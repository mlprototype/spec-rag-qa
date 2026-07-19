from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from typing import Any

from ragqa.agent_eval.advanced_models import (
    AdvancedEvaluationError,
    StabilityDimensionResult,
    StabilityEvaluationResult,
)
from ragqa.agent_eval.assertions import resolve_json_path
from ragqa.agent_eval.judge import JudgeAdapter, JudgeError
from ragqa.agent_eval.models import AgentEvalCase, AgentRunTrace


async def evaluate_stability(
    case: AgentEvalCase,
    indexed_traces: Sequence[tuple[int, AgentRunTrace]],
    execution_errors: Sequence[AdvancedEvaluationError],
    judge: JudgeAdapter,
) -> StabilityEvaluationResult:
    """Evaluate deterministic signatures and Judge-based answer semantics."""

    traces = [trace for _, trace in indexed_traces]
    dimensions = {
        "route": _dimension(
            "route",
            [trace.output.route for trace in traces],
            requested_runs=case.repeat,
            execution_error_count=len(execution_errors),
        ),
        "query_type": _dimension(
            "query_type",
            [trace.output.query_type for trace in traces],
            requested_runs=case.repeat,
            execution_error_count=len(execution_errors),
        ),
        "tool_names": _dimension(
            "tool_names",
            [_tool_name_signature(trace) for trace in traces],
            requested_runs=case.repeat,
            execution_error_count=len(execution_errors),
        ),
        "tool_arguments": _dimension(
            "tool_arguments",
            [_tool_argument_signature(case, trace) for trace in traces],
            requested_runs=case.repeat,
            execution_error_count=len(execution_errors),
        ),
        "citation_set": _dimension(
            "citation_set",
            [_citation_signature(trace) for trace in traces],
            requested_runs=case.repeat,
            execution_error_count=len(execution_errors),
        ),
    }

    semantic_consistency: StabilityDimensionResult | None = None
    semantic_judge = None
    semantic_error = None
    if case.repeat >= 2:
        if len(traces) < 2:
            semantic_consistency = StabilityDimensionResult(
                dimension="answer_semantic",
                observations=len(traces),
                distinct_values=0,
                mode_share=None,
                all_match=False,
                mode_value=None,
            )
        else:
            try:
                decision = await judge.judge_semantic_consistency(
                    case.id,
                    [trace.output.answer for trace in traces],
                    schema_version=case.schema_version,
                )
                semantic_consistency = _dimension(
                    "answer_semantic",
                    [str(group_id) for group_id in decision.response.group_ids],
                    requested_runs=case.repeat,
                    execution_error_count=len(execution_errors),
                )
                semantic_judge = decision.judge
            except JudgeError as exc:
                semantic_error = AdvancedEvaluationError(
                    schema_version=case.schema_version,
                    case_id=case.id,
                    stage="semantic_judge",
                    error_type=type(exc).__name__,
                    message=str(exc),
                )

    return StabilityEvaluationResult(
        schema_version=case.schema_version,
        case_id=case.id,
        requested_runs=case.repeat,
        successful_runs=len(traces),
        execution_error_count=len(execution_errors),
        execution_errors=list(execution_errors),
        dimensions=dimensions,
        semantic_consistency=semantic_consistency,
        semantic_judge=semantic_judge,
        semantic_judge_error=semantic_error,
    )


def _dimension(
    name: str,
    values: Sequence[str],
    *,
    requested_runs: int,
    execution_error_count: int,
) -> StabilityDimensionResult:
    counts = Counter(values)
    observations = len(values)
    if requested_runs < 2 or observations < 2:
        mode_value = counts.most_common(1)[0][0] if counts else None
        return StabilityDimensionResult(
            dimension=name,
            observations=observations,
            distinct_values=len(counts),
            mode_share=None,
            all_match=None if requested_runs < 2 else False,
            mode_value=mode_value,
        )

    mode_value, mode_count = sorted(
        counts.items(), key=lambda item: (-item[1], item[0])
    )[0]
    return StabilityDimensionResult(
        dimension=name,
        observations=observations,
        distinct_values=len(counts),
        mode_share=round(mode_count / observations, 6),
        all_match=(
            execution_error_count == 0
            and observations == requested_runs
            and len(counts) == 1
        ),
        mode_value=mode_value,
    )


def _tool_name_signature(trace: AgentRunTrace) -> str:
    return _canonical_json([call.name for call in trace.tool_calls])


def _tool_argument_signature(case: AgentEvalCase, trace: AgentRunTrace) -> str:
    signature: list[dict[str, Any]] = []
    for call in trace.tool_calls:
        paths = _major_argument_paths(case, call.name)
        if not paths:
            arguments: Any = call.arguments
        else:
            arguments = {}
            for path in paths:
                found, value = resolve_json_path(call.arguments, path)
                arguments[path] = {"found": found, "value": value}
        signature.append({"tool_name": call.name, "arguments": arguments})
    return _canonical_json(signature)


def _major_argument_paths(case: AgentEvalCase, tool_name: str) -> list[str]:
    paths = {
        assertion.path
        for assertion in case.expected.tool_argument_assertions.get(tool_name, [])
    }
    schema = case.expected.tool_argument_schemas.get(tool_name, {})
    properties = schema.get("properties")
    if isinstance(properties, dict):
        paths.update(str(key) for key in properties)
    return sorted(paths)


def _citation_signature(trace: AgentRunTrace) -> str:
    return _canonical_json(
        sorted(
            {
                (citation.citation_id, citation.source_id)
                for citation in trace.citations
            }
        )
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
