from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from ragqa.agent_eval.advanced_models import (
    AdvancedEvaluationError,
    JudgeRequest,
)
from ragqa.agent_eval.judge import StructuredJudgeAdapter
from ragqa.agent_eval.models import AgentEvalCase, AgentRunTrace
from ragqa.agent_eval.stability import evaluate_stability


class SemanticJudgeTransport:
    def __init__(self, group_ids: list[int]) -> None:
        self.group_ids = group_ids
        self.requests: list[JudgeRequest] = []

    async def complete(self, request: JudgeRequest) -> Mapping[str, Any]:
        self.requests.append(request)
        return {
            "schema_version": request.schema_version,
            "group_ids": self.group_ids,
            "reason": "Test semantic grouping.",
        }


def _repeat_case(case: AgentEvalCase, repeat: int = 3) -> AgentEvalCase:
    repeated = case.model_copy(deep=True)
    repeated.repeat = repeat
    return repeated


def _indexed_traces(trace: AgentRunTrace, count: int = 3) -> list[tuple[int, AgentRunTrace]]:
    traces = []
    for index in range(1, count + 1):
        copied = trace.model_copy(deep=True)
        copied.run_id = f"{trace.run_id}-{index}"
        traces.append((index, copied))
    return traces


def test_stability_reports_mode_share_and_all_match(
    synthetic_cases: list[AgentEvalCase],
    synthetic_traces: list[AgentRunTrace],
) -> None:
    case = _repeat_case(synthetic_cases[0])
    traces = _indexed_traces(synthetic_traces[0])
    traces[2][1].output.route = "retrieval"
    transport = SemanticJudgeTransport([0, 0, 1])
    judge = StructuredJudgeAdapter(transport, judge_model="semantic-mock")

    result = asyncio.run(evaluate_stability(case, traces, [], judge))

    assert result.dimensions["route"].mode_share == pytest.approx(2 / 3)
    assert result.dimensions["route"].all_match is False
    assert result.dimensions["query_type"].mode_share == 1.0
    assert result.dimensions["query_type"].all_match is True
    assert result.semantic_consistency is not None
    assert result.semantic_consistency.mode_share == pytest.approx(2 / 3)
    assert result.semantic_consistency.all_match is False


def test_semantic_consistency_uses_judge_not_exact_string_match(
    synthetic_cases: list[AgentEvalCase],
    synthetic_traces: list[AgentRunTrace],
) -> None:
    case = _repeat_case(synthetic_cases[0])
    traces = _indexed_traces(synthetic_traces[0])
    traces[0][1].output.answer = "The timeout is thirty minutes."
    traces[1][1].output.answer = "30 minutes is the configured timeout."
    traces[2][1].output.answer = "Configuration: a half-hour timeout."
    transport = SemanticJudgeTransport([4, 4, 4])
    judge = StructuredJudgeAdapter(transport, judge_model="semantic-mock")

    result = asyncio.run(evaluate_stability(case, traces, [], judge))

    assert result.semantic_consistency is not None
    assert result.semantic_consistency.mode_share == 1.0
    assert result.semantic_consistency.all_match is True
    assert len(transport.requests) == 1


def test_execution_error_is_explicit_and_prevents_all_match(
    synthetic_cases: list[AgentEvalCase],
    synthetic_traces: list[AgentRunTrace],
) -> None:
    case = _repeat_case(synthetic_cases[0])
    traces = _indexed_traces(synthetic_traces[0], count=2)
    error = AdvancedEvaluationError(
        schema_version="1.0",
        case_id=case.id,
        run_index=3,
        stage="runner",
        error_type="RunnerTimeoutError",
        message="timeout",
    )
    judge = StructuredJudgeAdapter(
        SemanticJudgeTransport([0, 0]), judge_model="semantic-mock"
    )

    result = asyncio.run(evaluate_stability(case, traces, [error], judge))

    assert result.execution_error_count == 1
    assert result.execution_errors[0].run_index == 3
    assert result.dimensions["route"].mode_share == 1.0
    assert result.dimensions["route"].all_match is False
    assert result.semantic_consistency is not None
    assert result.semantic_consistency.all_match is False


def test_tool_argument_stability_ignores_uncontracted_arguments(
    synthetic_cases: list[AgentEvalCase],
    synthetic_traces: list[AgentRunTrace],
) -> None:
    pair = next(
        (case, trace)
        for case in synthetic_cases
        for trace in synthetic_traces
        if case.id == trace.case_id and trace.tool_calls
    )
    case = _repeat_case(pair[0])
    traces = _indexed_traces(pair[1])
    traces[0][1].tool_calls[0].arguments["debug_nonce"] = "one"
    traces[1][1].tool_calls[0].arguments["debug_nonce"] = "two"
    traces[2][1].tool_calls[0].arguments["debug_nonce"] = "three"
    judge = StructuredJudgeAdapter(
        SemanticJudgeTransport([0, 0, 0]), judge_model="semantic-mock"
    )

    result = asyncio.run(evaluate_stability(case, traces, [], judge))

    assert result.dimensions["tool_arguments"].mode_share == 1.0
    assert result.dimensions["tool_arguments"].all_match is True


def test_citation_stability_compares_citation_id_and_source_mapping(
    synthetic_cases: list[AgentEvalCase],
    synthetic_traces: list[AgentRunTrace],
) -> None:
    pair = next(
        (case, trace)
        for case in synthetic_cases
        for trace in synthetic_traces
        if case.id == trace.case_id and trace.citations
    )
    case = _repeat_case(pair[0])
    traces = _indexed_traces(pair[1])
    traces[2][1].citations[0].citation_id = "renumbered-citation"
    judge = StructuredJudgeAdapter(
        SemanticJudgeTransport([0, 0, 0]), judge_model="semantic-mock"
    )

    result = asyncio.run(evaluate_stability(case, traces, [], judge))

    assert result.dimensions["citation_set"].mode_share == pytest.approx(2 / 3)
    assert result.dimensions["citation_set"].all_match is False


def test_semantic_judge_uses_case_schema_version(
    synthetic_cases: list[AgentEvalCase],
    synthetic_traces: list[AgentRunTrace],
) -> None:
    case = _repeat_case(synthetic_cases[0], repeat=2)
    case.schema_version = "2.0"
    traces = _indexed_traces(synthetic_traces[0], count=2)
    transport = SemanticJudgeTransport([0, 0])
    judge = StructuredJudgeAdapter(transport, judge_model="semantic-mock")

    asyncio.run(evaluate_stability(case, traces, [], judge))

    assert transport.requests[0].schema_version == "2.0"


def test_repeat_one_has_no_stability_score(
    synthetic_cases: list[AgentEvalCase],
    synthetic_traces: list[AgentRunTrace],
) -> None:
    judge = StructuredJudgeAdapter(
        SemanticJudgeTransport([]), judge_model="semantic-mock"
    )

    case = _repeat_case(synthetic_cases[0], repeat=1)
    result = asyncio.run(
        evaluate_stability(
            case,
            [(1, synthetic_traces[0])],
            [],
            judge,
        )
    )

    assert result.dimensions["route"].mode_share is None
    assert result.dimensions["route"].all_match is None
    assert result.semantic_consistency is None
