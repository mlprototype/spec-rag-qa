from __future__ import annotations

from collections.abc import Iterable, Sequence

from ragqa.agent_eval.metrics import (
    evaluate_answer_format,
    evaluate_citation_presence,
    evaluate_citation_validity,
    evaluate_latency_budget,
    evaluate_required_tool_calls,
    evaluate_route,
    evaluate_task_success,
    evaluate_tool_argument_schema,
    evaluate_tool_argument_semantics,
    evaluate_unexpected_tool_calls,
)
from ragqa.agent_eval.models import (
    AgentEvalCase,
    AgentEvaluationResult,
    AgentRunTrace,
    CaseEvaluationResult,
    CheckResult,
    MetricAggregate,
)
from ragqa.agent_eval.runner import (
    FixtureTraceMismatchError,
    FixtureTraceNotFoundError,
)


def evaluate_case(
    case: AgentEvalCase, trace: AgentRunTrace
) -> CaseEvaluationResult:
    """Run every deterministic Issue #10 check for one case and trace."""

    if trace.case_id != case.id:
        raise FixtureTraceMismatchError(
            f"Trace case id {trace.case_id!r} does not match case id: {case.id}"
        )
    if trace.input.question != case.input.question:
        raise FixtureTraceMismatchError(
            f"Trace input mismatch for case id: {case.id}"
        )

    checks = [
        evaluate_route(case, trace),
        evaluate_required_tool_calls(case, trace),
        evaluate_unexpected_tool_calls(case, trace),
        evaluate_tool_argument_schema(case, trace),
        evaluate_tool_argument_semantics(case, trace),
        evaluate_citation_presence(case, trace),
        evaluate_citation_validity(case, trace),
        evaluate_answer_format(case, trace),
        evaluate_latency_budget(case, trace),
    ]
    task_success = evaluate_task_success(checks, schema_version=case.schema_version)
    checks.append(task_success)

    return CaseEvaluationResult(
        schema_version=case.schema_version,
        case_id=case.id,
        run_id=trace.run_id,
        passed=task_success.passed,
        checks=checks,
    )


def evaluate_cases(
    cases: Sequence[AgentEvalCase], traces: Iterable[AgentRunTrace]
) -> AgentEvaluationResult:
    """Evaluate a batch and compute numerator/denominator based aggregate rates."""

    traces_by_case_id: dict[str, AgentRunTrace] = {}
    for trace in traces:
        if trace.case_id in traces_by_case_id:
            raise ValueError(f"Duplicate trace for case id: {trace.case_id}")
        traces_by_case_id[trace.case_id] = trace

    case_results: list[CaseEvaluationResult] = []
    for case in cases:
        trace = traces_by_case_id.get(case.id)
        if trace is None:
            raise FixtureTraceNotFoundError(
                f"Trace not found for case id: {case.id}"
            )
        case_results.append(evaluate_case(case, trace))

    schema_version = cases[0].schema_version if cases else "1.0"
    return AgentEvaluationResult(
        schema_version=schema_version,
        cases=case_results,
        metrics=aggregate_metrics(case_results),
    )


def aggregate_metrics(
    case_results: Sequence[CaseEvaluationResult],
) -> dict[str, MetricAggregate]:
    """Aggregate check counters; a zero denominator is represented as N/A."""

    specs = {
        "route_selection_accuracy": (
            "route_selection",
            "valid_count",
            "evaluated_count",
        ),
        "required_tool_call_rate": (
            "required_tool_calls",
            "called_required_count",
            "required_count",
        ),
        "unexpected_tool_call_rate": (
            "unexpected_tool_calls",
            "unexpected_count",
            "actual_count",
        ),
        "tool_argument_schema_compliance": (
            "tool_argument_schema",
            "valid_count",
            "evaluated_count",
        ),
        "tool_argument_semantic_accuracy": (
            "tool_argument_semantics",
            "valid_count",
            "evaluated_count",
        ),
        "citation_presence_rate": (
            "citation_presence",
            "present_count",
            "required_count",
        ),
        "citation_validity_rate": (
            "citation_validity",
            "valid_count",
            "citation_count",
        ),
        "answer_format_compliance": (
            "answer_format",
            "valid_count",
            "evaluated_count",
        ),
        "latency_budget_compliance": (
            "latency_budget",
            "valid_count",
            "evaluated_count",
        ),
        "task_success_rate": (
            "task_success",
            "valid_count",
            "evaluated_count",
        ),
    }

    checks = [check for result in case_results for check in result.checks]
    return {
        metric_id: _aggregate_check_counts(
            metric_id,
            checks,
            check_id=check_id,
            numerator_key=numerator_key,
            denominator_key=denominator_key,
        )
        for metric_id, (check_id, numerator_key, denominator_key) in specs.items()
    }


def _aggregate_check_counts(
    metric_id: str,
    checks: Sequence[CheckResult],
    check_id: str,
    numerator_key: str,
    denominator_key: str,
) -> MetricAggregate:
    matching = [check for check in checks if check.check_id == check_id]
    numerator = sum(int(check.details.get(numerator_key, 0)) for check in matching)
    denominator = sum(
        int(check.details.get(denominator_key, 0)) for check in matching
    )
    value = round(numerator / denominator, 6) if denominator else None
    return MetricAggregate(
        metric_id=metric_id,
        numerator=numerator,
        denominator=denominator,
        value=value,
    )
