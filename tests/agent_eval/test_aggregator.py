from __future__ import annotations

from ragqa.agent_eval import AgentEvalCase, AgentRunTrace, evaluate_cases
from ragqa.agent_eval.aggregator import (
    aggregate_agent_evaluation,
    build_route_confusion_matrix,
    percentile,
)


def test_percentile_uses_nearest_rank_and_empty_is_na() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert percentile(values, 0) == 1.0
    assert percentile(values, 50) == 3.0
    assert percentile(values, 80) == 4.0
    assert percentile(values, 95) == 5.0
    assert percentile([], 95) is None


def test_route_confusion_matrix_counts_expected_and_actual_routes(
    synthetic_cases: list[AgentEvalCase],
    synthetic_traces: list[AgentRunTrace],
) -> None:
    cases = [case.model_copy(deep=True) for case in synthetic_cases[:2]]
    traces = [trace.model_copy(deep=True) for trace in synthetic_traces[:2]]
    traces[1].output.route = "retrieval"

    confusion = build_route_confusion_matrix(cases, traces)

    assert confusion["matrix"]["direct"]["direct"] == 1
    assert confusion["matrix"]["direct"]["retrieval"] == 1


def test_zero_denominator_remains_na(
    synthetic_cases: list[AgentEvalCase],
    synthetic_traces: list[AgentRunTrace],
) -> None:
    cases = [case for case in synthetic_cases if case.category == "direct"]
    case_ids = {case.id for case in cases}
    traces = [trace for trace in synthetic_traces if trace.case_id in case_ids]
    evaluation = evaluate_cases(cases, traces)

    aggregation = aggregate_agent_evaluation(cases, traces, evaluation, [])

    required_tools = aggregation["metrics"]["required_tool_call_rate"]
    citations = aggregation["metrics"]["citation_validity_rate"]
    assert required_tools["denominator"] == 0
    assert required_tools["value"] is None
    assert citations["denominator"] == 0
    assert citations["value"] is None


def test_execution_error_is_not_counted_as_evaluation_failure(
    synthetic_cases: list[AgentEvalCase],
    synthetic_traces: list[AgentRunTrace],
) -> None:
    failed_case = synthetic_cases[0]
    evaluated_cases = synthetic_cases[1:]
    evaluated_traces = synthetic_traces[1:]
    evaluation = evaluate_cases(evaluated_cases, evaluated_traces)
    execution_errors = [
        {
            "case_id": failed_case.id,
            "error_type": "RunnerTimeoutError",
            "message": "timed out",
        }
    ]

    aggregation = aggregate_agent_evaluation(
        synthetic_cases,
        evaluated_traces,
        evaluation,
        execution_errors,
    )

    assert aggregation["counts"]["execution_errors"] == 1
    assert aggregation["counts"]["failed_cases"] == 0
    outcome = next(
        item
        for item in aggregation["case_outcomes"]
        if item["case_id"] == failed_case.id
    )
    assert outcome["status"] == "execution_error"
