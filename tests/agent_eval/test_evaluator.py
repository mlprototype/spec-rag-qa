import pytest

from ragqa.agent_eval import (
    AgentEvalCase,
    AgentRunTrace,
    SchemaVersionMismatchError,
    evaluate_case,
    evaluate_cases,
)


def test_smoke_batch_passes_and_aggregates_metrics(
    smoke_cases: list[AgentEvalCase], smoke_traces: list[AgentRunTrace]
) -> None:
    result = evaluate_cases(smoke_cases, smoke_traces)

    assert all(case_result.passed for case_result in result.cases)
    assert result.metrics["route_selection_accuracy"].value == 1.0
    assert result.metrics["required_tool_call_rate"].value == 1.0
    assert result.metrics["unexpected_tool_call_rate"].value == 0.0
    assert result.metrics["tool_argument_schema_compliance"].value == 1.0
    assert result.metrics["tool_argument_semantic_accuracy"].value == 1.0
    assert result.metrics["citation_presence_rate"].value == 1.0
    assert result.metrics["citation_validity_rate"].value == 1.0
    assert result.metrics["answer_format_compliance"].value == 1.0
    assert result.metrics["latency_budget_compliance"].value == 1.0
    assert result.metrics["task_success_rate"].value == 1.0


def test_case_failure_propagates_to_task_success(
    smoke_pair: tuple[AgentEvalCase, AgentRunTrace],
) -> None:
    case, original = smoke_pair
    trace = original.model_copy(deep=True)
    trace.output.route = "wrong-route"

    result = evaluate_case(case, trace)
    checks = {check.check_id: check for check in result.checks}
    assert checks["route_selection"].passed is False
    assert checks["task_success"].passed is False
    assert result.passed is False


def test_missing_tool_is_counted_in_schema_and_semantic_denominators(
    smoke_pair: tuple[AgentEvalCase, AgentRunTrace],
) -> None:
    case, original = smoke_pair
    trace = original.model_copy(deep=True)
    trace.tool_calls = []

    result = evaluate_cases([case], [trace])
    schema = result.metrics["tool_argument_schema_compliance"]
    semantic = result.metrics["tool_argument_semantic_accuracy"]
    assert (schema.numerator, schema.denominator, schema.value) == (0, 1, 0.0)
    assert (semantic.numerator, semantic.denominator, semantic.value) == (
        0,
        1,
        0.0,
    )
    assert result.metrics["task_success_rate"].value == 0.0


def test_case_and_trace_schema_versions_must_match(
    smoke_pair: tuple[AgentEvalCase, AgentRunTrace],
) -> None:
    case, original = smoke_pair
    trace = original.model_copy(deep=True)
    trace.schema_version = "2.0"

    with pytest.raises(SchemaVersionMismatchError, match=case.id):
        evaluate_case(case, trace)


def test_batch_rejects_mixed_schema_versions(
    smoke_cases: list[AgentEvalCase], smoke_traces: list[AgentRunTrace]
) -> None:
    cases = [case.model_copy(deep=True) for case in smoke_cases[:2]]
    traces = [trace.model_copy(deep=True) for trace in smoke_traces[:2]]
    cases[1].schema_version = "2.0"
    traces[1].schema_version = "2.0"

    with pytest.raises(SchemaVersionMismatchError, match="Mixed case"):
        evaluate_cases(cases, traces)


def test_batch_rejects_mixed_trace_schema_versions(
    smoke_cases: list[AgentEvalCase], smoke_traces: list[AgentRunTrace]
) -> None:
    cases = [case.model_copy(deep=True) for case in smoke_cases[:2]]
    traces = [trace.model_copy(deep=True) for trace in smoke_traces[:2]]
    traces[1].schema_version = "2.0"

    with pytest.raises(SchemaVersionMismatchError, match="Mixed trace"):
        evaluate_cases(cases, traces)
