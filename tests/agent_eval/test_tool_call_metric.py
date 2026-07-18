from ragqa.agent_eval import (
    REQUIRED_TOOL_NOT_CALLED,
    TOOL_ARGUMENT_SCHEMA_INVALID,
    TOOL_ARGUMENT_SEMANTIC_MISMATCH,
    UNEXPECTED_TOOL_CALLED,
    AgentEvalCase,
    AgentRunTrace,
    ToolCallTrace,
)
from ragqa.agent_eval.metrics.tool_call import (
    evaluate_required_tool_calls,
    evaluate_tool_argument_schema,
    evaluate_tool_argument_semantics,
    evaluate_unexpected_tool_calls,
)


def test_required_tool_missing_fails_schema_and_semantics_too(
    smoke_pair: tuple[AgentEvalCase, AgentRunTrace],
) -> None:
    case, original = smoke_pair
    trace = original.model_copy(deep=True)
    trace.tool_calls = []

    checks = [
        evaluate_required_tool_calls(case, trace),
        evaluate_tool_argument_schema(case, trace),
        evaluate_tool_argument_semantics(case, trace),
    ]
    assert all(check.passed is False for check in checks)
    assert all(check.failure_type == REQUIRED_TOOL_NOT_CALLED for check in checks)
    assert checks[1].details["evaluated_count"] == 1
    assert checks[1].score == 0.0


def test_unexpected_tool_call_failure_type(
    smoke_pair: tuple[AgentEvalCase, AgentRunTrace],
) -> None:
    case, original = smoke_pair
    trace = original.model_copy(deep=True)
    trace.tool_calls.append(ToolCallTrace(name="delete_all"))

    check = evaluate_unexpected_tool_calls(case, trace)
    assert check.passed is False
    assert check.failure_type == UNEXPECTED_TOOL_CALLED
    assert check.details["unexpected_count"] == 1


def test_tool_argument_schema_invalid_is_separate_check(
    smoke_pair: tuple[AgentEvalCase, AgentRunTrace],
) -> None:
    case, original = smoke_pair
    trace = original.model_copy(deep=True)
    trace.tool_calls[0].arguments = {"query": 409}

    check = evaluate_tool_argument_schema(case, trace)
    assert check.passed is False
    assert check.failure_type == TOOL_ARGUMENT_SCHEMA_INVALID


def test_tool_argument_semantic_mismatch_is_separate_check(
    smoke_pair: tuple[AgentEvalCase, AgentRunTrace],
) -> None:
    case, original = smoke_pair
    trace = original.model_copy(deep=True)
    trace.tool_calls[0].arguments = {"query": "セッション"}

    schema_check = evaluate_tool_argument_schema(case, trace)
    semantic_check = evaluate_tool_argument_semantics(case, trace)
    assert schema_check.passed is True
    assert semantic_check.passed is False
    assert semantic_check.failure_type == TOOL_ARGUMENT_SEMANTIC_MISMATCH
