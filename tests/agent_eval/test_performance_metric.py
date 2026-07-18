from ragqa.agent_eval import (
    LATENCY_BUDGET_EXCEEDED,
    AgentEvalCase,
    AgentRunTrace,
)
from ragqa.agent_eval.metrics.performance import evaluate_latency_budget


def test_latency_equal_to_budget_passes(
    smoke_pair: tuple[AgentEvalCase, AgentRunTrace],
) -> None:
    case, original = smoke_pair
    trace = original.model_copy(deep=True)
    trace.timing.latency_ms = case.budgets.max_latency_ms
    assert evaluate_latency_budget(case, trace).passed is True


def test_latency_budget_exceeded_failure_type(
    smoke_pair: tuple[AgentEvalCase, AgentRunTrace],
) -> None:
    case, original = smoke_pair
    trace = original.model_copy(deep=True)
    trace.timing.latency_ms = case.budgets.max_latency_ms + 0.1

    check = evaluate_latency_budget(case, trace)
    assert check.passed is False
    assert check.failure_type == LATENCY_BUDGET_EXCEEDED
