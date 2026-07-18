from ragqa.agent_eval import ROUTE_MISMATCH, AgentEvalCase, AgentRunTrace
from ragqa.agent_eval.metrics.routing import evaluate_route


def test_route_matches_allowed_set(
    smoke_pair: tuple[AgentEvalCase, AgentRunTrace],
) -> None:
    case, trace = smoke_pair
    assert evaluate_route(case, trace).passed is True


def test_route_mismatch_failure_type(
    smoke_pair: tuple[AgentEvalCase, AgentRunTrace],
) -> None:
    case, original = smoke_pair
    trace = original.model_copy(deep=True)
    trace.output.route = "direct_answer"

    check = evaluate_route(case, trace)
    assert check.passed is False
    assert check.failure_type == ROUTE_MISMATCH
