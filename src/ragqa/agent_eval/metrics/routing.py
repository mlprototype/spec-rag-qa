from ragqa.agent_eval.failure_types import ROUTE_MISMATCH
from ragqa.agent_eval.models import AgentEvalCase, AgentRunTrace, CheckResult


CHECK_ID = "route_selection"


def evaluate_route(case: AgentEvalCase, trace: AgentRunTrace) -> CheckResult:
    """Pass when the observed route belongs to the case's allowed route set."""

    expected = list(case.expected.routes)
    passed = trace.output.route in expected
    return CheckResult(
        schema_version=case.schema_version,
        check_id=CHECK_ID,
        passed=passed,
        failure_type=None if passed else ROUTE_MISMATCH,
        score=1.0 if passed else 0.0,
        message=None if passed else f"Route {trace.output.route!r} is not allowed",
        details={
            "actual_route": trace.output.route,
            "expected_routes": expected,
            "valid_count": int(passed),
            "evaluated_count": 1,
        },
    )
