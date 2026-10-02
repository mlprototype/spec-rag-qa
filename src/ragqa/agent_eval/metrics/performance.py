from ragqa.agent_eval.failure_types import LATENCY_BUDGET_EXCEEDED
from ragqa.agent_eval.models import AgentEvalCase, AgentRunTrace, CheckResult


CHECK_ID = "latency_budget"


def evaluate_latency_budget(
    case: AgentEvalCase, trace: AgentRunTrace
) -> CheckResult:
    budget = case.budgets.max_latency_ms
    actual = trace.timing.latency_ms
    passed = actual <= budget
    return CheckResult(
        schema_version=case.schema_version,
        check_id=CHECK_ID,
        passed=passed,
        failure_type=None if passed else LATENCY_BUDGET_EXCEEDED,
        score=1.0 if passed else 0.0,
        message=None if passed else f"Latency {actual}ms exceeds budget {budget}ms",
        details={
            "latency_ms": actual,
            "max_latency_ms": budget,
            "valid_count": int(passed),
            "evaluated_count": 1,
        },
    )
