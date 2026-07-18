from ragqa.agent_eval import TASK_FAILED, CheckResult
from ragqa.agent_eval.metrics.task_success import evaluate_task_success


def _check(check_id: str, passed: bool, required: bool = True) -> CheckResult:
    return CheckResult(
        schema_version="1.0",
        check_id=check_id,
        passed=passed,
        required=required,
        score=1.0 if passed else 0.0,
    )


def test_task_success_is_required_check_and_not_average() -> None:
    checks = [_check("critical", False)] + [
        _check(f"passing-{index}", True) for index in range(20)
    ]
    result = evaluate_task_success(checks)
    assert result.passed is False
    assert result.failure_type == TASK_FAILED
    assert result.score == 0.0


def test_optional_failure_does_not_fail_task() -> None:
    result = evaluate_task_success(
        [_check("required", True), _check("optional", False, required=False)]
    )
    assert result.passed is True
