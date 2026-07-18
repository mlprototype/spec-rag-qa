from collections.abc import Sequence

from ragqa.agent_eval.failure_types import TASK_FAILED
from ragqa.agent_eval.models import CheckResult


CHECK_ID = "task_success"


def evaluate_task_success(
    checks: Sequence[CheckResult], schema_version: str = "1.0"
) -> CheckResult:
    """Pass only when every required non-task check passes."""

    required_checks = [
        check for check in checks if check.required and check.check_id != CHECK_ID
    ]
    failed_check_ids = [
        check.check_id for check in required_checks if not check.passed
    ]
    passed = not failed_check_ids

    return CheckResult(
        schema_version=schema_version,
        check_id=CHECK_ID,
        passed=passed,
        failure_type=None if passed else TASK_FAILED,
        score=1.0 if passed else 0.0,
        message=(
            None
            if passed
            else f"Required check(s) failed: {', '.join(failed_check_ids)}"
        ),
        details={
            "required_check_ids": [check.check_id for check in required_checks],
            "failed_check_ids": failed_check_ids,
            "valid_count": int(passed),
            "evaluated_count": 1,
        },
    )
