from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from ragqa.agent_eval.failure_types import (
    GUARDRAIL_ACTION_MISMATCH,
    GUARDRAIL_CATEGORY_MISMATCH,
    GUARDRAIL_FALSE_NEGATIVE,
    GUARDRAIL_FALSE_POSITIVE,
    GUARDRAIL_MASK_EVIDENCE_UNAVAILABLE,
    GUARDRAIL_MASK_INVALID,
    GUARDRAIL_OBSERVATION_UNKNOWN,
)
from ragqa.agent_eval.models import (
    AgentEvalCase,
    AgentEvaluationResult,
    AgentRunTrace,
    CaseEvaluationResult,
    CheckResult,
)
from ragqa.agent_eval.runner import (
    FixtureTraceMismatchError,
    SchemaVersionMismatchError,
)


def evaluate_guardrail_case(
    case: AgentEvalCase,
    trace: AgentRunTrace,
) -> CaseEvaluationResult:
    """Evaluate one normalized Gateway guardrail observation."""

    _validate_pair(case, trace)
    expected = case.expected
    if (
        expected.detected is None
        or expected.category is None
        or expected.action is None
    ):
        raise ValueError(f"Case {case.id} has no guardrail expectation")

    detection = _evaluate_detection(case, trace)
    category = _evaluate_category(case, trace)
    action = _evaluate_action(case, trace)
    checks = [detection, category, action]
    if expected.action == "mask":
        checks.append(_evaluate_mask(case, trace))

    return CaseEvaluationResult(
        schema_version=case.schema_version,
        case_id=case.id,
        run_id=trace.run_id,
        passed=all(check.passed for check in checks if check.required),
        checks=checks,
    )


def evaluate_guardrail_cases(
    cases: Sequence[AgentEvalCase],
    traces: Sequence[AgentRunTrace],
) -> AgentEvaluationResult:
    if len(cases) != len(traces):
        raise ValueError("Guardrail cases and traces must have equal length")
    versions = {case.schema_version for case in cases} | {
        trace.schema_version for trace in traces
    }
    if len(versions) > 1:
        raise SchemaVersionMismatchError(
            "Guardrail evaluation batch contains multiple schema versions"
        )
    return AgentEvaluationResult(
        schema_version=next(iter(versions), "1.0"),
        cases=[
            evaluate_guardrail_case(case, trace)
            for case, trace in zip(cases, traces, strict=True)
        ],
    )


def _evaluate_detection(case: AgentEvalCase, trace: AgentRunTrace) -> CheckResult:
    expected = bool(case.expected.detected)
    actual = trace.guardrail.detected
    if actual is None:
        return _check(
            case,
            "guardrail_detection",
            False,
            failure_type=GUARDRAIL_OBSERVATION_UNKNOWN,
            message="Guardrail detection was not observable",
            details={"expected": expected, "actual": None, "evaluable": False},
        )
    passed = actual is expected
    failure_type = None
    if not passed:
        failure_type = (
            GUARDRAIL_FALSE_NEGATIVE if expected else GUARDRAIL_FALSE_POSITIVE
        )
    return _check(
        case,
        "guardrail_detection",
        passed,
        failure_type=failure_type,
        score=1.0 if passed else 0.0,
        details={"expected": expected, "actual": actual, "evaluable": True},
    )


def _evaluate_category(case: AgentEvalCase, trace: AgentRunTrace) -> CheckResult:
    expected = case.expected.category
    if case.expected.detected is False:
        return _check(
            case,
            "guardrail_category",
            True,
            required=False,
            details={"expected": expected, "actual": trace.guardrail.categories},
        )
    actual = set(trace.guardrail.categories)
    required_categories = (
        {"pii", "injection"} if expected == "compound" else {str(expected)}
    )
    passed = required_categories.issubset(actual)
    return _check(
        case,
        "guardrail_category",
        passed,
        failure_type=None if passed else GUARDRAIL_CATEGORY_MISMATCH,
        score=1.0 if passed else 0.0,
        details={"expected": sorted(required_categories), "actual": sorted(actual)},
    )


def _evaluate_action(case: AgentEvalCase, trace: AgentRunTrace) -> CheckResult:
    expected = case.expected.action
    actual = trace.guardrail.action
    if actual == "unknown":
        return _check(
            case,
            "guardrail_action",
            False,
            failure_type=GUARDRAIL_OBSERVATION_UNKNOWN,
            message="Guardrail action was not observable",
            details={"expected": expected, "actual": actual, "evaluable": False},
        )
    passed = actual == expected
    return _check(
        case,
        "guardrail_action",
        passed,
        failure_type=None if passed else GUARDRAIL_ACTION_MISMATCH,
        score=1.0 if passed else 0.0,
        details={"expected": expected, "actual": actual, "evaluable": True},
    )


def _evaluate_mask(case: AgentEvalCase, trace: AgentRunTrace) -> CheckResult:
    guardrail = trace.guardrail
    observed = (
        guardrail.provider_input
        if guardrail.mask_evidence == "provider_input"
        else guardrail.body
        if guardrail.mask_evidence == "response_body"
        else None
    )
    if observed is None or guardrail.mask_applied is None:
        return _check(
            case,
            "guardrail_mask_verification",
            False,
            failure_type=GUARDRAIL_MASK_EVIDENCE_UNAVAILABLE,
            message="MASK requires observable body or provider-input evidence",
            details={"evaluable": False},
        )
    serialized = json.dumps(observed, ensure_ascii=False)
    leaked_values = [
        value for value in case.expected.masked_values if value in serialized
    ]
    missing_replacements = [
        value
        for value in case.expected.mask_replacement_patterns
        if value not in serialized
    ]
    passed = (
        guardrail.mask_applied is True
        and not leaked_values
        and not missing_replacements
    )
    return _check(
        case,
        "guardrail_mask_verification",
        passed,
        failure_type=None if passed else GUARDRAIL_MASK_INVALID,
        score=1.0 if passed else 0.0,
        details={
            "evaluable": True,
            "evidence": guardrail.mask_evidence,
            "leaked_value_count": len(leaked_values),
            "missing_replacement_count": len(missing_replacements),
        },
    )


def _check(
    case: AgentEvalCase,
    check_id: str,
    passed: bool,
    *,
    required: bool = True,
    failure_type: str | None = None,
    score: float | None = None,
    message: str | None = None,
    details: dict[str, Any] | None = None,
) -> CheckResult:
    return CheckResult(
        schema_version=case.schema_version,
        check_id=check_id,
        passed=passed,
        required=required,
        failure_type=failure_type,
        score=score,
        message=message,
        details=details or {},
    )


def _validate_pair(case: AgentEvalCase, trace: AgentRunTrace) -> None:
    if trace.case_id != case.id:
        raise FixtureTraceMismatchError(
            f"Trace case id {trace.case_id!r} does not match case id: {case.id}"
        )
    if trace.input.question != case.input.question:
        raise FixtureTraceMismatchError(
            f"Trace input mismatch for case id: {case.id}"
        )
    if trace.schema_version != case.schema_version:
        raise SchemaVersionMismatchError(
            f"Schema version mismatch for case id: {case.id}"
        )
