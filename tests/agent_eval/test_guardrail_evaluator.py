from __future__ import annotations

from ragqa.agent_eval import AgentEvalCase, AgentRunTrace
from ragqa.agent_eval.failure_types import (
    GUARDRAIL_FALSE_NEGATIVE,
    GUARDRAIL_FALSE_POSITIVE,
    GUARDRAIL_MASK_EVIDENCE_UNAVAILABLE,
    GUARDRAIL_MASK_INVALID,
    GUARDRAIL_OBSERVATION_UNKNOWN,
)
from ragqa.agent_eval.guardrail import (
    evaluate_guardrail_case,
    evaluate_guardrail_cases,
)


def _pair(
    cases: list[AgentEvalCase],
    traces: list[AgentRunTrace],
    *,
    detected: bool,
    action: str | None = None,
) -> tuple[AgentEvalCase, AgentRunTrace]:
    case = next(
        item
        for item in cases
        if item.expected.detected is detected
        and (action is None or item.expected.action == action)
    )
    trace = next(item for item in traces if item.case_id == case.id).model_copy(
        deep=True
    )
    return case, trace


def test_guardrail_fixture_evaluation_passes(
    guardrail_cases: list[AgentEvalCase],
    guardrail_traces: list[AgentRunTrace],
) -> None:
    result = evaluate_guardrail_cases(guardrail_cases, guardrail_traces)

    assert len(result.cases) == 30
    assert all(case.passed for case in result.cases)


def test_false_negative_has_distinct_failure_type(
    guardrail_cases: list[AgentEvalCase],
    guardrail_traces: list[AgentRunTrace],
) -> None:
    case, trace = _pair(guardrail_cases, guardrail_traces, detected=True)
    trace.guardrail.detected = False

    result = evaluate_guardrail_case(case, trace)

    detection = next(
        check for check in result.checks if check.check_id == "guardrail_detection"
    )
    assert detection.failure_type == GUARDRAIL_FALSE_NEGATIVE


def test_false_positive_has_distinct_failure_type(
    guardrail_cases: list[AgentEvalCase],
    guardrail_traces: list[AgentRunTrace],
) -> None:
    case, trace = _pair(guardrail_cases, guardrail_traces, detected=False)
    trace.guardrail.detected = True

    result = evaluate_guardrail_case(case, trace)

    detection = next(
        check for check in result.checks if check.check_id == "guardrail_detection"
    )
    assert detection.failure_type == GUARDRAIL_FALSE_POSITIVE


def test_unknown_observation_fails_without_becoming_allow(
    guardrail_cases: list[AgentEvalCase],
    guardrail_traces: list[AgentRunTrace],
) -> None:
    case, trace = _pair(guardrail_cases, guardrail_traces, detected=False)
    trace.guardrail.detected = None
    trace.guardrail.action = "unknown"

    result = evaluate_guardrail_case(case, trace)

    assert result.passed is False
    failures = {
        check.failure_type for check in result.checks if not check.passed
    }
    assert failures == {GUARDRAIL_OBSERVATION_UNKNOWN}


def test_mask_requires_target_removal_and_replacement(
    guardrail_cases: list[AgentEvalCase],
    guardrail_traces: list[AgentRunTrace],
) -> None:
    case, trace = _pair(
        guardrail_cases, guardrail_traces, detected=True, action="mask"
    )
    target = case.expected.masked_values[0]
    trace.guardrail.provider_input = {
        "messages": [{"role": "user", "content": target}]
    }

    result = evaluate_guardrail_case(case, trace)

    mask = next(
        check
        for check in result.checks
        if check.check_id == "guardrail_mask_verification"
    )
    assert mask.passed is False
    assert mask.failure_type == GUARDRAIL_MASK_INVALID
    assert mask.details["leaked_value_count"] == 1


def test_mask_status_200_without_evidence_does_not_pass(
    guardrail_cases: list[AgentEvalCase],
    guardrail_traces: list[AgentRunTrace],
) -> None:
    case, trace = _pair(
        guardrail_cases, guardrail_traces, detected=True, action="mask"
    )
    trace.guardrail.provider_input = None
    trace.guardrail.mask_applied = None
    trace.guardrail.mask_evidence = None
    trace.guardrail.http_status = 200

    result = evaluate_guardrail_case(case, trace)

    mask = next(
        check
        for check in result.checks
        if check.check_id == "guardrail_mask_verification"
    )
    assert mask.failure_type == GUARDRAIL_MASK_EVIDENCE_UNAVAILABLE
    assert mask.details["evaluable"] is False
