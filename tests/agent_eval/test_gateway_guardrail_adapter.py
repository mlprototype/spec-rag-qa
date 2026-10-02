from __future__ import annotations

import json

import pytest

from ragqa.agent_eval import AgentEvalCase
from ragqa.agent_eval.adapters.gateway import (
    GatewayGuardrailAdapter,
    GatewayHttpRunner,
    GatewayInvalidResponseError,
    GatewayTransportError,
    _RejectRedirectHandler,
)


def test_gateway_adapter_normalizes_injection_block(
    guardrail_cases: list[AgentEvalCase],
) -> None:
    case = next(
        item
        for item in guardrail_cases
        if item.id == "guardrail-injection-block-01"
    )
    adapter = GatewayGuardrailAdapter()

    trace = adapter.normalize(
        case,
        status_code=403,
        headers={
            "X-Gateway-Security-Blocked": "true",
            "X-Gateway-Block-Reason": "INJECTION_DETECTED",
            "X-Gateway-Security-Score": "80",
            "X-Gateway-Security-Categories": "INSTRUCTION_OVERRIDE,JAILBREAK",
            "Authorization": "must-not-be-copied",
        },
        body=json.dumps({"status": 403, "error": "Forbidden"}),
        latency_ms=12.5,
    )

    assert trace.guardrail.detected is True
    assert trace.guardrail.action == "block"
    assert trace.guardrail.blocked is True
    assert trace.guardrail.http_status == 403
    assert "injection" in trace.guardrail.categories
    assert "instruction_override" in trace.guardrail.categories
    assert "Authorization" not in trace.guardrail.security_headers
    assert trace.guardrail.body == {"status": 403, "error": "Forbidden"}


def test_missing_security_headers_remain_unknown_not_allow(
    guardrail_cases: list[AgentEvalCase],
) -> None:
    case = guardrail_cases[0]
    trace = GatewayGuardrailAdapter().normalize(
        case,
        status_code=200,
        headers={"X-Gateway-Provider": "openai"},
        body=json.dumps({"id": "chatcmpl-test", "choices": []}),
        latency_ms=8.0,
    )

    assert trace.guardrail.detected is None
    assert trace.guardrail.action == "unknown"
    assert trace.guardrail.triggered is False


def test_explicit_allow_header_is_observable(
    guardrail_cases: list[AgentEvalCase],
) -> None:
    case = next(item for item in guardrail_cases if not item.expected.detected)
    trace = GatewayGuardrailAdapter().normalize(
        case,
        status_code=200,
        headers={"X-Gateway-Security-Action": "ALLOW"},
        body=json.dumps({"id": "chatcmpl-test", "choices": []}),
        latency_ms=8.0,
    )

    assert trace.guardrail.detected is False
    assert trace.guardrail.action == "allow"


@pytest.mark.parametrize(
    ("category", "expected_category"),
    [("INJECTION", "injection"), ("PII", "pii")],
)
def test_allow_with_detection_category_keeps_detection_and_action_independent(
    guardrail_cases: list[AgentEvalCase],
    category: str,
    expected_category: str,
) -> None:
    trace = GatewayGuardrailAdapter().normalize(
        guardrail_cases[0],
        status_code=200,
        headers={
            "X-Gateway-Security-Action": "ALLOW",
            "X-Gateway-Security-Category": category,
        },
        body=json.dumps({"id": "chatcmpl-test", "choices": []}),
        latency_ms=8.0,
    )

    assert trace.guardrail.detected is True
    assert trace.guardrail.action == "allow"
    assert expected_category in trace.guardrail.categories


def test_block_reason_without_action_header_is_block(
    guardrail_cases: list[AgentEvalCase],
) -> None:
    trace = GatewayGuardrailAdapter().normalize(
        guardrail_cases[0],
        status_code=403,
        headers={"X-Gateway-Block-Reason": "INJECTION_DETECTED"},
        body=json.dumps({"status": 403, "error": "Forbidden"}),
        latency_ms=8.0,
    )

    assert trace.guardrail.detected is True
    assert trace.guardrail.action == "block"
    assert trace.guardrail.blocked is True
    assert "injection" in trace.guardrail.categories


def test_category_without_action_is_detected_with_unknown_action(
    guardrail_cases: list[AgentEvalCase],
) -> None:
    trace = GatewayGuardrailAdapter().normalize(
        guardrail_cases[0],
        status_code=200,
        headers={"X-Gateway-Security-Category": "PII"},
        body=json.dumps({"id": "chatcmpl-test", "choices": []}),
        latency_ms=8.0,
    )

    assert trace.guardrail.detected is True
    assert trace.guardrail.action == "unknown"
    assert trace.guardrail.categories == ["pii"]


@pytest.mark.parametrize("action", ["WARN", "MASK", "BLOCK"])
def test_controlling_action_implies_detection(
    guardrail_cases: list[AgentEvalCase], action: str
) -> None:
    trace = GatewayGuardrailAdapter().normalize(
        guardrail_cases[0],
        status_code=200,
        headers={"X-Gateway-Security-Action": action},
        body=json.dumps({"id": "chatcmpl-test", "choices": []}),
        latency_ms=8.0,
    )

    assert trace.guardrail.detected is True
    assert trace.guardrail.action == action.lower()


def test_provider_input_redaction_is_normalized_as_mask(
    guardrail_cases: list[AgentEvalCase],
) -> None:
    case = next(
        item for item in guardrail_cases if item.expected.action == "mask"
    )
    trace = GatewayGuardrailAdapter().normalize(
        case,
        status_code=200,
        headers={},
        body=json.dumps({"id": "chatcmpl-test", "choices": []}),
        provider_input={
            "messages": [
                {"role": "user", "content": "連絡先は [EMAIL_REDACTED] です。"}
            ]
        },
        latency_ms=9.0,
    )

    assert trace.guardrail.detected is True
    assert trace.guardrail.action == "mask"
    assert trace.guardrail.mask_applied is True
    assert trace.guardrail.mask_evidence == "provider_input"
    assert trace.guardrail.categories[0] == "pii"


@pytest.mark.parametrize("body", ["not-json", "[]", b"\xff"])
def test_invalid_gateway_body_is_an_execution_error(
    guardrail_cases: list[AgentEvalCase], body: str | bytes
) -> None:
    with pytest.raises(GatewayInvalidResponseError):
        GatewayGuardrailAdapter().normalize(
            guardrail_cases[0],
            status_code=200,
            headers={},
            body=body,
            latency_ms=1.0,
        )


def test_gateway_api_key_requires_https() -> None:
    endpoint = "http://gateway.example.test/v1/chat/completions"
    token = "synthetic-test-token"

    with pytest.raises(ValueError, match="requires an HTTPS") as exc_info:
        GatewayHttpRunner(
            endpoint,
            api_key=token,
            allowed_hosts={"gateway.example.test"},
        )

    assert endpoint not in str(exc_info.value)
    assert token not in str(exc_info.value)


def test_gateway_host_must_match_allowlist() -> None:
    endpoint = "https://unapproved.example.test/v1/chat/completions"

    with pytest.raises(ValueError, match="configured allowlist") as exc_info:
        GatewayHttpRunner(
            endpoint,
            allowed_hosts={"approved.example.test"},
        )

    assert endpoint not in str(exc_info.value)
    assert "unapproved.example.test" not in str(exc_info.value)


def test_gateway_allowlist_is_required_when_omitted() -> None:
    endpoint = "https://gateway.example.test/v1/chat/completions"

    with pytest.raises(ValueError, match="allowlist must not be empty") as exc_info:
        GatewayHttpRunner(endpoint)

    assert endpoint not in str(exc_info.value)


@pytest.mark.parametrize("allowed_hosts", [None, set()])
def test_gateway_allowlist_rejects_none_or_empty(
    allowed_hosts: object,
) -> None:
    endpoint = "https://gateway.example.test/v1/chat/completions"

    with pytest.raises(ValueError, match="allowlist must not be empty") as exc_info:
        GatewayHttpRunner(endpoint, allowed_hosts=allowed_hosts)  # type: ignore[arg-type]

    assert endpoint not in str(exc_info.value)


def test_gateway_allowlist_exact_match_allows_construction() -> None:
    GatewayHttpRunner(
        "https://gateway.example.test/v1/chat/completions",
        allowed_hosts={"gateway.example.test"},
    )


def test_gateway_allowlist_normalizes_case_and_trailing_dot() -> None:
    GatewayHttpRunner(
        "https://GATEWAY.EXAMPLE.TEST./v1/chat/completions",
        allowed_hosts={"Gateway.Example.Test."},
    )


def test_gateway_allowlist_does_not_match_subdomains() -> None:
    endpoint = "https://child.gateway.example.test/v1/chat/completions"

    with pytest.raises(ValueError, match="configured allowlist") as exc_info:
        GatewayHttpRunner(
            endpoint,
            allowed_hosts={"gateway.example.test"},
        )

    assert endpoint not in str(exc_info.value)


@pytest.mark.parametrize(
    "allowed_host",
    ["*.example.test", "https://gateway.example.test"],
)
def test_gateway_allowlist_rejects_wildcard_and_url_values(
    allowed_host: str,
) -> None:
    endpoint = "https://gateway.example.test/v1/chat/completions"

    with pytest.raises(ValueError, match="allowed host entry is invalid") as exc_info:
        GatewayHttpRunner(endpoint, allowed_hosts={allowed_host})

    assert endpoint not in str(exc_info.value)


def test_gateway_redirects_remain_rejected_without_echoing_response() -> None:
    response_body = "PRIVATE_RESPONSE_BODY"

    with pytest.raises(GatewayTransportError, match="redirects are not allowed") as exc_info:
        _RejectRedirectHandler().redirect_request(
            None,
            None,
            302,
            response_body,
            {},
            "https://gateway.example.test/redirected",
        )

    assert response_body not in str(exc_info.value)
