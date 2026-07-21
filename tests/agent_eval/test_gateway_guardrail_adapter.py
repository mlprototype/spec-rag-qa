from __future__ import annotations

import json

import pytest

from ragqa.agent_eval import AgentEvalCase
from ragqa.agent_eval.adapters.gateway import (
    GatewayGuardrailAdapter,
    GatewayHttpRunner,
    GatewayInvalidResponseError,
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
