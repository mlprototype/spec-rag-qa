from __future__ import annotations

import asyncio
import json
import socket
import time
from collections.abc import Collection, Mapping
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from ragqa.agent_eval.models import (
    AgentEvalCase,
    AgentRunInput,
    AgentRunOutput,
    AgentRunTrace,
    ControlTrace,
    GuardrailTrace,
    TimingTrace,
    UsageTrace,
)
from ragqa.agent_eval.runner import AgentRunnerError, RunnerTimeoutError


SECURITY_ACTION_HEADER = "X-Gateway-Security-Action"
SECURITY_BLOCKED_HEADER = "X-Gateway-Security-Blocked"
BLOCK_REASON_HEADER = "X-Gateway-Block-Reason"
SECURITY_SCORE_HEADER = "X-Gateway-Security-Score"
SECURITY_CATEGORIES_HEADER = "X-Gateway-Security-Categories"
SECURITY_CATEGORY_HEADER = "X-Gateway-Security-Category"
_SECURITY_HEADER_NAMES = {
    SECURITY_ACTION_HEADER.lower(),
    SECURITY_BLOCKED_HEADER.lower(),
    BLOCK_REASON_HEADER.lower(),
    SECURITY_SCORE_HEADER.lower(),
    SECURITY_CATEGORIES_HEADER.lower(),
    SECURITY_CATEGORY_HEADER.lower(),
}
_MASK_MARKERS = (
    "[EMAIL_REDACTED]",
    "[PHONE_REDACTED]",
    "[CARD_REDACTED]",
    "[KEY_REDACTED]",
)


class GatewayRunnerError(AgentRunnerError):
    """Base class for failures before a Gateway response can be evaluated."""


class GatewayInvalidResponseError(GatewayRunnerError):
    """Raised when a Gateway response cannot be normalized safely."""


class GatewayTransportError(GatewayRunnerError):
    """Raised when the Gateway endpoint cannot return an HTTP response."""


class GatewayGuardrailAdapter:
    """Normalize observable Gateway HTTP facts into the shared run trace."""

    def normalize(
        self,
        case: AgentEvalCase,
        *,
        status_code: int,
        headers: Mapping[str, str],
        body: bytes | str,
        latency_ms: float,
        provider_input: Any | None = None,
    ) -> AgentRunTrace:
        if not 100 <= status_code <= 599:
            raise GatewayInvalidResponseError(
                f"Gateway returned an invalid HTTP status for case id: {case.id}"
            )
        body_value, answer = _parse_json_body(body, case.id)
        header_lookup = {
            str(name).strip().lower(): str(value).strip()
            for name, value in headers.items()
        }
        security_headers = {
            _canonical_header_name(name): value
            for name, value in header_lookup.items()
            if name in _SECURITY_HEADER_NAMES
        }

        action = _read_explicit_action(header_lookup)
        blocked = (
            header_lookup.get(SECURITY_BLOCKED_HEADER.lower(), "").lower()
            == "true"
        )
        raw_block_reason = header_lookup.get(BLOCK_REASON_HEADER.lower())
        block_reason = raw_block_reason.upper() if raw_block_reason else None
        categories = _read_categories(header_lookup, block_reason)
        mask_applied, mask_evidence = _observe_mask(provider_input, body_value)

        if action == "unknown" and blocked:
            action = "block"
        if action == "unknown" and mask_applied is True:
            action = "mask"
            if "pii" not in categories:
                categories.insert(0, "pii")

        if action == "allow":
            detected: bool | None = False
        elif action in {"warn", "mask", "block"}:
            detected = True
        elif blocked or block_reason is not None or categories:
            detected = True
        else:
            detected = None

        codes = []
        if block_reason:
            codes.append(block_reason)
        for category in categories:
            if category not in codes:
                codes.append(category)

        guardrail = GuardrailTrace(
            triggered=detected is True,
            blocked=action == "block",
            codes=codes,
            detected=detected,
            action=action,
            categories=categories,
            http_status=status_code,
            security_headers=security_headers,
            body=body_value,
            provider_input=provider_input,
            mask_applied=mask_applied,
            mask_evidence=mask_evidence,
        )
        return AgentRunTrace(
            schema_version=case.schema_version,
            run_id=f"gateway-{case.id}",
            case_id=case.id,
            target="policy-aware-llm-gateway",
            input=AgentRunInput(question=case.input.question),
            output=AgentRunOutput(
                answer=answer,
                query_type="guardrail",
                route="gateway",
                confidence=1.0,
            ),
            guardrail=guardrail,
            control=ControlTrace(),
            usage=UsageTrace(),
            timing=TimingTrace(latency_ms=latency_ms),
        )


class GatewayHttpRunner:
    """Execute one guardrail case against an HTTP Gateway endpoint."""

    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str | None = None,
        allowed_hosts: Collection[str] | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Gateway timeout must be greater than zero")
        parsed = _parse_endpoint(endpoint)
        if api_key and parsed.scheme != "https":
            raise ValueError("Gateway API key requires an HTTPS endpoint")
        allowed = {
            _normalize_allowed_host(host) for host in (allowed_hosts or [])
        }
        hostname = (parsed.hostname or "").lower().rstrip(".")
        if allowed and hostname not in allowed:
            raise ValueError("Gateway endpoint host is not in the configured allowlist")
        self._endpoint = endpoint
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._adapter = GatewayGuardrailAdapter()
        self._opener = urllib_request.build_opener(_RejectRedirectHandler())

    async def run(self, case: AgentEvalCase) -> AgentRunTrace:
        return await asyncio.to_thread(self._run_sync, case)

    def _run_sync(self, case: AgentEvalCase) -> AgentRunTrace:
        payload = {
            "model": str(case.metadata.get("gateway_model", "gpt-4o-mini")),
            "messages": [{"role": "user", "content": case.input.question}],
            "max_tokens": int(case.metadata.get("max_tokens", 16)),
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        provider = case.metadata.get("gateway_provider")
        if isinstance(provider, str) and provider.strip():
            headers["X-Gateway-Requested-Provider"] = provider.strip()
        request = urllib_request.Request(
            self._endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        started = time.monotonic()
        try:
            with self._opener.open(
                request,
                timeout=self._timeout_seconds,
            ) as response:
                status = int(response.status)
                response_headers = dict(response.headers.items())
                body = response.read()
        except urllib_error.HTTPError as exc:
            status = int(exc.code)
            response_headers = dict(exc.headers.items()) if exc.headers else {}
            body = exc.read()
        except (TimeoutError, socket.timeout) as exc:
            raise RunnerTimeoutError(
                f"Gateway timed out for case id: {case.id}"
            ) from exc
        except (OSError, urllib_error.URLError) as exc:
            raise GatewayTransportError(
                f"Gateway request failed for case id: {case.id}"
            ) from exc
        latency_ms = (time.monotonic() - started) * 1000.0
        return self._adapter.normalize(
            case,
            status_code=status,
            headers=response_headers,
            body=body,
            latency_ms=latency_ms,
        )


def _parse_json_body(body: bytes | str, case_id: str) -> tuple[Any, str]:
    try:
        text = body.decode("utf-8") if isinstance(body, bytes) else str(body)
    except UnicodeDecodeError as exc:
        raise GatewayInvalidResponseError(
            f"Gateway response was not UTF-8 for case id: {case_id}"
        ) from exc
    if not text.strip():
        raise GatewayInvalidResponseError(
            f"Gateway response body was empty for case id: {case_id}"
        )
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GatewayInvalidResponseError(
            f"Gateway response was not valid JSON for case id: {case_id}"
        ) from exc
    if not isinstance(value, Mapping):
        raise GatewayInvalidResponseError(
            f"Gateway response was not a JSON object for case id: {case_id}"
        )
    return value, json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _read_explicit_action(headers: Mapping[str, str]) -> str:
    value = headers.get(SECURITY_ACTION_HEADER.lower(), "").strip().lower()
    return value if value in {"allow", "warn", "mask", "block"} else "unknown"


def _read_categories(
    headers: Mapping[str, str], block_reason: str | None
) -> list[str]:
    categories: list[str] = []
    if block_reason == "PII_DETECTED":
        categories.append("pii")
    elif block_reason == "INJECTION_DETECTED":
        categories.append("injection")
    explicit = headers.get(SECURITY_CATEGORY_HEADER.lower(), "")
    raw_categories = headers.get(SECURITY_CATEGORIES_HEADER.lower(), "")
    for value in (explicit, raw_categories):
        for item in value.split(","):
            normalized = item.strip().lower()
            if not normalized:
                continue
            if normalized != "pii" and "injection" not in categories:
                categories.append("injection")
            if normalized not in categories:
                categories.append(normalized)
    return categories


def _observe_mask(
    provider_input: Any | None, body: Any
) -> tuple[bool | None, str | None]:
    if provider_input is not None:
        serialized = json.dumps(provider_input, ensure_ascii=False)
        return any(marker in serialized for marker in _MASK_MARKERS), "provider_input"
    serialized_body = json.dumps(body, ensure_ascii=False)
    if any(marker in serialized_body for marker in _MASK_MARKERS):
        return True, "response_body"
    return None, None


def _canonical_header_name(name: str) -> str:
    return "-".join(part.capitalize() for part in name.split("-"))


def _parse_endpoint(url: str) -> urllib_parse.SplitResult:
    try:
        endpoint = urllib_parse.urlsplit(url)
        _ = endpoint.port
    except ValueError:
        raise ValueError("Gateway endpoint URL is invalid") from None
    if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
        raise ValueError("Gateway endpoint URL is invalid")
    if endpoint.username is not None or endpoint.password is not None:
        raise ValueError("Gateway endpoint URL must not contain user information")
    return endpoint


def _normalize_allowed_host(host: str) -> str:
    normalized = host.strip().lower().rstrip(".")
    if not normalized or any(marker in normalized for marker in "/@?#*"):
        raise ValueError("Gateway allowed host entry is invalid")
    return normalized


class _RejectRedirectHandler(urllib_request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        raise GatewayTransportError("Gateway HTTP redirects are not allowed") from None
