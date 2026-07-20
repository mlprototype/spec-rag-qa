from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Callable, Collection, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Protocol, TypeVar, runtime_checkable
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from pydantic import BaseModel, ValidationError

from ragqa.agent_eval.advanced_models import (
    GroundednessJudgeDecision,
    GroundednessJudgeResponse,
    JudgeExecutionMetadata,
    JudgeRequest,
    SemanticConsistencyJudgeDecision,
    SemanticConsistencyJudgeResponse,
)
from ragqa.agent_eval.models import AgentRunTrace, ToolCallTrace


GROUNDING_PROMPT_VERSION = "groundedness.claim-support.v2"
SEMANTIC_STABILITY_PROMPT_VERSION = "stability.semantic-groups.v1"

STRUCTURED_QUERY_EVIDENCE_KIND = "structured_query_result"
_ALLOWED_GROUNDEDNESS_TOOL = "structured_query_tool"
_STRUCTURED_QUERY_ARGUMENT_FIELDS = (
    "operation",
    "target_metric",
    "filters",
    "target_dataset",
)
_STRUCTURED_QUERY_RESULT_CONTEXT_FIELDS = (
    "operation",
    "target_metric",
    "filters",
    "target_dataset",
    "row_count",
)
_STRUCTURED_QUERY_FACT_FIELDS = (
    "value",
    "values",
    "result_value",
    "aggregate",
    "aggregates",
    "count",
    "total",
    "average",
    "minimum",
    "maximum",
    "rows",
    "records",
    "metrics",
)
_PROVENANCE_ONLY_KEYS = {
    "citation_id",
    "citation_ids",
    "document_id",
    "document_ids",
    "source_count",
    "source_id",
    "source_ids",
}
_MAX_EVIDENCE_ROWS = 20
_MAX_EVIDENCE_FIELDS = 20
_MAX_EVIDENCE_DEPTH = 4
_MAX_EVIDENCE_STRING_LENGTH = 1_000
_MAX_PROJECTED_EVIDENCE_BYTES = 16_384

GROUNDING_PROMPT = """You are an independent factual-grounding evaluator.
Split the answer into independently verifiable claims. Judge each evaluable claim only
against the supplied sources and the projected factual values in eligible deterministic
Tool results. Tool names, arguments, source IDs, and row/source counts alone are not
claim support. Never use the Agent's confidence, critic, answer_ok, self-assessment, or
similar self-evaluation as evidence. Return JSON with schema_version and claims. Each
claim contains claim, evaluable, supported, source_ids, tool_result_ids, and reason. Use
supported=null only when evaluable=false. A supported claim must cite at least one
supplied source_id or tool_result_id. Return claims=[] when the answer has no claim."""

SEMANTIC_STABILITY_PROMPT = """You are an independent semantic-consistency evaluator.
Group answers that express the same material conclusion even when wording, ordering, or
formatting differs. Do not use exact string equality. Return JSON with schema_version,
one non-negative group_id per answer in input order, and a concise reason."""


class JudgeError(RuntimeError):
    """Base class for independent Judge execution failures."""


class JudgeTransportError(JudgeError):
    """Raised when a Judge provider cannot return a response."""


class JudgeMalformedResponseError(JudgeError):
    """Raised after the bounded malformed-response retry is exhausted."""

    def __init__(self, task: str, attempts: int, detail: str) -> None:
        super().__init__(
            f"Malformed Judge response for {task} after {attempts} attempt(s): {detail}"
        )
        self.task = task
        self.attempts = attempts


@runtime_checkable
class JudgeTransport(Protocol):
    """Provider transport kept independent from the Agent runner."""

    async def complete(self, request: JudgeRequest) -> str | Mapping[str, Any]:
        ...


@runtime_checkable
class JudgeAdapter(Protocol):
    """Advanced evaluation contract for Groundedness and semantic stability."""

    judge_model: str
    grounding_prompt_version: str
    semantic_prompt_version: str

    async def judge_groundedness(
        self, trace: AgentRunTrace
    ) -> GroundednessJudgeDecision:
        ...

    async def judge_semantic_consistency(
        self,
        case_id: str,
        answers: Sequence[str],
        *,
        schema_version: str = "1.0",
    ) -> SemanticConsistencyJudgeDecision:
        ...


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class StructuredJudgeAdapter:
    """Validate provider output and retry malformed JSON at most once."""

    grounding_prompt_version = GROUNDING_PROMPT_VERSION
    semantic_prompt_version = SEMANTIC_STABILITY_PROMPT_VERSION

    def __init__(
        self,
        transport: JudgeTransport,
        *,
        judge_model: str,
        malformed_retries: int = 1,
    ) -> None:
        if not judge_model.strip():
            raise ValueError("judge_model must not be empty")
        if malformed_retries not in {0, 1}:
            raise ValueError("malformed_retries must be 0 or 1")
        self.transport = transport
        self.judge_model = judge_model
        self.malformed_retries = malformed_retries

    async def judge_groundedness(
        self, trace: AgentRunTrace
    ) -> GroundednessJudgeDecision:
        sources = [
            {
                "source_id": source.source_id,
                "title": source.title,
                "uri": source.uri,
                "snippet": source.snippet,
            }
            for source in trace.sources
            if source.snippet is not None and source.snippet.strip()
        ]
        tool_results = [
            projected
            for index, call in enumerate(trace.tool_calls)
            if (projected := _project_tool_evidence(index, call)) is not None
        ]
        request = JudgeRequest(
            schema_version=trace.schema_version,
            task="groundedness",
            prompt_version=self.grounding_prompt_version,
            prompt=GROUNDING_PROMPT,
            input={
                "case_id": trace.case_id,
                "run_id": trace.run_id,
                "question": trace.input.question,
                "answer": trace.output.answer,
                "sources": sources,
                "tool_results": tool_results,
            },
        )
        source_ids = {str(source["source_id"]) for source in sources}
        tool_result_ids = {
            str(tool_result["tool_result_id"]) for tool_result in tool_results
        }

        def validate(response: GroundednessJudgeResponse) -> None:
            if response.schema_version != trace.schema_version:
                raise ValueError("Judge schema_version does not match the trace")
            for claim in response.claims:
                unknown_sources = set(claim.source_ids) - source_ids
                unknown_tools = set(claim.tool_result_ids) - tool_result_ids
                if unknown_sources or unknown_tools:
                    raise ValueError("Judge claim references unknown evidence")
                if claim.evaluable and claim.supported and not (
                    claim.source_ids or claim.tool_result_ids
                ):
                    raise ValueError("A supported claim must reference supplied evidence")

        response, metadata = await self._request_structured(
            request,
            GroundednessJudgeResponse,
            validate,
        )
        return GroundednessJudgeDecision(response=response, judge=metadata)

    async def judge_semantic_consistency(
        self,
        case_id: str,
        answers: Sequence[str],
        *,
        schema_version: str = "1.0",
    ) -> SemanticConsistencyJudgeDecision:
        if len(answers) < 2:
            raise ValueError("Semantic consistency requires at least two answers")
        request = JudgeRequest(
            schema_version=schema_version,
            task="semantic_stability",
            prompt_version=self.semantic_prompt_version,
            prompt=SEMANTIC_STABILITY_PROMPT,
            input={
                "case_id": case_id,
                "answers": [
                    {"index": index, "answer": answer}
                    for index, answer in enumerate(answers)
                ],
            },
        )

        def validate(response: SemanticConsistencyJudgeResponse) -> None:
            if response.schema_version != request.schema_version:
                raise ValueError("Judge schema_version does not match the request")
            if len(response.group_ids) != len(answers):
                raise ValueError("Judge must return one group_id per answer")

        response, metadata = await self._request_structured(
            request,
            SemanticConsistencyJudgeResponse,
            validate,
        )
        return SemanticConsistencyJudgeDecision(response=response, judge=metadata)

    async def _request_structured(
        self,
        request: JudgeRequest,
        response_model: type[ResponseModel],
        validate: Callable[[ResponseModel], None],
    ) -> tuple[ResponseModel, JudgeExecutionMetadata]:
        max_attempts = self.malformed_retries + 1
        last_detail = "unknown validation error"
        for attempt in range(1, max_attempts + 1):
            raw = await self.transport.complete(request)
            try:
                payload = json.loads(raw) if isinstance(raw, str) else dict(raw)
                response = response_model.model_validate(payload)
                validate(response)
            except (
                json.JSONDecodeError,
                TypeError,
                ValidationError,
                ValueError,
            ) as exc:
                last_detail = _safe_validation_detail(exc)
                continue
            return response, JudgeExecutionMetadata(
                schema_version=request.schema_version,
                judge_model=self.judge_model,
                judge_prompt_version=request.prompt_version,
                evaluated_at=datetime.now(timezone.utc),
                attempts=attempt,
            )
        raise JudgeMalformedResponseError(request.task, max_attempts, last_detail)


class HttpJudgeTransport:
    """POST a provider-neutral JudgeRequest and return its structured response."""

    def __init__(
        self,
        url: str,
        *,
        api_key: str | None = None,
        allowed_hosts: Collection[str],
        timeout_seconds: float = 60.0,
    ) -> None:
        endpoint = _parse_judge_endpoint(url)
        if api_key and endpoint.scheme != "https":
            raise ValueError(
                "Judge endpoint must use HTTPS when an API key is configured"
            )
        allowed_host_values = (
            (allowed_hosts,)
            if isinstance(allowed_hosts, str)
            else allowed_hosts
        )
        normalized_hosts = frozenset(
            _normalize_allowed_host(host)
            for host in allowed_host_values
            if host.strip()
        )
        if not normalized_hosts:
            raise ValueError("Judge endpoint host allowlist must not be empty")
        endpoint_host = _normalize_allowed_host(endpoint.hostname or "")
        if endpoint_host not in normalized_hosts:
            raise ValueError(
                "Judge endpoint host is not in the configured allowlist"
            )
        if timeout_seconds <= 0:
            raise ValueError("Judge timeout must be greater than zero")
        self.url = url
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._opener = urllib_request.build_opener(_RejectRedirectHandler())

    async def complete(self, request: JudgeRequest) -> str:
        return await asyncio.to_thread(self._complete_sync, request)

    def _complete_sync(self, request: JudgeRequest) -> str:
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = request.model_dump_json().encode("utf-8")
        try:
            provider_request = urllib_request.Request(
                self.url,
                data=body,
                headers=headers,
                method="POST",
            )
            with self._opener.open(
                provider_request,
                timeout=self.timeout_seconds,
            ) as response:
                return response.read().decode("utf-8")
        except JudgeTransportError:
            raise
        except urllib_error.HTTPError as exc:
            raise JudgeTransportError(
                f"Judge HTTP request failed with status {exc.code}"
            ) from None
        except (
            urllib_error.URLError,
            UnicodeDecodeError,
            TimeoutError,
            ValueError,
        ):
            raise JudgeTransportError("Judge HTTP request failed") from None


class DeterministicMockJudgeTransport:
    """Offline plumbing-only Judge; its scores are never suitable as a quality Gate."""

    async def complete(self, request: JudgeRequest) -> Mapping[str, Any]:
        if request.task == "semantic_stability":
            answers = request.input.get("answers", [])
            return {
                "schema_version": request.schema_version,
                "group_ids": [0 for _ in answers],
                "reason": "Offline mock groups all supplied answers together.",
            }

        answer = str(request.input.get("answer", "")).strip()
        if not answer:
            return {"schema_version": request.schema_version, "claims": []}
        sources = request.input.get("sources", [])
        tool_results = request.input.get("tool_results", [])
        source_id = (
            str(sources[0]["source_id"])
            if isinstance(sources, list) and sources
            else None
        )
        tool_result_id = (
            str(tool_results[0]["tool_result_id"])
            if isinstance(tool_results, list) and tool_results
            else None
        )
        evaluable = source_id is not None or tool_result_id is not None
        return {
            "schema_version": request.schema_version,
            "claims": [
                {
                    "claim": answer,
                    "evaluable": evaluable,
                    "supported": True if evaluable else None,
                    "source_ids": [source_id] if source_id is not None else [],
                    "tool_result_ids": (
                        [tool_result_id] if tool_result_id is not None else []
                    ),
                    "reason": "Offline mock response for adapter and report testing.",
                }
            ],
        }


def _tool_result_id(index: int, tool_name: str) -> str:
    return f"tool:{index}:{tool_name}"


def _project_tool_evidence(
    index: int,
    call: ToolCallTrace,
) -> dict[str, Any] | None:
    """Project explicitly approved Tool facts; every other Tool is denied."""

    if call.error is not None or call.result is None:
        return None
    if call.metadata.get("deterministic") is not True:
        return None
    if call.name != _ALLOWED_GROUNDEDNESS_TOOL:
        return None
    if call.metadata.get("evidence_kind") != STRUCTURED_QUERY_EVIDENCE_KIND:
        return None
    if not isinstance(call.result, Mapping):
        return None
    if call.result.get("success") is not True:
        return None

    projected_facts: dict[str, Any] = {}
    for field in _STRUCTURED_QUERY_FACT_FIELDS:
        if field not in call.result:
            continue
        value = _project_fact_value(call.result[field])
        if _has_factual_content(value):
            projected_facts[field] = value
    if not projected_facts:
        return None

    projected_arguments = _project_selected_fields(
        call.arguments,
        _STRUCTURED_QUERY_ARGUMENT_FIELDS,
    )
    projected_result = {
        "success": True,
        **_project_selected_fields(
            call.result,
            _STRUCTURED_QUERY_RESULT_CONTEXT_FIELDS,
        ),
        **projected_facts,
    }
    projected = {
        "tool_result_id": _tool_result_id(index, call.name),
        "tool_name": call.name,
        "evidence_kind": STRUCTURED_QUERY_EVIDENCE_KIND,
        "arguments": projected_arguments,
        "result": projected_result,
    }
    try:
        encoded = json.dumps(
            projected,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return projected if len(encoded) <= _MAX_PROJECTED_EVIDENCE_BYTES else None


def _project_selected_fields(
    value: Mapping[str, Any],
    fields: Sequence[str],
) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    for field in fields:
        if field not in value:
            continue
        item = _project_fact_value(value[field])
        if _has_factual_content(item):
            projected[field] = item
    return projected


def _project_fact_value(value: Any, *, depth: int = 0) -> Any | None:
    if depth > _MAX_EVIDENCE_DEPTH:
        return None
    if isinstance(value, Mapping):
        if len(value) > _MAX_EVIDENCE_FIELDS:
            return None
        projected = {}
        for key, item in value.items():
            normalized_key = _normalize_evidence_key(key)
            if (
                _is_self_evaluation_key(normalized_key)
                or normalized_key in _PROVENANCE_ONLY_KEYS
            ):
                continue
            projected_item = _project_fact_value(item, depth=depth + 1)
            if _has_factual_content(projected_item):
                projected[str(key)] = projected_item
        return projected or None
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_EVIDENCE_ROWS:
            return None
        projected_items = [
            projected
            for item in value
            if (projected := _project_fact_value(item, depth=depth + 1))
            is not None
        ]
        return projected_items or None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or len(stripped) > _MAX_EVIDENCE_STRING_LENGTH:
            return None
        return stripped
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (bool, int, float)):
        return value
    return None


def _has_factual_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (Mapping, list, tuple, str)):
        return bool(value)
    return isinstance(value, (bool, int, float))


def _normalize_evidence_key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _is_self_evaluation_key(normalized_key: str) -> bool:
    parts = set(normalized_key.split("_"))
    if parts.intersection({"confidence", "critic", "critique", "judge"}):
        return True
    if "answer" in parts and parts.intersection(
        {"correct", "evaluation", "ok", "quality", "score", "verdict"}
    ):
        return True
    if "self" in parts and parts.intersection(
        {"assessment", "eval", "evaluation", "review", "score"}
    ):
        return True
    return normalized_key in {
        "answer_ok",
        "groundedness_score",
        "self_assessment",
        "support_score",
    }


def _safe_validation_detail(exc: Exception) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return "response was not valid JSON"
    if isinstance(exc, ValidationError):
        return "response did not match the required schema"
    if isinstance(exc, TypeError):
        return "response was not a JSON object"
    message = str(exc)
    safe_messages = {
        "A supported claim must reference supplied evidence",
        "Judge claim references unknown evidence",
        "Judge must return one group_id per answer",
        "Judge schema_version does not match the request",
        "Judge schema_version does not match the trace",
    }
    return (
        message
        if message in safe_messages
        else "response failed semantic validation"
    )


def _parse_judge_endpoint(url: str) -> urllib_parse.SplitResult:
    try:
        endpoint = urllib_parse.urlsplit(url)
        _ = endpoint.port
    except ValueError:
        raise ValueError("Judge endpoint URL is invalid") from None
    if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
        raise ValueError("Judge endpoint URL is invalid")
    if endpoint.username is not None or endpoint.password is not None:
        raise ValueError("Judge endpoint URL must not contain user information")
    return endpoint


def _normalize_allowed_host(host: str) -> str:
    normalized = host.strip().lower().rstrip(".")
    invalid_markers = ("/", "@", "?", "#", "*")
    if not normalized or any(marker in normalized for marker in invalid_markers):
        raise ValueError("Judge allowed host entry is invalid")
    return normalized


class _RejectRedirectHandler(urllib_request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        raise JudgeTransportError("Judge HTTP redirects are not allowed") from None
