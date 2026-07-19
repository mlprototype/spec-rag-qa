from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Protocol, TypeVar, runtime_checkable
from urllib import error as urllib_error
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


GROUNDING_PROMPT_VERSION = "groundedness.claim-support.v1"
SEMANTIC_STABILITY_PROMPT_VERSION = "stability.semantic-groups.v1"

GROUNDING_PROMPT = """You are an independent factual-grounding evaluator.
Split the answer into independently verifiable claims. Judge each evaluable claim only
against the supplied sources and successful deterministic Tool results. Never use the
Agent's confidence, critic, answer_ok, or self-assessment as evidence. Return JSON with
schema_version and claims. Each claim contains claim, evaluable, supported, source_ids,
tool_result_ids, and reason. Use supported=null only when evaluable=false. A supported
claim must cite at least one supplied source_id or tool_result_id. Return claims=[] when
the answer contains no evaluable or non-evaluable claim."""

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
        ]
        tool_results = [
            {
                "tool_result_id": _tool_result_id(index, call.name),
                "tool_name": call.name,
                "arguments": _sanitize_evidence(call.arguments),
                "result": _sanitize_evidence(call.result),
            }
            for index, call in enumerate(trace.tool_calls)
            if _is_deterministic_tool_result(call)
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
                last_detail = str(exc)
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
        timeout_seconds: float = 60.0,
    ) -> None:
        if not url.startswith(("https://", "http://")):
            raise ValueError("Judge URL must use http or https")
        if timeout_seconds <= 0:
            raise ValueError("Judge timeout must be greater than zero")
        self.url = url
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    async def complete(self, request: JudgeRequest) -> str:
        return await asyncio.to_thread(self._complete_sync, request)

    def _complete_sync(self, request: JudgeRequest) -> str:
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = request.model_dump_json().encode("utf-8")
        provider_request = urllib_request.Request(
            self.url,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib_request.urlopen(
                provider_request, timeout=self.timeout_seconds
            ) as response:
                return response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            raise JudgeTransportError(
                f"Judge HTTP request failed with status {exc.code}"
            ) from exc
        except (urllib_error.URLError, UnicodeDecodeError, TimeoutError) as exc:
            raise JudgeTransportError("Judge HTTP request failed") from exc


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


def _is_deterministic_tool_result(call: ToolCallTrace) -> bool:
    if call.error is not None or call.result is None:
        return False
    if call.metadata.get("deterministic") is False:
        return False
    normalized_name = call.name.lower().replace("-", "_")
    self_evaluation_markers = (
        "critic",
        "judge",
        "answer_check",
        "answer_ok",
        "self_eval",
        "confidence",
    )
    return not any(marker in normalized_name for marker in self_evaluation_markers)


def _sanitize_evidence(value: Any) -> Any:
    forbidden_keys = {"answer_ok", "critic", "confidence", "self_assessment"}
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_evidence(item)
            for key, item in value.items()
            if str(key).lower() not in forbidden_keys
        }
    if isinstance(value, list):
        return [_sanitize_evidence(item) for item in value]
    return value
