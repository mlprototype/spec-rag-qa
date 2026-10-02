from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from ragqa.agent_eval.models import AgentRunTrace
from ragqa.agent_eval.runner import RunnerInvalidJSONError


def parse_agent_trace(payload: Any, source: str = "runner output") -> AgentRunTrace:
    """Normalize ai-agent-rag#6 observations and validate the shared trace contract."""

    if not isinstance(payload, dict):
        raise RunnerInvalidJSONError(f"Agent trace must be a JSON object: {source}")

    normalized = deepcopy(payload)
    _normalize_timing(normalized)
    _normalize_usage(normalized)
    _normalize_source_citations(normalized)

    try:
        return AgentRunTrace.model_validate(normalized)
    except ValidationError as exc:
        raise RunnerInvalidJSONError(f"Invalid AgentRunTrace JSON: {source}") from exc


def _normalize_timing(payload: dict[str, Any]) -> None:
    timing = payload.get("timing")
    if not isinstance(timing, dict):
        return
    if "latency_ms" not in timing and "total_latency_ms" in timing:
        timing["latency_ms"] = timing.pop("total_latency_ms")


def _normalize_usage(payload: dict[str, Any]) -> None:
    if payload.get("usage") is None:
        payload["usage"] = {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "cost_usd": None,
        }


def _normalize_source_citations(payload: dict[str, Any]) -> None:
    sources = payload.get("sources")
    if not isinstance(sources, list):
        return

    citations = payload.get("citations")
    if citations is None:
        citations = []
        payload["citations"] = citations
    if not isinstance(citations, list):
        return

    known_citation_ids = {
        str(item.get("citation_id"))
        for item in citations
        if isinstance(item, dict) and item.get("citation_id") is not None
    }
    for source in sources:
        if not isinstance(source, dict) or "citation_id" not in source:
            continue
        citation_id = str(source.pop("citation_id"))
        source_id = source.get("source_id")
        if source_id is None or citation_id in known_citation_ids:
            continue
        citations.append({"citation_id": citation_id, "source_id": str(source_id)})
        known_citation_ids.add(citation_id)

    for citation in citations:
        if isinstance(citation, dict) and citation.get("citation_id") is not None:
            citation["citation_id"] = str(citation["citation_id"])
