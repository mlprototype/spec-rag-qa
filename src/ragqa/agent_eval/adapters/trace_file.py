from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ragqa.agent_eval.adapters.trace_codec import parse_agent_trace
from ragqa.agent_eval.models import AgentEvalCase, AgentRunTrace
from ragqa.agent_eval.runner import (
    FixtureTraceMismatchError,
    RunnerCaseIdMismatchError,
    RunnerInvalidJSONError,
    TraceFileNotFoundError,
)


class TraceFileRunner:
    """Re-evaluate saved traces from one JSON file or a directory of JSON files."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    async def run(self, case: AgentEvalCase) -> AgentRunTrace:
        traces = load_saved_traces(self.path)
        matches = [trace for trace in traces if trace.case_id == case.id]
        if not matches:
            available = sorted(trace.case_id for trace in traces)
            raise RunnerCaseIdMismatchError(
                f"Saved trace case_id mismatch for {case.id}; available={available}"
            )
        if len(matches) > 1:
            raise RunnerInvalidJSONError(
                f"Duplicate saved traces for case id: {case.id}"
            )
        trace = matches[0]
        if trace.input.question != case.input.question:
            raise FixtureTraceMismatchError(
                f"Saved trace input mismatch for case id: {case.id}"
            )
        return trace.model_copy(deep=True)


def load_saved_traces(path: str | Path) -> list[AgentRunTrace]:
    trace_path = Path(path)
    if not trace_path.exists():
        raise TraceFileNotFoundError(f"Saved trace path not found: {trace_path}")

    if trace_path.is_dir():
        files = sorted(trace_path.glob("*.json"))
        if not files:
            raise TraceFileNotFoundError(
                f"No JSON trace files found in directory: {trace_path}"
            )
        traces: list[AgentRunTrace] = []
        for file_path in files:
            traces.extend(_load_json_trace_payload(file_path))
        return traces

    return _load_json_trace_payload(trace_path)


def _load_json_trace_payload(path: Path) -> list[AgentRunTrace]:
    try:
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RunnerInvalidJSONError(f"Invalid JSON in saved trace: {path}") from exc

    items: list[Any]
    if isinstance(payload, list):
        items = payload
    else:
        items = [payload]
    return [parse_agent_trace(item, source=str(path)) for item in items]
