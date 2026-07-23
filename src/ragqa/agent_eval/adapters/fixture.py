from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from ragqa.agent_eval.models import AgentEvalCase, AgentRunTrace
from ragqa.agent_eval.runner import FixtureTraceNotFoundError


class DuplicateFixtureTraceError(ValueError):
    """Raised when multiple fixture traces target the same case ID."""


class FixtureRunner:
    """Asynchronous runner that returns deterministic traces keyed by case ID."""

    def __init__(self, traces: Iterable[AgentRunTrace]) -> None:
        self._traces: dict[str, AgentRunTrace] = {}
        for trace in traces:
            if trace.case_id in self._traces:
                raise DuplicateFixtureTraceError(
                    f"Duplicate fixture trace for case id: {trace.case_id}"
                )
            self._traces[trace.case_id] = trace

    @classmethod
    def from_json(cls, path: str | Path) -> FixtureRunner:
        """Build a runner from a UTF-8 JSON array of fixture traces."""

        fixture_path = Path(path)
        with fixture_path.open(encoding="utf-8") as file:
            payload = json.load(file)

        if not isinstance(payload, list):
            raise ValueError(f"Fixture traces must be a JSON array: {fixture_path}")

        return cls(AgentRunTrace.model_validate(item) for item in payload)

    async def run(self, case: AgentEvalCase) -> AgentRunTrace:
        """Return a deep copy of the trace registered for ``case.id``."""

        try:
            trace = self._traces[case.id]
        except KeyError as exc:
            raise FixtureTraceNotFoundError(
                f"Fixture trace not found for case id: {case.id}"
            ) from exc
        return trace.model_copy(deep=True)

