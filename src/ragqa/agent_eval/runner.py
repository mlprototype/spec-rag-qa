from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from ragqa.agent_eval.models import AgentEvalCase, AgentRunTrace


class DuplicateCaseIdError(ValueError):
    """Raised when an evaluation case collection contains duplicate IDs."""


class FixtureTraceNotFoundError(LookupError):
    """Raised when a fixture has no trace for a requested case."""


class FixtureTraceMismatchError(ValueError):
    """Raised when a fixture trace does not correspond to the requested case."""


class SchemaVersionMismatchError(ValueError):
    """Raised when incompatible Agent evaluation schema versions are mixed."""


class AgentRunnerError(RuntimeError):
    """Base class for execution failures outside deterministic evaluation."""


class RunnerTimeoutError(AgentRunnerError):
    """Raised when an Agent runner exceeds its execution timeout."""


class RunnerLaunchError(AgentRunnerError):
    """Raised when an Agent subprocess cannot be started."""


class RunnerNonZeroExitError(AgentRunnerError):
    """Raised when an Agent subprocess exits unsuccessfully."""

    def __init__(self, case_id: str, returncode: int, stderr: str = "") -> None:
        super().__init__(
            f"Agent subprocess failed for case id {case_id} with exit code {returncode}"
        )
        self.case_id = case_id
        self.returncode = returncode
        self.stderr = stderr


class RunnerInvalidJSONError(AgentRunnerError):
    """Raised when runner output is not valid AgentRunTrace JSON."""


class RunnerCaseIdMismatchError(AgentRunnerError):
    """Raised when runner output belongs to a different case."""


class TraceFileNotFoundError(AgentRunnerError):
    """Raised when a requested saved trace cannot be found."""


@runtime_checkable
class AgentRunner(Protocol):
    """Target-independent contract for executing one Agent evaluation case."""

    async def run(self, case: AgentEvalCase) -> AgentRunTrace:
        ...


def load_cases(path: str | Path) -> list[AgentEvalCase]:
    """Load and validate an Agent evaluation case list from UTF-8 JSON."""

    case_path = Path(path)
    with case_path.open(encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError(f"Agent evaluation cases must be a JSON array: {case_path}")

    cases = [AgentEvalCase.model_validate(item) for item in payload]
    _ensure_unique_case_ids(cases, case_path)
    return cases


def _ensure_unique_case_ids(
    cases: Sequence[AgentEvalCase], source: str | Path = "case collection"
) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for case in cases:
        if case.id in seen:
            duplicates.add(case.id)
        seen.add(case.id)

    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise DuplicateCaseIdError(f"Duplicate case id(s) in {source}: {duplicate_list}")
