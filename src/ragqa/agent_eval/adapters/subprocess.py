from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

from ragqa.agent_eval.adapters.trace_codec import parse_agent_trace
from ragqa.agent_eval.models import AgentEvalCase, AgentRunTrace
from ragqa.agent_eval.runner import (
    FixtureTraceMismatchError,
    RunnerCaseIdMismatchError,
    RunnerInvalidJSONError,
    RunnerLaunchError,
    RunnerNonZeroExitError,
    RunnerTimeoutError,
)


class SubprocessAgentRunner:
    """Invoke the ai-agent-rag#6 trace CLI without a shell."""

    def __init__(
        self,
        command: Sequence[str],
        timeout_seconds: float = 30.0,
        cwd: str | Path | None = None,
    ) -> None:
        if not command:
            raise ValueError("Subprocess command must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self.command = tuple(command)
        self.timeout_seconds = timeout_seconds
        self.cwd = Path(cwd) if cwd is not None else None

    async def run(self, case: AgentEvalCase) -> AgentRunTrace:
        with tempfile.TemporaryDirectory(prefix="ragqa-agent-trace-") as temp_dir:
            output_path = Path(temp_dir) / "trace.json"
            command = [
                *self.command,
                "--case-id",
                case.id,
                "--question",
                case.input.question,
                "--output",
                str(output_path),
            ]
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=str(self.cwd) if self.cwd is not None else None,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except OSError as exc:
                raise RunnerLaunchError(
                    f"Agent subprocess could not start for case id: {case.id}"
                ) from exc
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=self.timeout_seconds
                )
            except asyncio.TimeoutError as exc:
                process.kill()
                await process.communicate()
                raise RunnerTimeoutError(
                    f"Agent subprocess timed out for case id {case.id} "
                    f"after {self.timeout_seconds}s"
                ) from exc

            if process.returncode != 0:
                raise RunnerNonZeroExitError(
                    case.id,
                    process.returncode or -1,
                    stderr.decode("utf-8", errors="replace"),
                )

            payload = _read_subprocess_payload(output_path, stdout, case.id)
            trace = parse_agent_trace(payload, source=f"subprocess case {case.id}")
            if trace.case_id != case.id:
                raise RunnerCaseIdMismatchError(
                    f"Subprocess returned case id {trace.case_id!r}; expected {case.id!r}"
                )
            if trace.input.question != case.input.question:
                raise FixtureTraceMismatchError(
                    f"Subprocess trace input mismatch for case id: {case.id}"
                )
            return trace


def _read_subprocess_payload(
    output_path: Path, stdout: bytes, case_id: str
) -> object:
    try:
        if output_path.exists():
            with output_path.open(encoding="utf-8") as file:
                return json.load(file)
        if stdout.strip():
            return json.loads(stdout.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RunnerInvalidJSONError(
            f"Agent subprocess returned invalid JSON for case id: {case_id}"
        ) from exc
    raise RunnerInvalidJSONError(
        f"Agent subprocess produced no trace JSON for case id: {case_id}"
    )
