from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from ragqa.agent_eval import (
    AgentEvalCase,
    AgentRunTrace,
    AgentRunner,
    RunnerCaseIdMismatchError,
    RunnerInvalidJSONError,
    TraceFileRunner,
)


def test_trace_file_runner_replays_saved_trace(
    synthetic_cases: list[AgentEvalCase], tmp_path: Path
) -> None:
    source = Path(__file__).resolve().parents[2] / (
        "data/agent_eval/fixtures/phase6_synthetic_traces.json"
    )
    runner = TraceFileRunner(source)

    assert isinstance(runner, AgentRunner)
    trace = asyncio.run(runner.run(synthetic_cases[0]))
    assert trace.case_id == synthetic_cases[0].id


def test_trace_file_runner_supports_one_trace_per_file(
    synthetic_cases: list[AgentEvalCase],
    synthetic_traces: list[AgentRunTrace],
    tmp_path: Path,
) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    for index, trace in enumerate(synthetic_traces[:2]):
        (trace_dir / f"{index}.json").write_text(
            trace.model_dump_json(), encoding="utf-8"
        )

    result = asyncio.run(TraceFileRunner(trace_dir).run(synthetic_cases[1]))

    assert result.case_id == synthetic_cases[1].id


def test_trace_file_runner_rejects_invalid_json(tmp_path: Path) -> None:
    trace_path = tmp_path / "invalid.json"
    trace_path.write_text("{invalid", encoding="utf-8")
    case_payload = {
        "schema_version": "1.0",
        "id": "case-1",
        "category": "direct",
        "severity": "low",
        "input": {"question": "question"},
        "expected": {
            "query_types": ["direct"],
            "routes": ["direct"],
            "citation_required": False,
        },
        "budgets": {"max_latency_ms": 1, "max_cost_usd": 0},
    }
    case = AgentEvalCase.model_validate(case_payload)

    with pytest.raises(RunnerInvalidJSONError, match="Invalid JSON"):
        asyncio.run(TraceFileRunner(trace_path).run(case))


def test_trace_file_runner_rejects_case_id_mismatch(
    synthetic_cases: list[AgentEvalCase],
    synthetic_traces: list[AgentRunTrace],
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(synthetic_traces[1].model_dump(mode="json"), ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(RunnerCaseIdMismatchError, match=synthetic_cases[0].id):
        asyncio.run(TraceFileRunner(trace_path).run(synthetic_cases[0]))
