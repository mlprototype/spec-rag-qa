from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from ragqa.agent_eval import (
    AgentEvalCase,
    AgentRunTrace,
    RunnerCaseIdMismatchError,
    RunnerInvalidJSONError,
    RunnerLaunchError,
    RunnerNonZeroExitError,
    RunnerTimeoutError,
    SubprocessAgentRunner,
)
from ragqa.agent_eval.adapters.trace_codec import parse_agent_trace


def _write_fake_trace_cli(path: Path) -> None:
    path.write_text(
        """\
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--trace-source', type=Path, required=True)
parser.add_argument('--case-id', required=True)
parser.add_argument('--question', required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
traces = json.loads(args.trace_source.read_text(encoding='utf-8'))
trace = next(item for item in traces if item['case_id'] == args.case_id)
args.output.write_text(json.dumps(trace, ensure_ascii=False), encoding='utf-8')
""",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "category",
    ["direct", "structured_query", "retrieval", "compare"],
)
def test_subprocess_runner_executes_representative_routes(
    category: str,
    synthetic_cases: list[AgentEvalCase],
    tmp_path: Path,
) -> None:
    trace_source = Path(__file__).resolve().parents[2] / (
        "data/agent_eval/fixtures/phase6_synthetic_traces.json"
    )
    script = tmp_path / "fake_trace_cli.py"
    _write_fake_trace_cli(script)
    case = next(item for item in synthetic_cases if item.category == category)
    runner = SubprocessAgentRunner(
        [sys.executable, str(script), "--trace-source", str(trace_source)],
        timeout_seconds=2,
    )

    trace = asyncio.run(runner.run(case))

    assert trace.case_id == case.id
    assert trace.input.question == case.input.question


def test_subprocess_runner_raises_timeout(
    synthetic_cases: list[AgentEvalCase], tmp_path: Path
) -> None:
    script = tmp_path / "sleep.py"
    script.write_text("import time\ntime.sleep(1)\n", encoding="utf-8")
    runner = SubprocessAgentRunner(
        [sys.executable, str(script)], timeout_seconds=0.01
    )

    with pytest.raises(RunnerTimeoutError, match=synthetic_cases[0].id):
        asyncio.run(runner.run(synthetic_cases[0]))


def test_subprocess_runner_raises_non_zero_exit(
    synthetic_cases: list[AgentEvalCase], tmp_path: Path
) -> None:
    script = tmp_path / "fail.py"
    script.write_text(
        "import sys\nsys.stderr.write('private detail')\nsys.exit(7)\n",
        encoding="utf-8",
    )
    runner = SubprocessAgentRunner([sys.executable, str(script)])

    with pytest.raises(RunnerNonZeroExitError) as exc_info:
        asyncio.run(runner.run(synthetic_cases[0]))

    assert exc_info.value.returncode == 7
    assert exc_info.value.stderr == "private detail"
    assert "private detail" not in str(exc_info.value)


def test_subprocess_runner_raises_launch_error(
    synthetic_cases: list[AgentEvalCase], tmp_path: Path
) -> None:
    runner = SubprocessAgentRunner([str(tmp_path / "missing-command")])

    with pytest.raises(RunnerLaunchError, match=synthetic_cases[0].id):
        asyncio.run(runner.run(synthetic_cases[0]))


def test_subprocess_runner_rejects_invalid_json(
    synthetic_cases: list[AgentEvalCase], tmp_path: Path
) -> None:
    script = tmp_path / "invalid.py"
    script.write_text("print('{invalid')\n", encoding="utf-8")
    runner = SubprocessAgentRunner([sys.executable, str(script)])

    with pytest.raises(RunnerInvalidJSONError, match="invalid JSON"):
        asyncio.run(runner.run(synthetic_cases[0]))


def test_subprocess_runner_rejects_case_id_mismatch(
    synthetic_cases: list[AgentEvalCase],
    synthetic_traces: list[AgentRunTrace],
    tmp_path: Path,
) -> None:
    script = tmp_path / "mismatch.py"
    trace = synthetic_traces[0].model_copy(deep=True)
    trace.case_id = "different-case"
    script.write_text(
        """\
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--case-id')
parser.add_argument('--question')
parser.add_argument('--output', type=Path)
args = parser.parse_args()
args.output.write_text(%r, encoding='utf-8')
"""
        % trace.model_dump_json(),
        encoding="utf-8",
    )
    runner = SubprocessAgentRunner([sys.executable, str(script)])

    with pytest.raises(RunnerCaseIdMismatchError, match="different-case"):
        asyncio.run(runner.run(synthetic_cases[0]))


def test_ai_agent_trace_codec_normalizes_nullable_usage_and_inline_citation(
    synthetic_traces: list[AgentRunTrace],
) -> None:
    payload = synthetic_traces[3].model_dump(mode="json")
    payload["usage"] = None
    payload["timing"]["total_latency_ms"] = payload["timing"].pop("latency_ms")
    citation = payload["citations"].pop()
    source = next(
        item for item in payload["sources"] if item["source_id"] == citation["source_id"]
    )
    source["citation_id"] = 1
    payload["output"]["answer"] = payload["output"]["answer"].replace(
        f"[{citation['citation_id']}]", "[1]"
    )

    trace = parse_agent_trace(payload)

    assert trace.timing.latency_ms is not None
    assert trace.usage.total_tokens is None
    assert trace.citations[0].citation_id == "1"
    assert trace.citations[0].source_id == source["source_id"]
