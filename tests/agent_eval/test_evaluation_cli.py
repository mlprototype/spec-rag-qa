from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_agent_evaluation.py"
TRACES = ROOT / "data" / "agent_eval" / "fixtures" / "phase6_synthetic_traces.json"
BASELINE = ROOT / "data" / "agent_eval" / "baseline" / "agent_baseline.json"


@pytest.mark.parametrize("runner", ["fixture", "trace-file"])
def test_cli_completes_without_api_key(
    runner: str, tmp_path: Path
) -> None:
    output_path = tmp_path / f"{runner}.json"
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env["PYTHONPATH"] = str(ROOT / "src")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--runner",
            runner,
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["total_cases"] == 20
    assert report["evaluated_cases"] == 20
    assert report["execution_errors"] == []
    assert report["evaluation"]["metrics"]["task_success_rate"]["value"] == 1.0
    assert report["gate"]["passed"] is True
    assert report["baseline"]["updated"] is False
    assert output_path.with_suffix(".md").exists()


def test_cli_returns_one_when_evaluation_fails_quality_gate(
    tmp_path: Path,
) -> None:
    traces = json.loads(TRACES.read_text(encoding="utf-8"))
    traces[0]["output"]["route"] = "retrieval"
    traces_path = tmp_path / "regressed_traces.json"
    traces_path.write_text(
        json.dumps(traces, ensure_ascii=False), encoding="utf-8"
    )
    output_path = tmp_path / "regression.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--runner",
            "trace-file",
            "--traces",
            str(traces_path),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )

    assert completed.returncode == 1, completed.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["gate"]["passed"] is False
    assert report["aggregation"]["counts"]["failed_cases"] == 1
    assert report["aggregation"]["counts"]["execution_errors"] == 0


def test_cli_does_not_update_baseline_without_explicit_flag(
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline.json"
    original = BASELINE.read_bytes()
    baseline_path.write_bytes(original)
    output_path = tmp_path / "report.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--runner",
            "fixture",
            "--baseline",
            str(baseline_path),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert baseline_path.read_bytes() == original
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["baseline"]["updated"] is False


def test_cli_reports_runner_failure_separately_from_task_success(
    tmp_path: Path,
) -> None:
    failing_runner = tmp_path / "failing_runner.py"
    failing_runner.write_text("raise SystemExit(9)\n", encoding="utf-8")
    output_path = tmp_path / "subprocess.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--runner",
            "subprocess",
            "--subprocess-command",
            f"{sys.executable} {failing_runner}",
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )

    assert completed.returncode == 2
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["evaluated_cases"] == 0
    assert len(report["execution_errors"]) == report["total_cases"]
    assert report["evaluation"]["cases"] == []
