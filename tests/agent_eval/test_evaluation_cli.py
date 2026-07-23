from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_agent_evaluation.py"


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
