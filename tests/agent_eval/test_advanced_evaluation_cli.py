from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_agent_advanced_evaluation.py"


def test_advanced_cli_runs_offline_with_mock_judge(tmp_path: Path) -> None:
    json_path = tmp_path / "advanced.json"
    markdown_path = tmp_path / "advanced.md"
    env = os.environ.copy()
    env.pop("AGENT_EVAL_JUDGE_API_KEY", None)
    env["PYTHONPATH"] = str(ROOT / "src")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--runner",
            "fixture",
            "--judge",
            "mock",
            "--report-json",
            str(json_path),
            "--report-markdown",
            str(markdown_path),
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
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["monitor_only"] == [
        "groundedness",
        "answer_semantic_consistency",
        "cost",
    ]
    assert report["summary"]["runs"]["requested"] == 30
    assert report["summary"]["runs"]["execution_errors"] == 0
    assert report["pricing_version"]
    assert markdown_path.is_file()


def test_http_judge_preflight_failure_is_exit_two_and_writes_reports(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "advanced-error.json"
    markdown_path = tmp_path / "advanced-error.md"
    env = os.environ.copy()
    env.pop("AGENT_EVAL_JUDGE_API_KEY", None)
    env["PYTHONPATH"] = str(ROOT / "src")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--judge",
            "http",
            "--judge-model",
            "external-model",
            "--report-json",
            str(json_path),
            "--report-markdown",
            str(markdown_path),
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
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["preflight_error"]["error_type"] == "ValueError"
    assert "--judge-url" in report["preflight_error"]["message"]
    assert markdown_path.is_file()
