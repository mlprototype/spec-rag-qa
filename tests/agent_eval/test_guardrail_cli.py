from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_guardrail_evaluation.py"
TRACES = (
    ROOT
    / "data"
    / "agent_eval"
    / "fixtures"
    / "guardrail_synthetic_traces.json"
)


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("GATEWAY_API_KEY", None)
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def test_guardrail_fixture_cli_passes_without_api_key(tmp_path: Path) -> None:
    json_path = tmp_path / "guardrail.json"
    markdown_path = tmp_path / "guardrail.md"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--runner",
            "fixture",
            "--report-json",
            str(json_path),
            "--report-markdown",
            str(markdown_path),
        ],
        cwd=ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["aggregation"]["counts"]["total_cases"] == 30
    assert report["aggregation"]["guardrail"]["overall"]["recall"] == 1.0
    assert report["gate"]["passed"] is True
    assert "Guardrail Confusion Matrix" in markdown_path.read_text(encoding="utf-8")


def test_guardrail_cli_returns_one_for_detection_regression(
    tmp_path: Path,
) -> None:
    traces = json.loads(TRACES.read_text(encoding="utf-8"))
    traces[0]["guardrail"]["detected"] = False
    traces_path = tmp_path / "regressed.json"
    traces_path.write_text(
        json.dumps(traces, ensure_ascii=False), encoding="utf-8"
    )
    report_path = tmp_path / "report.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--runner",
            "fixture",
            "--traces",
            str(traces_path),
            "--report-json",
            str(report_path),
            "--report-markdown",
            str(tmp_path / "report.md"),
        ],
        cwd=ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )

    assert completed.returncode == 1, completed.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["gate"]["passed"] is False
    assert report["aggregation"]["guardrail"]["overall"]["confusion_matrix"]["fn"] == 1


def test_guardrail_http_cli_requires_endpoint_and_allowlist(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "error.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--runner",
            "http",
            "--report-json",
            str(report_path),
            "--report-markdown",
            str(tmp_path / "error.md"),
        ],
        cwd=ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )

    assert completed.returncode == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["preflight_error"]["error_type"] == "DatasetValidationError"
