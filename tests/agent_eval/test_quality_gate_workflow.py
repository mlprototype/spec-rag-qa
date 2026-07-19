from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "agent-quality-gate.yml"
REPORTS_PATH = ROOT / "data" / "agent_eval" / "reports"


def test_offline_ci_uses_a_fresh_untracked_artifact_directory() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    offline_job = workflow.split("  real-agent-evaluation:", maxsplit=1)[0]

    cleanup_index = offline_job.index("rm -rf .artifacts/agent-quality")
    evaluation_index = offline_job.index("python scripts/run_agent_evaluation.py")

    assert cleanup_index < evaluation_index
    assert "mkdir -p .artifacts/agent-quality" in offline_job
    assert "--report-json .artifacts/agent-quality/report.json" in offline_job
    assert "--report-markdown .artifacts/agent-quality/report.md" in offline_job
    assert "tee .artifacts/agent-quality/run.log" in offline_job
    assert "set -o pipefail" in offline_job
    assert "data/agent_eval/reports/latest" not in offline_job


def test_offline_ci_always_uploads_current_reports_and_log() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    upload_step = workflow.split(
        "- name: Upload Agent quality reports", maxsplit=1
    )[1].split("  real-agent-evaluation:", maxsplit=1)[0]

    assert "if: always()" in upload_step
    assert "include-hidden-files: true" in upload_step
    assert ".artifacts/agent-quality/report.json" in upload_step
    assert ".artifacts/agent-quality/report.md" in upload_step
    assert ".artifacts/agent-quality/run.log" in upload_step


def test_generated_artifacts_are_ignored_and_tracked_reports_are_examples() -> None:
    gitignore_entries = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert ".artifacts/" in gitignore_entries
    assert (REPORTS_PATH / "example.json").is_file()
    assert (REPORTS_PATH / "example.md").is_file()
    assert not (REPORTS_PATH / "latest.json").exists()
    assert not (REPORTS_PATH / "latest.md").exists()
