from __future__ import annotations

import json
from pathlib import Path

from ragqa.agent_eval.report import render_markdown, write_reports


def test_report_preserves_na_in_json_and_markdown(tmp_path: Path) -> None:
    report = {
        "generated_at": "fixed",
        "runner": "fixture",
        "baseline": {"updated": False},
        "execution_errors": [],
        "aggregation": {
            "counts": {
                "total_cases": 1,
                "evaluated_cases": 1,
                "passed_cases": 1,
                "failed_cases": 0,
                "execution_errors": 0,
            },
            "metrics": {
                "citation_validity_rate": {
                    "numerator": 0,
                    "denominator": 0,
                    "value": None,
                }
            },
            "latency": {
                "count": 0,
                "average_ms": None,
                "p50_ms": None,
                "p95_ms": None,
                "max_ms": None,
            },
            "route_confusion_matrix": {
                "expected_labels": [],
                "actual_labels": [],
                "matrix": {},
            },
            "distributions": {"category": {}, "severity": {}},
            "top_failure_types": [],
        },
        "gate": {"passed": True, "checks": []},
    }
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"

    write_reports(report, json_path, markdown_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert payload["aggregation"]["metrics"]["citation_validity_rate"]["value"] is None
    assert "| 0 | 0 | N/A |" in markdown
    assert "100.00%" not in markdown
    assert render_markdown(report) == markdown
