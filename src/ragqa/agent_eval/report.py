from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ragqa.agent_eval.models import AgentEvaluationResult


METRIC_LABELS = {
    "task_success_rate": "Task Success Rate",
    "route_selection_accuracy": "Route Selection Accuracy",
    "required_tool_call_rate": "Required Tool Call Rate",
    "unexpected_tool_call_rate": "Unexpected Tool Call Rate",
    "tool_argument_schema_compliance": "Tool Argument Schema Compliance",
    "tool_argument_semantic_accuracy": "Tool Argument Semantic Accuracy",
    "citation_presence_rate": "Citation Presence",
    "citation_validity_rate": "Citation Validity",
    "answer_format_compliance": "Answer Format Compliance",
    "latency_budget_compliance": "Latency Budget Compliance",
}


def build_report(
    *,
    runner: str,
    cases_path: str | Path,
    traces_path: str | Path,
    evaluation: AgentEvaluationResult,
    aggregation: Mapping[str, Any],
    execution_errors: list[dict[str, str]],
    gate: Mapping[str, Any],
    baseline_path: str | Path,
    baseline_updated: bool,
    generated_at: str | None = None,
) -> dict[str, Any]:
    counts = aggregation["counts"]
    category_metrics = {
        category: details["metrics"]
        for category, details in aggregation["distributions"]["category"].items()
    }
    return {
        "schema_version": evaluation.schema_version,
        "generated_at": generated_at or _utc_now(),
        "runner": runner,
        "dataset": {
            "cases_path": str(cases_path),
            "traces_path": str(traces_path),
        },
        "baseline": {
            "path": str(baseline_path),
            "updated": baseline_updated,
        },
        # Backward-compatible summary fields from the Issue #11 CLI report.
        "total_cases": counts["total_cases"],
        "evaluated_cases": counts["evaluated_cases"],
        "execution_errors": execution_errors,
        "evaluation": evaluation.model_dump(mode="json"),
        "category_metrics": category_metrics,
        "aggregation": dict(aggregation),
        "gate": dict(gate),
    }


def write_reports(
    report: Mapping[str, Any],
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    json_report_path = Path(json_path)
    markdown_report_path = Path(markdown_path)
    json_report_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
    json_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_report_path.write_text(
        render_markdown(report),
        encoding="utf-8",
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    if "aggregation" not in report:
        return _render_error_report(report)

    aggregation = report["aggregation"]
    gate = report.get("gate", {})
    counts = aggregation["counts"]
    lines = [
        "# Agent Quality Report",
        "",
        f"- Generated: {report.get('generated_at', 'N/A')}",
        f"- Runner: `{report.get('runner', 'N/A')}`",
        f"- Gate: **{'PASS' if gate.get('passed') else 'FAIL'}**",
        f"- Baseline: `{report.get('baseline', {}).get('path', 'N/A')}`",
        f"- Baseline updated: `{str(report.get('baseline', {}).get('updated', False)).lower()}`",
        "",
        "## Summary",
        "",
        "| Total | Evaluated | Passed | Evaluation FAIL | Execution Error |",
        "|---:|---:|---:|---:|---:|",
        (
            f"| {counts['total_cases']} | {counts['evaluated_cases']} | "
            f"{counts['passed_cases']} | {counts['failed_cases']} | "
            f"{counts['execution_errors']} |"
        ),
        "",
        "## Quality Metrics",
        "",
        "| Metric | Numerator | Denominator | Value |",
        "|:---|---:|---:|---:|",
    ]
    for metric_id, metric in aggregation["metrics"].items():
        label = METRIC_LABELS.get(metric_id, metric_id)
        lines.append(
            f"| {label} | {metric['numerator']} | {metric['denominator']} | "
            f"{_format_rate(metric.get('value'))} |"
        )

    latency = aggregation["latency"]
    lines.extend(
        [
            "",
            "## Latency",
            "",
            "| Count | Average | p50 | p95 | Max |",
            "|---:|---:|---:|---:|---:|",
            (
                f"| {latency['count']} | {_format_ms(latency.get('average_ms'))} | "
                f"{_format_ms(latency.get('p50_ms'))} | "
                f"{_format_ms(latency.get('p95_ms'))} | "
                f"{_format_ms(latency.get('max_ms'))} |"
            ),
            "",
            "## Route Confusion Matrix",
            "",
        ]
    )
    confusion = aggregation["route_confusion_matrix"]
    actual_labels = confusion["actual_labels"]
    if not confusion["expected_labels"]:
        lines.append("N/A")
    else:
        lines.append("| Expected \\ Actual | " + " | ".join(actual_labels) + " |")
        lines.append("|:---|" + "---:|" * len(actual_labels))
        for expected in confusion["expected_labels"]:
            values = [str(confusion["matrix"][expected][actual]) for actual in actual_labels]
            lines.append(f"| {expected} | " + " | ".join(values) + " |")

    for distribution_name in ("category", "severity"):
        lines.extend(
            [
                "",
                f"## By {distribution_name.title()}",
                "",
                "| Group | Total | Passed | Evaluation FAIL | Execution Error | Task Success |",
                "|:---|---:|---:|---:|---:|---:|",
            ]
        )
        for group, details in aggregation["distributions"][distribution_name].items():
            group_counts = details["counts"]
            task_success = details["metrics"]["task_success_rate"]
            lines.append(
                f"| {group} | {group_counts['total_cases']} | "
                f"{group_counts['passed_cases']} | {group_counts['failed_cases']} | "
                f"{group_counts['execution_errors']} | "
                f"{_format_rate(task_success.get('value'))} |"
            )

    lines.extend(["", "## Failure Types", ""])
    top_failures = aggregation["top_failure_types"]
    if not top_failures:
        lines.append("No evaluation failures.")
    else:
        lines.extend(
            [
                "| Failure Type | Count | Owner |",
                "|:---|---:|:---|",
            ]
        )
        for failure in top_failures:
            lines.append(
                f"| {failure['failure_type']} | {failure['count']} | "
                f"{failure['owner']} |"
            )

    lines.extend(["", "## Execution Errors", ""])
    execution_errors = report.get("execution_errors", [])
    if not execution_errors:
        lines.append("No execution errors.")
    else:
        lines.extend(
            [
                "| Case | Error Type | Message |",
                "|:---|:---|:---|",
            ]
        )
        for error in execution_errors:
            lines.append(
                f"| {error.get('case_id', '')} | {error.get('error_type', '')} | "
                f"{_escape_cell(error.get('message', ''))} |"
            )

    lines.extend(
        [
            "",
            "## Quality Gate",
            "",
            "| Gate | Type | Status | Actual | Baseline | Threshold | Reason |",
            "|:---|:---|:---|---:|---:|---:|:---|",
        ]
    )
    for check in gate.get("checks", []):
        lines.append(
            f"| {check['gate_id']} | {check['gate_type']} | {check['status']} | "
            f"{_format_value(check.get('actual'))} | "
            f"{_format_value(check.get('baseline'))} | "
            f"{_format_value(check.get('threshold'))} | "
            f"{_escape_cell(check.get('reason', ''))} |"
        )
    return "\n".join(lines) + "\n"


def _render_error_report(report: Mapping[str, Any]) -> str:
    error = report.get("preflight_error", {})
    return "\n".join(
        [
            "# Agent Quality Report",
            "",
            "- Gate: **ERROR**",
            f"- Runner: `{report.get('runner', 'N/A')}`",
            "",
            "## Preflight Error",
            "",
            f"- Type: `{error.get('error_type', 'Unknown')}`",
            f"- Message: {_escape_cell(str(error.get('message', '')))}",
            "",
        ]
    )


def _format_rate(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def _format_ms(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.3f} ms"


def _format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
