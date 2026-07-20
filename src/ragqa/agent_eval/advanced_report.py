from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ragqa.agent_eval.advanced_models import AdvancedEvaluationResult


def build_advanced_report(
    result: AdvancedEvaluationResult,
    *,
    runner: str,
    judge_adapter: str,
    cases_path: str | Path,
    traces_path: str | Path,
    pricing_path: str | Path,
) -> dict[str, Any]:
    report = result.model_dump(mode="json")
    report.update(
        {
            "runner": runner,
            "judge_adapter": judge_adapter,
            "dataset": {
                "cases_path": str(cases_path),
                "traces_path": str(traces_path),
            },
            "pricing_path": str(pricing_path),
        }
    )
    return report


def write_advanced_reports(
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
        render_advanced_markdown(report),
        encoding="utf-8",
    )


def render_advanced_markdown(report: Mapping[str, Any]) -> str:
    if "summary" not in report:
        error = report.get("preflight_error", {})
        return "\n".join(
            [
                "# Agent Advanced Evaluation Report",
                "",
                "- Mode: **MONITOR ONLY**",
                "- Status: **ERROR**",
                f"- Type: `{error.get('error_type', 'Unknown')}`",
                f"- Message: {_escape(error.get('message', ''))}",
                "",
            ]
        )

    summary = report["summary"]
    runs = summary["runs"]
    grounding = summary["groundedness"]
    cost = summary["cost"]
    lines = [
        "# Agent Advanced Evaluation Report",
        "",
        "- Mode: **MONITOR ONLY**",
        f"- Generated: {report.get('generated_at', 'N/A')}",
        f"- Runner: `{report.get('runner', 'N/A')}`",
        f"- Judge adapter: `{report.get('judge_adapter', 'N/A')}`",
        f"- Judge model: `{report.get('judge_model', 'N/A')}`",
        (
            "- Groundedness prompt: `"
            f"{report.get('judge_prompt_versions', {}).get('groundedness', 'N/A')}`"
        ),
        (
            "- Semantic prompt: `"
            f"{report.get('judge_prompt_versions', {}).get('answer_semantic_consistency', 'N/A')}`"
        ),
        f"- Pricing version: `{report.get('pricing_version', 'N/A')}`",
        "",
        "## Run Status",
        "",
        "| Requested | Completed | Runner Errors | Judge Errors |",
        "|---:|---:|---:|---:|",
        (
            f"| {runs['requested']} | {runs['completed']} | "
            f"{runs['execution_errors']} | {runs['judge_errors']} |"
        ),
        "",
        "## Groundedness",
        "",
        "| Supported Claims | Evaluable Claims | Score | Evaluated Runs |",
        "|---:|---:|---:|---:|",
        (
            f"| {grounding['supported_claims']} | {grounding['evaluable_claims']} | "
            f"{_format_rate(grounding.get('score'))} | "
            f"{grounding['evaluated_runs']} |"
        ),
        "",
        "## Stability",
        "",
        "| Dimension | Cases | Average Mode Share | All-Match |",
        "|:---|---:|---:|---:|",
    ]
    for name, dimension in summary["stability"].items():
        lines.append(
            f"| {name} | {dimension['cases']} | "
            f"{_format_rate(dimension.get('average_mode_share'))} | "
            f"{dimension['all_match_cases']}/{dimension['all_match_denominator']} "
            f"({_format_rate(dimension.get('all_match_rate'))}) |"
        )

    lines.extend(
        [
            "",
            "## Claim Results",
            "",
            "| Case | Run | Evaluable | Supported | Claim | Evidence |",
            "|:---|---:|:---:|:---:|:---|:---|",
        ]
    )
    claim_rows = 0
    for result in report.get("groundedness", []):
        for claim in result.get("claims", []):
            evidence = [
                *claim.get("source_ids", []),
                *claim.get("tool_result_ids", []),
            ]
            lines.append(
                f"| {result['case_id']} | {result['run_index']} | "
                f"{_format_bool(claim.get('evaluable'))} | "
                f"{_format_bool(claim.get('supported'))} | "
                f"{_escape(_truncate(claim.get('claim', '')))} | "
                f"{_escape(', '.join(evidence) or 'N/A')} |"
            )
            claim_rows += 1
    if claim_rows == 0:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A |")

    lines.extend(
        [
            "",
            "## Cost",
            "",
            "| Cost Coverage | Known Cost | Complete Estimated Total | Unavailable Runs |",
            "|---:|---:|---:|---:|",
            (
                f"| {_format_rate(cost.get('coverage'))} | "
                f"{_format_usd(cost.get('known_cost_usd'))} | "
                f"{_format_usd(cost.get('estimated_total_cost_usd'))} | "
                f"{cost['unavailable_runs']} |"
            ),
            "",
            "## Cost by Run",
            "",
            (
                "| Case | Run | Model | Input Tokens | Output Tokens | "
                "Token Cost | Tool Cost | Total | Status |"
            ),
            "|:---|---:|:---|---:|---:|---:|---:|---:|:---|",
        ]
    )
    costs = report.get("costs", [])
    if not costs:
        lines.append(
            "| N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
        )
    for item in costs:
        lines.append(
            f"| {item['case_id']} | {item['run_index']} | {item['model']} | "
            f"{_format_integer(item.get('input_tokens'))} | "
            f"{_format_integer(item.get('output_tokens'))} | "
            f"{_format_usd(item.get('estimated_token_cost_usd'))} | "
            f"{_format_usd(item.get('estimated_tool_cost_usd'))} | "
            f"{_format_usd(item.get('estimated_total_cost_usd'))} | "
            f"{item['status']} |"
        )
    if any(item.get("status") == "model_not_priced" for item in costs):
        lines.extend(
            [
                "",
                (
                    "> `model_not_priced` is `N/A`: the Trace either did not "
                    "report an actual model identifier (`model=unreported`) or "
                    "that model is absent from the pricing config. Agent `target` "
                    "names are never treated as model identifiers."
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Repeat Cases",
            "",
            "| Case | Runs | Errors | Route Mode | Route All | Semantic Mode | Semantic All |",
            "|:---|---:|---:|---:|:---:|---:|:---:|",
        ]
    )
    repeat_cases = [
        item for item in report.get("stability", []) if item["requested_runs"] >= 2
    ]
    if not repeat_cases:
        lines.append("| N/A | 0 | 0 | N/A | N/A | N/A | N/A |")
    for item in repeat_cases:
        route = item["dimensions"]["route"]
        semantic = item.get("semantic_consistency") or {}
        lines.append(
            f"| {item['case_id']} | {item['successful_runs']}/{item['requested_runs']} | "
            f"{item['execution_error_count']} | {_format_rate(route.get('mode_share'))} | "
            f"{_format_bool(route.get('all_match'))} | "
            f"{_format_rate(semantic.get('mode_share'))} | "
            f"{_format_bool(semantic.get('all_match'))} |"
        )

    lines.extend(["", "## Errors", ""])
    errors = report.get("errors", [])
    if not errors:
        lines.append("No Runner or Judge errors.")
    else:
        lines.extend(
            [
                "| Case | Run | Stage | Type | Message |",
                "|:---|---:|:---|:---|:---|",
            ]
        )
        for error in errors:
            lines.append(
                f"| {error['case_id']} | {error.get('run_index') or 'N/A'} | "
                f"{error['stage']} | {error['error_type']} | "
                f"{_escape(error.get('message', ''))} |"
            )
    return "\n".join(lines) + "\n"


def _format_rate(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def _format_usd(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"${float(value):.8f}"


def _format_bool(value: Any) -> str:
    if value is None:
        return "N/A"
    return "yes" if value else "no"


def _format_integer(value: Any) -> str:
    return "N/A" if value is None else str(value)


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _truncate(value: Any, limit: int = 120) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"
