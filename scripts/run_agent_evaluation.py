from __future__ import annotations

import argparse
import asyncio
import shlex
import sys
from pathlib import Path
from typing import Any

from ragqa.agent_eval import (
    AgentEvalCase,
    AgentRunTrace,
    AgentRunner,
    AgentRunnerError,
    DatasetValidationError,
    FixtureRunner,
    FixtureTraceMismatchError,
    SubprocessAgentRunner,
    TraceFileRunner,
    aggregate_metrics,
    evaluate_cases,
    load_cases,
    load_saved_traces,
    validate_case_contracts,
    validate_dataset,
)
from ragqa.agent_eval.aggregator import aggregate_agent_evaluation
from ragqa.agent_eval.gate import (
    AgentQualityGateError,
    build_baseline,
    evaluate_quality_gate,
    load_baseline,
    load_gate_config,
    maybe_update_baseline,
)
from ragqa.agent_eval.report import build_report, write_reports


DEFAULT_CASES = Path("data/agent_eval/cases/phase6_synthetic.json")
DEFAULT_TRACES = Path("data/agent_eval/fixtures/phase6_synthetic_traces.json")
DEFAULT_GATE_CONFIG = Path("config/agent_quality_gate.yml")
DEFAULT_BASELINE = Path("data/agent_eval/baseline/agent_baseline.json")
DEFAULT_REPORT_JSON = Path("data/agent_eval/reports/latest.json")
DEFAULT_REPORT_MARKDOWN = Path("data/agent_eval/reports/latest.md")
FALLBACK_CATEGORIES = {"fallback", "degraded"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic Agent evaluation")
    parser.add_argument(
        "--runner",
        choices=("fixture", "trace-file", "subprocess"),
        default="fixture",
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument(
        "--subprocess-command",
        help="ai-agent-rag trace CLI command, parsed without a shell",
    )
    parser.add_argument("--subprocess-cwd", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--output",
        type=Path,
        help="Backward-compatible alias for --report-json",
    )
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--report-markdown", type=Path)
    parser.add_argument("--gate-config", type=Path, default=DEFAULT_GATE_CONFIG)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Explicitly replace the reviewed Baseline after absolute Gates pass",
    )
    return parser


async def run_evaluation(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    cases = load_cases(args.cases)
    gate_config = load_gate_config(args.gate_config)
    runner = _build_runner(args, cases)
    successful_cases: list[AgentEvalCase] = []
    traces: list[AgentRunTrace] = []
    execution_errors: list[dict[str, str]] = []

    for case in cases:
        try:
            trace = await runner.run(case)
        except (AgentRunnerError, FixtureTraceMismatchError) as exc:
            execution_errors.append(
                {
                    "case_id": case.id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            continue
        successful_cases.append(case)
        traces.append(trace)

    evaluation = evaluate_cases(successful_cases, traces)
    result_by_case_id = {result.case_id: result for result in evaluation.cases}
    category_metrics: dict[str, dict[str, Any]] = {}
    for category in sorted({case.category for case in successful_cases}):
        category_results = [
            result_by_case_id[case.id]
            for case in successful_cases
            if case.category == category
        ]
        category_metrics[category] = {
            key: value.model_dump(mode="json")
            for key, value in aggregate_metrics(category_results).items()
        }

    normal_results = [
        result_by_case_id[case.id]
        for case in successful_cases
        if case.category not in FALLBACK_CATEGORIES
    ]
    normal_metrics = {
        key: value.model_dump(mode="json")
        for key, value in aggregate_metrics(normal_results).items()
    }
    failure_owners = gate_config.get("failure_owners", {})
    aggregation = aggregate_agent_evaluation(
        cases,
        traces,
        evaluation,
        execution_errors,
        failure_owners=failure_owners,
    )

    baseline_updated = False
    if args.update_baseline:
        baseline = build_baseline(aggregation)
    else:
        baseline = load_baseline(args.baseline)
    gate = evaluate_quality_gate(aggregation, gate_config, baseline)

    if args.update_baseline and gate["passed"] and not execution_errors:
        baseline_updated = maybe_update_baseline(
            args.baseline,
            aggregation,
            enabled=True,
        )

    report = build_report(
        runner=args.runner,
        cases_path=args.cases,
        traces_path=args.traces,
        evaluation=evaluation,
        aggregation=aggregation,
        execution_errors=execution_errors,
        gate=gate,
        baseline_path=args.baseline,
        baseline_updated=baseline_updated,
    )
    # Preserve the Issue #11 normal-path view for existing report consumers.
    report["normal_metrics"] = normal_metrics
    report["category_metrics"] = category_metrics

    if execution_errors:
        return report, 2
    return report, 0 if gate["passed"] else 1


def _build_runner(args: argparse.Namespace, cases: list[AgentEvalCase]) -> AgentRunner:
    if args.runner in {"fixture", "trace-file"}:
        traces = load_saved_traces(args.traces)
        validate_dataset(cases, traces)
        if args.runner == "fixture":
            return FixtureRunner(traces)
        return TraceFileRunner(args.traces)

    validate_case_contracts(cases)
    if not args.subprocess_command:
        raise DatasetValidationError(
            "--subprocess-command is required for subprocess runner"
        )
    command = shlex.split(args.subprocess_command)
    return SubprocessAgentRunner(
        command,
        timeout_seconds=args.timeout_seconds,
        cwd=args.subprocess_cwd,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    json_path, markdown_path = _report_paths(args)
    try:
        report, exit_code = asyncio.run(run_evaluation(args))
    except (
        AgentQualityGateError,
        AgentRunnerError,
        DatasetValidationError,
        ValueError,
    ) as exc:
        report = {
            "runner": args.runner,
            "preflight_error": {
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        }
        exit_code = 2

    write_reports(report, json_path, markdown_path)
    print(
        f"Agent quality gate: "
        f"{'PASS' if exit_code == 0 else 'FAIL'} "
        f"(JSON: {json_path}, Markdown: {markdown_path})"
    )
    return exit_code


def _report_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    json_path = args.output or args.report_json or DEFAULT_REPORT_JSON
    if args.report_markdown is not None:
        markdown_path = args.report_markdown
    elif args.output is not None or args.report_json is not None:
        markdown_path = json_path.with_suffix(".md")
    else:
        markdown_path = DEFAULT_REPORT_MARKDOWN
    return json_path, markdown_path


if __name__ == "__main__":
    sys.exit(main())
