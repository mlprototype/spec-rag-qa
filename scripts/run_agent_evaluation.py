from __future__ import annotations

import argparse
import asyncio
import json
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


DEFAULT_CASES = Path("data/agent_eval/cases/phase6_synthetic.json")
DEFAULT_TRACES = Path("data/agent_eval/fixtures/phase6_synthetic_traces.json")
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
    parser.add_argument("--output", type=Path)
    return parser


async def run_evaluation(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    cases = load_cases(args.cases)
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
    report = {
        "schema_version": evaluation.schema_version,
        "runner": args.runner,
        "total_cases": len(cases),
        "evaluated_cases": len(successful_cases),
        "execution_errors": execution_errors,
        "evaluation": evaluation.model_dump(mode="json"),
        "normal_metrics": normal_metrics,
        "category_metrics": category_metrics,
    }

    if execution_errors:
        return report, 2
    if any(not result.passed for result in evaluation.cases):
        return report, 1
    return report, 0


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
    try:
        report, exit_code = asyncio.run(run_evaluation(args))
    except (AgentRunnerError, DatasetValidationError, ValueError) as exc:
        report = {
            "runner": args.runner,
            "preflight_error": {
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        }
        exit_code = 2

    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
