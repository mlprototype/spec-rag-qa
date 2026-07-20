from __future__ import annotations

import argparse
import asyncio
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from ragqa.agent_eval.adapters.fixture import FixtureRunner
from ragqa.agent_eval.adapters.subprocess import SubprocessAgentRunner
from ragqa.agent_eval.adapters.trace_file import TraceFileRunner, load_saved_traces
from ragqa.agent_eval.advanced import run_advanced_evaluation
from ragqa.agent_eval.advanced_report import (
    build_advanced_report,
    write_advanced_reports,
)
from ragqa.agent_eval.cost import PricingConfigError, load_pricing_config
from ragqa.agent_eval.judge import (
    DeterministicMockJudgeTransport,
    HttpJudgeTransport,
    JudgeAdapter,
    JudgeError,
    StructuredJudgeAdapter,
)
from ragqa.agent_eval.models import AgentEvalCase
from ragqa.agent_eval.runner import AgentRunner, AgentRunnerError, load_cases
from ragqa.agent_eval.validation import (
    DatasetValidationError,
    validate_case_contracts,
    validate_dataset,
)


DEFAULT_CASES = Path("data/agent_eval/cases/phase6_synthetic.json")
DEFAULT_TRACES = Path("data/agent_eval/fixtures/phase6_synthetic_traces.json")
DEFAULT_PRICING = Path("config/agent_pricing.json")
DEFAULT_REPORT_JSON = Path(".artifacts/agent-advanced/report.json")
DEFAULT_REPORT_MARKDOWN = Path(".artifacts/agent-advanced/report.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run monitor-only advanced Agent evaluation"
    )
    parser.add_argument(
        "--runner",
        choices=("fixture", "trace-file", "subprocess"),
        default="fixture",
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--pricing", type=Path, default=DEFAULT_PRICING)
    parser.add_argument(
        "--judge",
        choices=("mock", "http"),
        default="mock",
    )
    parser.add_argument("--judge-url")
    parser.add_argument("--judge-model")
    parser.add_argument(
        "--judge-allowed-host",
        action="append",
        default=[],
        help="Exact Judge hostname allowed for outbound evaluation data; repeatable",
    )
    parser.add_argument("--judge-timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--subprocess-command",
        help="Agent trace CLI command, parsed without a shell",
    )
    parser.add_argument("--subprocess-cwd", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument(
        "--report-markdown",
        type=Path,
        default=DEFAULT_REPORT_MARKDOWN,
    )
    return parser


async def run_monitoring(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    cases = load_cases(args.cases)
    runner = _build_runner(args, cases)
    judge = _build_judge(args)
    pricing = load_pricing_config(args.pricing)
    result = await run_advanced_evaluation(cases, runner, judge, pricing)
    report = build_advanced_report(
        result,
        runner=args.runner,
        judge_adapter=args.judge,
        cases_path=args.cases,
        traces_path=args.traces,
        pricing_path=args.pricing,
    )
    return report, 2 if result.errors else 0


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
    return SubprocessAgentRunner(
        shlex.split(args.subprocess_command),
        timeout_seconds=args.timeout_seconds,
        cwd=args.subprocess_cwd,
    )


def _build_judge(args: argparse.Namespace) -> JudgeAdapter:
    if args.judge == "mock":
        return StructuredJudgeAdapter(
            DeterministicMockJudgeTransport(),
            judge_model=args.judge_model or "deterministic-mock-judge-v1",
        )
    if not args.judge_url:
        raise ValueError("--judge-url is required for the HTTP Judge")
    if not args.judge_model:
        raise ValueError("--judge-model is required for the HTTP Judge")
    transport = HttpJudgeTransport(
        args.judge_url,
        api_key=os.environ.get("AGENT_EVAL_JUDGE_API_KEY"),
        allowed_hosts=args.judge_allowed_host,
        timeout_seconds=args.judge_timeout_seconds,
    )
    return StructuredJudgeAdapter(
        transport,
        judge_model=args.judge_model,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report, exit_code = asyncio.run(run_monitoring(args))
    except (
        AgentRunnerError,
        DatasetValidationError,
        JudgeError,
        PricingConfigError,
        ValueError,
    ) as exc:
        report = {
            "runner": args.runner,
            "judge_adapter": args.judge,
            "monitor_only": [
                "groundedness",
                "answer_semantic_consistency",
                "cost",
            ],
            "preflight_error": {
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        }
        exit_code = 2

    write_advanced_reports(report, args.report_json, args.report_markdown)
    print(
        "Agent advanced monitoring: "
        f"{'COMPLETED' if exit_code == 0 else 'ERROR'} "
        f"(JSON: {args.report_json}, Markdown: {args.report_markdown})"
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
