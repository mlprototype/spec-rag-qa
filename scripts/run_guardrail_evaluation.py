from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from ragqa.agent_eval.adapters.gateway import GatewayHttpRunner
from ragqa.agent_eval.aggregator import aggregate_guardrail_evaluation
from ragqa.agent_eval.gate import (
    AgentQualityGateError,
    evaluate_quality_gate,
    load_gate_config,
)
from ragqa.agent_eval.guardrail import evaluate_guardrail_cases
from ragqa.agent_eval.models import AgentEvalCase, AgentRunTrace
from ragqa.agent_eval.report import build_guardrail_report, write_reports
from ragqa.agent_eval.runner import (
    AgentRunner,
    AgentRunnerError,
    FixtureTraceMismatchError,
    load_cases,
)
from ragqa.agent_eval.validation import (
    DatasetValidationError,
    validate_case_contracts,
    validate_dataset,
)
from ragqa.agent_eval.adapters.fixture import FixtureRunner
from ragqa.agent_eval.adapters.trace_file import load_saved_traces


DEFAULT_CASES = Path("data/agent_eval/cases/guardrail_synthetic.json")
DEFAULT_TRACES = Path("data/agent_eval/fixtures/guardrail_synthetic_traces.json")
DEFAULT_GATE_CONFIG = Path("config/guardrail_quality_gate.yml")
DEFAULT_REPORT_JSON = Path(".artifacts/guardrail-quality/report.json")
DEFAULT_REPORT_MARKDOWN = Path(".artifacts/guardrail-quality/report.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run policy-aware-llm-gateway Guardrail evaluation"
    )
    parser.add_argument("--runner", choices=("fixture", "http"), default="fixture")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--gateway-url")
    parser.add_argument(
        "--gateway-allowed-host",
        action="append",
        default=[],
        help="Exact Gateway hostname allowed for outbound evaluation; repeatable",
    )
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--gate-config", type=Path, default=DEFAULT_GATE_CONFIG)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument(
        "--report-markdown", type=Path, default=DEFAULT_REPORT_MARKDOWN
    )
    return parser


async def run_evaluation(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    cases = load_cases(args.cases)
    validate_case_contracts(cases)
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

    evaluation = evaluate_guardrail_cases(successful_cases, traces)
    aggregation = aggregate_guardrail_evaluation(
        cases,
        traces,
        evaluation,
        execution_errors,
        failure_owners=gate_config.get("failure_owners", {}),
    )
    gate = evaluate_quality_gate(aggregation, gate_config, {})
    report = build_guardrail_report(
        runner=args.runner,
        cases_path=args.cases,
        traces_path=args.traces if args.runner == "fixture" else None,
        evaluation=evaluation,
        aggregation=aggregation,
        execution_errors=execution_errors,
        gate=gate,
    )
    if execution_errors:
        return report, 2
    return report, 0 if gate["passed"] else 1


def _build_runner(
    args: argparse.Namespace, cases: list[AgentEvalCase]
) -> AgentRunner:
    if args.runner == "fixture":
        traces = load_saved_traces(args.traces)
        validate_dataset(cases, traces)
        return FixtureRunner(traces)
    if not args.gateway_url:
        raise DatasetValidationError("--gateway-url is required for HTTP runner")
    if not args.gateway_allowed_host:
        raise DatasetValidationError(
            "--gateway-allowed-host is required for HTTP runner"
        )
    return GatewayHttpRunner(
        args.gateway_url,
        api_key=os.environ.get("GATEWAY_API_KEY"),
        allowed_hosts=args.gateway_allowed_host,
        timeout_seconds=args.timeout_seconds,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report, exit_code = asyncio.run(run_evaluation(args))
    except (
        AgentQualityGateError,
        AgentRunnerError,
        DatasetValidationError,
        OSError,
        ValueError,
    ) as exc:
        report = {
            "report_type": "guardrail",
            "runner": args.runner,
            "preflight_error": {
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        }
        exit_code = 2

    write_reports(report, args.report_json, args.report_markdown)
    print(
        "Guardrail quality gate: "
        f"{'PASS' if exit_code == 0 else 'FAIL'} "
        f"(JSON: {args.report_json}, Markdown: {args.report_markdown})"
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
