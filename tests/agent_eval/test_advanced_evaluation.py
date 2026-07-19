from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ragqa.agent_eval.adapters.fixture import FixtureRunner
from ragqa.agent_eval.advanced import run_advanced_evaluation
from ragqa.agent_eval.advanced_models import JudgeRequest
from ragqa.agent_eval.advanced_report import (
    build_advanced_report,
    render_advanced_markdown,
)
from ragqa.agent_eval.cost import load_pricing_config
from ragqa.agent_eval.judge import (
    DeterministicMockJudgeTransport,
    StructuredJudgeAdapter,
)
from ragqa.agent_eval.models import AgentEvalCase, AgentRunTrace
from ragqa.agent_eval.runner import RunnerTimeoutError


ROOT = Path(__file__).resolve().parents[2]
PRICING_PATH = ROOT / "config" / "agent_pricing.json"


class CountingRunner:
    def __init__(self, trace: AgentRunTrace, fail_runs: set[int] | None = None) -> None:
        self.trace = trace
        self.fail_runs = fail_runs or set()
        self.calls = 0

    async def run(self, case: AgentEvalCase) -> AgentRunTrace:
        self.calls += 1
        if self.calls in self.fail_runs:
            raise RunnerTimeoutError(f"timeout on run {self.calls}")
        trace = self.trace.model_copy(deep=True)
        trace.run_id = f"{self.trace.run_id}-{self.calls}"
        return trace


class AlwaysMalformedTransport:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: JudgeRequest) -> str:
        self.calls += 1
        return "malformed"


class UnsupportedClaimTransport:
    async def complete(self, request: JudgeRequest) -> dict[str, Any]:
        return {
            "schema_version": request.schema_version,
            "claims": [
                {
                    "claim": "This claim has no supplied support.",
                    "evaluable": True,
                    "supported": False,
                    "source_ids": [],
                    "tool_result_ids": [],
                    "reason": "No evidence supports the claim.",
                }
            ],
        }


def _case_with_repeat(case: AgentEvalCase, repeat: int) -> AgentEvalCase:
    copied = case.model_copy(deep=True)
    copied.repeat = repeat
    return copied


def test_synthetic_dataset_has_cross_category_repeat_cases(
    synthetic_cases: list[AgentEvalCase],
) -> None:
    repeat_cases = [case for case in synthetic_cases if case.repeat >= 3]

    assert len(repeat_cases) == 5
    assert {case.category for case in repeat_cases} == {
        "direct",
        "definition",
        "structured_query",
        "compare",
        "fallback",
    }


def test_advanced_evaluation_runs_runner_exactly_case_repeat_times(
    synthetic_cases: list[AgentEvalCase],
    synthetic_traces: list[AgentRunTrace],
) -> None:
    case = _case_with_repeat(synthetic_cases[0], 3)
    runner = CountingRunner(synthetic_traces[0])
    judge = StructuredJudgeAdapter(
        DeterministicMockJudgeTransport(),
        judge_model="offline-mock",
    )

    result = asyncio.run(
        run_advanced_evaluation(
            [case], runner, judge, load_pricing_config(PRICING_PATH)
        )
    )

    assert runner.calls == 3
    assert result.summary["runs"]["requested"] == 3
    assert result.summary["runs"]["completed"] == 3
    assert result.stability[0].dimensions["route"].all_match is True
    assert result.stability[0].semantic_consistency is not None


def test_repeat_runner_error_is_explicit_and_does_not_stop_remaining_runs(
    synthetic_cases: list[AgentEvalCase],
    synthetic_traces: list[AgentRunTrace],
) -> None:
    case = _case_with_repeat(synthetic_cases[0], 3)
    runner = CountingRunner(synthetic_traces[0], fail_runs={2})
    judge = StructuredJudgeAdapter(
        DeterministicMockJudgeTransport(),
        judge_model="offline-mock",
    )

    result = asyncio.run(
        run_advanced_evaluation(
            [case], runner, judge, load_pricing_config(PRICING_PATH)
        )
    )

    assert runner.calls == 3
    assert result.summary["runs"]["completed"] == 2
    assert result.summary["runs"]["execution_errors"] == 1
    assert result.errors[0].stage == "runner"
    assert result.errors[0].run_index == 2
    assert result.stability[0].dimensions["route"].all_match is False


def test_judge_failure_is_explicit_not_an_unsupported_claim(
    synthetic_cases: list[AgentEvalCase],
    synthetic_traces: list[AgentRunTrace],
) -> None:
    case = _case_with_repeat(synthetic_cases[0], 1)
    transport = AlwaysMalformedTransport()
    judge = StructuredJudgeAdapter(transport, judge_model="broken-judge")

    result = asyncio.run(
        run_advanced_evaluation(
            [case],
            FixtureRunner([synthetic_traces[0]]),
            judge,
            load_pricing_config(PRICING_PATH),
        )
    )

    assert transport.calls == 2
    assert result.groundedness == []
    assert result.summary["groundedness"]["evaluable_claims"] == 0
    assert result.summary["groundedness"]["score"] is None
    assert result.errors[0].stage == "groundedness_judge"


def test_groundedness_failure_is_monitor_only_not_an_execution_error(
    synthetic_cases: list[AgentEvalCase],
    synthetic_traces: list[AgentRunTrace],
) -> None:
    case = _case_with_repeat(synthetic_cases[0], 1)
    judge = StructuredJudgeAdapter(
        UnsupportedClaimTransport(),
        judge_model="offline-mock",
    )

    result = asyncio.run(
        run_advanced_evaluation(
            [case],
            FixtureRunner([synthetic_traces[0]]),
            judge,
            load_pricing_config(PRICING_PATH),
        )
    )

    assert result.summary["groundedness"]["score"] == 0.0
    assert result.errors == []
    assert "passed" not in result.model_dump()


def test_advanced_report_preserves_unavailable_cost_as_na(
    synthetic_cases: list[AgentEvalCase],
    synthetic_traces: list[AgentRunTrace],
) -> None:
    case = _case_with_repeat(synthetic_cases[0], 1)
    trace = synthetic_traces[0].model_copy(deep=True)
    trace.usage.input_tokens = None
    trace.usage.output_tokens = None
    judge = StructuredJudgeAdapter(
        DeterministicMockJudgeTransport(),
        judge_model="offline-mock",
    )
    result = asyncio.run(
        run_advanced_evaluation(
            [case],
            FixtureRunner([trace]),
            judge,
            load_pricing_config(PRICING_PATH),
        )
    )
    report = build_advanced_report(
        result,
        runner="fixture",
        judge_adapter="mock",
        cases_path="cases.json",
        traces_path="traces.json",
        pricing_path=PRICING_PATH,
    )

    markdown = render_advanced_markdown(report)

    assert result.costs[0].estimated_total_cost_usd is None
    assert result.summary["cost"]["coverage"] == 0.0
    assert result.summary["cost"]["estimated_total_cost_usd"] is None
    assert "N/A" in markdown
    assert "## Claim Results" in markdown
    assert "## Cost by Run" in markdown
