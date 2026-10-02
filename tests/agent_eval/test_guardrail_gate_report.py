from __future__ import annotations

from pathlib import Path

from ragqa.agent_eval import (
    AgentEvalCase,
    AgentEvaluationResult,
    AgentRunTrace,
)
from ragqa.agent_eval.aggregator import aggregate_guardrail_evaluation
from ragqa.agent_eval.gate import evaluate_quality_gate, load_gate_config
from ragqa.agent_eval.guardrail import evaluate_guardrail_cases
from ragqa.agent_eval.report import (
    build_guardrail_report,
    render_guardrail_markdown,
)


ROOT = Path(__file__).resolve().parents[2]
GATE_CONFIG = ROOT / "config" / "guardrail_quality_gate.yml"


def _aggregation(
    cases: list[AgentEvalCase], traces: list[AgentRunTrace]
) -> tuple[dict[str, object], AgentEvaluationResult]:
    evaluation = evaluate_guardrail_cases(cases, traces)
    aggregation = aggregate_guardrail_evaluation(
        cases, traces, evaluation, []
    )
    return aggregation, evaluation


def test_guardrail_gate_passes_reviewed_fixture(
    guardrail_cases: list[AgentEvalCase],
    guardrail_traces: list[AgentRunTrace],
) -> None:
    aggregation, _ = _aggregation(guardrail_cases, guardrail_traces)

    gate = evaluate_quality_gate(
        aggregation, load_gate_config(GATE_CONFIG), {}
    )

    assert gate["passed"] is True
    assert gate["failed_count"] == 0


def test_guardrail_gate_fails_critical_false_negative(
    guardrail_cases: list[AgentEvalCase],
    guardrail_traces: list[AgentRunTrace],
) -> None:
    traces = [trace.model_copy(deep=True) for trace in guardrail_traces]
    critical_positive = next(
        case
        for case in guardrail_cases
        if case.severity == "critical" and case.expected.detected is True
    )
    trace = next(item for item in traces if item.case_id == critical_positive.id)
    trace.guardrail.detected = False
    aggregation, _ = _aggregation(guardrail_cases, traces)

    gate = evaluate_quality_gate(
        aggregation, load_gate_config(GATE_CONFIG), {}
    )

    failed_ids = {
        check["gate_id"]
        for check in gate["checks"]
        if check["status"] == "failed"
    }
    assert gate["passed"] is False
    assert "critical_guardrail_recall" in failed_ids


def test_guardrail_gate_fails_when_mask_evidence_is_unavailable(
    guardrail_cases: list[AgentEvalCase],
    guardrail_traces: list[AgentRunTrace],
) -> None:
    traces = [trace.model_copy(deep=True) for trace in guardrail_traces]
    mask_case = next(
        case for case in guardrail_cases if case.expected.action == "mask"
    )
    trace = next(item for item in traces if item.case_id == mask_case.id)
    trace.guardrail.provider_input = None
    trace.guardrail.mask_applied = None
    trace.guardrail.mask_evidence = None
    aggregation, _ = _aggregation(guardrail_cases, traces)

    gate = evaluate_quality_gate(
        aggregation, load_gate_config(GATE_CONFIG), {}
    )

    failed_ids = {
        check["gate_id"]
        for check in gate["checks"]
        if check["status"] == "failed"
    }
    assert "mask_evidence_availability" in failed_ids


def test_guardrail_report_contains_metrics_and_na(
    guardrail_cases: list[AgentEvalCase],
    guardrail_traces: list[AgentRunTrace],
) -> None:
    aggregation, evaluation = _aggregation(
        guardrail_cases, guardrail_traces
    )
    aggregation["guardrail"]["categories"]["pii"]["precision"] = None
    gate = evaluate_quality_gate(
        aggregation, load_gate_config(GATE_CONFIG), {}
    )
    report = build_guardrail_report(
        runner="fixture",
        cases_path="cases.json",
        traces_path="traces.json",
        evaluation=evaluation,
        aggregation=aggregation,
        execution_errors=[],
        gate=gate,
        generated_at="2026-07-21T00:00:00Z",
    )

    markdown = render_guardrail_markdown(report)

    assert "## Guardrail Confusion Matrix" in markdown
    assert "## Detection Metrics" in markdown
    assert "## Action Correctness" in markdown
    assert "## MASK Verification" in markdown
    assert "| pii | N/A |" in markdown
