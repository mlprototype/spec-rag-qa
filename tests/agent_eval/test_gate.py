from __future__ import annotations

import json
from pathlib import Path

from ragqa.agent_eval import AgentEvalCase, AgentRunTrace, evaluate_cases
from ragqa.agent_eval.aggregator import aggregate_agent_evaluation
from ragqa.agent_eval.gate import (
    build_baseline,
    evaluate_quality_gate,
    load_gate_config,
    maybe_update_baseline,
)


ROOT = Path(__file__).resolve().parents[2]
GATE_CONFIG_PATH = ROOT / "config" / "agent_quality_gate.yml"


def _aggregation(
    cases: list[AgentEvalCase], traces: list[AgentRunTrace]
) -> dict:
    evaluation = evaluate_cases(cases, traces)
    return aggregate_agent_evaluation(cases, traces, evaluation, [])


def test_critical_failure_is_an_absolute_gate_failure(
    synthetic_cases: list[AgentEvalCase],
    synthetic_traces: list[AgentRunTrace],
) -> None:
    baseline_aggregation = _aggregation(synthetic_cases, synthetic_traces)
    traces = [trace.model_copy(deep=True) for trace in synthetic_traces]
    critical_case = next(case for case in synthetic_cases if case.severity == "critical")
    critical_trace = next(trace for trace in traces if trace.case_id == critical_case.id)
    critical_trace.output.route = "wrong-route"
    current = _aggregation(synthetic_cases, traces)

    result = evaluate_quality_gate(
        current,
        load_gate_config(GATE_CONFIG_PATH),
        build_baseline(baseline_aggregation, generated_at="fixed"),
    )

    check = next(
        item for item in result["checks"] if item["gate_id"] == "critical_task_success"
    )
    assert result["passed"] is False
    assert check["gate_type"] == "absolute"
    assert check["status"] == "failed"


def test_baseline_relative_quality_regression_fails_gate(
    synthetic_cases: list[AgentEvalCase],
    synthetic_traces: list[AgentRunTrace],
) -> None:
    baseline_aggregation = _aggregation(synthetic_cases, synthetic_traces)
    traces = [trace.model_copy(deep=True) for trace in synthetic_traces]
    noncritical_case = next(case for case in synthetic_cases if case.severity == "low")
    trace = next(item for item in traces if item.case_id == noncritical_case.id)
    trace.output.route = "retrieval"
    current = _aggregation(synthetic_cases, traces)

    result = evaluate_quality_gate(
        current,
        load_gate_config(GATE_CONFIG_PATH),
        build_baseline(baseline_aggregation, generated_at="fixed"),
    )

    failed_ids = {
        check["gate_id"]
        for check in result["checks"]
        if check["status"] == "failed"
    }
    assert "overall_task_success_regression" in failed_ids
    assert "route_accuracy_regression" in failed_ids


def test_baseline_relative_latency_regression_fails_gate(
    synthetic_cases: list[AgentEvalCase],
    synthetic_traces: list[AgentRunTrace],
) -> None:
    baseline_aggregation = _aggregation(synthetic_cases, synthetic_traces)
    current = json.loads(json.dumps(baseline_aggregation))
    current["latency"]["p95_ms"] = 628.0

    result = evaluate_quality_gate(
        current,
        load_gate_config(GATE_CONFIG_PATH),
        build_baseline(baseline_aggregation, generated_at="fixed"),
    )

    check = next(
        item
        for item in result["checks"]
        if item["gate_id"] == "latency_p95_regression"
    )
    assert check["gate_type"] == "baseline_relative"
    assert check["threshold"] == 627.0
    assert check["status"] == "failed"


def test_na_gate_is_explicitly_not_applicable() -> None:
    aggregation = {"metrics": {"citation_validity_rate": {"value": None}}}
    config = {
        "schema_version": "1.0",
        "absolute_gates": [
            {
                "id": "citation_validity",
                "metric_path": "metrics.citation_validity_rate.value",
                "operator": "gte",
                "threshold": 1.0,
                "allow_na": True,
            }
        ],
        "baseline_relative_gates": [],
    }

    result = evaluate_quality_gate(aggregation, config, {})

    assert result["passed"] is True
    assert result["checks"][0]["status"] == "not_applicable"
    assert result["checks"][0]["actual"] is None


def test_baseline_is_not_written_without_explicit_update(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    original = {"schema_version": "1.0", "sentinel": "keep"}
    baseline_path.write_text(json.dumps(original), encoding="utf-8")

    updated = maybe_update_baseline(
        baseline_path,
        {"counts": {}, "metrics": {}, "latency": {}},
        enabled=False,
    )

    assert updated is False
    assert json.loads(baseline_path.read_text(encoding="utf-8")) == original


def test_baseline_is_written_with_explicit_update(
    tmp_path: Path,
    synthetic_cases: list[AgentEvalCase],
    synthetic_traces: list[AgentRunTrace],
) -> None:
    baseline_path = tmp_path / "baseline.json"
    aggregation = _aggregation(synthetic_cases, synthetic_traces)

    updated = maybe_update_baseline(
        baseline_path,
        aggregation,
        enabled=True,
        generated_at="2026-01-01T00:00:00Z",
    )

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert updated is True
    assert baseline["generated_at"] == "2026-01-01T00:00:00Z"
    assert baseline["case_count"] == 20
    assert baseline["metrics"]["task_success_rate"] == 1.0
