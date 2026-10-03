import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ragqa.agent_eval.aggregator import aggregate_guardrail_evaluation
from ragqa.agent_eval.gate import evaluate_quality_gate, load_gate_config
from ragqa.agent_eval.guardrail import evaluate_guardrail_cases
from ragqa.agent_eval.models import AgentEvalCase, AgentRunTrace


ROOT = Path(__file__).resolve().parents[2]
GATE_CONFIG = ROOT / "config/guardrail_quality_gate.yml"


def _pair(case_id, action, severity="medium"):
    category = "pii" if action == "mask" else "injection"
    detected = action != "allow"
    case = AgentEvalCase.model_validate({
        "schema_version": "1.0", "id": case_id, "category": "guardrail",
        "severity": severity, "input": {"question": f"Synthetic {case_id}"},
        "expected": {
            "query_types": ["guardrail"], "routes": ["gateway"],
            "citation_required": False, "detected": detected,
            "category": category, "action": action,
            "masked_values": ["test@example.test"] if action == "mask" else [],
            "mask_replacement_patterns": ["[EMAIL_REDACTED]"] if action == "mask" else [],
        },
        "budgets": {"max_latency_ms": 100, "max_cost_usd": 0},
    })
    trace = AgentRunTrace.model_validate({
        "schema_version": "1.0", "run_id": case_id, "case_id": case_id,
        "target": "synthetic-gateway", "input": {"question": case.input.question},
        "output": {"answer": "synthetic", "query_type": "guardrail",
                   "route": "gateway", "confidence": 1},
        "guardrail": {
            "detected": detected, "action": action,
            "categories": [category] if detected else [],
            "provider_input": "[EMAIL_REDACTED]" if action == "mask" else None,
            "mask_applied": True if action == "mask" else None,
            "mask_evidence": "provider_input" if action == "mask" else None,
        },
        "control": {}, "usage": {}, "timing": {"latency_ms": 1},
    })
    return case, trace


@pytest.fixture
def action_pairs():
    return [
        _pair("warn", "warn", "critical"), _pair("mask", "mask"),
        _pair("block", "block"), _pair("allow", "allow"),
    ]


def _evaluate(pairs):
    cases, traces = map(list, zip(*pairs))
    evaluation = evaluate_guardrail_cases(cases, traces)
    aggregation = aggregate_guardrail_evaluation(cases, traces, evaluation, [])
    gate = evaluate_quality_gate(aggregation, load_gate_config(GATE_CONFIG), {})
    return evaluation, aggregation, gate


def test_warn_to_allow_mismatch_fails_final_gate(action_pairs):
    assert _evaluate(action_pairs)[2]["passed"] is True
    action_pairs[0][1].guardrail.action = "allow"

    evaluation, aggregation, gate = _evaluate(action_pairs)

    assert evaluation.cases[0].passed is False
    assert aggregation["counts"]["failed_cases"] == 1
    assert gate["passed"] is False
    assert {check["gate_id"] for check in gate["checks"] if not check["passed"]} == {
        "overall_action_correctness"
    }


def test_noncritical_detection_miss_still_uses_recall_threshold():
    pairs = [_pair("critical", "block", "critical"), _pair("mask", "mask")]
    pairs.extend(_pair(f"warn-{i}", "warn") for i in range(18))
    pairs.append(_pair("allow", "allow"))
    pairs[-2][1].guardrail.detected = False

    evaluation, aggregation, gate = _evaluate(pairs)

    assert any(not case.passed for case in evaluation.cases)
    assert aggregation["guardrail"]["overall"]["recall"] == 0.95
    assert aggregation["guardrail"]["action"]["accuracy"] == 1.0
    assert gate["passed"] is True


def test_warn_to_allow_mismatch_returns_cli_failure(tmp_path, action_pairs):
    action_pairs[0][1].guardrail.action = "allow"
    cases_path, traces_path = tmp_path / "cases.json", tmp_path / "traces.json"
    cases_path.write_text(json.dumps([c.model_dump(mode="json") for c, _ in action_pairs]))
    traces_path.write_text(json.dumps([t.model_dump(mode="json") for _, t in action_pairs]))
    report_path = tmp_path / "report.json"
    env = os.environ.copy()
    env.pop("GATEWAY_API_KEY", None)
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/run_guardrail_evaluation.py"),
        "--runner", "fixture", "--cases", str(cases_path), "--traces", str(traces_path),
        "--report-json", str(report_path), "--report-markdown", str(tmp_path / "report.md"),
    ], cwd=ROOT, env=env, capture_output=True, text=True, timeout=20)

    assert result.returncode == 1, result.stderr
    report = json.loads(report_path.read_text())
    assert report["aggregation"]["counts"]["failed_cases"] == 1
    assert report["gate"]["passed"] is False
