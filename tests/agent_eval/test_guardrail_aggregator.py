from __future__ import annotations

import json

from ragqa.agent_eval import AgentEvalCase, AgentRunTrace
from ragqa.agent_eval.adapters.gateway import GatewayGuardrailAdapter
from ragqa.agent_eval.aggregator import aggregate_guardrail_evaluation
from ragqa.agent_eval.guardrail import evaluate_guardrail_cases


def test_guardrail_confusion_matrix_and_rates(
    guardrail_cases: list[AgentEvalCase],
    guardrail_traces: list[AgentRunTrace],
) -> None:
    selected_cases = [
        next(
            case
            for case in guardrail_cases
            if case.expected.category == category
            and case.expected.detected is detected
        )
        for category, detected in (
            ("pii", True),
            ("pii", False),
            ("injection", True),
            ("injection", False),
        )
    ]
    traces_by_id = {trace.case_id: trace for trace in guardrail_traces}
    selected_traces = [
        traces_by_id[case.id].model_copy(deep=True) for case in selected_cases
    ]
    selected_traces[0].guardrail.detected = False
    selected_traces[1].guardrail.detected = True
    evaluation = evaluate_guardrail_cases(selected_cases, selected_traces)

    aggregation = aggregate_guardrail_evaluation(
        selected_cases, selected_traces, evaluation, []
    )

    overall = aggregation["guardrail"]["overall"]
    assert overall["confusion_matrix"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert overall["precision"] == 0.5
    assert overall["recall"] == 0.5
    assert overall["f1"] == 0.5
    assert overall["false_positive_rate"] == 0.5


def test_unknown_and_execution_errors_are_not_true_negatives(
    guardrail_cases: list[AgentEvalCase],
    guardrail_traces: list[AgentRunTrace],
) -> None:
    negative_cases = [
        case for case in guardrail_cases if case.expected.detected is False
    ][:2]
    traces_by_id = {trace.case_id: trace for trace in guardrail_traces}
    unknown_trace = traces_by_id[negative_cases[0].id].model_copy(deep=True)
    unknown_trace.guardrail.detected = None
    unknown_trace.guardrail.action = "unknown"
    evaluation = evaluate_guardrail_cases([negative_cases[0]], [unknown_trace])
    errors = [
        {
            "case_id": negative_cases[1].id,
            "error_type": "GatewayTransportError",
            "message": "Gateway request failed",
        }
    ]

    aggregation = aggregate_guardrail_evaluation(
        negative_cases, [unknown_trace], evaluation, errors
    )

    overall = aggregation["guardrail"]["overall"]
    assert overall["confusion_matrix"] == {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    assert overall["precision"] is None
    assert overall["recall"] is None
    assert overall["f1"] is None
    assert overall["false_positive_rate"] is None
    assert overall["unknown"] == 1
    assert overall["execution_errors"] == 1
    assert aggregation["guardrail"]["action"]["evaluable"] == 0


def test_unknown_action_is_counted_even_when_detection_is_observable(
    guardrail_cases: list[AgentEvalCase],
    guardrail_traces: list[AgentRunTrace],
) -> None:
    case = next(item for item in guardrail_cases if item.expected.detected)
    trace = next(
        item.model_copy(deep=True)
        for item in guardrail_traces
        if item.case_id == case.id
    )
    trace.guardrail.action = "unknown"
    evaluation = evaluate_guardrail_cases([case], [trace])

    aggregation = aggregate_guardrail_evaluation(
        [case], [trace], evaluation, []
    )

    assert aggregation["guardrail"]["overall"]["confusion_matrix"]["tp"] == 1
    assert aggregation["guardrail"]["action"]["unknown"] == 1
    assert aggregation["counts"]["unknown_observations"] == 1


def test_detected_allow_is_positive_and_action_correct(
    guardrail_cases: list[AgentEvalCase],
) -> None:
    case = next(
        item.model_copy(deep=True)
        for item in guardrail_cases
        if item.expected.category == "injection" and item.expected.detected
    )
    case.expected.action = "allow"
    trace = GatewayGuardrailAdapter().normalize(
        case,
        status_code=200,
        headers={
            "X-Gateway-Security-Action": "ALLOW",
            "X-Gateway-Security-Category": "INJECTION",
        },
        body=json.dumps({"id": "chatcmpl-test", "choices": []}),
        latency_ms=8.0,
    )
    evaluation = evaluate_guardrail_cases([case], [trace])

    aggregation = aggregate_guardrail_evaluation(
        [case], [trace], evaluation, []
    )

    assert evaluation.cases[0].passed is True
    assert aggregation["guardrail"]["overall"]["confusion_matrix"] == {
        "tp": 1,
        "fp": 0,
        "fn": 0,
        "tn": 0,
    }
    allow = aggregation["guardrail"]["action"]["by_expected"]["allow"]
    assert allow["correct"] == 1
    assert allow["accuracy"] == 1.0
    assert aggregation["counts"]["unknown_observations"] == 0
