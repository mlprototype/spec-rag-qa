from __future__ import annotations

from collections import Counter

import pytest

from ragqa.agent_eval import (
    AgentEvalCase,
    AgentRunTrace,
    DatasetValidationError,
    validate_case_contracts,
    validate_dataset,
)
from ragqa.agent_eval.aggregator import aggregate_guardrail_evaluation
from ragqa.agent_eval.guardrail import evaluate_guardrail_cases


def test_guardrail_dataset_is_balanced_and_public(
    guardrail_cases: list[AgentEvalCase],
) -> None:
    categories = Counter(case.expected.category for case in guardrail_cases)
    labels = Counter(case.expected.detected for case in guardrail_cases)

    assert len(guardrail_cases) >= 25
    assert categories == {"injection": 12, "pii": 12, "compound": 6}
    assert labels == {True: 15, False: 15}
    for case in guardrail_cases:
        assert {"synthetic", "public", "guardrail"}.issubset(case.tags)
        assert case.expected.detected is not None
        assert case.expected.category is not None
        assert case.expected.action is not None
        assert len(case.input.question) <= 120


def test_mask_cases_declare_verifiable_targets(
    guardrail_cases: list[AgentEvalCase],
) -> None:
    mask_cases = [
        case for case in guardrail_cases if case.expected.action == "mask"
    ]

    assert len(mask_cases) == 4
    for case in mask_cases:
        assert case.expected.masked_values
        assert case.expected.mask_replacement_patterns


def test_guardrail_fixture_coverage_and_aggregate_pass(
    guardrail_cases: list[AgentEvalCase],
    guardrail_traces: list[AgentRunTrace],
) -> None:
    validate_dataset(guardrail_cases, guardrail_traces)
    evaluation = evaluate_guardrail_cases(guardrail_cases, guardrail_traces)
    aggregation = aggregate_guardrail_evaluation(
        guardrail_cases, guardrail_traces, evaluation, []
    )

    assert aggregation["guardrail"]["overall"]["confusion_matrix"] == {
        "tp": 15,
        "fp": 0,
        "fn": 0,
        "tn": 15,
    }
    assert aggregation["guardrail"]["overall"]["precision"] == 1.0
    assert aggregation["guardrail"]["overall"]["recall"] == 1.0
    assert aggregation["guardrail"]["categories"]["pii"]["recall"] == 1.0
    assert aggregation["guardrail"]["categories"]["injection"]["recall"] == 1.0
    assert aggregation["guardrail"]["action"]["accuracy"] == 1.0
    assert aggregation["guardrail"]["mask"]["accuracy"] == 1.0


def test_guardrail_contract_requires_complete_expectation(
    guardrail_cases: list[AgentEvalCase],
) -> None:
    case = guardrail_cases[0].model_copy(deep=True)
    case.expected.action = None

    with pytest.raises(DatasetValidationError, match="detected, category, and action"):
        validate_case_contracts([case])


def test_mask_contract_requires_target_and_replacement(
    guardrail_cases: list[AgentEvalCase],
) -> None:
    case = next(
        item.model_copy(deep=True)
        for item in guardrail_cases
        if item.expected.action == "mask"
    )
    case.expected.masked_values = []

    with pytest.raises(DatasetValidationError, match="MASK values"):
        validate_case_contracts([case])
