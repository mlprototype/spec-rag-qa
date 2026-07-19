from __future__ import annotations

from collections import Counter

import pytest

from ragqa.agent_eval import (
    AgentEvalCase,
    AgentRunTrace,
    DatasetValidationError,
    evaluate_cases,
    validate_case_contracts,
    validate_dataset,
)


def test_synthetic_dataset_has_required_category_distribution(
    synthetic_cases: list[AgentEvalCase],
) -> None:
    counts = Counter(case.category for case in synthetic_cases)

    assert len(synthetic_cases) >= 20
    assert counts["direct"] >= 3
    assert counts["definition"] + counts["retrieval"] >= 4
    assert counts["retrieval_complex"] >= 3
    assert counts["structured_query"] >= 4
    assert counts["compare"] >= 3
    assert counts["insufficient_evidence"] >= 2
    assert counts["fallback"] + counts["degraded"] >= 1


def test_synthetic_dataset_is_public_and_has_complete_expectations(
    synthetic_cases: list[AgentEvalCase],
) -> None:
    for case in synthetic_cases:
        assert {"synthetic", "public"}.issubset(case.tags)
        assert case.severity
        assert case.expected.query_types
        assert case.expected.routes
        assert case.budgets.max_latency_ms > 0
        assert case.budgets.max_cost_usd >= 0


def test_synthetic_dataset_uses_common_agent_routes(
    synthetic_cases: list[AgentEvalCase],
    synthetic_traces: list[AgentRunTrace],
) -> None:
    common_routes = {"direct", "structured_query", "retrieval", "compare"}

    assert {
        route for case in synthetic_cases for route in case.expected.routes
    } <= common_routes
    assert {trace.output.route for trace in synthetic_traces} <= common_routes


def test_structured_query_cases_match_real_agent_contract(
    synthetic_cases: list[AgentEvalCase],
    synthetic_traces: list[AgentRunTrace],
) -> None:
    cases = [case for case in synthetic_cases if case.category == "structured_query"]
    traces_by_id = {trace.case_id: trace for trace in synthetic_traces}

    assert len(cases) == 4
    for case in cases:
        schema = case.expected.tool_argument_schemas["structured_query_tool"]
        assert set(schema["properties"]) == {
            "operation",
            "target_metric",
            "filters",
            "target_dataset",
        }
        assert case.expected.citation_required is False
        assert case.expected.answer_format is not None
        assert case.expected.answer_format.format_type == "natural_language"
        assert traces_by_id[case.id].citations == []


def test_compare_cases_match_real_agent_contract(
    synthetic_cases: list[AgentEvalCase],
) -> None:
    expected_sections = [
        "共通点",
        "相違点",
        "向いているケース（使い分けの指針）",
        "注意点",
    ]
    cases = [case for case in synthetic_cases if case.category == "compare"]

    assert len(cases) == 3
    for case in cases:
        schema = case.expected.tool_argument_schemas["compare_documents"]
        assert set(schema["properties"]) == {"left", "right", "aspects"}
        assert case.expected.answer_format is not None
        assert case.expected.answer_format.format_type == "natural_language"
        assert case.expected.answer_format.required_sections == expected_sections


def test_synthetic_fixture_coverage_and_evaluation_pass(
    synthetic_cases: list[AgentEvalCase],
    synthetic_traces: list[AgentRunTrace],
) -> None:
    validate_dataset(synthetic_cases, synthetic_traces)

    evaluation = evaluate_cases(synthetic_cases, synthetic_traces)

    assert len(evaluation.cases) == len(synthetic_cases)
    assert all(result.passed for result in evaluation.cases)
    assert evaluation.metrics["task_success_rate"].value == 1.0


def test_preflight_rejects_missing_fixture(
    synthetic_cases: list[AgentEvalCase],
    synthetic_traces: list[AgentRunTrace],
) -> None:
    missing_id = synthetic_cases[-1].id

    with pytest.raises(DatasetValidationError, match=missing_id):
        validate_dataset(synthetic_cases, synthetic_traces[:-1])


def test_preflight_rejects_duplicate_saved_trace(
    synthetic_cases: list[AgentEvalCase],
    synthetic_traces: list[AgentRunTrace],
) -> None:
    duplicate = synthetic_traces[0].model_copy(deep=True)

    with pytest.raises(DatasetValidationError, match="Duplicate trace"):
        validate_dataset(synthetic_cases, [*synthetic_traces, duplicate])


def test_preflight_rejects_dangling_tool_schema(
    synthetic_cases: list[AgentEvalCase],
) -> None:
    case = synthetic_cases[0].model_copy(deep=True)
    case.expected.tool_argument_schemas["not-required"] = {"type": "object"}

    with pytest.raises(DatasetValidationError, match="unknown tool"):
        validate_case_contracts([case])


def test_preflight_rejects_unresolved_local_schema_reference(
    synthetic_cases: list[AgentEvalCase],
) -> None:
    case = synthetic_cases[3].model_copy(deep=True)
    tool_name = case.expected.tool_calls[0]
    case.expected.tool_argument_schemas[tool_name] = {
        "$defs": {"query": {"type": "string"}},
        "type": "object",
        "properties": {"query": {"$ref": "#/$defs/missing"}},
    }

    with pytest.raises(DatasetValidationError, match="unresolved.*missing"):
        validate_case_contracts([case])


def test_preflight_rejects_natural_language_with_json_schema(
    synthetic_cases: list[AgentEvalCase],
) -> None:
    case = next(
        item.model_copy(deep=True)
        for item in synthetic_cases
        if item.category == "structured_query"
    )
    assert case.expected.answer_format is not None
    case.expected.answer_format.json_schema = {"type": "object"}

    with pytest.raises(DatasetValidationError, match="natural-language"):
        validate_case_contracts([case])
