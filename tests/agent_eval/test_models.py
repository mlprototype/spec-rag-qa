from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ragqa.agent_eval import (
    AgentEvalCase,
    AgentRunTrace,
    CaseEvaluationResult,
    CheckResult,
    DuplicateCaseIdError,
    TimingTrace,
    ToolCallTrace,
    UsageTrace,
    load_cases,
)


ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = ROOT / "data" / "agent_eval" / "cases" / "smoke.json"
TRACES_PATH = ROOT / "data" / "agent_eval" / "fixtures" / "smoke_traces.json"


def _case_payload() -> dict[str, object]:
    with CASES_PATH.open(encoding="utf-8") as file:
        return json.load(file)[0]


def _trace_payload() -> dict[str, object]:
    with TRACES_PATH.open(encoding="utf-8") as file:
        return json.load(file)[0]


def test_smoke_cases_and_traces_are_valid_contracts() -> None:
    cases = load_cases(CASES_PATH)
    with TRACES_PATH.open(encoding="utf-8") as file:
        traces = [AgentRunTrace.model_validate(item) for item in json.load(file)]

    assert len(cases) >= 3
    assert len(traces) >= 3
    assert {case.id for case in cases} == {trace.case_id for trace in traces}
    assert {case.id: case.input.question for case in cases} == {
        trace.case_id: trace.input.question for trace in traces
    }


def test_smoke_citations_resolve_to_observed_sources() -> None:
    cases = {case.id: case for case in load_cases(CASES_PATH)}
    with TRACES_PATH.open(encoding="utf-8") as file:
        traces = [AgentRunTrace.model_validate(item) for item in json.load(file)]

    for trace in traces:
        source_ids = {source.source_id for source in trace.sources}
        assert all(citation.source_id in source_ids for citation in trace.citations)
        assert all(
            citation.citation_id in trace.output.answer
            for citation in trace.citations
        )
        if cases[trace.case_id].expected.citation_required:
            assert trace.citations


def test_schema_version_is_required() -> None:
    payload = _case_payload()
    payload.pop("schema_version")

    with pytest.raises(ValidationError, match="schema_version"):
        AgentEvalCase.model_validate(payload)


@pytest.mark.parametrize(
    ("field_path", "invalid_value"),
    [
        (("expected", "routes"), [""]),
        (("budgets", "max_latency_ms"), -1),
        (("budgets", "max_cost_usd"), -0.01),
    ],
)
def test_invalid_case_values_are_rejected(
    field_path: tuple[str, str], invalid_value: object
) -> None:
    payload = _case_payload()
    parent, field = field_path
    nested = payload[parent]
    assert isinstance(nested, dict)
    nested[field] = invalid_value

    with pytest.raises(ValidationError):
        AgentEvalCase.model_validate(payload)


def test_invalid_trace_values_are_rejected() -> None:
    payload = _trace_payload()
    output = payload["output"]
    timing = payload["timing"]
    assert isinstance(output, dict)
    assert isinstance(timing, dict)

    output["confidence"] = 1.1
    timing["latency_ms"] = -1

    with pytest.raises(ValidationError):
        AgentRunTrace.model_validate(payload)

    with pytest.raises(ValidationError):
        UsageTrace(cost_usd=-0.001)
    with pytest.raises(ValidationError):
        TimingTrace(latency_ms=-0.1)


def test_unknown_fields_are_rejected_and_metadata_is_the_extension_point() -> None:
    payload = _trace_payload()
    payload["passed"] = True

    with pytest.raises(ValidationError, match="passed"):
        AgentRunTrace.model_validate(payload)

    payload.pop("passed")
    payload["metadata"] = {"vendor_trace_id": "trace-123"}
    trace = AgentRunTrace.model_validate(payload)
    assert trace.metadata["vendor_trace_id"] == "trace-123"


def test_mutable_defaults_are_not_shared() -> None:
    first = ToolCallTrace(name="search")
    second = ToolCallTrace(name="search")
    first.arguments["query"] = "one"
    first.metadata["source"] = "fixture"

    assert second.arguments == {}
    assert second.metadata == {}


def test_duplicate_case_ids_raise_clear_error(tmp_path: Path) -> None:
    payload = [_case_payload(), _case_payload()]
    path = tmp_path / "duplicate_cases.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(DuplicateCaseIdError, match="Duplicate case id.*agent-smoke"):
        load_cases(path)


def test_evaluation_results_are_separate_from_run_trace() -> None:
    check = CheckResult(
        schema_version="1.0",
        check_id="route",
        passed=True,
        score=1.0,
    )
    result = CaseEvaluationResult(
        schema_version="1.0",
        case_id="case-1",
        run_id="run-1",
        passed=True,
        checks=[check],
    )

    assert result.passed is True
    assert "passed" not in AgentRunTrace.model_fields
