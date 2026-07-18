from ragqa.agent_eval import (
    CITATION_INVALID,
    CITATION_MISSING,
    AgentEvalCase,
    AgentRunTrace,
)
from ragqa.agent_eval.metrics.citation import (
    evaluate_citation_presence,
    evaluate_citation_validity,
    extract_answer_citation_ids,
)


def test_citation_missing_failure_type(
    smoke_pair: tuple[AgentEvalCase, AgentRunTrace],
) -> None:
    case, original = smoke_pair
    trace = original.model_copy(deep=True)
    trace.citations = []
    trace.output.answer = "## 回答\nCitationなしの回答"

    check = evaluate_citation_presence(case, trace)
    assert check.passed is False
    assert check.failure_type == CITATION_MISSING


def test_citation_source_must_resolve(
    smoke_pair: tuple[AgentEvalCase, AgentRunTrace],
) -> None:
    case, original = smoke_pair
    trace = original.model_copy(deep=True)
    trace.citations[0].source_id = "missing-source"

    presence = evaluate_citation_presence(case, trace)
    validity = evaluate_citation_validity(case, trace)
    assert presence.passed is True
    assert validity.passed is False
    assert validity.failure_type == CITATION_INVALID


def test_recorded_citation_must_appear_in_answer(
    smoke_pair: tuple[AgentEvalCase, AgentRunTrace],
) -> None:
    case, original = smoke_pair
    trace = original.model_copy(deep=True)
    trace.output.answer = "引用IDを含まない回答"

    assert evaluate_citation_presence(case, trace).failure_type == CITATION_MISSING
    assert evaluate_citation_validity(case, trace).failure_type == CITATION_INVALID


def test_citation_id_must_match_exactly(
    smoke_pair: tuple[AgentEvalCase, AgentRunTrace],
) -> None:
    case, original = smoke_pair
    trace = original.model_copy(deep=True)
    trace.citations[0].citation_id = "cite-1"
    trace.output.answer = "回答です。[cite-10]"

    assert extract_answer_citation_ids(trace.output.answer) == {"cite-10"}
    assert evaluate_citation_presence(case, trace).passed is True
    validity = evaluate_citation_validity(case, trace)
    assert validity.passed is False
    assert validity.failure_type == CITATION_INVALID


def test_unrecorded_answer_citation_is_invalid(
    smoke_pair: tuple[AgentEvalCase, AgentRunTrace],
) -> None:
    case, original = smoke_pair
    trace = original.model_copy(deep=True)
    trace.output.answer += " [hallucinated-citation]"

    validity = evaluate_citation_validity(case, trace)
    assert validity.passed is False
    assert validity.failure_type == CITATION_INVALID
    assert any(
        item["reason"] == "citation_not_recorded"
        for item in validity.details["invalid_citations"]
    )
