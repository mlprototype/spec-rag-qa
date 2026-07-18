from ragqa.agent_eval import (
    CITATION_INVALID,
    CITATION_MISSING,
    AgentEvalCase,
    AgentRunTrace,
)
from ragqa.agent_eval.metrics.citation import (
    evaluate_citation_presence,
    evaluate_citation_validity,
)


def test_citation_missing_failure_type(
    smoke_pair: tuple[AgentEvalCase, AgentRunTrace],
) -> None:
    case, original = smoke_pair
    trace = original.model_copy(deep=True)
    trace.citations = []

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
