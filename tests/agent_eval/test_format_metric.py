from ragqa.agent_eval import (
    ANSWER_FORMAT_INVALID,
    AgentEvalCase,
    AgentRunTrace,
    AnswerFormatExpectation,
)
from ragqa.agent_eval.metrics.format import evaluate_answer_format


def test_required_section_format_failure(
    smoke_pair: tuple[AgentEvalCase, AgentRunTrace],
) -> None:
    case, original = smoke_pair
    trace = original.model_copy(deep=True)
    trace.output.answer = "この回答には回答セクションを書きません。"

    check = evaluate_answer_format(case, trace)
    assert check.passed is False
    assert check.failure_type == ANSWER_FORMAT_INVALID


def test_markdown_heading_is_a_required_section(
    smoke_pair: tuple[AgentEvalCase, AgentRunTrace],
) -> None:
    case, trace = smoke_pair
    assert evaluate_answer_format(case, trace).passed is True


def test_json_schema_answer_format_passes_and_fails(
    smoke_pair: tuple[AgentEvalCase, AgentRunTrace],
) -> None:
    original_case, original_trace = smoke_pair
    case = original_case.model_copy(deep=True)
    case.expected.answer_format = AnswerFormatExpectation(
        json_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
    )
    trace = original_trace.model_copy(deep=True)
    trace.output.answer = '{"answer": "8文字以上"}'
    assert evaluate_answer_format(case, trace).passed is True

    trace.output.answer = '{"unexpected": true}'
    check = evaluate_answer_format(case, trace)
    assert check.passed is False
    assert check.failure_type == ANSWER_FORMAT_INVALID


def test_json_required_section_uses_dot_path(
    smoke_pair: tuple[AgentEvalCase, AgentRunTrace],
) -> None:
    original_case, original_trace = smoke_pair
    case = original_case.model_copy(deep=True)
    case.expected.answer_format = AnswerFormatExpectation(
        required_sections=["result.summary"]
    )
    trace = original_trace.model_copy(deep=True)
    trace.output.answer = '{"result": {"summary": null}}'
    assert evaluate_answer_format(case, trace).passed is True

    trace.output.answer = '{"result": {}}'
    assert evaluate_answer_format(case, trace).failure_type == ANSWER_FORMAT_INVALID


def test_natural_language_format_rejects_json_and_empty_answer(
    smoke_pair: tuple[AgentEvalCase, AgentRunTrace],
) -> None:
    original_case, original_trace = smoke_pair
    case = original_case.model_copy(deep=True)
    case.expected.answer_format = AnswerFormatExpectation(
        format_type="natural_language"
    )
    trace = original_trace.model_copy(deep=True)
    trace.output.answer = "該当するデータは3件です。"
    assert evaluate_answer_format(case, trace).passed is True

    trace.output.answer = '{"result": 3}'
    check = evaluate_answer_format(case, trace)
    assert check.passed is False
    assert check.failure_type == ANSWER_FORMAT_INVALID

    trace.output.answer = "  "
    assert evaluate_answer_format(case, trace).passed is False
