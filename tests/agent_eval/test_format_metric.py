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
    trace.output.answer = "必要な見出しがありません"

    check = evaluate_answer_format(case, trace)
    assert check.passed is False
    assert check.failure_type == ANSWER_FORMAT_INVALID


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
