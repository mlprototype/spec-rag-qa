from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import ValidationError

from ragqa.agent_eval.advanced_models import ClaimJudgment, JudgeRequest
from ragqa.agent_eval.judge import (
    GROUNDING_PROMPT_VERSION,
    JudgeMalformedResponseError,
    StructuredJudgeAdapter,
)
from ragqa.agent_eval.models import AgentRunTrace, ToolCallTrace


class QueueJudgeTransport:
    def __init__(self, responses: list[str | Mapping[str, Any]]) -> None:
        self.responses = list(responses)
        self.requests: list[JudgeRequest] = []

    async def complete(self, request: JudgeRequest) -> str | Mapping[str, Any]:
        self.requests.append(request)
        return self.responses.pop(0)


def test_groundedness_judge_validates_schema_and_saves_metadata(
    synthetic_traces: list[AgentRunTrace],
) -> None:
    trace = next(item for item in synthetic_traces if item.sources)
    source_id = trace.sources[0].source_id
    transport = QueueJudgeTransport(
        [
            {
                "schema_version": "1.0",
                "claims": [
                    {
                        "claim": "The configured timeout is 30 minutes.",
                        "evaluable": True,
                        "supported": True,
                        "source_ids": [source_id],
                        "tool_result_ids": [],
                        "reason": "The supplied source states the timeout.",
                    }
                ],
            }
        ]
    )
    judge = StructuredJudgeAdapter(transport, judge_model="mock-model")

    decision = asyncio.run(judge.judge_groundedness(trace))

    assert decision.response.claims[0].supported is True
    assert decision.judge.judge_model == "mock-model"
    assert decision.judge.judge_prompt_version == GROUNDING_PROMPT_VERSION
    assert decision.judge.attempts == 1
    request_input = transport.requests[0].input
    assert "confidence" not in request_input
    assert "critic" not in request_input
    assert "answer_ok" not in request_input


def test_agent_self_evaluation_is_removed_from_tool_evidence(
    synthetic_traces: list[AgentRunTrace],
) -> None:
    trace = next(item for item in synthetic_traces if item.tool_calls).model_copy(
        deep=True
    )
    trace.tool_calls[0].result = {
        "value": "business evidence",
        "answer_ok": True,
        "critic": {"confidence": 0.99},
    }
    trace.tool_calls.append(
        ToolCallTrace(name="answer_critic", result={"answer_ok": True})
    )
    tool_result_id = f"tool:0:{trace.tool_calls[0].name}"
    transport = QueueJudgeTransport(
        [
            {
                "schema_version": "1.0",
                "claims": [
                    {
                        "claim": "The business value is supported.",
                        "evaluable": True,
                        "supported": True,
                        "source_ids": [],
                        "tool_result_ids": [tool_result_id],
                        "reason": "Deterministic Tool result contains the value.",
                    }
                ],
            }
        ]
    )
    judge = StructuredJudgeAdapter(transport, judge_model="mock-model")

    asyncio.run(judge.judge_groundedness(trace))

    tool_results = transport.requests[0].input["tool_results"]
    assert len(tool_results) == 1
    assert tool_results[0]["result"] == {"value": "business evidence"}


def test_nondeterministic_tool_result_is_not_judge_evidence(
    synthetic_traces: list[AgentRunTrace],
) -> None:
    trace = next(item for item in synthetic_traces if item.tool_calls).model_copy(
        deep=True
    )
    trace.tool_calls[0].metadata["deterministic"] = False
    transport = QueueJudgeTransport(
        [{"schema_version": trace.schema_version, "claims": []}]
    )
    judge = StructuredJudgeAdapter(transport, judge_model="mock-model")

    asyncio.run(judge.judge_groundedness(trace))

    assert transport.requests[0].input["tool_results"] == []


def test_malformed_judge_response_retries_once_then_succeeds(
    synthetic_traces: list[AgentRunTrace],
) -> None:
    trace = next(item for item in synthetic_traces if item.sources)
    transport = QueueJudgeTransport(
        [
            "not-json",
            {"schema_version": "1.0", "claims": []},
        ]
    )
    judge = StructuredJudgeAdapter(transport, judge_model="mock-model")

    decision = asyncio.run(judge.judge_groundedness(trace))

    assert decision.judge.attempts == 2
    assert len(transport.requests) == 2


def test_semantic_judge_retries_wrong_group_count_once() -> None:
    transport = QueueJudgeTransport(
        [
            {
                "schema_version": "1.0",
                "group_ids": [0],
                "reason": "Missing one answer assignment.",
            },
            {
                "schema_version": "1.0",
                "group_ids": [0, 0],
                "reason": "Both answers are semantically equivalent.",
            },
        ]
    )
    judge = StructuredJudgeAdapter(transport, judge_model="mock-model")

    decision = asyncio.run(
        judge.judge_semantic_consistency("case-1", ["answer one", "answer two"])
    )

    assert decision.judge.attempts == 2
    assert decision.response.group_ids == [0, 0]


def test_malformed_judge_response_is_not_converted_to_unsupported(
    synthetic_traces: list[AgentRunTrace],
) -> None:
    transport = QueueJudgeTransport(["bad-1", "bad-2"])
    judge = StructuredJudgeAdapter(transport, judge_model="mock-model")

    with pytest.raises(JudgeMalformedResponseError, match="after 2 attempt"):
        asyncio.run(judge.judge_groundedness(synthetic_traces[0]))

    assert len(transport.requests) == 2


def test_unknown_judge_evidence_is_a_malformed_response(
    synthetic_traces: list[AgentRunTrace],
) -> None:
    response = {
        "schema_version": "1.0",
        "claims": [
            {
                "claim": "Unsupported evidence reference.",
                "evaluable": True,
                "supported": True,
                "source_ids": ["missing-source"],
                "tool_result_ids": [],
                "reason": "Invalid test fixture.",
            }
        ],
    }
    transport = QueueJudgeTransport([response, response])
    judge = StructuredJudgeAdapter(transport, judge_model="mock-model")

    with pytest.raises(JudgeMalformedResponseError, match="unknown evidence"):
        asyncio.run(judge.judge_groundedness(synthetic_traces[0]))


def test_claim_schema_distinguishes_non_evaluable_from_unsupported() -> None:
    with pytest.raises(ValidationError, match="requires a supported decision"):
        ClaimJudgment(
            claim="claim",
            evaluable=True,
            supported=None,
            reason="missing decision",
        )

    with pytest.raises(ValidationError, match="must use supported=null"):
        ClaimJudgment(
            claim="claim",
            evaluable=False,
            supported=False,
            reason="not evaluable",
        )


def test_judge_retry_is_bounded_to_at_most_one() -> None:
    with pytest.raises(ValueError, match="must be 0 or 1"):
        StructuredJudgeAdapter(
            QueueJudgeTransport([]),
            judge_model="mock-model",
            malformed_retries=2,
        )
