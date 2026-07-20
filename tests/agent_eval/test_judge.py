from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import ValidationError

from ragqa.agent_eval.advanced_models import ClaimJudgment, JudgeRequest
from ragqa.agent_eval.judge import (
    GROUNDING_PROMPT_VERSION,
    STRUCTURED_QUERY_EVIDENCE_KIND,
    HttpJudgeTransport,
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


def _trace_with_tool(
    traces: list[AgentRunTrace],
    tool_call: ToolCallTrace,
) -> AgentRunTrace:
    trace = traces[0].model_copy(deep=True)
    trace.sources = []
    trace.tool_calls = [tool_call]
    return trace


def _empty_groundedness_response(schema_version: str = "1.0") -> dict[str, Any]:
    return {"schema_version": schema_version, "claims": []}


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


def test_source_without_snippet_is_not_groundedness_evidence(
    synthetic_traces: list[AgentRunTrace],
) -> None:
    trace = next(item for item in synthetic_traces if item.sources).model_copy(
        deep=True
    )
    for source in trace.sources:
        source.snippet = None
    transport = QueueJudgeTransport([_empty_groundedness_response()])
    judge = StructuredJudgeAdapter(transport, judge_model="mock-model")

    asyncio.run(judge.judge_groundedness(trace))

    assert transport.requests[0].input["sources"] == []


def test_unknown_tool_is_excluded_from_groundedness_evidence(
    synthetic_traces: list[AgentRunTrace],
) -> None:
    trace = _trace_with_tool(
        synthetic_traces,
        ToolCallTrace(
            name="unknown_tool",
            metadata={
                "deterministic": True,
                "evidence_kind": STRUCTURED_QUERY_EVIDENCE_KIND,
            },
            result={"success": True, "value": 42},
        ),
    )
    transport = QueueJudgeTransport([_empty_groundedness_response()])
    judge = StructuredJudgeAdapter(transport, judge_model="mock-model")

    asyncio.run(judge.judge_groundedness(trace))

    assert transport.requests[0].input["tool_results"] == []


def test_deterministic_metadata_is_required_for_tool_evidence(
    synthetic_traces: list[AgentRunTrace],
) -> None:
    trace = _trace_with_tool(
        synthetic_traces,
        ToolCallTrace(
            name="structured_query_tool",
            metadata={"evidence_kind": STRUCTURED_QUERY_EVIDENCE_KIND},
            result={"success": True, "rows": [{"result": 3}]},
        ),
    )
    transport = QueueJudgeTransport([_empty_groundedness_response()])
    judge = StructuredJudgeAdapter(transport, judge_model="mock-model")

    asyncio.run(judge.judge_groundedness(trace))

    assert transport.requests[0].input["tool_results"] == []


@pytest.mark.parametrize("tool_name", ["hybrid_search", "compare_documents"])
def test_source_reference_tool_results_are_not_groundedness_evidence(
    synthetic_traces: list[AgentRunTrace],
    tool_name: str,
) -> None:
    trace = _trace_with_tool(
        synthetic_traces,
        ToolCallTrace(
            name=tool_name,
            metadata={
                "deterministic": True,
                "evidence_kind": STRUCTURED_QUERY_EVIDENCE_KIND,
            },
            result={"source_ids": ["source-1"], "source_count": 1},
        ),
    )
    transport = QueueJudgeTransport([_empty_groundedness_response()])
    judge = StructuredJudgeAdapter(transport, judge_model="mock-model")

    asyncio.run(judge.judge_groundedness(trace))

    assert transport.requests[0].input["tool_results"] == []


def test_structured_query_source_metadata_only_is_not_evidence(
    synthetic_traces: list[AgentRunTrace],
) -> None:
    trace = _trace_with_tool(
        synthetic_traces,
        ToolCallTrace(
            name="structured_query_tool",
            metadata={
                "deterministic": True,
                "evidence_kind": STRUCTURED_QUERY_EVIDENCE_KIND,
            },
            result={
                "success": True,
                "source_ids": ["table-1"],
                "source_count": 1,
                "row_count": 1,
            },
        ),
    )
    transport = QueueJudgeTransport([_empty_groundedness_response()])
    judge = StructuredJudgeAdapter(transport, judge_model="mock-model")

    asyncio.run(judge.judge_groundedness(trace))

    assert transport.requests[0].input["tool_results"] == []


def test_explicit_deterministic_structured_query_facts_are_minimally_projected(
    synthetic_traces: list[AgentRunTrace],
) -> None:
    trace = _trace_with_tool(
        synthetic_traces,
        ToolCallTrace(
            name="structured_query_tool",
            metadata={
                "deterministic": True,
                "evidence_kind": STRUCTURED_QUERY_EVIDENCE_KIND,
            },
            arguments={
                "operation": "count",
                "target_metric": "orders",
                "filters": {"period": "2025-Q1", "confidence": 0.9},
                "target_dataset": "sales",
                "raw_query": "must not be sent",
            },
            result={
                "success": True,
                "operation": "count",
                "source_name": "private database name",
                "row_count": 1,
                "rows": [
                    {
                        "result": 3,
                        "answer_ok": True,
                        "critic": {"confidence": 0.99},
                        "source_ids": ["not-evidence"],
                    }
                ],
                "self_assessment": "correct",
            },
        ),
    )
    tool_result_id = "tool:0:structured_query_tool"
    response = {
        "schema_version": "1.0",
        "claims": [
            {
                "claim": "There are three orders.",
                "evaluable": True,
                "supported": True,
                "source_ids": [],
                "tool_result_ids": [tool_result_id],
                "reason": "The projected row contains the count.",
            }
        ],
    }
    transport = QueueJudgeTransport([response])
    judge = StructuredJudgeAdapter(transport, judge_model="mock-model")

    asyncio.run(judge.judge_groundedness(trace))

    assert transport.requests[0].input["tool_results"] == [
        {
            "tool_result_id": tool_result_id,
            "tool_name": "structured_query_tool",
            "evidence_kind": STRUCTURED_QUERY_EVIDENCE_KIND,
            "arguments": {
                "operation": "count",
                "target_metric": "orders",
                "filters": {"period": "2025-Q1"},
                "target_dataset": "sales",
            },
            "result": {
                "success": True,
                "operation": "count",
                "row_count": 1,
                "rows": [{"result": 3}],
            },
        }
    ]


def test_excluded_tool_result_id_is_a_malformed_judge_response(
    synthetic_traces: list[AgentRunTrace],
) -> None:
    trace = _trace_with_tool(
        synthetic_traces,
        ToolCallTrace(
            name="hybrid_search",
            metadata={"deterministic": True},
            result={"source_ids": ["source-1"]},
        ),
    )
    response = {
        "schema_version": "1.0",
        "claims": [
            {
                "claim": "Excluded Tool evidence.",
                "evaluable": True,
                "supported": True,
                "source_ids": [],
                "tool_result_ids": ["tool:0:hybrid_search"],
                "reason": "This Tool result was not supplied.",
            }
        ],
    }
    transport = QueueJudgeTransport([response, response])
    judge = StructuredJudgeAdapter(transport, judge_model="mock-model")

    with pytest.raises(JudgeMalformedResponseError, match="unknown evidence"):
        asyncio.run(judge.judge_groundedness(trace))

    assert len(transport.requests) == 2


def test_api_key_requires_https_for_judge_transport() -> None:
    endpoint = "http://judge.example.test/evaluate"
    token = "test-bearer-token"

    with pytest.raises(ValueError, match="must use HTTPS") as exc_info:
        HttpJudgeTransport(
            endpoint,
            api_key=token,
            allowed_hosts={"judge.example.test"},
        )

    assert endpoint not in str(exc_info.value)
    assert token not in str(exc_info.value)


def test_judge_transport_rejects_host_outside_allowlist() -> None:
    endpoint = "https://unapproved.example.test/evaluate"

    with pytest.raises(ValueError, match="configured allowlist") as exc_info:
        HttpJudgeTransport(
            endpoint,
            allowed_hosts={"approved.example.test"},
        )

    assert endpoint not in str(exc_info.value)
    assert "unapproved.example.test" not in str(exc_info.value)


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


def test_malformed_judge_error_does_not_echo_source_content(
    synthetic_traces: list[AgentRunTrace],
) -> None:
    trace = next(item for item in synthetic_traces if item.sources).model_copy(
        deep=True
    )
    sensitive_content = "PRIVATE_SOURCE_CONTENT_MUST_NOT_BE_LOGGED"
    trace.sources[0].snippet = sensitive_content
    malformed = {
        "schema_version": trace.schema_version,
        "claims": sensitive_content,
    }
    transport = QueueJudgeTransport([malformed, malformed])
    judge = StructuredJudgeAdapter(transport, judge_model="mock-model")

    with pytest.raises(JudgeMalformedResponseError) as exc_info:
        asyncio.run(judge.judge_groundedness(trace))

    assert sensitive_content not in str(exc_info.value)
    assert "required schema" in str(exc_info.value)


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
