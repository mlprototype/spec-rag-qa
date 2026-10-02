from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from ragqa.agent_eval.advanced_models import JudgeRequest
from ragqa.agent_eval.groundedness import evaluate_groundedness
from ragqa.agent_eval.judge import StructuredJudgeAdapter
from ragqa.agent_eval.models import AgentRunTrace


class StaticJudgeTransport:
    def __init__(self, response: Mapping[str, Any]) -> None:
        self.response = response

    async def complete(self, request: JudgeRequest) -> Mapping[str, Any]:
        return self.response


def test_groundedness_is_supported_claims_over_evaluable_claims(
    synthetic_traces: list[AgentRunTrace],
) -> None:
    trace = next(item for item in synthetic_traces if item.sources)
    source_id = trace.sources[0].source_id
    judge = StructuredJudgeAdapter(
        StaticJudgeTransport(
            {
                "schema_version": "1.0",
                "claims": [
                    {
                        "claim": "supported",
                        "evaluable": True,
                        "supported": True,
                        "source_ids": [source_id],
                        "tool_result_ids": [],
                        "reason": "supported by source",
                    },
                    {
                        "claim": "unsupported",
                        "evaluable": True,
                        "supported": False,
                        "source_ids": [],
                        "tool_result_ids": [],
                        "reason": "no supplied evidence",
                    },
                    {
                        "claim": "opinion",
                        "evaluable": False,
                        "supported": None,
                        "source_ids": [],
                        "tool_result_ids": [],
                        "reason": "not factually evaluable",
                    },
                ],
            }
        ),
        judge_model="mock-model",
    )

    result = asyncio.run(evaluate_groundedness(trace, judge))

    assert result.supported_claims == 1
    assert result.evaluable_claims == 2
    assert result.score == 0.5


def test_groundedness_claim_zero_is_na(
    synthetic_traces: list[AgentRunTrace],
) -> None:
    judge = StructuredJudgeAdapter(
        StaticJudgeTransport({"schema_version": "1.0", "claims": []}),
        judge_model="mock-model",
    )

    result = asyncio.run(evaluate_groundedness(synthetic_traces[0], judge))

    assert result.supported_claims == 0
    assert result.evaluable_claims == 0
    assert result.score is None
