from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from ragqa.agent_eval.adapters.trace_file import load_saved_traces
from ragqa.agent_eval.judge import HttpJudgeTransport, StructuredJudgeAdapter


ROOT = Path(__file__).resolve().parents[2]
TRACES_PATH = (
    ROOT / "data" / "agent_eval" / "fixtures" / "phase6_synthetic_traces.json"
)


def test_external_judge_contract_when_explicitly_enabled() -> None:
    if os.environ.get("RUN_EXTERNAL_AGENT_JUDGE_TESTS") != "1":
        pytest.skip("External Judge test is opt-in")
    judge_url = os.environ.get("AGENT_EVAL_JUDGE_URL")
    judge_model = os.environ.get("AGENT_EVAL_JUDGE_MODEL")
    if not judge_url or not judge_model:
        pytest.fail("External Judge URL and model must be configured")
    trace = next(item for item in load_saved_traces(TRACES_PATH) if item.sources)
    judge = StructuredJudgeAdapter(
        HttpJudgeTransport(
            judge_url,
            api_key=os.environ.get("AGENT_EVAL_JUDGE_API_KEY"),
        ),
        judge_model=judge_model,
    )

    decision = asyncio.run(judge.judge_groundedness(trace))

    assert decision.judge.judge_model == judge_model
    assert decision.response.schema_version == trace.schema_version
