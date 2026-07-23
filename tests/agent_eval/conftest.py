from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragqa.agent_eval import AgentEvalCase, AgentRunTrace, load_cases


ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = ROOT / "data" / "agent_eval" / "cases" / "smoke.json"
TRACES_PATH = ROOT / "data" / "agent_eval" / "fixtures" / "smoke_traces.json"


@pytest.fixture
def smoke_cases() -> list[AgentEvalCase]:
    return load_cases(CASES_PATH)


@pytest.fixture
def smoke_traces() -> list[AgentRunTrace]:
    with TRACES_PATH.open(encoding="utf-8") as file:
        return [AgentRunTrace.model_validate(item) for item in json.load(file)]


@pytest.fixture
def smoke_pair(
    smoke_cases: list[AgentEvalCase], smoke_traces: list[AgentRunTrace]
) -> tuple[AgentEvalCase, AgentRunTrace]:
    return smoke_cases[0], smoke_traces[0]
