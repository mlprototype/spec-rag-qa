from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragqa.agent_eval import AgentEvalCase, AgentRunTrace, load_cases
from ragqa.agent_eval.adapters.trace_file import load_saved_traces


ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = ROOT / "data" / "agent_eval" / "cases" / "smoke.json"
TRACES_PATH = ROOT / "data" / "agent_eval" / "fixtures" / "smoke_traces.json"
SYNTHETIC_CASES_PATH = (
    ROOT / "data" / "agent_eval" / "cases" / "phase6_synthetic.json"
)
SYNTHETIC_TRACES_PATH = (
    ROOT
    / "data"
    / "agent_eval"
    / "fixtures"
    / "phase6_synthetic_traces.json"
)


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


@pytest.fixture
def synthetic_cases() -> list[AgentEvalCase]:
    return load_cases(SYNTHETIC_CASES_PATH)


@pytest.fixture
def synthetic_traces() -> list[AgentRunTrace]:
    return load_saved_traces(SYNTHETIC_TRACES_PATH)
