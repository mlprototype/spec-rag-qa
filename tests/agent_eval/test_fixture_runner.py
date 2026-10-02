from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from ragqa.agent_eval import (
    AgentRunTrace,
    AgentRunner,
    DuplicateFixtureTraceError,
    FixtureRunner,
    FixtureTraceMismatchError,
    FixtureTraceNotFoundError,
    load_cases,
)


ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = ROOT / "data" / "agent_eval" / "cases" / "smoke.json"
TRACES_PATH = ROOT / "data" / "agent_eval" / "fixtures" / "smoke_traces.json"


def test_fixture_runner_satisfies_async_runner_protocol() -> None:
    runner = FixtureRunner.from_json(TRACES_PATH)
    assert isinstance(runner, AgentRunner)


def test_fixture_runner_returns_trace_by_case_id() -> None:
    case = load_cases(CASES_PATH)[0]
    runner = FixtureRunner.from_json(TRACES_PATH)

    trace = asyncio.run(runner.run(case))

    assert trace.case_id == case.id
    assert trace.input.question == case.input.question


def test_fixture_runner_returns_a_deep_copy() -> None:
    case = load_cases(CASES_PATH)[0]
    runner = FixtureRunner.from_json(TRACES_PATH)

    first = asyncio.run(runner.run(case))
    second = asyncio.run(runner.run(case))
    first.output.warning_codes.append("LOCAL_MUTATION")

    assert second.output.warning_codes == []


def test_fixture_runner_owns_a_deep_copy_of_input_trace() -> None:
    with TRACES_PATH.open(encoding="utf-8") as file:
        trace = AgentRunTrace.model_validate(json.load(file)[0])
    runner = FixtureRunner([trace])
    trace.output.warning_codes.append("EXTERNAL_MUTATION")

    case = load_cases(CASES_PATH)[0]
    result = asyncio.run(runner.run(case))

    assert result.output.warning_codes == []


def test_fixture_runner_raises_clear_error_for_missing_trace() -> None:
    case = load_cases(CASES_PATH)[0].model_copy(update={"id": "missing-case"})
    runner = FixtureRunner.from_json(TRACES_PATH)

    with pytest.raises(FixtureTraceNotFoundError, match="missing-case"):
        asyncio.run(runner.run(case))


def test_fixture_runner_rejects_mismatched_case_input() -> None:
    case = load_cases(CASES_PATH)[0]
    case = case.model_copy(
        update={
            "input": case.input.model_copy(
                update={"question": "fixtureと一致しない質問"}
            )
        }
    )
    runner = FixtureRunner.from_json(TRACES_PATH)

    with pytest.raises(FixtureTraceMismatchError, match=case.id):
        asyncio.run(runner.run(case))


def test_fixture_runner_rejects_duplicate_trace_case_ids() -> None:
    with TRACES_PATH.open(encoding="utf-8") as file:
        trace = AgentRunTrace.model_validate(json.load(file)[0])

    with pytest.raises(DuplicateFixtureTraceError, match=trace.case_id):
        FixtureRunner([trace, trace.model_copy(deep=True)])
