from __future__ import annotations

from collections import defaultdict

from jsonschema import SchemaError
from jsonschema.validators import validator_for

from ragqa.agent_eval.assertions import evaluate_assertion
from ragqa.agent_eval.failure_types import (
    REQUIRED_TOOL_NOT_CALLED,
    TOOL_ARGUMENT_SCHEMA_INVALID,
    TOOL_ARGUMENT_SEMANTIC_MISMATCH,
    UNEXPECTED_TOOL_CALLED,
)
from ragqa.agent_eval.models import (
    AgentEvalCase,
    AgentRunTrace,
    CheckResult,
    ToolCallTrace,
)


REQUIRED_TOOL_CHECK_ID = "required_tool_calls"
UNEXPECTED_TOOL_CHECK_ID = "unexpected_tool_calls"
TOOL_SCHEMA_CHECK_ID = "tool_argument_schema"
TOOL_SEMANTIC_CHECK_ID = "tool_argument_semantics"


def evaluate_required_tool_calls(
    case: AgentEvalCase, trace: AgentRunTrace
) -> CheckResult:
    required_tools = list(dict.fromkeys(case.expected.tool_calls))
    called_tools = {call.name for call in trace.tool_calls}
    missing = [name for name in required_tools if name not in called_tools]
    called_required_count = len(required_tools) - len(missing)
    required = bool(required_tools)
    passed = not missing
    score = called_required_count / len(required_tools) if required_tools else None

    return CheckResult(
        schema_version=case.schema_version,
        check_id=REQUIRED_TOOL_CHECK_ID,
        passed=passed,
        required=required,
        failure_type=None if passed else REQUIRED_TOOL_NOT_CALLED,
        score=score,
        message=None if passed else f"Required tool(s) not called: {', '.join(missing)}",
        details={
            "required_tools": required_tools,
            "called_tools": sorted(called_tools),
            "missing_tools": missing,
            "called_required_count": called_required_count,
            "required_count": len(required_tools),
        },
    )


def evaluate_unexpected_tool_calls(
    case: AgentEvalCase, trace: AgentRunTrace
) -> CheckResult:
    expected_tools = set(case.expected.tool_calls)
    forbidden_tools = set(case.expected.forbidden_tool_calls)
    actual_tools = [call.name for call in trace.tool_calls]
    unexpected = [name for name in actual_tools if name not in expected_tools]
    unexpected_rate = len(unexpected) / len(actual_tools) if actual_tools else 0.0
    passed = not unexpected

    return CheckResult(
        schema_version=case.schema_version,
        check_id=UNEXPECTED_TOOL_CHECK_ID,
        passed=passed,
        failure_type=None if passed else UNEXPECTED_TOOL_CALLED,
        score=1.0 - unexpected_rate,
        message=None if passed else f"Unexpected tool call(s): {', '.join(unexpected)}",
        details={
            "expected_tools": sorted(expected_tools),
            "forbidden_tools": sorted(forbidden_tools),
            "actual_tools": actual_tools,
            "unexpected_tools": unexpected,
            "unexpected_count": len(unexpected),
            "actual_count": len(actual_tools),
        },
    )


def evaluate_tool_argument_schema(
    case: AgentEvalCase, trace: AgentRunTrace
) -> CheckResult:
    schemas = case.expected.tool_argument_schemas
    calls_by_name = _calls_by_name(trace)
    required_tools = set(case.expected.tool_calls)
    valid_count = 0
    evaluated_count = 0
    missing_tools: list[str] = []
    errors: list[dict[str, object]] = []

    for tool_name, schema in schemas.items():
        calls = calls_by_name.get(tool_name, [])
        if not calls:
            if tool_name in required_tools:
                evaluated_count += 1
                missing_tools.append(tool_name)
            continue

        validator_class = validator_for(schema)
        try:
            validator_class.check_schema(schema)
        except SchemaError as exc:
            raise ValueError(f"Invalid JSON Schema for tool {tool_name!r}") from exc
        validator = validator_class(schema)

        for call_index, call in enumerate(calls):
            evaluated_count += 1
            validation_errors = sorted(
                validator.iter_errors(call.arguments),
                key=lambda error: tuple(str(part) for part in error.absolute_path),
            )
            if not validation_errors:
                valid_count += 1
                continue
            errors.append(
                {
                    "tool": tool_name,
                    "call_index": call_index,
                    "messages": [error.message for error in validation_errors],
                }
            )

    required = bool(evaluated_count or schemas)
    passed = not missing_tools and not errors
    failure_type = None
    if missing_tools:
        failure_type = REQUIRED_TOOL_NOT_CALLED
    elif errors:
        failure_type = TOOL_ARGUMENT_SCHEMA_INVALID

    return CheckResult(
        schema_version=case.schema_version,
        check_id=TOOL_SCHEMA_CHECK_ID,
        passed=passed,
        required=required,
        failure_type=failure_type,
        score=(valid_count / evaluated_count) if evaluated_count else None,
        message=_tool_check_message(missing_tools, errors),
        details={
            "configured_tools": sorted(schemas),
            "missing_tools": missing_tools,
            "errors": errors,
            "valid_count": valid_count,
            "evaluated_count": evaluated_count,
        },
    )


def evaluate_tool_argument_semantics(
    case: AgentEvalCase, trace: AgentRunTrace
) -> CheckResult:
    assertions_by_tool = case.expected.tool_argument_assertions
    calls_by_name = _calls_by_name(trace)
    required_tools = set(case.expected.tool_calls)
    valid_count = 0
    evaluated_count = 0
    missing_tools: list[str] = []
    errors: list[dict[str, object]] = []

    for tool_name, assertions in assertions_by_tool.items():
        if not assertions:
            continue
        calls = calls_by_name.get(tool_name, [])
        if not calls:
            if tool_name in required_tools:
                evaluated_count += 1
                missing_tools.append(tool_name)
            continue

        for call_index, call in enumerate(calls):
            evaluated_count += 1
            document = {"arguments": call.arguments}
            failed_assertions = [
                assertion
                for assertion in assertions
                if not evaluate_assertion(assertion, document)
            ]
            if not failed_assertions:
                valid_count += 1
                continue
            errors.append(
                {
                    "tool": tool_name,
                    "call_index": call_index,
                    "assertions": [
                        assertion.model_dump(mode="json")
                        for assertion in failed_assertions
                    ],
                }
            )

    configured = any(assertions_by_tool.values())
    required = bool(evaluated_count or configured)
    passed = not missing_tools and not errors
    failure_type = None
    if missing_tools:
        failure_type = REQUIRED_TOOL_NOT_CALLED
    elif errors:
        failure_type = TOOL_ARGUMENT_SEMANTIC_MISMATCH

    return CheckResult(
        schema_version=case.schema_version,
        check_id=TOOL_SEMANTIC_CHECK_ID,
        passed=passed,
        required=required,
        failure_type=failure_type,
        score=(valid_count / evaluated_count) if evaluated_count else None,
        message=_tool_check_message(missing_tools, errors),
        details={
            "configured_tools": sorted(assertions_by_tool),
            "missing_tools": missing_tools,
            "errors": errors,
            "valid_count": valid_count,
            "evaluated_count": evaluated_count,
        },
    )


def _calls_by_name(trace: AgentRunTrace) -> dict[str, list[ToolCallTrace]]:
    calls: dict[str, list[ToolCallTrace]] = defaultdict(list)
    for call in trace.tool_calls:
        calls[call.name].append(call)
    return calls


def _tool_check_message(
    missing_tools: list[str], errors: list[dict[str, object]]
) -> str | None:
    if missing_tools:
        return f"Required tool(s) not called: {', '.join(missing_tools)}"
    if errors:
        return f"Tool argument validation failed for {len(errors)} call(s)"
    return None
