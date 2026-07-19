from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from jsonschema import SchemaError
from jsonschema.validators import validator_for

from ragqa.agent_eval.models import AgentEvalCase, AgentRunTrace


class DatasetValidationError(ValueError):
    """Raised when cases and saved traces do not form a coherent dataset."""


def validate_case_contracts(cases: Sequence[AgentEvalCase]) -> None:
    """Validate cross-field tool and JSON Schema references before execution."""

    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise DatasetValidationError(f"Duplicate case id: {case.id}")
        seen.add(case.id)

        required_tools = set(case.expected.tool_calls)
        forbidden_tools = set(case.expected.forbidden_tool_calls)
        overlap = sorted(required_tools & forbidden_tools)
        if overlap:
            raise DatasetValidationError(
                f"Case {case.id} requires and forbids the same tool(s): {overlap}"
            )

        schema_tools = set(case.expected.tool_argument_schemas)
        assertion_tools = set(case.expected.tool_argument_assertions)
        dangling_tools = sorted((schema_tools | assertion_tools) - required_tools)
        if dangling_tools:
            raise DatasetValidationError(
                f"Case {case.id} has schema/assertion for unknown tool(s): "
                f"{dangling_tools}"
            )

        for tool_name, schema in case.expected.tool_argument_schemas.items():
            validator_class = validator_for(schema)
            try:
                validator_class.check_schema(schema)
            except SchemaError as exc:
                raise DatasetValidationError(
                    f"Case {case.id} has invalid schema for tool {tool_name}"
                ) from exc
            _validate_schema_refs(schema, schema, case.id, tool_name)

        answer_format = case.expected.answer_format
        if answer_format is not None and answer_format.json_schema is not None:
            if answer_format.format_type == "natural_language":
                raise DatasetValidationError(
                    f"Case {case.id} combines natural-language format with JSON Schema"
                )
            schema = answer_format.json_schema
            validator_class = validator_for(schema)
            try:
                validator_class.check_schema(schema)
            except SchemaError as exc:
                raise DatasetValidationError(
                    f"Case {case.id} has invalid answer format schema"
                ) from exc
            _validate_schema_refs(schema, schema, case.id, "answer_format")


def validate_trace_coverage(
    cases: Sequence[AgentEvalCase], traces: Iterable[AgentRunTrace]
) -> None:
    """Require exactly one compatible saved trace for every selected case."""

    traces_by_case_id: dict[str, AgentRunTrace] = {}
    for trace in traces:
        if trace.case_id in traces_by_case_id:
            raise DatasetValidationError(
                f"Duplicate trace for case id: {trace.case_id}"
            )
        traces_by_case_id[trace.case_id] = trace

    case_ids = {case.id for case in cases}
    trace_ids = set(traces_by_case_id)
    missing = sorted(case_ids - trace_ids)
    if missing:
        raise DatasetValidationError(f"Fixture trace missing for case id(s): {missing}")
    unknown = sorted(trace_ids - case_ids)
    if unknown:
        raise DatasetValidationError(
            f"Fixture contains unknown case id(s): {unknown}"
        )

    for case in cases:
        trace = traces_by_case_id[case.id]
        if trace.schema_version != case.schema_version:
            raise DatasetValidationError(
                f"Schema version mismatch for case id: {case.id}"
            )
        if trace.input.question != case.input.question:
            raise DatasetValidationError(
                f"Fixture input mismatch for case id: {case.id}"
            )


def validate_dataset(
    cases: Sequence[AgentEvalCase], traces: Iterable[AgentRunTrace]
) -> None:
    trace_list = list(traces)
    validate_case_contracts(cases)
    validate_trace_coverage(cases, trace_list)


def _validate_schema_refs(
    node: Any,
    root: Mapping[str, Any],
    case_id: str,
    schema_name: str,
) -> None:
    if isinstance(node, Mapping):
        ref = node.get("$ref")
        if isinstance(ref, str):
            if not ref.startswith("#/") or not _json_pointer_exists(root, ref[2:]):
                raise DatasetValidationError(
                    f"Case {case_id} has unresolved $ref {ref!r} in {schema_name}"
                )
        for value in node.values():
            _validate_schema_refs(value, root, case_id, schema_name)
    elif isinstance(node, list):
        for value in node:
            _validate_schema_refs(value, root, case_id, schema_name)


def _json_pointer_exists(root: Mapping[str, Any], pointer: str) -> bool:
    current: Any = root
    for raw_part in pointer.split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    return True
