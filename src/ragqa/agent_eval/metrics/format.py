from __future__ import annotations

import json
import re
from typing import Any

from jsonschema import SchemaError
from jsonschema.validators import validator_for

from ragqa.agent_eval.assertions import resolve_json_path
from ragqa.agent_eval.failure_types import ANSWER_FORMAT_INVALID
from ragqa.agent_eval.models import AgentEvalCase, AgentRunTrace, CheckResult


CHECK_ID = "answer_format"


def evaluate_answer_format(
    case: AgentEvalCase, trace: AgentRunTrace
) -> CheckResult:
    expectation = case.expected.answer_format
    configured = expectation is not None and bool(
        expectation.format_type is not None
        or expectation.json_schema is not None
        or expectation.required_sections
    )
    if not configured or expectation is None:
        return CheckResult(
            schema_version=case.schema_version,
            check_id=CHECK_ID,
            passed=True,
            required=False,
            score=None,
            details={"valid_count": 0, "evaluated_count": 0},
        )

    errors: list[str] = []
    answer = trace.output.answer
    parsed_answer: Any = None
    answer_is_json = False

    if (
        expectation.format_type in {"json", "natural_language"}
        or expectation.json_schema is not None
        or expectation.required_sections
    ):
        try:
            parsed_answer = json.loads(answer)
            answer_is_json = True
        except json.JSONDecodeError:
            pass

    if expectation.format_type == "json" and not answer_is_json:
        errors.append("Answer is not valid JSON")
    elif expectation.format_type == "natural_language":
        if not answer.strip():
            errors.append("Answer is empty")
        elif answer_is_json:
            errors.append("Answer must be natural language, not JSON")

    if expectation.json_schema is not None:
        if not answer_is_json:
            if "Answer is not valid JSON" not in errors:
                errors.append("Answer is not valid JSON")
        else:
            validator_class = validator_for(expectation.json_schema)
            try:
                validator_class.check_schema(expectation.json_schema)
            except SchemaError as exc:
                raise ValueError("Invalid answer JSON Schema") from exc
            validation_errors = sorted(
                validator_class(expectation.json_schema).iter_errors(parsed_answer),
                key=lambda error: tuple(
                    str(part) for part in error.absolute_path
                ),
            )
            errors.extend(error.message for error in validation_errors)

    if answer_is_json:
        missing_sections = [
            section
            for section in expectation.required_sections
            if not _json_section_exists(parsed_answer, section)
        ]
    else:
        missing_sections = [
            section
            for section in expectation.required_sections
            if not _markdown_heading_exists(answer, section)
        ]
    if missing_sections:
        errors.append(f"Missing required sections: {', '.join(missing_sections)}")

    passed = not errors
    return CheckResult(
        schema_version=case.schema_version,
        check_id=CHECK_ID,
        passed=passed,
        failure_type=None if passed else ANSWER_FORMAT_INVALID,
        score=1.0 if passed else 0.0,
        message=None if passed else errors[0],
        details={
            "errors": errors,
            "missing_sections": missing_sections,
            "valid_count": int(passed),
            "evaluated_count": 1,
        },
    )


def _markdown_heading_exists(answer: str, section: str) -> bool:
    pattern = re.compile(
        rf"(?m)^#{{1,6}}[ \t]+{re.escape(section)}[ \t]*$"
    )
    return pattern.search(answer) is not None


def _json_section_exists(document: Any, section: str) -> bool:
    try:
        found, _ = resolve_json_path(document, section)
    except ValueError:
        return False
    return found
