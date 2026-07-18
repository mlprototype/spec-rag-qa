from __future__ import annotations

import json

from jsonschema import SchemaError
from jsonschema.validators import validator_for

from ragqa.agent_eval.failure_types import ANSWER_FORMAT_INVALID
from ragqa.agent_eval.models import AgentEvalCase, AgentRunTrace, CheckResult


CHECK_ID = "answer_format"


def evaluate_answer_format(
    case: AgentEvalCase, trace: AgentRunTrace
) -> CheckResult:
    expectation = case.expected.answer_format
    configured = expectation is not None and bool(
        expectation.json_schema is not None or expectation.required_sections
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

    if expectation.json_schema is not None:
        try:
            parsed_answer = json.loads(answer)
        except json.JSONDecodeError as exc:
            errors.append(f"Answer is not valid JSON: {exc.msg}")
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

    missing_sections = [
        section
        for section in expectation.required_sections
        if section not in answer
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
