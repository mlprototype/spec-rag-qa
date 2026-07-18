from ragqa.agent_eval.failure_types import CITATION_INVALID, CITATION_MISSING
from ragqa.agent_eval.models import AgentEvalCase, AgentRunTrace, CheckResult


PRESENCE_CHECK_ID = "citation_presence"
VALIDITY_CHECK_ID = "citation_validity"


def evaluate_citation_presence(
    case: AgentEvalCase, trace: AgentRunTrace
) -> CheckResult:
    emitted = [
        citation.citation_id
        for citation in trace.citations
        if citation.citation_id in trace.output.answer
    ]
    required = case.expected.citation_required
    passed = not required or bool(emitted)

    return CheckResult(
        schema_version=case.schema_version,
        check_id=PRESENCE_CHECK_ID,
        passed=passed,
        required=required,
        failure_type=None if passed else CITATION_MISSING,
        score=(1.0 if emitted else 0.0) if required else None,
        message=None if passed else "The answer contains no recorded citation",
        details={
            "citation_ids": emitted,
            "present_count": int(bool(emitted)),
            "required_count": int(required),
        },
    )


def evaluate_citation_validity(
    case: AgentEvalCase, trace: AgentRunTrace
) -> CheckResult:
    source_ids = {source.source_id for source in trace.sources}
    invalid: list[dict[str, str]] = []

    for citation in trace.citations:
        if citation.citation_id not in trace.output.answer:
            invalid.append(
                {
                    "citation_id": citation.citation_id,
                    "source_id": citation.source_id,
                    "reason": "citation_id_not_in_answer",
                }
            )
        elif citation.source_id not in source_ids:
            invalid.append(
                {
                    "citation_id": citation.citation_id,
                    "source_id": citation.source_id,
                    "reason": "source_not_found",
                }
            )

    citation_count = len(trace.citations)
    valid_count = citation_count - len(invalid)
    required = citation_count > 0
    passed = not invalid

    return CheckResult(
        schema_version=case.schema_version,
        check_id=VALIDITY_CHECK_ID,
        passed=passed,
        required=required,
        failure_type=None if passed else CITATION_INVALID,
        score=(valid_count / citation_count) if citation_count else None,
        message=None if passed else f"Invalid citation(s): {len(invalid)}",
        details={
            "invalid_citations": invalid,
            "valid_count": valid_count,
            "citation_count": citation_count,
        },
    )
