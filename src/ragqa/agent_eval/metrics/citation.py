import re

from ragqa.agent_eval.failure_types import CITATION_INVALID, CITATION_MISSING
from ragqa.agent_eval.models import (
    AgentEvalCase,
    AgentRunTrace,
    CheckResult,
    CitationTrace,
)


PRESENCE_CHECK_ID = "citation_presence"
VALIDITY_CHECK_ID = "citation_validity"
CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_.:-]+)\]")


def extract_answer_citation_ids(answer: str) -> set[str]:
    """Extract exact bracketed citation IDs from an answer."""

    return set(CITATION_PATTERN.findall(answer))


def evaluate_citation_presence(
    case: AgentEvalCase, trace: AgentRunTrace
) -> CheckResult:
    answer_citation_ids = extract_answer_citation_ids(trace.output.answer)
    required = case.expected.citation_required
    passed = not required or bool(answer_citation_ids)

    return CheckResult(
        schema_version=case.schema_version,
        check_id=PRESENCE_CHECK_ID,
        passed=passed,
        required=required,
        failure_type=None if passed else CITATION_MISSING,
        score=(1.0 if answer_citation_ids else 0.0) if required else None,
        message=None if passed else "The answer contains no citation marker",
        details={
            "citation_ids": sorted(answer_citation_ids),
            "present_count": int(bool(answer_citation_ids)),
            "required_count": int(required),
        },
    )


def evaluate_citation_validity(
    case: AgentEvalCase, trace: AgentRunTrace
) -> CheckResult:
    answer_citation_ids = extract_answer_citation_ids(trace.output.answer)
    recorded_citations: dict[str, CitationTrace] = {}
    duplicate_recorded_ids: set[str] = set()
    for citation in trace.citations:
        if citation.citation_id in recorded_citations:
            duplicate_recorded_ids.add(citation.citation_id)
        recorded_citations[citation.citation_id] = citation

    source_ids = {source.source_id for source in trace.sources}
    evaluated_ids = answer_citation_ids | set(recorded_citations)
    invalid: list[dict[str, object]] = []
    valid_count = 0

    for citation_id in sorted(evaluated_ids):
        citation = recorded_citations.get(citation_id)
        if citation is None:
            invalid.append(
                {
                    "citation_id": citation_id,
                    "source_id": None,
                    "reason": "citation_not_recorded",
                }
            )
        elif citation_id not in answer_citation_ids:
            invalid.append(
                {
                    "citation_id": citation_id,
                    "source_id": citation.source_id,
                    "reason": "recorded_citation_not_in_answer",
                }
            )
        elif citation_id in duplicate_recorded_ids:
            invalid.append(
                {
                    "citation_id": citation_id,
                    "source_id": citation.source_id,
                    "reason": "duplicate_recorded_citation_id",
                }
            )
        elif citation.source_id not in source_ids:
            invalid.append(
                {
                    "citation_id": citation_id,
                    "source_id": citation.source_id,
                    "reason": "source_not_found",
                }
            )
        else:
            valid_count += 1

    citation_count = len(evaluated_ids)
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
            "answer_citation_ids": sorted(answer_citation_ids),
            "recorded_citation_ids": sorted(recorded_citations),
            "invalid_citations": invalid,
            "valid_count": valid_count,
            "citation_count": citation_count,
        },
    )
