from typing import List

from evaluator.fail_detector import detect_fail_type
from schemas.evaluation import EvaluationResult


def evaluate_answer(
    answer: str,
    retrieved_chunks: List[str],
) -> EvaluationResult:
    fail_type = detect_fail_type(answer, retrieved_chunks)

    if fail_type is None:
        return EvaluationResult(verdict="PASS")

    return EvaluationResult(
        verdict="FAIL",
        fail_type=fail_type,
        reasons=[f"Detected failure type: {fail_type}"],
    )
