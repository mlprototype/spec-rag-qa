from typing import List, Optional


def detect_fail_type(
    answer: str,
    retrieved_chunks: List[str],
) -> Optional[str]:
    if not retrieved_chunks:
        return "NO_RETRIEVAL"

    if not answer or len(answer.strip()) < 10:
        return "EMPTY_ANSWER"

    for chunk in retrieved_chunks:
        if answer.strip() in chunk:
            return None

    return "HALLUCINATION"
