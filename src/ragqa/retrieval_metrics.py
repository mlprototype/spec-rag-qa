from __future__ import annotations

import math
from typing import Any


def parse_source_ref(ref: Any) -> tuple[str, int | None] | None:
    if isinstance(ref, dict):
        doc_id = ref.get("doc_id")
        if doc_id is None:
            return None
        chunk_id = ref.get("chunk_id")
        return (str(doc_id), int(chunk_id) if chunk_id is not None else None)

    if isinstance(ref, str):
        if "#" in ref:
            doc_id, _, chunk = ref.partition("#")
            if not doc_id:
                return None
            try:
                return (doc_id, int(chunk))
            except ValueError:
                return (doc_id, None)
        return (ref, None) if ref else None

    return None


def _expected_sources_from_case(case: dict) -> list[Any]:
    """
    Prefer explicit expected_sources. Fallback to legacy sources to keep
    compatibility with older ground truth files.
    """
    expected = case.get("expected_sources")
    if expected:
        return list(expected)
    legacy = case.get("sources")
    if legacy:
        return list(legacy)
    return []


def _is_hit(pred: tuple[str, int | None], exp: tuple[str, int | None]) -> bool:
    if exp[1] is not None:
        return pred[0] == exp[0] and pred[1] == exp[1]
    return pred[0] == exp[0]


def first_hit_rank(
    predicted: list[tuple[str, int | None]],
    expected: list[tuple[str, int | None]],
) -> int | None:
    for rank, pred in enumerate(predicted, start=1):
        if any(_is_hit(pred, exp) for exp in expected):
            return rank
    return None


def percentile(values: list[float], q: int) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if q <= 0:
        return float(sorted_values[0])
    if q >= 100:
        return float(sorted_values[-1])
    idx = math.ceil((q / 100.0) * len(sorted_values)) - 1
    return float(sorted_values[max(0, min(idx, len(sorted_values) - 1))])


def compute_retrieval_metrics(cases: list[dict], details: list[dict]) -> dict:
    details_by_id: dict[str, dict] = {
        str(d["id"]): d for d in details if isinstance(d, dict) and "id" in d
    }

    eval_items: list[dict] = []
    all_latencies: list[float] = []
    per_case: list[dict] = []

    for case in cases:
        case_id = str(case.get("id"))
        detail = details_by_id.get(case_id, {})
        latency_ms = float(detail.get("latency_ms", 0.0) or 0.0)
        all_latencies.append(latency_ms)

        expected_raw = _expected_sources_from_case(case)
        parsed_expected = [parse_source_ref(x) for x in expected_raw]
        expected = [x for x in parsed_expected if x is not None]

        citations_raw = detail.get("citations", []) or []
        parsed_predicted = [parse_source_ref(x) for x in citations_raw]
        predicted = [x for x in parsed_predicted if x is not None]

        if not expected:
            continue

        rank = first_hit_rank(predicted, expected)
        hit_at_1 = rank is not None and rank <= 1
        hit_at_5 = rank is not None and rank <= 5
        reciprocal_rank = (1.0 / rank) if rank is not None else 0.0

        eval_items.append(
            {
                "hit_at_1": hit_at_1,
                "hit_at_5": hit_at_5,
                "rr": reciprocal_rank,
            }
        )
        per_case.append(
            {
                "id": case_id,
                "rank": rank,
                "hit_at_1": hit_at_1,
                "hit_at_5": hit_at_5,
                "reciprocal_rank": round(reciprocal_rank, 6),
                "expected_sources": expected_raw,
                "citations": citations_raw,
            }
        )

    if not eval_items:
        mean_latency = (
            round(sum(all_latencies) / len(all_latencies), 3) if all_latencies else 0.0
        )
        return {
            "recall_at_1": 0.0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
            "failure_rate": 1.0,
            "p50_latency_ms": percentile(all_latencies, 50),
            "p95_latency_ms": percentile(all_latencies, 95),
            "mean_latency_ms": mean_latency,
            "evaluable_cases": 0,
            "total_cases": len(cases),
            "per_case": per_case,
        }

    n = len(eval_items)
    recall_at_1 = sum(1.0 for x in eval_items if x["hit_at_1"]) / n
    recall_at_5 = sum(1.0 for x in eval_items if x["hit_at_5"]) / n
    mrr = sum(x["rr"] for x in eval_items) / n
    mean_latency = (
        round(sum(all_latencies) / len(all_latencies), 3) if all_latencies else 0.0
    )

    return {
        "recall_at_1": round(recall_at_1, 4),
        "recall_at_5": round(recall_at_5, 4),
        "mrr": round(mrr, 4),
        "failure_rate": round(1.0 - recall_at_5, 4),
        "p50_latency_ms": percentile(all_latencies, 50),
        "p95_latency_ms": percentile(all_latencies, 95),
        "mean_latency_ms": mean_latency,
        "evaluable_cases": n,
        "total_cases": len(cases),
        "per_case": per_case,
    }
