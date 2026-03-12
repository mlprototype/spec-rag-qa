from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from ragqa.config import cfg
from ragqa.hybrid_retriever import HybridRetriever
from ragqa.retrieval_metrics import compute_retrieval_metrics

GROUND_TRUTH_PATH = Path(
    os.getenv("RETRIEVAL_GROUND_TRUTH_PATH", "data/eval/ground_truth_phase0_expanded.json")
)
BASELINE_PATH = Path(
    os.getenv("RETRIEVAL_BASELINE_PATH", "data/eval/phase0_vector_baseline_expanded.json")
)
REPORT_PATH = Path(
    os.getenv("RETRIEVAL_REPORT_PATH", "data/eval/phase4_hybrid_retrieval_report.json")
)


def _parse_expected_doc_id(ref: Any) -> str | None:
    if isinstance(ref, dict):
        doc_id = ref.get("doc_id")
        return str(doc_id) if doc_id else None
    if isinstance(ref, str):
        doc_id, _, _ = ref.partition("#")
        return doc_id or None
    return None


def _expected_sources_from_case(case: dict) -> list[Any]:
    expected = case.get("expected_sources")
    if expected:
        return list(expected)
    legacy = case.get("sources")
    if legacy:
        return list(legacy)
    return []


def _load_thresholds() -> tuple[float, float, float, str]:
    if BASELINE_PATH.exists():
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        summary = baseline.get("summary", {})
        try:
            base_r5 = float(summary["recall_at_5"])
            base_mrr = float(summary["mrr"])
            base_fr = float(summary["failure_rate"])
        except Exception as exc:
            raise RuntimeError(
                f"Invalid baseline schema at {BASELINE_PATH}. Expected summary.recall_at_5/mrr/failure_rate."
            ) from exc

        min_r5 = base_r5 * float(os.getenv("RETRIEVAL_RECALL5_MIN_RATIO", "0.90"))
        min_mrr = base_mrr * float(os.getenv("RETRIEVAL_MRR_MIN_RATIO", "0.90"))
        max_fr = base_fr * float(os.getenv("RETRIEVAL_FAILURE_MAX_RATIO", "1.20"))
        mode = f"baseline-relative ({BASELINE_PATH})"
    else:
        min_r5 = float(os.getenv("RETRIEVAL_MIN_RECALL_AT_5", "0.60"))
        min_mrr = float(os.getenv("RETRIEVAL_MIN_MRR", "0.35"))
        max_fr = float(os.getenv("RETRIEVAL_MAX_FAILURE_RATE", "0.40"))
        mode = "absolute fallback"

    return min_r5, min_mrr, max_fr, mode


def _validate_corpus_alignment(cases: list[dict], retriever: HybridRetriever) -> None:
    expected_doc_ids: set[str] = set()
    for case in cases:
        for src in _expected_sources_from_case(case):
            doc_id = _parse_expected_doc_id(src)
            if doc_id:
                expected_doc_ids.add(doc_id)

    indexed_doc_ids = {
        str(m["doc_id"]) for m in retriever.vs.meta if isinstance(m, dict) and "doc_id" in m
    }
    missing = sorted(expected_doc_ids - indexed_doc_ids)
    if missing:
        joined = ", ".join(missing[:6])
        if len(missing) > 6:
            joined += ", ..."
        print("ERROR: Ground truth and indexed corpus are misaligned.")
        print(f"Missing expected doc_ids in index ({len(missing)}): {joined}")
        print(
            "Hint: build index with matching docs, e.g. "
            "`RAGQA_DOCS_DIR=data/phase0_expanded/docs python -m ragqa.ingest`."
        )
        sys.exit(1)


def _print_summary(metrics: dict, evaluable: int, total: int) -> None:
    print()
    print("=" * 56)
    print(f"  Recall@1       {metrics['recall_at_1']:10.4f}")
    print(f"  Recall@5       {metrics['recall_at_5']:10.4f}")
    print(f"  MRR            {metrics['mrr']:10.4f}")
    print(f"  Failure Rate   {metrics['failure_rate']:10.4f}")
    print(f"  P50 Latency    {metrics['p50_latency_ms']:10.4f}")
    print(f"  P95 Latency    {metrics['p95_latency_ms']:10.4f}")
    print(f"  Mean Latency   {metrics['mean_latency_ms']:10.4f}")
    print(f"  Evaluable      {evaluable} / {total}")
    print("=" * 56)


def main() -> None:
    for path in [cfg.faiss_index_path, cfg.meta_path, cfg.bm25_index_path, cfg.bm25_postings_path]:
        if not path.exists():
            print(f"ERROR: Index not found: {path}. Run `python -m ragqa.ingest` first.")
            sys.exit(1)

    if not GROUND_TRUTH_PATH.exists():
        print(f"ERROR: ground truth not found: {GROUND_TRUTH_PATH}")
        sys.exit(1)

    cases = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    min_r5, min_mrr, max_fr, mode = _load_thresholds()

    print(f"SLO mode: {mode}")
    print(
        f"Thresholds: Recall@5>={min_r5:.4f}  MRR>={min_mrr:.4f}  FailureRate<={max_fr:.4f}"
    )

    retriever = HybridRetriever.load()
    _validate_corpus_alignment(cases, retriever)

    details: list[dict] = []
    for case in cases:
        start = time.perf_counter()
        hits = retriever.retrieve(case["question"])
        latency_ms = (time.perf_counter() - start) * 1000.0
        details.append(
            {
                "id": case["id"],
                "citations": [f"{h['doc_id']}#{h['chunk_id']}" for h in hits],
                "latency_ms": round(latency_ms, 3),
            }
        )

    metrics = compute_retrieval_metrics(cases, details)
    evaluable = int(metrics.get("evaluable_cases", 0))
    total = int(metrics.get("total_cases", len(cases)))

    if evaluable == 0:
        print("ERROR: evaluable_cases == 0. Check ground_truth expected_sources.")
        sys.exit(1)

    _print_summary(metrics, evaluable, total)

    report = {
        "phase": "phase4_retrieval_gate",
        "ground_truth_path": str(GROUND_TRUTH_PATH),
        "baseline_path": str(BASELINE_PATH),
        "docs_dir": str(cfg.docs_dir),
        "summary": metrics,
        "slo": {
            "mode": mode,
            "min_recall_at_5": min_r5,
            "min_mrr": min_mrr,
            "max_failure_rate": max_fr,
            "recall_at_5_passed": metrics["recall_at_5"] >= min_r5,
            "mrr_passed": metrics["mrr"] >= min_mrr,
            "failure_rate_passed": metrics["failure_rate"] <= max_fr,
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report saved: {REPORT_PATH}")

    failures: list[str] = []
    if not report["slo"]["recall_at_5_passed"]:
        failures.append(f"Recall@5 {metrics['recall_at_5']:.4f} < {min_r5:.4f}")
    if not report["slo"]["mrr_passed"]:
        failures.append(f"MRR {metrics['mrr']:.4f} < {min_mrr:.4f}")
    if not report["slo"]["failure_rate_passed"]:
        failures.append(f"FailureRate {metrics['failure_rate']:.4f} > {max_fr:.4f}")

    if failures:
        print("SLO FAILED:")
        for msg in failures:
            print(f"  - {msg}")
        sys.exit(1)

    print("SLO PASSED.")


if __name__ == "__main__":
    main()
