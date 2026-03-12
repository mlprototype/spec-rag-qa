from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

from ragqa.retrieval_metrics import compute_retrieval_metrics

DEFAULT_GROUND_TRUTH_PATH = Path(
    os.getenv("RETRIEVAL_GROUND_TRUTH_PATH", "data/eval/ground_truth_phase0_expanded.json")
)
DEFAULT_BASELINE_PATH = Path(
    os.getenv("RETRIEVAL_BASELINE_PATH", "data/eval/phase0_vector_baseline_expanded.json")
)
DEFAULT_REPORT_PATH = Path(
    os.getenv("RETRIEVAL_REPORT_PATH", "data/eval/phase5_grid_search_report.json")
)
DEFAULT_BEST_PATH = Path(os.getenv("RETRIEVAL_BEST_PATH", "data/eval/phase5_best_config.json"))
DEFAULT_REPORT_MD_PATH = Path(
    os.getenv("RETRIEVAL_REPORT_MD_PATH", "data/eval/phase5_grid_search_report.md")
)

STAGE1_FIXED_BOOST_ALPHA = 1.5
STAGE1_FIXED_BOOST_BETA = 2.0
STAGE1_VECTOR_CANDIDATE_K = [10, 15, 20]
STAGE1_BM25_CANDIDATE_K = [10, 15, 20]
STAGE1_RRF_K = [30, 60, 90]
STAGE1_FINAL_TOP_K = [3, 5, 7]
STAGE2_BOOST_ALPHA = [1.2, 1.5, 1.8, 2.0]
STAGE2_BOOST_BETA = [1.5, 2.0, 2.5, 3.0]

EPSILON = 1e-6
MIN_GAIN = 0.01


def _expected_sources_from_case(case: dict) -> list[Any]:
    expected = case.get("expected_sources")
    if expected:
        return list(expected)
    legacy = case.get("sources")
    if legacy:
        return list(legacy)
    return []


def _parse_expected_doc_id(ref: Any) -> str | None:
    if isinstance(ref, dict):
        doc_id = ref.get("doc_id")
        return str(doc_id) if doc_id else None
    if isinstance(ref, str):
        doc_id, _, _ = ref.partition("#")
        return doc_id or None
    return None


def validate_corpus_alignment(cases: list[dict], meta: list[dict]) -> None:
    expected_doc_ids: set[str] = set()
    for case in cases:
        for src in _expected_sources_from_case(case):
            doc_id = _parse_expected_doc_id(src)
            if doc_id:
                expected_doc_ids.add(doc_id)
    indexed_doc_ids = {
        str(m["doc_id"]) for m in meta if isinstance(m, dict) and "doc_id" in m
    }
    missing = sorted(expected_doc_ids - indexed_doc_ids)
    if missing:
        joined = ", ".join(missing[:6])
        if len(missing) > 6:
            joined += ", ..."
        raise RuntimeError(
            "Ground truth and index are misaligned. "
            f"Missing doc_ids in index ({len(missing)}): {joined}"
        )


def build_stage1_params() -> list[dict]:
    params: list[dict] = []
    for vector_k, bm25_k, rrf_k, final_k in product(
        STAGE1_VECTOR_CANDIDATE_K,
        STAGE1_BM25_CANDIDATE_K,
        STAGE1_RRF_K,
        STAGE1_FINAL_TOP_K,
    ):
        params.append(
            {
                "vector_candidate_k": vector_k,
                "bm25_candidate_k": bm25_k,
                "rrf_k": rrf_k,
                "final_top_k": final_k,
                "boost_alpha": STAGE1_FIXED_BOOST_ALPHA,
                "boost_beta": STAGE1_FIXED_BOOST_BETA,
            }
        )
    return params


def build_stage2_params(stage1_best: dict) -> list[dict]:
    params: list[dict] = []
    for alpha, beta in product(STAGE2_BOOST_ALPHA, STAGE2_BOOST_BETA):
        params.append(
            {
                "vector_candidate_k": int(stage1_best["vector_candidate_k"]),
                "bm25_candidate_k": int(stage1_best["bm25_candidate_k"]),
                "rrf_k": int(stage1_best["rrf_k"]),
                "final_top_k": int(stage1_best["final_top_k"]),
                "boost_alpha": float(alpha),
                "boost_beta": float(beta),
            }
        )
    return params


def build_slo_thresholds(baseline_summary: dict) -> dict:
    ratio_r5 = float(os.getenv("RETRIEVAL_RECALL5_MIN_RATIO", "0.90"))
    ratio_mrr = float(os.getenv("RETRIEVAL_MRR_MIN_RATIO", "0.90"))
    ratio_fr = float(os.getenv("RETRIEVAL_FAILURE_MAX_RATIO", "1.20"))
    return {
        "min_recall_at_5": float(baseline_summary["recall_at_5"]) * ratio_r5,
        "min_mrr": float(baseline_summary["mrr"]) * ratio_mrr,
        "max_failure_rate": float(baseline_summary["failure_rate"]) * ratio_fr,
    }


def is_eligible(metrics: dict, thresholds: dict) -> bool:
    return (
        float(metrics["recall_at_5"]) >= float(thresholds["min_recall_at_5"])
        and float(metrics["mrr"]) >= float(thresholds["min_mrr"])
        and float(metrics["failure_rate"]) <= float(thresholds["max_failure_rate"])
    )


def ranking_key(trial: dict) -> tuple:
    metrics = trial["metrics"]
    return (
        -float(metrics["recall_at_5"]),
        -float(metrics["mrr"]),
        float(metrics["failure_rate"]),
        float(metrics["p95_latency_ms"]),
        -float(metrics["recall_at_1"]),
        int(trial["trial_id"]),
    )


def build_improvement(final_metrics: dict | None, baseline_summary: dict) -> dict:
    if final_metrics is None:
        final_metrics = {
            "recall_at_5": 0.0,
            "mrr": 0.0,
        }
    baseline_r5 = float(baseline_summary["recall_at_5"])
    baseline_mrr = float(baseline_summary["mrr"])
    final_r5 = float(final_metrics.get("recall_at_5", 0.0))
    final_mrr = float(final_metrics.get("mrr", 0.0))

    return {
        "epsilon": EPSILON,
        "min_gain": MIN_GAIN,
        "baseline_recall_at_5": baseline_r5,
        "baseline_mrr": baseline_mrr,
        "final_recall_at_5": final_r5,
        "final_mrr": final_mrr,
        "improved_r5": final_r5 > baseline_r5 + EPSILON,
        "meaningful_improved_r5": final_r5 >= baseline_r5 + MIN_GAIN,
        "improved_mrr": final_mrr > baseline_mrr + EPSILON,
        "meaningful_improved_mrr": final_mrr >= baseline_mrr + MIN_GAIN,
    }


def evaluate_trial(retriever: Any, cases: list[dict], params: dict) -> dict:
    details: list[dict] = []
    for case in cases:
        case_start = time.perf_counter()
        hits = retriever.retrieve(
            case["question"],
            final_top_k=int(params["final_top_k"]),
            vector_candidate_k=int(params["vector_candidate_k"]),
            bm25_candidate_k=int(params["bm25_candidate_k"]),
            rrf_k=int(params["rrf_k"]),
            boost_alpha=float(params["boost_alpha"]),
            boost_beta=float(params["boost_beta"]),
        )
        latency_ms = (time.perf_counter() - case_start) * 1000.0
        details.append(
            {
                "id": case["id"],
                "citations": [f"{h['doc_id']}#{h['chunk_id']}" for h in hits],
                "latency_ms": round(latency_ms, 3),
            }
        )
    return compute_retrieval_metrics(cases, details)


def _run_trials(
    retriever: Any,
    cases: list[dict],
    thresholds: dict,
    params_list: list[dict],
    stage_name: str,
    starting_trial_id: int,
    budget: int | None,
) -> tuple[list[dict], int]:
    trials: list[dict] = []
    trial_id = starting_trial_id
    for params in params_list:
        if budget is not None and budget <= 0:
            break
        metrics = evaluate_trial(retriever, cases, params)
        eligible = is_eligible(metrics, thresholds)
        trials.append(
            {
                "trial_id": trial_id,
                "stage": stage_name,
                "params": params,
                "metrics": metrics,
                "eligible": eligible,
            }
        )
        trial_id += 1
        if budget is not None:
            budget -= 1
    return trials, trial_id


def select_best_eligible(trials: list[dict]) -> dict | None:
    eligible_trials = [t for t in trials if t["eligible"]]
    if not eligible_trials:
        return None
    eligible_trials.sort(key=ranking_key)
    return eligible_trials[0]


def render_markdown_report(report: dict, topn: int) -> str:
    lines: list[str] = []
    lines.append("# Phase 5 Grid Search Report")
    lines.append("")
    lines.append(f"- Status: `{report['status']}`")
    lines.append(f"- Ground Truth: `{report['ground_truth_path']}`")
    lines.append(f"- Docs Dir: `{report['docs_dir']}`")
    lines.append(f"- Total Trials: `{report['eligibility']['total_trials']}`")
    lines.append(f"- Eligible Trials: `{report['eligibility']['eligible_trials']}`")
    lines.append("")
    if report["best"] is None:
        lines.append("No eligible trial found under SLO constraints.")
        return "\n".join(lines)

    best = report["best"]
    lines.append("## Best Params")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(best["params"], ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Best Metrics")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(best["metrics"], ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append(f"## Top {topn} Eligible Trials")
    lines.append("")
    lines.append("| trial_id | stage | recall@5 | mrr | failure_rate | p95_ms |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for trial in report["topn"]:
        metrics = trial["metrics"]
        lines.append(
            f"| {trial['trial_id']} | {trial['stage']} | "
            f"{metrics['recall_at_5']:.4f} | {metrics['mrr']:.4f} | "
            f"{metrics['failure_rate']:.4f} | {metrics['p95_latency_ms']:.3f} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5 grid search for hybrid retrieval")
    parser.add_argument("--max-trials", type=int, default=None, help="Max number of trials")
    parser.add_argument("--topn", type=int, default=10, help="Top N results to keep")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_REPORT_MD_PATH)
    parser.add_argument("--output-best", type=Path, default=DEFAULT_BEST_PATH)
    return parser.parse_args()


def _load_baseline_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"baseline not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"baseline.summary missing: {path}")
    for key in ("recall_at_5", "mrr", "failure_rate"):
        if key not in summary:
            raise ValueError(f"baseline.summary.{key} missing: {path}")
    return summary


def _get_budget(max_trials: int | None) -> int | None:
    if max_trials is None:
        return None
    if max_trials <= 0:
        raise ValueError("--max-trials must be > 0")
    return max_trials


def main() -> None:
    args = parse_args()
    budget = _get_budget(args.max_trials)
    topn = max(1, int(args.topn))

    # Import here to keep unit tests independent from optional native deps.
    from ragqa.config import cfg
    from ragqa.hybrid_retriever import HybridRetriever

    ground_truth_path = Path(
        os.getenv("RETRIEVAL_GROUND_TRUTH_PATH", str(DEFAULT_GROUND_TRUTH_PATH))
    )
    baseline_path = Path(os.getenv("RETRIEVAL_BASELINE_PATH", str(DEFAULT_BASELINE_PATH)))

    for path in [
        cfg.faiss_index_path,
        cfg.meta_path,
        cfg.bm25_index_path,
        cfg.bm25_postings_path,
        ground_truth_path,
    ]:
        if not path.exists():
            print(f"ERROR: required file not found: {path}")
            sys.exit(1)

    cases = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    baseline_summary = _load_baseline_summary(baseline_path)
    thresholds = build_slo_thresholds(baseline_summary)

    retriever = HybridRetriever.load()
    validate_corpus_alignment(cases, retriever.vs.meta)

    stage1_params = build_stage1_params()
    stage1_trials, next_trial_id = _run_trials(
        retriever=retriever,
        cases=cases,
        thresholds=thresholds,
        params_list=stage1_params,
        stage_name="stage1",
        starting_trial_id=1,
        budget=budget,
    )

    if budget is not None:
        budget -= len(stage1_trials)

    stage1_best = select_best_eligible([t for t in stage1_trials if t["stage"] == "stage1"])
    stage2_trials: list[dict] = []
    if stage1_best is not None and (budget is None or budget > 0):
        stage2_params = build_stage2_params(stage1_best["params"])
        stage2_trials, next_trial_id = _run_trials(
            retriever=retriever,
            cases=cases,
            thresholds=thresholds,
            params_list=stage2_params,
            stage_name="stage2",
            starting_trial_id=next_trial_id,
            budget=budget,
        )

    all_trials = stage1_trials + stage2_trials
    eligible_trials = [t for t in all_trials if t["eligible"]]
    eligible_trials.sort(key=ranking_key)
    best = eligible_trials[0] if eligible_trials else None
    top_trials = eligible_trials[:topn]

    final_metrics = best["metrics"] if best else None
    improvement = build_improvement(final_metrics, baseline_summary)
    status = "ok" if best is not None else "no_eligible"

    report = {
        "phase": "phase5_grid_search",
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "status": status,
        "ground_truth_path": str(ground_truth_path),
        "docs_dir": str(cfg.docs_dir),
        "baseline_path": str(baseline_path),
        "baseline": {
            "recall_at_5": float(baseline_summary["recall_at_5"]),
            "mrr": float(baseline_summary["mrr"]),
            "failure_rate": float(baseline_summary["failure_rate"]),
        },
        "search_space": {
            "stage1": {
                "vector_candidate_k": STAGE1_VECTOR_CANDIDATE_K,
                "bm25_candidate_k": STAGE1_BM25_CANDIDATE_K,
                "rrf_k": STAGE1_RRF_K,
                "final_top_k": STAGE1_FINAL_TOP_K,
                "boost_alpha_fixed": STAGE1_FIXED_BOOST_ALPHA,
                "boost_beta_fixed": STAGE1_FIXED_BOOST_BETA,
            },
            "stage2": {
                "boost_alpha": STAGE2_BOOST_ALPHA,
                "boost_beta": STAGE2_BOOST_BETA,
            },
        },
        "slo": thresholds,
        "eligibility": {
            "eligible_trials": len(eligible_trials),
            "total_trials": len(all_trials),
        },
        "improvement": improvement,
        "trials": all_trials,
        "best": best,
        "topn": top_trials,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )

    best_payload = (
        {
            "trial_id": best["trial_id"],
            "stage": best["stage"],
            "params": best["params"],
            "metrics": best["metrics"],
        }
        if best
        else {"status": "no_eligible", "best": None}
    )
    args.output_best.parent.mkdir(parents=True, exist_ok=True)
    args.output_best.write_text(
        json.dumps(best_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown_report(report, topn), encoding="utf-8")

    print(f"Report JSON: {args.output_json}")
    print(f"Best Config: {args.output_best}")
    print(f"Report MD:   {args.output_md}")
    print(f"Status: {status}")

    if best is None:
        print("No eligible trial under SLO constraints.")
        sys.exit(1)

    metrics = best["metrics"]
    print(
        "Best metrics: "
        f"recall@5={metrics['recall_at_5']:.4f}, "
        f"mrr={metrics['mrr']:.4f}, "
        f"failure_rate={metrics['failure_rate']:.4f}, "
        f"p95={metrics['p95_latency_ms']:.3f}ms"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
