from __future__ import annotations

import json
import importlib.util
from pathlib import Path


def _load_phase5_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "run_phase5_grid_search.py"
    spec = importlib.util.spec_from_file_location("phase5_grid_search", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gs00_default_baseline_is_available_for_ci_grid_search():
    root = Path(__file__).resolve().parents[1]
    baseline_path = root / "data" / "eval" / "phase0_vector_baseline_expanded.json"
    assert baseline_path.exists(), (
        "Phase 5 grid search requires the expanded Phase 0 baseline file in CI."
    )

    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    summary = payload.get("summary")
    assert isinstance(summary, dict)
    for key in ("recall_at_5", "mrr", "failure_rate"):
        assert key in summary


def test_gs01_stage1_param_count_and_fixed_boost():
    mod = _load_phase5_module()
    params = mod.build_stage1_params()
    assert len(params) == 81
    assert all(p["boost_alpha"] == 1.5 for p in params)
    assert all(p["boost_beta"] == 2.0 for p in params)


def test_gs02_stage2_param_count_and_inherited_fields():
    mod = _load_phase5_module()
    base = {
        "vector_candidate_k": 15,
        "bm25_candidate_k": 20,
        "rrf_k": 60,
        "final_top_k": 5,
        "boost_alpha": 1.5,
        "boost_beta": 2.0,
    }
    params = mod.build_stage2_params(base)
    assert len(params) == 16
    assert all(p["vector_candidate_k"] == 15 for p in params)
    assert all(p["bm25_candidate_k"] == 20 for p in params)
    assert all(p["rrf_k"] == 60 for p in params)
    assert all(p["final_top_k"] == 5 for p in params)


def test_gs03_ranking_tie_break_by_recall_at_1_then_trial_id():
    mod = _load_phase5_module()
    trials = [
        {
            "trial_id": 2,
            "metrics": {
                "recall_at_5": 0.8,
                "mrr": 0.5,
                "failure_rate": 0.2,
                "p95_latency_ms": 12.0,
                "recall_at_1": 0.2,
            },
        },
        {
            "trial_id": 1,
            "metrics": {
                "recall_at_5": 0.8,
                "mrr": 0.5,
                "failure_rate": 0.2,
                "p95_latency_ms": 12.0,
                "recall_at_1": 0.4,
            },
        },
    ]
    ranked = sorted(trials, key=mod.ranking_key)
    assert ranked[0]["trial_id"] == 1


def test_gs04_slo_eligibility_filter():
    mod = _load_phase5_module()
    thresholds = {
        "min_recall_at_5": 0.72,
        "min_mrr": 0.40,
        "max_failure_rate": 0.24,
    }
    ok_metrics = {"recall_at_5": 0.80, "mrr": 0.45, "failure_rate": 0.20}
    ng_metrics = {"recall_at_5": 0.90, "mrr": 0.41, "failure_rate": 0.30}
    assert mod.is_eligible(ok_metrics, thresholds) is True
    assert mod.is_eligible(ng_metrics, thresholds) is False


def test_gs05_best_selection_is_deterministic():
    mod = _load_phase5_module()
    trials = [
        {
            "trial_id": 2,
            "eligible": True,
            "metrics": {
                "recall_at_5": 0.8,
                "mrr": 0.5,
                "failure_rate": 0.2,
                "p95_latency_ms": 10.0,
                "recall_at_1": 0.3,
            },
        },
        {
            "trial_id": 1,
            "eligible": True,
            "metrics": {
                "recall_at_5": 0.8,
                "mrr": 0.5,
                "failure_rate": 0.2,
                "p95_latency_ms": 10.0,
                "recall_at_1": 0.3,
            },
        },
        {
            "trial_id": 3,
            "eligible": False,
            "metrics": {
                "recall_at_5": 0.95,
                "mrr": 0.9,
                "failure_rate": 0.4,
                "p95_latency_ms": 9.0,
                "recall_at_1": 0.8,
            },
        },
    ]
    best = mod.select_best_eligible(trials)
    assert best is not None
    assert best["trial_id"] == 1


def test_gs06_improvement_uses_epsilon_and_min_gain():
    mod = _load_phase5_module()
    baseline = {"recall_at_5": 0.8, "mrr": 0.4475, "failure_rate": 0.2}
    final_metrics = {"recall_at_5": 0.80001, "mrr": 0.44761}
    improvement = mod.build_improvement(final_metrics, baseline)
    assert improvement["improved_r5"] is True
    assert improvement["meaningful_improved_r5"] is False


def test_gs07_markdown_report_contains_best_section():
    mod = _load_phase5_module()
    report = {
        "status": "ok",
        "ground_truth_path": "x.json",
        "docs_dir": "data/phase0_expanded/docs",
        "eligibility": {"total_trials": 10, "eligible_trials": 2},
        "best": {
            "params": {"rrf_k": 60},
            "metrics": {"recall_at_5": 0.8, "mrr": 0.45, "failure_rate": 0.2, "p95_latency_ms": 5.0},
        },
        "topn": [
            {
                "trial_id": 1,
                "stage": "stage1",
                "metrics": {"recall_at_5": 0.8, "mrr": 0.45, "failure_rate": 0.2, "p95_latency_ms": 5.0},
            }
        ],
    }
    text = mod.render_markdown_report(report, topn=10)
    assert "## Best Params" in text
    assert "## Top 10 Eligible Trials" in text


def test_gs08_smoke_recall_positive():
    """
    GS-01: recall_at_5 > 0.0 を保証する検索スモーク。
    """
    mod = _load_phase5_module()

    class FakeRetriever:
        def retrieve(self, query: str, **_: object):
            if query == "target query":
                return [{"doc_id": "doc.md", "chunk_id": 0}]
            return []

    params = {
        "vector_candidate_k": 10,
        "bm25_candidate_k": 10,
        "rrf_k": 60,
        "final_top_k": 5,
        "boost_alpha": 1.5,
        "boost_beta": 2.0,
    }
    cases = [{"id": "q1", "question": "target query", "expected_sources": ["doc.md"]}]
    metrics = mod.evaluate_trial(FakeRetriever(), cases, params)
    assert metrics["evaluable_cases"] > 0
    assert metrics["recall_at_5"] > 0.0
