from ragqa.retrieval_metrics import (
    compute_retrieval_metrics,
    first_hit_rank,
    percentile,
)


def test_rm01_doc_level_hit():
    predicted = [("doc.md", 0), ("other.md", 1)]
    expected = [("doc.md", None)]
    assert first_hit_rank(predicted, expected) == 1


def test_rm02_chunk_level_strict_hit():
    predicted = [("doc.md", 0), ("doc.md", 3)]
    expected = [("doc.md", 3)]
    assert first_hit_rank(predicted, expected) == 2
    assert first_hit_rank(predicted, [("doc.md", 9)]) is None


def test_rm03_recall_top1_miss_top5_hit():
    cases = [{"id": "q1", "expected_sources": ["target.md"]}]
    details = [
        {
            "id": "q1",
            "citations": ["a.md#0", "b.md#0", "c.md#0", "target.md#0", "e.md#0"],
            "latency_ms": 5.0,
        }
    ]
    metrics = compute_retrieval_metrics(cases, details)
    assert metrics["recall_at_1"] == 0.0
    assert metrics["recall_at_5"] == 1.0


def test_rm04_no_hit():
    cases = [{"id": "q1", "expected_sources": ["target.md"]}]
    details = [{"id": "q1", "citations": ["a.md#0", "b.md#0"], "latency_ms": 5.0}]
    metrics = compute_retrieval_metrics(cases, details)
    assert metrics["recall_at_5"] == 0.0
    assert metrics["mrr"] == 0.0
    assert metrics["per_case"][0]["rank"] is None


def test_rm05_no_evaluable_cases():
    cases = [{"id": "q1", "expected_sources": []}]
    details = [{"id": "q1", "citations": ["a.md#0"], "latency_ms": 5.0}]
    metrics = compute_retrieval_metrics(cases, details)
    assert metrics["evaluable_cases"] == 0
    assert metrics["recall_at_5"] == 0.0
    assert metrics["failure_rate"] == 1.0


def test_rm06_mrr_average_matches_manual():
    cases = [{"id": "q1", "expected_sources": ["t.md"]}]
    cases += [{"id": "q2", "expected_sources": ["t.md"]}]
    cases += [{"id": "q3", "expected_sources": ["t.md"]}]
    details = [
        {"id": "q1", "citations": ["t.md#0", "x.md#0"], "latency_ms": 1.0},
        {"id": "q2", "citations": ["x.md#0", "t.md#0"], "latency_ms": 2.0},
        {"id": "q3", "citations": ["x.md#0", "y.md#0"], "latency_ms": 3.0},
    ]
    metrics = compute_retrieval_metrics(cases, details)
    expected_mrr = (1.0 + 0.5 + 0.0) / 3.0
    assert abs(metrics["mrr"] - expected_mrr) < 1e-9


def test_rm07_percentile_empty():
    assert percentile([], 95) == 0.0


def test_rm08_percentile_nearest_rank():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 80) == 4.0
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 5.0


def test_rm09_legacy_sources_fallback():
    # Legacy ground truth schema uses `sources` (without `expected_sources`).
    cases = [{"id": "q1", "sources": ["doc.md"]}]
    details = [{"id": "q1", "citations": ["doc.md#0"], "latency_ms": 10.0}]
    metrics = compute_retrieval_metrics(cases, details)
    assert metrics["evaluable_cases"] == 1
    assert metrics["recall_at_5"] == 1.0
