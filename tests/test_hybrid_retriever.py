import numpy as np

from ragqa.bm25_store import BM25Store
from ragqa.chunking import Chunk
from ragqa.hybrid_retriever import HybridRetriever, _rrf_fuse
from ragqa.vectorstore import VectorStore


def _make_vs(chunks: list[Chunk]) -> VectorStore:
    """3次元ダミー埋め込みで FAISS インデックスを構築する。"""
    dim = 3
    vs = VectorStore(dim)
    rng = np.random.default_rng(seed=42)
    embs = rng.standard_normal((len(chunks), dim)).astype("float32")
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / norms
    vs.add(embs, chunks)
    return vs


class _FakeEmbedder:
    """固定ベクトルを返すフェイク埋め込みモデル。"""

    def __init__(self, dim: int = 3, seed: int = 0):
        rng = np.random.default_rng(seed=seed)
        v = rng.standard_normal(dim).astype("float32")
        self._vec = (v / np.linalg.norm(v)).reshape(1, -1)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec


def _make_retriever(chunks: list[Chunk]) -> HybridRetriever:
    vs = _make_vs(chunks)
    bm25 = BM25Store()
    bm25.build(chunks)
    return HybridRetriever(vs=vs, bm25=bm25, embedder=_FakeEmbedder())


def test_hyb01_both_lists_highest_score():
    meta = [
        {"doc_id": "d.md", "chunk_id": 0, "text": "A"},
        {"doc_id": "d.md", "chunk_id": 1, "text": "B"},
    ]
    vector_hits = [
        {"doc_id": "d.md", "chunk_id": 0, "text": "A", "score": 0.9},
        {"doc_id": "d.md", "chunk_id": 1, "text": "B", "score": 0.8},
    ]
    bm25_hits = [{"chunk_idx": 0, "bm25_score": 3.0, "exact_hits": 1}]

    scores = _rrf_fuse(vector_hits, bm25_hits, meta, rrf_k=60)
    assert scores[("d.md", 0)] > scores[("d.md", 1)]


def test_hyb02_single_list_items_included():
    meta = [
        {"doc_id": "d.md", "chunk_id": 0, "text": "A"},
        {"doc_id": "d.md", "chunk_id": 1, "text": "B"},
    ]
    vector_hits = [{"doc_id": "d.md", "chunk_id": 0, "text": "A", "score": 0.9}]
    bm25_hits = [{"chunk_idx": 1, "bm25_score": 3.0, "exact_hits": 0}]

    scores = _rrf_fuse(vector_hits, bm25_hits, meta, rrf_k=60)
    assert ("d.md", 0) in scores
    assert ("d.md", 1) in scores


def test_hyb03_rrf_k_affects_scores():
    meta = [{"doc_id": "d.md", "chunk_id": 0, "text": "A"}]
    vector_hits = [{"doc_id": "d.md", "chunk_id": 0, "text": "A", "score": 1.0}]
    bm25_hits = []

    s1 = _rrf_fuse(vector_hits, bm25_hits, meta, rrf_k=1)[("d.md", 0)]
    s2 = _rrf_fuse(vector_hits, bm25_hits, meta, rrf_k=100)[("d.md", 0)]
    assert s1 != s2


def test_hyb04_final_top_k_limits_results():
    chunks = [Chunk("d.md", i, f"チャンク{i}") for i in range(5)]
    retriever = _make_retriever(chunks)
    results = retriever.retrieve("テスト", final_top_k=2)
    assert len(results) == 2, f"final_top_k=2 なのに {len(results)} 件返った"


def test_hyb05_result_keys_complete():
    chunks = [Chunk("d.md", i, f"テキスト{i}") for i in range(3)]
    retriever = _make_retriever(chunks)
    results = retriever.retrieve("テキスト", final_top_k=3)

    assert len(results) > 0, "結果が0件"
    for r in results:
        assert "doc_id" in r
        assert "chunk_id" in r
        assert "text" in r
        assert "score" in r


def test_hyb06_empty_lists_no_error():
    meta = [{"doc_id": "d.md", "chunk_id": 0, "text": "A"}]
    assert _rrf_fuse([], [], meta, rrf_k=60) == {}

    scores = _rrf_fuse(
        [],
        [{"chunk_idx": 0, "bm25_score": 1.0, "exact_hits": 0}],
        meta,
        rrf_k=60,
    )
    assert scores != {}
    assert all(v > 0 for v in scores.values())
