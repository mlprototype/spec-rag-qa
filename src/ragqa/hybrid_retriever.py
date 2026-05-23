from __future__ import annotations

from .bm25_store import BM25Store
from .config import cfg
from .embedder import Embedder
from .vectorstore import VectorStore


def _rrf_fuse(
    vector_hits: list[dict],
    bm25_hits: list[dict],
    meta: list[dict],
    rrf_k: int,
) -> dict[tuple[str, int], float]:
    """
    Reciprocal Rank Fusion.

    Returns:
      {(doc_id, chunk_id): rrf_score}
    """
    scores: dict[tuple[str, int], float] = {}

    for rank, hit in enumerate(vector_hits):
        key = (hit["doc_id"], hit["chunk_id"])
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)

    for rank, hit in enumerate(bm25_hits):
        chunk_idx = hit["chunk_idx"]
        if chunk_idx < 0 or chunk_idx >= len(meta):
            continue
        m = meta[chunk_idx]
        key = (m["doc_id"], m["chunk_id"])
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)

    return scores


class HybridRetriever:
    def __init__(
        self,
        vs: VectorStore,
        bm25: BM25Store,
        embedder: Embedder,
    ) -> None:
        self.vs = vs
        self.bm25 = bm25
        self.embedder = embedder
        # 改善案: meta lookup は __init__ で1回だけ構築する
        self._meta_lookup = {(m["doc_id"], m["chunk_id"]): m for m in self.vs.meta}

    @classmethod
    def load(cls) -> "HybridRetriever":
        """本番用ファクトリ。インデックスファイルからロードする。"""
        if not cfg.faiss_index_path.exists() or not cfg.meta_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {cfg.faiss_index_path}. Run ingest first."
            )
        if not cfg.bm25_index_path.exists() or not cfg.bm25_postings_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found: {cfg.bm25_index_path}. Run ingest first."
            )
        vs = VectorStore.load(cfg.faiss_index_path, cfg.meta_path)
        bm25 = BM25Store.load(cfg.bm25_index_path, cfg.bm25_postings_path)
        embedder = Embedder()
        return cls(vs, bm25, embedder)

    def retrieve(
        self,
        query: str,
        final_top_k: int | None = None,
        *,
        vector_candidate_k: int | None = None,
        bm25_candidate_k: int | None = None,
        rrf_k: int | None = None,
        boost_alpha: float | None = None,
        boost_beta: float | None = None,
    ) -> list[dict]:
        """
        ハイブリッド検索を実行して上位 final_top_k 件を返す。
        戻り値のキーは VectorStore.search() と同一:
        [{doc_id, chunk_id, text, score}, ...]  (score = RRF スコア)
        """
        if final_top_k is None:
            final_top_k = cfg.final_top_k
        if vector_candidate_k is None:
            vector_candidate_k = cfg.vector_candidate_k
        if bm25_candidate_k is None:
            bm25_candidate_k = cfg.bm25_candidate_k
        if rrf_k is None:
            rrf_k = cfg.rrf_k
        if boost_alpha is None:
            boost_alpha = cfg.boost_alpha
        if boost_beta is None:
            boost_beta = cfg.boost_beta
        if final_top_k <= 0:
            return []

        q_emb = self.embedder.embed_query(query)
        vector_hits = self.vs.search(q_emb, vector_candidate_k)
        bm25_hits = self.bm25.search(
            query,
            bm25_candidate_k,
            boost_alpha=boost_alpha,
            boost_beta=boost_beta,
        )

        rrf_scores = _rrf_fuse(vector_hits, bm25_hits, self.vs.meta, rrf_k)
        top = sorted(
            rrf_scores.items(),
            key=lambda x: (-x[1], x[0][0], x[0][1]),
        )[:final_top_k]

        results: list[dict] = []
        for (doc_id, chunk_id), rrf_score in top:
            meta = self._meta_lookup.get((doc_id, chunk_id))
            if meta is None:
                continue
            results.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "text": meta["text"],
                    "score": rrf_score,
                }
            )
        return results
