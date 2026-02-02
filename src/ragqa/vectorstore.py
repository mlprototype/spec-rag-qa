from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from .chunking import Chunk


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine相当（正規化済み前提）
        self.meta: list[dict] = []

    def add(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        assert embeddings.shape[0] == len(chunks)
        self.index.add(embeddings)
        for c in chunks:
            self.meta.append(asdict(c))

    def save(self, index_path: Path, meta_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        meta_path.write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def load(index_path: Path, meta_path: Path) -> "VectorStore":
        index = faiss.read_index(str(index_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        vs = VectorStore(index.d)
        vs.index = index
        vs.meta = meta
        return vs

    def search(self, query_emb: np.ndarray, top_k: int) -> list[dict]:
        scores, idxs = self.index.search(query_emb, top_k)
        results: list[dict] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            m = self.meta[idx].copy()
            m["score"] = float(score)
            results.append(m)
        return results
