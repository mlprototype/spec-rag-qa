from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from .chunking import Chunk
from .tokenizer_ja import detect_special_tokens, tokenize


class BM25Store:
    def __init__(self, b: float = 0.75, k1: float = 2.0) -> None:
        self.b = float(b)
        self.k1 = float(k1)
        self.N = 0
        self.avgdl = 0.0

        self.tokens_by_chunk: list[list[str]] = []
        self.dl: list[int] = []
        self.tf: list[dict[str, int]] = []
        self.df: dict[str, int] = {}
        self.postings: dict[str, list[list[int]]] = {}

    def build(self, chunks: list[Chunk]) -> None:
        self.tokens_by_chunk = []
        self.dl = []
        self.tf = []
        self.df = {}
        self.postings = {}

        for chunk in chunks:
            tokens = tokenize(chunk.text)
            tf_map = dict(Counter(tokens))

            self.tokens_by_chunk.append(tokens)
            self.dl.append(len(tokens))
            self.tf.append(tf_map)

            for term in tf_map:
                self.df[term] = self.df.get(term, 0) + 1

        self.N = len(chunks)
        self.avgdl = (sum(self.dl) / self.N) if self.N > 0 else 0.0

        postings: dict[str, list[list[int]]] = {term: [] for term in self.df}
        for chunk_idx, tf_map in enumerate(self.tf):
            for term, tf_value in tf_map.items():
                postings[term].append([chunk_idx, int(tf_value)])
        self.postings = postings

    def score(self, chunk_idx: int, query_tokens: list[str]) -> float:
        if not query_tokens or self.N == 0:
            return 0.0
        if not (0 <= chunk_idx < self.N):
            raise IndexError(f"chunk_idx out of range: {chunk_idx}")
        if self.avgdl <= 0:
            return 0.0

        tf_map = self.tf[chunk_idx]
        dl = self.dl[chunk_idx]
        score_value = 0.0

        for term in query_tokens:
            df_value = self.df.get(term, 0)
            if df_value == 0:
                continue

            tf_value = tf_map.get(term, 0)
            if tf_value == 0:
                continue

            idf = math.log((self.N - df_value + 0.5) / (df_value + 0.5))
            denom = tf_value + self.k1 * (
                1.0 - self.b + self.b * (dl / self.avgdl)
            )
            if denom == 0:
                continue
            tf_norm = tf_value * (self.k1 + 1.0) / denom
            score_value += idf * tf_norm

        return float(score_value)

    def search(
        self,
        query: str,
        top_k: int,
        boost_alpha: float = 0.0,
        boost_beta: float = 0.0,
    ) -> list[dict]:
        if top_k <= 0 or self.N == 0:
            return []

        query_tokens = tokenize(query)
        special_tokens = detect_special_tokens(query)
        results: list[dict] = []

        for chunk_idx in range(self.N):
            bm25_raw = self.score(chunk_idx, query_tokens)
            exact_hit_count = sum(1 for t in special_tokens if t in self.tf[chunk_idx])
            all_hit_flag = (
                1
                if special_tokens and exact_hit_count == len(special_tokens)
                else 0
            )
            bm25_boosted = (
                bm25_raw + boost_alpha * exact_hit_count + boost_beta * all_hit_flag
            )

            results.append(
                {
                    "chunk_idx": chunk_idx,
                    "bm25_score": float(bm25_boosted),
                    "exact_hits": int(exact_hit_count),
                }
            )

        results.sort(key=lambda x: (-x["bm25_score"], x["chunk_idx"]))
        return results[:top_k]

    def save(self, index_path: Path, postings_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        postings_path.parent.mkdir(parents=True, exist_ok=True)

        with index_path.open("w", encoding="utf-8") as f:
            for chunk_idx, tokens in enumerate(self.tokens_by_chunk):
                row = {
                    "chunk_idx": chunk_idx,
                    "tokens": tokens,
                    "dl": self.dl[chunk_idx],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        with postings_path.open("w", encoding="utf-8") as f:
            stats = {
                "type": "stats",
                "N": self.N,
                "avgdl": self.avgdl,
                "b": self.b,
                "k1": self.k1,
            }
            f.write(json.dumps(stats, ensure_ascii=False) + "\n")
            for term in sorted(self.postings):
                row = {
                    "term": term,
                    "df": self.df.get(term, 0),
                    "postings": self.postings[term],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, index_path: Path, postings_path: Path) -> "BM25Store":
        stats: dict | None = None
        postings: dict[str, list[list[int]]] = {}
        df: dict[str, int] = {}

        with postings_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("type") == "stats":
                    stats = row
                    continue
                term = row["term"]
                term_postings = [
                    [int(p[0]), int(p[1])] for p in row.get("postings", [])
                ]
                postings[term] = term_postings
                df[term] = int(row.get("df", len(term_postings)))

        b = float(stats.get("b", 0.75)) if stats else 0.75
        k1 = float(stats.get("k1", 2.0)) if stats else 2.0
        store = cls(b=b, k1=k1)

        rows: list[dict] = []
        with index_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        rows.sort(key=lambda x: int(x["chunk_idx"]))

        for row in rows:
            tokens = list(row.get("tokens", []))
            store.tokens_by_chunk.append(tokens)
            store.dl.append(int(row.get("dl", len(tokens))))
            store.tf.append(dict(Counter(tokens)))

        store.N = int(stats.get("N", len(store.tokens_by_chunk))) if stats else len(
            store.tokens_by_chunk
        )
        if stats and "avgdl" in stats:
            store.avgdl = float(stats["avgdl"])
        else:
            store.avgdl = (
                sum(store.dl) / len(store.dl)
                if store.dl
                else 0.0
            )

        store.postings = postings
        if df:
            store.df = df
        else:
            inferred_df: dict[str, int] = {}
            for tf_map in store.tf:
                for term in tf_map:
                    inferred_df[term] = inferred_df.get(term, 0) + 1
            store.df = inferred_df

        return store
