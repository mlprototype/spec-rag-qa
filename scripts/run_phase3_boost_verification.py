"""
Phase 3: Exact Match Boost エンドツーエンド検証スクリプト。
BM25インデックスをロードし、Boost あり/なしのランキングを比較する。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ragqa.bm25_store import BM25Store
from ragqa.config import cfg
from ragqa.tokenizer_ja import detect_special_tokens


VERIFY_CASES = [
    {
        "id": "exp-001",
        "query": "登録済みメールアドレスで会員登録した場合のステータスコードは何ですか？",
        "expected_docs": ["sample_spec_v2.md", "error_code_reference.md"],
    },
    {
        "id": "exp-002",
        "query": "重複アカウント登録時の応答コードを教えてください。",
        "expected_docs": ["api_contract_signup.md", "error_code_reference.md"],
    },
    {
        "id": "exp-004",
        "query": "signup のエンドポイント定義はどの文書にありますか？",
        "expected_docs": ["api_contract_signup.md", "sample_spec_v2.md"],
    },
    {
        "id": "exp-011",
        "query": "重複メールは 422 ですか、それとも 409 ですか？",
        "expected_docs": ["error_code_reference.md", "sample_spec_v2.md"],
    },
    {
        "id": "exp-006",
        "query": "監査ログに USER_ID は必須ですか？",
        "expected_docs": ["audit_logging_standard.md"],
    },
    {
        "id": "exp-017",
        "query": "メール重複と HTTP 409 のマッピングを教えてください。",
        "expected_docs": ["error_code_reference.md", "sample_spec_v2.md"],
    },
]


def _top5_chunks(
    bm25: BM25Store,
    meta: list[dict],
    query: str,
    boost_alpha: float,
    boost_beta: float,
) -> list[tuple[str, int]]:
    """
    上位5件を (doc_id, chunk_id) タプルで返す。
    """
    hits = bm25.search(
        query,
        top_k=5,
        boost_alpha=boost_alpha,
        boost_beta=boost_beta,
    )

    results: list[tuple[str, int]] = []
    for h in hits:
        idx = h["chunk_idx"]
        if 0 <= idx < len(meta):
            m = meta[idx]
            results.append((m["doc_id"], m["chunk_id"]))
    return results


def _hit(top5_chunks: list[tuple[str, int]], expected_docs: list[str]) -> bool:
    return any(doc_id in expected_docs for doc_id, _ in top5_chunks)


def _best_rank(top5_chunks: list[tuple[str, int]], expected_docs: list[str]) -> int | None:
    for rank, (doc_id, _) in enumerate(top5_chunks, start=1):
        if doc_id in expected_docs:
            return rank
    return None


def main() -> None:
    if not cfg.bm25_index_path.exists() or not cfg.bm25_postings_path.exists():
        print(f"ERROR: BM25 index not found: {cfg.bm25_index_path}")
        print("Run: TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python -m ragqa.ingest")
        sys.exit(1)
    if not cfg.meta_path.exists():
        print(f"ERROR: meta not found: {cfg.meta_path}")
        print("Run: TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python -m ragqa.ingest")
        sys.exit(1)

    print("Loading BM25 index...")
    bm25 = BM25Store.load(cfg.bm25_index_path, cfg.bm25_postings_path)
    meta: list[dict] = json.loads(cfg.meta_path.read_text(encoding="utf-8"))
    print(f"  chunks={bm25.N}  vocab={len(bm25.df)}  avgdl={bm25.avgdl:.1f}")

    results = []
    hit_no_boost = 0
    hit_boosted = 0

    print()
    print("=" * 70)
    for case in VERIFY_CASES:
        query = case["query"]
        expected = case["expected_docs"]
        special_toks = detect_special_tokens(query)

        no_boost_top5 = _top5_chunks(
            bm25,
            meta,
            query,
            boost_alpha=0.0,
            boost_beta=0.0,
        )
        boosted_top5 = _top5_chunks(
            bm25,
            meta,
            query,
            boost_alpha=cfg.boost_alpha,
            boost_beta=cfg.boost_beta,
        )

        no_boost_hit_flag = _hit(no_boost_top5, expected)
        boosted_hit_flag = _hit(boosted_top5, expected)
        no_boost_rank = _best_rank(no_boost_top5, expected)
        boosted_rank = _best_rank(boosted_top5, expected)

        rank_improved = (
            (boosted_rank is not None)
            and (no_boost_rank is None or boosted_rank < no_boost_rank)
        )

        if no_boost_hit_flag:
            hit_no_boost += 1
        if boosted_hit_flag:
            hit_boosted += 1

        result = {
            "id": case["id"],
            "query": query,
            "special_tokens": special_toks,
            "expected_docs": expected,
            "no_boost_top5_chunks": no_boost_top5,
            "boosted_top5_chunks": boosted_top5,
            "no_boost_hit": no_boost_hit_flag,
            "boosted_hit": boosted_hit_flag,
            "no_boost_rank": no_boost_rank,
            "boosted_rank": boosted_rank,
            "rank_improved": rank_improved,
        }
        results.append(result)

        boost_indicator = "↑" if rank_improved else ("→" if boosted_hit_flag == no_boost_hit_flag else "↓")
        print(f"[{case['id']}] {boost_indicator}  special={special_toks}")
        print(f"  no_boost : rank={no_boost_rank}  top5_chunks={no_boost_top5}")
        print(f"  boosted  : rank={boosted_rank}   top5_chunks={boosted_top5}")
        print()

    n = len(VERIFY_CASES)
    recall_no_boost = hit_no_boost / n
    recall_boosted = hit_boosted / n

    summary = {
        "phase": "Phase 3 Boost Verification",
        "bm25_only_recall_at_5_no_boost": recall_no_boost,
        "bm25_only_recall_at_5_boosted": recall_boosted,
        "boost_alpha": cfg.boost_alpha,
        "boost_beta": cfg.boost_beta,
        "cases": results,
    }

    out = Path("data/eval/phase3_boost_verification.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print(f"BM25 Recall@5  no_boost={recall_no_boost:.2f}  boosted={recall_boosted:.2f}")
    print(f"Report saved: {out}")


if __name__ == "__main__":
    main()
