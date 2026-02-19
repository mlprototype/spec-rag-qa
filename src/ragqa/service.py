from __future__ import annotations

from .config import cfg
from .embedder import Embedder
from .llm import run_llm
from .prompt import build_prompt
from .schemas import AnswerResult, Source, Verification
from .utils import (
    _extract_json_object,
    _parse_fallback,
    build_evidence_check_prompt,
    # merge_missing_points_into_answer は削除（使用しないため）
)
from .vectorstore import VectorStore


if cfg.enable_langsmith and cfg.langsmith_tracing and cfg.langsmith_api_key:
    from langsmith import traceable as _traceable
else:

    def _traceable(*args, **kwargs):  # type: ignore[no-redef]
        def decorator(func):
            return func

        return decorator


@_traceable(name="RAG Pipeline")
def answer_question(question: str) -> AnswerResult:
    """
    質問を受け取り、RAGを実行し、検証結果を含めた構造化データを返す。
    UI依存（printなど）は一切行わない。
    """
    # 1. インデックスのロード
    if not cfg.faiss_index_path.exists() or not cfg.meta_path.exists():
        raise FileNotFoundError("Index not found. Run ingest first.")

    vs = VectorStore.load(cfg.faiss_index_path, cfg.meta_path)
    embedder = Embedder()
    q_emb = embedder.embed_query(question)

    # 2. 検索 (Retrieve)
    hits = vs.search(q_emb, cfg.top_k)

    # Sourceオブジェクトのリストに変換
    sources = [
        Source(
            doc_id=h["doc_id"],
            chunk_id=h["chunk_id"],
            text=h["text"],
            score=h.get("score", 0.0),
        )
        for h in hits
    ]

    # コンテキスト作成
    contexts = []
    for s in sources:
        tag = f"[source: {s.doc_id}#{s.chunk_id}]"
        contexts.append(tag + "\n" + s.text)

    # 3. 回答生成 (Generate)
    prompt = build_prompt(question, contexts)
    initial_answer = run_llm(prompt, contexts)

    # 4. 検証 (Verify)
    check_prompt = build_evidence_check_prompt(question, initial_answer, contexts)
    raw_check = run_llm(check_prompt, contexts)

    try:
        check_data = _extract_json_object(raw_check)
    except Exception:
        check_data = _parse_fallback(raw_check)

    # 検証結果オブジェクトの作成
    verification = Verification(
        verdict=check_data.get("verdict", "insufficient"),
        confidence=float(check_data.get("confidence", 0)),
        missing_points=check_data.get("missing_points", []) or [],
        supported_claims=check_data.get("supported_claims", []) or [],
        unsupported_claims=check_data.get("unsupported_claims", []) or [],
    )

    # 【変更点】Verifierによる回答の書き換え（merge）を廃止
    # AnswerResultには「純粋な回答」と「検証結果」を分けて格納する
    final_answer = initial_answer

    # 5. 結果のパッキング (Return)
    return AnswerResult(
        question=question,
        answer=final_answer,
        verification=verification,
        sources=sources,
    )
