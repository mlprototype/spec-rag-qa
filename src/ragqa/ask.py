from __future__ import annotations

import json
import re
import sys
from typing import Any, Dict, List

from .config import cfg
from .embedder import Embedder
from .llm import run_llm
from .prompt import build_prompt
from .vectorstore import VectorStore

if not cfg.faiss_index_path.exists() or not cfg.meta_path.exists():
    raise SystemExit("Index not found. Run: PYTHONPATH=src python -m ragqa.ingest")


# =========================
# Evidence check helpers
# =========================
def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    LLMが前置き/後置きを混ぜても、最初に見つかったJSONオブジェクトを抽出する。
    """
    # まずは全体をそのままJSONとして試す
    try:
        return json.loads(text)
    except Exception:
        pass

    # ```json ... ``` の中身を優先
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    # 最初の { から最後の } までを雑に抜く（最小実装）
    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in verifier output.")
    return json.loads(m.group(1))


def build_evidence_check_prompt(question: str, answer: str, contexts: List[str]) -> str:
    # contexts は既に "[source: doc#chunk]\n本文" なので、そのまま証拠として渡せる
    evidence = "\n\n".join(contexts)

    return f"""あなたは厳格な仕様レビュー担当です。
ユーザー質問・回答・根拠（引用チャンク本文）を読み、回答が根拠だけで十分に支持されているか判定してください。

# ルール（重要）
- 根拠に書いていないことを推測で補完してはいけません。
- 「一般的には〜」など外部知識で正当化してはいけません。
- 根拠から直接言えることだけを supported_claims に入れてください。
- 根拠が不足している主張は unsupported_claims に入れてください。
- 仕様に無い/根拠が足りない場合は verdict を "insufficient" にしてください。
- 質問が「一覧」「すべて」「例外条件は？」のように網羅性を求める場合、
  根拠が限定的で “これが全て” と断言できないなら verdict を "insufficient" にしてください。
  その場合 missing_points に「例外条件の一覧が他にないか確認」などを入れてください。
- 次の条件を満たさない限り、網羅性が必要な質問（例: 例外条件は？）では verdict を "insufficient" にしてください：
  (A) 根拠中に「例外条件は以下の通り（全て）」など “網羅” を示す表現がある
  または
  (B) 根拠中に「例外条件はこの2つのみ」等の “限定” が明示されている
- 「全て」「網羅」「のみ」など網羅性/限定を主張する場合、
  根拠チャンクからそのことを示す短い引用を coverage_evidence に入れてください（20語以内）。
  引用が取れない場合、verdict は必ず "insufficient" にしてください。


# 入力
[question]
{question}

[answer]
{answer}

[evidence_chunks]
{evidence}

# 出力（JSONのみ、余計な文章は禁止）
{{
  "verdict": "sufficient" or "insufficient",
  "confidence": 0-100,
  "reasons": ["..."],
  "missing_points": ["..."],
  "supported_claims": ["..."],
  "unsupported_claims": ["..."],
  "coverage_evidence": "",
  "coverage_source_id": ""
}}
"""


def merge_missing_points_into_answer(answer: str, missing_points: List[str]) -> str:
    if not missing_points:
        return answer
    # 既存フォーマットを壊さず、末尾に追記（まずこれで十分）
    extra = "\n- 根拠の十分性チェックで不足と判定された点:\n" + "\n".join(
        [f"  - {p}" for p in missing_points]
    )
    return answer + extra


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m ragqa.ask "質問文"')

    question = sys.argv[1]

    vs = VectorStore.load(cfg.faiss_index_path, cfg.meta_path)
    embedder = Embedder()
    q = embedder.embed_query(question)

    hits = vs.search(q, cfg.top_k)

    contexts = []
    for h in hits:
        doc_type = h.get("doc_type", "unknown")
        tag = f"[source: {h['doc_id']}#{h['chunk_id']} role={doc_type}]"
        contexts.append(tag + "\n" + h["text"])

    # 1) 通常回答
    prompt = build_prompt(question, contexts)
    ans = run_llm(prompt, contexts)

    # 2) 根拠の十分性チェック（Verifier）
    check_prompt = build_evidence_check_prompt(question, ans, contexts)

    # run_llm が contexts を内部で参照しても良いように渡す（不要なら [] でもOK）
    raw_check = run_llm(check_prompt, contexts)

    try:
        check = _extract_json_object(raw_check)
    except Exception as e:
        check = {
            "verdict": "insufficient",
            "confidence": 0,
            "reasons": [f"Verifier JSON parse failed: {e}"],
            "missing_points": [
                "Verifierの出力がJSONになっていません。プロンプトを見直してください。"
            ],
            "supported_claims": [],
            "unsupported_claims": [],
        }

    verdict = check.get("verdict", "insufficient")
    missing_points = (
        check.get("missing_points", [])
        if isinstance(check.get("missing_points"), list)
        else []
    )

    # insufficient のときは Answer に追記（まずはここまで）
    if verdict == "insufficient":
        ans = merge_missing_points_into_answer(ans, missing_points)

    print("==== Retrieved ====")
    for h in hits:
        print(f"- {h['doc_id']}#{h['chunk_id']} score={h['score']:.3f}")

    print("\n==== Answer ====")
    print(ans)

    print("\n==== Evidence Check ====")

    supported = check.get("supported_claims", [])
    unsupported = check.get("unsupported_claims", [])

    if supported:
        print("- supported_claims:")
        for c in supported[:10]:
            print(f"  - {c}")

    if unsupported:
        print("- unsupported_claims:")
        for c in unsupported[:10]:
            print(f"  - {c}")

    print(f"- verdict: {check.get('verdict')} (confidence={check.get('confidence')})")
    for r in check.get("reasons", [])[:10]:
        print(f"- reason: {r}")

    # 便利なので missing_points も見える化
    if missing_points:
        print("- missing_points:")
        for p in missing_points[:20]:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
