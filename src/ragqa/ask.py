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


def _parse_fallback(text: str) -> Dict[str, Any]:
    """
    JSON解析に失敗した場合の救済措置（Fallback）
    テキストから正規表現で無理やり判定（Verdict）を読み取る
    """
    text_lower = text.lower()

    # デフォルトは安全側に倒して insufficient
    verdict = "insufficient"

    # 正規表現で "verdict" : "sufficient/insufficient" のパターンを探す
    # コロンの前後のスペース、クォートの有無(' or ")を許容
    match = re.search(
        r"['\"]?verdict['\"]?\s*[:=]\s*['\"]?(sufficient|insufficient)['\"]?",
        text_lower,
    )

    if match:
        verdict = match.group(1)
    else:
        # キーが見つからない場合、単語の出現で推測（最後の手段）
        # insufficient が含まれていれば NG とみなす
        if "insufficient" in text_lower:
            verdict = "insufficient"
        elif "sufficient" in text_lower:
            verdict = "sufficient"

    return {
        "verdict": verdict,
        "confidence": 10,  # 救済措置なので信頼度は低く設定
        "reasons": ["JSON parse failed, recovered by text fallback logic."],
        "missing_points": ["(Check raw_verifier_json for details)"],
        "supported_claims": [],
        "unsupported_claims": [],
    }


def build_evidence_check_prompt(question: str, answer: str, contexts: List[str]) -> str:
    # contexts は既に "\n本文" なので、そのまま証拠として渡せる
    evidence = "\n\n".join(contexts)

    return f"""あなたは厳格な仕様レビュー担当です。
ユーザー質問・回答・根拠（引用チャンク本文）を読み、回答が根拠だけで十分に支持されているか判定してください。

  # ルール
- 根拠に書いていないことを推測で補完してはいけません。
- 根拠から直接言えることだけを supported_claims に入れてください。
- 仕様に無い/根拠が足りない場合は verdict を "insufficient" にしてください。
- **回答が主観的な評価（「使いにくい」「良い」「悪い」など）を含んでいる場合、その評価という事実そのものが根拠テキストに明記されていなければ、たとえ論理的に正しくても "insufficient" にしてください。**
  # 網羅性の判定基準
- 質問が「一覧」「条件は？」などを求めている場合でも、以下の場合は verdict を "sufficient" と判定してください：
  1. 根拠が見出し（例: ## 入力）を含み、その配下に箇条書き等で項目が列挙されている場合、そのセクションが「条件の全て」であるとみなして良い。
  2. 設計書(design.md)に「将来要件」「不要」と明記されている場合、それを根拠に「必須ではない」と断定して良い。
  3. 根拠となる文書の文脈から、それ以外の条件が存在しないことが合理的に推測できる場合。

- ただし、根拠テキストが文の途中で切れている（チャンク切れ）と思われる場合は "insufficient" にしてください。


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
    # 既存フォーマットを壊さず、末尾に追記
    extra = "\n- 根拠の十分性チェックで不足と判定された点:\n" + "\n".join(
        [f"  - {p}" for p in missing_points]
    )
    return answer + extra


def ask_question(question: str) -> Dict[str, Any]:
    """
    質問を受け取り、RAGの回答と検証結果を辞書で返す（評価スクリプト用）
    """
    if not cfg.faiss_index_path.exists() or not cfg.meta_path.exists():
        return {"error": "Index not found. Run ingest first."}

    vs = VectorStore.load(cfg.faiss_index_path, cfg.meta_path)
    embedder = Embedder()
    q = embedder.embed_query(question)

    # 1. 検索
    hits = vs.search(q, cfg.top_k)
    contexts = []
    for h in hits:
        # ログ等で見やすいようにタグ付け
        tag = f"[source: {h['doc_id']}#{h['chunk_id']}]"
        contexts.append(tag + "\n" + h["text"])

    # 2. 回答生成
    prompt = build_prompt(question, contexts)
    ans = run_llm(prompt, contexts)

    # 3. 根拠の十分性チェック（Verifier）
    check_prompt = build_evidence_check_prompt(question, ans, contexts)
    raw_check = run_llm(check_prompt, contexts)

    try:
        # 構造化データとしての抽出を試みる
        check = _extract_json_object(raw_check)
    except Exception:
        # 失敗した場合、テキストFallback戦略に切り替え
        check = _parse_fallback(raw_check)
        # デバッグ用に生出力を保存しておく
        check["raw_text_on_error"] = raw_check

    verdict = check.get("verdict", "insufficient")
    missing_points = (
        check.get("missing_points", [])
        if isinstance(check.get("missing_points"), list)
        else []
    )

    # insufficient のときは Answer に追記
    final_answer = ans
    if verdict == "insufficient":
        final_answer = merge_missing_points_into_answer(ans, missing_points)

    # 結果を構造化して返す
    return {
        "question": question,
        "answer": final_answer,
        "verdict": verdict,
        "confidence": check.get("confidence", 0),
        "supported_claims": check.get("supported_claims", []),
        "unsupported_claims": check.get("unsupported_claims", []),
        "missing_points": missing_points,
        "hits": hits,
        "raw_verifier_json": check,  # デバッグ用に生JSONも持っておく
    }


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m ragqa.ask "質問文"')

    question = sys.argv[1]

    # リファクタリングした関数を呼び出し
    result = ask_question(question)

    if "error" in result:
        print(result["error"])
        return

    # === 以下、表示ロジック ===
    print("==== Retrieved ====")
    for h in result["hits"]:
        print(f"- {h['doc_id']}#{h['chunk_id']} score={h['score']:.3f}")

    print("\n==== Answer ====")
    print(result["answer"])

    print("\n==== Evidence Check ====")

    # 検証結果の表示
    if result["supported_claims"]:
        print("- supported_claims:")
        for c in result["supported_claims"][:10]:
            print(f"  - {c}")

    if result["unsupported_claims"]:
        print("- unsupported_claims:")
        for c in result["unsupported_claims"][:10]:
            print(f"  - {c}")

    print(f"- verdict: {result['verdict']} (confidence={result['confidence']})")

    if result["missing_points"]:
        print("- missing_points:")
        for p in result["missing_points"][:20]:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
