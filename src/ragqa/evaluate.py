# src/ragqa/evaluate.py

import json
import time
from pathlib import Path
from typing import Dict, List

from .ask import ask_question  # Step 1で修正した関数をインポート

# 設定
EVAL_FILE = Path("data/eval/ground_truth.json")
REPORT_FILE = Path("data/eval/report.json")


def load_ground_truth() -> List[Dict]:
    if not EVAL_FILE.exists():
        print(f"Error: {EVAL_FILE} not found. Please create it first.")
        return []
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def check_answer(case: Dict, result: Dict) -> Dict:
    """
    質問タイプに応じた厳密な合否判定ロジック
    """
    # 基本情報の取得
    q_type = case.get("type", "explicit")  # デフォルトは明示質問
    expected_verdict = case.get("expected_verdict")
    actual_verdict = result.get("verdict")

    # 判定フラグ
    is_pass = True
    fail_reasons = []

    # === 共通チェック: Verdictの一致 ===
    if expected_verdict and expected_verdict != actual_verdict:
        is_pass = False
        fail_reasons.append(
            f"Verdict mismatch: expected {expected_verdict}, got {actual_verdict}"
        )

    # === タイプ別チェック ===

    # ① 明示仕様質問 (Explicit)
    if q_type == "explicit":
        # sufficient であることが必須（共通チェックでカバー済みだが念押し）
        if actual_verdict != "sufficient":
            # 既に共通チェックでFalseになっているが、理由を明確化
            pass

    # ② 未定義検出質問 (Undefined)
    elif q_type == "undefined":
        # insufficient であること + 「書いてない」という根拠が出ているか
        # unsupported_claims または missing_points に何か入っているべき
        has_unsupported = len(result.get("unsupported_claims", [])) > 0
        has_missing = len(result.get("missing_points", [])) > 0

        if not (has_unsupported or has_missing):
            is_pass = False
            fail_reasons.append(
                "Type Undefined error: No missing points or unsupported claims detected"
            )

    # ③ 横断整合性質問 (Cross-doc)
    elif q_type == "cross_doc":
        # 複数のドキュメントを参照しているかチェック
        # hits からユニークな doc_id を抽出
        hits = result.get("hits", [])
        unique_docs = set(h["doc_id"] for h in hits)

        if len(unique_docs) < 2:
            # 必須ではないかもしれないが、Cross-docなら複数見ていないと怪しい
            # ここでは厳格にFAILにするか、WARNにするか選べる。今回は厳格に判定。
            is_pass = False
            fail_reasons.append(
                f"Type Cross-doc error: Referred only {len(unique_docs)} doc(s). Expected >= 2"
            )

    # ④ 判断禁止質問 (Non-judgmental)
    elif q_type == "non_judgmental":
        # 絶対に sufficient になってはいけない
        if actual_verdict == "sufficient":
            is_pass = False
            fail_reasons.append(
                "Type Non-judgmental error: AI made a judgment (sufficient) on subjective topic"
            )

    # === キーワードチェック (must_include) ===
    must_include = case.get("must_include", [])
    answer_text = result.get("answer", "")
    missing_words = []

    for word in must_include:
        if word not in answer_text:
            is_pass = False
            missing_words.append(word)

    if missing_words:
        fail_reasons.append(f"Missing keywords: {missing_words}")

    return {
        "id": case.get("id"),
        "question": case.get("question"),
        "type": q_type,  # レポートにもタイプを出力
        "pass": is_pass,
        "details": {
            "fail_reasons": fail_reasons,  # 失敗理由をリストで返す
            "expected_verdict": expected_verdict,
            "actual_verdict": actual_verdict,
            "actual_answer": answer_text,
        },
    }


def main():
    cases = load_ground_truth()
    if not cases:
        return

    print(f"Starting evaluation of {len(cases)} cases...\n")

    results = []
    passed_count = 0
    start_time = time.time()

    for i, case in enumerate(cases):
        print(f"[{i + 1}/{len(cases)}] {case['question']} ... ", end="", flush=True)

        # RAG実行
        try:
            rag_result = ask_question(case["question"])
            if "error" in rag_result:
                print(f"ERROR: {rag_result['error']}")
                results.append(
                    {"id": case["id"], "pass": False, "error": rag_result["error"]}
                )
                continue

            # 判定
            eval_result = check_answer(case, rag_result)
            results.append(eval_result)

            if eval_result["pass"]:
                print("✅ PASS")
                passed_count += 1
            else:
                print("❌ FAIL")
        except Exception as e:
            print(f"EXCEPTION: {e}")
            results.append({"id": case["id"], "pass": False, "error": str(e)})

    duration = time.time() - start_time
    score = (passed_count / len(cases)) * 100 if cases else 0

    # レポート出力
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "score": score,
        "duration_seconds": duration,
        "total": len(cases),
        "passed": passed_count,
        "failed": len(cases) - passed_count,
        "details": results,
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n==============================")
    print("Evaluation Complete")
    print(f"Score: {score:.1f}% ({passed_count}/{len(cases)})")
    print(f"Time:  {duration:.2f}s")
    print(f"Report saved to: {REPORT_FILE}")
    print("==============================")


if __name__ == "__main__":
    main()
