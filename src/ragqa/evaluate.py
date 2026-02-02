import json
import time
from collections import Counter, defaultdict

# serviceからロジックを呼ぶ
from ragqa.ask import answer_question

# 改善アクションカタログをインポート
from ragqa.improvement_catalog import get_suggestion

GROUND_TRUTH_PATH = "data/eval/ground_truth.json"
REPORT_PATH = "data/eval/report.json"


def detect_fail_type(
    case: dict, result, verdict_ok: bool, must_include_ok: bool
) -> str | None:
    """
    eval_policy.md に基づき、FAILの原因を分類する
    """
    answer_text = result.answer.strip()

    # 1. 基本的なエラー
    if not answer_text or len(answer_text) < 5:
        return "EMPTY_ANSWER"

    # 2. RAGの自己評価と期待値のズレ
    if not verdict_ok:
        if (
            case["expected_verdict"] == "sufficient"
            and result.verification.verdict == "insufficient"
        ):
            return "RETRIEVAL_FAILURE (Evidence Missing)"

        if (
            case["expected_verdict"] == "insufficient"
            and result.verification.verdict == "sufficient"
        ):
            return "HALLUCINATION / OVERCONFIDENCE"

        return "VERDICT_MISMATCH"

    # 3. キーワード不足
    if not must_include_ok:
        if case["type"] == "omission_detection":
            return "OMISSION (Critical Condition Missing)"
        if case["type"] == "priority_conflict":
            return "PRIORITY_ERROR (Wrong Rule Applied)"
        if case["type"] == "opinion_guard":
            return "OPINION_LEAK (Subjective)"
        return "FACTUAL_ERROR (Keyword Missing)"

    return None  # PASS


def generate_trend_hints(distribution: dict, total_fail: int) -> list[str]:
    """分布データから、人間向けの分析コメントを生成する"""
    hints = []
    if total_fail == 0:
        return ["FAILはありません。素晴らしい状態です！"]

    # 1. CRITICALチェック
    crit_count = distribution["by_priority"].get("CRITICAL", 0)
    if crit_count > 0:
        hints.append(
            f"CRITICALなFAILが {crit_count}件 あります。これらは即時修正が必要です。"
        )

    # 2. 犯人探し（Owner分析）
    by_owner = distribution["by_owner"]
    if by_owner:
        # 最も多いOwnerを見つける
        top_owner, count = max(by_owner.items(), key=lambda x: x[1])
        if count >= total_fail * 0.5:  # 過半数を占める場合
            hints.append(
                f"{top_owner} 起因のFAILが全体の {count / total_fail * 100:.0f}% を占めています。ここの改善が効果的です。"
            )

    return hints


def run_evaluation():
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        cases = json.load(f)

    results = []

    # === 集計用カウンター ===
    cnt_fail_type = Counter()
    cnt_owner = Counter()
    cnt_priority = Counter()
    cnt_q_type = Counter()  # FAILした質問タイプのカウント(分布用)

    # 【復活】カテゴリ別スコア計算用 (True/Falseのリスト)
    by_type_score = defaultdict(list)

    print(f"Starting evaluation of {len(cases)} cases (Distribution Analysis)...\n")
    start = time.time()

    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['question']} ... ", end="")

        result = answer_question(case["question"])

        # --- 評価ロジック ---
        verdict_ok = result.verification.verdict == case["expected_verdict"]

        must_include_ok = True
        missing_words = []
        for word in case.get("must_include", []):
            if word not in result.answer:
                must_include_ok = False
                missing_words.append(word)

        # Fail Type の判定
        fail_type = detect_fail_type(case, result, verdict_ok, must_include_ok)

        # 改善アクションの取得
        suggested_action = get_suggestion(fail_type) if fail_type else None

        passed = fail_type is None
        status = "PASS" if passed else "FAIL"
        print("✅ PASS" if passed else f"❌ FAIL ({fail_type})")

        # === 集計処理 ===
        by_type_score[case["type"]].append(passed)  # スコア計算用に記録

        if not passed:
            cnt_fail_type[fail_type] += 1
            cnt_q_type[case["type"]] += 1

            if suggested_action:
                cnt_owner[suggested_action["owner"]] += 1
                cnt_priority[suggested_action["priority"]] += 1
            else:
                cnt_owner["unknown"] += 1
                cnt_priority["unknown"] += 1

        results.append(
            {
                "id": case["id"],
                "type": case["type"],
                "question": case["question"],
                "result": status,
                "fail_type": fail_type,
                "suggested_action": suggested_action,
                "rag_verdict": result.verification.verdict,
                "expected_verdict": case["expected_verdict"],
                "answer": result.answer,
                "missing_words": missing_words,
                "citations": [f"{s.doc_id}#{s.chunk_id}" for s in result.sources],
            }
        )

    elapsed = time.time() - start

    # === レポート生成 ===
    total_cases = len(results)
    passed_count = sum(r["result"] == "PASS" for r in results)
    failed_count = total_cases - passed_count

    # 分布オブジェクトの作成
    distribution = {
        "by_fail_type": dict(cnt_fail_type),
        "by_owner": dict(cnt_owner),
        "by_priority": dict(cnt_priority),
        "by_question_type": dict(cnt_q_type),
    }

    # トレンド分析
    trend_hint = generate_trend_hints(distribution, failed_count)

    summary = {
        "total": total_cases,
        "passed": passed_count,
        "failed": failed_count,
        "score": passed_count / total_cases * 100 if total_cases else 0,
        "time_sec": round(elapsed, 2),
    }

    report = {
        "summary": summary,
        "distribution": distribution,
        "trend_hint": trend_hint,
        "details": results,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n==============================")
    print("Distribution Report Generated")
    print(f"Score: {summary['score']:.1f}% ({passed_count}/{total_cases})")

    # 【復活】カテゴリ別スコアの表示
    for t, v in by_type_score.items():
        type_score = sum(v) / len(v) * 100
        print(f"{t}: {type_score:.1f}%")

    print("--- Trend Hints ---")
    for hint in trend_hint:
        print(f"👉 {hint}")
    print(f"\nReport saved to: {REPORT_PATH}")
    print("==============================")


if __name__ == "__main__":
    run_evaluation()
