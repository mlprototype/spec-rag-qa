import csv
import datetime
import json
import os
import sys
import time
from collections import Counter, defaultdict

# serviceからロジックを呼ぶ
from ragqa.service import answer_question

# OpenAIクライアント用
from ragqa.config import cfg

# 改善アクションカタログをインポート
from ragqa.improvement_catalog import get_suggestion
from ragqa.retrieval_metrics import compute_retrieval_metrics

# Default keeps current low-cost behavior (5-case set).
# For expanded benchmark in CI/local, set:
# RAGQA_EVAL_GROUND_TRUTH_PATH=data/eval/ground_truth_phase0_expanded.json
GROUND_TRUTH_PATH = os.getenv("RAGQA_EVAL_GROUND_TRUTH_PATH", "data/eval/ground_truth.json")
REPORT_PATH = "data/eval/report.json"
TREND_CSV_PATH = "data/eval/trend.csv"


def check_assertion_with_llm(question: str, answer: str, assertion: str) -> bool:
    """
    回答が指定された判定基準（assertion）を満たしているか、LLMにジャッジさせる
    """
    from openai import OpenAI

    # 設定からAPIキーを読み込む
    client = OpenAI(api_key=cfg.openai_api_key)

    prompt = f"""
    あなたは公平な評価者です。以下のRAGシステムの回答が、判定基準を満たしているか判定してください。

    # 質問
    {question}

    # システムの回答
    {answer}

    # 判定基準
    {assertion}

    # 指示
    判定基準を満たしていれば "true"、満たしていなければ "false" とだけ出力してください。
    理由や説明は一切不要です。
    """

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",  # 評価能力が高いモデル推奨
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        result = resp.choices[0].message.content.strip().lower()
        return "true" in result
    except Exception as e:
        print(f"Warning: LLM assertion check failed: {e}")
        return False


def append_trend_csv(report: dict):
    """
    評価結果を時系列CSVに追記する（DB代わり）
    Lv5: Actionable Trend 対応
    """
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = report["summary"]
    dist = report["distribution"]
    run_id = int(time.time())

    top_fail_type = "None"
    top_owner = "None"
    recommended_action = "None"

    if summary["failed"] > 0:
        if dist["by_fail_type"]:
            top_fail_type, _ = max(dist["by_fail_type"].items(), key=lambda x: x[1])
            suggestion = get_suggestion(top_fail_type)
            if suggestion:
                recommended_action = suggestion["example_fix"].replace("\n", " / ")
            else:
                recommended_action = "要調査（カタログ未定義）"

        if dist["by_owner"]:
            top_owner, _ = max(dist["by_owner"].items(), key=lambda x: x[1])
    else:
        recommended_action = "All Green! 素晴らしい状態です"

    row = {
        "timestamp": now_str,
        "run_id": run_id,
        "score": summary["score"],
        "fail_rate": summary["failed"] / summary["total"]
        if summary["total"] > 0
        else 0,
        "critical_count": dist["by_priority"].get("CRITICAL", 0),
        "fail_factual_basic": dist["by_question_type"].get("factual_basic", 0),
        "fail_omission_detection": dist["by_question_type"].get(
            "omission_detection", 0
        ),
        "fail_priority_conflict": dist["by_question_type"].get("priority_conflict", 0),
        "fail_opinion_guard": dist["by_question_type"].get("opinion_guard", 0),
        "owner_spec": dist["by_owner"].get("spec", 0),
        "owner_retrieval": dist["by_owner"].get("rag", 0),
        "owner_prompt": dist["by_owner"].get("prompt", 0),
        "owner_system": dist["by_owner"].get("system", 0),
        "top_fail_type": top_fail_type,
        "top_owner": top_owner,
        "recommended_action": recommended_action,
    }

    os.makedirs(os.path.dirname(TREND_CSV_PATH), exist_ok=True)
    file_exists = os.path.exists(TREND_CSV_PATH)

    fieldnames = [
        "timestamp",
        "run_id",
        "score",
        "fail_rate",
        "critical_count",
        "fail_factual_basic",
        "fail_omission_detection",
        "fail_priority_conflict",
        "fail_opinion_guard",
        "owner_spec",
        "owner_retrieval",
        "owner_prompt",
        "owner_system",
        "top_fail_type",
        "top_owner",
        "recommended_action",
    ]

    with open(TREND_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"📈 Trend data appended to: {TREND_CSV_PATH}")


def detect_fail_type(
    case: dict, result, verdict_ok: bool, assertion_ok: bool
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

    # 3. アサーション（意味判定）NG
    if not assertion_ok:
        if case["type"] == "omission_detection":
            return "OMISSION (Critical Condition Missing)"
        if case["type"] == "priority_conflict":
            return "PRIORITY_ERROR (Wrong Rule Applied)"
        if case["type"] == "opinion_guard":
            return "OPINION_LEAK (Subjective)"
        # デフォルトの意味不一致
        return "ASSERTION_FAILED (Semantic Mismatch)"

    return None  # PASS


def generate_trend_hints(distribution: dict, total_fail: int) -> list[str]:
    hints = []
    if total_fail == 0:
        return ["FAILはありません。素晴らしい状態です！"]

    crit_count = distribution["by_priority"].get("CRITICAL", 0)
    if crit_count > 0:
        hints.append(
            f"CRITICALなFAILが {crit_count}件 あります。これらは即時修正が必要です。"
        )

    by_owner = distribution["by_owner"]
    if by_owner:
        top_owner, count = max(by_owner.items(), key=lambda x: x[1])
        if count >= total_fail * 0.5:
            hints.append(
                f"{top_owner} 起因のFAILが全体の {count / total_fail * 100:.0f}% を占めています。ここの改善が効果的です。"
            )

    return hints


def run_evaluation():
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        cases = json.load(f)

    results = []
    cnt_fail_type = Counter()
    cnt_owner = Counter()
    cnt_priority = Counter()
    cnt_q_type = Counter()
    by_type_score = defaultdict(list)

    print(f"Starting AI-Judge evaluation of {len(cases)} cases...\n")
    start = time.time()

    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['question']} ... ", end="", flush=True)

        # RAG実行 (per-case latency)
        case_start = time.time()
        result = answer_question(case["question"])
        latency_ms = (time.time() - case_start) * 1000.0

        # --- 評価ロジック ---
        # 1. 自己評価Verdictの一致チェック
        verdict_ok = result.verification.verdict == case["expected_verdict"]

        # 2. AIによる意味判定 (Assertion Check)
        assertion_ok = True
        assertion_msg = "PASS"
        if "assertion" in case:
            # LLMにジャッジさせる
            if not check_assertion_with_llm(
                case["question"], result.answer, case["assertion"]
            ):
                assertion_ok = False
                assertion_msg = f"Failed assertion: {case['assertion']}"

        # 判定
        fail_type = detect_fail_type(case, result, verdict_ok, assertion_ok)
        suggested_action = get_suggestion(fail_type) if fail_type else None

        passed = fail_type is None
        status = "PASS" if passed else "FAIL"

        # 結果表示
        if passed:
            print("✅ PASS")
        else:
            print(f"❌ FAIL ({fail_type})")

        # === 集計処理 ===
        by_type_score[case["type"]].append(passed)

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
                "rag_verdict": result.verification.verdict,
                "expected_verdict": case["expected_verdict"],
                "answer": result.answer,
                "assertion_result": assertion_msg,
                "citations": [f"{s.doc_id}#{s.chunk_id}" for s in result.sources],
                "latency_ms": round(latency_ms, 3),
            }
        )

    elapsed = time.time() - start

    # === レポート生成 ===
    total_cases = len(results)
    passed_count = sum(r["result"] == "PASS" for r in results)
    failed_count = total_cases - passed_count

    distribution = {
        "by_fail_type": dict(cnt_fail_type),
        "by_owner": dict(cnt_owner),
        "by_priority": dict(cnt_priority),
        "by_question_type": dict(cnt_q_type),
    }

    trend_hint = generate_trend_hints(distribution, failed_count)

    summary = {
        "total": total_cases,
        "passed": passed_count,
        "failed": failed_count,
        "score": passed_count / total_cases * 100 if total_cases else 0,
        "time_sec": round(elapsed, 2),
    }
    retrieval = compute_retrieval_metrics(cases, results)

    report = {
        "summary": summary,
        "retrieval": retrieval,
        "distribution": distribution,
        "trend_hint": trend_hint,
        "details": results,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # CSV追記
    append_trend_csv(report)

    print("\n==============================")
    print("AI-Judge Evaluation Report Generated")
    print(f"Score: {summary['score']:.1f}% ({passed_count}/{total_cases})")
    print(f"Time: {summary['time_sec']}s")
    print(
        "Retrieval: "
        f"Recall@1={retrieval['recall_at_1']:.4f} "
        f"Recall@5={retrieval['recall_at_5']:.4f} "
        f"MRR={retrieval['mrr']:.4f} "
        f"FailureRate={retrieval['failure_rate']:.4f} "
        f"(evaluable={retrieval['evaluable_cases']}/{retrieval['total_cases']})"
    )

    for t, v in by_type_score.items():
        type_score = sum(v) / len(v) * 100
        print(f"{t}: {type_score:.1f}%")

    print(f"\nReport saved to: {REPORT_PATH}")
    print("==============================")


if __name__ == "__main__":
    run_evaluation()

    # Quality Gate
    try:
        with open(REPORT_PATH, encoding="utf-8") as f:
            report = json.load(f)

        if report["summary"]["failed"] > 0:
            print("\n🚫 Quality Gate Failed: FAIL detected.")
            sys.exit(1)

        if report["summary"]["score"] < 95.0:
            print(
                f"\n🚫 Quality Gate Failed: Score {report['summary']['score']} is below 95.0."
            )
            sys.exit(1)

        print("\n✅ Quality Gate Passed.")
        sys.exit(0)
    except Exception as e:
        print(f"\n🚫 System Error during Quality Gate check: {e}")
        sys.exit(1)
