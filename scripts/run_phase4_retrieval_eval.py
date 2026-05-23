# -*- coding: utf-8 -*-
"""
このファイルは、仕様書QA RAGにおけるRetrieval（検索）品質の自動測定およびSLO判定を担当します。
全体フローの中では「Layer 2（測定・ゲート層）」に位置し、CI/CD環境やローカルでのPR品質検証で機能します。
主な入力: 期待される検索ソースが定義された ground_truth JSON と、比較基準となる baseline JSON。
主な出力: 検索の精度（Recall@5, MRR, Failure Rate, Latency）を測定・検証したレポート JSON / Markdown。
重要な副作用: SLOを満たさない結果が検出された場合、`sys.exit(1)` を投げて異常終了させ、GitHub PR の自動マージなどをブロックします。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from ragqa.config import cfg
from ragqa.hybrid_retriever import HybridRetriever
from ragqa.retrieval_metrics import compute_retrieval_metrics

# 評価データやベースラインファイルのパスは、テスト規模やフェーズに応じて動的に切り替えられるよう
# 環境変数からの取得を優先し、未設定の場合は expanded（拡張版25ケース）をデフォルトとします。
GROUND_TRUTH_PATH = Path(
    os.getenv("RETRIEVAL_GROUND_TRUTH_PATH", "data/eval/ground_truth_phase0_expanded.json")
)
BASELINE_PATH = Path(
    os.getenv("RETRIEVAL_BASELINE_PATH", "data/eval/phase0_vector_baseline_expanded.json")
)
REPORT_PATH = Path(
    os.getenv("RETRIEVAL_REPORT_PATH", "data/eval/phase4_hybrid_retrieval_report.json")
)


def _parse_expected_doc_id(ref: Any) -> str | None:
    """
    正解ドキュメントの参照表現から doc_id をパースして抽出します。
    
    何を受け取り、何を返すか:
      - 入力 `ref`: 辞書型（例: {"doc_id": "xxx"}）または文字列（例: "xxx#chunk_0"）
      - 出力: 抽出されたドキュメントID（例: "xxx"）または None
    """
    # 辞書で定義されている場合はキーから直接取得します
    if isinstance(ref, dict):
        doc_id = ref.get("doc_id")
        return str(doc_id) if doc_id else None
    # 文字列の場合は 'doc_id#chunk_id' 形式を想定し、アンカー記号 '#' で分割して前部を取得します
    if isinstance(ref, str):
        doc_id, _, _ = ref.partition("#")
        return doc_id or None
    return None


def _expected_sources_from_case(case: dict) -> list[Any]:
    """
    テストケースから期待される正解ソースのリストを安全に取得します。
    旧仕様のキー名(sources)と新仕様のキー名(expected_sources)の両方をサポートします。
    
    何を受け取り、何を返すか:
      - 入力 `case`: テストケースの辞書オブジェクト
      - 出力: 正解ソースの配列
    """
    expected = case.get("expected_sources")
    if expected:
        return list(expected)
    legacy = case.get("sources")
    if legacy:
        return list(legacy)
    return []


def _load_thresholds() -> tuple[float, float, float, str]:
    """
    ベースラインファイルを読み込み、SLO（合格基準）となる閾値を計算して返します。
    ベースラインが存在しない場合は、安全のためにあらかじめ定義された固定の最低ラインにフォールバックします。
    
    何を受け取り、何を返すか:
      - 受け取るもの: なし（BASELINE_PATH を参照）
      - 返すもの: (最小許容Recall@5, 最小許容MRR, 最大許容FailureRate, 閾値モード名称) のタプル
    
    例外吸収を行う理由:
      ベースラインファイルが破損している、または古い形式のままになっている場合に、
      不正確な閾値で品質チェックを通過させてしまうのを防ぐため、明示的に検証を行い、不正なら例外を投げます。
    """
    # なぜここで条件分岐が必要か:
    # 基準となる過去の測定データ（ベースライン）がある場合は、そこからの「相対評価（ラチェット構造）」を適用し、
    # 新規プロジェクトの開始時などでベースラインがない場合は「絶対評価（最低ライン保証）」で判定するため。
    if BASELINE_PATH.exists():
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        summary = baseline.get("summary", {})
        try:
            base_r5 = float(summary["recall_at_5"])
            base_mrr = float(summary["mrr"])
            base_fr = float(summary["failure_rate"])
        except Exception as exc:
            raise RuntimeError(
                f"Invalid baseline schema at {BASELINE_PATH}. Expected summary.recall_at_5/mrr/failure_rate."
            ) from exc

        # 【閾値/マジックナンバーの意味と判断目的】
        # - RETRIEVAL_RECALL5_MIN_RATIO (0.90): Recall@5の性能劣化を基準値の10%以内に抑える（90%以上を維持する）
        # - RETRIEVAL_MRR_MIN_RATIO (0.90): 検索順位の質（MRR）の低下を基準値の10%以内に抑える（90%以上を維持する）
        # - RETRIEVAL_FAILURE_MAX_RATIO (1.20): 検索が全くヒットしない失敗率の増加を基準値の20%増（1.2倍）までに抑える
        min_r5 = base_r5 * float(os.getenv("RETRIEVAL_RECALL5_MIN_RATIO", "0.90"))
        min_mrr = base_mrr * float(os.getenv("RETRIEVAL_MRR_MIN_RATIO", "0.90"))
        max_fr = base_fr * float(os.getenv("RETRIEVAL_FAILURE_MAX_RATIO", "1.20"))
        mode = f"baseline-relative ({BASELINE_PATH})"
    else:
        # 【閾値/マジックナンバーの意味と判断目的】
        # ベースラインがない場合の絶対的な最低ライン（絶対フォールバック）。
        # - Recall@5 (0.60): 最低でも6割以上の確率で正解ドキュメントを上位5件に含める
        # - MRR (0.35): 平均して3件目前後には正解を表示させる品質を保証する
        # - Failure Rate (0.40): 検索失敗率を最大でも4割以下に抑える
        min_r5 = float(os.getenv("RETRIEVAL_MIN_RECALL_AT_5", "0.60"))
        min_mrr = float(os.getenv("RETRIEVAL_MIN_MRR", "0.35"))
        max_fr = float(os.getenv("RETRIEVAL_MAX_FAILURE_RATE", "0.40"))
        mode = "absolute fallback"

    return min_r5, min_mrr, max_fr, mode


def _validate_corpus_alignment(cases: list[dict], retriever: HybridRetriever) -> None:
    """
    テストデータの正解ソースに指定されている doc_id が、検索エンジン側のインデックスに
    すべて登録されているかを検証し、アライメントの不整合を防ぎます。
    
    何を受け取り、何を返すか:
      - 入力 `cases`: テストケース一覧のリスト
      - 入力 `retriever`: 検索を実行する retriever インスタンス
      - 出力: なし（不整合がある場合は sys.exit(1) で処理を終了）
    """
    expected_doc_ids: set[str] = set()
    for case in cases:
        for src in _expected_sources_from_case(case):
            doc_id = _parse_expected_doc_id(src)
            if doc_id:
                expected_doc_ids.add(doc_id)

    indexed_doc_ids = {
        str(m["doc_id"]) for m in retriever.vs.meta if isinstance(m, dict) and "doc_id" in m
    }
    
    # なぜここで差分判定（alignment検証）が必要か:
    # インデックスに登録されていないドキュメントを正解ソース（expected_sources）として指定している場合、
    # 検索エンジンがそれを物理的にヒットさせることができず、評価の測定値（Recall等）が不当に低下します。
    # これはプログラムのバグではなくテスト設定の誤り（インジェストのし忘れなど）であるため、評価を実行する前に検知します。
    missing = sorted(expected_doc_ids - indexed_doc_ids)
    if missing:
        # なぜ 6件 に絞っているか: コンソールのログが大量のファイル名で埋まるのを防ぐための表示上の閾値
        joined = ", ".join(missing[:6])
        if len(missing) > 6:
            joined += ", ..."
        print("ERROR: Ground truth and indexed corpus are misaligned.")
        print(f"Missing expected doc_ids in index ({len(missing)}): {joined}")
        print(
            "Hint: build index with matching docs, e.g. "
            "`RAGQA_DOCS_DIR=data/phase0_expanded/docs python -m ragqa.ingest`."
        )
        sys.exit(1)


def _print_summary(metrics: dict, evaluable: int, total: int) -> None:
    """
    測定された各指標をコンソールへ整形して出力します。
    """
    print()
    print("=" * 56)
    print(f"  Recall@1       {metrics['recall_at_1']:10.4f}")
    print(f"  Recall@5       {metrics['recall_at_5']:10.4f}")
    print(f"  MRR            {metrics['mrr']:10.4f}")
    print(f"  Failure Rate   {metrics['failure_rate']:10.4f}")
    print(f"  P50 Latency    {metrics['p50_latency_ms']:10.4f}")
    print(f"  P95 Latency    {metrics['p95_latency_ms']:10.4f}")
    print(f"  Mean Latency   {metrics['mean_latency_ms']:10.4f}")
    print(f"  Evaluable      {evaluable} / {total}")
    print("=" * 56)


def main() -> None:
    """
    検索エンジンのインデックス状態を確認し、全テストケースに対する検索を実行して、
    測定された品質指標が指定の SLO 閾値をクリアしているかを最終ジャッジします。
    """
    # 検索エンジンの実行に必要なインデックスファイル群が揃っているか確認します。
    for path in [cfg.faiss_index_path, cfg.meta_path, cfg.bm25_index_path, cfg.bm25_postings_path]:
        if not path.exists():
            print(f"ERROR: Index not found: {path}. Run `python -m ragqa.ingest` first.")
            sys.exit(1)

    if not GROUND_TRUTH_PATH.exists():
        print(f"ERROR: ground truth not found: {GROUND_TRUTH_PATH}")
        sys.exit(1)

    cases = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    min_r5, min_mrr, max_fr, mode = _load_thresholds()

    print(f"SLO mode: {mode}")
    print(
        f"Thresholds: Recall@5>={min_r5:.4f}  MRR>={min_mrr:.4f}  FailureRate<={max_fr:.4f}"
    )

    retriever = HybridRetriever.load()
    _validate_corpus_alignment(cases, retriever)

    # 全クエリを検索エンジンに入力し、返却されたドキュメントIDと検索遅延（レイテンシ）を測定します
    details: list[dict] = []
    for case in cases:
        start = time.perf_counter()
        hits = retriever.retrieve(case["question"])
        latency_ms = (time.perf_counter() - start) * 1000.0
        details.append(
            {
                "id": case["id"],
                "citations": [f"{h['doc_id']}#{h['chunk_id']}" for h in hits],
                "latency_ms": round(latency_ms, 3),
            }
        )

    # 測定データを集計して Recall, MRR などの品質指標を算出
    metrics = compute_retrieval_metrics(cases, details)
    evaluable = int(metrics.get("evaluable_cases", 0))
    total = int(metrics.get("total_cases", len(cases)))

    # なぜここで判定が必要か:
    # 正解データ（expected_sources）が1件も定義されていない場合、RecallやMRRの分母が0になり
    # 正常な品質評価が行えません。テストデータの不備として早期に処理を中断させる必要があります。
    if evaluable == 0:
        print("ERROR: evaluable_cases == 0. Check ground_truth expected_sources.")
        sys.exit(1)

    _print_summary(metrics, evaluable, total)

    # CI等で後続のジョブが結果を参照できるよう、結果と閾値を合わせたJSONレポートを出力します
    report = {
        "phase": "phase4_retrieval_gate",
        "ground_truth_path": str(GROUND_TRUTH_PATH),
        "baseline_path": str(BASELINE_PATH),
        "docs_dir": str(cfg.docs_dir),
        "summary": metrics,
        "slo": {
            "mode": mode,
            "min_recall_at_5": min_r5,
            "min_mrr": min_mrr,
            "max_failure_rate": max_fr,
            "recall_at_5_passed": metrics["recall_at_5"] >= min_r5,
            "mrr_passed": metrics["mrr"] >= min_mrr,
            "failure_rate_passed": metrics["failure_rate"] <= max_fr,
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report saved: {REPORT_PATH}")

    # 各指標が閾値をクリアしているかを判定し、未達成のものがあればエラーメッセージを構築します
    failures: list[str] = []
    if not report["slo"]["recall_at_5_passed"]:
        failures.append(f"Recall@5 {metrics['recall_at_5']:.4f} < {min_r5:.4f}")
    if not report["slo"]["mrr_passed"]:
        failures.append(f"MRR {metrics['mrr']:.4f} < {min_mrr:.4f}")
    if not report["slo"]["failure_rate_passed"]:
        failures.append(f"FailureRate {metrics['failure_rate']:.4f} > {max_fr:.4f}")

    # なぜここで failures の有無を判定し、異常終了させる必要があるか:
    # 品質ゲート（品質ラチェット）として機能させるため、SLOに1項目でも違反している場合は
    # プログラムが異常ステータス（sys.exit(1)）で終了し、CI（GitHub Actions等）が失敗扱いとなり、
    # 品質が劣化した変更がPRマージされるのを自動的に防止するため。
    if failures:
        print("SLO FAILED:")
        for msg in failures:
            print(f"  - {msg}")
        sys.exit(1)

    print("SLO PASSED.")


if __name__ == "__main__":
    main()

