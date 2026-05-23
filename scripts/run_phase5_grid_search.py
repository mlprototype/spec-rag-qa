# -*- coding: utf-8 -*-
"""
このファイルは、仕様書QA RAGのハイブリッド検索における超パラメータ（超引数）空間を探索し、
SLO制約を満たした上で最良となる設定（Best Config）を決定論的に特定する Grid Search エンジンです。
全体フローの中では「Layer 3（最適化・自動調整層）」を担当し、Nightly ビルドや手動で実行されます。
主な入力: 期待される正解ソースが含まれる ground_truth JSON と、基準となる baseline JSON。
主な出力: 探索結果レポート（JSON/Markdown）および特定された最良パラメータ設定の config JSON。
設計上の注意点: 探索空間が指数関数的に増大するのを防ぐため、探索は二段階（Stage 1: kパラメータ系, Stage 2: boostパラメータ系）に分割して段階的に最良候補を絞り込みます。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

from ragqa.retrieval_metrics import compute_retrieval_metrics

# 評価データ、ベースライン、および出力先のパス設定。環境変数がない場合のデフォルトは expanded（拡張版25ケース）を使用します。
DEFAULT_GROUND_TRUTH_PATH = Path(
    os.getenv("RETRIEVAL_GROUND_TRUTH_PATH", "data/eval/ground_truth_phase0_expanded.json")
)
DEFAULT_BASELINE_PATH = Path(
    os.getenv("RETRIEVAL_BASELINE_PATH", "data/eval/phase0_vector_baseline_expanded.json")
)
DEFAULT_REPORT_PATH = Path(
    os.getenv("RETRIEVAL_REPORT_PATH", "data/eval/phase5_grid_search_report.json")
)
DEFAULT_BEST_PATH = Path(os.getenv("RETRIEVAL_BEST_PATH", "data/eval/phase5_best_config.json"))
DEFAULT_REPORT_MD_PATH = Path(
    os.getenv("RETRIEVAL_REPORT_MD_PATH", "data/eval/phase5_grid_search_report.md")
)

# 【閾値/マジックナンバーの意味と判断目的】
# 探索パラメータおよび固定パラメータの設定値：
# - STAGE1_FIXED_BOOST_ALPHA (1.5): Stage 1（k値系探索）で固定する、BM25の一単語完全一致ブースト係数の基準値。
# - STAGE1_FIXED_BOOST_BETA (2.0): Stage 1で固定する、全検索語完全一致ブースト係数の基準値。
# - STAGE1_VECTOR_CANDIDATE_K [10, 15, 20]: Vector検索で取得する候補数。多様性と精度を考慮した探索空間。
# - STAGE1_BM25_CANDIDATE_K [10, 15, 20]: BM25検索で取得する候補数。
# - STAGE1_RRF_K [30, 60, 90]: RRFによる順位融合時の定数。値が大きいほど下位の順位による加点の影響が滑らかになる。
# - STAGE1_FINAL_TOP_K [3, 5, 7]: 最終的にLLMへ渡すコンテキストの最大件数。
# - STAGE2_BOOST_ALPHA [1.2, 1.5, 1.8, 2.0]: Stage 2（ブースト系探索）で走査するα候補。
# - STAGE2_BOOST_BETA [1.5, 2.0, 2.5, 3.0]: Stage 2で走査するβ候補。
STAGE1_FIXED_BOOST_ALPHA = 1.5
STAGE1_FIXED_BOOST_BETA = 2.0
STAGE1_VECTOR_CANDIDATE_K = [10, 15, 20]
STAGE1_BM25_CANDIDATE_K = [10, 15, 20]
STAGE1_RRF_K = [30, 60, 90]
STAGE1_FINAL_TOP_K = [3, 5, 7]
STAGE2_BOOST_ALPHA = [1.2, 1.5, 1.8, 2.0]
STAGE2_BOOST_BETA = [1.5, 2.0, 2.5, 3.0]

# 【閾値/マジックナンバーの意味と判断目的】
# 統計的な改善の有無を判定するための数値基準：
# - EPSILON (1e-6): 浮動小数点演算に伴う丸め誤差を無視して「少しでも値が改善したか」を安全に判定するための閾値。
# - MIN_GAIN (0.01): 1%の有意差フロア。偶然の誤差や微細なノイズによる指標の変動を「改善」と見なすのを防ぎ、
#   実質的な効果がある改善（1%以上）であるかを識別するために使用されます。
EPSILON = 1e-6
MIN_GAIN = 0.01


def _expected_sources_from_case(case: dict) -> list[Any]:
    """
    テストケースから正解ソースの配列を安全に取得します（新旧のJSONキーに対応）。
    
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


def _parse_expected_doc_id(ref: Any) -> str | None:
    """
    正解ドキュメントの参照（辞書型または文字列型）から doc_id をパースして抽出します。
    
    何を受け取り、何を返すか:
      - 入力 `ref`: 正解ソースの参照
      - 出力: ドキュメントID、またはパース不能なら None
    """
    if isinstance(ref, dict):
        doc_id = ref.get("doc_id")
        return str(doc_id) if doc_id else None
    if isinstance(ref, str):
        doc_id, _, _ = ref.partition("#")
        return doc_id or None
    return None


def validate_corpus_alignment(cases: list[dict], meta: list[dict]) -> None:
    """
    テストデータの正解ソースが、現在インジェストされているインデックスのメタデータと一致するか確認します。
    不一致がある場合は、評価が不当に低くなるのを防ぐため、早期に例外を発生させます。
    
    何を受け取り、何を返すか:
      - 入力 `cases`: テストケースのリスト
      - 入力 `meta`: インデックスから読み込まれたメタデータ配列
      - 出力: なし（不整合検出時は RuntimeError を送出）
    """
    expected_doc_ids: set[str] = set()
    for case in cases:
        for src in _expected_sources_from_case(case):
            doc_id = _parse_expected_doc_id(src)
            if doc_id:
                expected_doc_ids.add(doc_id)
    indexed_doc_ids = {
        str(m["doc_id"]) for m in meta if isinstance(m, dict) and "doc_id" in m
    }
    
    # なぜここでアライメント（整合性）判定が必要か:
    # 存在しないドキュメントが正解ソースとしてテストデータにある場合、いかなるパラメータであっても
    # 検索が100%ヒットしないため、測定結果に不当なペナルティが加わります。
    # これは探索アルゴリズムの良し悪しとは無関係な「テストの前提の誤り」であるため、無駄な試行を走らせる前に検知します。
    missing = sorted(expected_doc_ids - indexed_doc_ids)
    if missing:
        # なぜ 6件 に絞っているか: ログがファイル名で溢れかえるのを防ぐための表示上の閾値
        joined = ", ".join(missing[:6])
        if len(missing) > 6:
            joined += ", ..."
        raise RuntimeError(
            "Ground truth and index are misaligned. "
            f"Missing doc_ids in index ({len(missing)}): {joined}"
        )


def build_stage1_params() -> list[dict]:
    """
    第一段階（Stage 1）のパラメータ候補リストを生成します。
    ここでは k パラメータ系（vector_k, bm25_k, rrf_k, final_k）の組み合わせを生成し、ブースト係数は固定します。
    
    何を受け取り、何を返すか:
      - 受け取るもの: なし（定義された定数リストを参照）
      - 返すもの: Stage 1 で試行するパラメータ設定の辞書配列 (81通り)
    """
    params: list[dict] = []
    # 各パラメータ候補の直積（組み合わせ）を決定論的に生成します
    for vector_k, bm25_k, rrf_k, final_k in product(
        STAGE1_VECTOR_CANDIDATE_K,
        STAGE1_BM25_CANDIDATE_K,
        STAGE1_RRF_K,
        STAGE1_FINAL_TOP_K,
    ):
        params.append(
            {
                "vector_candidate_k": vector_k,
                "bm25_candidate_k": bm25_k,
                "rrf_k": rrf_k,
                "final_top_k": final_k,
                "boost_alpha": STAGE1_FIXED_BOOST_ALPHA,
                "boost_beta": STAGE1_FIXED_BOOST_BETA,
            }
        )
    return params


def build_stage2_params(stage1_best: dict) -> list[dict]:
    """
    第二段階（Stage 2）のパラメータ候補リストを生成します。
    Stage 1 の最良の k パラメータ設定を固定したまま、ブースト係数（alpha, beta）の組み合わせを生成します。
    
    何を受け取り、何を返すか:
      - 入力 `stage1_best`: Stage 1 で選ばれた最良パラメータの辞書オブジェクト
      - 出力: Stage 2 で試行するパラメータ設定の辞書配列 (16通り)
    """
    params: list[dict] = []
    for alpha, beta in product(STAGE2_BOOST_ALPHA, STAGE2_BOOST_BETA):
        params.append(
            {
                "vector_candidate_k": int(stage1_best["vector_candidate_k"]),
                "bm25_candidate_k": int(stage1_best["bm25_candidate_k"]),
                "rrf_k": int(stage1_best["rrf_k"]),
                "final_top_k": int(stage1_best["final_top_k"]),
                "boost_alpha": float(alpha),
                "boost_beta": float(beta),
            }
        )
    return params


def build_slo_thresholds(baseline_summary: dict) -> dict:
    """
    ベースラインの成績まとめ（summary）から、合格基準となる各指標の SLO 閾値辞書を算出します。
    
    何を受け取り、何を返すか:
      - 入力 `baseline_summary`: ベースラインファイルのサマリー辞書
      - 出力: {"min_recall_at_5": float, "min_mrr": float, "max_failure_rate": float} の辞書
    """
    # 【閾値/マジックナンバーの意味と判断目的】
    # - RETRIEVAL_RECALL5_MIN_RATIO (0.90): ベースラインと比較して、Recall@5の劣化を10%以内に抑える
    # - RETRIEVAL_MRR_MIN_RATIO (0.90): MRRの劣化を10%以内に抑える
    # - RETRIEVAL_FAILURE_MAX_RATIO (1.20): 検索失敗率（Failure Rate）の増加をベースラインの1.2倍以内に抑える
    ratio_r5 = float(os.getenv("RETRIEVAL_RECALL5_MIN_RATIO", "0.90"))
    ratio_mrr = float(os.getenv("RETRIEVAL_MRR_MIN_RATIO", "0.90"))
    ratio_fr = float(os.getenv("RETRIEVAL_FAILURE_MAX_RATIO", "1.20"))
    return {
        "min_recall_at_5": float(baseline_summary["recall_at_5"]) * ratio_r5,
        "min_mrr": float(baseline_summary["mrr"]) * ratio_mrr,
        "max_failure_rate": float(baseline_summary["failure_rate"]) * ratio_fr,
    }


def is_eligible(metrics: dict, thresholds: dict) -> bool:
    """
    ある試行（trial）で測定された品質指標が、ベースライン基準の SLO 閾値を全て満たしているかチェックします。
    
    何を受け取り、何を返すか:
      - 入力 `metrics`: 測定された品質指標（Recall@5, MRR, Failure Rate）
      - 入力 `thresholds`: 合格ラインとなる閾値辞書
      - 出力: 適合していれば True, 1点でも満たしていない場合は False
    """
    # 全ての指標が合格ラインを同時に満たしていること（AND条件）を判定します。
    return (
        float(metrics["recall_at_5"]) >= float(thresholds["min_recall_at_5"])
        and float(metrics["mrr"]) >= float(thresholds["min_mrr"])
        and float(metrics["failure_rate"]) <= float(thresholds["max_failure_rate"])
    )


def ranking_key(trial: dict) -> tuple:
    """
    合格した試行の中から最良の設定を選出するために、辞書式ソート用のタプルキーを作成します。
    
    何を受け取り、何を返すか:
      - 入力 `trial`: 試行データ（metrics と trial_id を含む）
      - 出力: 優先順に並べた指標値のタプル（値が小さいほどソートで先頭に配置されるよう、最大化したい指標には負符号を付与）
    """
    metrics = trial["metrics"]
    # なぜこの指標順序でソートするのか（システムの優先する価値判断）：
    # 1. Recall@5 (降順): 必要な情報を取りこぼさないことが最重要
    # 2. MRR (降順): できるだけ上位に正解を表示させてノイズを減らす
    # 3. Failure Rate (昇順): 完全な取りこぼしは少ないほど良い
    # 4. p95_latency_ms (昇順): ユーザー体験を守るため遅延の95%値を考慮
    # 5. Recall@1 (降順): 1発命中の精度
    # 6. trial_id (昇順): 指標が全く同じ場合は、より安定した早い試行を優先し、結果を一意に決定論化する
    return (
        -float(metrics["recall_at_5"]),
        -float(metrics["mrr"]),
        float(metrics["failure_rate"]),
        float(metrics["p95_latency_ms"]),
        -float(metrics["recall_at_1"]),
        int(trial["trial_id"]),
    )


def build_improvement(final_metrics: dict | None, baseline_summary: dict) -> dict:
    """
    探索によって決定された最良の設定値が、元のベースラインと比較して
    統計的または実質的な改善（有意差）を果たしたかを算出して辞書としてまとめます。
    
    何を受け取り、何を返すか:
      - 入力 `final_metrics`: 最良設定から得られた品質指標（見つからなかった場合は None）
      - 入力 `baseline_summary`: ベースラインの指標サマリー
      - 出力: 改善ステータスを格納した辞書
    """
    # なぜここで分岐が必要か:
    # SLO制約を満たす設定が1つも見つからなかった（final_metrics が None）場合の
    # ゼロ安全処理（例外防止）と、初期化のため。
    if final_metrics is None:
        final_metrics = {
            "recall_at_5": 0.0,
            "mrr": 0.0,
        }
    baseline_r5 = float(baseline_summary["recall_at_5"])
    baseline_mrr = float(baseline_summary["mrr"])
    final_r5 = float(final_metrics.get("recall_at_5", 0.0))
    final_mrr = float(final_metrics.get("mrr", 0.0))

    # EPSILON (微少差改善) および MIN_GAIN (1%以上の本質的改善) と比較して改善フラグを立てます
    return {
        "epsilon": EPSILON,
        "min_gain": MIN_GAIN,
        "baseline_recall_at_5": baseline_r5,
        "baseline_mrr": baseline_mrr,
        "final_recall_at_5": final_r5,
        "final_mrr": final_mrr,
        "improved_r5": final_r5 > baseline_r5 + EPSILON,
        "meaningful_improved_r5": final_r5 >= baseline_r5 + MIN_GAIN,
        "improved_mrr": final_mrr > baseline_mrr + EPSILON,
        "meaningful_improved_mrr": final_mrr >= baseline_mrr + MIN_GAIN,
    }


def evaluate_trial(retriever: Any, cases: list[dict], params: dict) -> dict:
    """
    指定されたパラメータ設定を用いて、全てのテストケースに対し検索を実行し、
    その設定における品質指標を計測して返します。
    
    何を受け取り、何を返すか:
      - 入力 `retriever`: 検索エンジンインスタンス
      - 入力 `cases`: テストケース一覧のリスト
      - 入力 `params`: 今回試行する検索パラメータの辞書
      - 出力: 計算された各品質指標の辞書
    """
    details: list[dict] = []
    for case in cases:
        case_start = time.perf_counter()
        # 検索パラメータを指定してハイブリッド検索を実行します
        hits = retriever.retrieve(
            case["question"],
            final_top_k=int(params["final_top_k"]),
            vector_candidate_k=int(params["vector_candidate_k"]),
            bm25_candidate_k=int(params["bm25_candidate_k"]),
            rrf_k=int(params["rrf_k"]),
            boost_alpha=float(params["boost_alpha"]),
            boost_beta=float(params["boost_beta"]),
        )
        latency_ms = (time.perf_counter() - case_start) * 1000.0
        details.append(
            {
                "id": case["id"],
                "citations": [f"{h['doc_id']}#{h['chunk_id']}" for h in hits],
                "latency_ms": round(latency_ms, 3),
            }
        )
    return compute_retrieval_metrics(cases, details)


def _run_trials(
    retriever: Any,
    cases: list[dict],
    thresholds: dict,
    params_list: list[dict],
    stage_name: str,
    starting_trial_id: int,
    budget: int | None,
) -> tuple[list[dict], int]:
    """
    パラメータリストの各設定をループで実行し、各試行の判定結果と次の trial_id を返します。
    試行回数予算（budget）が指定されている場合は、途中で走査を打ち切ります。
    
    何を受け取り、何を返すか:
      - 入力 `params_list`: 走査するパラメータ群のリスト
      - 入力 `starting_trial_id`: 今回の開始ID
      - 入力 `budget`: 残りの試行回数上限（未制限の場合は None）
      - 出力: (全試行の履歴リスト, 次の試行に割り当てるべきID値) のタプル
    """
    trials: list[dict] = []
    trial_id = starting_trial_id
    for params in params_list:
        # なぜここで判定が必要か:
        # 探索の上限回数（--max-trials）が指定されている場合、計算コストを抑えるため
        # 割り当てられた予算を使い果たした時点で探索ループを緊急脱出させるため。
        if budget is not None and budget <= 0:
            break
        metrics = evaluate_trial(retriever, cases, params)
        eligible = is_eligible(metrics, thresholds)
        trials.append(
            {
                "trial_id": trial_id,
                "stage": stage_name,
                "params": params,
                "metrics": metrics,
                "eligible": eligible,
            }
        )
        trial_id += 1
        # 予算を1つ消費
        if budget is not None:
            budget -= 1
    return trials, trial_id


def select_best_eligible(trials: list[dict]) -> dict | None:
    """
    合格した（eligible == True）試行の中から、多目的ランキングのソート基準に従って最良の試行を選択します。
    
    何を受け取り、何を返すか:
      - 入力 `trials`: 試行履歴リスト
      - 出力: 最良の試行データ、合格試行がなければ None
    """
    # なぜここで is_eligible で絞り込む必要があるか:
    # SLO制約を満たしていない（RecallやFailureRateが非常に悪い）設定は、
    # どんなに他の指標（例えばレイテンシ）が良くてもシステムとして採用できないため、
    # あらかじめ選考対象から除外（足切り）し、不合格の設定が選ばれるのを防ぎます。
    eligible_trials = [t for t in trials if t["eligible"]]
    if not eligible_trials:
        return None
    eligible_trials.sort(key=ranking_key)
    return eligible_trials[0]


def render_markdown_report(report: dict, topn: int) -> str:
    """
    探索結果レポートを Markdown 形式のテキスト文字列としてレンダリングします。
    """
    lines: list[str] = []
    lines.append("# Phase 5 Grid Search Report")
    lines.append("")
    lines.append(f"- Status: `{report['status']}`")
    lines.append(f"- Ground Truth: `{report['ground_truth_path']}`")
    lines.append(f"- Docs Dir: `{report['docs_dir']}`")
    lines.append(f"- Total Trials: `{report['eligibility']['total_trials']}`")
    lines.append(f"- Eligible Trials: `{report['eligibility']['eligible_trials']}`")
    lines.append("")
    if report["best"] is None:
        lines.append("No eligible trial found under SLO constraints.")
        return "\n".join(lines)

    best = report["best"]
    lines.append("## Best Params")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(best["params"], ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Best Metrics")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(best["metrics"], ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append(f"## Top {topn} Eligible Trials")
    lines.append("")
    lines.append("| trial_id | stage | recall@5 | mrr | failure_rate | p95_ms |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for trial in report["topn"]:
        metrics = trial["metrics"]
        lines.append(
            f"| {trial['trial_id']} | {trial['stage']} | "
            f"{metrics['recall_at_5']:.4f} | {metrics['mrr']:.4f} | "
            f"{metrics['failure_rate']:.4f} | {metrics['p95_latency_ms']:.3f} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """
    コマンドライン引数をパースします。
    """
    parser = argparse.ArgumentParser(description="Phase 5 grid search for hybrid retrieval")
    parser.add_argument("--max-trials", type=int, default=None, help="Max number of trials")
    parser.add_argument("--topn", type=int, default=10, help="Top N results to keep")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_REPORT_MD_PATH)
    parser.add_argument("--output-best", type=Path, default=DEFAULT_BEST_PATH)
    return parser.parse_args()


def _load_baseline_summary(path: Path) -> dict:
    """
    ベースラインファイルを読み込み、品質サマリー部分をチェックして返します。
    破損や必須キーの欠損がある場合はエラーを出します。
    """
    if not path.exists():
        raise FileNotFoundError(f"baseline not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"baseline.summary missing: {path}")
    for key in ("recall_at_5", "mrr", "failure_rate"):
        if key not in summary:
            raise ValueError(f"baseline.summary.{key} missing: {path}")
    return summary


def _get_budget(max_trials: int | None) -> int | None:
    """
    試行回数の上限（バジェット）が正しいかを検証します。
    """
    if max_trials is None:
        return None
    if max_trials <= 0:
        raise ValueError("--max-trials must be > 0")
    return max_trials


def main() -> None:
    """
    インデックス等の存在を確認し、ベースラインに基づくSLOを算出した上で、
    二段階（Stage 1 & Stage 2）のグリッドサーチを実行して最良設定をファイルに出力します。
    """
    args = parse_args()
    budget = _get_budget(args.max_trials)
    topn = max(1, int(args.topn))

    # オプション的な依存モジュールのインポート時における、ユニットテストの独立性維持のための遅延インポート
    from ragqa.config import cfg
    from ragqa.hybrid_retriever import HybridRetriever

    ground_truth_path = Path(
        os.getenv("RETRIEVAL_GROUND_TRUTH_PATH", str(DEFAULT_GROUND_TRUTH_PATH))
    )
    baseline_path = Path(os.getenv("RETRIEVAL_BASELINE_PATH", str(DEFAULT_BASELINE_PATH)))

    # 必要なリソースファイルが存在するか確認
    for path in [
        cfg.faiss_index_path,
        cfg.meta_path,
        cfg.bm25_index_path,
        cfg.bm25_postings_path,
        ground_truth_path,
    ]:
        if not path.exists():
            print(f"ERROR: required file not found: {path}")
            sys.exit(1)

    cases = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    baseline_summary = _load_baseline_summary(baseline_path)
    thresholds = build_slo_thresholds(baseline_summary)

    retriever = HybridRetriever.load()
    validate_corpus_alignment(cases, retriever.vs.meta)

    # ==========================================
    # STAGE 1: kパラメータ系の走査 (81通り)
    # ==========================================
    stage1_params = build_stage1_params()
    stage1_trials, next_trial_id = _run_trials(
        retriever=retriever,
        cases=cases,
        thresholds=thresholds,
        params_list=stage1_params,
        stage_name="stage1",
        starting_trial_id=1,
        budget=budget,
    )

    if budget is not None:
        budget -= len(stage1_trials)

    # Stage 1 の合格候補の中から最良の設定値（Best k-params）を選出します。
    stage1_best = select_best_eligible([t for t in stage1_trials if t["stage"] == "stage1"])
    stage2_trials: list[dict] = []
    
    # なぜここで stage1_best の有無を判定するのか:
    # Stage 1（kパラメータ系）でSLOを満たす候補が1つも見つからなかった場合、
    # その最良設定を固定して行う Stage 2（ブースト系パラメータ探索）は実行不可能であり、意味をなしません。
    # したがって、Stage 1で合格者が出た場合のみ Stage 2 に進むようにします。
    if stage1_best is not None and (budget is None or budget > 0):
        # Stage 1 の最良の k値を固定して、Stage 2 のブースト係数パラメータ空間(16通り)を走査します。
        stage2_params = build_stage2_params(stage1_best["params"])
        stage2_trials, next_trial_id = _run_trials(
            retriever=retriever,
            cases=cases,
            thresholds=thresholds,
            params_list=stage2_params,
            stage_name="stage2",
            starting_trial_id=next_trial_id,
            budget=budget,
        )

    # Stage 1 と Stage 2 の全試行データをマージし、総合評価します。
    all_trials = stage1_trials + stage2_trials
    eligible_trials = [t for t in all_trials if t["eligible"]]
    eligible_trials.sort(key=ranking_key)
    best = eligible_trials[0] if eligible_trials else None
    top_trials = eligible_trials[:topn]

    final_metrics = best["metrics"] if best else None
    improvement = build_improvement(final_metrics, baseline_summary)
    status = "ok" if best is not None else "no_eligible"

    # 詳細な試行データと結果をまとめたJSONレポートオブジェクトを構築
    report = {
        "phase": "phase5_grid_search",
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "status": status,
        "ground_truth_path": str(ground_truth_path),
        "docs_dir": str(cfg.docs_dir),
        "baseline_path": str(baseline_path),
        "baseline": {
            "recall_at_5": float(baseline_summary["recall_at_5"]),
            "mrr": float(baseline_summary["mrr"]),
            "failure_rate": float(baseline_summary["failure_rate"]),
        },
        "search_space": {
            "stage1": {
                "vector_candidate_k": STAGE1_VECTOR_CANDIDATE_K,
                "bm25_candidate_k": STAGE1_BM25_CANDIDATE_K,
                "rrf_k": STAGE1_RRF_K,
                "final_top_k": STAGE1_FINAL_TOP_K,
                "boost_alpha_fixed": STAGE1_FIXED_BOOST_ALPHA,
                "boost_beta_fixed": STAGE1_FIXED_BOOST_BETA,
            },
            "stage2": {
                "boost_alpha": STAGE2_BOOST_ALPHA,
                "boost_beta": STAGE2_BOOST_BETA,
            },
        },
        "slo": thresholds,
        "eligibility": {
            "eligible_trials": len(eligible_trials),
            "total_trials": len(all_trials),
        },
        "improvement": improvement,
        "trials": all_trials,
        "best": best,
        "topn": top_trials,
    }

    # 各種ファイルへの保存処理
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )

    # 外部のサービス（HybridRetrieverのロード等）がそのまま設定値を読み込めるよう、最良パラメータのみの config を保存します。
    best_payload = (
        {
            "trial_id": best["trial_id"],
            "stage": best["stage"],
            "params": best["params"],
            "metrics": best["metrics"],
        }
        if best
        else {"status": "no_eligible", "best": None}
    )
    args.output_best.parent.mkdir(parents=True, exist_ok=True)
    args.output_best.write_text(
        json.dumps(best_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown_report(report, topn), encoding="utf-8")

    print(f"Report JSON: {args.output_json}")
    print(f"Best Config: {args.output_best}")
    print(f"Report MD:   {args.output_md}")
    print(f"Status: {status}")

    # 最良設定が見つからなかった場合は異常ステータスで終了させます。
    if best is None:
        print("No eligible trial under SLO constraints.")
        sys.exit(1)

    metrics = best["metrics"]
    print(
        "Best metrics: "
        f"recall@5={metrics['recall_at_5']:.4f}, "
        f"mrr={metrics['mrr']:.4f}, "
        f"failure_rate={metrics['failure_rate']:.4f}, "
        f"p95={metrics['p95_latency_ms']:.3f}ms"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
