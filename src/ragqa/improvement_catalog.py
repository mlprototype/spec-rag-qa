from typing import Dict, Optional

# evaluate.py で定義した fail_type 文字列をキーにする
FAIL_IMPROVEMENT_CATALOG: Dict[str, dict] = {
    "EMPTY_ANSWER": {
        "owner": "system",
        "priority": "CRITICAL",
        "message": "回答が生成されていません。システムエラーまたはフィルタリング過剰の可能性があります。",
        "example_fix": "エラーログ確認 / リトライ処理の実装",
    },
    "RETRIEVAL_FAILURE (Evidence Missing)": {
        "owner": "rag",
        "priority": "HIGH",
        "message": "関連ドキュメントが検索できていません。チャンク設計またはメタデータを見直してください。",
        "example_fix": "・Chunkサイズの拡大\n・キーワード検索(Hybrid Search)の併用",
    },
    "HALLUCINATION / OVERCONFIDENCE": {
        "owner": "prompt",
        "priority": "HIGH",
        "message": "根拠がないのに自信満々に回答しています（幻覚）。",
        "example_fix": "プロンプトに『根拠がない場合は不明と答えよ』と強調する",
    },
    "VERDICT_MISMATCH": {
        "owner": "prompt",
        "priority": "MEDIUM",
        "message": "RAGの自己評価と期待値がズレています。",
        "example_fix": "Verifier（検証用プロンプト）の判定基準を見直す",
    },
    "OMISSION (Critical Condition Missing)": {
        "owner": "spec",
        "priority": "HIGH",
        "message": "仕様書に重要条件（例外・制約）の記載が漏れているか、検索できていません。",
        "example_fix": "## 例外条件\n- メール重複時\n- パスワード条件未満\nを仕様書に追記する",
    },
    "PRIORITY_ERROR (Wrong Rule Applied)": {
        "owner": "spec",
        "priority": "MEDIUM",
        "message": "仕様間の優先順位が不明確です。",
        "example_fix": "ドキュメント冒頭に『本仕様は設計書よりも優先される』と明記",
    },
    "OPINION_LEAK (Subjective)": {
        "owner": "prompt",
        "priority": "CRITICAL",
        "message": "AIが主観的判断（使いやすい等）を行っています。越権行為です。",
        "example_fix": "System Promptに『客観的事実のみを述べよ』と制約を追加",
    },
    "FACTUAL_ERROR (Keyword Missing)": {
        "owner": "rag",
        "priority": "HIGH",
        "message": "必須キーワードが含まれていません。",
        "example_fix": "類義語辞書の追加 / 検索クエリの拡張",
    },
}


def get_suggestion(fail_type: str) -> Optional[dict]:
    """fail_type に対応する改善アクションを返す"""
    return FAIL_IMPROVEMENT_CATALOG.get(fail_type)
