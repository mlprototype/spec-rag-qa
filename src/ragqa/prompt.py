def build_prompt(question: str, contexts: list[str]) -> str:
    joined = "\n\n---\n\n".join(contexts)
    return f"""あなたは業務仕様・設計書のQAアシスタントです。
与えられたコンテキスト（仕様/設計）だけを根拠に、曖昧なら曖昧と明言し、追加で確認すべき点を箇条書きで出してください。

# 質問
{question}

# コンテキスト
{joined}

# 出力フォーマット
- 結論（短く）
- 根拠（どの記述に基づくか）
- 不明点・確認事項（必要なら）
"""
