from __future__ import annotations

from .config import cfg


def _should_wrap_langsmith() -> bool:
    return (
        cfg.enable_langsmith and cfg.langsmith_tracing and bool(cfg.langsmith_api_key)
    )


def _build_openai_client():
    # openai>=1.x
    from openai import OpenAI

    raw_client = OpenAI(api_key=cfg.openai_api_key)
    if not _should_wrap_langsmith():
        return raw_client

    try:
        from langsmith.wrappers import wrap_openai

        return wrap_openai(raw_client)
    except Exception:
        # トレーシング設定不備で本処理を止めない
        return raw_client


def answer_with_openai(prompt: str) -> str:
    client = _build_openai_client()

    resp = client.chat.completions.create(
        model=cfg.openai_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def fallback_answer(contexts: list[str], reason: str) -> str:
    bullets = "\n".join([f"- {c[:240].replace(chr(10), ' ')}..." for c in contexts])
    return f"""（{reason}）

# 検索で見つかった関連箇所
{bullets}

# 次にやると良いこと
- .env に OPENAI_API_KEY を設定し、依存関係を確認すると生成回答を利用できます
"""


def run_llm(prompt: str, contexts: list[str]) -> str:
    if cfg.openai_api_key:
        try:
            return answer_with_openai(prompt)
        except Exception as e:
            return f"(OpenAI呼び出しでエラー: {e})\n\n" + fallback_answer(
                contexts, "OpenAI呼び出しに失敗したため、生成回答の代わりに検索結果を表示します"
            )
    return fallback_answer(
        contexts, "OpenAI APIキーが未設定のため、生成回答の代わりに検索結果を表示します"
    )
