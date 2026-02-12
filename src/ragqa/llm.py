from __future__ import annotations

from langsmith.wrappers import wrap_openai

from .config import cfg


def answer_with_openai(prompt: str) -> str:
    # openai>=1.x
    from openai import OpenAI

    # クライアント作成時に wrap_openai で包むことで、LangSmithにトレースできるようにする。
    raw_client = OpenAI(api_key=cfg.openai_api_key)
    client = wrap_openai(raw_client)

    resp = client.chat.completions.create(
        model=cfg.openai_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def fallback_answer(prompt: str, contexts: list[str]) -> str:
    # キー無しでも動く：検索結果を返す
    bullets = "\n".join([f"- {c[:240].replace(chr(10), ' ')}..." for c in contexts])
    return f"""（APIキーは設定されていますが、SDK初期化に失敗しました、生成回答の代わりに検索結果を表示します）

# 検索で見つかった関連箇所
{bullets}

# 次にやると良いこと
- .env に OPENAI_API_KEY を設定すると、コンテキストを根拠に回答を生成します
"""


def run_llm(prompt: str, contexts: list[str]) -> str:
    if cfg.openai_api_key:
        try:
            return answer_with_openai(prompt)
        except Exception as e:
            return f"(OpenAI呼び出しでエラー: {e})\n\n" + fallback_answer(
                prompt, contexts
            )
    return fallback_answer(prompt, contexts)
