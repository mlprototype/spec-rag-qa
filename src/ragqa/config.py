import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    docs_dir: Path = Path("data/docs")
    index_dir: Path = Path("data/index")
    faiss_index_path: Path = Path("data/index/faiss.index")
    meta_path: Path = Path("data/index/meta.json")

    chunk_size: int = 700  # だいたい文字数
    chunk_overlap: int = 120

    top_k: int = 5

    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_model: str = os.getenv("OPENAI_MODEL") or "gpt-3.5-turbo"
    enable_langsmith: bool = _is_truthy(os.getenv("RAGQA_ENABLE_LANGSMITH"))
    langsmith_tracing: bool = _is_truthy(
        os.getenv("LANGCHAIN_TRACING_V2") or os.getenv("LANGSMITH_TRACING")
    )
    langsmith_api_key: str | None = (
        os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY") or None
    )


cfg = Config()
