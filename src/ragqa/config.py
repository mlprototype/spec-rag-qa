import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    docs_dir: Path = Path(os.getenv("RAGQA_DOCS_DIR") or "data/docs")
    index_dir: Path = Path("data/index")
    faiss_index_path: Path = Path("data/index/faiss.index")
    meta_path: Path = Path("data/index/meta.json")
    bm25_index_path: Path = Path("data/index/bm25_index.jsonl")
    bm25_postings_path: Path = Path("data/index/bm25_postings.jsonl")
    manifest_path: Path = Path("data/index/manifest.json")

    chunk_size: int = 700  # だいたい文字数
    chunk_overlap: int = 120

    top_k: int = 5

    bm25_b: float = 0.75
    bm25_k1: float = 2.0
    vector_candidate_k: int = 15
    bm25_candidate_k: int = 15
    rrf_k: int = 60
    final_top_k: int = 5
    boost_alpha: float = 1.5
    boost_beta: float = 2.0

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
