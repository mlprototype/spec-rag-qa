from typing import List

from pydantic import BaseModel, Field


class Source(BaseModel):
    """回答の根拠となるドキュメントチャンク"""

    doc_id: str
    chunk_id: int
    text: str
    score: float


class Verification(BaseModel):
    """LLMによる回答精度の検証結果"""

    verdict: str  # "sufficient" | "insufficient"
    confidence: float = 0.0
    missing_points: List[str] = Field(default_factory=list)
    supported_claims: List[str] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)


class AnswerResult(BaseModel):
    """RAGの実行結果全体を包むコンテナ"""

    question: str
    answer: str
    verification: Verification
    sources: List[Source]
