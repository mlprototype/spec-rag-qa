from typing import List, Optional

from pydantic import BaseModel


class AnswerResult(BaseModel):
    answer: str
    retrieved_chunks: List[str]
    citations: List[str] = []
    confidence: Optional[float] = None
