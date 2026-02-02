from typing import List, Literal, Optional

from pydantic import BaseModel


class EvaluationResult(BaseModel):
    verdict: Literal["PASS", "FAIL"]
    fail_type: Optional[str] = None
    reasons: List[str] = []
