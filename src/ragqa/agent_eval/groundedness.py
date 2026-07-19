from __future__ import annotations

from ragqa.agent_eval.advanced_models import GroundednessEvaluationResult
from ragqa.agent_eval.judge import JudgeAdapter
from ragqa.agent_eval.models import AgentRunTrace


async def evaluate_groundedness(
    trace: AgentRunTrace,
    judge: JudgeAdapter,
    *,
    run_index: int = 1,
) -> GroundednessEvaluationResult:
    """Compute supported/evaluable claims without consulting Agent self-evaluation."""

    decision = await judge.judge_groundedness(trace)
    evaluable_claims = sum(claim.evaluable for claim in decision.response.claims)
    supported_claims = sum(
        claim.evaluable and claim.supported is True
        for claim in decision.response.claims
    )
    score = (
        round(supported_claims / evaluable_claims, 6)
        if evaluable_claims
        else None
    )
    return GroundednessEvaluationResult(
        schema_version=trace.schema_version,
        case_id=trace.case_id,
        run_id=trace.run_id,
        run_index=run_index,
        supported_claims=supported_claims,
        evaluable_claims=evaluable_claims,
        score=score,
        claims=decision.response.claims,
        judge=decision.judge,
    )
