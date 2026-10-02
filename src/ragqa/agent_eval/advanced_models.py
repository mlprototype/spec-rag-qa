from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from ragqa.agent_eval.models import (
    AgentEvalModel,
    Confidence,
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
)


class JudgeRequest(AgentEvalModel):
    """Provider-neutral request sent to an independent evaluation Judge."""

    schema_version: NonEmptyStr
    task: Literal["groundedness", "semantic_stability"]
    prompt_version: NonEmptyStr
    prompt: NonEmptyStr
    input: dict[str, Any]


class ClaimJudgment(AgentEvalModel):
    """Judge decision for one answer claim and its evidence references."""

    claim: NonEmptyStr
    evaluable: bool = True
    supported: bool | None
    source_ids: list[NonEmptyStr] = Field(default_factory=list)
    tool_result_ids: list[NonEmptyStr] = Field(default_factory=list)
    reason: NonEmptyStr

    @model_validator(mode="after")
    def validate_evaluable_state(self) -> ClaimJudgment:
        if self.evaluable and self.supported is None:
            raise ValueError("An evaluable claim requires a supported decision")
        if not self.evaluable and self.supported is not None:
            raise ValueError("A non-evaluable claim must use supported=null")
        if not self.evaluable and (self.source_ids or self.tool_result_ids):
            raise ValueError("A non-evaluable claim must not cite evidence")
        return self


class GroundednessJudgeResponse(AgentEvalModel):
    """Strict structured output expected from a Groundedness Judge."""

    schema_version: NonEmptyStr
    claims: list[ClaimJudgment] = Field(default_factory=list)


class SemanticConsistencyJudgeResponse(AgentEvalModel):
    """Semantic equivalence group assigned to each answer in input order."""

    schema_version: NonEmptyStr
    group_ids: list[NonNegativeInt] = Field(min_length=2)
    reason: NonEmptyStr


class JudgeExecutionMetadata(AgentEvalModel):
    """Audit metadata for one successful Judge invocation."""

    schema_version: NonEmptyStr
    judge_model: NonEmptyStr
    judge_prompt_version: NonEmptyStr
    evaluated_at: datetime
    attempts: Annotated[int, Field(ge=1, le=2)]


class GroundednessJudgeDecision(AgentEvalModel):
    response: GroundednessJudgeResponse
    judge: JudgeExecutionMetadata


class SemanticConsistencyJudgeDecision(AgentEvalModel):
    response: SemanticConsistencyJudgeResponse
    judge: JudgeExecutionMetadata


class GroundednessEvaluationResult(AgentEvalModel):
    """Claim-level Groundedness score for one observed Agent run."""

    schema_version: NonEmptyStr
    case_id: NonEmptyStr
    run_id: NonEmptyStr
    run_index: Annotated[int, Field(ge=1)]
    supported_claims: NonNegativeInt
    evaluable_claims: NonNegativeInt
    score: Confidence | None
    claims: list[ClaimJudgment] = Field(default_factory=list)
    judge: JudgeExecutionMetadata


class StabilityDimensionResult(AgentEvalModel):
    """Mode share and complete agreement for one stability dimension."""

    dimension: NonEmptyStr
    observations: NonNegativeInt
    distinct_values: NonNegativeInt
    mode_share: Confidence | None
    all_match: bool | None
    mode_value: Any | None = None


class AdvancedEvaluationError(AgentEvalModel):
    """Runner or Judge failure that must remain distinct from a quality score."""

    schema_version: NonEmptyStr
    case_id: NonEmptyStr
    run_index: Annotated[int, Field(ge=1)] | None = None
    stage: Literal["runner", "groundedness_judge", "semantic_judge"]
    error_type: NonEmptyStr
    message: str


class StabilityEvaluationResult(AgentEvalModel):
    """Repeat-run stability for one case."""

    schema_version: NonEmptyStr
    case_id: NonEmptyStr
    requested_runs: Annotated[int, Field(ge=1)]
    successful_runs: NonNegativeInt
    execution_error_count: NonNegativeInt
    execution_errors: list[AdvancedEvaluationError] = Field(default_factory=list)
    dimensions: dict[NonEmptyStr, StabilityDimensionResult] = Field(
        default_factory=dict
    )
    semantic_consistency: StabilityDimensionResult | None = None
    semantic_judge: JudgeExecutionMetadata | None = None
    semantic_judge_error: AdvancedEvaluationError | None = None


class ModelTokenPricing(AgentEvalModel):
    input_usd_per_unit: NonNegativeFloat
    output_usd_per_unit: NonNegativeFloat


class ToolCallPricing(AgentEvalModel):
    usd_per_call: NonNegativeFloat


class PricingConfig(AgentEvalModel):
    """Versioned model and Tool pricing table."""

    schema_version: NonEmptyStr
    pricing_version: NonEmptyStr
    currency: Literal["USD"]
    token_unit: Annotated[int, Field(gt=0)]
    models: dict[NonEmptyStr, ModelTokenPricing] = Field(default_factory=dict)
    tools: dict[NonEmptyStr, ToolCallPricing] = Field(default_factory=dict)


class CostEvaluationResult(AgentEvalModel):
    """Estimated run cost; unavailable usage is represented with null amounts."""

    schema_version: NonEmptyStr
    case_id: NonEmptyStr
    run_id: NonEmptyStr
    run_index: Annotated[int, Field(ge=1)]
    model: NonEmptyStr
    pricing_version: NonEmptyStr
    input_tokens: NonNegativeInt | None
    output_tokens: NonNegativeInt | None
    estimated_input_cost_usd: NonNegativeFloat | None
    estimated_output_cost_usd: NonNegativeFloat | None
    estimated_token_cost_usd: NonNegativeFloat | None
    estimated_tool_cost_usd: NonNegativeFloat | None
    estimated_total_cost_usd: NonNegativeFloat | None
    observed_cost_usd: NonNegativeFloat | None
    status: Literal[
        "estimated",
        "usage_unavailable",
        "model_not_priced",
        "tool_not_priced",
    ]


class AdvancedEvaluationResult(AgentEvalModel):
    """Monitor-only Groundedness, Stability, and Cost evaluation batch."""

    schema_version: NonEmptyStr
    generated_at: datetime
    monitor_only: list[NonEmptyStr]
    judge_model: NonEmptyStr
    judge_prompt_versions: dict[NonEmptyStr, NonEmptyStr]
    pricing_version: NonEmptyStr
    groundedness: list[GroundednessEvaluationResult] = Field(default_factory=list)
    stability: list[StabilityEvaluationResult] = Field(default_factory=list)
    costs: list[CostEvaluationResult] = Field(default_factory=list)
    errors: list[AdvancedEvaluationError] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
