from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
NonNegativeInt = Annotated[int, Field(ge=0)]
Confidence = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class AgentEvalModel(BaseModel):
    """Base model for strict, explicitly extensible Agent evaluation contracts."""

    model_config = ConfigDict(extra="forbid")

    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentEvalInput(AgentEvalModel):
    """Input supplied to the Agent under evaluation."""

    question: NonEmptyStr


class AgentEvalExpected(AgentEvalModel):
    """Allowed outcomes and answer expectations for one evaluation case."""

    query_types: list[NonEmptyStr] = Field(min_length=1)
    routes: list[NonEmptyStr] = Field(min_length=1)
    tool_calls: list[NonEmptyStr] = Field(default_factory=list)
    citation_required: bool
    answer_assertions: list[NonEmptyStr] = Field(default_factory=list)


class AgentEvalBudgets(AgentEvalModel):
    """Per-run resource limits declared by an evaluation case."""

    max_latency_ms: NonNegativeFloat
    max_cost_usd: NonNegativeFloat


class AgentEvalCase(AgentEvalModel):
    """Target-independent definition of one Agent evaluation scenario."""

    schema_version: NonEmptyStr
    id: NonEmptyStr
    category: NonEmptyStr
    severity: NonEmptyStr
    input: AgentEvalInput
    expected: AgentEvalExpected
    budgets: AgentEvalBudgets
    repeat: Annotated[int, Field(ge=1)] = 1
    tags: list[NonEmptyStr] = Field(default_factory=list)


class AgentRunInput(AgentEvalModel):
    """Input actually observed for an Agent run."""

    question: NonEmptyStr


class AgentRunOutput(AgentEvalModel):
    """Output facts observed from an Agent run."""

    answer: str
    query_type: NonEmptyStr
    route: NonEmptyStr
    confidence: Confidence
    warning_codes: list[NonEmptyStr] = Field(default_factory=list)


class ToolCallTrace(AgentEvalModel):
    """Observed invocation of one tool, without judging whether it was correct."""

    name: NonEmptyStr
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any | None = None
    error: NonEmptyStr | None = None
    duration_ms: NonNegativeFloat | None = None


class SourceTrace(AgentEvalModel):
    """Source exposed to or cited by the Agent during a run."""

    source_id: NonEmptyStr
    title: NonEmptyStr | None = None
    uri: NonEmptyStr | None = None
    snippet: str | None = None
    score: Annotated[float, Field(allow_inf_nan=False)] | None = None


class GuardrailTrace(AgentEvalModel):
    """Observed guardrail activity for a run."""

    triggered: bool = False
    blocked: bool = False
    codes: list[NonEmptyStr] = Field(default_factory=list)
    messages: list[NonEmptyStr] = Field(default_factory=list)


class ControlTrace(AgentEvalModel):
    """Observed execution-control facts that are independent of evaluation."""

    attempt_count: Annotated[int, Field(ge=1)] = 1
    fallback_used: bool = False
    stop_reason: NonEmptyStr | None = None


class UsageTrace(AgentEvalModel):
    """Observed token and monetary usage for a run."""

    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0
    total_tokens: NonNegativeInt = 0
    cost_usd: NonNegativeFloat = 0.0


class TimingTrace(AgentEvalModel):
    """Observed end-to-end and optional tool latency for a run."""

    latency_ms: NonNegativeFloat
    tool_latency_ms: NonNegativeFloat | None = None


class AgentRunTrace(AgentEvalModel):
    """Observed facts from one Agent run; deliberately contains no PASS/FAIL."""

    schema_version: NonEmptyStr
    run_id: NonEmptyStr
    case_id: NonEmptyStr
    target: NonEmptyStr
    input: AgentRunInput
    output: AgentRunOutput
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    sources: list[SourceTrace] = Field(default_factory=list)
    guardrail: GuardrailTrace
    control: ControlTrace
    usage: UsageTrace
    timing: TimingTrace


class CheckResult(AgentEvalModel):
    """Evaluation judgment produced by one named check."""

    schema_version: NonEmptyStr
    check_id: NonEmptyStr
    passed: bool
    score: Annotated[float, Field(allow_inf_nan=False)] | None = None
    message: str | None = None


class CaseEvaluationResult(AgentEvalModel):
    """Evaluation judgments for one run, separate from its observation trace."""

    schema_version: NonEmptyStr
    case_id: NonEmptyStr
    run_id: NonEmptyStr
    passed: bool
    checks: list[CheckResult] = Field(default_factory=list)

