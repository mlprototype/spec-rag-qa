from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from statistics import fmean

from ragqa.agent_eval.advanced_models import (
    AdvancedEvaluationError,
    AdvancedEvaluationResult,
    CostEvaluationResult,
    GroundednessEvaluationResult,
    PricingConfig,
    StabilityDimensionResult,
    StabilityEvaluationResult,
)
from ragqa.agent_eval.cost import estimate_cost
from ragqa.agent_eval.groundedness import evaluate_groundedness
from ragqa.agent_eval.judge import JudgeAdapter, JudgeError
from ragqa.agent_eval.models import AgentEvalCase, AgentRunTrace
from ragqa.agent_eval.runner import (
    AgentRunner,
    FixtureTraceMismatchError,
    SchemaVersionMismatchError,
)
from ragqa.agent_eval.stability import evaluate_stability


async def run_advanced_evaluation(
    cases: Sequence[AgentEvalCase],
    runner: AgentRunner,
    judge: JudgeAdapter,
    pricing: PricingConfig,
) -> AdvancedEvaluationResult:
    """Run monitor-only Groundedness, Stability, and Cost evaluation."""

    schema_versions = {case.schema_version for case in cases}
    if len(schema_versions) > 1:
        raise SchemaVersionMismatchError(
            f"Mixed case schema versions: {sorted(schema_versions)}"
        )

    groundedness_results: list[GroundednessEvaluationResult] = []
    stability_results: list[StabilityEvaluationResult] = []
    cost_results: list[CostEvaluationResult] = []
    errors: list[AdvancedEvaluationError] = []

    for case in cases:
        indexed_traces: list[tuple[int, AgentRunTrace]] = []
        runner_errors: list[AdvancedEvaluationError] = []
        for run_index in range(1, case.repeat + 1):
            try:
                trace = await runner.run(case)
                _validate_trace(case, trace)
            except Exception as exc:
                error = AdvancedEvaluationError(
                    schema_version=case.schema_version,
                    case_id=case.id,
                    run_index=run_index,
                    stage="runner",
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
                runner_errors.append(error)
                errors.append(error)
                continue

            indexed_traces.append((run_index, trace))
            cost_results.append(
                estimate_cost(trace, pricing, run_index=run_index)
            )
            try:
                groundedness_results.append(
                    await evaluate_groundedness(
                        trace,
                        judge,
                        run_index=run_index,
                    )
                )
            except JudgeError as exc:
                errors.append(
                    AdvancedEvaluationError(
                        schema_version=case.schema_version,
                        case_id=case.id,
                        run_index=run_index,
                        stage="groundedness_judge",
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )

        stability = await evaluate_stability(
            case,
            indexed_traces,
            runner_errors,
            judge,
        )
        stability_results.append(stability)
        if stability.semantic_judge_error is not None:
            errors.append(stability.semantic_judge_error)

    summary = _build_summary(
        cases,
        groundedness_results,
        stability_results,
        cost_results,
        errors,
    )
    schema_version = next(iter(schema_versions), "1.0")
    return AdvancedEvaluationResult(
        schema_version=schema_version,
        generated_at=datetime.now(timezone.utc),
        monitor_only=["groundedness", "answer_semantic_consistency", "cost"],
        judge_model=judge.judge_model,
        judge_prompt_versions={
            "groundedness": judge.grounding_prompt_version,
            "answer_semantic_consistency": judge.semantic_prompt_version,
        },
        pricing_version=pricing.pricing_version,
        groundedness=groundedness_results,
        stability=stability_results,
        costs=cost_results,
        errors=errors,
        summary=summary,
    )


def _validate_trace(case: AgentEvalCase, trace: AgentRunTrace) -> None:
    if trace.case_id != case.id or trace.input.question != case.input.question:
        raise FixtureTraceMismatchError(
            f"Repeated trace does not match case id: {case.id}"
        )
    if trace.schema_version != case.schema_version:
        raise SchemaVersionMismatchError(
            f"Repeated trace schema mismatch for case id: {case.id}"
        )


def _build_summary(
    cases: Sequence[AgentEvalCase],
    groundedness: Sequence[GroundednessEvaluationResult],
    stability: Sequence[StabilityEvaluationResult],
    costs: Sequence[CostEvaluationResult],
    errors: Sequence[AdvancedEvaluationError],
) -> dict[str, object]:
    supported_claims = sum(item.supported_claims for item in groundedness)
    evaluable_claims = sum(item.evaluable_claims for item in groundedness)
    groundedness_score = (
        round(supported_claims / evaluable_claims, 6)
        if evaluable_claims
        else None
    )

    stability_dimensions: dict[str, dict[str, object]] = {}
    dimension_names = {
        name
        for result in stability
        for name in result.dimensions
    }
    for dimension_name in sorted(dimension_names):
        dimensions = [
            result.dimensions[dimension_name]
            for result in stability
            if dimension_name in result.dimensions
        ]
        stability_dimensions[dimension_name] = _aggregate_dimensions(dimensions)
    semantic_dimensions = [
        result.semantic_consistency
        for result in stability
        if result.semantic_consistency is not None
    ]
    stability_dimensions["answer_semantic"] = _aggregate_dimensions(
        semantic_dimensions
    )

    known_costs = [
        item.estimated_total_cost_usd
        for item in costs
        if item.estimated_total_cost_usd is not None
    ]
    complete_cost = bool(costs) and len(known_costs) == len(costs)
    requested_runs = sum(case.repeat for case in cases)
    runner_error_count = sum(error.stage == "runner" for error in errors)
    judge_error_count = sum(error.stage != "runner" for error in errors)
    return {
        "runs": {
            "requested": requested_runs,
            "completed": len(costs),
            "execution_errors": runner_error_count,
            "judge_errors": judge_error_count,
        },
        "groundedness": {
            "supported_claims": supported_claims,
            "evaluable_claims": evaluable_claims,
            "score": groundedness_score,
            "evaluated_runs": len(groundedness),
        },
        "stability": stability_dimensions,
        "cost": {
            "evaluated_runs": len(known_costs),
            "unavailable_runs": len(costs) - len(known_costs),
            "coverage": round(len(known_costs) / len(costs), 6) if costs else None,
            "known_cost_usd": round(sum(known_costs), 12) if known_costs else None,
            "estimated_total_cost_usd": (
                round(sum(known_costs), 12) if complete_cost else None
            ),
        },
    }


def _aggregate_dimensions(
    dimensions: Sequence[StabilityDimensionResult],
) -> dict[str, object]:
    mode_shares = [
        dimension.mode_share
        for dimension in dimensions
        if dimension.mode_share is not None
    ]
    all_matches = [
        dimension.all_match
        for dimension in dimensions
        if dimension.all_match is not None
    ]
    return {
        "cases": len(mode_shares),
        "average_mode_share": (
            round(fmean(mode_shares), 6) if mode_shares else None
        ),
        "all_match_cases": sum(value is True for value in all_matches),
        "all_match_denominator": len(all_matches),
        "all_match_rate": (
            round(sum(value is True for value in all_matches) / len(all_matches), 6)
            if all_matches
            else None
        ),
    }
