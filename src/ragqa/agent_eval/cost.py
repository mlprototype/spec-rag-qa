from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from ragqa.agent_eval.advanced_models import (
    CostEvaluationResult,
    PricingConfig,
)
from ragqa.agent_eval.models import AgentRunTrace


class PricingConfigError(ValueError):
    """Raised when a versioned pricing table cannot be loaded."""


def load_pricing_config(path: str | Path) -> PricingConfig:
    pricing_path = Path(path)
    try:
        with pricing_path.open(encoding="utf-8") as file:
            payload = json.load(file)
        return PricingConfig.model_validate(payload)
    except OSError as exc:
        raise PricingConfigError(
            f"Pricing config could not be read: {pricing_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise PricingConfigError(
            f"Pricing config is invalid JSON: {pricing_path}"
        ) from exc
    except ValidationError as exc:
        raise PricingConfigError(
            f"Pricing config does not match its schema: {pricing_path}"
        ) from exc


def estimate_cost(
    trace: AgentRunTrace,
    pricing: PricingConfig,
    *,
    run_index: int = 1,
) -> CostEvaluationResult:
    """Estimate model and Tool cost while preserving unavailable usage as N/A."""

    model = _resolve_model(trace)
    usage = trace.usage
    observed_cost = usage.cost_usd if usage is not None else None
    input_tokens = usage.input_tokens if usage is not None else None
    output_tokens = usage.output_tokens if usage is not None else None
    common = {
        "schema_version": trace.schema_version,
        "case_id": trace.case_id,
        "run_id": trace.run_id,
        "run_index": run_index,
        "model": model,
        "pricing_version": pricing.pricing_version,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "observed_cost_usd": observed_cost,
    }

    if input_tokens is None or output_tokens is None:
        return CostEvaluationResult(
            **common,
            estimated_input_cost_usd=None,
            estimated_output_cost_usd=None,
            estimated_token_cost_usd=None,
            estimated_tool_cost_usd=None,
            estimated_total_cost_usd=None,
            status="usage_unavailable",
        )

    model_pricing = pricing.models.get(model)
    if model_pricing is None:
        return CostEvaluationResult(
            **common,
            estimated_input_cost_usd=None,
            estimated_output_cost_usd=None,
            estimated_token_cost_usd=None,
            estimated_tool_cost_usd=None,
            estimated_total_cost_usd=None,
            status="model_not_priced",
        )

    input_cost = _round_cost(
        input_tokens / pricing.token_unit * model_pricing.input_usd_per_unit
    )
    output_cost = _round_cost(
        output_tokens / pricing.token_unit * model_pricing.output_usd_per_unit
    )
    token_cost = _round_cost(input_cost + output_cost)

    missing_tool_price = any(
        call.name not in pricing.tools for call in trace.tool_calls
    )
    if missing_tool_price:
        return CostEvaluationResult(
            **common,
            estimated_input_cost_usd=input_cost,
            estimated_output_cost_usd=output_cost,
            estimated_token_cost_usd=token_cost,
            estimated_tool_cost_usd=None,
            estimated_total_cost_usd=None,
            status="tool_not_priced",
        )

    tool_cost = _round_cost(
        sum(pricing.tools[call.name].usd_per_call for call in trace.tool_calls)
    )
    return CostEvaluationResult(
        **common,
        estimated_input_cost_usd=input_cost,
        estimated_output_cost_usd=output_cost,
        estimated_token_cost_usd=token_cost,
        estimated_tool_cost_usd=tool_cost,
        estimated_total_cost_usd=_round_cost(token_cost + tool_cost),
        status="estimated",
    )


def _resolve_model(trace: AgentRunTrace) -> str:
    candidates = []
    if trace.usage is not None:
        candidates.append(trace.usage.metadata.get("model"))
    candidates.append(trace.metadata.get("model"))
    candidates.append(trace.target)
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return trace.target


def _round_cost(value: float) -> float:
    return round(value, 12)
