from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragqa.agent_eval.adapters.trace_codec import parse_agent_trace
from ragqa.agent_eval.advanced_models import (
    ModelTokenPricing,
    PricingConfig,
    ToolCallPricing,
)
from ragqa.agent_eval.cost import (
    PricingConfigError,
    estimate_cost,
    load_pricing_config,
)
from ragqa.agent_eval.models import AgentRunTrace


ROOT = Path(__file__).resolve().parents[2]
PRICING_PATH = ROOT / "config" / "agent_pricing.json"


def test_cost_uses_versioned_model_and_tool_prices(
    synthetic_traces: list[AgentRunTrace],
) -> None:
    trace = next(item for item in synthetic_traces if item.tool_calls)
    model = str(trace.metadata["model"])
    assert model != trace.target
    pricing = PricingConfig(
        schema_version="1.0",
        pricing_version="test-v1",
        currency="USD",
        token_unit=1000,
        models={
            model: ModelTokenPricing(
                input_usd_per_unit=1.0,
                output_usd_per_unit=2.0,
            )
        },
        tools={
            trace.tool_calls[0].name: ToolCallPricing(usd_per_call=0.25)
        },
    )

    result = estimate_cost(trace, pricing)

    assert trace.usage.input_tokens is not None
    assert trace.usage.output_tokens is not None
    expected_token_cost = (
        trace.usage.input_tokens / 1000
        + trace.usage.output_tokens / 1000 * 2
    )
    assert result.estimated_token_cost_usd == pytest.approx(expected_token_cost)
    assert result.estimated_tool_cost_usd == 0.25
    assert result.estimated_total_cost_usd == pytest.approx(
        expected_token_cost + 0.25
    )
    assert result.pricing_version == "test-v1"
    assert result.status == "estimated"


def test_missing_usage_is_na_not_zero(
    synthetic_traces: list[AgentRunTrace],
) -> None:
    trace = synthetic_traces[0].model_copy(deep=True)
    trace.usage.input_tokens = None
    trace.usage.output_tokens = None
    trace.usage.total_tokens = None
    trace.usage.cost_usd = None

    result = estimate_cost(trace, load_pricing_config(PRICING_PATH))

    assert result.status == "usage_unavailable"
    assert result.estimated_token_cost_usd is None
    assert result.estimated_tool_cost_usd is None
    assert result.estimated_total_cost_usd is None


def test_null_usage_from_agent_contract_is_na_not_zero(
    synthetic_traces: list[AgentRunTrace],
) -> None:
    payload = synthetic_traces[0].model_dump(mode="json")
    payload["usage"] = None
    trace = parse_agent_trace(payload)

    result = estimate_cost(trace, load_pricing_config(PRICING_PATH))

    assert result.status == "usage_unavailable"
    assert result.input_tokens is None
    assert result.output_tokens is None
    assert result.estimated_total_cost_usd is None


def test_unknown_model_cost_is_na(
    synthetic_traces: list[AgentRunTrace],
) -> None:
    trace = synthetic_traces[0].model_copy(deep=True)
    trace.metadata["model"] = "unpriced-model"

    result = estimate_cost(trace, load_pricing_config(PRICING_PATH))

    assert result.status == "model_not_priced"
    assert result.estimated_total_cost_usd is None


def test_real_agent_target_is_not_used_as_a_model_identifier(
    synthetic_traces: list[AgentRunTrace],
) -> None:
    trace = synthetic_traces[0].model_copy(deep=True)
    trace.target = "ai-agent-rag"
    trace.metadata.pop("model", None)
    trace.usage.metadata = {"origin": "langchain_callbacks"}
    pricing = load_pricing_config(PRICING_PATH).model_copy(deep=True)
    pricing.models["ai-agent-rag"] = ModelTokenPricing(
        input_usd_per_unit=1.0,
        output_usd_per_unit=2.0,
    )

    result = estimate_cost(trace, pricing)

    assert result.model == "unreported"
    assert result.metadata["model_source"] == "unreported"
    assert result.status == "model_not_priced"
    assert result.estimated_total_cost_usd is None


def test_unpriced_tool_keeps_total_cost_na(
    synthetic_traces: list[AgentRunTrace],
) -> None:
    trace = next(item for item in synthetic_traces if item.tool_calls).model_copy(
        deep=True
    )
    pricing = load_pricing_config(PRICING_PATH).model_copy(deep=True)
    pricing.tools.pop(trace.tool_calls[0].name)

    result = estimate_cost(trace, pricing)

    assert result.status == "tool_not_priced"
    assert result.estimated_token_cost_usd is not None
    assert result.estimated_tool_cost_usd is None
    assert result.estimated_total_cost_usd is None


def test_pricing_config_is_versioned_and_strict() -> None:
    pricing = load_pricing_config(PRICING_PATH)

    assert pricing.schema_version == "1.0"
    assert pricing.pricing_version == "phase6-synthetic-2026-07-20-v2"


def test_invalid_pricing_config_raises_clear_error(tmp_path: Path) -> None:
    pricing_path = tmp_path / "pricing.json"
    pricing_path.write_text(
        json.dumps({"schema_version": "1.0"}),
        encoding="utf-8",
    )

    with pytest.raises(PricingConfigError, match="does not match its schema"):
        load_pricing_config(pricing_path)
