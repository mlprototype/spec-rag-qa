from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


class AgentQualityGateError(ValueError):
    """Raised when Gate configuration or Baseline input is unusable."""


def load_gate_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        with config_path.open(encoding="utf-8") as file:
            payload = yaml.safe_load(file)
    except OSError as exc:
        raise AgentQualityGateError(
            f"Agent quality gate config could not be read: {config_path}"
        ) from exc
    except yaml.YAMLError as exc:
        raise AgentQualityGateError(
            f"Agent quality gate config is invalid YAML: {config_path}"
        ) from exc

    if not isinstance(payload, dict):
        raise AgentQualityGateError("Agent quality gate config must be an object")
    for key in ("absolute_gates", "baseline_relative_gates"):
        if not isinstance(payload.get(key), list):
            raise AgentQualityGateError(f"Gate config field {key!r} must be a list")
    if not isinstance(payload.get("failure_owners", {}), dict):
        raise AgentQualityGateError("Gate config failure_owners must be an object")
    return payload


def load_baseline(path: str | Path) -> dict[str, Any]:
    baseline_path = Path(path)
    try:
        with baseline_path.open(encoding="utf-8") as file:
            payload = json.load(file)
    except OSError as exc:
        raise AgentQualityGateError(
            f"Agent quality baseline could not be read: {baseline_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise AgentQualityGateError(
            f"Agent quality baseline is invalid JSON: {baseline_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise AgentQualityGateError("Agent quality baseline must be an object")
    return payload


def build_baseline(
    aggregation: Mapping[str, Any], generated_at: str | None = None
) -> dict[str, Any]:
    """Capture only the stable values used by baseline-relative gates."""

    return {
        "schema_version": "1.0",
        "generated_at": generated_at or _utc_now(),
        "case_count": _read_path(aggregation, "counts.total_cases"),
        "metrics": {
            "task_success_rate": _read_path(
                aggregation, "metrics.task_success_rate.value"
            ),
            "route_selection_accuracy": _read_path(
                aggregation, "metrics.route_selection_accuracy.value"
            ),
            "latency_p95_ms": _read_path(aggregation, "latency.p95_ms"),
        },
    }


def maybe_update_baseline(
    path: str | Path,
    aggregation: Mapping[str, Any],
    *,
    enabled: bool,
    generated_at: str | None = None,
) -> bool:
    """Write a Baseline only after an explicit opt-in from the caller."""

    if not enabled:
        return False
    baseline_path = Path(path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_baseline(aggregation, generated_at=generated_at)
    baseline_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return True


def evaluate_quality_gate(
    aggregation: Mapping[str, Any],
    config: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    absolute_rules = _rules(config, "absolute_gates")
    relative_rules = _rules(config, "baseline_relative_gates")

    for rule in absolute_rules:
        actual = _read_path(aggregation, _required_string(rule, "metric_path"))
        check = _evaluate_absolute_rule(rule, actual)
        checks.append(check)

    for rule in relative_rules:
        actual = _read_path(aggregation, _required_string(rule, "metric_path"))
        baseline_value = _read_path(
            baseline, _required_string(rule, "baseline_path")
        )
        check = _evaluate_relative_rule(rule, actual, baseline_value)
        checks.append(check)

    failed = [check for check in checks if check["status"] == "failed"]
    return {
        "schema_version": str(config.get("schema_version", "1.0")),
        "passed": not failed,
        "failed_count": len(failed),
        "not_applicable_count": sum(
            check["status"] == "not_applicable" for check in checks
        ),
        "checks": checks,
    }


def _evaluate_absolute_rule(
    rule: Mapping[str, Any], actual: Any
) -> dict[str, Any]:
    gate_id = _required_string(rule, "id")
    threshold = _number(rule.get("threshold"), gate_id, "threshold")
    operator = _required_string(rule, "operator")
    if actual is None:
        return _na_result(rule, "absolute", "Current metric is N/A")
    actual_number = _number(actual, gate_id, "actual")
    if operator == "gte":
        passed = actual_number >= threshold
        comparison = f">= {threshold}"
    elif operator == "lte":
        passed = actual_number <= threshold
        comparison = f"<= {threshold}"
    else:
        raise AgentQualityGateError(
            f"Unsupported absolute gate operator for {gate_id}: {operator}"
        )
    return {
        "gate_id": gate_id,
        "gate_type": "absolute",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "actual": actual_number,
        "threshold": threshold,
        "baseline": None,
        "message": f"actual {actual_number} must be {comparison}",
        "reason": str(rule.get("reason", "")),
    }


def _evaluate_relative_rule(
    rule: Mapping[str, Any], actual: Any, baseline: Any
) -> dict[str, Any]:
    gate_id = _required_string(rule, "id")
    if actual is None or baseline is None:
        return _na_result(rule, "baseline_relative", "Current or baseline metric is N/A")

    actual_number = _number(actual, gate_id, "actual")
    baseline_number = _number(baseline, gate_id, "baseline")
    direction = _required_string(rule, "direction")
    if direction == "higher_is_better":
        ratio = _number(rule.get("min_ratio"), gate_id, "min_ratio")
        threshold = baseline_number * ratio
        passed = actual_number >= threshold
        message = (
            f"actual {actual_number} must be >= baseline {baseline_number} "
            f"x {ratio} ({threshold})"
        )
    elif direction == "lower_is_better":
        ratio = _number(rule.get("max_ratio"), gate_id, "max_ratio")
        threshold = baseline_number * ratio
        passed = actual_number <= threshold
        message = (
            f"actual {actual_number} must be <= baseline {baseline_number} "
            f"x {ratio} ({threshold})"
        )
    else:
        raise AgentQualityGateError(
            f"Unsupported relative gate direction for {gate_id}: {direction}"
        )
    return {
        "gate_id": gate_id,
        "gate_type": "baseline_relative",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "actual": actual_number,
        "threshold": threshold,
        "baseline": baseline_number,
        "message": message,
        "reason": str(rule.get("reason", "")),
    }


def _na_result(
    rule: Mapping[str, Any], gate_type: str, message: str
) -> dict[str, Any]:
    allow_na = bool(rule.get("allow_na", False))
    return {
        "gate_id": _required_string(rule, "id"),
        "gate_type": gate_type,
        "status": "not_applicable" if allow_na else "failed",
        "passed": None if allow_na else False,
        "actual": None,
        "threshold": rule.get("threshold"),
        "baseline": None,
        "message": message,
        "reason": str(rule.get("reason", "")),
    }


def _rules(config: Mapping[str, Any], key: str) -> Sequence[Mapping[str, Any]]:
    raw_rules = config.get(key)
    if not isinstance(raw_rules, list):
        raise AgentQualityGateError(f"Gate config field {key!r} must be a list")
    if not all(isinstance(rule, dict) for rule in raw_rules):
        raise AgentQualityGateError(f"Gate config field {key!r} contains invalid rules")
    return raw_rules


def _read_path(document: Mapping[str, Any], path: str) -> Any:
    current: Any = document
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _required_string(rule: Mapping[str, Any], key: str) -> str:
    value = rule.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AgentQualityGateError(f"Gate rule requires non-empty {key!r}")
    return value


def _number(value: Any, gate_id: str, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AgentQualityGateError(
            f"Gate {gate_id} field {field!r} must be numeric"
        )
    return float(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
