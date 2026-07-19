from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any

from ragqa.agent_eval.evaluator import aggregate_metrics
from ragqa.agent_eval.models import (
    AgentEvalCase,
    AgentEvaluationResult,
    AgentRunTrace,
    CaseEvaluationResult,
)


def percentile(values: Sequence[float], q: float) -> float | None:
    """Return the nearest-rank percentile, or N/A for an empty collection."""

    if not values:
        return None
    if not 0 <= q <= 100:
        raise ValueError("Percentile must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    if q == 0:
        return ordered[0]
    rank = math.ceil((q / 100.0) * len(ordered))
    return ordered[min(max(rank - 1, 0), len(ordered) - 1)]


def build_route_confusion_matrix(
    cases: Sequence[AgentEvalCase], traces: Sequence[AgentRunTrace]
) -> dict[str, Any]:
    """Count expected-route labels against observed common routes."""

    cases_by_id = {case.id: case for case in cases}
    pairs: list[tuple[str, str]] = []
    for trace in traces:
        case = cases_by_id.get(trace.case_id)
        if case is None:
            continue
        expected = _expected_route_label(case)
        pairs.append((expected, trace.output.route))

    expected_labels = sorted({expected for expected, _ in pairs})
    actual_labels = sorted({actual for _, actual in pairs})
    matrix = {
        expected: {actual: 0 for actual in actual_labels}
        for expected in expected_labels
    }
    for expected, actual in pairs:
        matrix[expected][actual] += 1

    return {
        "expected_labels": expected_labels,
        "actual_labels": actual_labels,
        "matrix": matrix,
    }


def aggregate_agent_evaluation(
    cases: Sequence[AgentEvalCase],
    traces: Sequence[AgentRunTrace],
    evaluation: AgentEvaluationResult,
    execution_errors: Sequence[Mapping[str, str]],
    failure_owners: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the complete, runner-independent Agent quality summary."""

    owners = dict(failure_owners or {})
    results_by_id = {result.case_id: result for result in evaluation.cases}
    traces_by_id = {trace.case_id: trace for trace in traces}
    errors_by_id = {
        str(error.get("case_id")): dict(error) for error in execution_errors
    }

    passed = sum(1 for result in evaluation.cases if result.passed)
    failed = sum(1 for result in evaluation.cases if not result.passed)
    counts = {
        "total_cases": len(cases),
        "evaluated_cases": len(evaluation.cases),
        "passed_cases": passed,
        "failed_cases": failed,
        "execution_errors": len(execution_errors),
    }

    latencies = [float(trace.timing.latency_ms) for trace in traces]
    latency = {
        "count": len(latencies),
        "average_ms": round(fmean(latencies), 3) if latencies else None,
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "max_ms": max(latencies) if latencies else None,
    }

    failure_counts: Counter[str] = Counter()
    for result in evaluation.cases:
        for check in result.checks:
            if not check.passed and check.failure_type is not None:
                failure_counts[check.failure_type] += 1

    failure_type_distribution = {
        failure_type: {
            "count": count,
            "owner": owners.get(failure_type, "unassigned"),
        }
        for failure_type, count in sorted(failure_counts.items())
    }
    top_failure_types = [
        {
            "failure_type": failure_type,
            "count": count,
            "owner": owners.get(failure_type, "unassigned"),
        }
        for failure_type, count in sorted(
            failure_counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]

    execution_error_types = Counter(
        str(error.get("error_type", "UNKNOWN_EXECUTION_ERROR"))
        for error in execution_errors
    )

    case_outcomes = []
    for case in cases:
        result = results_by_id.get(case.id)
        trace = traces_by_id.get(case.id)
        execution_error = errors_by_id.get(case.id)
        if execution_error is not None:
            status = "execution_error"
        elif result is not None and result.passed:
            status = "passed"
        else:
            status = "failed"
        case_outcomes.append(
            {
                "case_id": case.id,
                "category": case.category,
                "severity": case.severity,
                "status": status,
                "expected_route": _expected_route_label(case),
                "actual_route": trace.output.route if trace is not None else None,
                "latency_ms": (
                    float(trace.timing.latency_ms) if trace is not None else None
                ),
                "failure_types": (
                    [
                        check.failure_type
                        for check in result.checks
                        if not check.passed and check.failure_type is not None
                    ]
                    if result is not None
                    else []
                ),
                "execution_error": execution_error,
            }
        )

    return {
        "counts": counts,
        "metrics": {
            metric_id: metric.model_dump(mode="json")
            for metric_id, metric in evaluation.metrics.items()
        },
        "latency": latency,
        "route_confusion_matrix": build_route_confusion_matrix(cases, traces),
        "distributions": {
            "category": _group_distribution(
                cases, evaluation.cases, execution_errors, "category"
            ),
            "severity": _group_distribution(
                cases, evaluation.cases, execution_errors, "severity"
            ),
            "failure_type": failure_type_distribution,
            "execution_error_type": dict(sorted(execution_error_types.items())),
        },
        "top_failure_types": top_failure_types,
        "case_outcomes": case_outcomes,
    }


def _group_distribution(
    cases: Sequence[AgentEvalCase],
    results: Sequence[CaseEvaluationResult],
    execution_errors: Sequence[Mapping[str, str]],
    field: str,
) -> dict[str, Any]:
    results_by_id = {result.case_id: result for result in results}
    error_ids = {str(error.get("case_id")) for error in execution_errors}
    groups: dict[str, list[AgentEvalCase]] = {}
    for case in cases:
        key = str(getattr(case, field))
        groups.setdefault(key, []).append(case)

    distribution: dict[str, Any] = {}
    for key, group_cases in sorted(groups.items()):
        group_results = [
            results_by_id[case.id]
            for case in group_cases
            if case.id in results_by_id
        ]
        group_metrics = aggregate_metrics(group_results)
        distribution[key] = {
            "counts": {
                "total_cases": len(group_cases),
                "evaluated_cases": len(group_results),
                "passed_cases": sum(result.passed for result in group_results),
                "failed_cases": sum(not result.passed for result in group_results),
                "execution_errors": sum(case.id in error_ids for case in group_cases),
            },
            "metrics": {
                metric_id: metric.model_dump(mode="json")
                for metric_id, metric in group_metrics.items()
            },
        }
    return distribution


def _expected_route_label(case: AgentEvalCase) -> str:
    if len(case.expected.routes) == 1:
        return case.expected.routes[0]
    return " | ".join(case.expected.routes)
