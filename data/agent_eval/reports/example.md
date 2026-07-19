# Agent Quality Report

- Generated: 2026-07-19T16:19:37.541955Z
- Runner: `fixture`
- Gate: **PASS**
- Baseline: `data/agent_eval/baseline/agent_baseline.json`
- Baseline updated: `false`

## Summary

| Total | Evaluated | Passed | Evaluation FAIL | Execution Error |
|---:|---:|---:|---:|---:|
| 20 | 20 | 20 | 0 | 0 |

## Quality Metrics

| Metric | Numerator | Denominator | Value |
|:---|---:|---:|---:|
| Route Selection Accuracy | 20 | 20 | 100.00% |
| Required Tool Call Rate | 17 | 17 | 100.00% |
| Unexpected Tool Call Rate | 0 | 17 | 0.00% |
| Tool Argument Schema Compliance | 17 | 17 | 100.00% |
| Tool Argument Semantic Accuracy | 17 | 17 | 100.00% |
| Citation Presence | 10 | 10 | 100.00% |
| Citation Validity | 16 | 16 | 100.00% |
| Answer Format Compliance | 20 | 20 | 100.00% |
| Latency Budget Compliance | 20 | 20 | 100.00% |
| Task Success Rate | 20 | 20 | 100.00% |

## Latency

| Count | Average | p50 | p95 | Max |
|---:|---:|---:|---:|---:|
| 20 | 345.250 ms | 260.000 ms | 570.000 ms | 1450.000 ms |

## Route Confusion Matrix

| Expected \ Actual | compare | direct | retrieval | structured_query |
|:---|---:|---:|---:|---:|
| compare | 3 | 0 | 0 | 0 |
| direct | 0 | 3 | 0 | 0 |
| retrieval | 0 | 0 | 10 | 0 |
| structured_query | 0 | 0 | 0 | 4 |

## By Category

| Group | Total | Passed | Evaluation FAIL | Execution Error | Task Success |
|:---|---:|---:|---:|---:|---:|
| compare | 3 | 3 | 0 | 0 | 100.00% |
| definition | 2 | 2 | 0 | 0 | 100.00% |
| direct | 3 | 3 | 0 | 0 | 100.00% |
| fallback | 1 | 1 | 0 | 0 | 100.00% |
| insufficient_evidence | 2 | 2 | 0 | 0 | 100.00% |
| retrieval | 2 | 2 | 0 | 0 | 100.00% |
| retrieval_complex | 3 | 3 | 0 | 0 | 100.00% |
| structured_query | 4 | 4 | 0 | 0 | 100.00% |

## By Severity

| Group | Total | Passed | Evaluation FAIL | Execution Error | Task Success |
|:---|---:|---:|---:|---:|---:|
| critical | 4 | 4 | 0 | 0 | 100.00% |
| high | 9 | 9 | 0 | 0 | 100.00% |
| low | 3 | 3 | 0 | 0 | 100.00% |
| medium | 4 | 4 | 0 | 0 | 100.00% |

## Failure Types

No evaluation failures.

## Execution Errors

No execution errors.

## Quality Gate

| Gate | Type | Status | Actual | Baseline | Threshold | Reason |
|:---|:---|:---|---:|---:|---:|:---|
| execution_errors | absolute | passed | 0 | N/A | 0 | Runner failure is not an evaluable Agent result and must block the deterministic gate. |
| critical_task_success | absolute | passed | 1 | N/A | 1 | A Critical case failure cannot be offset by lower-severity averages. |
| required_tool_call_rate | absolute | passed | 1 | N/A | 1 | Every required Tool call must be observed when the dataset contains Tool cases. |
| tool_argument_schema_compliance | absolute | passed | 1 | N/A | 1 | Invalid Tool arguments can cause unsafe or failed downstream execution. |
| citation_validity | absolute | passed | 1 | N/A | 1 | Every emitted Citation must resolve to an observed Source. |
| critical_format_compliance | absolute | passed | 1 | N/A | 1 | Critical answers must preserve their machine-checkable response contract. |
| overall_task_success_regression | baseline_relative | passed | 1 | 1 | 1 | Overall Task Success must not regress from the reviewed Baseline. |
| route_accuracy_regression | baseline_relative | passed | 1 | 1 | 1 | Route Selection Accuracy must not regress from the reviewed Baseline. |
| latency_p95_regression | baseline_relative | passed | 570 | 570 | 627 | p95 latency may vary, but a regression above 10 percent requires review. |
