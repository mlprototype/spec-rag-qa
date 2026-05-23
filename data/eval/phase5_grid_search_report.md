# Phase 5 Grid Search Report

- Status: `ok`
- Ground Truth: `data/eval/ground_truth_phase0_expanded.json`
- Docs Dir: `data/docs`
- Total Trials: `2`
- Eligible Trials: `1`

## Best Params

```json
{
  "vector_candidate_k": 10,
  "bm25_candidate_k": 10,
  "rrf_k": 30,
  "final_top_k": 5,
  "boost_alpha": 1.5,
  "boost_beta": 2.0
}
```

## Best Metrics

```json
{
  "recall_at_1": 0.6,
  "recall_at_5": 0.9,
  "mrr": 0.6908,
  "failure_rate": 0.1,
  "p50_latency_ms": 4.563,
  "p95_latency_ms": 7.641,
  "mean_latency_ms": 5.257,
  "evaluable_cases": 20,
  "total_cases": 25,
  "per_case": [
    {
      "id": "exp-001",
      "rank": 4,
      "hit_at_1": false,
      "hit_at_5": true,
      "reciprocal_rank": 0.25,
      "expected_sources": [
        "sample_spec_v2.md",
        "error_code_reference.md"
      ],
      "citations": [
        "incident_runbook_auth.md#0",
        "unrelated_marketing_doc.md#1",
        "support_playbook_identity.md#0",
        "sample_spec_v2.md#0",
        "account_lifecycle_spec.md#0"
      ]
    },
    {
      "id": "exp-002",
      "rank": null,
      "hit_at_1": false,
      "hit_at_5": false,
      "reciprocal_rank": 0.0,
      "expected_sources": [
        "api_contract_signup.md",
        "error_code_reference.md"
      ],
      "citations": [
        "unrelated_marketing_doc.md#1",
        "account_lifecycle_spec.md#0",
        "support_playbook_identity.md#0",
        "retention_policy.md#0",
        "incident_runbook_auth.md#0"
      ]
    },
    {
      "id": "exp-003",
      "rank": 1,
      "hit_at_1": true,
      "hit_at_5": true,
      "reciprocal_rank": 1.0,
      "expected_sources": [
        "sample_spec_v2.md",
        "auth_design_detail.md",
        "password_policy_faq.md"
      ],
      "citations": [
        "auth_design_detail.md#0",
        "password_policy_faq.md#0",
        "account_lifecycle_spec.md#0",
        "sample_spec_v2.md#0",
        "password_policy_faq.md#5"
      ]
    },
    {
      "id": "exp-004",
      "rank": 5,
      "hit_at_1": false,
      "hit_at_5": true,
      "reciprocal_rank": 0.2,
      "expected_sources": [
        "api_contract_signup.md",
        "sample_spec_v2.md"
      ],
      "citations": [
        "retention_policy.md#0",
        "account_lifecycle_spec.md#1",
        "unrelated_company_history.md#0",
        "account_lifecycle_spec.md#0",
        "api_contract_signup.md#6"
      ]
    },
    {
      "id": "exp-005",
      "rank": 1,
      "hit_at_1": true,
      "hit_at_5": true,
      "reciprocal_rank": 1.0,
      "expected_sources": [
        "retention_policy.md",
        "account_lifecycle_spec.md"
      ],
      "citations": [
        "retention_policy.md#0",
        "account_lifecycle_spec.md#0",
        "retention_policy.md#4",
        "retention_policy.md#5",
        "unrelated_marketing_doc.md#0"
      ]
    },
    {
      "id": "exp-006",
      "rank": 1,
      "hit_at_1": true,
      "hit_at_5": true,
      "reciprocal_rank": 1.0,
      "expected_sources": [
        "audit_logging_standard.md"
      ],
      "citations": [
        "audit_logging_standard.md#0",
        "audit_logging_standard.md#6",
        "audit_logging_standard.md#7",
        "audit_logging_standard.md#5",
        "unrelated_company_history.md#0"
      ]
    },
    {
      "id": "exp-007",
      "rank": 1,
      "hit_at_1": true,
      "hit_at_5": true,
      "reciprocal_rank": 1.0,
      "expected_sources": [
        "support_playbook_identity.md",
        "incident_runbook_auth.md"
      ],
      "citations": [
        "incident_runbook_auth.md#0",
        "support_playbook_identity.md#0",
        "unrelated_marketing_doc.md#1",
        "unrelated_travel_policy.md#0",
        "api_contract_signup.md#0"
      ]
    },
    {
      "id": "exp-008",
      "rank": 2,
      "hit_at_1": false,
      "hit_at_5": true,
      "reciprocal_rank": 0.5,
      "expected_sources": [
        "session_management_design.md"
      ],
      "citations": [
        "account_lifecycle_spec.md#0",
        "session_management_design.md#0",
        "session_management_design.md#2",
        "session_management_design.md#6",
        "unrelated_marketing_doc.md#0"
      ]
    },
    {
      "id": "exp-009",
      "rank": 3,
      "hit_at_1": false,
      "hit_at_5": true,
      "reciprocal_rank": 0.333333,
      "expected_sources": [
        "auth_design_detail.md",
        "support_playbook_identity.md"
      ],
      "citations": [
        "password_policy_faq.md#0",
        "account_lifecycle_spec.md#0",
        "support_playbook_identity.md#0",
        "auth_design_detail.md#0",
        "unrelated_marketing_doc.md#0"
      ]
    },
    {
      "id": "exp-010",
      "rank": 1,
      "hit_at_1": true,
      "hit_at_5": true,
      "reciprocal_rank": 1.0,
      "expected_sources": [
        "data_classification_policy.md"
      ],
      "citations": [
        "data_classification_policy.md#0",
        "unrelated_marketing_doc.md#0",
        "account_lifecycle_spec.md#0",
        "data_classification_policy.md#1",
        "data_classification_policy.md#2"
      ]
    },
    {
      "id": "exp-011",
      "rank": 5,
      "hit_at_1": false,
      "hit_at_5": true,
      "reciprocal_rank": 0.2,
      "expected_sources": [
        "error_code_reference.md",
        "sample_spec_v2.md"
      ],
      "citations": [
        "account_lifecycle_spec.md#0",
        "incident_runbook_auth.md#0",
        "audit_logging_standard.md#0",
        "auth_design_detail.md#0",
        "error_code_reference.md#0"
      ]
    },
    {
      "id": "exp-012",
      "rank": 3,
      "hit_at_1": false,
      "hit_at_5": true,
      "reciprocal_rank": 0.333333,
      "expected_sources": [
        "account_lifecycle_spec.md"
      ],
      "citations": [
        "support_playbook_identity.md#0",
        "retention_policy.md#0",
        "account_lifecycle_spec.md#0",
        "account_lifecycle_spec.md#3",
        "account_lifecycle_spec.md#4"
      ]
    },
    {
      "id": "exp-013",
      "rank": null,
      "hit_at_1": false,
      "hit_at_5": false,
      "reciprocal_rank": 0.0,
      "expected_sources": [
        "incident_runbook_auth.md"
      ],
      "citations": [
        "unrelated_company_history.md#0",
        "auth_design_detail.md#0",
        "retention_policy.md#0",
        "account_lifecycle_spec.md#0",
        "unrelated_marketing_doc.md#0"
      ]
    },
    {
      "id": "exp-014",
      "rank": 1,
      "hit_at_1": true,
      "hit_at_5": true,
      "reciprocal_rank": 1.0,
      "expected_sources": [
        "password_policy_faq.md",
        "auth_design_detail.md"
      ],
      "citations": [
        "password_policy_faq.md#0",
        "retention_policy.md#0",
        "account_lifecycle_spec.md#0",
        "auth_design_detail.md#0",
        "unrelated_marketing_doc.md#0"
      ]
    },
    {
      "id": "exp-015",
      "rank": 1,
      "hit_at_1": true,
      "hit_at_5": true,
      "reciprocal_rank": 1.0,
      "expected_sources": [
        "api_contract_signup.md",
        "sample_spec_v2.md"
      ],
      "citations": [
        "api_contract_signup.md#0",
        "unrelated_marketing_doc.md#1",
        "sample_spec_v2.md#0",
        "password_policy_faq.md#0",
        "api_contract_signup.md#4"
      ]
    },
    {
      "id": "exp-016",
      "rank": 1,
      "hit_at_1": true,
      "hit_at_5": true,
      "reciprocal_rank": 1.0,
      "expected_sources": [
        "audit_logging_standard.md",
        "retention_policy.md"
      ],
      "citations": [
        "retention_policy.md#0",
        "data_classification_policy.md#0",
        "audit_logging_standard.md#0",
        "unrelated_marketing_doc.md#0",
        "audit_logging_standard.md#4"
      ]
    },
    {
      "id": "exp-017",
      "rank": 1,
      "hit_at_1": true,
      "hit_at_5": true,
      "reciprocal_rank": 1.0,
      "expected_sources": [
        "sample_spec_v2.md",
        "error_code_reference.md"
      ],
      "citations": [
        "error_code_reference.md#0",
        "incident_runbook_auth.md#0",
        "password_policy_faq.md#0",
        "auth_design_detail.md#0",
        "unrelated_marketing_doc.md#1"
      ]
    },
    {
      "id": "exp-018",
      "rank": 1,
      "hit_at_1": true,
      "hit_at_5": true,
      "reciprocal_rank": 1.0,
      "expected_sources": [
        "support_playbook_identity.md"
      ],
      "citations": [
        "support_playbook_identity.md#0",
        "password_policy_faq.md#0",
        "unrelated_marketing_doc.md#1",
        "account_lifecycle_spec.md#0",
        "unrelated_marketing_doc.md#0"
      ]
    },
    {
      "id": "exp-019",
      "rank": 1,
      "hit_at_1": true,
      "hit_at_5": true,
      "reciprocal_rank": 1.0,
      "expected_sources": [
        "account_lifecycle_spec.md",
        "retention_policy.md"
      ],
      "citations": [
        "retention_policy.md#0",
        "unrelated_company_history.md#0",
        "account_lifecycle_spec.md#2",
        "unrelated_marketing_doc.md#0",
        "account_lifecycle_spec.md#7"
      ]
    },
    {
      "id": "exp-020",
      "rank": 1,
      "hit_at_1": true,
      "hit_at_5": true,
      "reciprocal_rank": 1.0,
      "expected_sources": [
        "incident_runbook_auth.md",
        "error_code_reference.md"
      ],
      "citations": [
        "incident_runbook_auth.md#0",
        "unrelated_marketing_doc.md#1",
        "password_policy_faq.md#0",
        "incident_runbook_auth.md#4",
        "support_playbook_identity.md#0"
      ]
    }
  ]
}
```

## Top 10 Eligible Trials

| trial_id | stage | recall@5 | mrr | failure_rate | p95_ms |
|---:|---|---:|---:|---:|---:|
| 2 | stage1 | 0.9000 | 0.6908 | 0.1000 | 7.641 |