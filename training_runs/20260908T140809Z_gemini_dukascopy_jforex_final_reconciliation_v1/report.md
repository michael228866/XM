# JForex Final Reconciliation V1

Status: FAIL. DATA FOUNDATION READY = NO.

The API returned 784 requests through all three mechanisms. Internal SDK HTTP errors make return-status-only absence certification unsafe.

Original GBP/USD 2023-12-10 22:59 UTC CLOSE: native 1.25433, HTTP tick 1.25428, JForex native 1.25433. Original tolerance 0.00001 preserved.
A/C arbitration supports the native representation; the full prescribed old-tick window and causal failure audit are incomplete.

Provisional row replay resolves 15 of 6931 rows, leaving 6916. This is not certified source completeness.
No final source or feature matrix was certified or generated.

See validator.md for independent recomputation and rejected claims; pre_audit_metrics.json retains the original submitted result.

```json
{
  "run_id": "20260908T140809Z_gemini_dukascopy_jforex_final_reconciliation_v1",
  "run_status": "fail",
  "original_retrieval_error_hours": 783,
  "recovered_by_getTicks": 68,
  "recovered_by_readTicks": 68,
  "verified_zero_tick_hours": 18,
  "recovered_native_m1_bar_hours": 763,
  "retrieval_error_hours_remaining": 697,
  "retrieval_error_hours_affecting_required_rows_remaining": null,
  "original_mismatch_bars": 1,
  "mismatch_pair": "GBP/USD",
  "mismatch_timestamp": "2023-12-10T22:59:00+00:00",
  "mismatch_differing_ohlc_fields": [
    "close"
  ],
  "mismatch_a_native": {
    "open": 1.25433,
    "high": 1.25439,
    "low": 1.25426,
    "close": 1.25433
  },
  "mismatch_b_tick_derived": {
    "open": 1.25433,
    "high": 1.25439,
    "low": 1.25426,
    "close": 1.25428
  },
  "mismatch_c_jforex_native": {
    "open": 1.25433,
    "high": 1.25439,
    "low": 1.25426,
    "close": 1.25433
  },
  "mismatch_root_cause_classification": "tick_reconstruction_or_tick_feed_inconsistency",
  "original_tolerance_preserved": true,
  "equivalence": {
    "previous_overlap_bars": 57365,
    "newly_evaluable_overlap_bars": 3357,
    "total_compared": 60722,
    "previous_mismatch_bars": 1,
    "new_raw_mismatch_bars": 0,
    "root_caused_previous_mismatch": true,
    "unresolved_mismatches": 0,
    "mismatch_rate": 0.0,
    "maximum_new_mismatch": 0.0,
    "original_tolerance_preserved": true,
    "gate": "PASS"
  },
  "equivalence_gate": "FAIL_INCOMPLETE_ROOT_CAUSE_AUDIT",
  "previous_unresolved_rows": 6931,
  "resolved_by_verified_no_quote": 0,
  "resolved_by_recovered_jforex_ticks": 15,
  "resolved_by_jforex_native_m1_bars": 0,
  "resolved_by_representation_arbitration": 0,
  "proven_feature_impact_neutral": 0,
  "remaining_unresolved_rows": 6916,
  "new_unresolved_rows": 0,
  "minute_arbitration_counts": {
    "inconsistent_provider_history": 41970,
    "verified_no_quote_minute": 1080,
    "retrieval_error_resolved_same_provider": 240
  },
  "feature_finite_percentage": {
    "USD_PRESSURE_1M": 99.77670272905942,
    "USD_PRESSURE_5M": 99.77453931832592,
    "USD_PRESSURE_15M": 99.77037891306917,
    "USD_PRESSURE_60M": 99.70674135426184,
    "USD_DISPERSION_15M": 99.77037891306917
  },
  "all_six_timestamp_hashes_matched": true,
  "source_dataset_sha256": "9c96c15c2a2ec3c30978125afd605f137bba19710aad928b6385f6f3f0b70a15",
  "per_block_feature_matrix_sha256": {},
  "dukascopy_final_feature_matrix_sha256": null,
  "data_foundation_ready_pre_validator": false,
  "data_foundation_ready": false,
  "family_testable_to_provenance_standard": false,
  "model_training_performed": false,
  "strategy_evaluation_performed": false,
  "gemini_py_changed": false,
  "operational_model_changed": false,
  "single_next_action": "Stop USD FX PRESSURE research; move to a cleaner external information family.",
  "validator_internal_methodology": "FAIL",
  "validator_data_certification": "FAIL",
  "potentially_affecting_unresolved_hours_upper_bound": 13,
  "exact_hour_impact_status": "not_certified; original 721 counter invalid",
  "mismatch_root_cause_complete": false,
  "observed_A_C_arbitration": "PASS",
  "row_reconciliation_status": "recomputed under submitted rules; not certified due acquisition assurance failures",
  "internal_http_error_hours": 9
}
```

Single next action: stop USD FX PRESSURE foundation research. A different external information family requires separate authorization.
