# GEMINI DUKASCOPY FX TICK GAP ADJUDICATION V1

Data-only. No model, label, prediction, target, trade, or strategy metric was accessed.

Preliminary foundation ready: **NO**

```json
{
  "run_id": "20260907T114431Z_gemini_dukascopy_fx_tick_gap_adjudication_v1",
  "run_status": "pending_validator",
  "dukascopy_tick_retrieval": "FAIL",
  "gap_intervals_audited": 554,
  "requested_unique_pair_hours": 12823,
  "native_overlap_bars_compared": 57365,
  "native_tick_ohlc_equivalence": "FAIL",
  "mismatch_bars": 1,
  "mismatch_rate": 1.743223219733287e-05,
  "gap_classification": {
    "verified_no_tick_interval": {
      "intervals": 530,
      "minutes": 653215
    },
    "native_m1_missing_but_ticks_present": {
      "intervals": 0,
      "minutes": 0
    },
    "retrieval_error": {
      "intervals": 229,
      "minutes": 43290
    },
    "inconsistent_provider_history": {
      "intervals": 8,
      "minutes": 6826
    }
  },
  "previous_unresolved_rows": 8476,
  "resolved_by_verified_no_tick": 1545,
  "resolved_by_tick_reconstruction": 0,
  "remaining_unresolved_rows": 6931,
  "new_unresolved_rows": 0,
  "reconstructed_native_m1_minutes": {
    "EUR/USD": 0,
    "GBP/USD": 0,
    "USD/JPY": 0
  },
  "feature_coverage": {
    "USD_PRESSURE_1M": 99.76841520178797,
    "USD_PRESSURE_5M": 99.76678432292732,
    "USD_PRESSURE_15M": 99.76395524735273,
    "USD_PRESSURE_60M": 99.7060424061787,
    "USD_DISPERSION_15M": 99.76395524735273
  },
  "all_six_timestamp_hashes_matched": true,
  "source_provider_count": 1,
  "economic_data_provider": "Dukascopy",
  "representation_sources": "native_m1 + tick_reconstructed_m1",
  "source_dataset_sha256": "b4cdddd5e5a0402af9c76c1ec7986d732bdee068f1df18aebb16c901f0307a6b",
  "dukascopy_tick_reconciled_feature_matrix_sha256": null,
  "data_foundation_ready": false,
  "model_training_performed": false,
  "strategy_evaluation_performed": false,
  "gemini_py_changed": false,
  "operational_model_changed": false,
  "single_next_action": "Resolve only the preserved retrieval/equivalence defects; do not train."
}
```

## Independent validation

Internal methodology: **PASS**. Data certification: **FAIL**. Foundation ready: **NO**.
