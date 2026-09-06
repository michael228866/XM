# GEMINI USD FX PRESSURE FOUNDATION V1

Data-only. No model, label, prediction, outcome, or strategy metric was accessed.

## Preliminary decision

USD FX PRESSURE DATA FOUNDATION READY pending validator: **NO**

## Metrics

```json
{
  "run_id": "20260906T104638Z_gemini_usd_fx_pressure_foundation_v1",
  "run_status": "pending_independent_validator",
  "preliminary_data_readiness": false,
  "data_foundation_ready": false,
  "symbols": {
    "EUR/USD": "EURUSD#",
    "GBP/USD": "GBPUSD#",
    "USD/JPY": "USDJPY#"
  },
  "coverage": [
    {
      "instrument": "EUR/USD",
      "broker_symbol": "EURUSD#",
      "first_m1_open_utc": "2016-06-30T19:54:00+00:00",
      "last_m1_open_utc": "2024-12-31T18:01:00+00:00",
      "total_rows": 3162754,
      "duplicate_rows_removed": 0,
      "non_monotonic_after_canonicalization": 0,
      "invalid_ohlc_rows": 0,
      "zero_or_invalid_prices": 0,
      "acquisition_chunk_errors": 0,
      "unexplained_gap_count": 26,
      "maximum_unexplained_gap_minutes": 2882,
      "spans_required_prehistory": true,
      "spans_required_end": true
    },
    {
      "instrument": "GBP/USD",
      "broker_symbol": "GBPUSD#",
      "first_m1_open_utc": "2016-06-30T19:54:00+00:00",
      "last_m1_open_utc": "2024-12-31T18:01:00+00:00",
      "total_rows": 3019070,
      "duplicate_rows_removed": 0,
      "non_monotonic_after_canonicalization": 0,
      "invalid_ohlc_rows": 0,
      "zero_or_invalid_prices": 0,
      "acquisition_chunk_errors": 0,
      "unexplained_gap_count": 2436,
      "maximum_unexplained_gap_minutes": 2939,
      "spans_required_prehistory": true,
      "spans_required_end": true
    },
    {
      "instrument": "USD/JPY",
      "broker_symbol": "USDJPY#",
      "first_m1_open_utc": "2016-06-30T19:54:00+00:00",
      "last_m1_open_utc": "2024-12-31T18:01:00+00:00",
      "total_rows": 3011482,
      "duplicate_rows_removed": 0,
      "non_monotonic_after_canonicalization": 0,
      "invalid_ohlc_rows": 0,
      "zero_or_invalid_prices": 0,
      "acquisition_chunk_errors": 0,
      "unexplained_gap_count": 2571,
      "maximum_unexplained_gap_minutes": 2939,
      "spans_required_prehistory": true,
      "spans_required_end": true
    }
  ],
  "feature_coverage": {
    "USD_PRESSURE_1M": {
      "finite_rows": 2829070,
      "nan_rows": 175445,
      "finite_percentage": 94.1606215978286
    },
    "USD_PRESSURE_5M": {
      "finite_rows": 2817752,
      "nan_rows": 186763,
      "finite_percentage": 93.78392186426096
    },
    "USD_PRESSURE_15M": {
      "finite_rows": 2811090,
      "nan_rows": 193425,
      "finite_percentage": 93.56218890569693
    },
    "USD_PRESSURE_60M": {
      "finite_rows": 2806185,
      "nan_rows": 198330,
      "finite_percentage": 93.39893460342185
    },
    "USD_DISPERSION_15M": {
      "finite_rows": 2811090,
      "nan_rows": 193425,
      "finite_percentage": 93.56218890569693
    }
  },
  "all_six_timestamp_hashes_matched": true,
  "usd_fx_feature_matrix_sha256": "da226ee13e8470a919cc4b08770e76c376911e8d4613e0bc5ffc98a8395fdf81",
  "fx_source_dataset_sha256": "0251732cc3ce79c25a4015c49efeaa75a802dc949a440a2f7cfaf14804e56b37",
  "unknown_source_rows_union": 148867,
  "legitimate_nan_rows_union": 65277,
  "model_training_performed": false,
  "strategy_evaluation_performed": false,
  "labels_loaded": false,
  "predictions_loaded": false,
  "outcomes_loaded": false,
  "gemini_py_changed": false,
  "operational_model_changed": false,
  "single_next_action": "Resolve documented FX source/alignment defects in a new data-only run; do not train."
}
```

## Independent validation

Internal methodology: **PASS**. Data certification: **FAIL**.

USD FX PRESSURE DATA FOUNDATION READY = **NO**
