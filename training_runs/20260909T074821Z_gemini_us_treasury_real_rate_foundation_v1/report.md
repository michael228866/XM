# GEMINI US TREASURY REAL RATE FOUNDATION V1

Status: **PASS**

DATA-ONLY. No labels, predictions, trades, outcomes, or model fitting were accessed.

- Pre-run Git commit: `bf89f2c4211d6360c72a0347894521d3321c5edf`
- Nominal observations 2016-2024: **2,250** (2016-01-04 → 2024-12-31)
- Real observations 2016-2024: **2,250** (2016-01-04 → 2024-12-31)
- Nominal/real observation-date mismatch: **0**
- Required maturity N/A counts: `{'nominal_2y': 0, 'nominal_10y': 0, 'real_5y': 0, 'real_10y': 0}`
- Substantive repeated-download revisions: **0**
- Methodology break 2021-12-06 confirmed: **True**
- Official 6:00 PM ET availability evidence confirmed: **True**
- Frozen availability: observation date +2 calendar days, 00:00 America/New_York
- Six GOLD timestamp hashes matched: **True**
- Exact unresolved source rows: **0**

## Five frozen features

- `UST_REAL_10Y`
- `UST_REAL_5Y`
- `UST_REAL_10Y_CHG_1D`
- `UST_2S10S`
- `UST_BREAKEVEN_10Y_PROXY`

## Block coverage

- fold1_train: 530,218/530,218 constructable; unresolved 0; finite [100.0, 100.0, 100.0, 100.0, 100.0]%
- fold1_score: 1,058,080/1,058,080 constructable; unresolved 0; finite [100.0, 100.0, 100.0, 100.0, 100.0]%
- fold2_train: 532,563/532,563 constructable; unresolved 0; finite [100.0, 100.0, 100.0, 100.0, 100.0]%
- fold2_score: 708,197/708,197 constructable; unresolved 0; finite [100.0, 100.0, 100.0, 100.0, 100.0]%
- fold3_train: 533,580/533,580 constructable; unresolved 0; finite [100.0, 100.0, 100.0, 100.0, 100.0]%
- fold3_score: 708,020/708,020 constructable; unresolved 0; finite [100.0, 100.0, 100.0, 100.0, 100.0]%

- `us_treasury_source_dataset_sha256`: `6ed983e4d1522208938e73d4392b0ef277541cb8cbdd0464de1400ea179660f2`
- `us_treasury_real_rate_feature_matrix_sha256`: `6d6870536437f3e4c3a4edd3c939852f9998bde8f0be5fe946ac8fe561fff080`
- Independent validator: **PASS**

## Decision

**US TREASURY REAL RATE DATA FOUNDATION READY = YES**

- Model training performed: NO
- Strategy evaluation performed: NO
- gemini.py changed: NO
- Operational model changed: NO

Single next action: Authorize frozen B0 (31 execution-aligned features) vs B1 (31 + 5 Treasury features) experiment.
