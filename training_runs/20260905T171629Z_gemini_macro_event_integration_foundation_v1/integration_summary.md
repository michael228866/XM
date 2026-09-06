# GEMINI MACRO EVENT INTEGRATION FOUNDATION V1 — Final data certification

MACRO EVENT B0/B1 DATA FOUNDATION READY = YES

Run: `20260905T171629Z_gemini_macro_event_integration_foundation_v1`. Internal data methodology PASS. Final untouched strategy validity FAIL/not established: this is historical data certification only, not a model or performance study. No training follows this result.

## PCE

| Item | Result |
|---|---|
| Inherited PCE events | 143 |
| Canonical national releases retained | 119 |
| National releases inside required history | 101, plus the prehistory anchor and neutral older identity records |
| State-level exclusions | 19 |
| Other noncanonical exclusions (county/metropolitan income) | 5 |
| Missing canonical national releases | 0 |
| Ambiguous national releases | 0 |
| PCE history sufficient | YES |

Official national reference-period sequence covers May 2016–November 2024; reference periods are not public release dates. The shutdown-combined February/March 2019 release counts once. No public-release-month presence rule was used.

| Exact PCE correction impact (deduplicated timestamps) | Rows |
|---|---|
| EVENT_MINUTES_SINCE changed | 28,488 |
| Post-window flags changed | 5,190 |
| EVENT_TYPE_PCE changed | 5,370 |
| Any of eight features changed | 28,728 |

The 2016-10-04 state release affects 1,380 rows; the 2017-10-04 state release affects 1,379; the December 2015 state release affects 0. All 24 excluded events have separate isolated deletion counts in excluded_pce_event_impact.csv. These counts are not automatically additive.

## CPI and other families

| Item | Result |
|---|---|
| CPI 2016-06-16 timestamp recovered | NO |
| CPI source retention complete | NO |
| cpi_20160616_exact_feature_rows_affected | 0 |
| CPI feature-impact-neutral | YES |
| CPI history sufficient for exact rows | YES |
| EMPLOYMENT history sufficient | YES |
| FOMC history sufficient | YES; 74 unchanged canonical statements, including 3 unscheduled |

CPI: original HTTP 200 empty response and new HTTP 403 remain provenance defects. Every possible suspect-date timestamp is more than 1440 minutes before every required row, so its eight-feature effect is identically zero. A 2015-01-09 Employment timestamp omission was also independently identified and proven neutral; it is outside required history. Neither defect is silently treated as a recovered source.

## Six exact blocks

| Block | Rows | Constructable | Original broker timestamp SHA-256 |
|---|---:|---:|---|
| fold1_train | 530,218 | 530,218 | `6a7405a13f30c54e6863cf6e80ea1b2e9ee93a90902e5edd37fe4305d471aab6` |
| fold1_score | 1,058,080 | 1,058,080 | `47086d0837f09e86d8874874de302e96e7573efb29236f24720f4a14e0e33c94` |
| fold2_train | 532,563 | 532,563 | `691d8d2b01c3829f010ab559b1daa07c1899f8425a5b8b23b3007bf58467df44` |
| fold2_score | 708,197 | 708,197 | `1fc6500ff642dbf7172328f71f13f38712d11f6c0ce873c38524745710c608b8` |
| fold3_train | 533,580 | 533,580 | `3cf7f68ba2cf994d23d4db97c2e7ec10b2ed772245e3d4b776272ed20ee06b1b` |
| fold3_score | 708,020 | 708,020 | `1451d2069b1d087dc8b4bb7b3ed5840a7b2c03d4adfb548eadc611ed07803449` |

Deduplicated timestamp union: 3,004,515. All six hashes match retained C1 provenance. Source reading was DATE/TIME only; conversion uses the inherited audited XM EET/EEST convention, not newly inferred broker offsets.

- `macro_event_dataset_sha256`: `cd212f31ca0f9b8294c485e20aa863216eb859a6825540a82d40a46f81162592`
- `event_feature_matrix_sha256` (dtype/shape/values): `8f0bf303448bccb8ae418cd9b980447a835704731875336dddd4e0a64f74aac1`
- `event_feature_archive_sha256`: `85506f203de20ecb6b34c3c91cd9a60b7befc6195197c7f5a44f36487ea9139c`
- `incomplete_event_history_rows_exact`: **0**
- Causal alignment: **PASS**, release <= decision for every row; independent backward as-of reconstruction matches the entire frozen matrix.

## Validation and preservation

The initial audit FAIL and summary-write error are retained. Only output serialization, independent validator time units, neutral-prehistory proof and snapshot encoding were corrected. All seven recorded data artifact hashes remain exactly identical to their pre-correction values. The corrected internal validator passes. Nine previous finalized runs remain byte-identical.

Git stages: initial `57d47fd95b9f86445cbe3a0c43cf80be0775bc34`; source review `8f8bb7ec884065c7e7f189c95d136a8b9c1d6074`; output-only build revision `9fa631f3a88d08b8ad668c2d9188f4410d8a10b3`; corrected independent audit `59e15271e0cc63aaf04cf0559d3f78f9c1d8924c`. Each was pushed with a verified clean state before its next computation stage. Original and exact executed script snapshots are retained. Final archive commit identity is the Git commit that contains FINALIZED.json and this run's registry entry; it is not inserted into its own hashed content.

## Safety and next action

- model training performed = no
- strategy evaluation performed = no
- gemini.py changed = no
- operational model changed = no
- no candidate, no Generation 22, no production promotion

Single next action: wait for explicit authorization for GEMINI MACRO EVENT TIMING B0 VS B1 V1. Do not start automatically.

