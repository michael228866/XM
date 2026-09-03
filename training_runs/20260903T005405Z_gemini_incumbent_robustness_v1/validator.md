# GEMINI INCUMBENT ROBUSTNESS ATTRIBUTION V1 - independent validation

Overall: **FAIL**

Internal diagnostic methodology: **PASS**

Final untouched-test validity: **FAIL**

| Check | Verdict | Evidence | Failure or reason for pass | Required validation correction |
|---|---|---|---|---|
| chronology | PASS | oof_model_provenance.json records=12; diagnostic script:411 | All W0-W3 fold fits end before their scored interval and all preserved assertions reconcile. | none |
| feature leakage | PASS | drl_trading_v2.py:138; manifest.model.feature_count=31 | The frozen 31 features are shifted one completed bar and no diagnostic outcome field enters fitting. | none |
| label maturity | PASS | gold_generation11_execution_aligned.py:124; oof_model_provenance.json | Every fit has latest_training_label_bar strictly earlier than score_start. | none |
| OOF predictions | PASS | diagnostic_predictions.npz rows=2474297; schemes=4; script:411 | W0 is the preserved parent OOF vector; all nine new replicas fit only their own preceding window. | none |
| calibration | PASS | manifest.model.calibration_method; calibration_summary.csv; calibration_buckets.csv | No calibrator was fitted; fixed-bin calibration is descriptive and independently checked. | none |
| threshold selection | PASS | manifest.diagnostic_design; manifest.search.performed=false | One threshold/RSI/execution configuration and exactly four pre-specified training windows were reported; none was promoted. | none |
| purge/embargo | PASS | diagnostic script:397; oof_model_provenance.json | The full 240-row horizon is removed and exact label-end timestamps precede every score block. | none |
| holdout contamination | PASS | manifest.evidence_status.classification=development_diagnostic_only | No historical interval is labeled untouched and no production-performance claim is made. | none |
| recent-period reuse | PASS | manifest.evidence_status.previous_forward_status; metrics.live_monitoring.classification | The old cutoff remains contaminated and live outcomes are explicitly monitoring-only, with no tuning or selection. | none |
| execution alignment | PASS | gold_gemini_core_gate_v1.py:469; diagnostic_predictions.npz; trade_ledger.csv | All executable identities and W0-W3 fold metrics were rebuilt from the frozen event arrays. | none |
| cost assumptions | PASS | diagnostic_predictions.npz gross_pnl_price/spread_points/denominator/reward/stress_reward | Observed positive spread or a 30-point fallback plus 5/10-point extra costs reconcile for every OOF row. | none |
| multiple-testing risk | FAIL | README.md historical generations; manifest.evidence_status; no untouched final interval | The four windows were pre-specified and all are reported, but all scored regimes were repeatedly inspected and no untouched final data exists. Findings are hypothesis-generating only. | Freeze a later experiment before collecting a genuinely new forward interval; do not reuse this diagnostic history as untouched evidence. |

## Metric reconciliation

Window/fold metrics: **PASS**.
Executable trade identities: **PASS**.
Probability distributions: **PASS**.

| Scheme | Fold | Trades | Trades/day | WR | PF | Mean-R | PnL-R | Max DD-R |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| W0_expanding | 2018_2020 | 189 | 0.1724 | 25.40% | 0.3725 | -0.4460 | -84.29 | -88.42 |
| W0_expanding | 2021_2022 | 0 | 0.0000 | 0.00% | 0.0000 | 0.0000 | 0.00 | 0.00 |
| W0_expanding | 2023_2024 | 0 | 0.0000 | 0.00% | 0.0000 | 0.0000 | 0.00 | 0.00 |
| W0_expanding | pooled | 189 | 0.0739 | 25.40% | 0.3725 | -0.4460 | -84.29 | -88.42 |
| W1_trailing_24m | 2018_2020 | 502 | 0.4580 | 29.68% | 0.4438 | -0.3715 | -186.52 | -186.52 |
| W1_trailing_24m | 2021_2022 | 61 | 0.0836 | 19.67% | 0.2472 | -0.6139 | -37.45 | -41.88 |
| W1_trailing_24m | 2023_2024 | 38 | 0.0520 | 52.63% | 0.9539 | -0.0227 | -0.86 | -6.87 |
| W1_trailing_24m | pooled | 601 | 0.2350 | 30.12% | 0.4432 | -0.3741 | -224.83 | -230.83 |
| W2_trailing_18m | 2018_2020 | 433 | 0.3951 | 29.10% | 0.4280 | -0.3889 | -168.42 | -168.42 |
| W2_trailing_18m | 2021_2022 | 157 | 0.2151 | 28.03% | 0.4323 | -0.4223 | -66.30 | -69.83 |
| W2_trailing_18m | 2023_2024 | 57 | 0.0780 | 42.11% | 0.8797 | -0.0702 | -4.00 | -7.47 |
| W2_trailing_18m | pooled | 647 | 0.2530 | 29.98% | 0.4629 | -0.3690 | -238.71 | -239.09 |
| W3_trailing_12m | 2018_2020 | 339 | 0.3093 | 29.20% | 0.4348 | -0.3754 | -127.27 | -128.54 |
| W3_trailing_12m | 2021_2022 | 155 | 0.2123 | 29.68% | 0.4085 | -0.4368 | -67.70 | -70.31 |
| W3_trailing_12m | 2023_2024 | 99 | 0.1354 | 44.44% | 0.7901 | -0.1223 | -12.11 | -18.60 |
| W3_trailing_12m | pooled | 593 | 0.2319 | 31.87% | 0.4788 | -0.3492 | -207.08 | -214.84 |

The 60% target and positive-economic guardrails are not promotion claims in this diagnostic. No candidate was selected.

## Validation conclusion

No internal rerun is required; the only failed check is final multiple-testing/untouched validity, which historical recomputation cannot repair.

Submitted claim: `valid_internal_development_diagnostic_invalid_for_final_strategy_claim`.
