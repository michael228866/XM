# Generation 20 walk-forward validation

Overall: FAIL

This validator audits methodology only; it does not optimize or promote the strategy.

Internal chronological validation quality: PASS

Final untouched-test validity: FAIL

| Check | Verdict | Reason |
|---|---|---|
| chronology | PASS | Each research fold is fitted on history ending before the fold; 2025/recent are diagnostics only. |
| feature_leakage | PASS | Model features exclude net-return labels, outcomes, exit offsets, and spread_quality; relative cost transforms use prior blocks only. |
| label_maturity | PASS | Exact event exit offsets mature before calibration and policy boundaries. |
| oof_predictions | PASS | Fixed-cohort selections exactly reproduce frozen policy rules on fold predictions and add no signals in Phases 1-5. |
| calibration | PASS | P(net_R>0) isotonic calibration is fitted on a purged chronological calibration partition only. |
| threshold_selection | PASS | 0.50 and zero economic cutoffs are predeclared; top-75 uses the prior policy-tail prediction quantile, not evaluated outcomes. |
| purge_embargo | PASS | Fit/calibration events have exact label-end purges; the outer training helper removes the horizon tail. |
| holdout_contamination | FAIL | The repeatedly inspected 2025 interval is development data and cannot be an untouched final test. |
| recent_period_reuse | PASS | Recent results are stored only under development_diagnostics and do not select or freeze a candidate. |
| execution_alignment | PASS | Independent ledger metrics, position occupancy, event horizon, subset identity, and the Phase-6 gate reconcile. |
| cost_assumptions | PASS | Observed entry spreads are charged when available; unavailable/zero spread is charged 30 points, plus 5-point base and 10-point stress extras. |
| multiple_testing_risk | FAIL | Two targets and four policies were inspected after many prior generations, with no untouched future interval available. |

## Metric reconciliation

| Strategy / period | Trades | Trades/day | WR | PF | Mean-R | PnL | Max DD | Stress PF | Reconciled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gen17_parent:2018_2020 | 47 | 0.061 | 48.94% | 0.618 | -0.1987 | -125.56 | -13.42% | 0.590 | yes |
| p_net_ge_050:2018_2020 | 39 | 0.050 | 48.72% | 0.628 | -0.1944 | -103.32 | -11.38% | 0.599 | yes |
| e_net_positive:2018_2020 | 13 | 0.017 | 53.85% | 0.810 | -0.0894 | -17.09 | -6.95% | 0.771 | yes |
| joint_positive:2018_2020 | 12 | 0.015 | 50.00% | 0.688 | -0.1592 | -27.28 | -6.95% | 0.652 | yes |
| e_net_top75_past:2018_2020 | 47 | 0.061 | 48.94% | 0.618 | -0.1987 | -125.56 | -13.42% | 0.590 | yes |
| gen17_parent:2021_2022 | 132 | 0.256 | 62.88% | 1.105 | 0.0398 | 67.01 | -13.73% | 1.044 | yes |
| p_net_ge_050:2021_2022 | 36 | 0.070 | 55.56% | 0.856 | -0.0654 | -34.96 | -9.59% | 0.808 | yes |
| e_net_positive:2021_2022 | 20 | 0.039 | 50.00% | 0.675 | -0.1659 | -46.82 | -7.90% | 0.637 | yes |
| joint_positive:2021_2022 | 17 | 0.033 | 47.06% | 0.608 | -0.2120 | -50.46 | -7.38% | 0.574 | yes |
| e_net_top75_past:2021_2022 | 132 | 0.256 | 62.88% | 1.105 | 0.0398 | 67.01 | -13.73% | 1.044 | yes |
| gen17_parent:2023_2024 | 27 | 0.052 | 66.67% | 1.392 | 0.1330 | 49.67 | -5.56% | 1.328 | yes |
| p_net_ge_050:2023_2024 | 21 | 0.041 | 71.43% | 1.758 | 0.2205 | 65.51 | -3.75% | 1.678 | yes |
| e_net_positive:2023_2024 | 5 | 0.010 | 60.00% | 1.078 | 0.0319 | 1.87 | -2.83% | 1.028 | yes |
| joint_positive:2023_2024 | 5 | 0.010 | 60.00% | 1.078 | 0.0319 | 1.87 | -2.83% | 1.028 | yes |
| e_net_top75_past:2023_2024 | 27 | 0.052 | 66.67% | 1.392 | 0.1330 | 49.67 | -5.56% | 1.328 | yes |
| gen17_parent:selection_pooled | 206 | 0.114 | 60.19% | 0.994 | -0.0024 | -20.62 | -20.41% | 0.943 | yes |
| p_net_ge_050:selection_pooled | 96 | 0.053 | 56.25% | 0.876 | -0.0553 | -77.98 | -18.22% | 0.833 | yes |
| e_net_positive:selection_pooled | 38 | 0.021 | 52.63% | 0.765 | -0.1137 | -61.36 | -12.77% | 0.725 | yes |
| joint_positive:selection_pooled | 34 | 0.019 | 50.00% | 0.691 | -0.1575 | -74.63 | -12.27% | 0.654 | yes |
| e_net_top75_past:selection_pooled | 206 | 0.114 | 60.19% | 0.994 | -0.0024 | -20.62 | -20.41% | 0.943 | yes |
| gen17_parent:2025_2026_05_development | 54 | 0.155 | 59.26% | 1.098 | 0.0403 | 26.79 | -8.97% | 1.083 | yes |
| p_net_ge_050:2025_2026_05_development | 54 | 0.155 | 59.26% | 1.098 | 0.0403 | 26.79 | -8.97% | 1.083 | yes |
| e_net_positive:2025_2026_05_development | 31 | 0.089 | 61.29% | 1.206 | 0.0801 | 33.00 | -5.95% | 1.191 | yes |
| joint_positive:2025_2026_05_development | 31 | 0.089 | 61.29% | 1.206 | 0.0801 | 33.00 | -5.95% | 1.191 | yes |
| e_net_top75_past:2025_2026_05_development | 54 | 0.155 | 59.26% | 1.098 | 0.0403 | 26.79 | -8.97% | 1.083 | yes |
| gen17_parent:2026_recent_development | 59 | 0.787 | 54.24% | 0.853 | -0.0680 | -58.75 | -7.86% | 0.836 | yes |
| p_net_ge_050:2026_recent_development | 59 | 0.787 | 54.24% | 0.853 | -0.0680 | -58.75 | -7.86% | 0.836 | yes |
| e_net_positive:2026_recent_development | 5 | 0.067 | 40.00% | 0.483 | -0.3121 | -22.01 | -3.18% | 0.474 | yes |
| joint_positive:2026_recent_development | 5 | 0.067 | 40.00% | 0.483 | -0.3121 | -22.01 | -3.18% | 0.474 | yes |
| e_net_top75_past:2026_recent_development | 59 | 0.787 | 54.24% | 0.853 | -0.0680 | -58.75 | -7.86% | 0.836 | yes |

## Adversarial conclusion

No Phase-5 policy passed. Phase 6 was correctly not executed, no candidate was frozen, and no production promotion is valid.

The experiment is valid as internal chronological development evidence only. Final untouched-test validity remains FAIL until a genuinely new forward interval is collected without re-selection.
