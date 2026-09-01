# Generation 18 walk-forward adversarial validation

Overall: **FAIL**

| Validation scope | Verdict |
|---|---|
| Internal chronological validation quality | PASS |
| Final untouched-test validity | FAIL |

| Check | Verdict | Evidence | Failure or reason for pass | Required validation correction |
|---|---|---|---|---|
| chronology | PASS | Each outer fold trains before fold_start; inner rank policy is chosen in prior policy data; block ranks are computed before the current block is appended; 2025/recent run after frozen candidate ranking. | Each outer fold trains before fold_start; inner rank policy is chosen in prior policy data; block ranks are computed before the current block is appended; 2025/recent run after frozen candidate ranking. | None. |
| feature leakage | PASS | M1/MTF inputs are shifted, Gen17 normalized features remain entry-time known, and past family outcomes enter only after exit maturity+1. Outcomes in calibration reports are diagnostic labels, not rank inputs. | M1/MTF inputs are shifted, Gen17 normalized features remain entry-time known, and past family outcomes enter only after exit maturity+1. Outcomes in calibration reports are diagnostic labels, not rank inputs. | None. |
| label maturity | PASS | Fit label ends precede calibration starts and calibration label ends precede policy starts in every selection and diagnostic model. | Fit label ends precede calibration starts and calibration label ends precede policy starts in every selection and diagnostic model. | None. |
| OOF predictions | PASS | Outer scores come from models fit/calibrated before each fold; inner rank policies use the disjoint policy segment. All three reconstructed absolute short-trend ledgers exactly match Gen17. | Outer scores come from models fit/calibrated before each fold; inner rank policies use the disjoint policy segment. All three reconstructed absolute short-trend ledgers exactly match Gen17. | None. |
| calibration | PASS | Isotonic calibration is fit only on the chronological calibration partition. Reliability curves and deciles use outer outcomes for diagnosis only and never update a candidate cutoff. | Isotonic calibration is fit only on the chronological calibration partition. Reliability curves and deciles use outer outcomes for diagnosis only and never update a candidate cutoff. | None. |
| threshold selection | PASS | Exactly 12 predeclared inner policies per expert (3 rank modes x top 1/2/5/10%) are selected before the next outer fold; four fixed expert portfolios are compared only on 2018-2024 development folds. | Exactly 12 predeclared inner policies per expert (3 rank modes x top 1/2/5/10%) are selected before the next outer fold; four fixed expert portfolios are compared only on 2018-2024 development folds. | None. |
| purge/embargo | PASS | Outer training drops H=90 rows and inner fit/calibration boundaries use exact exit label ends. | Outer training drops H=90 rows and inner fit/calibration boundaries use exact exit label ends. | None. |
| holdout contamination | FAIL | The experiment explicitly retains 2025 as repeatedly inspected development/diagnostic data; no untouched historical outer test exists. | The experiment explicitly retains 2025 as repeatedly inspected development/diagnostic data; no untouched historical outer test exists. | Freeze a future qualified candidate before collecting a genuinely new embargoed forward interval. |
| recent-period reuse | PASS | Recent data are evaluated only after selection and explicitly excluded from candidate choice and promotion claims. | Recent data are evaluated only after selection and explicitly excluded from candidate choice and promotion claims. | None. |
| execution alignment | PASS | All four candidate paths and the frozen diagnostic reconcile from executable ledgers; HIGH/LOW first-touch starts after entry, ties are stop-first, and single-position occupancy is enforced. | All four candidate paths and the frozen diagnostic reconcile from executable ledgers; HIGH/LOW first-touch starts after entry, ties are stop-first, and single-position occupancy is enforced. | None. |
| cost assumptions | PASS | Phase 1 maps each ledger entry back to ATR, outcome, reward, and exit offset; base reward applies 30 spread + 5 extra points, stress applies 10 extra points with identical entries. TP-first and positive-net-return rates are separately reconciled. | Phase 1 maps each ledger entry back to ATR, outcome, reward, and exit offset; base reward applies 30 spread + 5 extra points, stress applies 10 extra points with identical entries. TP-first and positive-net-return rates are separately reconciled. | None. |
| multiple-testing risk | FAIL | Twelve inner rank policies per expert, four portfolios, prior generations, reused 2025, and monitored recent data have all been inspected; there is no untouched final test or selection-aware final claim. | Twelve inner rank policies per expert, four portfolios, prior generations, reused 2025, and monitored recent data have all been inspected; there is no untouched final test or selection-aware final claim. | Pre-register one qualified candidate and obtain a new untouched forward outer interval without further tuning on it. |

## Frozen diagnostic reconciliation

| Period | Profile | Trades | Trades/day | TP-first | Realized win | Payoff | Break-even | Edge | PF | Mean-R | PnL | Max DD | Match |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2018_2020 | base | 3148 | 4.062 | 46.98% | 47.30% | 0.6738 | 59.75% | -12.45% | 0.60 | -0.2137 | -999.94 | -99.99% | PASS |
| 2018_2020 | cost_stress | 3148 | 4.062 | 46.98% | 47.27% | 0.6320 | 61.27% | -14.01% | 0.57 | -0.2409 | -999.98 | -100.00% | PASS |
| 2021_2022 | base | 2375 | 4.603 | 54.06% | 54.11% | 0.5911 | 62.85% | -8.74% | 0.70 | -0.1417 | -992.33 | -99.24% | PASS |
| 2021_2022 | cost_stress | 2375 | 4.603 | 54.06% | 54.11% | 0.5588 | 64.15% | -10.05% | 0.66 | -0.1629 | -996.21 | -99.62% | PASS |
| 2023_2024 | base | 3644 | 7.062 | 55.49% | 55.57% | 0.5919 | 62.82% | -7.25% | 0.74 | -0.1176 | -998.05 | -99.82% | PASS |
| 2023_2024 | cost_stress | 3644 | 7.062 | 55.49% | 55.57% | 0.5606 | 64.08% | -8.51% | 0.70 | -0.1381 | -999.31 | -99.93% | PASS |
| 2025_2026_05_development | base | 1367 | 3.928 | 56.04% | 56.11% | 0.6768 | 59.64% | -3.53% | 0.87 | -0.0598 | -710.98 | -72.94% | PASS |
| 2025_2026_05_development | cost_stress | 1367 | 3.928 | 56.04% | 56.11% | 0.6588 | 60.29% | -4.18% | 0.84 | -0.0708 | -765.78 | -78.00% | PASS |
| 2026_recent_development | base | 236 | 3.147 | 58.05% | 58.05% | 0.7196 | 58.15% | -0.10% | 1.00 | -0.0018 | -22.58 | -17.61% | PASS |
| 2026_recent_development | cost_stress | 236 | 3.147 | 58.05% | 58.05% | 0.7073 | 58.57% | -0.52% | 0.98 | -0.0090 | -45.63 | -18.32% | PASS |

The validator did not optimize, rank, retrain, or alter the submitted strategy.
