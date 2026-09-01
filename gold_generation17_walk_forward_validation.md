# Generation 17 walk-forward adversarial validation

Overall: **FAIL**

| Validation scope | Verdict |
|---|---|
| Internal chronological validation quality | PASS |
| Final untouched-test validity | FAIL |

| Check | Verdict | Evidence | Required correction |
|---|---|---|---|
| chronology | PASS | Models use training_frame(history, fold_start); ranking is frozen after 2018-2024 folds and before reused 2025/recent diagnostics. | None. |
| feature leakage | PASS | M1/MTF inputs are shifted; bar structure uses t-1; family outcome state appears only at exit_offset+1 in actual maturity order; raw ATR is excluded from model inputs. | None. |
| label maturity | PASS | Every recorded fit label ends before calibration, and every calibration label ends before policy. | None. |
| OOF predictions | PASS | All five predeclared candidates have executable ledgers in each of three chronological development folds; each fold model trains only before its fold. | None. |
| calibration | PASS | Chronological 60/20/20 fit, isotonic calibration, and policy segments are disjoint with label-end purging. | None. |
| threshold selection | PASS | P(TP-first)=0.60, Expected-R=0, and top-2/expert/day are fixed; only three single families and two fixed portfolios were compared. | None. |
| purge/embargo | PASS | Outer training drops the last H=90 rows; internal boundaries require exact label end before the next stage. | None. |
| holdout contamination | FAIL | The experiment correctly labels 2025 as repeatedly inspected development data; it cannot support an untouched OOS claim. | Freeze a candidate, then collect a new embargoed future interval that no research decision has observed. |
| recent-period reuse | PASS | The reused recent interval is evaluated after selection and is explicitly monitoring/development-only, not a selection or promotion gate. | None. |
| execution alignment | PASS | HIGH/LOW first-touch starts at offset 1, ties are stop-first, position occupancy is enforced, and all persisted metrics reconcile. | None. |
| cost assumptions | PASS | Base applies the existing 30-point spread plus 5 extra points; stress keeps identical entries and raises extra points to 10. | None. |
| multiple-testing risk | FAIL | Generation 17 tested five fixed configurations after many prior generations; a raw-ATR implementation was also rejected pre-freeze for specification mismatch. All historical/recent intervals are development data. | Pre-register one frozen candidate and obtain a new untouched future outer test; do not use that interval for further tuning. |

## Metric reconciliation

| Period | Profile | Trades | Trades/day | Wins | Losses | Timeouts | Win | PF | PnL | Mean-R | Max DD | Match |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2018_2020 | base | 47 | 0.061 | 23 | 24 | 0 | 48.94% | 0.60 | -131.72 | -0.2095 | -13.87% | PASS |
| 2018_2020 | cost_stress | 47 | 0.061 | 23 | 24 | 0 | 48.94% | 0.57 | -141.81 | -0.2272 | -14.82% | PASS |
| 2021_2022 | base | 132 | 0.256 | 83 | 49 | 0 | 62.88% | 0.97 | -26.17 | -0.0101 | -14.27% | PASS |
| 2021_2022 | cost_stress | 132 | 0.256 | 83 | 49 | 0 | 62.88% | 0.92 | -64.16 | -0.0316 | -15.01% | PASS |
| 2023_2024 | base | 27 | 0.052 | 18 | 9 | 0 | 66.67% | 1.20 | 24.87 | 0.0692 | -5.89% | PASS |
| 2023_2024 | cost_stress | 27 | 0.052 | 18 | 9 | 0 | 66.67% | 1.15 | 17.69 | 0.0505 | -6.08% | PASS |
| 2025_2026_05_development | base | 54 | 0.155 | 32 | 22 | 0 | 59.26% | 1.07 | 16.73 | 0.0271 | -9.28% | PASS |
| 2025_2026_05_development | cost_stress | 54 | 0.155 | 32 | 22 | 0 | 59.26% | 1.05 | 12.06 | 0.0210 | -9.49% | PASS |
| 2026_recent_development | base | 59 | 0.787 | 32 | 27 | 0 | 54.24% | 0.84 | -63.87 | -0.0747 | -7.93% | PASS |
| 2026_recent_development | cost_stress | 59 | 0.787 | 32 | 27 | 0 | 54.24% | 0.82 | -70.40 | -0.0832 | -8.48% | PASS |

The validator did not optimize, resweep, retrain, or modify the strategy.
