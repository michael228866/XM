# Generation 16 walk-forward adversarial validation

Overall: **FAIL**
Submitted performance claim: **invalid**

| Check | Verdict | Evidence |
|---|---|---|
| chronology | PASS | Each fold trains with training_frame(history, fold_start); holdout and recent are evaluated only after the selection ranking is frozen. |
| feature leakage | PASS | Historical and MT5 feature builders shift M1 features one bar and higher-timeframe trends one completed bar before scoring. |
| label maturity | PASS | All recorded fit labels end before calibration and all calibration labels end before policy. |
| OOF predictions | PASS | All 72 predeclared candidates have fold-level ledgers; predictions come from models fit strictly before each evaluated fold. |
| calibration | PASS | Chronological fit, isotonic-calibration, and policy segments are disjoint; the policy segment is not reused to fit the calibrator. |
| threshold selection | PASS | P(TP-first)=0.60 and Expected-R=0 are fixed; family/context/top-k architectures are selected on 2018-2024 only, before holdout/recent evaluation. |
| purge/embargo | PASS | Outer training removes the final H=90 rows; internal fit/calibration/policy boundaries additionally require event label_end before the next stage. |
| holdout contamination | FAIL | The report explicitly identifies 2025-2026-05 as a reused historical holdout already inspected by earlier generations; it is not untouched. |
| recent-period reuse | PASS | The repeatedly observed 2026 recent period is explicitly monitoring-only and is excluded from candidate/model selection and promotion claims. |
| execution alignment | PASS | First-touch starts at offset 1, same-bar TP/SL ties are stop-first, and every persisted ledger is single-position and non-overlapping. |
| cost assumptions | PASS | Base uses 30 spread points plus 5 extra points; stress uses the same entries with 10 extra points. Ledgers reconcile and costs are applied once in stop-risk units. |
| multiple-testing risk | FAIL | Seventy-two Generation 16 architectures follow many prior generations, while the nominal holdout is already reused; there is no fresh untouched outer interval for a positive performance claim. |

## Metric reconciliation

| Period | Profile | Trades | Trades/day | Wins | Losses | Timeouts | Win | PF | PnL | Mean-R | DD | Match |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2018_2020 | base | 93 | 0.120 | 46 | 47 | 0 | 49.46% | 0.57 | -253.73 | -0.2198 | -25.37% | PASS |
| 2018_2020 | cost_stress | 93 | 0.120 | 46 | 47 | 0 | 49.46% | 0.54 | -273.60 | -0.2405 | -27.36% | PASS |
| 2021_2022 | base | 232 | 0.450 | 115 | 117 | 0 | 49.57% | 0.59 | -505.02 | -0.2114 | -50.86% | PASS |
| 2021_2022 | cost_stress | 232 | 0.450 | 115 | 117 | 0 | 49.57% | 0.55 | -544.82 | -0.2372 | -54.67% | PASS |
| 2023_2024 | base | 23 | 0.045 | 16 | 7 | 0 | 69.57% | 1.32 | 31.44 | 0.1001 | -6.06% | PASS |
| 2023_2024 | cost_stress | 23 | 0.045 | 16 | 7 | 0 | 69.57% | 1.25 | 24.93 | 0.0804 | -6.22% | PASS |
| 2025_2026_05_holdout | base | 228 | 0.655 | 110 | 118 | 0 | 48.25% | 0.59 | -502.74 | -0.2137 | -50.60% | PASS |
| 2025_2026_05_holdout | cost_stress | 228 | 0.655 | 110 | 118 | 0 | 48.25% | 0.57 | -525.71 | -0.2284 | -52.83% | PASS |
| 2026_recent | base | 1 | 0.013 | 1 | 0 | 0 | 100.00% | inf | 10.22 | 0.7297 | 0.00% | PASS |
| 2026_recent | cost_stress | 1 | 0.013 | 1 | 0 | 0 | 100.00% | inf | 10.12 | 0.7227 | 0.00% | PASS |

## Minimum validation-only correction

- Do not treat the reused 2025 holdout or recent monitoring window as untouched evidence.
- Pre-register and freeze a qualified future candidate before collecting a new embargoed forward interval that no research iteration has inspected.
- Run the same executable-event and cost-stress reconciliation once on that new interval without changing models, families, thresholds, or context rules.

The validator did not optimize, resweep, or change the strategy.
