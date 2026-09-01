# GEMINI FREQUENCY EXPANSION 1 — adversarial validation

Overall: **FAIL**

Internal chronological validation quality: **FAIL**
Final untouched-test validity: **FAIL**

| Check | Result | Evidence | Impact |
|---|---|---|---|
| chronology | FAIL | The current model was trained/selected on 2025-2026 data, then scored 2018-2024. | The backward replay cannot be interpreted as chronological OOS evidence. |
| feature leakage | PASS | Expansion rules use frozen production scores and completed-bar technical/MTF fields only; outcomes are not rule inputs. | No direct future outcome feature was identified in the expansion rules. |
| label maturity | PASS | Entries require mature first-touch outcomes and a positive exit offset; labels are used only for offline scoring. | The reported trade outcomes are mature within the available historical data. |
| OOF predictions | FAIL | The production score is not OOF for any 2018-2024 fold; model origin starts in 2025. | Winner/loser discrimination and expansion ranking are not independently estimated. |
| calibration | PASS | No new probability calibration is fitted or claimed in this study. | There is no calibration reuse layer, although the frozen production score remains non-OOF. |
| threshold selection | PASS | Production 0.75 is immutable; 0.65-0.75 and the small contextual rules were frozen before outcome calculation. | No post-result threshold rescue or broad sweep was performed. |
| purge/embargo | PASS | No expansion model is trained; executable positions are non-overlapping and production-preempted. | Overlapping event labels do not enter a fitted expansion model in this study. |
| holdout contamination | FAIL | All 2018-2024 folds were previously inspected development history; no untouched historical holdout remains. | The study cannot support a final promotion or shadow-performance claim. |
| recent-period reuse | FAIL | Model origin reports validation 2026-04-01T00:00:00+00:00 and test 2026-06-01T00:00:00+00:00; both were already inspected. | The production score embeds reused recent-development decisions. |
| execution alignment | FAIL | Non-overlap and production priority are causal, but exact broker account state, fills, slippage and export-time UTC mapping are unavailable. | The reconstructed ledger is comparable research execution, not exact MT5 execution. |
| cost assumptions | FAIL | Observed entry spread is used when valid and 30 points otherwise; exact commission/slippage and missing-spread costs are unobserved. | PF and Mean-R remain assumption-sensitive despite the 10-point extra-cost stress. |
| multiple-testing risk | FAIL | Six related portfolios are compared on repeatedly inspected development folds, and this repository contains many prior generations. | Selecting any apparent winner on these folds would have material research-overfitting risk. |

## Metric reconciliation

| Layer | Trades | Trades/day | WR | PF | Mean-R | PnL-R | Max DD-R |
|---|---:|---:|---:|---:|---:|---:|---:|
| Production | 763 | 0.2984 | 47.44% | 0.72 | -0.1504 | -114.78 | -116.16 |
| Diagnostic combined | 2088 | 0.8166 | 46.26% | 0.75 | -0.1350 | -281.85 | -284.57 |

Arithmetic reconciliation: **PASS**.

## Verdict

FAIL. No expansion candidate passed positive-expectancy discovery, and the reverse-time production scoring prevents an OOS claim. Do not create a sidecar, change production, or promote this research. The only valid next test is a fully frozen paper-shadow protocol on new post-cutoff data, without inspecting outcomes during collection.

No validation-only correction was applied, because the failures are evidence-design limitations rather than arithmetic/reporting defects.
