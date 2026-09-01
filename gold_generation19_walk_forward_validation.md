# Generation 19 walk-forward validation

Overall: FAIL

Internal chronological validation quality: PASS

Final untouched-test validity: FAIL

| Check | Verdict | Evidence | Failure or reason for pass | Required validation correction |
|---|---|---|---|---|
| chronology | PASS | gold_generation19_cost_aware.py:334,399 | Models, calibration, margin choice, and evaluated application are ordered train -> inner policy -> next block. | None |
| feature_leakage | PASS | gold_generation19_cost_aware.py:69 | Economic state uses entry-bar spread, shifted ATR/features, calibrated P(win), and no exit/outcome field as a live input. | None |
| label_maturity | PASS | gold_generation17_cross_regime.py:282 | Exact exit offsets end before each next stage boundary; horizon remains 90. | None |
| oof_predictions | PASS | gold_generation19_cost_aware.py:399 | Every claimed fold is scored by the frozen Gen17 recipe fitted strictly before the fold; Gen17 absolute ledgers are exactly reproduced or the run aborts. | None |
| calibration | PASS | gold_generation17_cross_regime.py:282 | Isotonic calibration is fit only on the chronological calibration partition. Poor cross-regime calibration is a performance failure, not leakage. | None |
| threshold_selection | PASS | gold_generation19_cost_aware.py:334 | Five predeclared safety margins are selected only on the prior policy tail; evaluated folds and monitored intervals do not select their own margin. | None |
| purge_embargo | PASS | gold_generation17_cross_regime.py:306 | Event-specific maximum label ends are strictly earlier than calibration/policy starts; training_frame also removes the horizon tail. | None |
| holdout_contamination | FAIL | gold_generation19_cost_aware.json:development_history_policy | The 2025 interval was viewed by prior generations and is explicitly development diagnostic data, so it cannot support a final OOS claim. | Freeze a candidate and collect a genuinely new future interval. |
| recent_period_reuse | PASS | gold_generation19_cost_aware.json:development_diagnostics | The recent interval is explicitly monitoring/development only, excluded from selection, discovery, and promotion claims. | None |
| execution_alignment | PASS | gold_generation16_independent_families.py:634 | Stored ledgers reconcile, are chronological and non-overlapping, use exit offsets <=90, and retain stop-first HIGH/LOW execution. | None |
| cost_assumptions | PASS | gold_generation19_transaction_cost_audit.json:verdict | GOLD# point units, Bid bars, observed spreads, missing-spread fallback, extra costs, and adverse-cost stress are traced and rewards reconcile. | None |
| multiple_testing_risk | FAIL | gold_generation19_cost_aware.json:selection_inventory | Gen19 inventories 10 dynamic policies per fold plus four diagnostic exit profiles, but no genuinely untouched final interval remains to absorb accumulated generation-level data snooping. | Use the same untouched future interval; do not select on 2025/recent diagnostics. |

## Metric reconciliation

| Strategy / period | Trades | Trades/day | Wins | Losses | Timeouts | Win rate | PF | Mean-R | PnL | Max DD | Stress PF | Reconciled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gen17_parent:2018_2020 | 47 | 0.061 | 23 | 24 | 0 | 48.94% | 0.618 | -0.1987 | -125.56 | -13.42% | 0.590 | yes |
| gen19_dynamic_short_trend:2018_2020 | 66 | 0.085 | 32 | 34 | 1 | 48.48% | 0.857 | -0.0768 | -74.91 | -15.20% | 0.797 | yes |
| gen19_dynamic_existing_portfolio:2018_2020 | 469 | 0.605 | 224 | 245 | 3 | 47.76% | 0.662 | -0.1812 | -707.16 | -71.63% | 0.626 | yes |
| gen17_parent:2021_2022 | 132 | 0.256 | 83 | 49 | 0 | 62.88% | 1.105 | 0.0398 | 67.01 | -13.73% | 1.044 | yes |
| gen19_dynamic_short_trend:2021_2022 | 197 | 0.382 | 115 | 82 | 0 | 58.38% | 0.912 | -0.0375 | -110.28 | -21.27% | 0.864 | yes |
| gen19_dynamic_existing_portfolio:2021_2022 | 204 | 0.395 | 119 | 85 | 0 | 58.33% | 0.914 | -0.0366 | -111.61 | -22.20% | 0.867 | yes |
| gen17_parent:2023_2024 | 27 | 0.052 | 18 | 9 | 0 | 66.67% | 1.392 | 0.1330 | 49.67 | -5.56% | 1.328 | yes |
| gen19_dynamic_short_trend:2023_2024 | 127 | 0.246 | 41 | 86 | 0 | 32.28% | 0.616 | -0.2758 | -398.32 | -41.73% | 0.561 | yes |
| gen19_dynamic_existing_portfolio:2023_2024 | 111 | 0.215 | 60 | 51 | 1 | 54.05% | 0.885 | -0.0529 | -86.89 | -12.88% | 0.854 | yes |
| gen19_dynamic_existing_portfolio:2025_2026_05_development | 3125 | 8.980 | 1716 | 1409 | 0 | 54.91% | 0.912 | -0.0400 | -862.77 | -86.78% | 0.896 | yes |
| gen17_parent:2025_2026_05_development | 54 | 0.155 | 32 | 22 | 0 | 59.26% | 1.098 | 0.0403 | 26.79 | -8.97% | 1.083 | yes |
| gen19_dynamic_existing_portfolio:2026_recent_development | 448 | 5.973 | 258 | 190 | 0 | 57.59% | 0.997 | -0.0014 | -40.51 | -21.45% | 0.980 | yes |
| gen17_parent:2026_recent_development | 59 | 0.787 | 32 | 27 | 0 | 54.24% | 0.853 | -0.0680 | -58.75 | -7.86% | 0.836 | yes |

The submitted performance claim is valid only as internal chronological development evidence. It is invalid as a final untouched OOS claim. No optimization or production recommendation was made by this validator.
