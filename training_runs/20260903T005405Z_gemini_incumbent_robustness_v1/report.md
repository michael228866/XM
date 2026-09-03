# GEMINI INCUMBENT ROBUSTNESS ATTRIBUTION V1

Status: **research_only diagnostic**. No candidate was selected and no operational artifact changed.

## Evidence classification

All historical intervals and live rows are development/monitoring evidence. The old forward cutoff remains contaminated; no new cutoff was created.

## Fold-by-fold fixed-control results

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

W0 is the byte-preserved Gen core-gate control. W1-W3 change only the historical training-window length.

## Original recent evidence versus W0 OOF and live monitoring

| Evidence | Scored interval | Trades | Trades/day | WR | PF | Mean-R | PnL | Max DD | Classification |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A_original_recent_validation | 2026-04-01 to 2026-06-01 | 20 | 0.3279 | 70.00% | 1.7557 | n/a | 82.03 | -0.07 | selected development evidence; formerly called validation/test; not untouched |
| B_original_recent_test | 2026-06-01 to 2026-08-25T05:49:00 | 15 | 0.1760 | 66.67% | 1.3447 | n/a | 34.25 | -0.07 | selected development evidence; formerly called validation/test; not untouched |
| C_W0_OOF_2018_2020 | 2018-01-02T00:00:00 to 2020-12-31T18:50:00 | 189 | 0.1724 | 25.40% | 0.3725 | -0.4460 | -84.29 | -88.42 | genuine chronological OOF development evidence |
| C_W0_OOF_2021_2022 | 2021-01-04T01:00:00 to 2022-12-30T23:57:00 | 0 | 0.0000 | 0.00% | 0.0000 | 0.0000 | 0.00 | 0.00 | genuine chronological OOF development evidence |
| C_W0_OOF_2023_2024 | 2023-01-03T01:00:00 to 2024-12-31T20:00:00 | 0 | 0.0000 | 0.00% | 0.0000 | 0.0000 | 0.00 | 0.00 | genuine chronological OOF development evidence |
| D_W0_OOF_pooled | 2018-01-02 to 2024-12-31 | 189 | 0.0739 | 25.40% | 0.3725 | -0.4460 | -84.29 | -88.42 | pooled chronological OOF development evidence |
| E_current_live_monitoring | 2026-08-25T06:52:00+00:00 to 2026-09-03T00:54:00+00:00 | 3 | 0.3428 | 66.67% | 1.5314 | 0.1286 | 5.24 | -9.86 | monitoring_only_contaminated_for_future_gate_selection |

Original PnL is risk-scaled account currency and DD is percent; OOF PnL/DD are net R. They are intentionally not treated as the same unit.

## Verified methodology differences

| Area | Original recent pipeline | Genuine OOF control | Materiality | Can explain gap |
|---|---|---|---|---|
| training history | 2025-01-01 to each 2026 score boundary; recent rolling fit | expanding 2014 history before each 2018-2024 fold | high | yes; different fitted models and regimes |
| training-window length | about 15-17 months | about 4, 7, and 9 years | high | tested directly by frozen W0-W3 diagnostic |
| model and class weighting | binary logistic, 220 trees, lr .05, depth 4, min-child 80, balanced binary weights, seed 42 | same model family, parameters, weights, and seed | low | no verified architecture-parameter difference |
| selection | 288 threshold/TP/SL/hold/session combinations selected on Apr-May 2026 | reported fixed T0_R0, though parent run also evaluated 20 threshold/RSI combinations | high | yes; old 70% validation is selection-conditioned and optimistic |
| artifact identity | exact operational artifact scored only Jun-Aug 2026 test | three historical fold replicas; exact operational artifact not scored | critical | yes; 25.40% is architecture replication evidence, not exact-artifact evidence |
| feature formulas/order | build_feature_frame uses add_indicators, shifted 31 features, backward MTF as-of | prepare_barrier_data uses same add_indicators and shifted 31-feature order | low | no verified formula difference |
| feature data source | dynamic MT5 copy_rates; raw snapshot not retained | local XM CSV exports with hashes | unknown | possible but not quantifiable because original raw snapshot is missing |
| timezone/session semantics | MT5 epoch converted to naive timestamp | naive CSV export timestamps; exact UTC mapping unproven | potentially high | unresolved; session-hour equivalence cannot be proven |
| label | clean-window long label, horizon 240, TP 1.8 ATR/min1.0, SL 1.2 ATR/min0.8 | same BARRIER_TARGET implementation | low | no verified label-formula difference |
| label maturity | last 240 rows removed before validation/final fit | last 240 pre-fold rows purged and latest label bar verified | low | no verified maturity difference |
| execution path | legacy simulator called without HIGH/LOW; close-only threshold exits | precomputed M1 HIGH/LOW first-touch, stop-first same-bar | critical | yes; directly changes TP/SL ordering and realized WR/PF |
| signal episode order | filters applied before rising-edge state | threshold episode formed before session/RSI/spread filters | high | yes; signals becoming eligible later in a persistent probability run are treated differently |
| live entry semantics | legacy filtered rising edge | pre-filter threshold episode | high | neither is exact live behavior; live checks every new completed eligible bar when flat |
| risk/cooldown state | legacy drawdown/loss-streak guards including a 120-tick pause after three losses | 15-minute cooldown after every loss plus independently reconstructed daily/rolling guards | high | yes; executable identities and risk-scaled PF are not definitionally identical |
| spread/cost | fixed 30-point spread plus 5-point extra; no spread gate | observed spread when positive, 30-point fallback, 5-point extra, spread gate | medium | yes for PF/trade count; direction is measurable in retained ledgers |
| metric units | risk-scaled account-currency PnL and percent DD | unscaled net R, PF, Mean-R, and DD-R | high for PnL/DD | yes for numeric PnL/DD; WR remains comparable only after execution semantics match |
| software environment | exact package lock not retained | Python/package versions retained | unknown | cannot be excluded, but no evidence it is primary |

## Probability diagnosis

Absolute P(long)=0.75 calibration stable: **False**.
Probability ranking stable: **False**.
See probability_distribution.csv, calibration_summary.csv, calibration_buckets.csv, probability_buckets.csv, ranking_summary.csv, and ranking_deciles.csv.

## Time-proximity diagnosis

Three-fold Spearman recency vs WR: -0.8660; vs PF: -0.8660; vs Mean-R: 0.8660; vs trades/day: -0.8660.
These are descriptive correlations across only three folds and are not inferential evidence.

## Attribution conclusion

The pooled W0 result is generated by fold-specific expanding-history replicas, not the exact operational artifact. Its >=0.75 executable cohort is concentrated according to the fold table, and that cohort loses under observed/fallback costs and HIGH/LOW first-touch. The old 66-70% figures came from a recent-window, selection-conditioned pipeline with close-only exit detection.

Direct comparability with the old result: **partial**.
Current core classification: **methodology mismatch not yet resolved**.
Most stable predefined window diagnostically: **W2_trailing_18m**; this is not a production selection.
Exact operational artifact demonstrably bad: **False**.
Later W0 fold stronger than both earlier folds: **False**.
Older expanding history diagnostically degrades the architecture: **True**.

## Required conclusions

1. 25.40% cause: The pooled W0 result is generated by fold-specific expanding-history replicas, not the exact operational artifact. Its >=0.75 executable cohort is concentrated according to the fold table, and that cohort loses under observed/fallback costs and HIGH/LOW first-touch. The old 66-70% figures came from a recent-window, selection-conditioned pipeline with close-only exit detection.
2. Directly comparable to the old 66-70% result: partial.
3. Exact differences are enumerated in methodology_reconciliation.csv and the table above.
4. Material differences are training history, selection conditioning, artifact identity, exit path, signal-episode order, risk/cooldown state, spread/cost handling, metric units, and unresolved timestamp/data identity.
5. Exact current artifact demonstrably bad: False; The exact artifact has a retained 15-trade selected-period result and only a very small live monitoring sample; W0 historical scores came from different fold replicas.
6. Historical fold replicas are weak outside the recent selected evidence, but architecture robustness is not equivalent to exact-artifact quality.
7. Performance improves nearer 2025-2026: False within comparable W0 folds; the 2026 comparison remains confounded.
8. Older training history degrades the architecture: True under the predefined diagnostic rule.
9. Diagnostically most stable window: W2_trailing_18m; no production selection is made.
10. P(long)=0.75 calibration stable: False.
11. Probability ranking stable: False.
12. Classification: methodology mismatch not yet resolved.
13. Single next hypothesis: EXECUTION-SEMANTICS RECONCILIATION: on one frozen prediction cohort, reproduce the exact live barwise eligibility order, position re-entry behavior, observed spread, and HIGH/LOW exits, then compare it with both retained simulators without selecting parameters.

## Single next hypothesis (not implemented)

EXECUTION-SEMANTICS RECONCILIATION: on one frozen prediction cohort, reproduce the exact live barwise eligibility order, position re-entry behavior, observed spread, and HIGH/LOW exits, then compare it with both retained simulators without selecting parameters.

## Operational safety

`gemini.py` and `gold_long_recent_candidate_xgb.json` hashes were checked before and after and remained unchanged.
