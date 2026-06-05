# Precious Metals Axis Research

This folder is for building a multi-symbol model around precious metals without changing the live `gemini.py`.

Core idea:
- Use `GOLD#` as the anchor market.
- Add related metals only when spread cost is reasonable for the chosen timeframe.
- Train with normalized features so different price scales can share one model.
- Keep per-symbol calibration for thresholds, spread limits, TP/SL, and minimum timeframe.

Initial MT5 probe result:

| Symbol | Best initial use | Reason |
|---|---|---|
| `GOLD#` | M1/M5 | Lowest spread cost versus ATR |
| `SILVER#` | M5+ | M1 cost is high, M5 is acceptable |
| `XAUEUR#` | M5+ | Better as a gold cross validation asset |
| `XPTUSD#` | M30+ | M1/M5 costs are too high |
| `XPDUSD#` | M30/H1+ | M1/M5 costs are too high |

Recommended model shape:
- Shared model trained on ATR-normalized features.
- Symbol identity as a feature, not separate hand-tuned code paths.
- Asset-specific gates for spread/ATR and minimum timeframe.
- Candidate promotion only if both anchor GOLD and at least one non-GOLD metal pass validation.

Run the probe:

```powershell
python .\precious_metals_axis_research\probe_precious_metals_axis.py
```

Run the GOLD-only smoke test before expanding to other metals:

```powershell
python .\precious_metals_axis_research\test_gold_axis_candidate.py
```

Run a first per-symbol smoke model on the new datasets:

```powershell
python .\precious_metals_axis_research\axis_symbol_smoke.py
```

Run the first shared model across GOLD#, SILVER#, and XAUEUR#:

```powershell
python .\precious_metals_axis_research\axis_shared_model.py
```

Train the all-metals H1 shared model:

```powershell
python .\precious_metals_axis_research\train_all_metals_h1_shared.py
```

Walk-forward check the best all-metals shared candidates:

```powershell
python .\precious_metals_axis_research\walk_forward_all_metals_shared.py
```

Optimize the XAUEUR# candidate inside the all-metals shared model:

```powershell
python .\precious_metals_axis_research\optimize_xaueur_shared_walk_forward.py
```

Prepare and run the multi-metal trading entrypoint:

```powershell
python .\precious_metals_axis_research\multi_precious_metals_trader.py --prepare-models
python .\precious_metals_axis_research\multi_precious_metals_trader.py --once
```

The multi-metal trader defaults to dry-run. Real orders require `--live`.

Run symbol-specific base timeframe smoke tests:

```powershell
python .\precious_metals_axis_research\axis_timeframe_smoke.py
```

Optimize the strongest current non-GOLD candidate:

```powershell
python .\precious_metals_axis_research\optimize_xaueur_m5.py
```

Run the cost-aware version using CSV spread:

```powershell
python .\precious_metals_axis_research\cost_aware_xaueur_m5.py
```

Run cost-aware walk-forward checks:

```powershell
python .\precious_metals_axis_research\walk_forward_xaueur_m5_cost.py
```

Run H1 cost-aware smoke tests for non-GOLD metals:

```powershell
python .\precious_metals_axis_research\long_tf_cost_smoke.py
```

Run walk-forward for the best long-timeframe candidates:

```powershell
python .\precious_metals_axis_research\walk_forward_long_tf_cost.py
```

Optimize the SILVER# H1 long-timeframe candidate:

```powershell
python .\precious_metals_axis_research\optimize_silver_h1_cost.py
```

Optimize SILVER# H1 directly on walk-forward folds:

```powershell
python .\precious_metals_axis_research\optimize_silver_h1_walk_forward.py
```

Stress-test and refine the SILVER# H1 walk-forward candidate:

```powershell
python .\precious_metals_axis_research\stress_test_silver_h1_cost.py
python .\precious_metals_axis_research\optimize_silver_h1_stress.py
python .\precious_metals_axis_research\refine_silver_h1_stress_fold2.py
```

Run extended SILVER# / XAUEUR# battle-readiness checks:

```powershell
python .\precious_metals_axis_research\readiness_silver_xaueur.py
python .\precious_metals_axis_research\optimize_silver_xaueur_readiness.py
python .\precious_metals_axis_research\optimize_silver_regime_readiness.py
python .\precious_metals_axis_research\refine_silver_regime_readiness.py
python .\precious_metals_axis_research\final_silver_xaueur_robustness.py
python .\precious_metals_axis_research\optimize_training_profiles_silver_xaueur.py
python .\precious_metals_axis_research\select_stable_training_profiles.py
python .\precious_metals_axis_research\train_selected_profile_models.py
```

Latest extended readiness result:
- `XAUEUR#` passes the 3x-cost battle gate across 3/3 folds: +13.27R, 41 trades, 82.93% win rate, worst fold +1.40R, recent paper fold +6.19R. Active research params: conf=0.56, tp/sl=2.6/4.8 ATR, hold=216 H1 bars, both directions.
- `SILVER#` only passes after adding a low-volatility regime filter: +16.68R, 5/5 positive folds, 5/5 passed folds, 62 trades, 64.52% win rate, PF 2.05, worst fold +1.11R, recent paper fold +3.15R. Active research params: conf=0.56, tp/sl=5.2/5.2 ATR, hold=216 H1 bars, long only, VOLA_RATIO <= 1.2, trend score >= 0.

Final robustness result:
- `XAUEUR#` remains valid through 5x spread stress: at 5x, +12.72R, 3/3 positive folds, 3/3 passed folds.
- `SILVER#` is valid through 3x spread stress only: at 3x, +16.68R and 5/5 folds passed; at 4x and 5x it fails the gate. Keep SILVER# risk smaller than XAUEUR# and require clean live-paper logs before increasing size.
- Parameter-neighborhood pass rates are narrow: `SILVER#` 32/288 variants passed, `XAUEUR#` 12/216 variants passed. Treat both as exact calibrated candidates, not broad always-on models.

Selected training-profile result:
- `SILVER#`: keep the current symbol model profile, but use the stable low-volatility params: conf=0.52, tp/sl=6.0/6.0 ATR, hold=336 H1 bars, VOLA_RATIO <= 1.0. It passes 1x/2x/3x cost gates, but fails 4x/5x.
- `XAUEUR#`: switch the shared model to `smooth_more_trees`: conf=0.54, tp/sl=2.2/4.2 ATR, hold=288 H1 bars, both directions. It passes all 1x through 5x cost gates.
- Selected model files: `silver_h1_regime_selected_xgb.json` and `all_metals_h1_smooth_more_trees_xgb.json`.

Run the four-actual-metal main-candidate check for platinum and palladium:

```powershell
python .\precious_metals_axis_research\optimize_xpt_xpd_main_candidates.py
```

Run the extended XPT/XPD slower-timeframe search:

```powershell
python .\precious_metals_axis_research\optimize_xpt_xpd_extended_timeframes.py
```

Refine XPTUSD# H4 fold-level trade coverage:

```powershell
python .\precious_metals_axis_research\refine_xpt_h4_fold_coverage.py
```

Test XPDUSD# with alternate direction targets and time-stop exits:

```powershell
python .\precious_metals_axis_research\xpd_alternate_target_exit.py
```

Build the four-metal main promotion plan:

```powershell
python .\precious_metals_axis_research\build_four_metal_main_promotion.py
```

Latest four-metal main promotion verdict:

| Symbol | Role | Gate result | Notes |
|---|---|---|---|
| `GOLD#` | Main anchor | Existing dedicated GOLD line | Keep as the anchor metal. |
| `SILVER#` | Main core | Passes 1x/2x/3x cost gates | Fails 4x/5x cost stress, so keep regime filter and lower size. |
| `XPTUSD#` | Slow main | Refined H4 passes 1x-5x cost gates; 3x = +12.97R | Slow H4 main line, 20% research allocation. |
| `XPDUSD#` | Slow main | Alternate H12 target passes 1x-3x cost gates; 3x = +9.22R | Slow H12 main line, 20% research allocation. |

`XAUEUR#` is a robust gold-cross auxiliary model, not a separate fourth metal. It can help diversify gold exposure, but it should not be counted as platinum or palladium exposure.

The four actual metals are now promoted to research-main lines in `four_metal_main_promotion_plan.json`. This still does not modify `gemini.py` or place live orders; it defines the strategy lines to use for forward paper validation.

Build the current precious-metals composite trading plan:

```powershell
python .\precious_metals_axis_research\build_precious_metals_composite.py
```

Train each metal separately with its own candidate timeframes and parameters:

```powershell
python .\precious_metals_axis_research\train_each_metal_custom.py
```

Train each metal with richer multi-timeframe reference features:

```powershell
python .\precious_metals_axis_research\train_each_metal_mtf_reference.py
```

Outputs:
- `precious_metals_axis_probe.csv`
- `gold_axis_smoke_test.json`
- `gold_axis_smoke_test.md`
- `axis_symbol_smoke_results.csv`
- `axis_symbol_smoke_results.json`
- `axis_symbol_smoke_report.md`
- `axis_shared_model_results.csv`
- `axis_shared_model_results.json`
- `axis_shared_model_report.md`
- `axis_shared_precious_metals_xgb.json`
- `all_metals_h1_shared_xgb.json`
- `all_metals_h1_shared_results.csv`
- `all_metals_h1_shared_results.json`
- `all_metals_h1_shared_report.md`
- `all_metals_h1_shared_walk_forward.csv`
- `all_metals_h1_shared_walk_forward.json`
- `all_metals_h1_shared_walk_forward.md`
- `xaueur_shared_walk_forward_optimized_results.csv`
- `xaueur_shared_walk_forward_optimized_results.json`
- `xaueur_shared_walk_forward_optimized_report.md`
- `xaueur_shared_walk_forward_best.json`
- `multi_precious_metals_trader.py`
- `multi_metal_trader_signal_log.csv`
- `silver_h1_strict_live_xgb.json`
- `silver_h1_strict_live_xgb.metadata.json`
- `axis_timeframe_smoke_results.csv`
- `axis_timeframe_smoke_results.json`
- `axis_timeframe_smoke_report.md`
- `xaueur_m5_optimized_results.csv`
- `xaueur_m5_optimized_results.json`
- `xaueur_m5_optimized_report.md`
- `xaueur_m5_best_candidate.json`
- `xaueur_m5_cost_aware_results.csv`
- `xaueur_m5_cost_aware_results.json`
- `xaueur_m5_cost_aware_report.md`
- `xaueur_m5_cost_aware_best.json`
- `xaueur_m5_cost_walk_forward.csv`
- `xaueur_m5_cost_walk_forward.json`
- `xaueur_m5_cost_walk_forward.md`
- `long_tf_cost_smoke_results.csv`
- `long_tf_cost_smoke_results.json`
- `long_tf_cost_smoke_report.md`
- `long_tf_cost_best_candidates.json`
- `long_tf_cost_walk_forward.csv`
- `long_tf_cost_walk_forward.json`
- `long_tf_cost_walk_forward.md`
- `silver_h1_cost_optimized_results.csv`
- `silver_h1_cost_optimized_results.json`
- `silver_h1_cost_optimized_report.md`
- `silver_h1_cost_best_candidate.json`
- `silver_h1_walk_forward_optimized_results.csv`
- `silver_h1_walk_forward_optimized_results.json`
- `silver_h1_walk_forward_optimized_report.md`
- `silver_h1_walk_forward_best_candidate.json`
- `silver_h1_cost_stress.csv`
- `silver_h1_cost_stress.json`
- `silver_h1_cost_stress.md`
- `silver_h1_stress_optimized_results.csv`
- `silver_h1_stress_optimized_results.json`
- `silver_h1_stress_optimized_report.md`
- `silver_h1_stress_best_candidate.json`
- `silver_h1_stress_fold2_refine_results.csv`
- `silver_h1_stress_fold2_refine_results.json`
- `silver_h1_stress_fold2_refine_report.md`
- `silver_h1_stress_fold2_refine_best.json`
- `silver_xaueur_readiness_folds.csv`
- `silver_xaueur_readiness.json`
- `silver_xaueur_readiness.md`
- `silver_xaueur_readiness_optimized_results.csv`
- `silver_xaueur_readiness_optimized_results.json`
- `silver_xaueur_readiness_optimized_report.md`
- `silver_xaueur_readiness_optimized_best.json`
- `silver_regime_readiness_results.csv`
- `silver_regime_readiness_results.json`
- `silver_regime_readiness_report.md`
- `silver_regime_readiness_best.json`
- `silver_regime_refine_results.csv`
- `silver_regime_refine_results.json`
- `silver_regime_refine_report.md`
- `silver_regime_refine_best.json`
- `final_silver_xaueur_robustness.csv`
- `final_silver_xaueur_robustness.json`
- `final_silver_xaueur_robustness.md`
- `training_profile_optimization_results.csv`
- `training_profile_optimization_results.json`
- `training_profile_optimization_report.md`
- `training_profile_optimization_best.json`
- `training_profile_stable_selection.json`
- `training_profile_stable_selection.md`
- `xpt_xpd_main_candidate_results.csv`
- `xpt_xpd_main_candidate_results.json`
- `xpt_xpd_main_candidate_report.md`
- `xpt_xpd_main_candidate_best.json`
- `xpt_xpd_extended_timeframe_results.csv`
- `xpt_xpd_extended_timeframe_results.json`
- `xpt_xpd_extended_timeframe_report.md`
- `xpt_xpd_extended_timeframe_best.json`
- `xpt_h4_fold_coverage_results.csv`
- `xpt_h4_fold_coverage_results.json`
- `xpt_h4_fold_coverage_report.md`
- `xpt_h4_fold_coverage_best.json`
- `xpd_alternate_target_exit_results.csv`
- `xpd_alternate_target_exit_results.json`
- `xpd_alternate_target_exit_report.md`
- `xpd_alternate_target_exit_best.json`
- `four_metal_main_promotion_plan.json`
- `four_metal_main_promotion_plan.md`
- `silver_h1_regime_selected_xgb.json`
- `silver_h1_regime_selected_xgb.metadata.json`
- `all_metals_h1_smooth_more_trees_xgb.json`
- `all_metals_h1_smooth_more_trees_xgb.metadata.json`
- `precious_metals_composite_plan.json`
- `precious_metals_composite_plan.md`
- `each_metal_custom_results.csv`
- `each_metal_custom_results.json`
- `each_metal_custom_report.md`
- `each_metal_custom_best_by_symbol.json`
- `each_metal_mtf_reference_results.csv`
- `each_metal_mtf_reference_results.json`
- `each_metal_mtf_reference_report.md`
- `each_metal_mtf_reference_best_by_symbol.json`

No live settings are modified.
