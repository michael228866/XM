# Generation 21 walk-forward validation

Overall: FAIL

Internal chronological validation quality: PASS

Final untouched-test validity: FAIL_pending_future_data

| Check | Verdict | Evidence | Failure or reason for pass | Required validation correction |
|---|---|---|---|---|
| chronology | PASS | gold_generation21_new_information.py:add_microstructure_features,add_cross_market_features | All ablations fit before each scored fold; microstructure uses completed bars and XAG uses backward-only as-of alignment. | None |
| feature_leakage | PASS | gold_generation21_new_information.json:feature_sets,source_inventory | No return target, outcome, exit offset, future calendar value, forward merge, or fabricated unavailable source enters a model. | None |
| label_maturity | PASS | gold_generation21_walk_forward_validation.json:boundary_audit | Exact event label ends precede calibration and policy partitions for all four ablations and all folds. | None |
| oof_predictions | PASS | gold_generation21_new_information.json:fold_results.*.fixed_cohort_prediction_ledger | Every stored score ledger exactly covers the frozen executable cohort and independently reproduces reported ranking metrics. | None |
| calibration | PASS | gold_generation21_new_information.py:score_fixed_cohort | P(net_R>0) isotonic calibration uses only the purged chronological calibration partition; Brier/ECE recompute exactly. | None |
| threshold_selection | PASS | gold_generation21_new_information.json:information_gate,candidate_construction | The information gate was predeclared, all versions failed it, and no post-hoc feature or selector became a candidate. | None |
| purge_embargo | PASS | gold_generation21_new_information.json:model_diagnostics | Fit and calibration labels end strictly before the next partition; outer fold training retains the horizon-tail purge. | None |
| holdout_contamination | FAIL | gold_generation21_new_information.json:development_history_policy | All inspected history is development data and no untouched final test has yet arrived after the new cutoff. | Wait for the registered future interval; do not inspect outcomes before a candidate is frozen. |
| recent_period_reuse | PASS | gold_generation21_forward_protocol.json:untouched_forward_cutoff_utc | Gen21 did not evaluate recent strategy outcomes and recorded a future cutoff later than the experiment run. | None |
| execution_alignment | PASS | gold_generation21_new_information.json:control_baseline | The experiment ranks the unchanged non-overlapping Gen17 cohort; independent cost, reward, PF, PnL, DD, and occupancy checks reconcile. | None |
| cost_assumptions | PASS | gold_generation21_new_information.json:frozen_comparability | Observed spread/fallback-30, 5-point base extra cost, and 10-point stress cost are unchanged and independently reconciled. | None |
| multiple_testing_risk | FAIL | gold_generation21_new_information.json:selection_inventory | Four feature ablations plus post-hoc univariate diagnostics were inspected after prior generations without an evaluated untouched test. | Test only a preregistered frozen hypothesis on genuinely new forward data. |

## Independent metric reconciliation

| Period | Trades | Days | Trades/day | Wins/Losses | WR | PF | Mean-R | PnL | Max DD | Stress PF | Reconciled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gen20_control:2018_2020 | 47 | 775 | 0.061 | 23/24 | 48.94% | 0.618 | -0.1987 | -125.56 | -13.42% | 0.590 | yes |
| gen20_control:2021_2022 | 132 | 516 | 0.256 | 83/49 | 62.88% | 1.105 | 0.0398 | 67.01 | -13.73% | 1.044 | yes |
| gen20_control:2023_2024 | 27 | 516 | 0.052 | 18/9 | 66.67% | 1.392 | 0.1330 | 49.67 | -5.56% | 1.328 | yes |
| gen20_control:selection_pooled | 206 | 1807 | 0.114 | 124/82 | 60.19% | 0.994 | -0.0024 | -20.62 | -20.41% | 0.943 | yes |

No feature family passed the frozen information gate. No strategy candidate was constructed or promoted.

Submitted historical evidence is valid for internal chronological development only and invalid as a final untouched OOS claim.
