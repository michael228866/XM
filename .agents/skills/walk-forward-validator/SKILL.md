---
name: walk-forward-validator
description: Independently audit GOLD# quantitative experiments for chronological integrity, leakage, validation reuse, and executable-backtest alignment. Use after win-frequency, false-positive, or missed-winner research is frozen; do not optimize or tune the strategy.
---

# Walk-Forward Validator

Act as an adversarial reviewer. Decide whether the reported out-of-sample win rate and executable frequency are trustworthy; do not improve the candidate, search parameters, design filters, or replace `gemini.py`.

Audit frozen outputs from `win-frequency-optimizer`, `false-positive-miner`, and `missed-winner-miner`. The default strategy target is OOS win rate at least 60%; PF, positive expectancy/PnL, costs, and drawdown remain guardrails. These objectives never justify weakening an audit check.

## Evidence contract

- Read the experiment code, configuration, candidate inventory, reports, signal/event identities, prediction artifacts, and exact data boundaries. Do not accept a report's `promotion_pass`, `OOF`, or `past_only` field as proof without tracing the producing code.
- Record the code revision or diff, dataset identity or hash when available, run timestamp, timezone, feature cutoff, label end, train/calibration/selection/holdout/recent intervals, costs, and every adaptive choice.
- Recompute boundary assertions and representative metrics from raw artifacts when possible. Missing or irreproducible evidence makes the verdict `FAIL` with reason `not verifiable`, never an assumed pass.
- Keep the audit independent: inspect the submitted frozen candidate and state the minimum validation correction required. Do not run another candidate sweep or recommend better strategy parameters.

Use repository conventions as comparison points, not proof. Relevant implementations include `training_frame` and `fold_pass` in `gold_expected_r_walk_forward.py`, `sequential_event_indices` and past-only calibration in `gold_generation12_executable_events.py`, and `add_targets` plus `execution_realized_metrics` in `gold_generation11_execution_aligned.py`. Trace the actual candidate path when Gen13 or a later generation overrides them.

## Required checks

Return exactly `PASS` or `FAIL` for every check below. A check passes only when all applicable directions, experts, folds, and portfolio-union paths satisfy it.

1. **Chronology** - All fitting, feature transforms, routing rules, rolling statistics, champion state, and candidate choices use timestamps strictly earlier than the evaluated signal. Verify timezone ordering and that higher-timeframe features use only closed bars.
2. **Feature leakage** - Every model or rule input is available at decision time. Reject centered/global transforms, full-sample normalization or imputation, future extrema/returns, exit fields, outcome-derived clusters, and target-encoded features fit beyond the training boundary.
3. **Label maturity** - A row may enter fitting, calibration, or selection only after its complete first-touch label interval has ended. Verify `label_end < next_stage_start`, or an equivalent horizon purge, and exclude incomplete tail labels. With the comparable profile, audit horizon 90, TP 1.3 ATR, SL 1.6 ATR, and future bars beginning after entry; report any intentional profile difference.
4. **OOF predictions** - Every probability, Expected-R value, meta-model input, loss cluster, missed-winner family score, and claimed validation trade is produced by a model that did not fit that event or its label interval. In-sample base-model predictions are not OOF.
5. **Calibration** - Probability calibration is fit on a past-only partition or genuinely OOF predictions separate from model fitting and evaluation. Rolling calibration must end before each scoring block. Reject calibration on training predictions, the evaluated fold, holdout, or recent outcomes.
6. **Threshold selection** - Thresholds, top-k, sessions, regimes, feature filters, family definitions, exit profiles, and champion rules are selected inside training/inner-validation history and frozen before the interval used to claim performance. Dynamic thresholds must be reconstructed using only history available before each block.
7. **Purge/embargo** - No fitted or calibrated sample has a label or execution interval overlapping the next stage. Verify event identities and source-bar intervals across fit/calibration/selection boundaries; use the exact maximum label end when available or at least the full horizon as a conservative purge. Any shared overlapping events, duplicated timestamps across splits, or unpurged boundary labels fail this check.
8. **Holdout contamination** - The holdout is absent from feature, rule, family, hyperparameter, threshold, calibration, and model-choice decisions. If a result was revised after viewing holdout outcomes, that interval is contaminated and cannot support a final OOS claim.
9. **Recent-period reuse** - Establish how many research iterations inspected the recent interval. It passes only if it was evaluated once after all choices were frozen, or is explicitly excluded from selection and promotion claims as monitored data. Repeatedly consulted recent results are not an untouched test.
10. **Execution alignment** - Rebuild trades from signal time through exit. Use intrabar `HIGH`/`LOW` first touch beginning after entry, a documented conservative rule when TP and SL are both reachable in one bar, actual exit offsets, chronological deduplication, and the same single-position occupancy as MT5. Raw qualifying bars, overlapping signals, or oracle-selected directions are not executable trades.
11. **Cost assumptions** - Verify price units and per-trade application of spread, commission, slippage, and swap where applicable. Distinguish observed historical spread from constants, document omissions, and require a stated adverse-cost stress case. Costs must affect rewards, PF, PnL, and selection consistently without using future quotes.
12. **Multiple-testing risk** - Inventory all tried models, features, filters, clusters, sessions, thresholds, top-k values, exit profiles, and generations, including discarded candidates. Pass only when selection is confined to inner walk-forward data and a still-untouched final test exists, or when a justified selection-aware statistical control supports the claim. Reporting only the winning candidate, applying a 60% cutoff, or reusing holdout/recent data does not control data snooping.

For false-positive mining, additionally trace each filter from discovery through OOS application and confirm that losers removed and winners accidentally removed use identical executable-event identities. For missed-winner mining, ensure future TP-first outcomes define offline discovery labels only, family rules are frozen before evaluation, and raw plus execution overlap with the incumbent is removed before counting unique trades. Include these findings under the twelve checks rather than adding unrequested verdict categories.

## Required output

Start with `Overall: PASS` only when all twelve checks pass; otherwise use `Overall: FAIL`. Then provide one table with these columns:

| Check | Verdict | Evidence | Failure or reason for pass | Required validation correction |
|---|---|---|---|---|

Use repository-relative `file:line`, artifact keys, timestamps, event counts, or reproducible assertions as evidence. List all twelve checks even when one early failure invalidates the experiment.

After the verdict table, independently reconcile the claimed metrics for every OOS fold, holdout, recent interval, and cost stress: executable trades, evaluated days, trades/day, wins and losses, win rate, PF, PnL or mean R, and max drawdown. State whether the default 60% target and guardrails were met, but do not turn metric success into an audit pass.

End with the smallest set of validation-only reruns needed to clear failed checks and label the submitted performance claim `valid` or `invalid`. Do not provide optimization ideas, new thresholds, or a production-promotion recommendation.
