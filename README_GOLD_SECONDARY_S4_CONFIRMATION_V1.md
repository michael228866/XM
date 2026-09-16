# GOLD secondary S4 confirmation v1

**Source-only implementation.** Formal research and formal validation are manual
steps after source review. Self-tests do not create a `training_runs` entry or
load historical market data. This family does not modify discovery sources,
finalized archives, `gemini.py`, or the operational model.

## Scope and interpretation

This study checks the robustness of the predeclared discovery candidate
`S4_SECONDARY_P075`, the only interesting candidate in
`20260915T151723Z_gold_independent_secondary_classifier_v1`.
The discovery effect was small: WR increased by 0.0005840676502704145 and
trades/day by 0.021118498240125128 (+7.837445573294621%).

Exactly two configurations are evaluated:

| Configuration | Entry rule |
|---|---|
| A0_BASELINE_075 | B0 >= 0.75 |
| S4_CONFIRMATION | primary OR (B0 < 0.75 AND secondary >= 0.75) |

Threshold comparisons are inclusive; primary and secondary signals cannot
overlap. Every alternate threshold and candidate name is rejected. There is no
threshold search, ranking, percentile, top-K, recalibration, feature tuning,
hyperparameter tuning, selection, or production promotion.

These are the same previously inspected historical intervals. Fresh fitting
and new evaluation slices **do not create an untouched test**. A confirmation
PASS supports further research under the specified gates, not a production claim.

## Frozen identities and fresh fitting

The [execution specification](execution_spec_gold_secondary_s4_confirmation_v1.json)
is independently hash-pinned by both new programs. It contains the complete
discovery model parameters, 31-feature order, six timestamp/feature block hashes,
three C1 target hashes, B0 model identities, two gates, stresses, and confirmation
rules. The discovery `FINALIZED.json` SHA-256 is:

`522b9a8f477e23a79a9200905667b386747bb4fc9e7e40c66a2bdadd4ba19f4f`

Formal execution verifies every finalized discovery file, its original validator
PASS, sole interesting S4 result, original Git-committed 0.75 specification, and
absence of production promotion. The blocked later validator invocation does
not replace the retained original PASS.

Reconstruction uses the same immutable archived GOLD/C1 helpers and exact causal
18-month training windows. All frozen timestamp, float32 feature matrix and
full C1 training-target identities are checked before fitting. B0 probabilities
are freshly computed from the corresponding archived fold model. Training is
restricted to that fold's exact `B0 TRAIN probability < 0.75` subset, with no
score-fold rows, cross-fold fit, or B0 probability feature. Earlier historical
score periods can lie within a later fold's original causal training window;
each model remains isolated from its own score interval.

Train three new specialists, one per original fold, using the **unchanged**
discovery configuration: XGBoost 3.2.0, seed 42, CPU, one thread, 220 estimators,
depth 4, learning rate 0.05, min child weight 80, subsample/colsample 0.85,
and all other exact spec parameters. Balanced weights are recomputed only from
that specialist's training subset. Both target classes are required.

Only freshly generated prediction arrays enter confirmation execution. Discovery
predictions are read for deterministic reference comparison after fresh fitting;
they never supply the confirmation gate. Exact subset, target, weight, training
configuration and numerical prediction-array equality is expected and recorded.
A mismatch aborts rather than silently weakening deterministic provenance.
Fresh model **file hashes are not required to match discovery model files**.

## Baseline and execution

A0 is executed and checked before secondary fitting. Every baseline value must
match with absolute tolerance 1e-12, zero relative tolerance; failure aborts.

| Metric | Frozen A0 |
|---|---:|
| trades / wins / losses | 689 / 390 / 299 |
| realized_wr | 0.5660377358490566 |
| trades_per_day | 0.26945639421196715 |
| pf | 0.8247098331219882 |
| mean_r | -0.07690539315994348 |
| pnl_r | -52.98781588720106 |
| max_dd_r | -59.57958015890608 |
| cost_stress_pf | 0.7838186120438296 |

Replay the exact frozen S5 implementation once over the combined chronological
score folds for each configuration. Change only the entry gate and derived raw
episode identifiers. Keep spread/cost, RSI, sessions, cooldown, occupancy, TP/SL,
timeout, risk sizing and same-bar semantics unchanged. Do not reset execution
at a fold or slice boundary. Never retrain for a slice or stress scenario.

## Calendar slices

Use `start + (end_exclusive - start) / 2`, independent of observed rows, scores,
labels, or trades. All intervals are start-inclusive/end-exclusive; an entry at
the midpoint belongs to H2. Timestamp semantics remain the frozen broker/API
score-interval semantics.

| Original fold | Start | Midpoint | End exclusive | Half duration |
|---|---|---|---|---:|
| 2018_2020 | 2018-01-01 | 2019-07-03 | 2021-01-01 | 548 days |
| 2021_2022 | 2021-01-01 | 2022-01-01 | 2023-01-01 | 365 days |
| 2023_2024 | 2023-01-01 | 2024-01-01 12:00 | 2025-01-01 | 365.5 days |

Report the three original folds, six `_H1`/`_H2` slices, `ALL_EARLY` (union of
the three H1 intervals), `ALL_LATE` (union of H2), and pooled: twelve scopes per
configuration. Each trade belongs to the interval containing its actual entry,
including its full completed return if the exit crosses the boundary. Each
denominator is exact elapsed seconds / 86400; disjoint durations are summed.
Slice drawdown starts at zero and uses the selected trades in ledger order.

Every scope reports trades, wins, losses, realized WR, trades/day, PF, Mean-R,
PnL-R, Max DD-R, standard cost-stress PF and break-even-adjusted edge. It also
reports primary/secondary counts, wins, losses, WR and trades/day. An empty
primary/secondary WR is null; overall empty-scope WR follows repository metrics
and is zero. Deltas always mean S4 minus A0, including the five required metrics.

## Entry displacement

Stable identity is `(score_fold, source_entry_index)`, where source entry index
is the original frozen reconstructed row, not a rounded/approximate timestamp.
The combined execution `entry_index` is retained for occupancy reconstruction.

For every scope, report baseline/S4 trade counts, retained baseline primary
entries, missing (displaced) baseline primary entries, new secondary entries,
and additional primary entries that were not executable in A0. The accounting
identity is:

`S4_count - A0_count = new_secondary + new_primary - displaced_primary`

`displaced_primary_wins/losses/realized_wr` at the top level describe all missing
baseline identities using their A0 realized outcomes. The nested
`occupancy_displacement` reports the same fields and exact identities for the
subset with an S4 position open at that entry phase. `nonoccupancy_displacement`
contains the remainder, so risk-state differences are not mislabelled occupancy.
Observed occupancy is not a claim of exclusive causal attribution.

A trade occupies a later row through its TP/SL exit row because those exits
occur after the entry phase. A timeout exit row is excluded: frozen S5 processes
timeout before eligibility. The occupying trade may have entered in an earlier
slice; occupancy always uses the full continuous ledger. New secondary trade
wins/losses/WR are reported separately.

## Post-trade diagnostic stresses

| Scenario | Additional round-trip R cost |
|---|---:|
| R0_STANDARD | 0.00 |
| R1_EXTRA_COST | 0.01 |
| R2_EXTRA_COST | 0.02 |

For every executed trade, calculate `stressed_r = net_r - extra_cost_r`.
Recompute wins as `stressed_r > 0` (zero is a loss), losses, WR, PF, mean, PnL
and drawdown. WR changes only on a zero crossing. Entries, occupancy, exits,
spread/RSI gates and risk decisions are never recomputed for these scenarios.
The inherited `cost_stress_pf` remains the original S5 stress measure; R1/R2 are
separate additional post-trade diagnostics. Non-finite JSON numbers serialize
as null following archived helpers; CSV retains inf/nan for exact audit.

## Exactly eight confirmation conditions

`confirmation_primary_pass` requires all of:

1. Pooled S4 WR strictly exceeds pooled A0 WR.
2. Pooled S4 trades/day strictly exceeds pooled A0 trades/day.
3. Original S4 guardrails: pooled PF >= 0.80, Mean-R >= -0.10690539315994348,
   cost-stress PF >= 0.75; every original fold has >= 10 trades, WR >= 0.45,
   PF >= 0.70.
4. At least four of six half-folds have both deltas >= 0.
5. `ALL_EARLY` does not have both primary deltas < 0.
6. `ALL_LATE` does not have both primary deltas < 0.
7. R1 pooled S4 PF >= R1 pooled A0 PF - 0.03.
8. R2 pooled S4 PF >= R2 pooled A0 PF - 0.05.

`confirmation_supportive = confirmation_primary_pass`.
`confirmation_strong` additionally requires pooled S4 WR >= 0.58 and
trades/day >= 0.40. No discovery-only gate, new selection criterion, or automatic
promotion is added.

## Formal artifacts and independent validation

Required outputs include `metrics.json`, `candidates.csv`, `fold_metrics.csv`,
`robustness_slices.csv`, `displacement_metrics.json`, `stress_metrics.csv`,
`trade_ledger.csv`, `secondary_evidence.npz`, `model_inventory.json`,
`baseline_reproduction.json`, and `source_provenance.json`. Also retain all three
new models, model hashes, identity audit, source/validator/spec snapshots,
operational pre/post hashes, and the report.

The validator never imports the new confirmation runner. It uses the archived
independent reconstruction/metric code and its own gate, midpoint, slice,
displacement, stress and confirmation calculations. It independently refits each
secondary model once, compares learned trees/intercept and exact predictions
against the new run, and replays frozen S5. It checks every JSON, CSV and report,
all 24 scoped metric rows, 72 stress rows, identities, and operational hashes.

`validator_attempt.json` is exclusively created before verification starts.
No retry is allowed after success, failure, or interruption. Preserve the marker
and every attempt artifact. Validator PASS certifies methodology/provenance
within this historical study, not untouched validity or production suitability.

## Source-review checks only

```powershell
.\.venv\Scripts\python.exe -m py_compile gold_secondary_s4_confirmation_v1.py validate_gold_secondary_s4_confirmation_v1.py
.\.venv\Scripts\python.exe -B gold_secondary_s4_confirmation_v1.py --self-test
.\.venv\Scripts\python.exe -B validate_gold_secondary_s4_confirmation_v1.py --self-test
git diff --check
```

Synthetic tests use generated matrices and toy trade records. They cover fixed
threshold rejection, routing and zero overlap, midpoint boundaries and fractional
days, early/late unions, exact entry identity, displacement and timeout ordering,
cost zero crossings, deterministic fitting, unchanged A0 gate, and positive and
negative cases for every confirmation condition. They do not reconstruct full
historical data, create a formal run, or call formal validation.

## Manual formal execution after review

These commands are documentation only. First verify clean, committed, pushed
sources, then manually create a unique run using `training_run_history.py create`
with this experiment name, this runner, exact command and seed 42. Use the path
returned by that workflow:

```powershell
.\.venv\Scripts\python.exe -B gold_secondary_s4_confirmation_v1.py --run-dir training_runs/<new-run-id>
.\.venv\Scripts\python.exe -B validate_gold_secondary_s4_confirmation_v1.py --run-dir training_runs/<new-run-id>
```

Use reviewed repository entry points, not the snapshots. The runner checks
HEAD/upstream/executed commit, clean creation provenance and its exact snapshot;
exclusive creation of `execution_spec.json` prevents rerunning the attempt.
No formal commands are executed as part of implementation. After manual result
review, follow the normal immutable finalize/register workflow. No candidate is
selected or promoted automatically.
