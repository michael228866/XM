# GOLD independent secondary classifier v1

Source-only, frozen research family. **Do not run formal research during source review.**
No run directory, research result, selection, or production promotion is produced by
implementation or synthetic self-tests. Closed CFTC direct-feature, score-ranking,
and anchored-episode families remain closed and unchanged.

## Hypothesis and fixed candidates

Retain archived B0 as the primary classifier. Train one independent specialist per
fold on exactly the TRAIN rows whose corresponding frozen B0 probability is `< 0.75`.
Use the original execution-aligned C1 target and the same ordered 31 technical
features. B0 probability routes observations; it is never a specialist feature.
Primary training scores are in-sample routing values, not OOF evaluation evidence.

| Candidate | Entry gate |
|---|---|
| A0_BASELINE_075 | B0 >= 0.75 |
| S1_SECONDARY_P060 | primary OR (B0 < 0.75 AND secondary >= 0.60) |
| S2_SECONDARY_P065 | primary OR (B0 < 0.75 AND secondary >= 0.65) |
| S3_SECONDARY_P070 | primary OR (B0 < 0.75 AND secondary >= 0.70) |
| S4_SECONDARY_P075 | primary OR (B0 < 0.75 AND secondary >= 0.75) |

Exactly five gates share three secondary fold models. No percentile, top-K,
anchor, lower-B0-threshold variant, per-fold threshold, search, calibration,
early stopping, or warm start. No candidate is automatically selected.

## Frozen reconstruction and training

The authoritative [execution specification](execution_spec_gold_independent_secondary_classifier_v1.json)
is hash-pinned independently by both source files. It records exact model/source,
timestamp-block, feature-matrix and target hashes, feature order, thresholds,
parameters, class weighting, diagnostic conventions, and assessments.

B0 source: `20260913T141817Z_gemini_cftc_gold_cot_b0_b1_v1`.
C1 target: `20260903T071729Z_gemini_execution_aligned_label_v1`.

| Score fold | Score interval (end exclusive) | Train window |
|---|---|---|
| 2018_2020 | 2018-01-01 to 2021-01-01 | preceding 18 months |
| 2021_2022 | 2021-01-01 to 2023-01-01 | preceding 18 months |
| 2023_2024 | 2023-01-01 to 2025-01-01 | preceding 18 months |

Use the exact archived broker/API timestamps and strict original maturity purge:
both legacy label information and C1 label maturity must be strictly before the
score boundary. Each model fits only its own prior training window; an earlier
fold's historical rows may appear in a later fold's training window under these
original chronological folds. No model fits its own score rows or pools fold fits.
Reconstruction checks all six timestamp/feature blocks and all three full C1
training target hashes **before any secondary training**:

| Fold | C1 target SHA-256 |
|---|---|
| 1 | d9bed87a073fd93c048b5a1ab8f0f7049f1a1bb8f3ff46f48a102c0b6763bbd3 |
| 2 | ff55356e8bdf1cb2cd5add50690156178ff32bc17f666f2103c8f51397fb3a12 |
| 3 | cd05d3647cd88a2fbd2b89fa3362ed5e2f9f13cbd611dee436f0e03e4a054259 |

XGBoost **3.2.0**, seed **42**, CPU, one thread, binary logistic, histogram,
220 estimators, learning rate 0.05, max depth 4, min child weight 80,
subsample/colsample_bytree 0.85. Other explicit parameters are in the spec.
The C1 balanced weighting convention is retained, recomputed only inside each
secondary subset: `weight = subset_N / (2 * subset_class_count)`. Both classes
must exist or the run fails. Feature inputs are named float32 columns, targets
int8, weights float64. Automatic XGBoost intercept estimation uses only the
secondary training subset. CPU/one thread are frozen for reproducible auditing;
there is no adaptive hardware choice or GPU fallback.

## Execution and baseline gate

Use the hash-verified original S5 simulator without changing its implementation.
Only `raw_signal` and its derived raw episode IDs change. B0 probabilities remain
in the price cohort. Spread/cost, RSI, sessions, cooldown, position occupancy,
TP/SL, timeout, risk sizing, and same-bar behavior are inherited unchanged.
Each candidate runs once across the combined chronological score folds, with
continuous execution state. Signal counts do not imply executable trade counts.

Before fitting any specialist, A0 must reproduce every value below with absolute
tolerance **1e-12**, zero relative tolerance. Failure aborts the attempt.

| Metric | Baseline |
|---|---:|
| trades / wins / losses | 689 / 390 / 299 |
| realized_wr | 0.5660377358490566 |
| trades_per_day | 0.26945639421196715 |
| pf | 0.8247098331219882 |
| mean_r | -0.07690539315994348 |
| pnl_r | -52.98781588720106 |
| max_dd_r | -59.57958015890608 |
| cost_stress_pf | 0.7838186120438296 |

## Assessment and diagnostics

Pooled guardrails: PF >= 0.80, Mean-R >= -0.10690539315994348,
cost-stress PF >= 0.75. Each fold: trades >= 10, realized WR >= 0.45,
PF >= 0.70. Missing/NaN guardrail values fail closed; positive infinity retains
the stated `>=` comparison. As in the archived metric helpers, non-finite JSON
statistics serialize as null; CSV preserves `inf`, and assessments use the
reconstructed numerical value.

All three categories require every guardrail to pass:

- Interesting: WR > baseline **and** trades/day > baseline.
- Strong: WR >= 0.58 **and** trades/day >= 0.40.
- Target: WR >= 0.60 **and** trades/day >= 0.50.

`metrics.json`, `candidates.csv`, and `fold_metrics.csv` report all requested
pooled/per-fold economic metrics, guardrails, primary/secondary trade counts,
secondary wins/losses/WR/trades per day, and raw pre-S5 signal counts.
`overlap_count` must be zero. Executed trades are classified using their actual
entry row; a secondary position may block a later primary entry under frozen S5.
Wins are net R > 0; zero-return trades are losses. Empty secondary WR is null.
Calendar score-window days are used as denominators, including days with no trades.

Secondary score distributions include count, mean, population std, linear
p50/p75/p90/p95/p99 and max, both for all score rows and for `B0 < 0.75` rows,
per fold and pooled. Empty distributions have count zero and null statistics.

Artifacts include all three models and complete training configurations,
subset indices/targets/weights, B0 train/score probabilities, secondary score
probabilities, row/timestamp identities, all gates, trade ledger, baseline
reproduction, source provenance, and operational pre/post hashes.

## Independent one-shot validator

The validator does **not import the new runner**. It uses the immutable archived
independent reconstruction/metric helpers and frozen S5 source. It independently
reconstructs the target and subsets, loads exact B0 models, recomputes B0 scores,
and refits the frozen secondary configuration **once per fold**. Trees, learned
intercept, feature order/types, complete training config, weights and predictions
must match the retained model/evidence. This refit is audit computation, with no
search, model selection, or replacement of retained artifacts.

It independently reconstructs every gate, replays complete S5 ledgers, checks all
JSON/CSV/report metrics and diagnostics, and checks operational file hashes.
An exclusively created `validator_attempt.json` prevents retries even after an
interrupt. Failed attempts preserve evidence; do not delete the marker or retry.
Validator PASS is methodology/provenance only, not a research success or permission
to promote. Historical intervals have been inspected repeatedly and are development
evidence, not an untouched final test. No selection-aware production claim is made.

## Source-review checks only

```powershell
.\.venv\Scripts\python.exe -m py_compile gold_independent_secondary_classifier_v1.py validate_gold_independent_secondary_classifier_v1.py
.\.venv\Scripts\python.exe -B gold_independent_secondary_classifier_v1.py --self-test
.\.venv\Scripts\python.exe -B validate_gold_independent_secondary_classifier_v1.py --self-test
git diff --check
```

Synthetic tests use small generated matrices and toy price rows. They cover the
strict secondary universe, B0 exclusion, inclusive boundaries, no overlap, fold
isolation, deterministic fitting, union gates, A0/S5 equivalence, saved-model
prediction persistence, and malformed inputs. They never load historical market
data, call a formal entry point, or create a training run.

## Manual formal execution after source review

The following is documentation only; implementation must not execute it.
Verify committed/pushed sources and a clean worktree first. Create a unique run
with `training_run_history.py create`, supplying this experiment name, runner,
exact command, and seed 42 according to repository policy. Then use the returned
path for these commands, once each:

```powershell
.\.venv\Scripts\python.exe -B gold_independent_secondary_classifier_v1.py --run-dir training_runs/<new-run-id>
.\.venv\Scripts\python.exe -B validate_gold_independent_secondary_classifier_v1.py --run-dir training_runs/<new-run-id>
```

Do not execute the snapshots directly: both entry points require the reviewed
repository source identities. The runner requires a clean creation manifest and
matching HEAD/upstream/executed commit. Its exclusive `execution_spec.json`
creation makes formal execution non-retryable. Preserve failed/partial evidence.
Finalize/register only after manual result review using the existing immutable
workflow. Never modify production or a finalized archive.
