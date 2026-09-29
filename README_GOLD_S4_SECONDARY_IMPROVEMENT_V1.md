# GOLD S4 Secondary Improvement v1

After independent infrastructure certification and committed approval, double-click
`RUN_TRAINING.bat`. This is the only normal entrypoint. The same launch runs the
reference control, historical research, independent validation, immutable
finalization and result display. The window pauses on completion or failure.
Only a USER Explorer launch receives the short-lived training receipt.

## Frozen reference

The reference trainer and config remain byte-identical. The control calls its
existing `train_folds` function, using the exact certified 21 GOLD# exports,
31 ordered features, C1 target, B0 training probability < .75 routing, seed 42,
three original folds and S5 execution. B0 is loaded, never retrained.
Control model hashes must match the verified 20260928T150232Z run exactly;
all ten pooled metrics must match within absolute tolerance 1e-12, relative
tolerance zero. Failure stops research; tolerance is never adapted.

The raw export cutoff is 2026-05-08 23:57 broker/export clock. Actual training
retains the three 18-month historical windows; the latest ends before 2023-01-01.
Scoring ends before 2025-01-01. Later CSV rows never expand these folds.
The existing data manager accepts only exact original hashes. Missing exact
files may be restored through its existing approved fallback; different fetched
bytes are rejected. This workflow does not authorize NEW_RETRAIN_DATASET.

## Predeclared 16 configurations

| Family | Configurations |
| --- | --- |
| 0 | Exact S4 control, threshold .75 |
| 1 | Depth 3; depth 5; min_child_weight 120 plus reg_lambda 3 |
| 2 | Balanced weights with positive multiplier .9 or 1.1 |
| 3 | Platt calibration inside each training fold |
| 4 | Reference model at .70, .725, .775, .80; .75 is family 0 |
| 5 | Remove clock group; redundant H2/H3/H6/H8; fast M2/M3/M4/M5/M6 |
| 6 | Three normalized features; three timeframe agreement features |

There are no family cross-products, adaptive candidates, importance-based
post-hoc subsets, external datasets or fine threshold searches. Ablation uses
fixed semantic/redundancy groups. All six new features are pointwise functions
of existing causal inputs; definitions and order are fixed in config. No future
bars, centered windows or backward filling are involved.

Platt calibration reserves the last three calendar months of each outer
training window. Secondary fitting excludes rows whose C1 maturity or legacy
horizon reaches the calibration start. Calibration rows remain inside the
original purged training fold. The model is not refitted after calibration.
Logistic settings are C=1, lbfgs, max_iter=1000, seed=42; nonconvergence stops the
run. Existing outer B0 in-sample routing remains frozen: this is not independent
B0 calibration evidence.

## Two stages and selection

Stage 1 uses the original 2018–2020 and 2021–2022 folds. Its pooled metrics must
beat the fixed full-period reference WR and trades/day, and pass pooled/fold
safety. Stage 1 replays only those two folds, including its own cohort end.
Only qualifying configurations proceed to the 2023–2024 fold in Stage 2.
Stage 2 replays all three original folds together, preserving cross-fold
execution semantics. The control itself always reproduces all three first.

These are previously inspected historical development periods. Stage 2 is
historical robustness evidence, not an untouched test or promotion evidence.
No locked future holdout data is used.

Safety: pooled PF >= .80, Mean-R >= -.10690539315994348, stress PF >= .75;
every included fold has trades >=10, WR >=.45 and PF >=.70. Both pooled WR and
trades/day must strictly beat the S4 reference. STRONG requires .58/.40 and
TARGET .60/.50 with the same safety checks.

The archived full-period Pareto frontier includes all safe full-period results,
including control. Stage-1-only losers remain recorded but are not compared
with full-period metrics. Selection requires both KPI improvements. Tie-breaks:
WR, trades/day, stress PF, PF, Mean-R descending; absolute drawdown ascending.
An exact numerical tie uses candidate ID ascending, explicitly recorded.

## Evidence and independent validation

The committed config fixes the full space before launch. The run writes it
exclusively before control evaluation and retains a hash-linked event journal
with begin/result events for every evaluated stage. Losing records are retained.
No post-hoc candidate insertion/removal or gate changes are permitted.

Shared train/score feature matrices are retained in 50,000-row compressed
chunks for exact reference feature-hash verification. Fold labels, timestamps,
maturities, conditioning, weights, score probabilities, calibration parameters
and complete ledgers remain auditable. All models in this small bounded search
are retained specifically so the independent validator can recompute their
probabilities. This can produce a substantial archive; no raw-backup claim is
made for local source CSVs.

The independent validator imports no trainer/search implementation. It checks
committed source, frozen matrices/labels, model identities and predictions,
calibration splits/weights, fixed causal feature formulas, S5 replay metrics,
all candidate records, Pareto decisions, holdout exclusion and production hashes.
Its validation attempt is one-shot. Successful finalization follows validation.

Execution PASS with `NO_IMPROVEMENT_FOUND` is valid and preserves all losing
results. `IMPROVEMENT_FOUND` means historical research only. Model bundles contain
fold-specific specialists; the last fold model is not a new full-history live
model. Calibration candidates also require their recorded calibrator. Neither
TARGET nor any other result permits automatic production promotion.

This implementation is certified with synthetic fixtures only. Codex does not
start the real search. Existing finalized research and capture automation remain
unchanged.
