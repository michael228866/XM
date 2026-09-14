# Frozen GOLD score-ranking research v1

This source-only research family changes only the conversion of archived B0
scores into entry eligibility. There is no training, calibration fit, feature
change, external data, model replacement or production promotion. Formal execution
is not authorized as part of implementation.

## Frozen score and execution source

Both A0 and every ranking candidate use the same retained float32 OOF scores and
score indices from the completed, finalized run:

`training_runs/20260913T141817Z_gemini_cftc_gold_cot_b0_b1_v1/`

Its `FINALIZED.json` SHA-256 is frozen as
`592b68e98c1a0e9dd6f5e5af1d6aeadbca86df5072f2cbe85a064adf06c5ebfe`.
The entire recorded inventory must verify before use. Only B0 score arrays and
B0 models are used; no CFTC feature matrix or B1 scores enter this study.
The original parent remains
`20260903T071729Z_gemini_execution_aligned_label_v1`, and UTC timestamps come from
the finalized `20260905T171629Z_gemini_macro_event_integration_foundation_v1` archive.

The runner uses the prior archived runner's frozen reconstruction and metric
helpers. The independent validator uses the prior archived validator's separate
reconstruction and metric helpers; it never imports `gold_score_ranking_v1.py`.
Both verify all frozen local dependency hashes and parent archive inventories.
The 31 feature names/order, six timestamp identities, six GOLD matrix hashes,
three C1 target hashes and chronological fold selection remain those already
verified by the B0 source. No historical training is performed, including during
a later formal run. The independent validator reloads the existing three B0
models and confirms their predictions exactly equal the archived scores; the
archived scores remain the inputs to its entry-gating recomputation.

XGBoost version 3.2.0, original model parameters and seed 42 are inherited from
those immutable models. Existing S5 spread/cost, entry eligibility, TP/SL,
90-minute wall-clock timeout, no trailing and risk state are unchanged. Only
`raw_signal` and its corresponding episode IDs are replaced after normal cohort
construction. Original `buy_prob` values are retained. S5 state is initialized
once per candidate and runs continuously across the combined chronological score
folds, exactly as B0; only ranking state resets at fold boundaries.

## Exactly 13 candidates

| Candidate | Ranking reference | Absolute floor | Percentile minimum |
|---|---|---:|---:|
| A0_BASELINE_075 | None | 0.75 | None |
| A1_ROLLING_1D_P95 | 1 UTC calendar day | None | 0.95 |
| A1_ROLLING_1D_P97 | 1 UTC calendar day | None | 0.97 |
| A1_ROLLING_3D_P95 | 3 UTC calendar days | None | 0.95 |
| A1_ROLLING_3D_P97 | 3 UTC calendar days | None | 0.97 |
| A1_ROLLING_5D_P95 | 5 UTC calendar days | None | 0.95 |
| A1_ROLLING_5D_P97 | 5 UTC calendar days | None | 0.97 |
| A2_SESSION_P95 | Same UTC date and bucket | None | 0.95 |
| A2_SESSION_P97 | Same UTC date and bucket | None | 0.97 |
| A3_3D_F065_P95 | 3 UTC calendar days | 0.65 | 0.95 |
| A3_3D_F065_P97 | 3 UTC calendar days | 0.65 | 0.97 |
| A3_3D_F070_P95 | 3 UTC calendar days | 0.70 | 0.95 |
| A3_3D_F070_P97 | 3 UTC calendar days | 0.70 | 0.97 |

The formally frozen source specification is `execution_spec_gold_score_ranking_v1.json`.
The runner and independent validator both reject drift from this specification.
The same specification is written to the run-local `execution_spec.json` before
ranking or simulation. Candidate additions, deletions and post-hoc retuning are forbidden.
There are no CLI overrides, additional candidates, arbitrary thresholds, adaptive
search, per-fold optimization or selected winner. All 13 gates are evaluated as
declared and reported as historical development evidence.

## Exact causal rules

For rolling window `W`, the reference set includes only scores in the current
fold whose UTC timestamp satisfies `t - W <= prior_timestamp < t`. Days are
24-hour UTC calendar days, not trading days. A3 uses the identical 3D reference
arrays computed for A1 3D. No history is carried across folds.

For A2, buckets are `UTC_S0 = [00:00,08:00)`, `UTC_S1 = [08:00,16:00)` and
`UTC_S2 = [16:00,24:00)`. Each UTC calendar date × bucket × fold has a separate
history. These are fixed UTC buckets, not geographically named market sessions.

`percentile = count(prior_score <= current_score) / N_prior`.
Ties count under `<=`, without jitter, averaging or interpolation. All rows with
timestamp equal to the current timestamp are excluded, including earlier rows
in the input carrying that same timestamp. Only already observed scores enter
the reference set; no full-sample percentile is computed.

For ranking candidates, `N_prior < 60` means ineligible. Evidence stores the
percentile as NaN in that case, plus the actual count and numerator; there is no
zero fill or imputation. A0 has no ranking warm-up and always uses `score >= 0.75`.
All absolute comparisons use the exact archived float32 score value widened to
float64, against the stated floor; no score normalization or rounding is applied.
P95/P97 comparisons use `percentile >= 0.95/0.97`. A3 also requires `score >= floor`.

The runner maintains a sorted set of prior scores (including duplicates). The
validator instead directly counts `<=` in each strictly prior time slice.
Ranking arrays are computed once per reference per fold and reused by the gates.

## Baseline gate and objective

A0 must match all fields below with zero relative tolerance and absolute tolerance
`1e-12`; otherwise the run stops before simulating other candidates.

| Field | Frozen B0 |
|---|---:|
| trades / wins / losses | 689 / 390 / 299 |
| realized_wr | 0.5660377358490566 |
| trades_per_day | 0.26945639421196715 |
| pf | 0.8247098331219882 |
| mean_r | -0.07690539315994348 |
| pnl_r | -52.98781588720106 |
| max_dd_r | -59.57958015890608 |
| cost_stress_pf | 0.7838186120438296 |

Both realized WR and trades/day must strictly exceed B0 for an interesting
candidate. Strong means WR >= 0.58 and trades/day >= 0.40; target means WR >= 0.60
and trades/day >= 0.50. These classifications also require the safety guards.

Reject if pooled PF < 0.80, Mean-R < baseline Mean-R - 0.03, stress PF < 0.75,
or any fold has fewer than 10 trades, WR < 0.45 or PF < 0.70. PnL and max DD are
reported without inventing additional cutoffs. These user-frozen research guards
apply to this family; they do not grant production promotion or alter the
repository's production promotion requirements. PF is not the ranking objective.
There is no automatic candidate selection based on later fold outcomes.

## Later formal outputs

The runner requires an existing, new in-progress `training_run_history.py` run
with experiment name `gold_score_ranking_v1`, committed/pushed source identity
and only that run dirty. It creates no run directory itself and refuses a retry
once `execution_spec.json` exists. It retains:

- `execution_spec.json`, `identity_audit.json`, `source_provenance.json`;
- `ranking_evidence.npz`: archived scores/indices/UTC times, reference counts,
  `<=` counts, percentiles and all candidate eligibility masks by fold;
- `candidates.csv`, `metrics.json`, `fold_metrics.csv`, `trade_ledger.csv`, `report.md`;
- operational pre/post hashes, executed source/validator snapshots and manifest
  artifact hashes. Models and original scores remain in the validated archive.

Every candidate includes trades, wins, losses, WR, trades/day, PF, Mean-R, PnL,
max DD, stress PF, all fold metrics, delta WR, delta trades/day and percent
frequency uplift. Guardrail failures and interesting/strong/target flags are
explicit; all outputs remain research only.

Independent validation verifies source provenance, exact definitions and model
identities, recomputes all ranks and masks, replays the same S5, matches the full
ledger and recomputes fold/pooled metrics and classifications. It writes
`validator.json` and `validator.md` with overall PASS/FAIL and failed check names.
It refuses finalized runs and repeated validation. A failure is retained for
review; do not tune or modify evidence to make it pass.

## Source-only verification

```powershell
.\.venv\Scripts\python.exe -m py_compile .\gold_score_ranking_v1.py
.\.venv\Scripts\python.exe -m py_compile .\validate_gold_score_ranking_v1.py
.\.venv\Scripts\python.exe .\gold_score_ranking_v1.py --self-test
.\.venv\Scripts\python.exe .\validate_gold_score_ranking_v1.py --self-test
```

The deterministic self-tests use synthetic scores and trades, temporary files
and frozen pure metric helpers. They cover causal exclusion, future mutation,
inclusive rolling boundaries, fold/session reset, cold starts, threshold edges,
ties, baseline reproduction, invalid candidates/thresholds, model/production
mutation and pooled aggregation. They do not train a model, reconstruct historical
data, simulate historical trading or create a `training_runs` directory.
