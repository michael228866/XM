# GOLD S4 secondary improvement v3

USER-only historical development workflow. After infrastructure certification and
approval are committed and pushed, manually double-click `RUN_TRAINING.bat`.
Codex must stop after implementation, synthetic certification and approval.
`CHECK_STATUS.bat` reads approval/run metadata only. No scheduler or CI entrypoint.

## Frozen reference and scope

`gold_s4_a_no_long_htf_reference_v1.json` binds the exact v2 A_NO_LONG_HTF
selected row (787 trades), three model hashes, 24 features, configuration, search
space, original seal, repair PASS and external amendment. Original FAIL remains.
Model bytes are checked when USER executes; infrastructure reads provenance
metadata only, without model loading, historical arrays or outcome recomputation.

The reference retains Daily, H1 and H4 trends. No HTF oscillator/volatility group
exists in that feature list. v3 does not invent absent groups or add removed HTFs.
The immutable v1 evidence supplies the same certified inputs and three folds;
the effective v2 reference supplies the existing control models.

## Search frozen before USER execution

23 total configurations: control 1, HTF ablation 5, compression 4, threshold 6,
calibration 2, weighting 2, execution timing 3. No Cartesian-product search,
adaptive shortlist, new data, external symbols or production updates.

- H: remove Daily, H4, H1, Daily+H4, or all three HTFs.
- HC: separately replace the three raw HTFs with trend consensus, positive
  direction count, alignment ratio or disagreement score.
- T: .70, .72, .74, .76, .78, .80; .75 is the control. These reuse reference
  models, preserving exact features, parameters and raw probabilities.
- CAL: PLATT and ISOTONIC, fitted only on the final three training months after
  a purged earlier training partition. Evaluation-fold labels never calibrate.
- W: balanced positive multipliers .975 and 1.025, around control 1.0.
  Frozen v2 summaries explain why NONE and 1.10/1.20/1.30 are not repeated.
- X: volatility-adjusted body quality, volatility-ratio-normalized distance
  from the 20-bar mean, and their predeclared pair. The distance is a proxy,
  not a newly computed price trigger. All six derived features are pointwise
  transforms of certified causal columns. No labels or future rows are inputs.

Control metrics require absolute tolerance 1e-12, zero relative tolerance;
integer counts reproduce exactly. Mismatch aborts before any new candidate fit
with `A_NO_LONG_HTF_REFERENCE_CONTROL_MISMATCH` and research `NOT_RUN`.

Both WR and trades/day must strictly exceed reference. Pooled PF >= .80,
Mean-R >= -.10690539315994348, stress PF >= .75; each of the three folds must
have trades >= 10, WR >= .45, PF >= .70. STRONG is WR >= .58 and TPD >= .40;
TARGET is WR >= .60 and TPD >= .50, with the same safety guards.

Pareto axes remain WR and trades/day. Economics are separately annotated.
Qualifying frontier tie-break: WR, TPD, PF, stress PF, Mean-R, PnL-R descending,
absolute drawdown ascending, candidate ID only for exact numerical ties.
PF .90 and positive expectancy are secondary milestones, never a reason to
waive either primary improvement. No subjective close-candidate override.
Economic status uses strict PF/Mean-R/PnL-R signs; mixed/zero values are
NEAR_BREAK_EVEN. No improvement is a successful execution with no winner.

## Chronology and ownership

The canonical event specification is in the frozen search space. Trainer and
validator implement it separately: sorted compact JSON, UTF-8, no NaN;
`reference_pass` hashes the exact raw control row. The status wrapper is only
in `reference_control.json`. Sequence is contiguous, UTC timestamps cannot go
backward, duplicate timestamps are legal. `research_complete` freezes the chain.
Validation and finalization use separate receipts and never append to that chain.

The launcher requires Explorer -> CMD ancestry, the BAT marker, a 30-second
single-use in-process USER receipt, committed approval, pinned sources and clean
pushed main. The independent validator also requires a single-use run-bound
permit and exclusive attempt file. Synthetic mode rejects real entrypoints and
historical/holdout access. Audit guards restrict real writes to the new run.

Completed research summaries are retained before independent validation, so a
later validator failure cannot relabel completed training as NOT_RUN.
The user workflow archives, registers, commits and pushes its own result.
No automatic retry; no modification to prior finalized evidence.

## Infrastructure certification

After the implementation source commit is pushed:

```powershell
.venv\Scripts\python.exe -B certify_gold_s4_secondary_improvement_v3.py --execute
```

This performs compilation, fake metric/selection checks, synthetic feature,
calibration isolation, weighting, threshold, event-chain and archive fixtures.
It performs no model fitting or real historical evaluation. Archive and push its
result, then use `--approve <infrastructure-run>` and commit/push the approval.
Do not launch the real Python launcher or BAT during Codex certification.

Infrastructure PASS proves the tested source contracts, not that a future real
USER run will pass or improve. All historical research is development evidence;
locked `future_holdout/gold_s4_v4` remains unused. No promotion is authorized.
