# GOLD S4 v4: Label / Objective research

Implementation and infrastructure certification only. Real training, label
generation, fitting, historical evaluation and validation belong exclusively to
the USER manually double-clicking `RUN_TRAINING.bat` after committed approval.
Codex must not execute that BAT or the real launcher.

## Audit completed before label redesign

`gold_s4_v4_label_pipeline_audit.json` records the source-level audit and hashes.
Current C1 is already **standalone S5 net realized R > 0**; zero maps to 0.
It is not a portfolio-selected trade label. Entry is the next executable row's
OPEN using preceding completed-bar features; entry-bar HIGH/LOW applies,
stop wins a same-bar conflict, and the original 90-minute wall-clock timeout,
spread fallback, costs and R denominator are unchanged.

The existing v1 fold evidence retains binary labels and maturity, but not the
continuous training R. The USER workflow regenerates R using the original frozen
C1 source and exact previously certified CSV cache. No fetch, extra CSV, new feed
or guessed path data is permitted. Regenerated timestamps, feature matrices,
binary targets and outcome maturity must match the sealed evidence before use.
Infrastructure inspects only source/provenance metadata, not those raw inputs.
Availability and exact reconstruction are rechecked at USER runtime; synthetic
certification does not establish that local raw files remain available.

The reference is unchanged A_NO_LONG_HTF from v2, with 787 trades and its exact
metrics, 24 features, model hashes, execution provenance and effective amendment.
The PASS v3 run found no improvement and does not replace this reference.

## Five labels, eight configurations

| Label | Positive class |
|---|---|
| L0_CURRENT | R > 0 |
| L1_R_GE_025 | R >= .25 |
| L1_R_GE_050 | R >= .50 |
| L2_NONLOSS_GT_M025 | R > -.25 |
| L2_NONLOSS_GT_M050 | R > -.50 |

The two nonloss labels are not win labels. Missing/nonfinite R aborts; no
imputation. Each definition binds the implementation function and source SHA256.

Configurations: immutable model control, five RAW label models, and two ISOTONIC
variants (L0 and L1_R_GE_025). Eight is intentionally below the preferred 10–18:
no redundant configurations are added to reach a count. R_GT_0 duplicates L0;
STRONG_WIN lacks a design-time guarantee of sufficient positives. Regression,
ordinal and two-stage objectives are excluded to retain one binary architecture
and avoid changing the score/model-validation contract at the same time.

All new models use the reference's fixed capacity and exact feature order.
No feature, execution, risk, exit, cost or R-accounting changes. Model inputs are
selected exclusively from the frozen feature names; outcome columns remain in
separate label arrays. Control reuses original models, balanced weighting and
.75 threshold; it is never retrained. New models use NONE weighting, including
the separately identified L0 training comparator.

## Time isolation and threshold policy

Use the original 18-month fold training window and existing dual C1/legacy
maturity purge. RAW fitting ends three months before the validation fold.
ISOTONIC fitting ends six months before validation; the next three months are
calibration. The last three training months are development in both cases.
Outcomes and legacy maturity must be strictly before the next partition boundary.
The model is not refitted after calibration or threshold selection.

Threshold grid: .55, .60, .65, .70, .75, .80. Development-only objective:
maximize balanced accuracy, then recall, then higher threshold. This explicit
alternative policy does not claim to optimize executable development trades/day.
Validation-fold labels/outcomes cannot choose the threshold. Each fold records
all grid scores, selected threshold, fit/calibration/development positions,
partition membership (0 purged, 1 fit, 2 calibration, 3 development), feature
cutoff, decision and outcome timestamps, target hashes and class statistics.
The feature cutoff is the conservative inclusive decision-time bound verified
against the shifted source pipeline, not an invented observed feature-bar time.

Each fitted partition, development partition and enabled calibration partition
requires at least 100 positives, 100 negatives and prevalence in [.01,.99].
Degeneracy fails the candidate and aborts the run, preserving evidence; candidates
are never silently removed and thresholds are never relaxed after failure.

## Research decision and events

Control must reproduce exact counts and all reference metrics within absolute
1e-12, relative zero tolerance. Mismatch stops before new label generation or
fitting with `REFERENCE_CONTROL_MISMATCH`, research NOT_RUN.

An official winner requires both WR > .5717916137229987 and trades/day >
.30778255768478685, pooled PF >= .80, Mean-R >= -.10690539315994348,
stress PF >= .75, and every fold trades >= 10, WR >= .45, PF >= .70.
STRONG requires WR >= .58 and TPD >= .40; TARGET requires .60 and .50.
The primary Pareto axes remain WR/TPD. Tie-break is WR, TPD, PF, stress PF,
Mean-R, PnL descending, absolute drawdown ascending, then ID for exact ties.
A separate PF/Mean-R frontier is diagnostic only. PF >= .90, PF >= 1.00,
Mean-R >= 0 and PnL > 0 are annotated milestones, never substitutes for the
primary gate. Economic classification uses the strict PF/Mean-R/PnL signs.

Canonical raw-payload events: space_frozen, reference_pass,
label_definition_freeze, candidate_start/candidate_end, pareto_complete,
selection_complete, research_freeze. The control has candidate_end immediately
after its reference pass/label freeze. `reference_pass` hashes the raw control
row without a status wrapper. UTC timestamps cannot go backward; contiguous
sequence and previous/self hashes are mandatory. Validation_result and
finalization are separate tip-bound receipts and do not extend the research chain.

No improvement is a successful execution with no winner, gate NONE, and the
reference unchanged. Original completed research is retained if later validation
fails. Real runs finalize/register/commit/push their own evidence. No automatic
retry, production promotion or edits to prior finalized runs.

## Infrastructure certification

After pushing the implementation source:

```powershell
.venv\Scripts\python.exe -B certify_gold_s4_secondary_improvement_v4_label_objective.py --execute
```

This runs only compile/static/synthetic checks. Archive and push its result,
then use `--approve <sealed-infrastructure-run>` and commit/push that approval.
Stop; only the USER may launch real Train + Val. Approval pins sources, policies
and certificate. A 30-second single-use Explorer/BAT USER receipt and separate
one-use validation permit guard real execution. Synthetic mode denies historical
and locked holdout inputs. `CHECK_STATUS.bat` reads metadata only.

All historical results remain development evidence. `future_holdout/gold_s4_v4`
is locked. `gemini.py` and the production model remain hash protected.
Infrastructure PASS is not real v4 historical validation or a promised improvement.
