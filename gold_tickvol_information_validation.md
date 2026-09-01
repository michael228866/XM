# TICKVOL Information Study - Walk-Forward Validation

Overall: **FAIL**

Internal chronological validation quality: **PASS_with_development_only_scope**

Final untouched-test validity: **FAIL**

| Check | PASS/FAIL | Evidence | Consequence |
|---|---|---|---|
| chronology | PASS | Each training frame ends before its next evaluation fold; folds are 2018-2020, 2021-2022, and 2023-2024; no random split is present. | Internal fold ranking is chronological. |
| feature leakage | PASS | Frozen H1 is implemented as shifted completed-bar log TICKVOL followed by two differences; rolling features are built after the shift and the self-check proves current-bar mutation cannot alter the same decision row. | TICKVOL values used at entry are available before the decision. |
| label maturity | PASS | Every fit label ends before calibration and every calibration label ends before policy scoring in all 12 fold/information-set fits. | Inner model stages do not train on immature net-R labels. |
| OOF predictions | PASS | Each fold model is fit only on the purged history before that fold and applied frozen to the fixed executable cohort in the next block. | Reported incremental ranking is internal chronological OOF, not in-sample ranking. |
| calibration | PASS | P(net-R>0) isotonic calibration uses the inner chronological calibration segment only; constant calibrated output is retained as an undefined Spearman rather than tuned away. | Calibration diagnostics are valid development diagnostics, although they are not final-test evidence. |
| threshold selection | PASS | No candidate threshold is selected; the primary comparison is continuous E(net-R) Spearman and there are zero threshold or hyperparameter variants. | No evaluated-fold threshold reuse affected the information verdict. |
| purge/embargo | PASS | Outer training uses the existing 180-row purge for a 90-row outcome horizon, and inner fit/calibration boundaries remove labels crossing the next stage. | Overlapping label windows do not cross training-stage boundaries. |
| holdout contamination | PASS | All inspected historical periods are explicitly development data. The study does not open an untouched_forward path and makes no final untouched-test claim. | The internal claim is correctly scoped; final untouched-test validity remains FAIL. |
| recent-period reuse | PASS | Evaluation stops at 2025-01-01 exclusive. 2025-2026 history is not used as an evaluation fold and post-2026-09-01T02:00:00Z data is not accessed. | Recent monitoring data did not select the TICKVOL result. |
| execution alignment | PASS | The test uses 206 existing Gen17 executable trades; ledger occupancy is non-overlapping within every fold and no raw-signal strategy candidate is constructed. | Ranking targets correspond to executable events, not raw qualifying rows. |
| cost assumptions | PASS | NET_REWARD is inherited unchanged from Gen19/20: observed entry spread when valid, 30-point fallback otherwise, and the existing extra-cost assumption. | The ablation changes only information, so economic targets remain comparable to control. |
| multiple-testing risk | FAIL | TV_ACCEL was discovered post hoc after multiple Gen21 microstructure features were inspected. This study tests four information sets and several preregistered volume features only on already-inspected development history; no multiplicity adjustment or untouched confirmation exists. | Positive raw H1 correlations cannot be promoted to confirmed evidence or justify Generation 22. |

## Adversarial decision

REJECT confirmation claim and STOP. The internal chronology is auditable, but H1 is post-hoc development evidence, the incremental model tests fail, and no untouched forward confirmation exists.

No methodology fix can turn already-inspected history into an untouched test. Confirmation requires a completely frozen specification and genuinely new post-cutoff data.
