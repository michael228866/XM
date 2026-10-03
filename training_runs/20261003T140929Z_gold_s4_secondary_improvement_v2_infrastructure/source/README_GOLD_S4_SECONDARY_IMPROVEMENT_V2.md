# GOLD S4 Improvement v2

Real training **and real validation are USER-only**. After infrastructure approval,
manually double-click `RUN_TRAINING.bat`. Codex, scheduling, startup and CI must not
launch it or its real Python equivalent. `CHECK_STATUS.bat` reads metadata only.

## Frozen reference and scope

F5_NO_REDUNDANT_HTF is the historical reference, bound by raw source-run seal,
model hashes, effective PASS amendment and revalidation receipt in
`gold_s4_f5_reference_v1.json`. Its final-fold model SHA256 is
`617e5724e2405e7423ec61b4104c4272d9094ed1fe29d6da95735326baf18753`.
Control loads the three existing F5 models and independently reproduces all ten
reference metrics to absolute tolerance 1e-12. **It never retrains F5.** A mismatch
aborts with `F5_REFERENCE_CONTROL_MISMATCH`, research `NOT_RUN`.

All inputs are already certified historical feature/label/price chunks retained
in the F5 source run. No external feed, new instrument, broker fetch or future
holdout is used. Each input is checked against its immutable archive hash, feature
and label hashes, original fold indices and maturity cutoffs. Loading the retained
31-column input does not reintroduce the original baseline: every candidate starts
from F5's 27-feature set; only those columns and explicitly named transforms are used.

## Predeclared 27 configurations

| Family | Count | Frozen scope |
|---|---:|---|
| Control / RAW / BALANCED | 1 | Existing F5 models, threshold 0.75 |
| A | 8 | Semantic momentum, oscillator, volatility, HTF and LTF removals |
| B | 2 | PLATT and ISOTONIC |
| C | 4 | NONE; balanced positive multipliers 1.10, 1.20, 1.30 |
| D | 8 | One local regularization change per configuration |
| E | 4 | Five total causal pointwise feature extensions |

`gold_s4_secondary_improvement_v2_search_space.json` lists every parameter and
feature. No candidate is added, dropped or shortlisted dynamically. Every candidate
uses all three original folds. The semantic ablations are hypotheses, not claims
of newly measured correlations. No real outcomes were inspected to choose groups.

The five derived features reuse certified decision-time inputs: MACD/absolute ATR,
body/absolute volatility ratio, HTF mean direction, LTF mean direction, and their
product. Denominators use float32 1e-6. These are pointwise transforms, with no
lookahead, fitted normalization, external data, or new price bars.

Calibration holds out the last three months *inside each training interval*.
Training labels and legacy maturity must precede the calibration cutoff. All
calibration labels must mature before the scoring fold. Require 2,000 calibration
samples and at least 100 per class, otherwise abort the whole run rather than alter
the frozen candidate inventory. PLATT uses C=1 logistic regression on clipped
logits. ISOTONIC uses monotonic regression with endpoint clipping. Neither fits on
the scoring fold. RAW is the fixed control and the default for other families.

## Gates and selection

INTERESTING requires WR > 0.5703125 and trades/day > 0.30035197497066873,
PF >= 0.80, Mean-R >= -0.10690539315994348, stress PF >= 0.75, and all three
folds with trades >= 10, WR >= 0.45, PF >= 0.70. STRONG additionally requires
WR >= 0.58 and trades/day >= 0.40; TARGET requires 0.60 and 0.50.

The safe WR/trades-per-day Pareto frontier is ranked by WR, frequency, stress PF,
PF, Mean-R, then smaller absolute drawdown. Candidate ID resolves only exact ties.
Only candidates improving both primary objectives are selectable. A correct run
without such a candidate returns execution PASS, NO_IMPROVEMENT_FOUND, gate NONE.

Economics is descriptive: POSITIVE_EXPECTANCY when PF > 1, Mean-R > 0 and PnL > 0;
NEGATIVE_EXPECTANCY when all three are below break-even; otherwise NEAR_BREAK_EVEN.
This label does not alter the research gate. All results remain historical
development evidence, never untouched promotion evidence. No automatic promotion.

## Ownership and provenance

The launcher requires the existing interactive Explorer -> CMD parent chain and
one-use, process-bound, 30-second receipt. The v2 policy prohibits automatic real
training, validation and research. Independent validation receives a separate
30-second single-use permit tied to the same active USER process and exact run.
Direct trainer/validator CLIs abort. This is an execution architecture, not a claim
to identify Codex or sandbox hostile code running as the Windows user.

Approval binds code/config raw hashes to a sealed synthetic certification. New
source files use Git `-text`; original historical byte hashes remain untouched.
F5 raw model/input hashes are bound through its archive, and the effective PASS
amendment preserves original FAIL evidence and the separate CRLF/LF adjudication.

The independent validator does not import the v2 trainer or selector. It checks
all configurations, event chronology, model/feature/label hashes, parameters,
weights, calibration isolation, predictions, historical execution metrics, ledger,
fold safety, Pareto, selection and economic labels. Its isotonic check uses an
independent weighted PAV implementation. Validation-only execution is not exposed
as a standalone real-data command. A one-shot attempt marker prevents retry.

## Implementation checks only

`test_gold_s4_secondary_improvement_v2.py` uses generated arrays and fake metrics.
Its audit hook rejects real historical exports, retained arrays/models and all
locked holdout access. Only frozen reference metadata can be read. Synthetic mode
cannot create a real execution session. It does not fit any candidate model.

`certify_gold_s4_secondary_improvement_v2.py --execute` archives static/synthetic
certification with all four real-execution flags false. After commit/push,
`--approve <sealed infrastructure run>` binds the launcher. Neither command starts
training or historical validation. Stop after approval; the USER starts Train + Val.
