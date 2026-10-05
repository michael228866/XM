# GOLD S4 Trade Economics v1

Implementation and certification are static/synthetic only. Real historical
execution belongs exclusively to the USER, by double-clicking `RUN_TRAINING.bat`.
No classifier training, prediction, calibration, label change or signal regeneration
occurs even during the real workflow. The retained historical name of the BAT does
not imply model training. No production promotion and no locked holdout access.

## Reference and scope

`A_NO_LONG_HTF`, original v2 run `20261003T172302Z_gold_s4_secondary_improvement_v2`,
is bound with its effective validator amendment, exact model/feature/config hashes,
original ledger and all 787 accepted entry timestamps, prices, eligibility and risk
distances. The frozen **accepted entry stream** has 787 entries; this is not a claim
that the raw classifier produced only 787 qualifying M1 rows. We do not rerun the
classifier or substitute new signals. The v4 run archived at `a270e86` did not replace
this reference. All intervals remain historical development, not untouched evidence.

Reference: 787 trades, 450 wins, 337 losses; WR 0.5717916137229987;
TPD 0.30778255768478685; PF 0.8521940468784924; Mean-R -0.06408993365369865;
PnL-R -50.438777785460836; DD -52.34913835149044; stress PF 0.8093582289789337.

Real execution first reproduces every reference exit, return and pooled/fold metric
using the sealed original M1 price evidence. Any discrepancy stops the run with
`REFERENCE_EXECUTION_MISMATCH`, research `NOT_RUN`, before other candidates run.
Reference reproduction is not claimed by infrastructure certification.

## Formal compatibility

Source of truth: **SINGLE_POSITION_REFERENCE_COMPATIBLE_ONLY**.

Every candidate starts with the complete frozen entry list. Before each entry:

* A previous position still open causes `REFERENCE_ENTRY_OVERLAP`.
* A nonpositive previous trade whose exit is less than 15 wall-clock minutes ago
  causes `REFERENCE_COOLDOWN_CONFLICT`. Exactly 15 minutes is allowed.
* Original S5 Taipei-calendar daily realized account loss guard remains active;
  violation causes `REFERENCE_DAILY_LOSS_CONFLICT`.

Any conflict marks the **entire candidate** `EXECUTION_COMPATIBILITY=FAIL`.
Its partial ledger and conflict entry ID are retained; no entry is skipped, moved,
replaced or treated as executable. Pooled economics are null and no economics gate,
official frontier or selection is allowed. The unmodified full entry list remains
in `signal_reference.json` for every candidate. There is no retry or relaxed rule.

S5 exits at OPEN (timeout; the predeclared no-progress OPEN exit) precede an entry
at that same timestamp. Intrabar barrier and cohort-end CLOSE exits do not precede
a same-timestamp OPEN entry. A positive trade does not erase the previous last-loss
timestamp. Existing risk budget is .014 of nonnegative account balance; rolling
30-trade account PF below 1.15 after 18 trades halves risk. The existing 120-minute
post-loss multiplier cap is 1.0. No resizing based on candidate stop distance.

The optional `OVERLAP_ALLOWED_DIAGNOSTIC` is **disabled**. No conditional diagnostic
can enter the official result. Fixed-entry comparisons still condition on historical
baseline acceptance and are not an untouched complete strategy validation.

## Frozen path and cost semantics

M1 HIGH/LOW, including entry bar; stop first if both stop and target are touched.
Entry is the immutable original OPEN. Baseline SL=max(1.6 ATR,.6), TP=max(1.3 ATR,1.5).
The baseline is not a symmetric 1R/1R payoff. Alternative stop/target levels use the
original SL price distance as unit. Return denominator is always original SL plus
entry spread*.01, so a wider stop cannot improve returns by redefining R.

Observed positive original spread, otherwise 30 points, is frozen at entry. Nominal
extra cost is 5 points, stress extra cost 10, point=.01, exactly the inherited combined
fee/slippage convention. Costs are never dropped and stress uses the same exits.
**Inherited gap limitation:** touched barriers fill at the barrier even across a gap,
as in the original S5. This is identical across candidates and is not live-fill proof.

Baseline timeout is 90 wall-clock minutes. Time exits use the first available OPEN
at or after the horizon, before that bar's HIGH/LOW. No-progress exits use only MFE
from preceding completed surviving bars. Stop changes caused by break-even or trailing
triggers activate next available bar, never favorably within the triggering bar.
MFE/MAE report completed surviving bars only; exit-bar unordered extremes are excluded.
These conservative diagnostics are not tick-accurate extrema. At the end of the exact
original universe, close at final CLOSE, as in S5; never fetch additional bars.

## Search space and gates

22 fixed configurations: control 1; stop 3 (.75/1.25/1.5 original SL); target 5
(.75/1/1.25/1.5/2 original SL); paired 3; timeout 4 (30/60/120/240 minutes);
break-even 2 (.5/.75); trailing 2 (trigger1, distance .5/.75); no-progress 2
(MFE below .2 at 30/60 minutes). Partial exits disabled: original accounting is
full-fill only. No Cartesian grid, adaptive generation or post-result edits.

Original three folds are retained. Official robustness requires execution compatibility,
WR>=.50, TPD>=.2770043019163082, stress PF>=.90; each fold >=10 trades, PF>=.85,
Mean-R>=-.15. The last floor was frozen from baseline fold means (-.0621151,
-.0254202,-.0730030), before candidate outcomes. PF without negative returns is
reported null and fails the finite official gate, rather than serializing infinity.

Only PF>1, Mean-R>0, PnL-R>0 and robustness can replace the economic reference.
STRONG requires PF>=1.10, Mean-R>=.05, stress PF>=1.00; TARGET requires PF>=1.20,
Mean-R>=.10, stress PF>=1.05. These are research labels only.

Official Pareto axes are PF and Mean-R, among compatible robustness-pass candidates.
Tie order: PF, Mean-R, stress PF, PnL-R descending; absolute DD ascending; TPD and WR
descending; candidate ID only for complete numerical ties. Incompatible candidates
never enter the frontier even if their hypothetical returns might be attractive.

Wins mean net R>0, losses net R<0, flats net R==0 and are reported separately.
`gross_win_r`/`gross_loss_r` mean sums of positive/negative **net** trade returns for
PF; pre-cost PnL and cost R are separately reported. Mean-R=PnL-R/trades. Drawdown is
the sequential unit-R ledger curve, not leveraged account drawdown. Each candidate
reports holding-time statistics, outcome counts, payoff ratio, fold and pooled metrics.

## Validation and archival

Independent validator separately implements exits, position/cooldown/daily compatibility,
accounting, gates, Pareto and selection. It compares full ledgers and raw event payload
hashes. The shared loader only verifies sealed data/entry identities; it neither trades
nor fits. `validator_attempt.json` is exclusive and one-shot; never delete to retry.

Research events end at `research_freeze`. `validation_result.json` and
`finalization.json` are separate tip-bound lifecycle records; they do not mutate the
research chain. Runs retain all source snapshots, approved configuration, full entry
reference, candidate/partial ledgers, conflict reasons, independent results and seal.
The manual workflow finalizes/registers then commits/pushes results, including aborts.

Infrastructure only: run compiler and synthetic tests, commit/push source, execute
`certify_gold_s4_trade_economics_v1.py --execute`, archive/push, prepare `--approve`,
commit/push approval, STOP. Certification never issues a USER receipt or reads real
price arrays. `CHECK_STATUS.bat` reads metadata only. Real sealed input availability,
baseline reproduction and economic improvement are left to the USER run.
