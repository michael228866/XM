# GOLD anchored episode expansion v1

Source-only, research-only family. The prior score-ranking family is CLOSED.
No formal run is authorized by implementation. Formal runner and validator are
invoked manually by the user, once each in a new preserved run.

## Frozen candidates

| Candidate | Window (minutes) | Score floor | Maximum decay |
|---|---:|---:|---:|
| A0_BASELINE_075 | 0 | 0.75 | None |
| E1_5M_F070 | 5 | 0.7 | None |
| E2_5M_F072 | 5 | 0.72 | None |
| E3_10M_F070 | 10 | 0.7 | None |
| E4_10M_F072 | 10 | 0.72 | None |
| C1_5M_F070_DECAY002 | 5 | 0.7 | 0.02 |
| C2_5M_F072_DECAY002 | 5 | 0.72 | 0.02 |
| C3_10M_F070_DECAY002 | 10 | 0.7 | 0.02 |
| C4_10M_F072_DECAY002 | 10 | 0.72 | 0.02 |

Exactly nine candidates; no candidate selection, tuning, extra candidates,
retraining, calibration, CLI parameter overrides, or production promotion.
A0 retains the ordinary score >= 0.75 gate and has no secondary entries.

## Episode and execution rules

All candidates reuse the exact archived B0 float32 OOF scores and score indices
from `20260913T141817Z_gemini_cftc_gold_cot_b0_b1_v1`.
The frozen FINALIZED.json hash is
`592b68e98c1a0e9dd6f5e5af1d6aeadbca86df5072f2cbe85a064adf06c5ebfe`.
The archived parent provides the exact GOLD reconstruction and frozen S5 code;
all dependency hashes are checked. No model is trained. The independent validator
reloads the archived B0 models and verifies exact prediction equality, while
using archived scores as its actual gate inputs.

- Process rows in archived index order within each chronological score fold.
- Every score >= 0.75 row retains baseline anchor entry eligibility, even when
  an earlier episode's secondary quota is exhausted or the anchor cannot trade.
- Only anchor rows open episodes. An anchor does not need to execute a trade.
- Publish a timestamp batch's last anchor only after the batch completes. For
  multiple anchors at one timestamp, the LAST in archived index order supplies
  the active anchor timestamp and anchor_score. Do not reject duplicates.
- No anchor at t may authorize secondary entry at t. A secondary row can use
  only the most recent anchor with strictly earlier timestamp. A pre-existing
  earlier anchor can still authorize rows in a batch before its new anchor is
  published; this never uses a same-timestamp anchor.
- A secondary row must satisfy 0 < elapsed time <= 5 or 10 calendar minutes,
  floor <= score < 0.75, and (for C1-C4) score >= anchor_score - 0.02.
  Boundaries are inclusive at the window end and at the score floor. Widen the
  exact archived float32 value to float64; do not round scores or decay floors.
- At most ONE actually executed secondary trade per episode. Only successful
  S5 open_trade consumes quota. Eligibility, existing positions, session, RSI,
  spread, cooldown, risk or any other execution rejection does not consume it.
  Later eligible rows may try while the episode is still valid.
- The episode ends when its secondary trade executes, its window expires, or
  a later anchor replaces it. Secondary rows cannot open or extend episodes.
- Episode state resets at each fold. S5 execution state runs continuously across
  the combined chronological folds separately for each candidate.

`execution_spec_gold_anchored_episode_expansion_v1.json` is checked against both
implementations and retained as run-local `execution_spec.json` before execution.
The exact S5 simulate function code is reused with private entry gate and
successful-open notification bindings. Shared modules and frozen source files
are never patched. Spread/cost, position occupancy, risk, TP/SL, timeout and
same-bar conventions remain unchanged. Anchor eligibility is preserved;
additional trades can change later occupancy/risk state, so candidate anchor
trade counts need not equal A0's count.

## Independent evidence and metrics

Runner batch-state episode reconstruction and validator strict-time searchsorted
reconstruction are separate; the validator does not import the new runner.
The validator recomputes all eligibility and actual-execution quota masks,
checks max one secondary per fold/anchor, matches every ledger field and all
pooled/fold metrics, verifies baseline reproduction and operational hashes.

Retain `episode_evidence.npz`: per-fold scores, indices, UTC nanoseconds and each
candidate's anchor flags, strictly prior anchor index, and secondary eligibility
before quota; plus combined candidate masks `allowed` (after quota, before S5
execution gates) and `executed` (actual opens). Prior indices in fold evidence are
fold-local; ledger anchor_row_index is the combined positional index.
Ledger rows additionally identify entry_kind, score_fold and anchor_row_index.

`metrics.json`, `candidates.csv`, `fold_metrics.csv`, `trade_ledger.csv` and
`report.md` retain every candidate, including failures, with trades, wins/losses,
realized WR, trades/day, PF, Mean-R, PnL, max DD, stress PF, break-even-adjusted
edge, guardrails and pooled baseline deltas. Each fold and pooled row also has:
anchor_trade_count, secondary_trade_count, secondary_wins, secondary_losses,
secondary_realized_wr and secondary_trades_per_day. Secondary wins use net_r > 0;
all other secondary trades are losses. No secondary trades means WR is null,
counts and frequency zero. Frequency uses full fold calendar-day denominators,
not active episode days. Identity/provenance and operational pre/post hashes are
retained, together with immutable source/validator snapshots.

## Baseline and frozen assessment

A0 must reproduce the archived B0 values with absolute tolerance 1e-12 and zero
relative tolerance before candidate evaluation continues:
689 trades, 390 wins, 299 losses; WR 0.5660377358490566;
trades/day 0.26945639421196715; PF 0.8247098331219882;
Mean-R -0.07690539315994348; PnL-R -52.98781588720106;
max DD-R -59.57958015890608; stress PF 0.7838186120438296.

Inherited frozen research guardrails reject pooled PF < 0.80,
Mean-R < baseline Mean-R - 0.03, stress PF < 0.75, or any fold with
trades < 10, WR < 0.45, PF < 0.70. These are research guardrails, not production
promotion criteria. Interesting requires BOTH WR and trades/day strictly above
baseline plus passing guardrails. Strong requires WR >= 0.58 and trades/day >=
0.40; target requires WR >= 0.60 and trades/day >= 0.50. Strong and target also
require guardrails. No winner is selected automatically. Historical results
remain development evidence, not untouched forward evidence.

## Source-only checks

```powershell
.\.venv\Scripts\python.exe -m py_compile .\gold_anchored_episode_expansion_v1.py
.\.venv\Scripts\python.exe -m py_compile .\validate_gold_anchored_episode_expansion_v1.py
.\.venv\Scripts\python.exe .\gold_anchored_episode_expansion_v1.py --self-test
.\.venv\Scripts\python.exe .\validate_gold_anchored_episode_expansion_v1.py --self-test
```

Self-tests use synthetic arrays and a five-row synthetic S5 cohort only. They
cover duplicate anchors, strict causality, future mutation, window/floor/decay
edges, fold reset, new-anchor renewal, occupancy/spread/RSI/risk rejection,
executed-only quota, original synthetic A0 ledger identity, preserved shared S5
functions and secondary metrics. They do not load historical market data, fit or
predict with models, run either formal entry point, or create training_runs.
Only a later manually authorized formal run can establish historical baseline
reproduction and independent methodology/provenance validity for this family.

## Manual formal lifecycle (not executed during implementation)

Use training_run_history.py create with experiment gold_anchored_episode_expansion_v1
and training script gold_anchored_episode_expansion_v1.py after a clean committed,
pushed pre-run gate. The runner accepts only --run-dir for a new in-progress run;
it verifies the committed snapshot and upstream identity and refuses retries.
Then manually execute validate_gold_anchored_episode_expansion_v1.py --run-dir
for that completed run once. Preserve any failure; never retry, tune, or repair
run evidence to obtain PASS. Finalization/registration is a separate operation.
Neither script creates a run or changes production. Closed-family source stays
unchanged.
