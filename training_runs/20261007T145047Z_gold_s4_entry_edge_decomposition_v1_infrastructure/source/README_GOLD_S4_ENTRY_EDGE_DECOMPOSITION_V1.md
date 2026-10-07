# GOLD S4 Entry Edge / Setup Quality Decomposition v1

This workflow describes the immutable 787 accepted A_NO_LONG_HTF trades. It does
not generate a signal, fit/predict/recalibrate a model, simulate exits, change a
trade, select an entry filter, or promote production. Historical development
results are not untouched OOS/forward evidence. The future holdout remains locked.

Only the USER may manually double-click RUN_TRAINING.bat after infrastructure
certification and committed approval. Codex may run the named synthetic test and
certification scripts, but must never run the BAT, historical runner or real
validator. CHECK_STATUS.bat reads approval/run metadata only.

## Frozen inputs and causality

`gold_s4_entry_edge_reference_v1.json` binds the original v2 ledger, the sealed
20261005T163507Z E0 control ledger, original score-matrix chunks, archived secondary
probabilities and source seals. It binds all trade fields by the complete ledger
hash, including timestamps, prices, exits, net R, stress R and costs. The original ledger
is the sole source of trade values; E0 only corroborates identity and supplies
holding time and completed-bar excursion diagnostics. Entry features
are joined by original fold-relative decision index and verified timestamp;
the independent validator joins by timestamp and cross-checks the index.
No price paths, training arrays, models, raw CSVs or holdout are loaded.

All 31 stored entry feature columns are retained as a snapshot. Approved fields
come from the existing shifted feature pipeline: preceding completed M1 values,
backward-joined shifted HTF trends, and original past-only fold predictions.
The frozen inventory is a schema/source review. `missing_rate: null` means not
measured by Codex; the USER run records actual missingness before grouping.
Missing values receive their own bucket; no trade disappears.

The secondary probability is **not** the sole confidence that accepted a trade:
A_NO_LONG_HTF includes the incumbent B0 union. Score buckets diagnose the stored
secondary probability across the unchanged accepted set, not calibration to
TP-first labels. Predicted-score mean and realized positive-trade WR are distinct.

Completed-bar MFE/MAE in E0 exclude unordered extremes on barrier exit bars. They
are lower-bound excursion diagnostics, not full intrabar excursions. They and
holding duration may be metrics only. Net R and stress R are copied unchanged:
original spread plus 5 cost points base, plus 10 stress, original R denominator.
TP-exit share is reported separately from realized positive-trade WR.

## Predeclared partition specification

10 dimensions: UTC four-hour windows, UTC weekday (including weekend buckets),
volatility ratio versus 1, M5 trend, H1/M5 alignment relative to LONG, MA20 price
deviation sign (trigger-extension proxy), five-bar return sign (momentum proxy),
RSI 30/70 zones, candle-body majority at 0.5, and user-specified score bands.
Volatility MEDIUM means exactly equal to the rolling baseline ratio of 1, not an
estimated middle quantile. Momentum/extension are sign proxies, not newly
reconstructed triggers or arbitrary magnitude thresholds. Numeric interval bins
are left-inclusive; equality goes to the upper interval. All rules are frozen
before real execution, with no fitted quantiles or outcome-based cutoffs.

Eight predefined pairs: time/alignment, volatility/alignment, score/alignment,
score/volatility, extension/volatility, momentum/alignment, RSI/alignment and
body/volatility. Univariate tables are written and hash-frozen before bivariate
work. Every dimension and pair is a complete partition of the same 787 entries;
all empty and missing cells are reported. No three-way or higher-order search.

Unsupported: momentum acceleration, recent-range position, decision-time spread
quote, cooldown snapshot and prior-outcome snapshot. No unsupported reconstruction.

## Interpretation

Univariate samples <30 are descriptive, 30-49 low confidence, >=50 usable.
Bivariate samples <20 are descriptive; 20-29 can meet the cell sample floor but
cannot receive a positive/negative edge label (edge labels still require >=30).
Missing buckets are always descriptive and never research candidates.

Positive: n>=30, PF>1, Mean-R>0 and PnL-R>0. Robust positive: n>=50, PF>=1.10,
Mean-R>=0.05, stress PF>=0.95. Negative: n>=30, PF<0.85, Mean-R<0.
PF with no loss denominator is null with an explicit status; it cannot qualify
for edge, stability or candidate labels. Empty means/WR are null.

STABLE requires all three folds >=8 trades with finite PF, at least two fold
PFs >1, and no fold PF<0.70. Inadequate/catastrophic folds are UNSTABLE; remaining
cases MIXED. Persistent weakness separately requires all folds >=8 with finite
PF<1, at least two PF<0.85 and all Mean-R<0. Positive research candidates require
n>=50, PF>=1.05, Mean-R>0, stress PF>=0.95 and STABLE. Exclusion research candidates
require n>=50, PF<=0.75, Mean-R<0, stress PF<=0.75 and persistent weakness.

Positive rankings use sample confidence, stress PF, PF, Mean-R, stability and
PnL-R lexicographically. Every qualifying bucket is retained, not truncated.
`EDGE_BUCKETS_FOUND` means at least one positive filter **research** candidate;
otherwise `NO_ROBUST_EDGE_BUCKETS_FOUND`. Neither creates a production filter.
Execution PASS and research result are separate. Reference mismatch stops before
grouping with NOT_RUN and REFERENCE_TRADESET_MISMATCH (category REFERENCE_MISMATCH).

All cells and fold metrics are retained. Counts in multiple_testing_inventory.json
include empty/missing cells and overlapping tests. Rankings are exploratory, not
confirmatory evidence. Bootstrap is disabled; no uncertainty claims are made.
Subset trades/day uses the fixed full fold duration and is descriptive, not a
backtest of an exclusion strategy. Entry filtering would change occupancy and
requires a separate, user-authorized preregistered experiment.

## Outputs and preservation

Five official CSVs and the four requested JSON summaries/inventories are emitted,
plus original ledger, all entry snapshots, measured inventory and reproduction
metrics. Canonical event hashes cover the raw event object excluding only its own
hash; one chain covers reference load through validation and finalization, with
artifact digests. The independent validator recalculates every official output.

Real runs require clean committed/pushed main, record the executed source commit,
use training_run_history.py, retain FAIL/aborted outcomes, validate preserved
provenance, then commit and push the archive. No finalized archive is modified.
Source arrays are retained in existing local immutable archives with Git-tracked
hash seals. No remote raw-data backup or full original raw MT5 retention is claimed.

Infrastructure commands (allowed for Codex):

    .venv\Scripts\python.exe -B test_gold_s4_entry_edge_decomposition_v1.py
    .venv\Scripts\python.exe -B certify_gold_s4_entry_edge_decomposition_v1.py --execute

Commit/push source before certification; commit/push certification before:

    .venv\Scripts\python.exe -B certify_gold_s4_entry_edge_decomposition_v1.py --approve <infrastructure-run>

Approval installs only the prospectively certified BAT text and workflow binding.
Commit/push approval, verify HEAD/remote and clean status, then STOP. Do not run BAT.
An infrastructure PASS proves static/synthetic logic and schema bindings only;
real artifact content, exact 787 reference reproduction, and feature joins remain
mandatory USER runtime checks. Python audit hooks enforce reviewed paths and writes;
they are not an OS sandbox against hostile native code.
