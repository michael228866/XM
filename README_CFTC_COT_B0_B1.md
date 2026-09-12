# Frozen CFTC GOLD COT B0/B1 comparison

Source-only implementation for a later research run. Creating these scripts and
running their self-tests does not execute the formal experiment. No production
promotion, feature selection, hyperparameter search, threshold search, transforms,
alternate lags or external features are permitted.

## Frozen inputs

- Execution-aligned parent: `20260903T071729Z_gemini_execution_aligned_label_v1`.
- Treasury B0 reference: `20260909T140906Z_gemini_us_treasury_real_rate_b0_b1_v1`.
- Timestamp archive: `20260905T171629Z_gemini_macro_event_integration_foundation_v1`.
- Certified CFTC source: `20260912T171048Z_gemini_cftc_gold_cot_foundation_v1`.

All four archives must have valid `FINALIZED.json` inventories and unchanged
recorded files. The CFTC manifest must have status `pass` and its independent
validator must report `PASS`. Both new scripts independently require:

| Identity | SHA-256 |
|---|---|
| Logical CFTC matrix | `be6245e7b79695b338ec0ec1c493f045123254a0fccd2952670ae553fa633a09` |
| Physical CFTC NPZ | `24ebca3f1a1dec99cb2b6979f922425237623ff9f1809b64e53b9c00e69467a1` |
| CFTC observations | `2713dc12ff22c8661ea3ce1c92f34c74019b137592e45edc75d78e9ec794b9ab` |

The six broker timestamp row counts/hashes, six float32 GOLD matrix hashes and
three int8 C1 training target hashes are literal constants in each script.
Every reconstructed block must match before any model fitting. The original
float64 CFTC blocks are appended without normalization, winsorization or search.

## Exactly two candidates

1. `B0_31_technical`: the same ordered 31 technical/MTF features as the Treasury B0.
2. `B1_31_plus_5_cftc_cot`: those 31 features followed by these five:
   `COT_MM_NET_PCT_OI`, `COT_MM_NET_CHG_1W`, `COT_PROD_NET_PCT_OI`,
   `COT_SWAP_NET_PCT_OI`, `COT_MM_VS_PROD_SPREAD`.

B1 has exactly 36 features. Neither candidate changes CFTC availability,
report-date priority, causal joining, label definitions or timestamp identities.

The runner uses the archived parent's actual reconstruction, target generation,
class-balanced binary training and S5 execution functions, following the Treasury
comparison's train/score selection. Local transitive Python dependencies are
frozen by SHA-256 before import; drift fails rather than silently changing S5 or
the feature pipeline. The operational model is never loaded for prediction by
this comparison or replaced.

Training is fixed at XGBoost 3.2.0, binary logistic, 220 trees, learning rate 0.05,
depth 4, minimum child weight 80, subsample/column sample 0.85, seed 42 and histogram
trees. The reference trainer's CUDA-if-available behavior and CPU scoring are
preserved. B0's reproduction gate catches numerical or hardware differences that
change its economics. Windows, purges and maturity rules remain the parent's
18-month training windows, with score folds 2018–2020, 2021–2022 and 2023–2024.
Threshold is 0.75. S5 retains open entry, entry-bar exits, HIGH/LOW stop-first
handling, no trailing, 90 wall-clock minute timeout, observed/fallback spread,
5-point nominal and 10-point stress extra costs, and the existing live eligibility,
spread gates and risk state. No S5 source file is changed.

## B0 reproduction gate

Every field below is required with zero relative tolerance and absolute tolerance
`1e-12`. A mismatch fails the formal run before fitting B1.

| Metric | Required value |
|---|---:|
| Trades | 689 |
| Wins / losses | 390 / 299 |
| Realized WR | 0.5660377358490566 |
| PF | 0.8247098331219882 |
| Mean-R | -0.07690539315994348 |
| PnL R | -52.98781588720106 |
| Max DD R | -59.57958015890608 |
| Stress PF | 0.7838186120438296 |
| Trades/day | 0.26945639421196715 |

## Evidence and interpretation

The later formal run retains six fold models, actual training configurations,
model inventory and hashes, paired OOF predictions, reconstruction identities,
CFTC provenance, execution specification, trade ledger, fold/pooled metrics,
exactly two candidates, report, environment and operational pre/post hashes.
The manifest references retained models and evidence for the repository's normal
finalize/register workflow. Historical raw GOLD CSVs are not duplicated; this
limitation is stated in the manifest. Any failed identity or B0 gate stops the run;
an interrupted/failed attempt must be retained using the existing aborted-run
workflow, never retried in the same directory.

The report includes both candidates' trades, wins, losses, WR, PF, Mean-R, PnL,
max DD, stress PF, trades/day, fold metrics and B1-minus-B0 deltas. Hard quality
requires pooled WR >= 0.60, PF > 1.05, Mean-R > 0, PnL > 0, break-even-adjusted
WR edge > 0, stress PF > 1.00 and no catastrophic fold. The parent's catastrophic
definition is retained: any fold with fewer than 10 trades, WR < 0.50, PF < 0.80,
Mean-R < -0.10 or max DD < -20 R fails. Preferred PF >= 1.15 and stress PF >= 1.05
are reported separately. Frequency matters only after hard quality passes; more
trades below the hard gate still means CFTC direct-feature family `FAIL`.
Even a quality `PASS` remains research-only historical development evidence.

The independent validator does not import the new runner. It has its own constants,
archive checks, fold reconstruction, arithmetic and quality checks. It reuses only
the hash-verified frozen parent feature/target pipeline and S5 simulator as the
reference semantics; it never invokes parent training. It reloads all retained
models, verifies feature order/rounds/configuration and reproduces every prediction,
then replays S5. Nominal/stress R and fold/pooled economics are independently
calculated from the matched ledger. It writes `validator.json`, `validator.md`,
overall PASS/FAIL and failed check names; validation PASS is an evidence integrity
result, not a production promotion or necessarily a family quality PASS.

## Static checks only for this task

```powershell
.\.venv\Scripts\python.exe -m py_compile .\cftc_gold_cot_b0_b1_v1.py
.\.venv\Scripts\python.exe -m py_compile .\validate_cftc_gold_cot_b0_b1_v1.py
.\.venv\Scripts\python.exe .\cftc_gold_cot_b0_b1_v1.py --self-test
.\.venv\Scripts\python.exe .\validate_cftc_gold_cot_b0_b1_v1.py --self-test
```

Self-tests use synthetic arrays and temporary directories. The validator also
fits a two-tree classifier to eight synthetic rows to check actual XGBoost
configuration serialization; no historical models or run directories are created.

Formal execution is intentionally separate: after explicit authorization, use
`training_run_history.py create` with experiment name
`gemini_cftc_gold_cot_b0_b1_v1` and this runner as the script, anchored to a committed
and pushed source revision. Invoke the runner with `--run-dir` pointing to that
new in-progress run, retain stdout, and then invoke the independent validator with
the same `--run-dir`. Do not validate into a finalized archive. Only after reviewing
the evidence should the existing finalize/register workflow be used; this document
does not authorize running it now.
