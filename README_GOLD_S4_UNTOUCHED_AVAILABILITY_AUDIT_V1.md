# GOLD S4 untouched availability audit v1

Source-only, read-only availability and provenance inspection for a **future**
one-shot protocol. This implementation does not run that protocol. The full
audit has not been run as part of implementation.

## Manual use

From `D:\XM\數據`, after reviewing this source:

```powershell
.\.venv\Scripts\python.exe -B audit_gold_s4_untouched_availability_v1.py --output gold_s4_untouched_availability_audit.json
```

The output path must be explicit, new, end in `.json`, and have an existing
parent. Existing files are never overwritten. Output inside `training_runs`,
`models`, `.git`, or `.venv` is refused. An optional repeatable
`--source-root D:\path\to\data` adds local read-only dataset sources. It does not
replace the canonical pipeline inputs. No network discovery occurs.

Only that JSON is written. No training run, registry entry, validator, model,
prediction, feature matrix, target, trade ledger, or performance file is created.
Failures before serialization return `AUDIT_ERROR` and exit 2; a successfully
written audit returns exit 0 even when its availability verdict is NOT_READY.
An interrupted write can leave a partial audit JSON; it is never overwritten.

## Frozen research chain

- Discovery: `training_runs/20260915T151723Z_gold_independent_secondary_classifier_v1`
  (`S4_SECONDARY_P075`, INTERESTING).
- Confirmation: `training_runs/20260919T124853Z_gold_secondary_s4_confirmation_v1`
  (SUPPORTIVE; primary/supportive true, strong false).
- B0 helper/model archive:
  `training_runs/20260913T141817Z_gemini_cftc_gold_cot_b0_b1_v1`.
- C1 archived `training_script.py` calls `barrier_final_train.py`, then
  `drl_trading_v2.py`; frozen S5 comes from
  `gold_gemini_execution_semantics_v1.py`.

The audit embeds the archived discovery dependency hashes, both finalized
archive hashes, and the 31-feature order. It checks live sources against those
identities and the archived runner/helper snapshots. Drift is a blocking issue,
never a reason to silently adopt new semantics. No repository research module
is imported. In particular, the data loader also constructs targets, so calling
it would violate this audit's scope.

The frozen future candidate remains primary `B0 >= 0.75`, or secondary
`B0 < 0.75 AND secondary >= 0.75`. Neither gate is evaluated here.
Historical confirmation is development robustness, not untouched promotion
evidence. No production selection, promotion, or change is authorized.

## Canonical data and checks

The traced loader reads repository-root `GOLD#_*.csv`, sorted lexically; the
second underscore-delimited filename component is its timeframe. Duplicate
timeframe files overwrite each other in the original loader. This audit reports
that ambiguity and does not choose a file. `XAUUSD` alternatives are inventoried
but never substituted for `GOLD#`.

The base is M1. The twenty separate native inputs are Daily, H12, H8, H6, H4,
H3, H2, H1, M30, M20, M15, M12, M10, M6, M5, M4, M3, M2, Weekly, Monthly.
The frozen loader does **not** resample M1 to manufacture them. All need
DATE and OHLC because `add_indicators` is called even for trend-only inputs.
Intraday timestamps need TIME or an unambiguous time-bearing DATE. M1 also
needs SPREAD. Angle-bracket headers are normalized as in the original source;
delimiter detection accepts comma, tab, and semicolon; UTF-8/BOM and UTF-16/BOM
exports and CSV gzip alternatives are supported.

Each recognized raw CSV is hashed and streamed for post-2025 rows, first/last
timestamps, distinct observed dates, duplicates, original-order reversals,
invalid timestamps, nonfinite/nonpositive prices, incoherent OHLC and spread
fallback use. Pre-2025 input is retained only as causal context. The audit does
not sort, repair, fill, or rewrite a dataset. A gap is an **unclassified elapsed
interval**, not automatically a missing bar: nominal M/H intervals, 24 hours,
7 days, or the maximum 31-day month separate examples. At most 200 gap examples
per file are retained with total counts and a truncation flag.

Other GOLD/XAUUSD CSVs/caches/NPZ/Parquet/NPY/pickle files are listed by identity;
outcome ledgers and unknown derived formats are not deserialized. Their unknown
timestamp coverage is explicit. Recursive inventory excludes Git internals,
environments, package caches, and links outside the enumerated source root.
Available local absolute dataset references are followed; additional sources
can be supplied explicitly. Unavailable, dynamic, untracked/deleted and external
sources cannot establish complete provenance.

## Warmup, execution and unknown completeness

M1 ATR(14) first exists at row index 13; its rolling(240) mean first exists at
252; the global one-row feature lag gives a **254-row structural minimum**
(first usable index 253). RSI/ATR execution inputs require their corresponding
lag (structural index 14). Other finite M1 windows are shorter. MACD uses
EMA12/EMA26 and EMA9 with `adjust=False`, initialized from the first source
close: **no finite truncated warmup guarantees exact historical EWM state**.
An attested original source prefix is required for exact reuse.

Each timeframe trend compares CLOSE with rolling(20), shifts one timeframe row,
joins backward with `merge_asof`, then receives the global M1 lag. Twenty genuine
prior closes plus the publication row and a following M1 row are required.
The original `np.where` yields -1 while rolling(20) is missing; that artificial
initial trend is not treated as genuine warmup completion.

The output distinguishes `structural_feature_lower_bound` and
`structural_execution_lower_bound` from **proven** earliest valid timestamps.
Naive broker/API timestamps, actual UTC/DST mapping, bar open/close labeling,
final-bar closure, higher-timeframe freshness, and original recursive-feature
prefix identity need provenance beyond OHLC row counts. The frozen entry-hour
and weekday gates are not a broker market calendar: they cannot prove holidays,
partial sessions, or missing market bars. This v1 has no authoritative broker
calendar/prefix attestation input and therefore deliberately does not certify
those conditions. `feature_pipeline_reconstructable` and
`execution_pipeline_reconstructable` mean **proven exact reconstruction**, not
merely the presence of columns, and remain false until those blockers are
resolved in a reviewed follow-up.

Accordingly actual raw timestamps/counts are reported, but proven feature/
execution starts, safe start/end, complete days/months, eligible M1 rows, and
complete sessions remain `null` where not established. Zero *certified* days
means no days have been certified, not that no raw data exists. The audit never
invents a market calendar or converts a research clock to UTC by assumption.

S5 needs TIME_DT, OPEN/HIGH/LOW/CLOSE, lagged ATR/RSI and SPREAD. The frozen
spread rule uses positive finite SPREAD or a 30-point fallback; a missing column
is a separate loader blocker. POINT=0.01, extra cost=5 points, stress extra=10
points, next-row open entry, entry-bar stop-first handling, 90 wall-clock minute
timeout, cooldown, occupancy, sizing and TP/SL are retained by source identity.
None is replayed. No RSI, ATR, signals, probabilities or execution paths are
calculated during this audit.

## Contamination and training overlap

Worktree source, READMEs, archived specs/reports, `TRAINING_RUNS.md`, and reachable
Git history blobs (`git rev-list --objects --all`) are searched. Evidence records
carry file/line or immutable Git blob identity and snippets. The text scan is a
conservative search for leads, not proof of execution:

- `SAFE_CONTEXT`: only raw ingestion/schema/warmup context is visible.
- `POTENTIAL_CONTAMINATION`: dated research references or uncertain context.
- `CONFIRMED_CONTAMINATION`: explicit repository policy evidence or verified
  candidate-period frozen-model fitting indices.

Source code alone is not labeled proof of fitting. Repository `AGENTS.md`
explicitly says the historical 2014–2026 periods were repeatedly inspected and
that data at/after `2026-09-01T02:00:00Z` was inspected. That is contamination
evidence, not a new safe boundary. This audit does not turn the cutoff into a
clean successor by adding a minute. Missing/unbounded contamination endpoints
leave the earliest safe untouched start unknown.

No matches do not prove untouched status. The scan cannot exclude manual or
off-repository inspection; unresolved leads and scan omissions block readiness.
Text blobs above 4 MiB and undecodable files are reported as scan limitations.
Reachable Git objects do not cover deleted uncommitted files, inaccessible
worktrees or every reflog/dangling object. Scanning large repositories/history
can take time; no performance data is computed while searching existing text.

For each of three specialists in each frozen archive, the audit reads only
`foldN_train_indices`, `foldN_subset_indices`, and `foldN_train_ns` from NPZ.
It verifies archive/member hashes, integer arrays, ordered training indices,
subset membership/hash, model SHA, objective, and 31-feature structure. Exact
first/latest fitting timestamps come from the subset's timestamp mapping.
Targets, weights and prediction arrays are never loaded. B0 artifacts are also
checked structurally without creating any model object.

Design A (already frozen models) needs no fitting and is structurally possible
if these artifacts verify, conditional on feature availability and untouched
provenance. A protocol must predeclare which frozen fold/model combination is
used. Design B (deployment-style refit) requires a new predeclared protocol,
training strictly before the untouched boundary with label horizon preserved.
This audit chooses neither design and performs neither.

## Duration and verdicts

Classification uses only **certified complete untouched calendar days**:

| Complete days | Classification |
|---|---|
| <90 | INSUFFICIENT_DURATION |
| 90 to <180 | LIMITED_DURATION |
| 180 to <365 | MODERATE_DURATION |
| >=365 | SUBSTANTIAL_DURATION |

Raw elapsed days and distinct observed dates are separate, never substitutes
for complete untouched days. Unknown completeness is not a measured shortage;
the provenance/contamination blocker takes priority over duration.

Verdict precedence: confirmed contamination, missing required data, incomplete
provenance, insufficient duration, then READY_FOR_PROTOCOL_DESIGN. All blocking
issues remain in the JSON regardless of the selected status. READY means only
ready to design a protocol, not a strategy pass or promotion. This v1's known
calendar/prefix/manual-provenance limitations prevent READY for the current
chain; they are explicit blockers, not assertions that the raw data is absent.

## Safety and permitted implementation checks

The implementation uses the standard library plus existing NumPy solely for
integer archive metadata. An import guard blocks model packages and repository
research/replay modules. A process audit hook rejects network access, mutation
outside the one explicit output, and subprocesses other than allowlisted
read-only Git operations. Bytecode generation is disabled in audit/self-test
mode. These guards prevent accidental execution paths; they are not a sandbox
for arbitrary hostile native extensions.

Synthetic tests use in-memory CSVs and arrays, never real datasets or archives.
They cover the date boundary, retained warmup, missing/invalid inputs,
duplicates/reversals/gaps, contamination classes, null/max safe boundaries,
duration thresholds, all verdict branches, warmup indices, complete versus
gapped synthetic calendar days, and the forbidden model import.

The only implementation validation commands are:

```powershell
.\.venv\Scripts\python.exe -m py_compile audit_gold_s4_untouched_availability_v1.py
.\.venv\Scripts\python.exe -B audit_gold_s4_untouched_availability_v1.py --self-test
git diff --check
```

The full audit, fitting, scoring, backtesting, validators and production files
are outside these implementation checks.
