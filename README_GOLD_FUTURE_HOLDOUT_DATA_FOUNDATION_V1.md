# GOLD future holdout data foundation v1

This is a **data/provenance foundation only**. It inspects local source text,
configuration references, raw schemas, timestamps and file identities. It
does not create a holdout. It determines which infrastructure and evidence
are still needed before an immutable Future Locked Holdout Protocol can be
frozen.

Historical 2025+ data is already contaminated for promotion-quality use. The
user's previous availability audit reported `NOT_READY_CONTAMINATED`, a stale
canonical endpoint of `2026-05-08T23:57:00`, and no certified untouched days.
That report is neither modified nor opened by this implementation. These
historical conclusions are recorded as user-supplied context, not recomputed.

No strategy result is evaluated; no model is loaded or trained; no predictions,
labels, signals, S5 replay, backtests or promotion evaluation occur. No
production change is made. Historical rows may establish schema, timestamp
format and warmup mechanics only, never promotion evidence.

## Files and manual invocation

- `gold_future_holdout_data_foundation_v1.py`
- `execution_spec_gold_future_holdout_data_foundation_v1.json`
- This README.

The full audit is **not run during implementation**. After source review, the
user may run:

```powershell
.\.venv\Scripts\python.exe -B `
  .\gold_future_holdout_data_foundation_v1.py `
  --output .\gold_future_holdout_data_foundation_report.json `
  --source-root "D:\some\raw\data"
```

`--source-root` is optional and repeatable; it may identify a local directory
or file. `--repo-root` optionally selects a checkout; default is the script's
directory. The frozen spec always comes from beside the script. Its canonical
JSON SHA256 is embedded in the implementation, so Windows newline conversion
does not invalidate the spec. Its source hashes detect changes to the traced
source/reference chain; they do not authorize changed pipeline semantics.

Only the explicit **new** JSON is written, UTF-8 with `ensure_ascii=False`.
Its parent must exist. Existing files, protected directories (`training_runs`,
`models`, `.git`, `.venv`) and UNC/network shares are refused. No directories
are created. Self-test writes nothing. An interrupted output write can leave
a partial new report; a later run must not overwrite it.

A report with NOT_READY is a successful inspection, exit 0. An inability to
complete the program safely returns `FOUNDATION_ERROR`, exit 2. File-level
inspection gaps appear as scan limitations and block readiness.

## Frozen pipeline trace

The spec records hashes for:

- Independent secondary discovery and S4 confirmation sources and archived
  runner snapshots.
- The repaired availability audit at `ef6e2f8`, whose Windows-safe subprocess
  guard is copied independently rather than imported.
- B0 and C1 archived helpers, `barrier_final_train.py`, `drl_trading_v2.py`,
  `gold_gemini_execution_semantics_v1.py` and `gemini.py`.
- Existing data-foundation audit and forward collector source.

These are inspected as text/bytes only. Research modules are never imported;
their loaders also construct targets and are unsuitable for this foundation.
The frozen conceptual gates remain primary B0 >=0.75 or secondary B0 <0.75
with specialist >=0.75. Neither condition is evaluated here.

The historical loader expects M1 and twenty **separate native timeframe
files**: Daily, H12, H8, H6, H4, H3, H2, H1, M30, M20, M15, M12, M10, M6,
M5, M4, M3, M2, Weekly, Monthly. Root `GOLD#_*.csv` files are read in lexical
order; duplicate timeframe names overwrite in the historical implementation.
This foundation reports ambiguity instead of reproducing that overwrite.

All timeframes need OHLC, since the loader invokes its indicator helper for
each input even when only a trend is used. M1 requires SPREAD. The exact
31-feature order and execution input names are frozen in the spec.
`SPREAD_POINTS` is recognized as an available raw alias but is not silently
converted to SPREAD. Exact S5 reconstruction additionally needs certified
clock, units and closed-bar semantics.

Deterministic resampling is technically possible only with complete input and
certified broker bar/calendar boundaries. Replacing the native-file inputs
would change the frozen protocol: `PROTOCOL_CHANGE_REQUIRED`. No resampling
or feature calculation occurs here, and no such change is authorized.

## Source discovery and limits

The user-run audit scans the checkout, local absolute source references and
explicit source roots. It excludes Git internals, virtual environments,
package caches and links outside the root. It inventories GOLD/XAUUSD-named
data and data under explicitly supplied roots. Native filenames and collector
`m1/GOLD_` directories supply timeframe hints; unknown naming stays unknown.
Filename hints do not attest exact broker symbol identity or native origin.
Per-file row `SOURCE_SYMBOL` declarations are retained where available.

Small Python, PowerShell and configuration files are searched for collector,
API, export, timezone, calendar, scheduling, terminal and integrity references.
Python literal paths are parsed with AST, never executed. Evidence records
carry path, line and category, without copying source bodies or credentials.
Static references do not prove a service is running. Dynamic paths, absent
files and scan failures remain unresolved. A literal drive root is not
automatically crawled. No broker/network requests, MT5 connection, terminal
launch or collection is performed.

Text scanning is capped at 1 MiB/file and 100 matching lines/file; omissions
are explicit. Model/result/ledger/cache artifacts and arbitrary JSON reports
are metadata-only. Unknown derived NPZ/NPY/Parquet/pickle contents are not
deserialized. README/result report bodies are not used as proof of fresh raw
data. The scan examines current local files, not unavailable/deleted sources
or remote infrastructure.

Each source/reference uses one of VERIFIED_LOCAL_RAW_SOURCE,
POTENTIAL_RAW_SOURCE, DERIVED_SOURCE, RESULT_OR_OUTCOME_ARTIFACT,
CONFIG_REFERENCE, UNRESOLVED_REFERENCE. VERIFIED_LOCAL_RAW_SOURCE means only
an accessible raw schema was inspected; it does not certify cleanliness,
native origin, continuing capture, or timezone. An outcome-like path or
outcome/target-bearing header is rejected before its body is interpreted.

CSV bodies are streamed. Duplicate detection retains timestamp identities in
memory; its memory use grows with distinct rows. Supported encodings are
UTF-8, UTF-8 BOM, UTF-16 BOM; delimiters are comma/tab/semicolon; CSV gzip is
supported. DATE/TIME, TIME_DT, TIMESTAMP, BAR_OPEN_UTC and TIMESTAMP_UTC text
timestamps are recognized. Numeric epoch units are not guessed. Unicode
paths use native `pathlib` and never terminal display text as identity.

Inventory includes first/last timestamps, row count, duplicate/reversed/invalid
timestamps, nonfinite/raw-price defects, spread positive-finite rate and
fallback dependence. Gap examples are observations, not missing-bar or DST
proof. Hashes are streamed; a size/mtime change during inspection invalidates
that source's verified classification. A file hash is a snapshot identity,
not a guarantee against later or covert mutation.

Files/partitions are not merged. The M1 reference is the unique historical
canonical input, otherwise the unique M1 candidate. All other candidates and
their timestamps remain visible. Multiple partitions need an authenticated
source manifest and overlap policy before they can be treated as a dataset.
Only root canonical files sharing the original naive clock are compared
directly with the supplied stale endpoint; different clock domains get null.

## Existing forward collector: useful but not certified

`gold_data_foundation_forward_collector.py` already contains M1/tick collection
and append operations. `data_foundation1_collector_config.json` references
`D:\XM2\terminal64.exe`; existing startup scripts reference terminal paths
and scheduled production startup. These references do not certify an active
raw-data collection schedule. None of those programs is executed here.

That collector uses a previous generation's cutoff, mutable state and an
overwritten manifest. Its UTC conversion depends on the empirical EET/EEST
helper in `gold_data_foundation1_audit.py`. It does not establish the new
protocol, a sealed per-snapshot previous-hash chain, or the twenty required
native higher timeframe captures. Its old cutoff/untouched declarations are
not adopted. The foundation assesses this design statically only.

## Freshness, timezone and prefix gates

`generated_at_utc` comes from the full audit's runtime clock. Operational
thresholds are frozen: CURRENT <=6h, RECENT >6h and <=72h, STALE >72h.
Unknown/naive/mixed clocks and future timestamps produce UNKNOWN with null
age. Explicit ISO offsets allow a declared-clock operational comparison;
they do not certify broker provenance. Existing collector rows carrying raw
broker epoch fields remain UNKNOWN because their transformation is unverified.
These thresholds say nothing about trading quality; weekends can be stale
operationally without proving missing bars.

Timezone is a required readiness gate. Static UTC logging or empirical
EET/EEST code is not authoritative evidence for every CSV. Broker epoch
meaning, DST transitions, bar-open/close labels, session rollover, weekends
and holidays require source-specific certification. V1 records static leads
and returns UNRESOLVED for this certification; no gap pattern can satisfy it.

The finite M1 feature warmup is 254 rows: ATR(14), rolling ATR mean(240),
global lag1. Each higher trend needs twenty genuine prior closes, its
publication row and subsequent M1 lag. Recursive MACD EMA12/26/9 with
`adjust=False` needs its complete original prefix or a separately predeclared
and certified initialization policy; a finite warmup cannot prove equality.
`prefix_available` records only minimum local context availability, while
`prefix_identity_attested` stays false. `context_start` and `holdout_start`
are separate. No context row becomes test evidence.

## Future append-only capture design

This is a design, not a live collector implementation or activation:

1. Persist original bytes for each sealed closed-bar snapshot; preserve raw
   spread and source identity. Never repair/overwrite a sealed snapshot.
2. Record a monotonically increasing manifest sequence, previous-manifest
   SHA256, raw SHA256, UTC capture time, first/last source timestamp, row count,
   schema fingerprint, duplicate/reversed counts, path and collector commit.
3. Hash canonical JSON metadata (sorted keys, compact UTF-8), link to the
   prior manifest, and anchor the chain tip independently. Sequence begins 0
   with a null predecessor.
4. Flush/fsync snapshots before committing their manifest. On restart recover
   from the last verified link, quarantine partial/orphan snapshots and
   deduplicate stable source/bar identities. Retain revisions as new evidence.
5. Restrict collector writes after sealing, retain raw data plus every manifest,
   and validate storage capacity, retention, recovery and timestamp authority.

Design feasibility booleans do not certify current ACLs, immutability,
running collection or durable storage. `writable_by_collector`,
`immutable_after_capture`, and `can_continue_forward` remain unknown without
operational evidence; no write probe is made. The pure manifest checker tests
schema, sequence, linkage and chronology on synthetic records, not actual
capture artifacts. Snapshot interval and exact manifest fields are in the spec.

## Readiness and future boundary

Readiness aggregates independent gates. One distinct blocker class gives its
named NOT_READY status; two or more give NOT_READY_MULTIPLE_BLOCKERS, with all
reasons retained. No blockers gives READY_TO_FREEZE_PROTOCOL only. Neither
status approves a strategy or production change.

This offline v1 cannot independently certify continuing live service,
authoritative timezone or operational capture integrity, so it reports those
blockers rather than accepting configuration assertions as proof. Structural
feature/execution availability is computed separately from exact certified
reconstruction. A later reviewed certification step is required to remove
these blockers; no CLI switch can bypass them.

`historical_start_allowed=false`, `historical_2025_plus_allowed=false`, and
`earliest_possible_future_start=null`. No historical cutoff or today's date
is selected. The boundary rule is:

> The first verified fresh GOLD M1 bar captured after the immutable protocol
> freeze and data-capture certification, with no outcome inspection before
> one-shot unlock.

Required certifications include protocol freeze commit, data source,
collector/hash-chain activation, timezone semantics and required raw pipeline.

Future workflow:

1. Implement this foundation.
2. User manually runs the foundation audit and reviews its result.
3. Resolve blockers; if READY_TO_FREEZE_PROTOCOL, write the immutable Future
   Locked Holdout Protocol.
4. Activate append-only capture and linked manifests.
5. The first verified post-freeze, post-certification bar defines the holdout.
6. Accumulate without outcome inspection.
7. One-shot unlock after the predeclared duration.
8. Only then evaluate promotion.

## Safety and implementation validation

The script uses only the standard library. Import guards block model/MT5 and
repository research modules; process audit hooks block sockets, filesystem
mutations other than the one new report, and subprocesses except direct
`git rev-parse`, `git status`, `git rev-list`, `git cat-file`. The repaired
Windows `(executable, args, cwd, env)` handling accepts list/tuple or simple
Windows command lines, including quoted git.exe paths, and rejects shell
chaining/ambiguous quotes/non-Git executables. No `shell=True` is used.
These are defense-in-depth guards, not a hostile-native-code sandbox.

The self-test uses in-memory fixtures and the local frozen spec only. It
tests freshness thresholds, missing timeframes/timezone, malformed raw rows,
spread absence, encodings/delimiters, gap ambiguity, historical exclusion,
null holdout boundary, manifest linkage, source classification, forbidden
imports/network/writes/subprocesses, Unicode paths, readiness precedence and
absence of strategy-performance fields in the report schema. Synthetic
`sys.audit` events test guards without launching processes or opening sockets.

Only these implementation checks are run:

```powershell
.\.venv\Scripts\python.exe -m py_compile .\gold_future_holdout_data_foundation_v1.py
.\.venv\Scripts\python.exe -B .\gold_future_holdout_data_foundation_v1.py --self-test
git diff --check
```

Static source review additionally checks forbidden active calls. No full audit,
live query, collection, research run, model operation, S5 replay, production
change, finalized-run change or existing-report overwrite is part of validation.
