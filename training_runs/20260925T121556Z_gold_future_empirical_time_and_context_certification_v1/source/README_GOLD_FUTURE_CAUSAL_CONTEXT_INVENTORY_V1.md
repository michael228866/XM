# GOLD future causal context inventory v1

This is a structural inventory, not a feature, model or strategy run. The four
inventory/manifest/compatibility/binding JSON files are generated exclusively
inside the unique formal run directory. Historical files are never rewritten.

## Frozen scope

Target: XMGlobal-MT5-6_GOLD, GOLD#, XM Global Limited, XMGlobal-MT5 6, demo.
The target identity does not certify the origin of a historical file.
Context envelope: [2024-01-01T00:00:00Z, 2026-09-24T00:00:00Z).
M1 requirement: 4096 bars; every native higher timeframe: 21 bars.
All 21 names and NATIVE_20TF_REQUIRED remain fixed; no resampling or tuning.
Context is CAUSAL_CONTEXT_ONLY, historically_equivalent=false,
outcome_tuning=false, holdout_evidence=false.

## Discovery and retention

Search the approved D:\XM tree for GOLD/XAUUSD-named CSV, compressed/array,
Parquet and pickle candidates. Paths are preregistered before the formal run.
Exclude Git/environment/tool/private directories, prior finalized research
archives and untouched_forward. Prior research archives are preserved evidence,
not substitute raw context; untouched capture is outside this historical task.
Symlinks are not followed. The exact exclusions and candidates are in the spec.

Read CSV headers first. Strategy schemas and model artifacts receive only byte
hashes, sizes and classification; their bodies are not interpreted/deserialized.
Unrecognized/headerless data is UNKNOWN_PROVENANCE. Only raw-schema candidates
have DATE/TIME columns parsed. Price, spread and volume values are not computed.
No MT5 connection or new data download occurs.

Existing loader references plus an MT5-style header justify compatibility
review, not CANONICAL_NATIVE_SOURCE. Without a file-bound export certificate,
native_source_confirmed=false. resampled=false means no derived flag was found;
it does not establish native origin. Such files remain POTENTIALLY_COMPATIBLE.

The file manifest hashes every candidate, including rejected candidates, and
records external retention honestly. The data bytes remain local and mutable.
Any subsequent byte change invalidates the manifest identity. No raw CSV enters
Git. Hashing model bytes is identity inspection, never loading a model.

## Timestamp and coverage semantics

Naive text remains naive. No legacy timezone conversion is applied. Envelope
comparisons are conditional wall-clock candidate selection, not certified UTC.
Select the latest required rows whose successor is inside the fixed envelope;
bind zero-based source row indexes, timestamps and successor timestamps.
Exclude the final source row. A successor is a conservative closure proxy,
conditional on BAR_OPEN and source/calendar authority; not closure proof.
certified_usable_causal_bars remains zero while those gates are unresolved.

M1_CONTEXT_BARS reports selected structural candidates, not all source rows.
Available-preboundary counts, all-source counts and certified usable counts are
separate. A satisfied warmup count does not imply approved causal context.
Historical Monthly bars are allowed; no post-activation 21-month wait is imposed.
The fixed source envelope is not an assertion that data cover its entire range.
A gap between file end and future activation needs a causal continuation bridge;
no state reset, resampling or imputation is silently permitted.

Continuity uses timestamp order, duplicates, positive spacing and nominal slot
counts. Monthly uses calendar-month differences. All gap interpretations remain
UNKNOWN without session/holiday authority; nominal missing slots are not proof
of corrupt or missing tradable data. OHLC validity is not certified by this audit.

## Binding and decisions

The prior validated MACD_HIST recursive dependency and native rolling20/lag1
requirements are bound to the frozen proposal and implementation hashes. The
implementation is parsed as source, never imported or executed. No new features
are computed and no warmup search occurs.

Root hash: sort manifest entries by timeframe and slash-normalized case-folded
absolute path; canonical UTF-8 JSON (sorted keys, compact separators) per entry;
SHA256 of the concatenated lowercase hex entry SHA256 values. File-manifest hash
is SHA256 of its exact stored bytes. Source/metadata/root hashes are distinct.

PARTIAL_TIMEZONE_PENDING is used when a useful structural inventory cannot
establish native origin/source/timezone semantics. It may include provenance
blockers beyond timezone. Missing/insufficient or incompatible required context
is FAIL. PASS_SOURCE_COMPATIBLE is never inferred from names or row counts.
approved=false and ready_after_timezone=false while provenance or continuation
also remain unresolved. No final protocol freeze, activation or boundary.

## Workflow

Run synthetic tests and compile source; commit/push clean main before --execute.
The runner snapshots source and generates the unique formal archive. Run its
archived independent validator once. It independently parses timestamps using
stdlib CSV/datetime, checks current raw bytes and hashes, and never imports the
builder. Negative research results can still pass truthful archive validation.
Finalize/register with training_run_history.py; validate provenance; commit/push
only finalized artifacts plus TRAINING_RUNS.md. Preserve the attempt marker.
Check protected production hashes before/after each major stage.
