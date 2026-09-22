# GOLD future capture certification v1

This is offline certification infrastructure between:

**future data foundation → capture certification → protocol freeze → activation
→ future untouched accumulation → one-shot unlock**.

Historical 2025+ GOLD remains contaminated for promotion-quality use. Copying,
renaming, hashing, sealing or certifying old files does not make them untouched.
The historical canonical endpoint remains `2026-05-08T23:57:00`. The additional
M1-looking source around September 2026 needs its own resolved provenance;
neither source is automatically selected or relabeled.

Implementation does not activate a collector, create a capture directory,
start a holdout, contact MT5/brokers, evaluate a strategy, load/train a model,
generate predictions/labels, replay S5, or change production. Source, timezone,
capture, native-timeframe and recursive-prefix policies must all be certified
first. The user runs certification manually after source review.

## Files

- `gold_future_capture_certification_v1.py`: attestation checks, collector AST
  inspection, integrated read-only chain verifier and synthetic self-test.
- `execution_spec_gold_future_capture_certification_v1.json`: frozen rules.
- `gold_future_capture_manifest_schema_v1.json`: exact flat JSON Schema.
- `gold_future_capture_source_attestation_template_v1.json`.
- `gold_future_capture_timezone_attestation_template_v1.json`.
- `gold_future_capture_activation_template_v1.json`.
- This README.

There is no separate verifier process or dynamically imported collector/helper.
No new dependency is required. Canonical hashes pin all five JSON release files
in the Python source. Whitespace/BOM/Windows newline changes do not change their
semantic identity; changing values requires a reviewed source/spec release.

## Later manual certification

Example, **not executed during implementation**:

```powershell
.\.venv\Scripts\python.exe -B .\gold_future_capture_certification_v1.py `
  --output .\gold_future_capture_certification_report.json `
  --repo-root "D:\XM\數據" `
  --collector "D:\XM\數據\proposed_collector.py" `
  --source-attestation "D:\chosen_capture\source_attestation.json" `
  --timezone-attestation "D:\chosen_capture\timezone_attestation.json" `
  --activation "D:\chosen_capture\activation.json" `
  --capture-root "D:\chosen_capture"
```

Only a new, explicit JSON report is written. Its parent must exist. Existing
outputs, UNC paths, capture-root descendants, and `training_runs`, `models`,
`.git`, `.venv`, `future_holdout_capture` output directories are rejected.
The capture root declared in an activation document is also protected, even
if `--capture-root` was omitted. No directory, attestation or capture artifact
is created automatically. Missing inputs produce NOT_READY/NOT_CERTIFIED.

Attestation paths default to the three conceptual filenames under an explicit
capture root. Repo defaults to this script's directory; collector has no
automatic default. A NOT_READY report is a successfully completed inspection
(exit 0). Unsafe output or unrecoverable execution errors return
`CERTIFICATION_ERROR` (exit 2). Error messages do not echo supplied values or
collector literal contents. An interrupted report write can leave a partial
new JSON; it must never be overwritten automatically.

## Source attestation

Templates have `template=true`, unresolved values, and cannot pass. A later
user-created completed attestation must retain the exact field set, set
`template=false`, and resolve required fields. No completed attestation is
created by this implementation.

Source binding includes broker/server, one source ID, exact case-sensitive
GOLD/XAUUSD symbol, feed/terminal reference, transport, BAR_M1 or TICK data type,
source timestamp field and BAR_OPEN/BAR_CLOSE semantics, clock/DST/calendar,
SPREAD field/POINTS units, precision/point size, optional volume field names,
reviewer, UTC review time and documentary evidence. Unknown demo/live
environment fails; this enum contains no account identifier. Native bar
certification cannot silently derive M1 from a TICK binding.

Secret-bearing keys (password/passwd/secret/token/api_key/login/account/
credential patterns) are rejected recursively. The sole attestation exception
is `source_account_environment`, restricted to the declared enum; completed
source certification allows only demo/live. Never include credentials in
free text either. Collector literal screening emits line numbers, not values.
Reports copy only selected binding fields, not notes or document bodies.

Every evidence item has the exact shape documented by
`evidence_object_shape` in the spec: authority, local_path, sha256, reviewed_by,
reviewed_at_utc. Supported authorities are BROKER_DOCUMENT, FEED_DOCUMENT,
SIGNED_ATTESTATION and INDEPENDENT_REVIEW. Local document bytes are hashed,
not executed or fetched. Relative evidence paths resolve beside their supplied
attestation. Observed gaps are not an accepted authority. Review timestamps
must be UTC and not future-dated.

These checks bind documents and reviewers' assertions; they cannot establish
the truth of a dishonest attestation, independently authenticate a broker
without contacting it, or verify a signature merely because an evidence item
is called SIGNED_ATTESTATION. Independent human review and externally anchored
identities remain part of the certification process. No signing is implemented.

## Timezone certification

Timezone attestation binds the canonical source-attestation hash and source ID.
Source and timezone documents must agree on timezone, DST, bar labeling,
rollover, weekend and holiday authority. It explicitly records historical bar
clock meaning, original timestamp encoding and MT5 epoch interpretation.

Supported states:

| State | Required interpretation |
|---|---|
| CERTIFIED_UTC | UTC, offset 0, DST NONE |
| CERTIFIED_FIXED_OFFSET | Integer offset -840..840 minutes, DST NONE |
| CERTIFIED_IANA_WITH_DST | IANA name, DST evidence, exact installed TZif SHA256 |
| CERTIFIED_BROKER_SERVER_RULE | Reviewed transition rule and contiguous UTC intervals with offset_minutes |
| UNRESOLVED | Certification fails |

IANA `iana_tzdb_identity` is the SHA256 of the actual local TZif file, found
under Python's `zoneinfo.TZPATH`, not a guessed version name. The tool uses that
verified file. If unavailable (including Windows without a local database),
certification fails closed; nothing is downloaded or installed.

Broker interval objects contain exactly start_utc, end_utc and offset_minutes.
They are start-inclusive/end-exclusive and must be continuous, ordered and
nonoverlapping. Source times outside the reviewed table fail. Ambiguous DST
folds and nonexistent local times fail unless the original timestamp carries
a matching explicit offset. UTC/fixed-offset originals must match that rule.

Original timestamps use ISO8601 or EPOCH_SECONDS. Epochs explicitly distinguish
ALREADY_UTC from LOCAL_EPOCH_REQUIRES_RULE; NOT_APPLICABLE is permitted only
when an epoch interpretation is unnecessary. The existing empirical EET/EEST
helper is evidence only. The certification does not import it, infer a timezone
from gaps, or certify a server based on a UTC-looking column label.

## Native timeframes, resampling and recursive prefix

No policy is selected automatically. Activation defaults to UNRESOLVED.

`NATIVE_20TF_REQUIRED` must contain exactly one record for M1 and each of Daily,
H12, H8, H6, H4, H3, H2, H1, M30, M20, M15, M12, M10, M6, M5, M4, M3, M2,
Weekly and Monthly. Each record has exactly timeframe, source_id, symbol,
source_mode=NATIVE, timestamp_semantics, schema_sha256, freshness_policy,
continuity_policy and evidence. Identities, semantics and canonical raw schema
must bind to the same source. Documentary review must address native origin,
freshness, continuity and the calendar for every timeframe. This is policy
certification, not proof that a feed is currently live.

`M1_DETERMINISTIC_RESAMPLE_PROPOSED` requires
`protocol_change_required=true` and a proposal with exact boundaries,
closed_bar_rule, calendar, missing_bar_rule and
immutable_approval_required=true. It returns PROTOCOL_CHANGE_REQUIRED and
blocks readiness. It never becomes equivalent to the historical native pipeline
through an override. A later immutable protocol approval is separate work.

Prefix policies are EXACT_CONTINUOUS_CERTIFIED_PREFIX,
REINITIALIZATION_PROTOCOL_CHANGE_REQUIRED or UNRESOLVED. Exact prefix requires
a reviewed evidence document binding the complete continuous prefix/state and
explicit context bounds no later than protocol freeze. A finite warmup alone
does not prove recursive EMA/MACD equivalence. Reinitialization remains a
protocol change and blocks this release's readiness. Context is never holdout
evidence; this tool computes no indicators.

## Collector certification

Collector Python text is parsed with AST, never executed or imported. Its exact
byte SHA256 binds the independent review; its source must also match the
specified Git commit in the selected checkout (CRLF/LF normalization only for
Git text comparison). Actual byte identity remains separately recorded.

The AST screen rejects mutable modes, unknown open modes, fixed-file
write_text/write_bytes, replacement/rename, deletion/truncation, low-level
os.open, strategy/model imports/calls, dynamic execution and detectable
credential-like literals. It requires exclusive creation, fsync, no-replace
`os.link` publication and defined verification/recovery/identity/revision
interfaces. This deliberately conservative release can reject a safe but
unrecognized implementation; it never runs it to find out.

An independent review bound to both collector SHA256 and commit must also
resolve every guarantee and recovery case in the spec. Review must cover
control flow, real implementations, dependencies, source/timezone mappings and
storage semantics; named operations or function names alone are not proof.
The synthetic AST fixture is intentionally not a deployable collector.

The review documents all nine cases: partial snapshot crash, orphan after
fsync, partial manifest crash, full-chain restart verification, stable duplicate
identity, retained broker revision, clock reversal quarantine, new source
attestation on symbol change, and collector-change recertification.

`STATICALLY_CONFORMANT` / CERTIFIED_FOR_ACTIVATION means that this static screen
and bound document review passed. It is not OPERATIONALLY_CERTIFIED. A later
manual run may report a pre-existing ACTIVE_UNVERIFIED activation, but this
tool does not promote it to operational certification. It makes no ACL,
durability, service-health or actual-broker claim from source text alone.

## Capture layout and raw snapshot format

Conceptual layout only; implementation creates none of it:

```text
<chosen_capture_root>/
  source_attestation.json
  timezone_attestation.json
  activation.json
  snapshots/000000000000_<timestamp>_<sha>.csv.gz
  manifests/000000000000.json
  chain_tip.json
```

V1 verifies the predeclared `gold_canonical_bars_v1` CSV format: UTF-8, comma,
exact ordered columns SOURCE_TIMESTAMP, ORIGINAL_SOURCE_TIMESTAMP, SOURCE_ID,
SYMBOL, TIMEFRAME, OPEN, HIGH, LOW, CLOSE, SPREAD, followed by any attested
TICK_VOLUME/REAL_VOLUME/VOLUME fields in their declared order. Extra fields
are rejected before body interpretation. Result/model artifacts are never
deserialized. Volume values are retained raw; no strategy computation occurs.

SOURCE_TIMESTAMP is normalized UTC ISO8601; ORIGINAL_SOURCE_TIMESTAMP preserves
the original attested time representation. Their conversion is independently
checked against the timezone document. Every row repeats the exact certified
source ID, symbol and timeframe. Raw prices and spread are checked only for
schema/finite values and integrity counters. Nonpositive finite spread is
recorded without calculating economics; missing/nonfinite spread fails quality
certification. No source data is repaired or rewritten.

Schema SHA256 hashes the canonical JSON object containing format, encoding,
delimiter and ordered columns. Raw SHA256 hashes actual stored bytes, including
compression for `.csv.gz`; equivalent decompressed rows do not substitute for
the original bytes. Original unconverted broker-export formats need a reviewed
predeclared adapter; v1 never guesses one. This is a capture-format decision,
not authorization to change native feature semantics.

The manifest embeds all sealed-snapshot metadata, including capture times,
quality counters, partial_snapshot=false and sealed=true. No separate mutable
snapshot metadata file is trusted. Row counts, bounds, schema and counters
are recomputed by streaming the raw snapshot. Duplicate detection retains
timestamp identities in memory; memory grows with snapshot size.

## Chain verifier and recovery

The integrated verifier enforces the release-pinned manifest schema's exact
keys, types, SHA formats, timestamp offsets and constants, plus its relational
rules. Sequence starts at 0; only genesis has a null predecessor. Snapshot and
manifest sequence agree. Missing/duplicate sequence files, duplicate snapshot
IDs/paths, clock reversals and path escapes fail.

Canonical manifest and attestation hashing is UTF-8 JSON, ensure_ascii=false,
sort_keys=true, separators=(",", ":"), allow_nan=false. Duplicate JSON keys and
NaN/Infinity are rejected. Each successor contains its predecessor's canonical
hash. Source/timezone/activation hashes must match the supplied documents.
Collector identity is constant within this chain; a code change requires
recertification. Capture start follows prior finish; finish precedes manifest
publication. Source timestamps cannot be after capture finish.

Snapshot IDs and paths are never reused. Ordinary snapshots must advance source
time within their timeframe without overlap. A revision names an earlier sealed
snapshot with the same timeframe, full interval and row count; old evidence is
retained. A different revision shape requires separately reviewed semantics.

Every snapshot hash, schema and raw integrity counter is recomputed. Partial,
unlinked and orphan artifacts require quarantine or explicit recovery; this
verifier never repairs, moves, fetches or deletes them. Snapshot stat changes
during inspection fail. Manifest filenames are twelve-digit sequence JSONs;
the scan limit is 100,000 manifests and 1 MiB per JSON/source file. Limit hits
fail closed.

`chain_tip` contains exactly manifest_sequence, manifest_sha256, updated_at_utc.
The full chain is recomputed before comparing the tip. A locally self-consistent
chain alone cannot prove an adversary did not rewrite everything: anchor tips
in a separate Git commit, read-only storage, signed record or independently
controlled location. No signatures are implemented here.

For later pre-activation readiness, the user must prepare a local root with
the completed bound documents and empty manifests/snapshots directories, no
tip and no orphan data. This tool does not prepare it. Empty ACTIVE chains fail.
Nonempty chains require an explicit effective ACTIVE_UNVERIFIED artifact;
capture must not predate activation. Pre-boundary source rows may remain context
but never become eligible merely by their presence in a sealed snapshot.

## Activation, readiness and holdout boundary

The supplied template is NOT_ACTIVATED with null freeze/effective times and
production_promotion=false. Completed pre-activation states can be NOT_ACTIVATED
or ACTIVATION_ARTIFACT_READY with null effective time. A later user-created
ACTIVE_UNVERIFIED artifact needs an immutable freeze commit and an effective
time after source/timezone certification and activation request. Freeze and
collector commits must exist locally; the tool never fetches them.

One blocker class produces its named NOT_READY status; multiple classes give
NOT_READY_MULTIPLE_BLOCKERS and retain every reason. READY_FOR_CAPTURE_ACTIVATION
requires completed source/timezone documents, native/prefix policy, a reviewed
statically conformant collector, valid prepared layout/schema/activation, and
bound protocol freeze. It does not mean capture is running, operational storage
is certified, holdout has begun, strategy passed or promotion is approved.

The report always leaves holdout_start and earliest_possible_holdout_start null.
It never calculates a historical or actual first eligible timestamp. Frozen rule:

> The untouched holdout begins at the first verified fresh GOLD M1 bar whose
> source timestamp is strictly after both the immutable protocol freeze and the
> effective certified capture activation, provided source, timezone, pipeline,
> and manifest-chain certifications remain valid. Pre-boundary rows may be used
> only as causal warmup/context and never as holdout evidence.

At a later activation stage, eligibility requires source time strictly greater
than max(freeze effective time, activation effective time), all certifications
valid, certified source membership and sealing before any outcome inspection.
Only that later stage may identify the first eligible bar.

After activation and before the predeclared one-shot unlock, operators must not
inspect S4 decisions, B0/secondary probabilities, entries/exits, WR, PF, PnL,
Mean-R, drawdown, trades/day or candidate comparisons. Permitted monitoring is
collector/process alive/dead, data freshness, raw row/schema/duplicate/order
checks, continuity, manifests, hashes and storage health. No outcome-derived
monitoring is allowed.

## Safety and implementation checks

Import guards block model libraries, MT5, HTTP clients and repository execution
modules. Process audit hooks block sockets and filesystem mutation except the
one new report. Subprocesses are restricted to Git rev-parse/status/rev-list/
cat-file with the repaired Windows audit-event handling. No shell=True,
terminal/service/task creation, network access or dynamic collector import.
CPython's pathlib preloads the inert urllib.parse utility; that existing parser
is permitted, while urllib network clients and socket operations remain blocked.
These are defense-in-depth safeguards, not a sandbox for hostile native code.

The self-test uses synthetic documents, in-memory raw bytes, inert collector
AST text and synthetic audit events. It does not create completed attestations
on disk, capture directories, snapshots or manifests. It covers valid/rejected
attestations, secret keys, timezone/label policies, missing TF/prefix, protocol
changes, activation exclusion, manifest/schema/hash/counter/revision/tip failures,
collector risks, static-vs-operational status, no historical holdout, Unicode,
output protection, readiness branches and forbidden report fields/execution.

Only these implementation validation commands are run:

```powershell
.\.venv\Scripts\python.exe -m py_compile .\gold_future_capture_certification_v1.py
.\.venv\Scripts\python.exe -B .\gold_future_capture_certification_v1.py --self-test
git diff --check
```

Static review also checks forbidden active calls. Full certification, collectors,
broker sessions, live capture, strategy evaluation and production changes are
outside implementation validation.
