# GOLD future capture enablement v1

Infrastructure research only. No model import, training, prediction or strategy
performance evaluation. Old collector, production files and finalized runs remain
unchanged. The formal runner requires clean, pushed main before creating a run.

## Commands

Run synthetic checks before committing:

```powershell
.\.venv\Scripts\python.exe -B test_gold_future_capture_collector_v1.py
.\.venv\Scripts\python.exe -B gold_mt5_timestamp_semantics_diagnostic_v1.py --self-test
.\.venv\Scripts\python.exe -B gold_mt5_native_timeframe_audit_v1.py --self-test
.\.venv\Scripts\python.exe -B gold_recursive_prefix_certification_v1.py --self-test
.\.venv\Scripts\python.exe -B validate_gold_future_capture_enablement_v1.py --self-test
```

After source commit/push, the explicitly authorized formal command is:

```powershell
.\.venv\Scripts\python.exe -B run_gold_future_capture_enablement_v1.py --execute
```

It creates one unique run, snapshots committed inputs, executes bounded read-only
diagnostics and offline certification, and stores stdout/stderr for each command.
Run the archived validator once, then use training_run_history.py finalize
--status fail or research_only as appropriate --register and validate. Never retry
an already attempted validator or modify finalized evidence.

## Diagnostic boundaries

Only GOLD# on the specified XM terminal/server/company is accepted. One current
tick, five recent M1 bars, one five-minute range and three closed bars for each of
21 native timeframes. Returned price values are never saved or analyzed. MT5
account information exists briefly in memory solely to extract company/server and
documented environment classification; no full structure or sensitive values are
printed. Exceptions persist type only. A failed query is UNRESOLVED.

The legacy European last-Sunday EET/EEST transform is a comparison hypothesis,
not a timezone certificate. Source-specific daily rollover, holiday authority,
weekend hours, DST and API-path consistency remain separate evidence gates.
Three-bar availability does not prove continuity or an exact recursive prefix.

## Storage and recovery review

Canonical raw CSV only, no gzip. Snapshot and manifest bytes are exclusively
created and fsynced in pending/, checked, then atomically hard-linked into their
final names on the same filesystem. No existing snapshot/manifest is replaced.
Manifest publication follows snapshot fsync/seal. Full chain, hashes, schema,
source, symbol, timeframe, code and attestation identities are verified before
each append. Revisions preserve original files and reference their exact interval.
Identical duplicate or reversed bars are rejected and copied to quarantine.

writer.lock is an exclusive directory. Concurrent writers fail. A process crash
may leave a stale lock: never steal it automatically; an operator must establish
that no writer remains before removing this disposable lock. Power-loss directory
durability is not claimed; restart verification detects missing publication.

Recovery cases:

- Snapshot write/fsync crash: pending bytes copied into quarantine with receipt;
  never automatically promoted. No accepted chain changes.
- Sealed snapshot without manifest: preserve original and quarantine copy; fail
  closed and require operator adjudication/new reviewed root.
- Manifest write crash: pending manifest is not accepted; orphan snapshot above
  still blocks. Invalid/orphan manifest is quarantined and blocks the root.
- Manifest sealed before tip update: full chain verification first; explicit
  `recover_chain(..., allow_stale_tip=True)` returns the validated expected index.
  Under writer_lock an operator may call publish_tip with that exact object.
  This rebuilds the index only and is never automatic during normal append.
- Pending copies matching accepted hashes remain harmless retained hard links;
  unmatched pending bytes get quarantine copies and receipts. Nothing is deleted.
- Changed source, symbol, timezone or activation rejects linkage. Changed code
  commit/version requires recertification and a separately attested chain.
- Broker revision: explicit record_revision/append_snapshot revision_of path,
  same timeframe/bar interval/count, differing bytes; old evidence retained.

## Known frozen-protocol incompatibilities

1. The verifier requires current chain_tip.json. Advancing this mutable,
   rebuildable index uses atomic os.replace; the unchanged static checker rejects
   every replace call without distinguishing index from sealed evidence. This is
   reported as NOT_STATICALLY_CONFORMANT, not bypassed with reflection or hidden
   dependency code. Full chain verification itself passes synthetic compatibility.
2. Existing source_id XMGlobal-MT5-6_GOLD# contains #; the frozen source identity
   and manifest regex exclude #. No silent rename or schema relaxation occurs.
3. Source-specific timezone/calendar and exact prefix remain uncertified until
   stronger evidence is provided. No historical rows become untouched evidence.

The execution spec explicitly disables activation. Collector static certification
failure is a major gate and yields formal FAIL even when synthetic storage tests
pass. It is not a strategy research failure or evidence of performance. No protocol
freeze or live capture is attempted. Mutable tip policy needs separate explicit
protocol adjudication; source implementation does not change the certifier.

The independent validator imports no execution module. It independently checks
source snapshots against Git, timing arithmetic, native-timeframe completeness,
prefix limitations, attestation links, exact static gate failures, report readiness
and production hashes. Validator PASS means honest provenance, not enablement PASS.
