# GOLD future capture protocol adjudication v1

v2 is an explicit protocol fork. v1 code, schema, templates, collectors and all
previous finalized runs remain unchanged. No MT5 call, model or strategy execution
is needed: retained diagnostics establish demo identity and structural native
availability. They do not certify annual timezone/calendar or recursive prefix.

## Decisions

- Machine source ID: XMGlobal-MT5-6_GOLD. Exact broker symbol: GOLD#.
  Migration is identifier representation only; broker/server/instrument unchanged.
- Evidence snapshots, manifests, attestations and protocol freeze are immutable.
- chain_tip.json has exactly three fields and is a non-authoritative index.
  Its bytes are derived only from a fresh full chain recomputation. Atomic replace
  is confined to that exact destination in one reviewed publisher function.
- Missing tip: chain valid with warning. Stale verified predecessor tip: warning.
  Conflicting tip: chain integrity can still pass, but readiness fails until explicit
  rebuild. Rebuild does not rewrite evidence; old tip can be retained in quarantine.
- Broker intervals are UTC start-inclusive/end-exclusive, contiguous and ordered,
  individually evidence-bound. Ambiguous local epochs or coverage gaps fail.
  No future offset recurrence is inferred from one diagnostic or a display clock.
- Native M1 plus 20 higher timeframes remain mandatory; no resampling fallback.
- Prefix exact history and approved causal reinitialization are distinct policies.
  Neither historical equivalence nor approval is claimed for the current proposal.

## Collector changes

collector_v2 is an explicit compatibility copy of v1: v2 versions and identity,
broker-rule/UTC/fixed/IANA epoch conversion, evidence-versus-index recovery, and
verified tip publication. Sealed CSV and manifest publication continue to use
exclusive create, flush/fsync and no-replace hard links. Revisions never mutate old
snapshots. The v1 canonical raw CSV schema hash is intentionally retained: columns
are unchanged. Only manifest protocol version changes.

The static certifier uses both path/purpose checks and an approved whole-source AST
identity, pinned in the committed v2 protocol. This is an allowlist of one reviewed,
tested implementation, not a claim to prove arbitrary Python safe. New collector
code needs new review, mutation tests and a new immutable approval identity.
Only publish_tip may call os.replace, only targeting file_under(root,
"chain_tip.json"), after recover_chain, exclusive temp creation and fsync. Alias,
dynamic destination, extra mutation or any source drift fails closed.

Recovery is read-only by default. Explicit append/recovery may retain quarantine
copies and receipts. Invalid manifests/orphan snapshots block; no automatic
adoption. A stale writer lock is never stolen. Directory durability after abrupt
power loss is not claimed; restart verifies all published evidence from sequence 0.

## Execution and archive

```powershell
.\.venv\Scripts\python.exe -B gold_future_capture_certification_v2.py --self-test
.\.venv\Scripts\python.exe -B validate_gold_future_capture_protocol_adjudication_v1.py --self-test
# Commit, push and verify clean source before this authorized formal command:
.\.venv\Scripts\python.exe -B run_gold_future_capture_protocol_adjudication_v1.py --execute
```

The run uses the repository create/finalize/register/validate lifecycle. PARTIAL
maps to archival status research_only, preserving the explicit PARTIAL verdict.
Its independent validator imports neither collector nor certifier. It independently
recomputes retained synthetic chains, hashes and index statuses. No live capture
root, strategy outcome, protocol freeze or holdout start is created while gates fail.

Source and result commits stay separate. Result commit identity is the Git commit
containing FINALIZED.json, avoiding a self-referential manifest hash. Source snapshots
retain exact bytes, including reviewed evidence and prior diagnostic copies.
