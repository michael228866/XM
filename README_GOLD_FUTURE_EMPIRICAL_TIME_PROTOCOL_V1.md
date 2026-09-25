# GOLD empirical time and causal context certification v1

## Explicit v3 policy fork

EMPIRICALLY_CERTIFIED_SOURCE_TIME_RULE replaces mandatory official broker
authority for this fork only. v1/v2 files and finalized runs remain unchanged.
Official evidence is SUPPLEMENTARY_ONLY; holiday/session calendars are
EXPLANATORY_METADATA. Neither missing official attestation nor a missing annual
holiday calendar is a v3 readiness blocker. No broker-internal semantics or
official guarantee is claimed.

The exact target remains XMGlobal-MT5-6_GOLD / GOLD# / XM Global Limited /
XMGlobal-MT5 6 / demo, digits2, point0.01. Identity is checked from actual feed
metadata. Account IDs and financial fields are never persisted or printed.

## Empirical rule and scope

Derive the nearest whole-hour offset from raw tick epoch minus observation UTC:
floor(delta/3600+0.5)*3600. Only then compare with the frozen candidate allowlist
[7200,10800] and require absolute normalized error <=5 seconds. The allowlist is
not evidence that both offsets were observed. There is no DST recurrence table.

Collect six predeclared five-minute historical M1 windows (winter, summer,
March before/after, October before/after), at most six timestamps each, and
three current windows with two tick/system-UTC observations each. Only timestamp
fields are read. Historical rows without independent anchors are labeled
EMPIRICAL_INTERNAL_CONSISTENCY with offset=null, never EXTERNAL_UTC_ANCHORED.
Query timestamps do not establish the UTC meaning of returned epochs.

OS UTC is operational evidence. Windows Time query status is retained but does
not independently guarantee maximum skew. Without an independently checked
clock certificate, live observations cannot be the sole basis of certification.
Two observed offset regimes and anchored transition windows are required by this
version; unobserved gaps are excluded from bounded certified segments. No future
transition dates or unconditional/open-ended guarantee is inferred.

## Runtime fail-closed behavior

The v3 collector preserves v2 canonical source identity, native21 policy,
exclusive fsynced publication, full-chain verification and non-authoritative tip.
A v3 manifest adds the runtime validation hash. Immutable JSON snapshots contain
raw market fields, raw epochs, normalized timestamps, source identity and the
runtime anchor envelope; no strategy fields. v2 schema is not changed.

Every admission checks source identity, allowed stable offset, clock certificate,
raw/normalized monotonicity, native interval spacing, successor closure proxy and
anchors covering the entire admitted interval. Offset transitions require a new
regime certificate; ambiguous spans are excluded. A gap is not filled. Only
actual source bars are admitted. The closed-bar test requires a later native
source bar and a bounding observed regime; no holiday-derived expected bars.

On conflict: quarantine before chain publication, append an immutable
PAUSED_TIME_RULE marker and block further admission. Recovery requires
re-certification and a new reviewed capture root; no automatic adaptation or
deletion of sealed evidence. Only chain_tip.json is replaceable. Orphans stop
recovery rather than being silently adopted.

The acquisition path checks the actual terminal identity and uses a 130-second
bracketing observation window. Native bars spanning beyond observed coverage
fail closed, including longer HTFs until suitable coverage exists. Thus static
conformance and synthetic runtime PASS are not operational capture readiness.
No capture is started while empirical/context gates remain unresolved.

## Causal context reuse and binding

The completed inventory run 20260924T191927Z_gold_future_causal_context_inventory_v1
is reused by immutable artifact hashes. Its independent validator is not rerun.
The new validator independently parses timestamp columns again and rehashes
current external bytes. Named inventory/manifest outputs are retained in the new
run alongside requested causal_context aliases; large raw CSV files stay local.

M1=4096 and each HTF=21 remain fixed in the envelope
[2024-01-01T00:00:00Z,2026-09-24T00:00:00Z). Historical Monthly context is allowed;
no post-activation 21-month accumulation is required. Only MACD_HIST is recursive;
pipeline/function hashes and EMA/pandas/dtype/lag mechanics remain frozen.

An empirical source-time policy does not prove that an old CSV was exported from
that source or establish an unobserved continuation bridge. Structural counts
and naive-time selections remain conditional until those facts are supported.
approved=false, historically_equivalent=false, outcome_tuning=false,
context_role=CAUSAL_CONTEXT_ONLY, holdout_evidence=false remain explicit.

## Execution and archival

Commit/push source first; verify clean main; execute one formal run. Run its
archived independent validator once. Preserve all attempts and diagnostics,
finalize/register and validate provenance before the result commit/push.
Both production hashes are checked before/after major stages. No models are
loaded and no strategy outputs are inspected. If empirical/context evidence is
insufficient, finalize PARTIAL and stop before freeze/activation/boundary.
