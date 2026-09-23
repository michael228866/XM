# Conditional M1 resampling proposal — NOT APPROVED

Use only if native availability cannot be established. No resampling occurs in
enablement v1. `protocol_change_required=true` for adopting this proposal.

- Proposed intraday boundaries: UTC midnight plus integer timeframe multiples.
- Proposed Daily boundaries: 00:00 UTC; Weekly: Monday 00:00 UTC; Monthly: first
  calendar day 00:00 UTC. Intervals are left-closed, right-open.
- Broker/session calendar: require a versioned, authoritative XMGlobal-MT5 6
  GOLD# calendar of expected open M1 slots, holidays, rollover and DST changes.
  That calendar is currently unresolved; no inferred gaps may substitute for it.
- Only sealed M1 bars whose full interval has closed may contribute. Output is
  labelled with interval opening UTC timestamp, never local ambiguous wall time.
- OPEN=first OPEN, HIGH=max HIGH, LOW=min LOW, CLOSE=last CLOSE; sum tick and real
  volume separately. Proposed SPREAD=max constituent spread in points, explicitly
  different from any undocumented broker-native aggregation.
- Missing expected M1 slot, conflicting revision, nonfinite input or incomplete
  interval: quarantine the aggregate; no fill-forward, interpolation or partial
  bar publication. Document scheduled closure slots rather than manufacturing bars.
- UTC boundaries do not shift with DST. The required calendar controls expected
  sessions, but never moves these aggregate boundaries implicitly.
- Broker native daily/week/month boundaries, spread semantics and historical
  offline-file construction may differ. Numerical equivalence is NOT claimed.
- Causal availability time is after interval close plus successful input seal;
  native-versus-resampled feature alignment needs a separately frozen decision.

Approval, calendar binding and recursive-state initialization remain prerequisites.
This proposal changes the historical native-file pipeline and is not approved here.
