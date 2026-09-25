# GOLD empirical time and context certification

Formal result: PARTIAL

Official broker attestation is no longer mandatory. Holiday calendar is explanatory metadata.

{
  "status": "PARTIAL",
  "policy": "EMPIRICALLY_SUPPORTED_BUT_INSUFFICIENT",
  "observed_offsets_seconds": [
    10800
  ],
  "allowed_offsets_seconds": [
    7200,
    10800
  ],
  "empirical_coverage_start": null,
  "empirical_coverage_end": null,
  "certified_segments": [],
  "unobserved_intervals_excluded": true,
  "official_attestation_required": false,
  "broker_internal_semantics_claimed": false,
  "runtime_fail_closed": true,
  "clock_quality": "OPERATIONAL_CHECK_ONLY",
  "blockers": [
    "Historical seasonal/transition samples lack independent offset anchors",
    "No certified temporal segment coverage; do not interpolate or invent recurrence",
    "System UTC is an operational anchor only"
  ]
}

Runtime PASS means synthetic implementation checks only. Capture was not activated.

Historical context inventory retained by exact bytes and external hashes; 21 structurally sufficient candidate timeframes, native file origin unconfirmed. Causal context only, never holdout evidence.

No model/strategy execution or production change. Prefix unapproved. No freeze, activation or boundary.

## Independent validation and scope

Independent validator PASS; failed_check_names=[]; executed once. Methodology PASS. 70 synthetic checks passed.

Actual source identity verified: XMGlobal-MT5-6_GOLD / GOLD# / XM Global Limited / XMGlobal-MT5 6 / demo. Six historical windows returned 36 timestamp rows; three current windows returned 9 rows plus six tick/OS-UTC observations. Current offset inferred as 10800 seconds; errors range from -0.635145902633667 to -0.6333940029144287 seconds. No 7200-second regime was directly observed.

Windows Time query completed successfully, but its service status does not independently guarantee clock skew. The OS clock remains one operational check. No certified historical/future coverage is claimed.

21 context timeframe files have sufficient structural candidate counts (M1=4096, each HTF=21). Native file origin confirmed=0; no claim that timeframe files are absent. Original naive-clock selection and external identity hashes remain unchanged. No new resampling or outcome tuning.

Runtime implementation PASS is synthetic only. Operational long-native-bar coverage remains unproven; the v3 collector rejects bars outside its observed anchor span. No official-authority requirement remains in v3 readiness gates.
