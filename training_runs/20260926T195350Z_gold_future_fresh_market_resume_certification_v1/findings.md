# GOLD fresh market resume certification

Formal result: PARTIAL

{
  "classification": "NO_FRESH_MARKET_DATA",
  "current_regime_status": "PARTIAL",
  "source_identity_status": "PASS",
  "fresh_source_data_available": false,
  "fresh_tick_sample_count": 0,
  "new_closed_m1_count": 0,
  "observed_offset_seconds": null,
  "normalized_clock_error_seconds_min": null,
  "normalized_clock_error_seconds_max": null,
  "latest_source_observation_age_seconds": 82560.89123702049,
  "evidence_admitted": false,
  "resume_authorized": false,
  "fresh_tick_max_age_seconds": 120,
  "fresh_closed_m1_max_age_seconds": 180,
  "max_clock_skew_seconds": 5
}

Fixed freshness screens: tick 120 seconds, closed M1 180 seconds under existing 10800. Observed feed progression is separate fresh evidence so live 7200 cannot hide as aged 10800 data. Offset and skew are evaluated only after freshness; stale rows carry null offset/skew. Five initial observations two seconds apart; stale batches stop. Fresh batches wait at most 130 seconds total for two closures.

ROOT_CAUSE=NOT_PROVEN. No runtime amendment or resume occurred in this certification phase. No healthy-idle claim. Original pause, quarantine, pending, context, boundary, chain and frozen runtime remain unchanged. No bars appended or backfilled. Feature eligibility not rerun without new sealed input. No model/strategy/outcome evaluation or production change.

Independent validator: PASS; failed_check_names=[]. No fresh samples, no current offset/skew inference, no resume.
