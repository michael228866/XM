# GOLD feature eligibility and continuous capture

Formal result: PARTIAL

{
  "formal_run_status": "PARTIAL",
  "candidate_bars_checked": 1,
  "feature_complete_candidate_count": 0,
  "expected_feature_count": 31,
  "missing_feature_count": 31,
  "nan_count": 0,
  "inf_count": 0,
  "context_hash_match": true,
  "pipeline_hash_match": true,
  "scheduled_task_installed": true,
  "supervisor_started": true,
  "heartbeat_pass": true,
  "chain_continuity_pass": true,
  "continuous_capture_status": "PAUSED",
  "validator_status": "PENDING"
}

Frozen seed-to-boundary M1 inputs and closed native HTF inputs are not fabricated or refetched. Evaluation start remains null pending independent eligibility. Supervisor honors all frozen collector pause markers, including failures the frozen collector treats as permanent. No model, strategy, outcome or production change.

Independent validator PASS; failed_check_names=[]. Formal result remains PARTIAL. M1 inputs 19:21, 19:22 and 19:23 UTC between frozen seed and boundary are unsealed. Known closed M2/M3/M4/M6/M12 inputs are also absent. No complete feature vector was computed; 31 features remain uncertified, NaN/Inf counts zero mean not computed, not proven finite.

The user-level scheduled task is installed and its principal was verified by SID equivalence. It was reused after the separately preserved aborted alias-comparison attempt. The supervisor started and the frozen collector created PAUSED_TIME_RULE with reason PAUSE_AND_QUARANTINE. Supervisor exited without automatic resume. No new evidence was admitted; last sequence 0, timestamp 2026-09-25T19:24:00+00:00. Current runtime time validation FAIL, existing chain PASS. No claim that continuous capture is running. Separate time-rule adjudication is required; no weekend exemption or policy amendment was introduced.
