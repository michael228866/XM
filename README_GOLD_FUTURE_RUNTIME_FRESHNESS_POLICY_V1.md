# GOLD healthy-idle runtime migration v1

This is an operational migration, not a finding that the original pause was invalid.
The original `PAUSED_TIME_RULE.json`, quarantine and finalized adjudications remain immutable.
Only its exact recorded hash is superseded operationally. Every new pause marker remains binding.

The new collector checks identity and freshness before current offset/skew inference.
Stale observations produce HEALTHY_IDLE with no snapshot or sequence advance.
Fresh observations still pass the unchanged v4 sample and payload validators before sealing v4-compatible chain entries.
Fresh 7200, unexpected offsets, excessive skew, reversal and integrity failures stop capture.
There is no historical paused-period catch-up: admission starts at the migration's next UTC minute.
The original HOLDOUT_START and evaluation start remain unchanged.

The supervisor uses the existing Windows mutex/PID lock and refreshes heartbeat every ten seconds while idle.
The single existing Limited/Interactive Scheduled Task is updated only after a committed, independently validated migration receipt exists.
No model, strategy or training launcher imports are permitted in the runtime graph.
