# Frozen feature eligibility and continuous raw capture

The feature checker reads only frozen context and a fixed prefix of the sealed
future chain. It binds the exact 31-feature ordered list, pipeline bytes and
pre-label feature AST. It never imports the pipeline module, which also contains
model and forward-label code. Only the verified feature statements may execute.
No model, classification threshold, signal, trade or outcome is evaluated.

Missing seed-to-boundary M1 rows block recursive state certification. Known native
HTF bars that have closed but are absent from sealed inputs also block eligibility.
No context refresh, interpolation or fabricated continuity is permitted. Historical
context remains causal context only. The original boundary is immutable; a separate
evaluation-start artifact records a verified timestamp or null.

The supervisor calls the unchanged frozen collector. It does not reimplement time,
identity or chain admission. A local Windows mutex and PID file prevent duplicate
instances. A stale PID file is reclaimed only after process absence is verified.
Cadence is the next UTC minute plus three seconds. Transient exceptions outside
collector policy states use 5/15/30/60/120/300-second bounded backoff. Every existing
PAUSED marker takes precedence; no task restart or user-logon start clears it.

Important frozen behavior: collector v4 creates a persistent pause even for some
connection or stale-clock failures. This supervisor cannot treat those failures as
retryable without changing frozen semantics. Market closure does not grant a clock
validation exemption. Any resulting pause is reported and requires adjudication.

The Scheduled Task runs at user logon using the interactive current-user token,
RunLevel Limited, no credentials and IgnoreNew instance policy. Policy pauses exit
successfully so Task Scheduler failure restarts cannot force re-admission. If task
registration is unavailable, a hidden user-level process may be started, with the
persistence limitation explicitly reported.

Heartbeat and coverage contain operational timestamps, counts and status only.
Source gaps are unclassified coverage gaps, not assumed market holidays. The frozen
365-certified-calendar-day rule is unchanged. New raw snapshots, manifests, pending
links, quarantine and health files remain local and hash chained; prior committed
boundary evidence remains tracked and immutable. Formal archives retain fixed
health/configuration views and chain hashes, not continuously changing files.
