# Native context and current-regime capture v4

This version preserves v1/v2/v3 source and archives. It certifies only the current
10800-second source-time regime using three bounded timestamp-only observations
against operational system UTC. The fixed five-second tolerance is an engineering
margin over the prior approximately 0.6-second observations, not a clock-authority
or future DST claim. Offset 7200 remains an uncertified candidate: persistent pause,
quarantine, separate certification and an immutable amendment are required.

The source identity addendum authorizes `account_info()` only for company, server
and immediate trade-mode normalization against package constants. No complete
account object, account identifier, financial value or credentials are serialized.
Identity is checked before and after every native fetch. Other MT5 calls are bounded
read-only metadata, tick and native bar requests. No trading API is called.

Native context requests are fixed at 4200 M1 and 25 bars per higher timeframe;
retain the latest 4096 M1 and 21 bars per higher timeframe. Exclude the newest bar
and require nominal source-calendar closure plus a successor and a current tick.
No resampling. Historical rows retain raw source epochs and are not assigned an
unverified historical UTC offset. Only the latest M1 endpoint uses the current
contemporaneously checked offset. Fetch completion is the conservative causal
upper bound and must precede protocol freeze and the future boundary.

Context is sealed in the unique formal archive before freeze. Activation copies
the exact sealed seed to `future_holdout/gold_s4_v4/context_seed/`. Every accepted
raw snapshot and manifest is published exclusively with fsync and a hash chain.
No accepted file can be overwritten. Failed acquisition or chain validation
creates a persistent pause and quarantine receipt; valid earlier evidence remains.

The formal validator is independent of execution modules and runs once. It reopens
all native context bytes, verifies hashes and time arithmetic, checks source Git
bindings and scans non-source JSON/text/log evidence for sensitive account values.
Source deny lists and documentation are not account disclosures.

Lifecycle: synthetic checks, source commit/push, one formal run, one independent
validator, immutable finalization and registry, result commit/push. Only PASS can
proceed to separate freeze, activation and boundary commits. Activation recertifies
frozen bytes. This task performs bounded raw M1 acquisition through the first legal
closed future bar, then stops; it does not install a background service.

S4 thresholds, the 31-feature implementation and S5 execution source identities
are frozen without importing or executing them. Holdout lasts 365 calendar days.
No probabilities, signals, trades or strategy metrics before separately authorized
one-shot unlock. Historical seed rows never enter holdout denominators. Evaluation
start stays null unless all feature prerequisites are independently established;
context availability alone does not prove seed-to-boundary continuity or finite
lagged feature values. No production promotion or production change is authorized.
