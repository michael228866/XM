# Enablement findings

Methodology/provenance PASS. Independent validator PASS, failed_check_names=[]. Formal enablement FAIL.

Controlled timing sample at 2026-09-23T06:25:11 UTC: tick epoch interpreted as UTC is approximately +10799.595 seconds ahead of system UTC; legacy conversion is approximately -0.405 seconds behind system UTC. time_msc agrees with seconds. Five M1 bars are monotonic with 60-second spacing. Classification LEGACY_CONVERSION_SUPPORTED applies to this small current sample only. The direct-UTC range request also returned timing metadata; full historical/API/DST equivalence is not proved. Do not certify a timezone from this observation alone.

Safe live metadata confirms XM Global Limited / XMGlobal-MT5 6 / GOLD#, terminal build 6182, digits 2, point 0.01, environment demo. No sensitive account fields or diagnostic price values were retained.

All 21 native timeframes returned three closed, schema-compatible bars. NATIVE_20TF_REQUIRED is the candidate policy; source-bound continuity, calendar and full attestation completion remain separate requirements. No resampling occurred. The conditional proposal remains unapproved and unused.

Exact recursive EMA/MACD prefix remains UNRESOLVED. No history or strategy performance was evaluated. Finite warmup is not claimed to reproduce exact initialization.

Collector storage tests pass, including a three-manifest/revision chain accepted by the existing frozen verifier. The existing static checker rejects os.replace for the mutable tip index and the incomplete guarantee review. Its all-or-nothing fields (including overwrites_existing_snapshot=true) are conservative AST classifications, not observed overwrites: the synthetic byte-preservation tests explicitly reject snapshot overwrite. No checker rule was changed.

The current collector supports only certified ALREADY_UTC epochs. The observed legacy-like source therefore cannot be activated through this implementation without a future reviewed broker-rule implementation and authoritative certification. No empirical transform was silently installed into collection.

Existing source_id contains #, which the frozen source/schema regex rejects. It was not silently renamed. A separately reviewed identity/protocol decision is required.

No protocol freeze, capture activation, holdout start, model load/training, strategy outcomes, production modification or promotion. Root attestation files are updated only with archived partial evidence and still have template=true; source snapshots preserve their pre-run versions.
