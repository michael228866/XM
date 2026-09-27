# GOLD manual training workflow binding v1

PARTIAL: infrastructure tests only. No approved single default, no actual training, no strategy evaluation, no live historical fetch. Production and capture preserved.

Latest related historical training is S4 confirmation 20260919T124853Z; this is not approval for a repeatable default or replacing frozen CSV reconstruction with fresh MT5 bars. Production legacy main overwrites protected model; B0 main also runs B1. Historical clock mapping, exact coverage, output adapter and independent training validator remain unresolved. See training_provenance_trace.json for all candidate hashes and native timeframes.

Launcher is NOT_READY with readable Chinese failure and pause. Process-local one-use 30-second receipt prevents accidental automated entry; no static secret. Data primitives tested on synthetic bars; CSV import supports reviewed canonical schema only. No actual model import/fit/prediction/replay. CPython audit hook is not hostile native-code OS isolation.

Capture status at certification is recorded separately in status_launcher_smoke_test.json. Pre-certification status observed PAUSED / heartbeat FAIL / chain PASS; heartbeat stopped at 2026-09-27T10:26:00.389145+00:00, PID 31912 absent. No capture restart or mutation was performed. Capture automation preserved means unchanged code/policy/task, not healthy runtime.

Independent validator PASS (144 checks; failed=[]). Methodology/provenance PASS; binding outcome PARTIAL. No actual training approved.
