# Training Run Registry

This is the append-only index for training and parameter-selection runs created under the prospective Training Run Preservation policy. Add rows with `training_run_history.py register`; never rewrite or delete an earlier row. PASS, FAIL, `research_only` and aborted runs all belong here.

The current `2026-08-25` operational model predates this policy and is not retroactively assigned a compliant `run_id`. Its known evidence and missing raw MT5/environment snapshot are documented in [README.md](README.md); no missing fields are fabricated here.

| run_id | date | experiment | parent/incumbent | model SHA-256 | data interval | selected configuration | trades/day | realized WR | PF | Mean-R | PnL | Max DD | validator | final status | artifact location |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| 20260902T074926Z_gemini_core_gate_v1 | 2026-09-02 | gemini_core_gate_v1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | aborted | training_runs/20260902T074926Z_gemini_core_gate_v1/ |
| 20260902T075259Z_gemini_core_gate_v1 | 2026-09-02 | gemini_core_gate_v1 | gold_long_recent_candidate_xgb.json@2dc32e3b3c0ea6ca8fa2e30187bebf8ff3f7e7e03109b39b3f70f013e3a755f2 | n/a | 2018-01-02T00:00:00..2024-12-31T20:00:00 | none_quality_pass_control_retained | 0.07391474384043802 | 0.25396825396825395 | 0.3724855998495003 | -0.4459526009202168 | -84.28504157392098 | -88.41767778745977 | FAIL | fail | training_runs/20260902T075259Z_gemini_core_gate_v1/ |
| 20260903T005036Z_gemini_incumbent_robustness_v1 | 2026-09-03 | gemini_incumbent_robustness_v1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | aborted | training_runs/20260903T005036Z_gemini_incumbent_robustness_v1/ |
