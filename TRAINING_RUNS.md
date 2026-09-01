# Training Run Registry

This is the append-only index for training and parameter-selection runs created under the prospective Training Run Preservation policy. Add rows with `training_run_history.py register`; never rewrite or delete an earlier row. PASS, FAIL, `research_only` and aborted runs all belong here.

The current `2026-08-25` operational model predates this policy and is not retroactively assigned a compliant `run_id`. Its known evidence and missing raw MT5/environment snapshot are documented in [README.md](README.md); no missing fields are fabricated here.

| run_id | date | experiment | parent/incumbent | model SHA-256 | data interval | selected configuration | trades/day | realized WR | PF | Mean-R | PnL | Max DD | validator | final status | artifact location |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
