# CFTC GOLD COT Positioning Foundation V1

This is a data-only foundation for the official CFTC **Disaggregated Commitments of Traders — Futures Only** GOLD contract (`088691`). It creates exactly five frozen features and causally joins them to the repository's six frozen GOLD timestamp blocks.

It does not read labels, predictions, trades, returns, or strategy outcomes. It does not train a model or change `gemini.py` or `gold_long_recent_candidate_xgb.json`.

## Frozen definition

Source: official annual CFTC `fut_disagg_txt_<year>.zip` files only, covering 2015 prehistory and 2016–2024 research history.

Feature order:

1. `COT_MM_NET_PCT_OI`
2. `COT_MM_NET_CHG_1W`
3. `COT_PROD_NET_PCT_OI`
4. `COT_SWAP_NET_PCT_OI`
5. `COT_MM_VS_PROD_SPREAD`

Availability is frozen as `Report_Date + 4 calendar days at 00:00:00 America/New_York`, converted to UTC with the timezone database. Each GOLD timestamp receives the latest observation whose availability time is not later than that timestamp. No nearest/forward join, interpolation, same-week anticipation, or zero fill is permitted.

Official raw schema names are introspected at runtime. A ZIP may contain multiple TXT/CSV members, but exactly one member must uniquely satisfy the frozen CFTC schema. For `report_date`, `Report_Date_as_YYYY-MM-DD` (normalized `reportdateasyyyymmdd`) takes deterministic priority: exactly one primary match is used even when As-of-Date aliases coexist; multiple primary matches fail. Only when the primary is absent must exactly one supported fallback (`asofdateinformyyyymmdd` or `asofdateinformyymmdd`) match; zero or multiple fallback matches fail. Selection uses column names only, never data values. All other canonical fields still require exactly one deterministic normalized alias. Zero or multiple qualifying members stop the run. This corrects a schema-mapping implementation bug, not a CFTC data failure, and is implemented independently in the foundation and validator.

The frozen `exact_timestamps.npz` source must come from the validated finalized run `20260905T171629Z_gemini_macro_event_integration_foundation_v1`. The foundation verifies that run's finalized archive alongside the Treasury and parent runs. Before loading timestamps, the independent validator checks `FINALIZED.json`, requires the recorded `exact_timestamps.npz` hash, and verifies the recorded archive file hashes. The six frozen timestamp row counts and hashes remain unchanged.

## Formal execution sequence

Run these commands from `D:\XM\數據` in PowerShell. Do not include unrelated working-tree files in the commits.

### 1. Prerequisite and static checks

```powershell
.\.venv\Scripts\python.exe -m py_compile .\cftc_gold_cot_foundation_v1.py
.\.venv\Scripts\python.exe -m py_compile .\validate_cftc_gold_cot_foundation_v1.py
.\.venv\Scripts\python.exe .\cftc_gold_cot_foundation_v1.py --self-test
.\.venv\Scripts\python.exe .\validate_cftc_gold_cot_foundation_v1.py --self-test
git status --short
git rev-parse HEAD
git rev-parse "@{u}"
```

Resolve unrelated dirty files before the formal run. The two commit IDs must match.

### 2. Commit the frozen implementation before computation

```powershell
git add -- cftc_gold_cot_foundation_v1.py validate_cftc_gold_cot_foundation_v1.py README_CFTC_COT.md
git commit -m "Prepare CFTC GOLD COT foundation"
git push
git status --short
git rev-parse HEAD
git rev-parse "@{u}"
```

The working tree should be clean and `HEAD` must equal upstream before creating the run.

### 3. Create one formal run

```powershell
$runDir = .\.venv\Scripts\python.exe .\training_run_history.py create `
  --experiment gemini_cftc_gold_cot_foundation_v1 `
  --training-script .\cftc_gold_cot_foundation_v1.py `
  --command ".\.venv\Scripts\python.exe .\cftc_gold_cot_foundation_v1.py --run-dir TRAINING_RUN_DIR" `
  --seed-note "No randomness; deterministic data-only acquisition, formulas, and causal as-of joins."
$runDir = $runDir.Trim()
```

### 4. Execute the foundation

```powershell
.\.venv\Scripts\python.exe .\cftc_gold_cot_foundation_v1.py --run-dir $runDir `
  *>&1 | Tee-Object -FilePath "$runDir\stdout.log"
```

Acquisition uses an explicit user agent, a 60-second timeout, and three bounded attempts. Any failed official annual download, ambiguous schema, wrong contract identity, timestamp mismatch, or unresolved feature row fails the foundation. No alternate provider is used.

### 5. Execute the independent validator

```powershell
.\.venv\Scripts\python.exe .\validate_cftc_gold_cot_foundation_v1.py --run-dir $runDir `
  *>&1 | Tee-Object -FilePath "$runDir\validator_stdout.log"
Get-Content "$runDir\validator.md"
```

The validator independently reparses every retained official ZIP, rebuilds the canonical observations and all five features, performs the six causal joins, compares every matrix element, and recomputes logical hashes. It does not import the foundation script.

### 6. Finalize, register, and validate provenance

Only an `INDEPENDENT_VALIDATOR_PASS` may be finalized with status `pass`:

```powershell
.\.venv\Scripts\python.exe .\training_run_history.py finalize $runDir --status pass --register
.\.venv\Scripts\python.exe .\training_run_history.py validate $runDir
```

If the data or independent validator fails, do not call it PASS. Preserve the formal failure under the repository policy:

```powershell
.\.venv\Scripts\python.exe .\training_run_history.py finalize $runDir --status fail --register
.\.venv\Scripts\python.exe .\training_run_history.py validate $runDir
```

### 7. Archive the formal evidence

```powershell
git status --short
git add -- TRAINING_RUNS.md $runDir
git commit -m "Archive CFTC GOLD COT foundation run"
git push
git rev-parse HEAD
git rev-parse "@{u}"
git status --short
```

The final commit IDs must match and the worktree must be clean. Committing run evidence is archival provenance only; it does not authorize model training, B0/B1 evaluation, production promotion, or operational-file changes.

## Expected run artifacts

The foundation produces:

```text
raw/
source_hashes.json
source_schema.json
cot_observations.csv
cftc_gold_cot_features.csv
cftc_gold_cot_features.npz
identity_audit.csv
coverage_audit.csv
metrics.json
report.md
validator.json
validator.md
```

The NPZ contains `feature_names` plus exactly `fold1_train`, `fold1_score`, `fold2_train`, `fold2_score`, `fold3_train`, and `fold3_score`. All matrices are frozen as `float64`, with shape `rows × 5`.
