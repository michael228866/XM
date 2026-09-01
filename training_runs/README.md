# Training run archive

Every model-training or parameter-selection execution started after this policy must use a unique directory:

```text
training_runs/<YYYYMMDDTHHMMSSZ>_<experiment_name>/
```

Example: `training_runs/20260901T153012Z_gemini_core_gate/`.

Directories are created with [training_run_history.py](../training_run_history.py), which refuses to reuse an existing `run_id`. A finalized run contains `FINALIZED.json` with SHA-256 values for every retained file; later modification makes validation fail. Corrections require a new run rather than overwriting the old one.

## Required contents

```text
manifest.json
training_script.<ext>   # immutable copy of the executed training script
report.md
metrics.json
environment.txt
stdout.log
candidates.csv          # required when search.performed=true
model.<ext>             # required when model.trained=true and retained in Git
model.sha256            # required when model.trained=true
FINALIZED.json          # written only after provenance validation passes
```

Large artifacts may be kept in external artifact storage. Their exact path, SHA-256 and `retention_status` must still be present in `manifest.json`; absence of a retained raw data snapshot must be reported as less than full reproducibility.

## Manifest contract

Run identity:

- `run_id`, `experiment_name`, `status`, `started_at_utc`, `finished_at_utc`
- `git_commit`, `git_dirty`
- `training_script`, `training_script_snapshot`, `training_script_sha256`
- `exact_command`, `arguments`, `random_seeds` or a reason no seed applies

Data identity:

- symbols, sources, source file paths and SHA-256 where feasible
- timezone and total data interval
- train, validation and test start/end/row count
- purge and embargo details
- whether the raw snapshot was retained and the allowed reproducibility claim
- for dynamic MT5 fetches: terminal path/info, broker info, exact fetch interval, retrieval timestamp and returned rows

Model identity:

- model type and complete parameters
- estimators/boosted rounds
- complete feature list and count
- label definition, horizon and label TP/SL semantics
- execution TP/SL semantics when different
- calibration method
- model path, SHA-256 and retention status

Search identity:

- whether search occurred
- the entire predefined search space
- `candidates.csv` containing every evaluated candidate, including failures
- candidate ID, parameters, fold, executable trades, trades/day, TP-first WR, realized WR, PF, Mean-R, PnL, Max DD, break-even WR, break-even-adjusted edge, cost stress and qualification verdict

The manifest also records registry summary values and whether promotion/replacement was separately requested and authorized. Training completion alone must leave `operational_artifact_changed=false`.

## Required workflow

Create the run before starting computation:

```powershell
$runDir = py -3.10 training_run_history.py create `
  --experiment gen22_example `
  --training-script gold_generation22_example.py `
  --command "py -3.10 gold_generation22_example.py --run-dir `$env:TRAINING_RUN_DIR" `
  --seed python=42 `
  --seed xgboost=42
$env:TRAINING_RUN_DIR = $runDir.Trim()
```

The training script must write its report, metrics, candidate table and model into `$env:TRAINING_RUN_DIR`, and update that directory's `manifest.json`. Capture the complete process output:

```powershell
py -3.10 gold_generation22_example.py --run-dir $env:TRAINING_RUN_DIR `
  *>&1 | Tee-Object -FilePath "$env:TRAINING_RUN_DIR\stdout.log"
```

Finalize, validate and append the registry only after all fields are populated:

```powershell
py -3.10 training_run_history.py finalize $env:TRAINING_RUN_DIR `
  --status research_only `
  --model "$env:TRAINING_RUN_DIR\model.json" `
  --register
py -3.10 training_run_history.py validate $env:TRAINING_RUN_DIR
```

Use `pass`, `fail`, `research_only` or `aborted`; aborted runs require `--aborted-reason`. A completed research failure is `fail` or `research_only`, not deleted.

If a legacy script cannot avoid writing a currently operational artifact, archive the operational model and its associated report/config at run creation, before training starts:

```powershell
py -3.10 training_run_history.py create `
  --experiment legacy_example `
  --training-script legacy_train.py `
  --command "py -3.10 legacy_train.py" `
  --archive gold_long_recent_candidate_xgb.json `
  --archive gold_long_recent_walk_forward.md
```

Prefer changing the training output path to the unique run directory. Archiving is protection against legacy behavior, not authorization to replace the model loaded by `gemini.py`.

Run the helper's isolated self-check with:

```powershell
py -3.10 training_run_history.py --self-check
```
