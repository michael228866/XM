from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RUNS_ROOT = ROOT / "training_runs"
REGISTRY = ROOT / "TRAINING_RUNS.md"
RUN_PATTERN = re.compile(r"^\d{8}T\d{6}Z_[a-z0-9][a-z0-9_-]*$")
SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
FINAL_STATUSES = {"pass", "fail", "research_only", "aborted"}
REQUIRED_FILES = {
    "manifest.json",
    "report.md",
    "metrics.json",
    "environment.txt",
    "stdout.log",
}
PACKAGES = ["numpy", "pandas", "scikit-learn", "xgboost", "MetaTrader5"]
CANDIDATE_COLUMNS = [
    "candidate_id",
    "parameters",
    "fold",
    "executable_trades",
    "trades_per_day",
    "tp_first_wr",
    "realized_wr",
    "pf",
    "mean_r",
    "pnl",
    "max_dd",
    "break_even_wr",
    "break_even_adjusted_edge",
    "cost_stress_result",
    "qualification_verdict",
]
BASE_FIELDS = [
    "schema_version",
    "run_id",
    "experiment_name",
    "status",
    "started_at_utc",
    "finished_at_utc",
    "git_commit",
    "git_dirty",
    "training_script",
    "training_script_snapshot",
    "training_script_sha256",
    "exact_command",
    "arguments",
    "random_seeds",
    "data",
    "model",
    "search",
    "registry",
    "promotion",
    "artifacts",
]
DATA_FIELDS = [
    "symbols",
    "data_sources",
    "timezone",
    "data_start_utc",
    "data_end_utc",
    "train_start_utc",
    "train_end_utc",
    "train_rows",
    "validation_start_utc",
    "validation_end_utc",
    "validation_rows",
    "test_start_utc",
    "test_end_utc",
    "test_rows",
    "purge_details",
    "embargo_details",
    "raw_snapshot_retained",
    "reproducibility_claim",
]
MT5_FIELDS = [
    "terminal_path",
    "terminal_info",
    "broker_info",
    "fetch_start_utc",
    "fetch_end_utc",
    "retrieved_at_utc",
    "returned_rows",
]
MODEL_FIELDS = [
    "model_type",
    "parameters",
    "boosted_rounds_or_estimators",
    "features",
    "feature_count",
    "label_definition",
    "horizon",
    "label_tp_sl_semantics",
    "execution_tp_sl_semantics",
    "calibration_method",
    "artifact_path",
    "artifact_sha256",
    "retention_status",
]
REGISTRY_FIELDS = [
    "parent_or_incumbent",
    "selected_configuration",
    "trades_per_day",
    "realized_win_rate",
    "pf",
    "mean_r",
    "pnl",
    "max_dd",
    "validator_result",
]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def missing(value: Any) -> bool:
    return value is None or value == ""


def require(
    value: dict[str, Any], fields: list[str], prefix: str, errors: list[str]
) -> None:
    for field in fields:
        if field not in value or missing(value[field]):
            errors.append(f"missing {prefix}{field}")


def artifact_path(run_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else run_dir / path


def git_state(root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable", True


def environment_text() -> str:
    lines = [f"python=={sys.version.split()[0]}"]
    for package in PACKAGES:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "not-installed"
        lines.append(f"{package}=={version}")
    return "\n".join(lines) + "\n"


def clean_experiment(value: str) -> str:
    value = re.sub(r"[^a-z0-9_-]+", "_", value.lower()).strip("_-")
    if not value:
        raise ValueError("experiment must contain a letter or number")
    return value


def parse_seeds(values: list[str]) -> dict[str, str]:
    seeds: dict[str, str] = {}
    for value in values:
        key, separator, seed = value.partition("=")
        if not separator or not key.strip() or not seed.strip():
            raise ValueError(f"Invalid seed {value!r}; expected NAME=VALUE")
        seeds[key.strip()] = seed.strip()
    return seeds


def new_manifest(
    run_id: str,
    experiment: str,
    script: Path,
    script_snapshot: str,
    command: str,
    arguments: list[str],
    seeds: dict[str, str],
    seed_note: str,
    root: Path,
) -> dict[str, Any]:
    commit, dirty = git_state(root)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "experiment_name": experiment,
        "status": "in_progress",
        "started_at_utc": utc_text(now_utc()),
        "finished_at_utc": None,
        "git_commit": commit,
        "git_dirty": dirty,
        "training_script": str(script),
        "training_script_snapshot": script_snapshot,
        "training_script_sha256": file_sha256(script),
        "exact_command": command,
        "arguments": arguments,
        "random_seeds": seeds,
        "random_seed_note": seed_note,
        "data": {
            "symbols": [],
            "data_sources": [],
            "source_files": [],
            "timezone": None,
            "data_start_utc": None,
            "data_end_utc": None,
            "train_start_utc": None,
            "train_end_utc": None,
            "train_rows": None,
            "validation_start_utc": None,
            "validation_end_utc": None,
            "validation_rows": None,
            "test_start_utc": None,
            "test_end_utc": None,
            "test_rows": None,
            "purge_details": None,
            "embargo_details": None,
            "raw_snapshot_retained": None,
            "reproducibility_claim": None,
            "mt5_fetch": {
                "used": False,
                **{field: None for field in MT5_FIELDS},
                "not_applicable_reason": None,
            },
        },
        "model": {
            "trained": False,
            **{field: None for field in MODEL_FIELDS},
            "parameters": {},
            "features": [],
            "feature_count": 0,
            "not_applicable_reason": None,
        },
        "search": {
            "performed": False,
            "predefined_search_space": {},
            "candidate_results_file": None,
            "not_applicable_reason": None,
        },
        "registry": {field: None for field in REGISTRY_FIELDS},
        "promotion": {
            "requested": False,
            "gate_result": "not_requested",
            "replacement_authorized": False,
            "operational_artifact_changed": False,
        },
        "artifacts": [],
        "aborted_reason": None,
    }


def create_run(
    experiment: str,
    script: Path,
    command: str,
    arguments: list[str] | None = None,
    seeds: dict[str, str] | None = None,
    seed_note: str = "",
    archives: list[Path] | None = None,
    runs_root: Path = RUNS_ROOT,
    root: Path = ROOT,
) -> Path:
    script = script.resolve()
    archives = [path.resolve() for path in (archives or [])]
    for path in [script, *archives]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if len({path.name for path in archives}) != len(archives):
        raise ValueError("Archived files must have unique names")
    experiment = clean_experiment(experiment)
    run_id = f"{now_utc().strftime('%Y%m%dT%H%M%SZ')}_{experiment}"
    run_dir = runs_root / run_id
    pre_run_commit, pre_run_dirty = git_state(root)
    run_dir.mkdir(parents=True, exist_ok=False)

    suffix = "".join(script.suffixes) or ".txt"
    snapshot = run_dir / f"training_script{suffix}"
    shutil.copy2(script, snapshot)
    manifest = new_manifest(
        run_id,
        experiment,
        script,
        snapshot.name,
        command,
        list(arguments or []),
        dict(seeds or {}),
        seed_note,
        root,
    )
    # new_manifest runs after the run directory exists, which makes a clean
    # worktree appear dirty. Preserve the state observed immediately before
    # the immutable run directory was created.
    manifest["git_commit"] = pre_run_commit
    manifest["git_dirty"] = pre_run_dirty
    if archives:
        archive_dir = run_dir / "incumbent"
        archive_dir.mkdir()
        for source in archives:
            destination = archive_dir / source.name
            if destination.exists():
                raise FileExistsError(destination)
            shutil.copy2(source, destination)
            manifest["artifacts"].append(
                {
                    "kind": "incumbent_snapshot",
                    "original_path": str(source),
                    "path": destination.relative_to(run_dir).as_posix(),
                    "sha256": file_sha256(destination),
                    "retention_status": "stored_in_run_directory",
                }
            )

    write_json(run_dir / "manifest.json", manifest)
    (run_dir / "report.md").write_text(
        f"# Training run {run_id}\n\nStatus: `in_progress`\n", encoding="utf-8"
    )
    write_json(run_dir / "metrics.json", {"summary": {}, "folds": []})
    (run_dir / "environment.txt").write_text(
        environment_text(), encoding="utf-8"
    )
    (run_dir / "stdout.log").touch()
    with (run_dir / "candidates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        csv.writer(file).writerow(CANDIDATE_COLUMNS)
    return run_dir


def validate_candidates(path: Path, errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"missing candidate results: {path}")
        return
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing_columns = [
            column
            for column in CANDIDATE_COLUMNS
            if column not in (reader.fieldnames or [])
        ]
        if missing_columns:
            errors.append("candidate columns missing: " + ", ".join(missing_columns))
        if next(reader, None) is None:
            errors.append("candidate results contain no evaluated candidates")


def validate_finalized(run_dir: Path, errors: list[str]) -> None:
    path = run_dir / "FINALIZED.json"
    if not path.is_file():
        return
    try:
        hashes = read_json(path).get("file_sha256", {})
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"invalid FINALIZED.json: {exc}")
        return
    if not hashes:
        errors.append("FINALIZED.json has no file hashes")
    for name, expected in hashes.items():
        retained = run_dir / name
        if not retained.is_file():
            errors.append(f"finalized file missing: {name}")
        elif file_sha256(retained) != expected:
            errors.append(f"finalized file changed: {name}")


def validate_run(run_dir: Path) -> list[str]:
    run_dir = run_dir.resolve()
    errors: list[str] = []
    if not run_dir.is_dir():
        return [f"run directory not found: {run_dir}"]
    if not RUN_PATTERN.fullmatch(run_dir.name):
        errors.append("invalid run directory name")
    for name in REQUIRED_FILES:
        if not (run_dir / name).is_file():
            errors.append(f"missing required file: {name}")
    try:
        manifest = read_json(run_dir / "manifest.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [*errors, f"invalid manifest.json: {exc}"]

    require(manifest, BASE_FIELDS, "manifest.", errors)
    if manifest.get("run_id") != run_dir.name:
        errors.append("manifest.run_id does not match directory")
    status = manifest.get("status")
    if status not in FINAL_STATUSES:
        errors.append(f"status is not final: {status!r}")
    if not isinstance(manifest.get("git_dirty"), bool):
        errors.append("manifest.git_dirty must be boolean")
    if not isinstance(manifest.get("arguments"), list):
        errors.append("manifest.arguments must be a list")
    if not isinstance(manifest.get("random_seeds"), dict):
        errors.append("manifest.random_seeds must be an object")
    if not manifest.get("random_seeds") and missing(manifest.get("random_seed_note")):
        errors.append("record random seeds or why none apply")

    script_hash = manifest.get("training_script_sha256", "")
    snapshot = artifact_path(run_dir, manifest.get("training_script_snapshot", ""))
    if not SHA_PATTERN.fullmatch(str(script_hash)):
        errors.append("invalid training_script_sha256")
    elif not snapshot.is_file() or file_sha256(snapshot) != script_hash:
        errors.append("training script snapshot/hash mismatch")

    environment = (run_dir / "environment.txt")
    if environment.is_file():
        keys = {
            line.partition("==")[0]
            for line in environment.read_text(encoding="utf-8").splitlines()
        }
        for package in ["python", *PACKAGES]:
            if package not in keys:
                errors.append(f"environment.txt missing {package}")
    try:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid metrics.json: {exc}")
        metrics = None

    if status == "aborted":
        if missing(manifest.get("aborted_reason")):
            errors.append("aborted run requires aborted_reason")
    else:
        report = run_dir / "report.md"
        if report.is_file() and "Status: `in_progress`" in report.read_text(
            encoding="utf-8"
        ):
            errors.append("report.md still has its placeholder")
        if metrics == {"summary": {}, "folds": []}:
            errors.append("metrics.json still has its placeholder")

        data = manifest.get("data", {})
        require(data, DATA_FIELDS, "manifest.data.", errors)
        if not data.get("symbols") or not data.get("data_sources"):
            errors.append("symbols and data_sources must not be empty")
        mt5 = data.get("mt5_fetch", {})
        if not data.get("source_files") and not mt5.get("used"):
            errors.append("source_files or MT5 fetch provenance is required")
        if not isinstance(data.get("raw_snapshot_retained"), bool):
            errors.append("raw_snapshot_retained must be boolean")
        if not data.get("raw_snapshot_retained") and data.get(
            "reproducibility_claim"
        ) == "full":
            errors.append("full reproducibility requires a retained raw snapshot")
        if mt5.get("used"):
            require(mt5, MT5_FIELDS, "manifest.data.mt5_fetch.", errors)
        elif missing(mt5.get("not_applicable_reason")):
            errors.append("MT5 not_applicable_reason is required")
        for index, source in enumerate(data.get("source_files", [])):
            require(
                source,
                ["path", "retention_status"],
                f"source_files[{index}].",
                errors,
            )
            source_hash = source.get("sha256")
            if not source_hash and missing(source.get("sha256_unavailable_reason")):
                errors.append(f"source_files[{index}] needs a hash or reason")
            if source_hash and not SHA_PATTERN.fullmatch(str(source_hash)):
                errors.append(f"source_files[{index}] has an invalid SHA-256")

        model = manifest.get("model", {})
        if model.get("trained"):
            require(model, MODEL_FIELDS, "manifest.model.", errors)
            if not model.get("features"):
                errors.append("model features must not be empty")
            if model.get("feature_count") != len(model.get("features", [])):
                errors.append("model feature_count mismatch")
            model_hash = model.get("artifact_sha256", "")
            model_file = artifact_path(run_dir, model.get("artifact_path", ""))
            external = str(model.get("retention_status", "")).startswith("external")
            if not SHA_PATTERN.fullmatch(str(model_hash)):
                errors.append("invalid model artifact SHA-256")
            elif model_file.is_file() and file_sha256(model_file) != model_hash:
                errors.append("model artifact SHA-256 mismatch")
            elif not model_file.is_file() and not external:
                errors.append(f"model artifact not found: {model_file}")
            sha_file = run_dir / "model.sha256"
            if not sha_file.is_file():
                errors.append("missing model.sha256")
            elif sha_file.read_text(encoding="utf-8").split()[0] != model_hash:
                errors.append("model.sha256 does not match manifest")
        elif missing(model.get("not_applicable_reason")):
            errors.append("model not_applicable_reason is required")

        search = manifest.get("search", {})
        if search.get("performed"):
            if not search.get("predefined_search_space"):
                errors.append("predefined_search_space must not be empty")
            candidate_file = search.get("candidate_results_file")
            if not candidate_file:
                errors.append("candidate_results_file is required")
            else:
                validate_candidates(artifact_path(run_dir, candidate_file), errors)
        elif missing(search.get("not_applicable_reason")):
            errors.append("search not_applicable_reason is required")
        require(
            manifest.get("registry", {}),
            REGISTRY_FIELDS,
            "manifest.registry.",
            errors,
        )

    promotion = manifest.get("promotion", {})
    require(
        promotion,
        [
            "requested",
            "gate_result",
            "replacement_authorized",
            "operational_artifact_changed",
        ],
        "manifest.promotion.",
        errors,
    )
    if promotion.get("operational_artifact_changed") and not (
        promotion.get("replacement_authorized")
        and str(promotion.get("gate_result")).lower() == "pass"
    ):
        errors.append("operational artifact changed without separate authorization")
    for index, item in enumerate(manifest.get("artifacts", [])):
        require(
            item,
            ["path", "sha256", "retention_status"],
            f"manifest.artifacts[{index}].",
            errors,
        )
        if item.get("sha256") and not SHA_PATTERN.fullmatch(str(item["sha256"])):
            errors.append(f"manifest.artifacts[{index}] has an invalid SHA-256")
    validate_finalized(run_dir, errors)
    return errors


def copy_model(run_dir: Path, manifest: dict[str, Any], source: Path) -> None:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        source.relative_to(run_dir)
        destination = source
    except ValueError:
        destination = run_dir / f"model{''.join(source.suffixes) or '.bin'}"
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copy2(source, destination)
    model_hash = file_sha256(destination)
    manifest["model"].update(
        {
            "trained": True,
            "artifact_path": destination.relative_to(run_dir).as_posix(),
            "artifact_sha256": model_hash,
            "retention_status": "stored_in_run_directory",
        }
    )
    (run_dir / "model.sha256").write_text(
        f"{model_hash}  {destination.name}\n", encoding="utf-8"
    )


def finalize_run(
    run_dir: Path,
    status: str,
    model: Path | None = None,
    aborted_reason: str | None = None,
) -> list[str]:
    run_dir = run_dir.resolve()
    if (run_dir / "FINALIZED.json").exists():
        raise FileExistsError(f"Run already finalized: {run_dir.name}")
    manifest = read_json(run_dir / "manifest.json")
    if model:
        copy_model(run_dir, manifest, model)
    manifest["status"] = status
    manifest["finished_at_utc"] = utc_text(now_utc())
    if status == "aborted":
        manifest["aborted_reason"] = aborted_reason
    write_json(run_dir / "manifest.json", manifest)
    errors = validate_run(run_dir)
    if errors:
        return errors
    hashes = {
        path.relative_to(run_dir).as_posix(): file_sha256(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "FINALIZED.json"
    }
    write_json(
        run_dir / "FINALIZED.json",
        {
            "run_id": manifest["run_id"],
            "finalized_at_utc": utc_text(now_utc()),
            "file_sha256": hashes,
        },
    )
    return []


def registry_text(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value).replace("|", "\\|").replace("\n", " ")


def register_run(run_dir: Path, registry: Path = REGISTRY) -> None:
    run_dir = run_dir.resolve()
    if not (run_dir / "FINALIZED.json").is_file():
        raise ValueError("Run must be finalized before registration")
    errors = validate_run(run_dir)
    if errors:
        raise ValueError("Run validation failed: " + "; ".join(errors))
    manifest = read_json(run_dir / "manifest.json")
    existing = registry.read_text(encoding="utf-8")
    if f"| {manifest['run_id']} |" in existing:
        raise ValueError(f"Run already registered: {manifest['run_id']}")
    data, model, summary = manifest["data"], manifest["model"], manifest["registry"]
    interval = "n/a"
    if data.get("data_start_utc") and data.get("data_end_utc"):
        interval = f"{data['data_start_utc']}..{data['data_end_utc']}"
    values = [
        manifest["run_id"],
        manifest["started_at_utc"][:10],
        manifest["experiment_name"],
        summary.get("parent_or_incumbent"),
        model.get("artifact_sha256"),
        interval,
        summary.get("selected_configuration"),
        summary.get("trades_per_day"),
        summary.get("realized_win_rate"),
        summary.get("pf"),
        summary.get("mean_r"),
        summary.get("pnl"),
        summary.get("max_dd"),
        summary.get("validator_result"),
        manifest["status"],
        model.get("artifact_path") or f"training_runs/{manifest['run_id']}/",
    ]
    # ponytail: add a file lock if concurrent registry writers appear.
    with registry.open("a", encoding="utf-8", newline="") as file:
        file.write("| " + " | ".join(map(registry_text, values)) + " |\n")


def self_check() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        script = root / "train.py"
        source = root / "gold.csv"
        script.write_text("print('training')\n", encoding="utf-8")
        source.write_text("time,close\n2026-01-01,1\n", encoding="utf-8")
        run_dir = create_run(
            "self_check",
            script,
            "python train.py --seed 42",
            ["--seed", "42"],
            {"python": "42", "xgboost": "42"},
            runs_root=root / "training_runs",
            root=root,
        )
        model = run_dir / "model.json"
        model.write_text("{}\n", encoding="utf-8")
        manifest = read_json(run_dir / "manifest.json")
        manifest["data"].update(
            {
                "symbols": ["GOLD#"],
                "data_sources": ["fixture"],
                "source_files": [
                    {
                        "path": str(source),
                        "sha256": file_sha256(source),
                        "retention_status": "fixture",
                    }
                ],
                "timezone": "UTC",
                "data_start_utc": "2026-01-01T00:00:00Z",
                "data_end_utc": "2026-01-02T00:00:00Z",
                "train_start_utc": "2026-01-01T00:00:00Z",
                "train_end_utc": "2026-01-01T12:00:00Z",
                "train_rows": 10,
                "validation_start_utc": "2026-01-01T12:00:00Z",
                "validation_end_utc": "2026-01-01T18:00:00Z",
                "validation_rows": 5,
                "test_start_utc": "2026-01-01T18:00:00Z",
                "test_end_utc": "2026-01-02T00:00:00Z",
                "test_rows": 5,
                "purge_details": "1 row",
                "embargo_details": "none",
                "raw_snapshot_retained": True,
                "reproducibility_claim": "full",
            }
        )
        manifest["data"]["mt5_fetch"]["not_applicable_reason"] = "fixture"
        manifest["model"].update(
            {
                "trained": True,
                "model_type": "fixture",
                "parameters": {"seed": 42},
                "boosted_rounds_or_estimators": 1,
                "features": ["close"],
                "feature_count": 1,
                "label_definition": "fixture",
                "horizon": 1,
                "label_tp_sl_semantics": "n/a",
                "execution_tp_sl_semantics": "n/a",
                "calibration_method": "none",
                "artifact_path": "model.json",
                "artifact_sha256": file_sha256(model),
                "retention_status": "stored_in_run_directory",
            }
        )
        manifest["search"].update(
            {
                "performed": True,
                "predefined_search_space": {"seed": [42]},
                "candidate_results_file": "candidates.csv",
            }
        )
        manifest["registry"].update(
            {
                "parent_or_incumbent": "fixture",
                "selected_configuration": {"seed": 42},
                "trades_per_day": 1,
                "realized_win_rate": 0.6,
                "pf": 1.1,
                "mean_r": 0.1,
                "pnl": 1,
                "max_dd": -0.01,
                "validator_result": "self_check",
            }
        )
        write_json(run_dir / "manifest.json", manifest)
        write_json(run_dir / "metrics.json", {"summary": {"pf": 1.1}})
        (run_dir / "report.md").write_text("# Self-check\n", encoding="utf-8")
        (run_dir / "model.sha256").write_text(
            f"{file_sha256(model)}  model.json\n", encoding="utf-8"
        )
        with (run_dir / "candidates.csv").open(
            "a", encoding="utf-8", newline=""
        ) as file:
            csv.writer(file).writerow(
                ["c1", "{}", "test", 1, 1, 1, 1, 2, 1, 1, 0, 0.5, 0.5, "pass", "pass"]
            )
        assert not finalize_run(run_dir, "fail")
        assert not validate_run(run_dir)
        (run_dir / "report.md").write_text("tampered\n", encoding="utf-8")
        assert any("finalized file changed" in error for error in validate_run(run_dir))
    print("TRAINING_RUN_HISTORY_SELF_CHECK_OK")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Preserve and audit training runs.")
    result.add_argument("--self-check", action="store_true")
    actions = result.add_subparsers(dest="action")
    create = actions.add_parser("create")
    create.add_argument("--experiment", required=True)
    create.add_argument("--training-script", type=Path, required=True)
    create.add_argument("--command", required=True)
    create.add_argument("--argument", action="append", default=[])
    create.add_argument("--seed", action="append", default=[])
    create.add_argument("--seed-note", default="")
    create.add_argument("--archive", type=Path, action="append", default=[])
    validate = actions.add_parser("validate")
    validate.add_argument("run_dir", type=Path)
    finalize = actions.add_parser("finalize")
    finalize.add_argument("run_dir", type=Path)
    finalize.add_argument("--status", choices=sorted(FINAL_STATUSES), required=True)
    finalize.add_argument("--model", type=Path)
    finalize.add_argument("--aborted-reason")
    finalize.add_argument("--register", action="store_true")
    register = actions.add_parser("register")
    register.add_argument("run_dir", type=Path)
    return result


def main() -> int:
    argument_parser = parser()
    args = argument_parser.parse_args()
    try:
        if args.self_check:
            self_check()
        elif args.action == "create":
            print(
                create_run(
                    args.experiment,
                    args.training_script,
                    args.command,
                    args.argument,
                    parse_seeds(args.seed),
                    args.seed_note,
                    args.archive,
                )
            )
        elif args.action == "validate":
            errors = validate_run(args.run_dir)
            message = (
                "TRAINING_RUN_PROVENANCE_PASS"
                if not errors
                else "TRAINING_RUN_PROVENANCE_FAIL"
            )
            print(message)
            for error in errors:
                print(f"- {error}")
            return int(bool(errors))
        elif args.action == "finalize":
            if args.status == "aborted" and not args.aborted_reason:
                argument_parser.error("--aborted-reason is required for aborted runs")
            errors = finalize_run(
                args.run_dir, args.status, args.model, args.aborted_reason
            )
            if errors:
                print("TRAINING_RUN_PROVENANCE_FAIL")
                for error in errors:
                    print(f"- {error}")
                return 1
            if args.register:
                register_run(args.run_dir)
            print("TRAINING_RUN_FINALIZED")
        elif args.action == "register":
            register_run(args.run_dir)
            print("TRAINING_RUN_REGISTERED")
        else:
            argument_parser.print_help()
            return 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
