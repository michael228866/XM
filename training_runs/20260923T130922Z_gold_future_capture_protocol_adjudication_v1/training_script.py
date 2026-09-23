"""Committed-source, offline v2 protocol adjudication; no capture or inference."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import training_run_history as history

ROOT = Path(__file__).resolve().parent
SPEC = "execution_spec_gold_future_capture_protocol_adjudication_v1.json"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def command(run, label, arguments):
    argv = [sys.executable, "-B", *arguments]
    with (run / (label + "_stdout.txt")).open("xb") as out, (run / (label + "_stderr.txt")).open("xb") as err:
        result = subprocess.run(argv, cwd=ROOT, stdout=out, stderr=err, check=False)
    with (run / "commands.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"label": label, "argv": argv, "exit_code": result.returncode}) + "\n")
    if result.returncode:
        raise RuntimeError("Formal command failed; preserve artifacts, do not retry: " + label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    spec = history.read_json(ROOT / SPEC)
    if git("status", "--porcelain", "-z"):
        raise RuntimeError("Clean source required")
    commit = git("rev-parse", "HEAD").decode().strip()
    remote = git("ls-remote", "origin", "refs/heads/main").decode().split()[0]
    if remote != commit or git("branch", "--show-current").decode().strip() != "main":
        raise RuntimeError("Pushed main required")
    protected = {name: sha(ROOT / name) for name in spec["protected_sha256"]}
    if protected != spec["protected_sha256"]:
        raise RuntimeError("Production hash changed")
    for name in spec["source_files"]:
        raw = (ROOT / name).read_bytes()
        blob = git("cat-file", "blob", commit + ":" + name)
        if raw.replace(b"\r\n", b"\n") != blob.replace(b"\r\n", b"\n"):
            raise RuntimeError("Uncommitted source input")
    run = history.create_run(spec["experiment_name"], Path(__file__),
        subprocess.list2cmdline([sys.executable, "-B", str(Path(__file__)), "--execute"]),
        arguments=["--execute"], seed_note="Deterministic offline protocol checks; no randomness or models")
    print("RUN_ID=" + run.relative_to(ROOT).as_posix(), flush=True)
    m = history.read_json(run / "manifest.json")
    m.update(source_commit=commit, pre_run_remote_commit=remote, git_branch="main", git_upstream="origin/main",
        pre_run_clean=True, pre_run_git_status="", protected_sha256_before=protected, input_snapshots=[],
        result_commit_binding="Git commit containing FINALIZED.json; immutable archive cannot contain its own commit hash")
    for name in spec["source_files"]:
        destination = run / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
        m["input_snapshots"].append({"source_path": name, "path": destination.relative_to(run).as_posix(),
            "sha256": sha(destination), "retention_status": "stored_in_run_directory_and_git"})
    shutil.copyfile(ROOT / SPEC, run / "execution_spec.json")
    shutil.copyfile(ROOT / "validate_gold_future_capture_protocol_adjudication_v1.py", run / "validator_script.py")
    history.write_json(run / "manifest.json", m)
    command(run, "self_test", ["gold_future_capture_certification_v2.py", "--self-test", "--export-synthetic", str(run / "synthetic_chain")])
    m["git_status_at_certification"] = git("status", "--porcelain", "-z").decode("utf-8")
    command(run, "certification", ["gold_future_capture_certification_v2.py", "--output", str(run / "capture_certification_report.json")])
    report = history.read_json(run / "capture_certification_report.json")
    protocol = history.read_json(ROOT / "gold_future_capture_protocol_v2.json")
    prefix = history.read_json(ROOT / "gold_recursive_prefix_protocol_v2.json")
    tz = history.read_json(ROOT / "gold_future_capture_timezone_attestation_v2.json")
    write(run / "protocol_decisions.json", {"source_id_policy": report["source_id_policy"],
        "identity_migration": protocol["identity_migration"], "chain_tip_policy": protocol["chain_tip_policy"],
        "replace_allowlist": ["chain_tip.json"], "authoritative_immutable_classes": protocol["immutable_classes"],
        "missing_tip": "VALID_CHAIN_WITH_WARNING", "stale_tip": "VALID_CHAIN_WITH_WARNING",
        "conflicting_tip": "CHAIN_CAN_REMAIN_VALID_READINESS_BLOCKED_UNTIL_REBUILD",
        "higher_timeframe_policy": report["higher_timeframe_policy"], "timezone_policy": report["timezone_policy"],
        "prefix_policy": report["prefix_policy"], "v1_changed": False, "production_changed": False})
    write(run / "timezone_adjudication.json", {"status": report["timezone_attestation_status"],
        "timezone_policy": report["timezone_policy"], "broker_offset_intervals": tz["broker_offset_intervals"],
        "recurrence_rule": tz["recurrence_rule"], "diagnostic_support": tz["diagnostic_support"],
        "decision": "No exact-server annual DST/session/holiday authority established. Prior +3h diagnostic is support only; no annual transitions inferred."})
    write(run / "prefix_adjudication.json", {"status": report["prefix_certification_status"], **prefix})
    write(run / "collector_static_review.json", report["collector_static_review"])
    metrics = {key: value for key, value in report.items() if key not in {"collector_static_review", "capture_integrity", "bindings", "readiness"}}
    metrics.update(collector_static_status=report["collector_static_review"]["static_status"],
        capture_certification_status=report["readiness"]["status"], methodology_status="PENDING_VALIDATOR")
    history.write_json(run / "metrics.json", metrics)
    m.update(formal_run_status=report["formal_run_status"], protected_sha256_after={name: sha(ROOT / name) for name in protected},
        collector_sha256=sha(ROOT / "gold_future_capture_collector_v2.py"), certifier_sha256=sha(ROOT / "gold_future_capture_certification_v2.py"),
        manifest_schema_sha256=sha(ROOT / "gold_future_capture_manifest_schema_v2.json"),
        v1_preserved_sha256=protocol["v1_preserved_sha256"], prior_evidence=protocol["prior_artifacts"],
        git_dirty_reason="Clean pushed source before create. Only the new run directory is untracked during computation.",
        strategy_outcome_inspected=False, model_loaded_for_holdout=False, model_trained_for_holdout=False)
    if m["protected_sha256_after"] != protected:
        raise RuntimeError("Protected files changed; stop")
    data = m["data"]
    data.update(symbols=["GOLD#"], data_sources=["Immutable prior timing/schema evidence; official document reviews; synthetic storage fixtures"],
        source_files=m["input_snapshots"], timezone="UNRESOLVED", raw_snapshot_retained=False,
        reproducibility_claim="Exact offline inputs retained; no new market samples. Synthetic chain retained and explicitly labelled.",
        purge_details="Not applicable: no strategy evaluation", embargo_details="No strategy outcome inspection")
    for key in ("data_start_utc", "data_end_utc", "train_start_utc", "train_end_utc", "validation_start_utc", "validation_end_utc", "test_start_utc", "test_end_utc"):
        data[key] = "NOT_APPLICABLE_NO_STRATEGY_DATASET"
    for key in ("train_rows", "validation_rows", "test_rows"):
        data[key] = 0
    data["mt5_fetch"]["not_applicable_reason"] = "No MT5 calls; reuse retained prior diagnostic"
    m["model"]["not_applicable_reason"] = "No models loaded or trained"
    m["search"]["not_applicable_reason"] = "No strategy search or performance"
    m["registry"].update({key: "N/A: protocol infrastructure only" for key in history.REGISTRY_FIELDS})
    m["registry"].update(parent_or_incumbent="GOLD capture enablement v1; prior FAIL preserved",
        selected_configuration="V2 ID and tip compatibility; " + report["formal_run_status"] + "; no activation or production change/promotion",
        validator_result="PENDING")
    history.write_json(run / "manifest.json", m)
    narrative = ("# GOLD future capture protocol adjudication v1\n\nFormal verdict: " + report["formal_run_status"]
        + "\n\nV2 canonical source ID preserves exact GOLD# instrument. Restricted atomic tip replacement is permitted only after full immutable chain verification. Static collector certification passes.\n"
        + "\nTimezone remains UNRESOLVED: no annual source-specific transitions/calendar inferred from the retained +3h sample. Native M1 plus 20 timeframes preserved; no diagnostic rerun or resampling.\n"
        + "\nPrefix remains UNRESOLVED. The deterministic non-equivalent reinitialization proposal lacks an exact certified source range and immutable approval; 4096 M1/21 native rows were not tuned on outcomes.\n"
        + "\nNo protocol freeze, capture activation, holdout start, models, strategy outcomes, production change or promotion. Previous runs and v1 files unchanged.\n"
        + "\nIndependent validator results are in validator.json; synthetic storage files are test fixtures, not live capture evidence.\n")
    (run / "report.md").write_text(narrative, encoding="utf-8")
    (run / "findings.md").write_text(narrative + "\nReadiness blockers:\n" + "\n".join("- " + b["status"] + ": " + b["reason"] for b in report["readiness"]["blockers"]) + "\n", encoding="utf-8")
    shutil.copyfile(run / "certification_stdout.txt", run / "stdout.txt")
    shutil.copyfile(run / "certification_stderr.txt", run / "stderr.txt")
    shutil.copyfile(run / "stdout.txt", run / "stdout.log")
    print("FORMAL_RUN_STATUS=" + report["formal_run_status"])


if __name__ == "__main__":
    main()
