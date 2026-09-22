"""Archive one offline capture certification, including an honest blocked result."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import training_run_history as history

ROOT = Path(__file__).resolve().parent
CERTIFIER = "gold_future_capture_certification_v1.py"
VALIDATOR = "validate_gold_future_capture_certification_run_v1.py"
ATTESTATIONS = ["gold_future_capture_source_attestation_v1.json",
                "gold_future_capture_timezone_attestation_v1.json",
                "gold_future_capture_activation_v1.json"]
RELEASE = ["execution_spec_gold_future_capture_certification_v1.json",
           "gold_future_capture_manifest_schema_v1.json",
           "gold_future_capture_source_attestation_template_v1.json",
           "gold_future_capture_timezone_attestation_template_v1.json",
           "gold_future_capture_activation_template_v1.json"]
COLLECTOR = "gold_data_foundation_forward_collector.py"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    before = git("status", "--porcelain", "-z")
    if before:
        raise RuntimeError("Clean worktree required before run creation")
    commit = git("rev-parse", "HEAD").decode().strip()
    remote = git("ls-remote", "origin", "refs/heads/main").decode().split()[0]
    if remote != commit or git("branch", "--show-current").decode().strip() != "main":
        raise RuntimeError("Source must be pushed to origin/main")
    inputs = [CERTIFIER, VALIDATOR, Path(__file__).name, "training_run_history.py",
              COLLECTOR, "gold_data_foundation1_audit.py", *ATTESTATIONS, *RELEASE]
    inputs += [p.relative_to(ROOT).as_posix() for p in sorted(
        (ROOT / "research_evidence/gold_future_capture_v1").iterdir()) if p.is_file()]
    for name in inputs:
        raw = (ROOT / name).read_bytes()
        blob = git("cat-file", "blob", commit + ":" + name)
        if raw.replace(b"\r\n", b"\n") != blob.replace(b"\r\n", b"\n"):
            raise RuntimeError("Uncommitted input: " + name)
    protected = {name: sha(ROOT / name) for name in
                 ("gemini.py", "gold_long_recent_candidate_xgb.json")}
    run = history.create_run("gold_future_capture_certification_v1", ROOT / CERTIFIER,
                             "pending exact subprocess command", seed_note="No randomness; offline document certification")
    manifest = history.read_json(run / "manifest.json")
    manifest.update(source_commit=commit, pre_run_remote_commit=remote, git_branch="main",
                    git_upstream="origin/main", pre_create_git_status="", input_snapshots=[],
                    protected_sha256_before=protected,
                    result_commit_binding="Archival commit is identified by Git history of FINALIZED.json; no self-referential commit hash")
    snapshot_root = run / "source"
    for name in inputs:
        destination = snapshot_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
        manifest["input_snapshots"].append({"source_path": name,
            "path": destination.relative_to(run).as_posix(), "sha256": sha(destination),
            "retention_status": "stored_in_run_directory_and_git"})
    shutil.copyfile(ROOT / VALIDATOR, run / "validator_script.py")
    shutil.copyfile(ROOT / RELEASE[0], run / "execution_spec.json")
    output = ROOT / ("gold_" + run.name + "_report.json")
    if output.exists():
        raise FileExistsError("Certification output already exists")
    command = [sys.executable, "-B", str(ROOT / CERTIFIER), "--output", str(output),
               "--repo-root", str(ROOT), "--collector", str(ROOT / COLLECTOR)]
    for option, name in zip(("--source-attestation", "--timezone-attestation", "--activation"), ATTESTATIONS):
        command.extend([option, str(ROOT / name)])
    manifest.update(exact_command=subprocess.list2cmdline(command), arguments=command[2:],
                    certification_command=command, certification_output_original=str(output),
                    git_status_at_certification=git("status", "--porcelain", "-z").decode("utf-8"))
    history.write_json(run / "manifest.json", manifest)
    with (run / "stdout.txt").open("xb") as out, (run / "stderr.txt").open("xb") as err:
        result = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err, check=False)
    shutil.copyfile(run / "stdout.txt", run / "stdout.log")
    manifest["certification_exit_code"] = result.returncode
    history.write_json(run / "manifest.json", manifest)
    if result.returncode != 0 or not output.is_file():
        print("RUN_ID=" + run.relative_to(ROOT).as_posix())
        raise RuntimeError("Certification failed to produce a valid report; preserve run, do not retry")
    shutil.copyfile(output, run / "certification_report.json")
    report = history.read_json(output)
    # This preparation stopped at phase 1. Never activate or upgrade its verdict.
    if report["readiness"]["status"] == "READY_FOR_CAPTURE_ACTIVATION":
        raise RuntimeError("Unexpected readiness; independent review required")
    manifest.update(research_verdict="FAIL", methodology_status="PENDING_VALIDATOR",
                    phase_stopped="PHASE_1_SOURCE_ATTESTATION", capture_activated=False,
                    strategy_outcome_inspected=False, model_loaded_for_holdout=False,
                    model_trained_for_holdout=False, production_change=False,
                    protected_sha256_after={name: sha(ROOT / name) for name in protected})
    data = manifest["data"]
    data.update(symbols=["GOLD#"], data_sources=["User-provided metadata and reviewed public documentation; no market retrieval"],
                source_files=manifest["input_snapshots"], timezone="UNRESOLVED",
                raw_snapshot_retained=False, reproducibility_claim="Exact certification documents and source retained; no raw market data used",
                purge_details="Not applicable: no strategy evaluation", embargo_details="Not applicable: no strategy evaluation")
    for key in ("data_start_utc", "data_end_utc", "train_start_utc", "train_end_utc",
                "validation_start_utc", "validation_end_utc", "test_start_utc", "test_end_utc"):
        data[key] = "NOT_APPLICABLE_NO_MARKET_DATA"
    for key in ("train_rows", "validation_rows", "test_rows"):
        data[key] = 0
    data["mt5_fetch"]["not_applicable_reason"] = "No terminal connection or query"
    manifest["model"]["not_applicable_reason"] = "No model loaded or trained"
    manifest["search"]["not_applicable_reason"] = "No strategy search or outcomes"
    manifest["registry"].update({key: "N/A: no strategy evaluation" for key in history.REGISTRY_FIELDS})
    manifest["registry"].update(parent_or_incumbent="GOLD S4 historical development only",
        selected_configuration="None; source/timezone unresolved; capture not activated; no production change/promotion",
        validator_result="PENDING")
    history.write_json(run / "manifest.json", manifest)
    history.write_json(run / "metrics.json", {"certification_readiness": report["readiness"],
        "research_verdict": "FAIL", "strategy_metrics_computed": False})
    (run / "report.md").write_text("# GOLD future capture certification\n\nResearch verdict: FAIL.\n"
        "\nMethodology/provenance and independent validator results are in validator.json.\n"
        "Source preparation stopped at phase 1: environment unknown and timezone/calendar uncertified.\n"
        "No candidate selected, protocol freeze, capture activation, holdout start, production promotion or production change.\n"
        "Historical evidence is development only. No strategy outcomes inspected or models loaded/trained.\n\n"
        + "\n".join("- " + x["status"] + ": " + x["reason"] for x in report["readiness"]["blockers"])
        + "\n\nSee source/research_evidence/gold_future_capture_v1/preparation_review.md for exact evidence gaps.\n",
        encoding="utf-8")
    print("RUN_ID=" + run.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
