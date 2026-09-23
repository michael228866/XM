"""One committed-source enablement run: diagnostics, document review, no capture."""
import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import training_run_history as history
import gold_future_capture_certification_v1 as certifier

ROOT = Path(__file__).resolve().parent
SPEC = "execution_spec_gold_future_capture_enablement_v1.json"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def execute(run, label, arguments):
    command = [sys.executable, "-B", *arguments]
    with (run / (label + "_stdout.txt")).open("xb") as out, (run / (label + "_stderr.txt")).open("xb") as err:
        result = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err, check=False, timeout=180)
    record = {"label": label, "argv": command, "exit_code": result.returncode}
    with (run / "commands.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    if result.returncode:
        raise RuntimeError("Command failed; preserve archive, do not retry: " + label)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    spec = history.read_json(ROOT / SPEC)
    if git("status", "--porcelain", "-z"):
        raise RuntimeError("Clean pre-run worktree required")
    commit = git("rev-parse", "HEAD").decode().strip()
    remote = git("ls-remote", "origin", "refs/heads/main").decode().split()[0]
    if remote != commit or git("branch", "--show-current").decode().strip() != "main":
        raise RuntimeError("Pushed main required")
    protected = {name: sha(ROOT / name) for name in spec["protected_sha256"]}
    if protected != spec["protected_sha256"]:
        raise RuntimeError("Protected source changed")
    inputs = spec["source_files"] + [p.relative_to(ROOT).as_posix() for p in sorted(
        (ROOT / "research_evidence/gold_future_capture_enablement_v1").iterdir()) if p.is_file()]
    for name in inputs:
        raw = (ROOT / name).read_bytes()
        if raw.replace(b"\r\n", b"\n") != git("cat-file", "blob", commit + ":" + name).replace(b"\r\n", b"\n"):
            raise RuntimeError("Uncommitted source input")
    run = history.create_run(spec["experiment_name"], Path(__file__),
        subprocess.list2cmdline([sys.executable, "-B", str(Path(__file__)), "--execute"]),
        arguments=["--execute"], seed_note="No models, randomness or strategy evaluation")
    print("RUN_ID=" + run.relative_to(ROOT).as_posix(), flush=True)
    manifest = history.read_json(run / "manifest.json")
    manifest.update(source_commit=commit, pre_run_remote_commit=remote, git_branch="main", git_upstream="origin/main",
        pre_run_clean=True, pre_run_git_status="", protected_sha256_before=protected,
        result_commit_binding="Git archival commit containing FINALIZED.json; no self-referential hash",
        python_version=sys.version, MetaTrader5_version=importlib.metadata.version("MetaTrader5"), input_snapshots=[])
    for name in inputs:
        target = run / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        manifest["input_snapshots"].append({"source_path": name, "path": target.relative_to(run).as_posix(),
            "sha256": sha(target), "retention_status": "stored_in_run_directory_and_git"})
    shutil.copyfile(ROOT / SPEC, run / "execution_spec.json")
    shutil.copyfile(ROOT / "validate_gold_future_capture_enablement_v1.py", run / "validator_script.py")
    history.write_json(run / "manifest.json", manifest)
    execute(run, "collector_self_test", ["test_gold_future_capture_collector_v1.py"])
    for label, script, output in (
        ("timestamp", "gold_mt5_timestamp_semantics_diagnostic_v1.py", "diagnostic_timestamp.json"),
        ("native", "gold_mt5_native_timeframe_audit_v1.py", "native_timeframe_audit.json"),
        ("prefix", "gold_recursive_prefix_certification_v1.py", "prefix_certification.json")):
        execute(run, label, [script, "--output", str(run / output)])
    timestamp = history.read_json(run / "diagnostic_timestamp.json")
    native = history.read_json(run / "native_timeframe_audit.json")
    prefix = history.read_json(run / "prefix_certification.json")
    derived = run / "attestations"
    derived.mkdir()
    names = ["gold_future_capture_source_attestation_v1.json", "gold_future_capture_timezone_attestation_v1.json",
             "gold_future_capture_activation_v1.json"]
    source, tz, activation = [history.read_json(ROOT / name) for name in names]
    evidence = history.read_json(ROOT / "research_evidence/gold_future_capture_enablement_v1/evidence_index.json")["evidence"]
    proof = [{key: item[key] for key in ("authority", "local_path", "sha256", "reviewed_by", "reviewed_at_utc")} for item in evidence]
    for item in proof:
        item["local_path"] = str((run / "source" / item["local_path"]).resolve())
    source.update(source_account_environment=timestamp.get("source_account_environment", "unknown"), evidence=proof,
        notes="Partial source preparation; live safe metadata preserved in diagnostic_timestamp.json. Source ID includes #, rejected by frozen schema. Calendar/timezone still uncertified. No automatic source rename.")
    tz.update(source_attestation_sha256=certifier.canonical_hash(source), evidence=proof,
        notes="Diagnostic classification: " + timestamp["diagnostic_classification"] + "; current evidence does not certify exact server DST/session/holiday rules. Status remains UNRESOLVED.")
    activation.update(source_attestation_sha256=certifier.canonical_hash(source), timezone_attestation_sha256=certifier.canonical_hash(tz),
        collector_commit=commit, collector_version="gold_future_capture_collector_v1",
        higher_timeframe_policy=native["candidate_policy"], prefix_policy=prefix["prefix_policy"],
        notes="Candidate native policy only; incomplete attestation. Static tip-publication conflict blocks activation. No protocol frozen, no holdout started.")
    if native["status"] != "PASS":
        activation.update(higher_timeframe_policy="M1_DETERMINISTIC_RESAMPLE_PROPOSED", protocol_change_required=True,
            resampling_proposal={"boundaries": "UTC midnight intraday; Monday weekly; first day monthly",
                "closed_bar_rule": "Only fully closed intervals", "calendar": "Requires exact authoritative broker session calendar; unresolved",
                "missing_bar_rule": "Quarantine if expected M1 slots missing", "immutable_approval_required": True})
        shutil.copyfile(ROOT / "gold_m1_resampling_protocol_change_proposal_v1.md", run / "resampling_proposal.md")
    review = activation["collector_review"]
    review.update(collector_sha256=sha(ROOT / "gold_future_capture_collector_v1.py"), collector_git_commit=commit,
        reviewed_by="Codex source review; independent archive validator is separate code", reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
        timezone_conversion_location="normalize_epoch: accepts certified direct UTC only", evidence=proof)
    for key in review["guarantees"]:
        review["guarantees"][key] = key not in {"writes_append_only", "dependencies_bound"}
    review["recovery_cases"] = {key: "See committed README storage and recovery review plus archived fault-injection self-test" for key in
        history.read_json(ROOT / "execution_spec_gold_future_capture_certification_v1.json")["recovery_cases"]}
    for name, document in zip(names, (source, tz, activation)):
        write(derived / name, document)
    static = certifier.inspect_collector((ROOT / "gold_future_capture_collector_v1.py").read_text(encoding="utf-8-sig"),
        review["collector_sha256"], commit, review, history.read_json(ROOT / "execution_spec_gold_future_capture_certification_v1.json"),
        lambda name: sha(Path(name)), datetime.now(timezone.utc))
    write(run / "collector_static_review.json", static)
    output = ROOT / ("gold_" + run.name + "_certification.json")
    manifest["git_status_at_certification"] = git("status", "--porcelain", "-z").decode("utf-8")
    execute(run, "certification", ["gold_future_capture_certification_v1.py", "--output", str(output),
        "--repo-root", str(ROOT), "--collector", str(ROOT / "gold_future_capture_collector_v1.py"),
        "--source-attestation", str(derived / names[0]), "--timezone-attestation", str(derived / names[1]), "--activation", str(derived / names[2])])
    shutil.copyfile(output, run / "capture_certification_report.json")
    report = history.read_json(output)
    # Major static certification gate fails independently of external evidence.
    verdict = "FAIL" if static["static_status"] != "STATICALLY_CONFORMANT" else "PARTIAL"
    metrics = {"formal_run_status": verdict, "timestamp_diagnostic_status": timestamp["diagnostic_classification"],
        "native_timeframe_audit_status": native["status"], "higher_timeframe_policy": activation["higher_timeframe_policy"],
        "prefix_policy": prefix["prefix_policy"], "prefix_certification_status": prefix["status"],
        "collector_static_status": static["static_status"], "capture_certification_status": report["readiness"]["status"],
        "source_attestation_status": "UNRESOLVED", "timezone_attestation_status": "UNRESOLVED",
        "protocol_frozen": False, "capture_activated": False, "holdout_started": False, "holdout_start": None,
        "strategy_outcome_inspected": False, "model_loaded_for_holdout": False, "model_trained_for_holdout": False,
        "production_changed": False, "production_promoted": False}
    history.write_json(run / "metrics.json", metrics)
    manifest.update(formal_run_status=verdict, protected_sha256_after={name: sha(ROOT / name) for name in protected},
        certification_output_original=str(output), certification_output_sha256=sha(output),
        terminal_build=timestamp.get("terminal_build"), broker_company=timestamp.get("broker_company"),
        broker_server=timestamp.get("broker_server"), symbol="GOLD#", collector_sha256=review["collector_sha256"],
        diagnostic_script_sha256=sha(ROOT / "gold_mt5_timestamp_semantics_diagnostic_v1.py"),
        derived_attestation_sha256={name: sha(derived / name) for name in names},
        manifest_schema_sha256=sha(ROOT / "gold_future_capture_manifest_schema_v1.json"),
        git_dirty_reason="Pre-create source clean. Run creation introduces only this new untracked run directory; certification records repo_dirty=true honestly.")
    if manifest["protected_sha256_after"] != protected:
        history.write_json(run / "manifest.json", manifest)
        raise RuntimeError("Production hash changed; stop")
    data = manifest["data"]
    data.update(symbols=["GOLD#"], data_sources=["Read-only MT5 timing/schema diagnostics; no retained market price arrays"],
        source_files=manifest["input_snapshots"], timezone="UNRESOLVED", raw_snapshot_retained=False,
        reproducibility_claim="Source, timing metadata, docs and exact outputs retained; live samples cannot be refetched identically",
        purge_details="Not applicable: no strategy evaluation", embargo_details="No strategy outcome inspection")
    for key in ("data_start_utc", "data_end_utc", "train_start_utc", "train_end_utc", "validation_start_utc", "validation_end_utc", "test_start_utc", "test_end_utc"):
        data[key] = "NOT_APPLICABLE_NO_STRATEGY_DATASET"
    for key in ("train_rows", "validation_rows", "test_rows"):
        data[key] = 0
    data["mt5_fetch"].update(used=True, terminal_path=spec["terminal_path"],
        terminal_info={"build": timestamp.get("terminal_build"), "diagnostic_error_type": timestamp.get("diagnostic_error_type")},
        broker_info={"company": timestamp.get("broker_company"), "server": timestamp.get("broker_server")},
        fetch_start_utc=timestamp["generated_at_utc"], fetch_end_utc=native["generated_at_utc"],
        retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
        returned_rows={"m1": len(timestamp["m1_sample"]), "native": sum(x["row_count"] for x in native["timeframes"])})
    manifest["model"]["not_applicable_reason"] = "No model loaded or trained"
    manifest["search"]["not_applicable_reason"] = "No candidate or strategy evaluation"
    manifest["registry"].update({key: "N/A: infrastructure only" for key in history.REGISTRY_FIELDS})
    manifest["registry"].update(parent_or_incumbent="GOLD future capture certification v1; prior FAIL preserved",
        selected_configuration="No selection; enablement " + verdict + "; no activation/promotion/production change", validator_result="PENDING")
    history.write_json(run / "manifest.json", manifest)
    (run / "report.md").write_text("# GOLD future capture enablement v1\n\nFormal verdict: " + verdict
        + "\n\nMethodology and independent validator: see validator.json.\n\n"
        + "Timestamp: " + timestamp["diagnostic_classification"] + "; native: " + native["status"] + "; prefix: " + prefix["prefix_policy"]
        + "\n\nStorage self-tests pass, including frozen verifier chain compatibility. Static certification fails because the unchanged checker rejects mutable tip replacement; no bypass or rule relaxation. Source ID also violates frozen regex. Calendar and exact recursive prefix remain uncertified.\n"
        + "\nNo protocol freeze, capture activation, holdout start, strategy inference, model training, production change or promotion.\n"
        + "\nOriginal report is retained outside training_runs to preserve the certifier output guard; the archived copy is byte-identical.\n", encoding="utf-8")
    (run / "stdout.txt").write_text("Formal run completed: " + verdict + "\n", encoding="utf-8")
    (run / "stderr.txt").touch()
    shutil.copyfile(run / "stdout.txt", run / "stdout.log")
    print("FORMAL_RUN_STATUS=" + verdict)


if __name__ == "__main__":
    main()
