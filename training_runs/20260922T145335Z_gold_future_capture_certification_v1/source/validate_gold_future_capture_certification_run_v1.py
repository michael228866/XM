"""Independent, one-shot validation of the phase-1-blocked certification archive.

Does not import the certifier or collector, contact MT5, or evaluate a strategy.
PASS means the blocked result is faithfully preserved, not activation readiness.
"""
import argparse
import ast
import hashlib
import json
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_HASH = "26d158735af2fffff3bbebad87cb93c463bb35e89405dec9f92e1f51c6409a38"
SPEC_HASH = "e9f489af6294076b2986965672993b8941aab7d28ba63557d26ba3f93334a370"
EXPECTED_BLOCKERS = {"NOT_READY_SOURCE_BINDING", "NOT_READY_TIMEZONE",
    "NOT_READY_HIGHER_TIMEFRAMES", "NOT_READY_PREFIX_POLICY", "NOT_READY_COLLECTOR",
    "NOT_READY_CAPTURE_INTEGRITY", "NOT_READY_PROTOCOL_FREEZE"}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode())


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def check_result(report, source, tz, activation):
    """Reconstruct why incomplete inputs cannot certify an activation."""
    return {
        "source_identity": source["source_id"] == "XMGlobal-MT5-6_GOLD#"
            and source["symbol"] == source["symbol_exact_case"] == "GOLD#",
        "source_incomplete": source["template"] is True
            and source["source_account_environment"] == "unknown"
            and report["source_binding"]["status"] == "NOT_CERTIFIED",
        "timezone_unresolved": tz["template"] is True and tz["status"] == "UNRESOLVED"
            and report["timezone_certification"]["status"] == "UNRESOLVED",
        "attestation_links": tz["source_attestation_sha256"] == canonical(source)
            == activation["source_attestation_sha256"]
            and activation["timezone_attestation_sha256"] == canonical(tz),
        "higher_timeframes": activation["higher_timeframe_policy"] == "UNRESOLVED"
            == report["higher_timeframe_policy"]["policy"]
            and activation["native_timeframes"] == []
            and activation["protocol_change_required"] is False,
        "prefix": activation["prefix_policy"] == "UNRESOLVED"
            == report["prefix_policy"]["policy"] and activation["prefix_evidence"] == [],
        "activation": activation["template"] is True and activation["status"] == "NOT_ACTIVATED"
            and activation["activation_effective_at_utc"] is None
            and activation["protocol_freeze_commit"] is None
            and report["capture_integrity"]["activation_status"] == "NOT_ACTIVATED",
        "collector_static": report["collector_certification"]["static_status"] == "NOT_STATICALLY_CONFORMANT",
        "capture_readiness": report["readiness"]["status"] == "NOT_READY_MULTIPLE_BLOCKERS"
            and {item["status"] for item in report["readiness"]["blockers"]} == EXPECTED_BLOCKERS,
        "capture_not_certified": report["capture_integrity"]["chain_valid"] is None
            and report["capture_integrity"]["append_only_certified"] is False
            and report["capture_integrity"]["attestation_hashes_valid"] is False,
        "historical_not_untouched": report["historical_context"]["historical_2025_plus_is_untouched"] is False
            and all(doc["historical_2025_plus_is_untouched"] is False for doc in (source, tz, activation)),
        "no_holdout": report["holdout_boundary"]["holdout_start"] is None
            and report["holdout_boundary"]["earliest_possible_holdout_start"] is None
            and all(doc["holdout_start"] is None and doc["earliest_possible_holdout_start"] is None
                    for doc in (source, tz, activation)),
        "no_promotion": report["production_promotion"] is False and report["production_change"] is False
            and all(doc["production_promotion"] is False for doc in (source, tz, activation)),
        "no_strategy_permission": all(doc["outcome_inspection_prohibited"] is True
            and doc["strategy_execution_prohibited"] is True for doc in (source, tz, activation)),
    }


def confined(status, run_relative):
    return all(entry == "?? " + run_relative + "/" or
               (entry.startswith("?? " + run_relative + "/") and ".." not in entry.split("/"))
               for entry in status.split("\0") if entry)


def validate(run):
    root = run.parent.parent
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root)
    manifest = load(run / "manifest.json")
    report = load(run / "certification_report.json")
    commit = manifest["source_commit"]
    source_root = run / "source"
    source = load(source_root / "gold_future_capture_source_attestation_v1.json")
    tz = load(source_root / "gold_future_capture_timezone_attestation_v1.json")
    activation = load(source_root / "gold_future_capture_activation_v1.json")
    checks = check_result(report, source, tz, activation)
    checks["repo_commit"] = commit == manifest["git_commit"] == report["repo_commit"] == git("rev-parse", "HEAD").decode().strip()
    checks["remote_source"] = commit == manifest["pre_run_remote_commit"] == git("ls-remote", "origin", "refs/heads/main").decode().split()[0]
    relative = run.relative_to(root).as_posix()
    checks["clean_source_provenance"] = (manifest["git_dirty"] is False
        and manifest["pre_create_git_status"] == ""
        and confined(manifest["git_status_at_certification"], relative)
        and confined(git("status", "--porcelain", "-z").decode("utf-8"), relative)
        and report["repo_dirty"] == bool(manifest["git_status_at_certification"]))
    expected_inputs = {"gold_future_capture_certification_v1.py", "training_run_history.py",
        "run_gold_future_capture_certification_v1.py", "validate_gold_future_capture_certification_run_v1.py",
        "gold_data_foundation_forward_collector.py", "gold_data_foundation1_audit.py",
        "execution_spec_gold_future_capture_certification_v1.json", "gold_future_capture_manifest_schema_v1.json",
        "gold_future_capture_source_attestation_v1.json", "gold_future_capture_timezone_attestation_v1.json",
        "gold_future_capture_activation_v1.json", "gold_future_capture_source_attestation_template_v1.json",
        "gold_future_capture_timezone_attestation_template_v1.json", "gold_future_capture_activation_template_v1.json"}
    checks["input_inventory"] = expected_inputs <= {item["source_path"] for item in manifest["input_snapshots"]}
    for item in manifest["input_snapshots"]:
        path = run / item["path"]
        original = root / item["source_path"]
        if not path.resolve().is_relative_to(source_root.resolve()) or not original.resolve().is_relative_to(root):
            raise ValueError("Input path escapes archive")
        raw = path.read_bytes()
        blob = git("cat-file", "blob", commit + ":" + item["source_path"])
        checks["input_hash:" + item["source_path"]] = digest(raw) == item["sha256"] == digest(original.read_bytes()) and raw.replace(b"\r\n", b"\n") == blob.replace(b"\r\n", b"\n")
    checks["executed_script"] = (run / "training_script.py").read_bytes() == (source_root / "gold_future_capture_certification_v1.py").read_bytes()
    checks["executed_script_hash"] = digest((run / "training_script.py").read_bytes()) == manifest["training_script_sha256"]
    checks["validator_snapshot"] = (run / "validator_script.py").read_bytes() == (source_root / "validate_gold_future_capture_certification_run_v1.py").read_bytes() == Path(__file__).read_bytes()
    spec = load(run / "execution_spec.json")
    schema = load(source_root / "gold_future_capture_manifest_schema_v1.json")
    checks["execution_spec"] = canonical(spec) == SPEC_HASH and (run / "execution_spec.json").read_bytes() == (source_root / "execution_spec_gold_future_capture_certification_v1.json").read_bytes()
    checks["manifest_schema"] = canonical(schema) == SCHEMA_HASH and schema["additionalProperties"] is False and set(schema["required"]) == set(schema["properties"])
    for doc in (source, tz):
        for item in doc["evidence"]:
            checks["evidence_hash:" + item["local_path"]] = digest((source_root / item["local_path"]).read_bytes()) == item["sha256"]
    collector = (source_root / "gold_data_foundation_forward_collector.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(collector)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    checks["collector_independent_defects"] = (any(isinstance(n.func, ast.Attribute) and n.func.attr == "replace" for n in calls)
        and any(isinstance(n.func, ast.Attribute) and n.func.attr == "open" and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "a" for n in calls)
        and not any(isinstance(n.func, ast.Attribute) and n.func.attr == "fsync" for n in calls))
    checks["report_copy"] = (run / "certification_report.json").read_bytes() == Path(manifest["certification_output_original"]).read_bytes()
    checks["execution_success"] = manifest["certification_exit_code"] == 0 and (run / "stderr.txt").read_bytes() == b"" and (run / "stdout.txt").read_text().strip() == report["readiness"]["status"]
    checks["research_fail"] = manifest["research_verdict"] == "FAIL" and load(run / "metrics.json")["research_verdict"] == "FAIL"
    checks["no_models_or_outcomes"] = all(manifest[k] is False for k in ("capture_activated", "strategy_outcome_inspected", "model_loaded_for_holdout", "model_trained_for_holdout", "production_change")) and manifest["model"]["trained"] is False and manifest["search"]["performed"] is False
    checks["production_unchanged"] = manifest["protected_sha256_before"] == manifest["protected_sha256_after"] == {name: digest((root / name).read_bytes()) for name in manifest["protected_sha256_before"]}
    return checks


def self_test():
    assert confined("?? training_runs/example/\0", "training_runs/example")
    assert not confined(" M gemini.py\0", "training_runs/example")
    assert not confined("?? training_runs/example/../other\0", "training_runs/example")
    assert canonical({"b": 2, "a": 1}) == canonical({"a": 1, "b": 2})
    assert digest(b"x") != digest(b"y")
    # Load only inert release/preparation documents, never outcomes or market files.
    root = Path(__file__).resolve().parent
    source = load(root / "gold_future_capture_source_attestation_v1.json")
    tz = load(root / "gold_future_capture_timezone_attestation_v1.json")
    activation = load(root / "gold_future_capture_activation_v1.json")
    report = {"source_binding": {"status": "NOT_CERTIFIED"},
        "timezone_certification": {"status": "UNRESOLVED"},
        "higher_timeframe_policy": {"policy": "UNRESOLVED"}, "prefix_policy": {"policy": "UNRESOLVED"},
        "capture_integrity": {"activation_status": "NOT_ACTIVATED", "chain_valid": None,
            "append_only_certified": False, "attestation_hashes_valid": False},
        "collector_certification": {"static_status": "NOT_STATICALLY_CONFORMANT"},
        "readiness": {"status": "NOT_READY_MULTIPLE_BLOCKERS", "blockers": [{"status": s} for s in EXPECTED_BLOCKERS]},
        "historical_context": {"historical_2025_plus_is_untouched": False},
        "holdout_boundary": {"holdout_start": None, "earliest_possible_holdout_start": None},
        "production_promotion": False, "production_change": False}
    assert all(check_result(report, source, tz, activation).values())
    for section, key, value in [("readiness", "status", "READY_FOR_CAPTURE_ACTIVATION"),
            ("historical_context", "historical_2025_plus_is_untouched", True),
            ("holdout_boundary", "holdout_start", "2030-01-01T00:00:00Z"),
            ("capture_integrity", "append_only_certified", True)]:
        bad = deepcopy(report)
        bad[section][key] = value
        assert not all(check_result(bad, source, tz, activation).values())
    print("SELF_TEST_PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.run:
            parser.error("Self-test cannot inspect a run")
        self_test()
        return 0
    if args.run is None:
        parser.error("Run required")
    run = args.run.resolve()
    if (run / "FINALIZED.json").exists():
        raise RuntimeError("Do not modify finalized runs")
    with (run / "validator_attempt.json").open("x", encoding="utf-8") as handle:
        json.dump({"started_at_utc": datetime.now(timezone.utc).isoformat(), "one_shot": True}, handle)
    try:
        checks = validate(run)
    except Exception as error:
        checks = {"validation_exception:" + type(error).__name__: False}
    failed = [name for name, passed in checks.items() if not passed]
    result = {"overall": "FAIL" if failed else "PASS", "failed_check_names": failed,
              "checks": checks, "research_verdict": "FAIL", "production_promotion": False}
    with (run / "validator.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    with (run / "validator.md").open("x", encoding="utf-8") as handle:
        handle.write("# Independent certification archive validation\n\nMethodology/provenance: " + result["overall"]
                     + "\nIndependent validator: " + result["overall"]
                     + "\nResearch: FAIL; no activation or production promotion.\n"
                     + "\n".join("- " + name for name in failed) + "\n")
    print("overall=" + result["overall"])
    print("failed_check_names=" + json.dumps(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
