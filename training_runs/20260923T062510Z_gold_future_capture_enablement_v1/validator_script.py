"""Independent archived-artifact validator. No collector/diagnostic imports."""
import argparse
import ast
import calendar
import hashlib
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROTECTED = {"gemini.py": "0ccb4a66c54981e3b207e0f20db1ca64a3f8d76ebe8a74784d1b9b6102fc4b07",
             "gold_long_recent_candidate_xgb.json": "2dc32e3b3c0ea6ca8fa2e30187bebf8ff3f7e7e03109b39b3f70f013e3a755f2"}
FRAMES = {"M1", "Daily", "H12", "H8", "H6", "H4", "H3", "H2", "H1", "M30", "M20", "M15",
          "M12", "M10", "M6", "M5", "M4", "M3", "M2", "Weekly", "Monthly"}


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode())


def epoch(text):
    return datetime.fromisoformat(text).timestamp()


def legacy_epoch(value):
    for offset in (7200, 10800):
        candidate = datetime.fromtimestamp(value - offset, timezone.utc)
        boundaries = []
        for month in (3, 10):
            end = datetime(candidate.year, month, calendar.monthrange(candidate.year, month)[1], 1, tzinfo=timezone.utc)
            boundaries.append(end - timedelta(days=(end.weekday() + 1) % 7))
        if offset == (10800 if boundaries[0] <= candidate < boundaries[1] else 7200):
            return value - offset
    raise ValueError("Unmapped legacy hypothesis")


def timing_valid(doc):
    tick = doc["tick_sample"]
    if tick is None:
        return doc["diagnostic_classification"] == "UNRESOLVED" and bool(doc.get("diagnostic_error_type")) and doc["m1_sample"] == []
    system = epoch(doc["system_current_utc"])
    t = tick["raw_time"]
    if epoch(tick["interpreted_as_utc"]) != t or abs(tick["delta_from_system_utc_seconds"] - (t - system)) > 1e-6:
        return False
    if epoch(tick["legacy_eet_eest_converted"]) != legacy_epoch(t) or abs(tick["legacy_delta_from_system_utc_seconds"] - (legacy_epoch(t) - system)) > 1e-6:
        return False
    rows = doc["m1_sample"]
    if not 1 <= len(rows) <= 5:
        return False
    for row in rows:
        if epoch(row["interpreted_as_utc"]) != row["raw_epoch"] or epoch(row["legacy_eet_eest_converted"]) != legacy_epoch(row["raw_epoch"]) or row["legacy_minus_direct_seconds"] != legacy_epoch(row["raw_epoch"]) - row["raw_epoch"]:
            return False
    epochs = [r["raw_epoch"] for r in rows]
    spacing = [b - a for a, b in zip(epochs, epochs[1:])]
    if spacing != doc["bar_spacing_seconds"] or doc["timestamp_monotonic"] != all(x > 0 for x in spacing):
        return False
    if tick["raw_time_msc"] // 1000 != t or t <= 0 or any(x <= 0 or x % 60 for x in spacing) or any(x % 60 for x in epochs):
        expected = "INCONSISTENT"
    elif not 0 <= t - epochs[-1] <= 180:
        expected = "UNRESOLVED"
    elif abs(t - system) <= 180 and abs(legacy_epoch(t) - system) > 180:
        expected = "DIRECT_UTC_SEMANTICS_SUPPORTED"
    elif abs(legacy_epoch(t) - system) <= 180 and abs(t - system) > 180:
        expected = "LEGACY_CONVERSION_SUPPORTED"
    else:
        expected = "UNRESOLVED"
    if "diagnostic_error_type" in doc:
        expected = "UNRESOLVED"
    query = doc.get("direct_utc_range_query")
    return doc["diagnostic_classification"] == expected and (query is None or epoch(query["end_utc"]) - epoch(query["start_utc"]) == 300)


def native_valid(doc):
    rows = doc["timeframes"]
    if len(rows) != 21 or {r["timeframe"] for r in rows} != FRAMES:
        return False
    for row in rows:
        epochs = row["raw_epochs"]
        if len(epochs) != row["row_count"] or not 0 <= len(epochs) <= 3 or row["source_symbol"] != "GOLD#":
            return False
        if epochs:
            if epoch(row["first_timestamp"]) != epochs[0] or epoch(row["last_timestamp"]) != epochs[-1] or row["monotonic"] != all(b > a for a, b in zip(epochs, epochs[1:])):
                return False
        if row["status"] == "AVAILABLE_NATIVE":
            if not (row["api_supported"] and row["sample_returned"] and row["spread_available"] and epochs and row["monotonic"] and {"time", "open", "high", "low", "close", "spread", "tick_volume", "real_volume"} <= set(row["schema"])):
                return False
        elif row["status"] not in {"API_CONSTANT_MISSING", "NO_DATA_RETURNED", "SCHEMA_MISMATCH", "UNRESOLVED"}:
            return False
    available = all(r["status"] == "AVAILABLE_NATIVE" for r in rows)
    return doc["status"] == ("PASS" if available else "PARTIAL") and doc["candidate_policy"] == ("NATIVE_20TF_REQUIRED" if available else "UNRESOLVED")


def safe_output(value):
    banned = {"login", "account_number", "name", "balance", "equity", "margin", "password", "credentials", "token",
              "open", "high", "low", "close", "bid", "ask", "profit", "pnl", "win_rate", "probability", "prediction"}
    if isinstance(value, dict):
        return not banned.intersection(value) and all(safe_output(v) for v in value.values())
    if isinstance(value, list):
        return all(safe_output(v) for v in value)
    return True


def validate(run):
    root = run.parent.parent
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root)
    m = load(run / "manifest.json")
    spec = load(run / "execution_spec.json")
    commit = m["source_commit"]
    checks = {"source_commit": commit == m["git_commit"] == git("rev-parse", "HEAD").decode().strip(),
        "remote_source_commit": commit == m["pre_run_remote_commit"] == git("ls-remote", "origin", "refs/heads/main").decode().split()[0],
        "clean_before_create": m["git_dirty"] is False and m["pre_run_clean"] is True and m["pre_run_git_status"] == ""}
    prefix_path = run.relative_to(root).as_posix() + "/"
    statuses = [git("status", "--porcelain", "-z").decode("utf-8"), m["git_status_at_certification"]]
    checks["source_unchanged_during_run"] = all(all(entry.startswith("?? " + prefix_path) for entry in status.split("\0") if entry) for status in statuses)
    checks["source_inventory"] = set(spec["source_files"]) <= {i["source_path"] for i in m["input_snapshots"]}
    for item in m["input_snapshots"]:
        path = (run / item["path"]).resolve()
        if not path.is_relative_to((run / "source").resolve()):
            raise ValueError("Source snapshot path escape")
        raw = path.read_bytes()
        blob = git("cat-file", "blob", commit + ":" + item["source_path"])
        checks["snapshot:" + item["source_path"]] = sha(raw) == item["sha256"] == sha((root / item["source_path"]).read_bytes()) and raw.replace(b"\r\n", b"\n") == blob.replace(b"\r\n", b"\n")
    checks["executed_source"] = sha((run / "training_script.py").read_bytes()) == m["training_script_sha256"] and (run / "training_script.py").read_bytes() == (run / "source/run_gold_future_capture_enablement_v1.py").read_bytes()
    checks["validator_source"] = Path(__file__).read_bytes() == (run / "validator_script.py").read_bytes() == (run / "source/validate_gold_future_capture_enablement_v1.py").read_bytes()
    checks["spec_snapshot"] = (run / "execution_spec.json").read_bytes() == (run / "source/execution_spec_gold_future_capture_enablement_v1.json").read_bytes()
    checks["production_hashes"] = m["protected_sha256_before"] == m["protected_sha256_after"] == PROTECTED == {name: sha((root / name).read_bytes()) for name in PROTECTED}
    checks["production_diff"] = not git("diff", "5b9a7329754ca6ae2bb9ac4b0d52748c63e844f4", "--", *PROTECTED)
    timestamp, native, prefix = [load(run / name) for name in ("diagnostic_timestamp.json", "native_timeframe_audit.json", "prefix_certification.json")]
    checks["timestamp_internal_consistency"] = timing_valid(timestamp)
    checks["native_timeframe_consistency"] = native_valid(native)
    checks["no_sensitive_or_price_outputs"] = all(safe_output(doc) for doc in (timestamp, native, prefix))
    checks["safe_live_identity"] = timestamp.get("broker_company", "XM Global Limited") == "XM Global Limited" and timestamp.get("broker_server", "XMGlobal-MT5 6") == "XMGlobal-MT5 6" and timestamp.get("source_account_environment", "unknown") in {"demo", "live", "unknown"}
    checks["prefix_honesty"] = prefix["prefix_policy"] == "UNRESOLVED" and prefix["status"] == "PARTIAL" and all(prefix[k] is False for k in ("continuous_history", "same_timestamp_semantics", "same_timeframe_semantics", "same_feature_initialization", "recursive_state_equivalent")) and prefix["required_history_start"] is None and prefix["available_history_start"] is None
    checks["prefix_pipeline_hash"] = prefix["pipeline_sha256"] == sha((run / "source/drl_trading_v2.py").read_bytes())
    docs = [load(run / "attestations" / name) for name in ("gold_future_capture_source_attestation_v1.json", "gold_future_capture_timezone_attestation_v1.json", "gold_future_capture_activation_v1.json")]
    source, tz, activation = docs
    checks["attestation_links"] = tz["source_attestation_sha256"] == canonical(source) == activation["source_attestation_sha256"] and activation["timezone_attestation_sha256"] == canonical(tz)
    checks["attestation_unresolved"] = all(doc["template"] is True for doc in docs) and tz["status"] == "UNRESOLVED" and source["source_account_environment"] == timestamp.get("source_account_environment", "unknown")
    checks["source_id_not_silently_changed"] = source["source_id"] == spec["source_id"] == "XMGlobal-MT5-6_GOLD#"
    for name, expected in m["derived_attestation_sha256"].items():
        checks["derived_attestation_hash:" + name] = sha((run / "attestations" / name).read_bytes()) == expected
    for item in source["evidence"]:
        path = Path(item["local_path"])
        checks["evidence:" + path.name] = path.resolve().is_relative_to((run / "source").resolve()) and sha(path.read_bytes()) == item["sha256"]
    schema = load(run / "source/gold_future_capture_manifest_schema_v1.json")
    checks["frozen_schema"] = canonical(schema) == "26d158735af2fffff3bbebad87cb93c463bb35e89405dec9f92e1f51c6409a38" and sha((run / "source/gold_future_capture_manifest_schema_v1.json").read_bytes()) == m["manifest_schema_sha256"]
    raw_collector = (run / "source/gold_future_capture_collector_v1.py").read_bytes()
    tree = ast.parse(raw_collector)
    mutable_tip = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "replace" for n in ast.walk(tree))
    static = load(run / "collector_static_review.json")
    checks["collector_static_gate_honesty"] = mutable_tip and static["static_status"] == "NOT_STATICALLY_CONFORMANT" and any("Mutable storage" in s for s in static["blockers"])
    checks["collector_hash"] = static["collector_sha256"] == m["collector_sha256"] == sha(raw_collector)
    checks["no_strategy_calls"] = not any(isinstance(n, ast.Attribute) and n.attr in {"fit", "predict", "predict_proba", "order_send", "order_check"} for n in ast.walk(tree))
    report = load(run / "capture_certification_report.json")
    metrics = load(run / "metrics.json")
    checks["certification_report"] = report["repo_commit"] == commit and report["repo_dirty"] == bool(m["git_status_at_certification"]) and report["readiness"]["status"] == metrics["capture_certification_status"] == "NOT_READY_MULTIPLE_BLOCKERS"
    checks["report_copy"] = sha((run / "capture_certification_report.json").read_bytes()) == m["certification_output_sha256"] == sha(Path(m["certification_output_original"]).read_bytes())
    checks["verdict"] = m["formal_run_status"] == metrics["formal_run_status"] == "FAIL" and metrics["collector_static_status"] == static["static_status"]
    checks["no_holdout_or_promotion"] = metrics["holdout_start"] is None and report["holdout_boundary"]["holdout_start"] is None and all(metrics[k] is False for k in ("protocol_frozen", "capture_activated", "holdout_started", "strategy_outcome_inspected", "model_loaded_for_holdout", "model_trained_for_holdout", "production_changed", "production_promoted")) and report["production_promotion"] is False and report["production_change"] is False and report["historical_context"]["historical_2025_plus_is_untouched"] is False
    checks["closed_activation_gate"] = spec["activation_permitted"] is False and activation["status"] == "NOT_ACTIVATED" and activation["protocol_freeze_commit"] is None
    commands = [json.loads(line) for line in (run / "commands.jsonl").read_text().splitlines()]
    checks["exact_commands_retained"] = {r["label"] for r in commands} == {"collector_self_test", "timestamp", "native", "prefix", "certification"} and all(r["exit_code"] == 0 and (run / (r["label"] + "_stderr.txt")).read_bytes() == b"" for r in commands)
    checks["collector_self_tests"] = load(run / "collector_self_test_stdout.txt")["overall"] == "PASS"
    if native["status"] == "PASS":
        checks["higher_timeframe_policy"] = activation["higher_timeframe_policy"] == metrics["higher_timeframe_policy"] == "NATIVE_20TF_REQUIRED"
    else:
        checks["higher_timeframe_policy"] = activation["higher_timeframe_policy"] == "M1_DETERMINISTIC_RESAMPLE_PROPOSED" and activation["protocol_change_required"] is True and (run / "resampling_proposal.md").is_file()
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        assert not safe_output({"login": "synthetic"})
        assert not safe_output({"nested": [{"probability": 0.5}]})
        assert safe_output({"source_account_environment": "demo", "raw_epoch": 100})
        assert not native_valid({"timeframes": []})
        assert timing_valid({"tick_sample": None, "m1_sample": [], "diagnostic_classification": "UNRESOLVED", "diagnostic_error_type": "RuntimeError"})
        assert not timing_valid({"tick_sample": None, "m1_sample": [], "diagnostic_classification": "DIRECT_UTC_SEMANTICS_SUPPORTED"})
        assert legacy_epoch(1593604800) == 1593604800 - 10800
        print("SELF_TEST_PASS")
        return 0
    if args.run is None:
        parser.error("Run required")
    run = args.run.resolve()
    if (run / "FINALIZED.json").exists():
        raise RuntimeError("Finalized run immutable")
    with (run / "validator_attempt.json").open("x", encoding="utf-8") as handle:
        json.dump({"started_at_utc": datetime.now(timezone.utc).isoformat(), "one_shot": True}, handle)
    try:
        checks = validate(run)
    except Exception as error:
        checks = {"exception:" + type(error).__name__: False}
    failed = [key for key, passed in checks.items() if not passed]
    result = {"overall": "FAIL" if failed else "PASS", "failed_check_names": failed, "checks": checks,
              "research_verdict": "FAIL", "production_promotion": False}
    with (run / "validator.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    with (run / "validator.md").open("x", encoding="utf-8") as handle:
        handle.write("# Independent enablement validation\n\nMethodology/provenance: " + result["overall"]
            + "\nIndependent validator: " + result["overall"] + "\nEnablement: FAIL; no activation/promotion.\n")
    print("overall=" + result["overall"])
    print("failed_check_names=" + json.dumps(failed))
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
