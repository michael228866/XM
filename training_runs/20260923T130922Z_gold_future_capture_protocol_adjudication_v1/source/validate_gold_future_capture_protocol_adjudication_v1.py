"""Independent offline archive and synthetic-chain verifier. No execution imports."""
import argparse
import ast
import csv
import hashlib
import io
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SOURCE_ID = "XMGlobal-MT5-6_GOLD"
PROTECTED = {"gemini.py": "0ccb4a66c54981e3b207e0f20db1ca64a3f8d76ebe8a74784d1b9b6102fc4b07",
             "gold_long_recent_candidate_xgb.json": "2dc32e3b3c0ea6ca8fa2e30187bebf8ff3f7e7e03109b39b3f70f013e3a755f2"}
FIELDS = ["SOURCE_TIMESTAMP", "ORIGINAL_SOURCE_TIMESTAMP", "SOURCE_ID", "SYMBOL", "TIMEFRAME", "OPEN", "HIGH", "LOW", "CLOSE", "SPREAD"]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(obj):
    return sha(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode())


def utc(text):
    value = datetime.fromisoformat(text)
    require(value.tzinfo is not None and value.utcoffset().total_seconds() == 0, "Non-UTC time")
    return value


def tip_status(entries, tip):
    if tip is None:
        return "MISSING"
    if not isinstance(tip, dict) or set(tip) != {"manifest_sequence", "manifest_sha256", "updated_at_utc"}:
        return "CONFLICTING"
    n = tip["manifest_sequence"]
    if type(n) is not int or not 0 <= n < len(entries):
        return "CONFLICTING"
    if tip["manifest_sha256"] != canonical(entries[n]) or tip["updated_at_utc"] != entries[n]["manifest_created_at_utc"]:
        return "CONFLICTING"
    return "MATCH" if n == len(entries) - 1 else "STALE"


def schema_check(item, schema):
    require(set(item) == set(schema["required"]), "Manifest fields changed")
    for name, rule in schema["properties"].items():
        value = item[name]
        if "const" in rule:
            require(type(value) is type(rule["const"]) and value == rule["const"], "Manifest constant")
        if "enum" in rule:
            require(value in rule["enum"], "Manifest enum")
        if "type" in rule:
            kinds = rule["type"] if isinstance(rule["type"], list) else [rule["type"]]
            require(any((value is None and k == "null") or (type(value) is int and k == "integer")
                        or (isinstance(value, str) and k == "string") for k in kinds), "Manifest type")
        if value is not None:
            if "pattern" in rule:
                require(re.fullmatch(rule["pattern"], value), "Manifest pattern")
            if "minimum" in rule:
                require(value >= rule["minimum"], "Manifest minimum")
            if "minLength" in rule:
                require(len(value) >= rule["minLength"], "Manifest text")
            if rule.get("format") == "date-time":
                utc(value)


def verify_chain(root, schema):
    source, tz, activation = [load(root / name) for name in ("source_attestation.json", "timezone_attestation.json", "activation.json")]
    require(all(doc["notes"] == "SYNTHETIC TEST ONLY" for doc in (source, tz, activation)), "Fixture cannot be live evidence")
    require(source["source_id"] == SOURCE_ID and source["symbol"] == "GOLD#", "Synthetic source mismatch")
    require(tz["source_attestation_sha256"] == canonical(source) == activation["source_attestation_sha256"]
            and activation["timezone_attestation_sha256"] == canonical(tz), "Synthetic attestation linkage")
    require(tz["timezone_status"] == "CERTIFIED_UTC" and tz["mt5_epoch_semantics"] == "ALREADY_UTC", "Only synthetic UTC fixtures accepted")
    columns = FIELDS + source["volume_fields_if_present"]
    expected_schema = canonical({"format": "gold_canonical_bars_v1", "encoding": "UTF-8", "delimiter": ",", "columns": columns})
    entries, by_id, paths, last_by_tf = [], {}, set(), {}
    previous = None
    for n, path in enumerate(sorted((root / "manifests").iterdir())):
        require(path.name == f"{n:012d}.json", "Sequence filename gap")
        item = load(path)
        schema_check(item, schema)
        require(item["sequence"] == item["snapshot_sequence"] == n and item["previous_manifest_sha256"] == previous, "Broken full chain")
        require(item["source_id"] == SOURCE_ID and item["symbol"] == "GOLD#" and item["collector_version"] == "gold_future_capture_collector_v2"
                and item["collector_commit"] == activation["collector_commit"], "Manifest source/code mismatch")
        for key, doc in (("source_attestation_sha256", source), ("timezone_attestation_sha256", tz), ("activation_sha256", activation)):
            require(item[key] == canonical(doc), "Manifest attestation hash")
        require(item["snapshot_id"] not in by_id and item["snapshot_path"] not in paths, "Duplicate snapshot")
        snapshot = (root / item["snapshot_path"]).resolve()
        require(snapshot.is_relative_to((root / "snapshots").resolve()), "Snapshot path escape")
        raw = snapshot.read_bytes()
        require(sha(raw) == item["raw_sha256"] and item["schema_sha256"] == expected_schema, "Raw/schema hash mismatch")
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
        require(reader.fieldnames == columns, "Raw schema/outcome field")
        rows = list(reader)
        require(len(rows) == item["row_count"] > 0, "Row count")
        timestamps = []
        nonpositive = 0
        for row in rows:
            require(None not in row and all(v is not None for v in row.values()), "Malformed row")
            require(row["SOURCE_ID"] == SOURCE_ID and row["SYMBOL"] == "GOLD#" and row["TIMEFRAME"] == item["timeframe"], "Row identity")
            stamp = utc(row["SOURCE_TIMESTAMP"])
            require(stamp.timestamp() == int(row["ORIGINAL_SOURCE_TIMESTAMP"]), "Raw/UTC mismatch")
            timestamps.append(stamp)
            prices = [float(row[key]) for key in ("OPEN", "HIGH", "LOW", "CLOSE")]
            require(all(math.isfinite(v) and v > 0 for v in prices) and prices[2] <= min(prices) and prices[1] >= max(prices), "Synthetic raw quality")
            spread = float(row["SPREAD"])
            require(math.isfinite(spread), "Missing spread")
            nonpositive += int(spread <= 0)
        require(all(b > a for a, b in zip(timestamps, timestamps[1:])), "Duplicate/reversed rows")
        require(utc(item["first_source_timestamp"]) == timestamps[0] and utc(item["last_source_timestamp"]) == timestamps[-1], "Timestamp bounds")
        require(item["spread_nonpositive_rows"] == nonpositive and all(item[key] == 0 for key in
            ("duplicate_timestamps", "non_monotonic_timestamps", "invalid_timestamp_rows", "invalid_price_rows", "spread_missing_rows")), "Quality counters")
        require(timestamps[-1] <= utc(item["capture_finished_at_utc"]) <= utc(item["manifest_created_at_utc"])
                and utc(item["capture_started_at_utc"]) <= utc(item["capture_finished_at_utc"]), "Capture chronology")
        if entries:
            require(utc(item["capture_started_at_utc"]) > utc(entries[-1]["capture_finished_at_utc"])
                    and utc(item["manifest_created_at_utc"]) > utc(entries[-1]["manifest_created_at_utc"]), "Chain clock reversal")
        revision = item["revision_of_snapshot_id"]
        if revision:
            require(revision in by_id, "Orphan revision")
            old = by_id[revision]
            require(all(item[k] == old[k] for k in ("timeframe", "row_count", "first_source_timestamp", "last_source_timestamp"))
                    and item["raw_sha256"] != old["raw_sha256"], "Invalid revision")
        else:
            require(item["timeframe"] not in last_by_tf or timestamps[0] > last_by_tf[item["timeframe"]], "Duplicate interval")
            last_by_tf[item["timeframe"]] = timestamps[-1]
        by_id[item["snapshot_id"]] = item
        paths.add(item["snapshot_path"])
        entries.append(item)
        previous = canonical(item)
    require(entries and {p.relative_to(root).as_posix() for p in (root / "snapshots").iterdir()} == paths, "Orphan snapshots")
    require(tip_status(entries, load(root / "chain_tip.json")) == "MATCH", "Current index mismatch")
    cases = load(root / "synthetic_cases.json")
    require(cases["synthetic_only"] is True, "Synthetic cases label")
    require([tip_status(entries, cases[name]) for name in ("missing", "stale", "conflicting", "matching")]
            == ["MISSING", "STALE", "CONFLICTING", "MATCH"], "Independent index case recomputation")
    return len(entries)


def validate(run):
    root = run.parent.parent
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root)
    m = load(run / "manifest.json")
    spec = load(run / "execution_spec.json")
    source_root = run / "source"
    protocol = load(source_root / "gold_future_capture_protocol_v2.json")
    commit = m["source_commit"]
    checks = {"source_commit": commit == m["git_commit"] == git("rev-parse", "HEAD").decode().strip(),
        "remote_source_commit": commit == m["pre_run_remote_commit"] == git("ls-remote", "origin", "refs/heads/main").decode().split()[0],
        "clean_pre_run": m["git_dirty"] is False and m["pre_run_clean"] is True and m["pre_run_git_status"] == ""}
    prefix = "?? " + run.relative_to(root).as_posix() + "/"
    checks["source_stayed_clean"] = all(all(entry.startswith(prefix) for entry in value.split("\0") if entry) for value in
        (m["git_status_at_certification"], git("status", "--porcelain", "-z").decode("utf-8")))
    checks["source_inventory"] = set(spec["source_files"]) == {item["source_path"] for item in m["input_snapshots"]}
    for item in m["input_snapshots"]:
        path = (run / item["path"]).resolve()
        require(path.is_relative_to(source_root.resolve()), "Source path escape")
        raw = path.read_bytes()
        blob = git("cat-file", "blob", commit + ":" + item["source_path"])
        checks["source:" + item["source_path"]] = sha(raw) == item["sha256"] == sha((root / item["source_path"]).read_bytes()) and raw.replace(b"\r\n", b"\n") == blob.replace(b"\r\n", b"\n")
    checks["validator_snapshot"] = Path(__file__).read_bytes() == (run / "validator_script.py").read_bytes() == (source_root / "validate_gold_future_capture_protocol_adjudication_v1.py").read_bytes()
    checks["executed_script_snapshot"] = sha((run / "training_script.py").read_bytes()) == m["training_script_sha256"] and (run / "training_script.py").read_bytes() == (source_root / "run_gold_future_capture_protocol_adjudication_v1.py").read_bytes()
    checks["protected_hashes"] = m["protected_sha256_before"] == m["protected_sha256_after"] == PROTECTED == {name: sha((root / name).read_bytes()) for name in PROTECTED}
    checks["v1_preserved"] = all(sha((root / name).read_bytes()) == expected for name, expected in protocol["v1_preserved_sha256"].items()) and not git("diff", spec["base_commit"], "--", *protocol["v1_preserved_sha256"], *PROTECTED)
    for name, item in protocol["prior_artifacts"].items():
        checks["prior:" + name] = sha((source_root / item["copy_path"]).read_bytes()) == item["sha256"] == sha((root / item["original_path"]).read_bytes())
    prior_final = load(source_root / protocol["prior_artifacts"]["FINALIZED.json"]["copy_path"])
    checks["prior_sealed_members"] = all(prior_final["file_sha256"][name] == item["sha256"] for name, item in protocol["prior_artifacts"].items() if name != "FINALIZED.json")
    old_schema = load(source_root / "gold_future_capture_manifest_schema_v1.json")
    schema = load(source_root / "gold_future_capture_manifest_schema_v2.json")
    comparable = json.loads(json.dumps(schema))
    comparable["title"] = old_schema["title"]
    comparable["properties"]["manifest_version"] = old_schema["properties"]["manifest_version"]
    checks["schema_not_weakened"] = comparable == old_schema and canonical(schema) == protocol["manifest_schema_sha256"]
    source = load(source_root / "gold_future_capture_source_attestation_v2.json")
    tz = load(source_root / "gold_future_capture_timezone_attestation_v2.json")
    activation = load(source_root / "gold_future_capture_activation_v2.json")
    prefix_doc = load(source_root / "gold_recursive_prefix_protocol_v2.json")
    migration = {"v1_source_id": "XMGlobal-MT5-6_GOLD#", "v2_source_id": SOURCE_ID, "symbol": "GOLD#", "meaning": "IDENTIFIER_REPRESENTATION_ONLY"}
    checks["explicit_source_identity"] = source["identity_migration"] == protocol["identity_migration"] == migration and source["source_id"] == SOURCE_ID and source["symbol"] == source["symbol_exact_case"] == "GOLD#" and source["broker_name"] == "XM Global Limited" and source["broker_server_name"] == "XMGlobal-MT5 6"
    diagnostic = load(source_root / protocol["prior_artifacts"]["diagnostic_timestamp.json"]["copy_path"])
    checks["demo_evidence"] = source["source_account_environment"] == diagnostic["source_account_environment"] == "demo"
    checks["attestation_links"] = tz["source_attestation_sha256"] == canonical(source) == activation["source_attestation_sha256"] and activation["timezone_attestation_sha256"] == canonical(tz)
    for item in source["evidence"]:
        checks["evidence:" + Path(item["local_path"]).name] = sha((source_root / item["local_path"]).read_bytes()) == item["sha256"] and all(key in item for key in ("source_url", "retrieved_at_utc", "authority", "reviewed_by", "reviewed_at_utc", "supported_claims", "limitations"))
    checks["timezone_honesty"] = tz["timezone_status"] == "UNRESOLVED" and tz["template"] is True and tz["broker_offset_intervals"] == [] and tz["recurrence_rule"] is None and bool(tz["diagnostic_support"])
    checks["prefix_honesty"] = prefix_doc["policy"] == "UNRESOLVED" and prefix_doc["approved"] is False and prefix_doc["source_range"] is None and prefix_doc["approval_commit"] is None and prefix_doc["historically_equivalent"] is False and prefix_doc["outcome_tuning"] is False
    native = load(source_root / protocol["prior_artifacts"]["native_timeframe_audit.json"]["copy_path"])
    checks["native_policy_preserved"] = protocol["higher_timeframe_policy"] == activation["higher_timeframe_policy"] == "NATIVE_20TF_REQUIRED" and len(native["timeframes"]) == 21 and {row["timeframe"] for row in native["timeframes"]} == set(schema["properties"]["timeframe"]["enum"]) and all(row["status"] == "AVAILABLE_NATIVE" for row in native["timeframes"])
    raw = (source_root / "gold_future_capture_collector_v2.py").read_bytes()
    tree = ast.parse(raw)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    publisher = functions["publish_tip"]
    replaces = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "os" and node.func.attr == "replace"]
    dest = ast.dump(ast.parse('file_under(root, "chain_tip.json")', mode="eval").body, include_attributes=False)
    publisher_calls = [n for n in ast.walk(publisher) if isinstance(n, ast.Call)]
    full_reads = [n for n in publisher_calls if isinstance(n.func, ast.Name) and n.func.id == "recover_chain"]
    fsync = [n for n in publisher_calls if isinstance(n.func, ast.Attribute) and n.func.attr == "fsync"]
    checks["restricted_tip_replace"] = len(replaces) == 1 and replaces[0] in publisher_calls and ast.dump(replaces[0].args[1], include_attributes=False) == dest and full_reads[0].lineno < fsync[0].lineno < replaces[0].lineno
    checks["reviewed_collector_ast"] = sha(ast.dump(tree, include_attributes=False).encode()) == protocol["approved_collector_ast_sha256"] and sha(ast.dump(publisher, include_attributes=False).encode()) == protocol["approved_tip_publisher_ast_sha256"]
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    checks["no_strategy_imports"] = not imports.intersection({"xgboost", "sklearn", "torch", "gemini", "drl_trading_v2"})
    checks["independent_full_chain"] = verify_chain(run / "synthetic_chain", schema) == 3
    self_test = load(run / "self_test_stdout.txt")
    checks["self_tests"] = self_test["overall"] == "PASS" and self_test["synthetic_only"] is True and len(self_test["checks"]) >= 32
    report = load(run / "capture_certification_report.json")
    static = load(run / "collector_static_review.json")
    checks["static_certification"] = report["collector_static_review"] == static and static["static_status"] == "STATICALLY_CONFORMANT" and static["blockers"] == [] and static["collector_sha256"] == sha(raw)
    checks["readiness_honesty"] = report["readiness"]["status"] == "NOT_READY_MULTIPLE_BLOCKERS" and {b["status"] for b in report["readiness"]["blockers"]} == {"NOT_READY_SOURCE_BINDING", "NOT_READY_TIMEZONE", "NOT_READY_PREFIX_POLICY", "NOT_READY_CAPTURE_INTEGRITY", "NOT_READY_PROTOCOL_FREEZE"}
    checks["formal_partial"] = report["formal_run_status"] == m["formal_run_status"] == load(run / "metrics.json")["formal_run_status"] == "PARTIAL"
    checks["no_activation_or_production"] = all(report[key] is False for key in ("protocol_frozen", "capture_activated", "holdout_started", "strategy_outcome_inspected", "model_loaded_for_holdout", "model_trained_for_holdout", "production_changed", "production_promoted")) and report["holdout_start"] is None and activation["status"] == "NOT_ACTIVATED" and activation["capture_root"] is None and protocol["activation_permitted"] is False
    decisions = load(run / "protocol_decisions.json")
    checks["index_non_authoritative"] = decisions["chain_tip_policy"] == protocol["chain_tip_policy"] == "NON_AUTHORITATIVE_REBUILDABLE_INDEX" and decisions["replace_allowlist"] == protocol["mutable_replace_destinations"] == ["chain_tip.json"]
    commands = [json.loads(line) for line in (run / "commands.jsonl").read_text(encoding="utf-8").splitlines()]
    checks["offline_commands_only"] = {c["label"] for c in commands} == {"self_test", "certification"} and all(c["exit_code"] == 0 and (run / (c["label"] + "_stderr.txt")).read_bytes() == b"" for c in commands)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        entries = [{"manifest_created_at_utc": "2030-01-01T00:00:00+00:00", "sequence": i} for i in range(2)]
        tip = {"manifest_sequence": 0, "manifest_sha256": canonical(entries[0]), "updated_at_utc": entries[0]["manifest_created_at_utc"]}
        assert tip_status(entries, None) == "MISSING" and tip_status(entries, tip) == "STALE"
        assert tip_status(entries, {**tip, "manifest_sha256": "f" * 64}) == "CONFLICTING"
        assert tip_status(entries, {**tip, "probability": 0.9}) == "CONFLICTING"
        assert tip_status(entries, {**tip, "manifest_sequence": 1, "manifest_sha256": canonical(entries[1])}) == "MATCH"
        print("SELF_TEST_PASS")
        return 0
    if args.run is None:
        parser.error("Run required")
    run = args.run.resolve()
    require(not (run / "FINALIZED.json").exists(), "Finalized run immutable")
    with (run / "validator_attempt.json").open("x", encoding="utf-8") as handle:
        json.dump({"started_at_utc": datetime.now(timezone.utc).isoformat(), "one_shot": True}, handle)
    try:
        checks = validate(run)
    except Exception as error:
        checks = {"exception:" + type(error).__name__: False}
    failed = [name for name, ok in checks.items() if not ok]
    result = {"overall": "FAIL" if failed else "PASS", "failed_check_names": failed, "checks": checks,
              "formal_run_status": "PARTIAL", "production_promotion": False}
    with (run / "validator.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    with (run / "validator.md").open("x", encoding="utf-8") as handle:
        handle.write("# Independent protocol adjudication validation\n\nMethodology/provenance: " + result["overall"]
            + "\nIndependent validator: " + result["overall"] + "\nFormal verdict: PARTIAL. No activation or production promotion.\n")
    print("overall=" + result["overall"])
    print("failed_check_names=" + json.dumps(failed))
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
