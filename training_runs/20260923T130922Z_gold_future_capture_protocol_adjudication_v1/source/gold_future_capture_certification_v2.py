"""Offline v2 certification: explicit identity migration and restricted tip index."""
import argparse
import ast
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import gold_future_capture_collector_v2 as storage

ROOT = Path(__file__).resolve().parent
SOURCE_ID = "XMGlobal-MT5-6_GOLD"
SYMBOL = "GOLD#"
MIGRATION = {"v1_source_id": "XMGlobal-MT5-6_GOLD#", "v2_source_id": SOURCE_ID,
             "symbol": SYMBOL, "meaning": "IDENTIFIER_REPRESENTATION_ONLY"}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def load(name):
    return storage.load(ROOT / name)


def canonical(value):
    return storage.sha(storage.encode(value))


def ast_hash(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def source_identity(source):
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", source["source_id"]), "Machine ID pattern")
    require(source["source_id"] == SOURCE_ID and source["symbol"] == source["symbol_exact_case"] == SYMBOL, "Source/symbol mismatch")
    require(source["broker_name"] == "XM Global Limited" and source["broker_server_name"] == "XMGlobal-MT5 6", "Instrument/source drift")
    require(source["identity_migration"] == MIGRATION, "Explicit representation-only migration required")
    return True


def check_evidence(items):
    require(isinstance(items, list) and items, "Authoritative evidence missing")
    for item in items:
        require(item["authority"] in {"BROKER_DOCUMENT", "FEED_DOCUMENT", "INDEPENDENT_REVIEW", "SIGNED_ATTESTATION"}, "Insufficient authority")
        require(item["reviewed_by"] and item["supported_claims"] and item["limitations"], "Evidence review absent")
        path = (ROOT / item["local_path"]).resolve()
        require(path.is_relative_to(ROOT) and path.is_file(), "Evidence outside retained checkout")
        require(storage.sha(path.read_bytes()) == item["sha256"], "Evidence hash mismatch")
        require(storage.utc(item["reviewed_at_utc"]) <= datetime.now(timezone.utc), "Future evidence review")


def complete_source(source, evidence_check=check_evidence):
    source_identity(source)
    template = load("gold_future_capture_source_attestation_template_v2.json")
    require(set(source) == set(template) and source["template"] is False, "Source preparation incomplete")
    require(source["attestation_version"] == template["attestation_version"], "Source version")
    for key in ("source_timezone", "timezone_authority", "dst_policy", "session_rollover", "weekend_policy",
                "holiday_policy_source", "attested_by", "attested_at_utc"):
        require(source[key] not in (None, "", "UNRESOLVED", "unknown"), "Source field unresolved: " + key)
    require(source["source_account_environment"] in {"demo", "live"} and source["spread_units"] == "POINTS"
            and source["price_digits"] == 2 and source["point_size"] == 0.01, "Source precision/environment")
    require(source["source_timestamp_semantics"] == "BAR_OPEN", "Frozen bar semantics")
    evidence_check(source["evidence"])
    return True


def timezone_check(tz, source, coverage=None, evidence_check=check_evidence):
    template = load("gold_future_capture_timezone_attestation_template_v2.json")
    require(set(tz) == set(template) and tz["template"] is False, "Timezone preparation incomplete")
    require(tz["attestation_version"] == template["attestation_version"], "Timezone version")
    require(tz["source_id"] == source["source_id"] and tz["source_attestation_sha256"] == canonical(source), "Timezone/source hash link")
    require(tz["timezone_status"] in {"CERTIFIED_BROKER_SERVER_RULE", "CERTIFIED_UTC", "CERTIFIED_FIXED_OFFSET", "CERTIFIED_IANA_WITH_DST"}, "Timezone unresolved")
    for key in ("timezone", "timezone_authority", "dst_policy", "server_display_clock_rule", "session_rollover",
                "weekend_policy", "holiday_calendar_authority", "attested_by", "attested_at_utc"):
        require(tz[key] not in (None, "", "UNRESOLVED", "unknown"), "Timezone rule unresolved: " + key)
    require(tz["bar_timestamp_semantics"] == source["source_timestamp_semantics"] == "BAR_OPEN"
            and tz["timestamp_encoding"] == "EPOCH_SECONDS", "Timestamp encoding/label")
    for tz_key, source_key in (("timezone", "source_timezone"), ("dst_policy", "dst_policy"), ("session_rollover", "session_rollover"),
                                ("weekend_policy", "weekend_policy"), ("holiday_calendar_authority", "holiday_policy_source")):
        require(tz[tz_key] == source[source_key], "Source/timezone declaration mismatch")
    evidence_check(tz["evidence"])
    start, end = tz["certified_coverage_start_utc"], tz["certified_coverage_end_utc"]
    require(start and end and storage.utc(start) < storage.utc(end), "Explicit certified coverage required")
    if coverage:
        require(storage.utc(start) <= storage.utc(coverage[0]) < storage.utc(coverage[1]) <= storage.utc(end), "Activation outside certified coverage")
    if tz["timezone_status"] == "CERTIFIED_BROKER_SERVER_RULE":
        intervals = storage.validate_intervals(tz, start, end)
        for interval in intervals:
            evidence_check(interval["evidence"])
        require(tz["broker_transition_rule"], "Documented broker rule absent")
        # No recurrence expansion: only the explicit reviewed table is usable.
        require(tz["recurrence_rule"] is None, "Recurrence requires a separately reviewed implementation")
    else:
        test_epoch = int(storage.utc(start).timestamp())
        storage.normalize_epoch(test_epoch, tz)
    return True


def prefix_approval(doc):
    blob = subprocess.check_output(["git", "cat-file", "blob", doc["approval_commit"] + ":gold_recursive_prefix_approval_v2.json"], cwd=ROOT)
    decision = json.loads(blob)
    definition = {k: v for k, v in doc.items() if k not in {"approved", "approval_commit"}}
    require(decision.get("prefix_definition_sha256") == canonical(definition)
            and decision.get("approved") is True and decision.get("source_id") == SOURCE_ID
            and decision.get("symbol") == SYMBOL, "Prefix approval artifact mismatch")


def prefix_check(doc, evidence_check=check_evidence, approval_check=prefix_approval):
    require(doc["source_id"] == SOURCE_ID and doc["symbol"] == SYMBOL, "Prefix source/symbol mismatch")
    require(doc["policy"] in {"EXACT_CONTINUOUS_CERTIFIED_PREFIX", "PREDECLARED_CAUSAL_REINITIALIZATION"}, "Prefix unresolved")
    require(doc["approved"] is True and re.fullmatch(r"[0-9a-f]{40}", doc["approval_commit"] or ""), "Prefix immutable approval missing")
    require(doc["outcome_tuning"] is False, "Outcome-dependent initialization prohibited")
    require(doc["pipeline_sha256"] == storage.sha((ROOT / "drl_trading_v2.py").read_bytes()), "Frozen feature source changed")
    source_range = doc["source_range"]
    require(isinstance(source_range, dict) and set(source_range) == {"start_utc", "end_utc", "sealed_context_sha256"}, "Exact prefix range missing")
    require(storage.utc(source_range["start_utc"]) < storage.utc(source_range["end_utc"])
            and re.fullmatch(r"[0-9a-f]{64}", source_range["sealed_context_sha256"]), "Prefix range/hash")
    evidence_check(doc["evidence"])
    approval_check(doc)
    if doc["policy"] == "EXACT_CONTINUOUS_CERTIFIED_PREFIX":
        history = doc["exact_history"]
        require(isinstance(history, dict) and history == {"continuous": True, "same_initialization": True,
            "same_timestamp_semantics": True, "same_native_semantics": True}, "Exact recursive equivalence unproven")
    else:
        proposed = load("gold_recursive_prefix_protocol_v2.json")
        require(doc["historically_equivalent"] is False and doc["initialization"] == proposed["initialization"]
                and doc["warmup_m1_rows"] == 4096 and doc["warmup_native_rows_per_timeframe"] == 21, "Reinitialization must match fixed non-equivalent proposal")
    return True


def activation_check(activation, source, tz, prefix, protocol):
    template = load("gold_future_capture_activation_template_v2.json")
    require(set(activation) == set(template) and activation["template"] is False
            and activation["activation_version"] == template["activation_version"], "Activation preparation incomplete")
    require(activation["source_attestation_sha256"] == canonical(source)
            and activation["timezone_attestation_sha256"] == canonical(tz)
            and activation["prefix_protocol_sha256"] == canonical(prefix), "Activation document bindings")
    require(activation["higher_timeframe_policy"] == "NATIVE_20TF_REQUIRED" and activation["prefix_policy"] == prefix["policy"]
            and activation["chain_tip_policy"] == "NON_AUTHORITATIVE_REBUILDABLE_INDEX", "Activation policies")
    require(activation["status"] in {"ACTIVATION_ARTIFACT_READY", "ACTIVE_UNVERIFIED"}, "Activation not prepared")
    root = Path(activation["capture_root"]).resolve()
    require(root.is_relative_to(ROOT / "future_holdout_capture"), "Research capture root required")
    for doc in (source, tz, activation):
        require(doc["production_promotion"] is False and doc["historical_2025_plus_is_untouched"] is False
                and doc["outcome_inspection_prohibited"] is True and doc["strategy_execution_prohibited"] is True
                and doc["holdout_start"] is None, "No-outcome/holdout policy")
    storage.verify_executable_binding(activation)
    require(activation["collector_version"] == "gold_future_capture_collector_v2", "Collector version")
    start, end = activation["coverage_start_utc"], activation["coverage_end_utc"]
    require((storage.utc(end) - storage.utc(start)).days >= 365, "Full holdout coverage required")
    timezone_check(tz, source, (start, end))
    return root


def freeze_check(activation, source, tz, prefix, protocol):
    commit = activation["protocol_freeze_commit"]
    require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit), "Holdout protocol not frozen")
    blob = subprocess.check_output(["git", "cat-file", "blob", commit + ":gold_future_locked_holdout_protocol_v2.json"], cwd=ROOT)
    frozen = json.loads(blob)
    require(canonical(frozen) == activation["protocol_freeze_sha256"], "Freeze artifact hash mismatch")
    expected = {"source_id": source["source_id"], "symbol": source["symbol"], "timezone_attestation_sha256": canonical(tz),
        "prefix_protocol_sha256": canonical(prefix), "higher_timeframe_policy": "NATIVE_20TF_REQUIRED",
        "collector_ast_sha256": protocol["approved_collector_ast_sha256"], "manifest_schema_sha256": protocol["manifest_schema_sha256"],
        "certifier_sha256": storage.sha(Path(__file__).read_bytes()), "holdout_duration_calendar_days": 365,
        "candidate": {"name": "S4", "primary_threshold": 0.75, "secondary_threshold": 0.75, "execution": "S5", "feature_count": 31},
        "no_peeking": True}
    require(all(frozen.get(k) == value for k, value in expected.items()), "Freeze definition mismatch")
    require(storage.utc(activation["protocol_freeze_effective_at_utc"]) <= storage.utc(activation["coverage_start_utc"]), "Coverage predates freeze")
    return True


def inspect_collector(text, protocol):
    tree = ast.parse(text)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    risks = []
    allowed_imports = {"argparse", "csv", "hashlib", "io", "json", "math", "os", "re", "subprocess", "sys", "uuid",
                       "contextlib", "datetime", "pathlib", "zoneinfo", "MetaTrader5"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(item.name not in allowed_imports for item in node.names):
                risks.append("Unreviewed/strategy/model import")
        elif isinstance(node, ast.ImportFrom) and node.module not in allowed_imports:
            risks.append("Unreviewed/strategy/model import")
        if isinstance(node, ast.Call):
            leaf = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else "dynamic"
            if leaf in {"exec", "eval", "__import__", "getattr", "setattr", "fit", "predict", "predict_proba", "order_send", "order_check", "unlink", "remove", "rename", "truncate", "write_text", "write_bytes"}:
                risks.append("Forbidden dynamic/strategy/mutation call")
    replacements = []
    for name, fn in functions.items():
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "os" and node.func.attr == "replace":
                replacements.append((name, node))
    expected_destination = ast.dump(ast.parse('file_under(root, "chain_tip.json")', mode="eval").body, include_attributes=False)
    if len(replacements) != 1:
        risks.append("Exactly one reviewed index replacement required")
    else:
        name, call = replacements[0]
        if name != "publish_tip" or len(call.args) != 2 or call.keywords or ast.dump(call.args[1], include_attributes=False) != expected_destination:
            risks.append("Replacement outside exact non-authoritative tip destination")
    publisher = functions.get("publish_tip")
    if publisher is None or ast_hash(publisher) != protocol["approved_tip_publisher_ast_sha256"]:
        risks.append("Tip publisher control flow changed; review required")
    if ast_hash(tree) != protocol["approved_collector_ast_sha256"]:
        risks.append("Collector implementation differs from reviewed AST")
    required = {"sealed_write", "recover_chain", "summarize", "validate_schema", "verify_previous_hash", "quarantine_partial",
                "record_revision", "binding", "append_snapshot", "publish_tip", "rebuild_tip", "assess_tip", "writer_lock"}
    if not required <= set(functions):
        risks.append("Required immutable storage/recovery interface missing")
    return {"static_status": "NOT_STATICALLY_CONFORMANT" if risks else "STATICALLY_CONFORMANT",
        "collector_sha256": hashlib.sha256(text.encode()).hexdigest(), "collector_ast_sha256": ast_hash(tree),
        "chain_tip_policy": "NON_AUTHORITATIVE_REBUILDABLE_INDEX", "replace_allowlist": ["chain_tip.json"],
        "blockers": sorted(set(risks)), "operational_status": "NOT_ACTIVATED",
        "limitations": "Certification applies only to the exact reviewed AST plus synthetic fault tests; no operational proof or arbitrary-code safety claim."}


def verify_capture(root, schema):
    activation = storage.load(storage.file_under(root, "activation.json"))
    if activation["status"] == "ACTIVATION_ARTIFACT_READY":
        require(not any(storage.file_under(root, "manifests").iterdir())
                and not any(storage.file_under(root, "snapshots").iterdir())
                and not (root / "chain_tip.json").exists(), "Prepared layout must be empty")
        return {"chain_valid": None, "verified_manifests": 0, "recomputed_tip": None,
                "index_status": "MISSING", "readiness_pass": True, "warning": "Prepared layout; no operational evidence"}
    entries, tip, state = storage.recover_chain(root, schema)
    return {"chain_valid": True, "verified_manifests": len(entries), "recomputed_tip": tip,
            "index_status": state["status"], "readiness_pass": state["readiness_pass"], "warning": state["warning"]}


def certify():
    protocol = load("gold_future_capture_protocol_v2.json")
    source = load("gold_future_capture_source_attestation_v2.json")
    tz = load("gold_future_capture_timezone_attestation_v2.json")
    activation = load("gold_future_capture_activation_v2.json")
    prefix = load("gold_recursive_prefix_protocol_v2.json")
    for name, expected in {**protocol["v1_preserved_sha256"], **protocol["protected_sha256"]}.items():
        require(storage.sha((ROOT / name).read_bytes()) == expected, "Protected/v1 artifact changed: " + name)
    schema = load("gold_future_capture_manifest_schema_v2.json")
    require(canonical(schema) == protocol["manifest_schema_sha256"], "V2 schema identity changed")
    prior = protocol["prior_artifacts"]
    for item in prior.values():
        require(storage.sha((ROOT / item["copy_path"]).read_bytes()) == item["sha256"] == storage.sha((ROOT / item["original_path"]).read_bytes()), "Prior evidence changed")
    native = load(prior["native_timeframe_audit.json"]["copy_path"])
    diagnostic = load(prior["diagnostic_timestamp.json"]["copy_path"])
    require(diagnostic["source_account_environment"] == source["source_account_environment"] == "demo", "Archived environment binding")
    blockers = []
    def gate(code, callback):
        try:
            callback()
            return True
        except (ValueError, TypeError, KeyError, OSError, subprocess.CalledProcessError) as error:
            blockers.append({"status": code, "reason": str(error) if type(error) is ValueError else type(error).__name__})
            return False
    identity = gate("NOT_READY_SOURCE_BINDING", lambda: source_identity(source))
    source_ready = gate("NOT_READY_SOURCE_BINDING", lambda: complete_source(source))
    timezone_ready = gate("NOT_READY_TIMEZONE", lambda: timezone_check(tz, source))
    prefix_ready = gate("NOT_READY_PREFIX_POLICY", lambda: prefix_check(prefix))
    frames = set(load("gold_future_capture_manifest_schema_v1.json")["properties"]["timeframe"]["enum"])
    higher = gate("NOT_READY_HIGHER_TIMEFRAMES", lambda: require(activation["higher_timeframe_policy"] == "NATIVE_20TF_REQUIRED"
        and protocol["higher_timeframe_policy"] == "NATIVE_20TF_REQUIRED" and len(native["timeframes"]) == 21
        and {r["timeframe"] for r in native["timeframes"]} == frames and all(r["status"] == "AVAILABLE_NATIVE" for r in native["timeframes"]), "Native structural evidence missing"))
    static = inspect_collector((ROOT / "gold_future_capture_collector_v2.py").read_text(encoding="utf-8"), protocol)
    static["collector_sha256"] = storage.sha((ROOT / "gold_future_capture_collector_v2.py").read_bytes())
    collector_ready = gate("NOT_READY_COLLECTOR", lambda: require(static["static_status"] == "STATICALLY_CONFORMANT", "Unapproved collector implementation"))
    integrity = None
    activation_ready = gate("NOT_READY_CAPTURE_INTEGRITY", lambda: activation_check(activation, source, tz, prefix, protocol))
    if activation_ready and activation["capture_root"]:
        def verify():
            nonlocal integrity
            capture_root = Path(activation["capture_root"])
            for name, doc in (("source_attestation.json", source), ("timezone_attestation.json", tz), ("activation.json", activation)):
                require(canonical(storage.load(storage.file_under(capture_root, name))) == canonical(doc), "Capture root binding mismatch")
            integrity = verify_capture(capture_root, schema)
            require(integrity["readiness_pass"], "Conflicting non-authoritative tip")
        gate("NOT_READY_CAPTURE_INTEGRITY", verify)
    else:
        blockers.append({"status": "NOT_READY_CAPTURE_INTEGRITY", "reason": "No activated capture layout; not prepared while external gates unresolved"})
    gate("NOT_READY_PROTOCOL_FREEZE", lambda: freeze_check(activation, source, tz, prefix, protocol))
    codes = {b["status"] for b in blockers}
    readiness = "READY_FOR_CAPTURE_ACTIVATION" if not codes else next(iter(codes)) if len(codes) == 1 else "NOT_READY_MULTIPLE_BLOCKERS"
    compatible = identity and collector_ready and higher
    verdict = "FAIL" if not compatible else "PASS" if source_ready and timezone_ready and prefix_ready else "PARTIAL"
    return {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "formal_run_status": verdict,
        "source_id_policy": "V2_CANONICALIZED" if identity else "FAIL", "source_id": source["source_id"], "symbol": source["symbol"],
        "source_attestation_status": "PASS" if source_ready else "UNRESOLVED",
        "timezone_attestation_status": "PASS" if timezone_ready else "UNRESOLVED", "timezone_policy": tz["timezone_status"],
        "higher_timeframe_policy": "NATIVE_20TF_REQUIRED" if higher else "UNRESOLVED",
        "prefix_policy": prefix["policy"], "prefix_certification_status": "PASS" if prefix_ready else "PARTIAL",
        "chain_tip_policy": protocol["chain_tip_policy"], "collector_static_review": static,
        "capture_integrity": integrity, "readiness": {"status": readiness, "blockers": blockers},
        "bindings": {"source_attestation_sha256": canonical(source), "timezone_attestation_sha256": canonical(tz),
                     "activation_sha256": canonical(activation), "prefix_protocol_sha256": canonical(prefix)},
        "protocol_frozen": False, "capture_activated": False, "holdout_started": False, "holdout_start": None,
        "strategy_outcome_inspected": False, "model_loaded_for_holdout": False, "model_trained_for_holdout": False,
        "production_changed": False, "production_promoted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--output", type=Path)
    parser.add_argument("--export-synthetic", type=Path)
    args = parser.parse_args()
    if args.self_test:
        from test_gold_future_capture_protocol_v2 import self_test
        self_test(args.export_synthetic)
        return
    require(args.export_synthetic is None, "Synthetic export requires self-test")
    report = certify()
    destination = args.output.resolve()
    require(destination.is_relative_to(ROOT) and destination.suffix == ".json", "Research output path required")
    require(destination.name == "capture_certification_report.json" and destination.parent.parent == ROOT / "training_runs", "New formal run output required")
    require(not (destination.parent / "FINALIZED.json").exists(), "Finalized run immutable")
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(report["readiness"]["status"])


if __name__ == "__main__":
    main()
