"""Synthetic-only protocol and storage mutation tests; no broker calls."""
import ast
import csv
import io
import json
import re
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path

import gold_future_capture_certification_v2 as cert
import gold_future_capture_collector_v2 as c

ROOT = Path(__file__).resolve().parent


def write(path, value):
    path.write_bytes(c.encode(value))


def reject(call):
    try:
        call()
    except (ValueError, KeyError, TypeError, OSError):
        return
    raise AssertionError("Expected rejection")


def fixture(root):
    for name in ("snapshots", "manifests", "pending", "quarantine"):
        (root / name).mkdir(parents=True)
    source = cert.load("gold_future_capture_source_attestation_v2.json")
    source.update(template=False, source_timezone="UTC", dst_policy="NONE", notes="SYNTHETIC TEST ONLY")
    tz = cert.load("gold_future_capture_timezone_attestation_v2.json")
    tz.update(template=False, timezone_status="CERTIFIED_UTC", timezone="UTC", dst_policy="NONE", utc_offset_minutes=0,
        mt5_epoch_semantics="ALREADY_UTC", source_attestation_sha256=cert.canonical(source), notes="SYNTHETIC TEST ONLY")
    activation = cert.load("gold_future_capture_activation_v2.json")
    activation.update(template=False, status="ACTIVE_UNVERIFIED", capture_root=str(root), collector_commit="c" * 40,
        collector_version=c.VERSION, source_attestation_sha256=cert.canonical(source), timezone_attestation_sha256=cert.canonical(tz),
        protocol_freeze_commit="d" * 40, activation_effective_at_utc="2020-01-01T00:00:00+00:00",
        prefix_policy="EXACT_CONTINUOUS_CERTIFIED_PREFIX", notes="SYNTHETIC TEST ONLY")
    for name, value in (("source_attestation.json", source), ("timezone_attestation.json", tz), ("activation.json", activation)):
        write(root / name, value)
    return source, tz, activation


def raw(stamp=1600000020, spread=13):
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(c.FIELDS + ["TICK_VOLUME", "REAL_VOLUME"])
    writer.writerow([c.datetime.fromtimestamp(stamp, c.timezone.utc).isoformat(), stamp, cert.SOURCE_ID, "GOLD#", "M1", 10, 12, 9, 11, spread, 2, 0])
    return output.getvalue().encode()


def self_test(export_dir=None):
    protocol = cert.load("gold_future_capture_protocol_v2.json")
    schema = cert.load("gold_future_capture_manifest_schema_v2.json")
    old_schema = cert.load("gold_future_capture_manifest_schema_v1.json")
    collector = (ROOT / "gold_future_capture_collector_v2.py").read_text(encoding="utf-8")
    checks = []
    assert not re.fullmatch(old_schema["properties"]["source_id"]["pattern"], "XMGlobal-MT5-6_GOLD#")
    source = cert.load("gold_future_capture_source_attestation_v2.json")
    assert cert.source_identity(source)
    for change in ({"symbol": "GOLD"}, {"source_id": "other"}, {"broker_server_name": "other"}):
        reject(lambda change=change: cert.source_identity({**source, **change}))
    checks.extend(["v1_hash_id_rejected", "v2_canonical_id", "exact_symbol", "identity_mismatch_rejected"])
    assert cert.inspect_collector(collector, protocol)["static_status"] == "STATICALLY_CONFORMANT"
    for destination in ("snapshots/x.csv", "manifests/x.json", "attestations/source.json", "protocol.json", "arbitrary.json"):
        mutated = collector.replace('os.replace(temporary, file_under(root, "chain_tip.json"))', 'os.replace(temporary, file_under(root, "' + destination + '"))')
        assert cert.inspect_collector(mutated, protocol)["static_status"] == "NOT_STATICALLY_CONFORMANT"
    for payload in ("import xgboost\n", "import gemini\n", "import torch\n", 'os.replace(a, b)\n', 'open("gemini.py", "w")\n'):
        assert cert.inspect_collector(collector + "\n" + payload, protocol)["static_status"] == "NOT_STATICALLY_CONFORMANT"
    checks.extend(["only_tip_replace_allowed", "snapshot_replace_rejected", "manifest_replace_rejected", "attestation_replace_rejected",
                   "arbitrary_replace_rejected", "no_strategy_imports", "no_model_imports", "no_production_mutation"])
    with tempfile.TemporaryDirectory(prefix="gold_v2_協議_") as temporary:
        root = Path(temporary) / "chain"
        source, tz, activation = fixture(root)
        events = []
        first = c.append_snapshot(root, raw(), schema, c.now(), c.now(), stage_hook=events.append)
        assert events.index("snapshot_after_fsync") < events.index("manifest_after_fsync")
        first_tip = c.load(root / "chain_tip.json")
        second = c.append_snapshot(root, raw(1600000080), schema, c.now(), c.now())
        revision = c.append_snapshot(root, raw(spread=15), schema, c.now(), c.now(), revision_of=first["snapshot_id"])
        entries, expected, state = c.recover_chain(root, schema)
        assert len(entries) == 3 and state["status"] == "MATCH"
        before = {p.relative_to(root).as_posix(): c.sha(p.read_bytes()) for directory in ("snapshots", "manifests") for p in (root / directory).iterdir()}
        (root / "chain_tip.json").unlink()  # Synthetic disposable index, not evidence.
        assert cert.verify_capture(root, schema)["chain_valid"] and cert.verify_capture(root, schema)["index_status"] == "MISSING"
        assert c.rebuild_tip(root, schema)["status"] == "MATCH"
        write(root / "chain_tip.json", first_tip)
        assert cert.verify_capture(root, schema)["index_status"] == "STALE" and cert.verify_capture(root, schema)["readiness_pass"]
        write(root / "chain_tip.json", {**expected, "manifest_sha256": "f" * 64})
        assert cert.verify_capture(root, schema)["chain_valid"] and not cert.verify_capture(root, schema)["readiness_pass"]
        reject(lambda: c.append_snapshot(root, raw(1600000140), schema, c.now(), c.now()))
        assert c.rebuild_tip(root, schema)["status"] == "MATCH"
        assert before == {name: c.sha((root / name).read_bytes()) for name in before}
        reject(lambda: c.sealed_write(root, root / first["snapshot_path"], b"overwrite"))
        reject(lambda: c.append_snapshot(root, raw(), schema, c.now(), c.now()))
        for field in ("source_attestation_sha256", "timezone_attestation_sha256", "activation_sha256", "previous_manifest_sha256"):
            bad = deepcopy(second)
            bad[field] = "f" * 64
            write(root / "manifests/000000000001.json", bad)
            reject(lambda: cert.verify_capture(root, schema))
            write(root / "manifests/000000000001.json", second)
        modified = raw().decode().replace("SPREAD,TICK_VOLUME", "probability,TICK_VOLUME").encode()
        reject(lambda: c.summarize(modified, source, tz))
        checks.extend(["missing_tip_valid_chain", "stale_tip_warning", "conflicting_tip_blocks", "full_chain_recomputation",
            "immutable_bytes_unchanged", "attestation_hash_links", "revision_retained", "exclusive_publication", "fsync_order", "no_outcome_fields", "unicode_paths"])
        if export_dir:
            export_dir = export_dir.resolve()
            assert export_dir.is_relative_to(ROOT / "training_runs") and not export_dir.exists()
            shutil.copytree(root, export_dir)
            write(export_dir / "synthetic_cases.json", {"synthetic_only": True, "missing": None, "stale": first_tip,
                "conflicting": {**expected, "manifest_sha256": "f" * 64}, "matching": expected})
    proof = [{"synthetic_authority": True}]
    intervals = {"broker_offset_intervals": [
        {"start_utc": "2030-01-01T00:00:00+00:00", "end_utc": "2030-06-01T00:00:00+00:00", "offset_minutes": 120, "evidence": proof},
        {"start_utc": "2030-06-01T00:00:00+00:00", "end_utc": "2031-01-01T00:00:00+00:00", "offset_minutes": 180, "evidence": proof}]}
    assert c.validate_intervals(intervals, "2030-01-01T00:00:00+00:00", "2031-01-01T00:00:00+00:00")
    for boundary in ("2030-05-01T00:00:00+00:00", "2030-07-01T00:00:00+00:00"):
        bad = deepcopy(intervals)
        bad["broker_offset_intervals"][1]["start_utc"] = boundary
        reject(lambda: c.validate_intervals(bad))
    broker_tz = {**intervals, "timezone_status": "CERTIFIED_BROKER_SERVER_RULE", "timestamp_encoding": "EPOCH_SECONDS", "mt5_epoch_semantics": "LOCAL_EPOCH_REQUIRES_RULE"}
    assert c.normalize_epoch(int(c.utc("2030-02-01T02:00:00+00:00").timestamp()), broker_tz) == "2030-02-01T00:00:00+00:00"
    reject(lambda: c.normalize_epoch(0, broker_tz))
    reject(lambda: cert.timezone_check(cert.load("gold_future_capture_timezone_attestation_v2.json"), cert.load("gold_future_capture_source_attestation_v2.json")))
    checks.extend(["interval_ordering", "overlap_rejected", "coverage_gap_rejected", "broker_rule_conversion", "out_of_coverage_rejected", "unresolved_timezone_blocks"])
    proposed = cert.load("gold_recursive_prefix_protocol_v2.json")
    complete = {**proposed, "policy": "PREDECLARED_CAUSAL_REINITIALIZATION", "approved": True, "approval_commit": "a" * 40,
        "source_range": {"start_utc": "2030-01-01T00:00:00+00:00", "end_utc": "2030-02-01T00:00:00+00:00", "sealed_context_sha256": "b" * 64}, "evidence": proof}
    assert cert.prefix_check(complete, lambda _: None, lambda _: None)
    exact = {**complete, "policy": "EXACT_CONTINUOUS_CERTIFIED_PREFIX", "exact_history": {"continuous": True,
        "same_initialization": True, "same_timestamp_semantics": True, "same_native_semantics": True}}
    assert cert.prefix_check(exact, lambda _: None, lambda _: None)
    reject(lambda: cert.prefix_check({**complete, "approved": False}, lambda _: None, lambda _: None))
    reject(lambda: cert.prefix_check(proposed))
    checks.extend(["exact_prefix_structural_pass", "causal_reinit_structural_pass", "unapproved_reinit_blocks", "unresolved_prefix_blocks"])
    assert protocol["higher_timeframe_policy"] == "NATIVE_20TF_REQUIRED"
    assert len(schema["properties"]["timeframe"]["enum"]) == 21
    comparable = deepcopy(schema)
    comparable["title"] = old_schema["title"]
    comparable["properties"]["manifest_version"] = old_schema["properties"]["manifest_version"]
    assert comparable == old_schema
    for name, expected in {**protocol["v1_preserved_sha256"], **protocol["protected_sha256"]}.items():
        assert c.sha((ROOT / name).read_bytes()) == expected
    checks.extend(["native_20tf_preserved", "no_silent_resampling", "schema_evidence_fields_preserved", "v1_unchanged", "protected_hashes_unchanged"])
    result = {"overall": "PASS", "checks": checks, "synthetic_only": True}
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    self_test()
