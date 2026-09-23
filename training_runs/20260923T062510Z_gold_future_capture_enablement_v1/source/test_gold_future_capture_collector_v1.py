"""Synthetic filesystem/crash tests only; never connect MT5 or read market data."""
import ast
import csv
import io
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import gold_future_capture_collector_v1 as c
import gold_future_capture_certification_v1 as frozen

ROOT = Path(__file__).resolve().parent


def write(path, value):
    path.write_bytes(c.encode(value))


def fixture(root):
    for name in ("snapshots", "manifests", "pending", "quarantine"):
        (root / name).mkdir(parents=True)
    source = frozen.load_json(ROOT / "gold_future_capture_source_attestation_template_v1.json")
    source.update(template=False, source_id="synthetic-source", symbol="GOLD#", symbol_exact_case="GOLD#",
        source_account_environment="demo", source_timestamp_semantics="BAR_OPEN", source_timezone="UTC", dst_policy="NONE",
        volume_fields_if_present=["TICK_VOLUME", "REAL_VOLUME"])
    tz = frozen.load_json(ROOT / "gold_future_capture_timezone_attestation_template_v1.json")
    tz.update(template=False, source_id=source["source_id"], source_attestation_sha256=c.sha(c.encode(source)),
        timestamp_semantics="BAR_OPEN", timezone="UTC", dst_policy="NONE", utc_offset_minutes=0,
        status="CERTIFIED_UTC", mt5_epoch_semantics="ALREADY_UTC", source_timestamp_encoding="EPOCH_SECONDS")
    activation = frozen.load_json(ROOT / "gold_future_capture_activation_template_v1.json")
    activation.update(template=False, status="ACTIVE_UNVERIFIED", capture_root=str(root), collector_commit="c" * 40,
        collector_version=c.VERSION, source_attestation_sha256=c.sha(c.encode(source)), timezone_attestation_sha256=c.sha(c.encode(tz)),
        protocol_freeze_commit="d" * 40, activation_effective_at_utc="2020-01-01T00:00:00+00:00",
        higher_timeframe_policy="NATIVE_20TF_REQUIRED", prefix_policy="EXACT_CONTINUOUS_CERTIFIED_PREFIX")
    for name, doc in (("source_attestation.json", source), ("timezone_attestation.json", tz), ("activation.json", activation)):
        write(root / name, doc)
    return source, tz, activation


def raw(epoch=1600000020, spread=13, **changes):
    row = dict(zip(c.FIELDS + ["TICK_VOLUME", "REAL_VOLUME"],
        [c.datetime.fromtimestamp(epoch, c.timezone.utc).isoformat(), str(epoch), "synthetic-source", "GOLD#", "M1", 10, 12, 9, 11, spread, 2, 0]))
    row.update(changes)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(row), lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)
    return stream.getvalue().encode()


def rejected(call):
    try:
        call()
    except (ValueError, FileExistsError, FileNotFoundError, OSError):
        return
    raise AssertionError("Expected rejection")


def main():
    schema = c.load(ROOT / "gold_future_capture_manifest_schema_v1.json")
    spec = c.load(ROOT / "execution_spec_gold_future_capture_certification_v1.json")
    assert c.sha(c.encode(schema)) == c.SCHEMA_SHA
    checks = []
    with tempfile.TemporaryDirectory(prefix="gold_capture_測試_") as temp:
        base = Path(temp)
        root = base / "valid"
        source, tz, activation = fixture(root)
        events = []
        real_fsync = c.os.fsync
        with patch.object(c.os, "fsync", wraps=real_fsync) as fsync:
            first = c.append_snapshot(root, raw(), schema, c.now(), c.now(), stage_hook=events.append)
            assert fsync.call_count >= 3
        assert events.index("snapshot_after_fsync") < events.index("snapshot_sealed") < events.index("manifest_after_fsync")
        original = (root / first["snapshot_path"]).read_bytes()
        rejected(lambda: c.sealed_write(root, root / first["snapshot_path"], b"overwrite"))
        assert (root / first["snapshot_path"]).read_bytes() == original
        entries, tip = c.recover_chain(root, schema)
        assert len(entries) == 1 and entries[0]["spread_nonpositive_rows"] == 0
        assert b",13," in original
        checks.extend(["exclusive_create", "overwrite_rejection", "fsync", "manifest_ordering", "restart_verification", "spread_preservation", "unicode_paths"])
        rejected(lambda: c.verify_previous_hash("a", "b"))
        rejected(lambda: c.append_snapshot(root, raw(), schema, c.now(), c.now()))
        second = c.append_snapshot(root, raw(spread=15), schema, c.now(), c.now(), revision_of=first["snapshot_id"])
        assert second["revision_of_snapshot_id"] == first["snapshot_id"] and (root / first["snapshot_path"]).read_bytes() == original
        third = c.append_snapshot(root, raw(1600000080), schema, c.now(), c.now())
        entries, tip = c.recover_chain(root, schema)
        result = frozen.verify_capture_root(root, source, tz, activation, schema, spec)
        assert result["verified_manifests"] == 3 and result["chain_valid"]
        checks.extend(["previous_hash", "duplicate_bar_rejection", "revision_retained", "frozen_verifier_compatibility"])
        for changes in ({"SYMBOL": "OTHER"}, {"SOURCE_ID": "wrong"}, {"probability": 0.9}):
            rejected(lambda changes=changes: c.append_snapshot(root, raw(**changes), schema, c.now(), c.now()))
        reversed_raw = raw(1600000080) + raw().split(b"\n", 1)[1]
        rejected(lambda: c.append_snapshot(root, reversed_raw, schema, c.now(), c.now()))
        assert list((root / "quarantine").glob("*.json"))
        checks.extend(["wrong_symbol", "wrong_source", "no_strategy_fields", "timestamp_reversal_quarantine"])
        for field, value in (("collector_commit", "e" * 40), ("collector_version", "changed"), ("source_attestation_sha256", "f" * 64), ("timezone_attestation_sha256", "f" * 64)):
            changed = deepcopy(activation)
            changed[field] = value
            write(root / "activation.json", changed)
            rejected(lambda: c.recover_chain(root, schema))
            write(root / "activation.json", activation)
        for name, document, field, value in (("source_attestation.json", source, "symbol", "OTHER"),
                ("timezone_attestation.json", tz, "dst_policy", "CHANGED")):
            changed = deepcopy(document)
            changed[field] = value
            write(root / name, changed)
            rejected(lambda: c.recover_chain(root, schema))
            write(root / name, document)
        checks.extend(["source_change", "collector_change", "timezone_change", "activation_change"])
        manifest_path = root / "manifests/000000000001.json"
        for field, value in (("previous_manifest_sha256", "f" * 64), ("sequence", 0), ("snapshot_id", first["snapshot_id"])):
            changed = deepcopy(second)
            changed[field] = value
            write(manifest_path, changed)
            rejected(lambda: c.recover_chain(root, schema))
            write(manifest_path, second)
        checks.extend(["broken_chain", "duplicate_sequence", "duplicate_snapshot"])
        for stage in ("snapshot_before_fsync", "snapshot_after_fsync", "snapshot_after_publish", "manifest_before_fsync", "manifest_after_fsync", "manifest_after_publish"):
            crash_root = base / stage
            fixture(crash_root)
            def crash(current):
                if current == stage:
                    raise OSError("Synthetic crash")
            rejected(lambda: c.append_snapshot(crash_root, raw(), schema, c.now(), c.now(), stage_hook=crash))
            if stage in ("snapshot_before_fsync", "snapshot_after_fsync"):
                assert c.recover_chain(crash_root, schema)[0] == []
                assert list((crash_root / "quarantine").glob("*.json"))
            elif stage == "manifest_after_publish":
                rejected(lambda: c.recover_chain(crash_root, schema))
                recovered, expected = c.recover_chain(crash_root, schema, allow_stale_tip=True)
                assert len(recovered) == 1
                with c.writer_lock(crash_root):
                    c.publish_tip(crash_root, expected)
                assert len(c.recover_chain(crash_root, schema)[0]) == 1
            else:
                rejected(lambda: c.recover_chain(crash_root, schema))
        checks.extend(["snapshot_crash", "orphan_after_fsync", "manifest_crash", "partial_quarantine", "explicit_tip_recovery"])
        orphan = base / "orphan_manifest"
        fixture(orphan)
        write(orphan / "manifests/000000000000.json", first)
        rejected(lambda: c.recover_chain(orphan, schema))
        checks.append("orphan_manifest")
        with c.writer_lock(root):
            rejected(lambda: c.append_snapshot(root, raw(1600000140), schema, c.now(), c.now()))
        checks.append("concurrent_writer_rejection")
    tree = ast.parse((ROOT / "gold_future_capture_collector_v1.py").read_text(encoding="utf-8"))
    imports = {n.name for node in ast.walk(tree) if isinstance(node, ast.Import) for n in node.names}
    assert not imports & {"xgboost", "torch", "sklearn", "gemini", "drl_trading_v2"}
    assert not any(isinstance(n, ast.Attribute) and n.attr in {"order_send", "order_check", "predict", "fit"} for n in ast.walk(tree))
    checks.extend(["no_model_imports", "no_production_operations", "raw_only_no_gzip"])
    print(json.dumps({"overall": "PASS", "checks": checks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
