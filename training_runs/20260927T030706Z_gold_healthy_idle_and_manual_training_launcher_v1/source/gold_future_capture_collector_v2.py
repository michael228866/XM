"""V2 raw capture: immutable evidence and a narrowly rebuildable tip index.

No strategy logic. All activation/timezone/prefix gates must be certified first.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import TZPATH, ZoneInfo

VERSION = "gold_future_capture_collector_v2"
ROOT = Path(__file__).resolve().parent
SCHEMA_SHA = "8f288343a8dbf4fb61025634f3485565255d324cf7bf535ff5ebb9986f19fbbd"
FIELDS = ["SOURCE_TIMESTAMP", "ORIGINAL_SOURCE_TIMESTAMP", "SOURCE_ID", "SYMBOL",
          "TIMEFRAME", "OPEN", "HIGH", "LOW", "CLOSE", "SPREAD"]
TIMEFRAMES = {"M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5, "M6": 6,
              "M10": 10, "M12": 12, "M15": 15, "M20": 20, "M30": 30,
              "H1": 16385, "H2": 16386, "H3": 16387, "H4": 16388,
              "H6": 16390, "H8": 16392, "H12": 16396, "Daily": 16408,
              "Weekly": 32769, "Monthly": 49153}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def now():
    return datetime.now(timezone.utc).isoformat()


def utc(text):
    value = datetime.fromisoformat(text)
    require(value.tzinfo is not None and value.utcoffset().total_seconds() == 0, "UTC required")
    return value


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique)


def file_under(root, relative):
    path = root / relative
    require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), "Redirected capture path")
    return path


def stable_bar_identity(row):
    return tuple(row[key] for key in ("SOURCE_ID", "SYMBOL", "TIMEFRAME", "SOURCE_TIMESTAMP"))


def verify_previous_hash(actual, expected):
    require(actual == expected, "Previous manifest hash mismatch")


def sealed_write(root, destination, raw, stage_hook=lambda stage: None):
    """Same-volume hard-link publication: fail if destination already exists."""
    pending = root / "pending" / (uuid.uuid4().hex + ".tmp")
    with pending.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        stage_hook("before_fsync")
        os.fsync(handle.fileno())
    stage_hook("after_fsync")
    require(pending.read_bytes() == raw, "Sealed write verification failed")
    os.link(pending, destination)
    stage_hook("after_publish")
    # Pending hard links are intentionally retained, never reused or overwritten.


def quarantine_partial(root, paths, reason):
    """Preserve bytes and a quarantine receipt; never alter accepted evidence."""
    receipt = {"created_at_utc": now(), "reason": reason, "artifacts": []}
    for path in paths:
        require(path.resolve().is_relative_to(root.resolve()) and not path.is_symlink(), "Quarantine path escape")
        if path.is_file():
            raw = path.read_bytes()
            target = root / "quarantine" / (uuid.uuid4().hex + ".bin")
            with target.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            receipt["artifacts"].append({"original": path.relative_to(root).as_posix(),
                "copy": target.relative_to(root).as_posix(), "sha256": sha(raw)})
    with (root / "quarantine" / (uuid.uuid4().hex + ".json")).open("xb") as handle:
        handle.write(encode(receipt))
        handle.flush()
        os.fsync(handle.fileno())
    return receipt


def validate_schema(item, schema):
    require(set(item) == set(schema["required"]), "Manifest field set")
    for key, rule in schema["properties"].items():
        value = item[key]
        if "const" in rule:
            require(type(value) is type(rule["const"]) and value == rule["const"], "Manifest constant")
        if "enum" in rule:
            require(value in rule["enum"], "Manifest enum")
        if "type" in rule:
            kinds = rule["type"] if isinstance(rule["type"], list) else [rule["type"]]
            require(any(value is None if k == "null" else type(value) is int if k == "integer"
                        else isinstance(value, str) for k in kinds), "Manifest type")
        if value is not None:
            if "pattern" in rule:
                require(re.fullmatch(rule["pattern"], value), "Manifest pattern")
            if "minimum" in rule:
                require(value >= rule["minimum"], "Manifest minimum")
            if "minLength" in rule:
                require(len(value) >= rule["minLength"], "Manifest empty text")
            if rule.get("format") == "date-time":
                utc(value)
    require(item["sequence"] == item["snapshot_sequence"], "Snapshot sequence mismatch")
    require((item["previous_manifest_sha256"] is None) == (item["sequence"] == 0), "Genesis link")
    require(utc(item["first_source_timestamp"]) <= utc(item["last_source_timestamp"]) <= utc(item["capture_finished_at_utc"]), "Future source timestamp")
    require(utc(item["capture_started_at_utc"]) <= utc(item["capture_finished_at_utc"]) <= utc(item["manifest_created_at_utc"]), "Capture chronology")


def validate_intervals(tz, coverage_start=None, coverage_end=None):
    intervals = tz["broker_offset_intervals"]
    require(intervals, "Reviewed broker intervals required")
    previous = None
    for item in intervals:
        require(set(item) == {"start_utc", "end_utc", "offset_minutes", "evidence"}, "Interval fields")
        start, end = utc(item["start_utc"]), utc(item["end_utc"])
        require(start < end and (previous is None or start == previous), "Interval gap/overlap/order")
        require(type(item["offset_minutes"]) is int and -840 <= item["offset_minutes"] <= 840, "Offset range")
        require(isinstance(item["evidence"], list) and item["evidence"], "Interval evidence binding required")
        previous = end
    if coverage_start is not None or coverage_end is not None:
        require(coverage_start is not None and coverage_end is not None, "Both coverage boundaries required")
        require(utc(intervals[0]["start_utc"]) <= utc(coverage_start) < utc(coverage_end)
                <= utc(intervals[-1]["end_utc"]), "Intervals do not cover requested period")
    return intervals


def normalize_epoch(epoch, tz):
    require(tz["timestamp_encoding"] == "EPOCH_SECONDS", "Epoch encoding required")
    raw = datetime.fromtimestamp(int(epoch), timezone.utc)
    status = tz["timezone_status"]
    if status == "CERTIFIED_UTC":
        require(tz["mt5_epoch_semantics"] == "ALREADY_UTC" and tz["utc_offset_minutes"] == 0
                and tz["dst_policy"] == "NONE", "UTC rule conflict")
        return raw.isoformat()
    if status == "CERTIFIED_FIXED_OFFSET":
        require(tz["dst_policy"] == "NONE" and type(tz["utc_offset_minutes"]) is int
                and -840 <= tz["utc_offset_minutes"] <= 840, "Fixed offset rule")
        offset = tz["utc_offset_minutes"] if tz["mt5_epoch_semantics"] == "LOCAL_EPOCH_REQUIRES_RULE" else 0
        require(tz["mt5_epoch_semantics"] in {"ALREADY_UTC", "LOCAL_EPOCH_REQUIRES_RULE"}, "Epoch rule unresolved")
        return (raw - timedelta(minutes=offset)).isoformat()
    if status == "CERTIFIED_IANA_WITH_DST":
        require(re.fullmatch(r"[A-Za-z0-9_+-]+(?:/[A-Za-z0-9_+-]+)*", tz["timezone"]), "Invalid zone name")
        files = [Path(base) / tz["timezone"] for base in TZPATH]
        path = next((p for p in files if p.is_file()), None)
        require(path is not None and sha(path.read_bytes()) == tz["iana_tzdb_identity"], "Pinned TZif unavailable")
        with path.open("rb") as handle:
            zone = ZoneInfo.from_file(handle)
        if tz["mt5_epoch_semantics"] == "ALREADY_UTC":
            return raw.isoformat()
        require(tz["mt5_epoch_semantics"] == "LOCAL_EPOCH_REQUIRES_RULE", "IANA epoch interpretation")
        wall = raw.replace(tzinfo=None)
        candidates = {wall.replace(tzinfo=zone, fold=fold).astimezone(timezone.utc) for fold in (0, 1)}
        candidates = {v for v in candidates if v.astimezone(zone).replace(tzinfo=None) == wall}
        require(len(candidates) == 1, "Ambiguous/nonexistent local timestamp")
        return candidates.pop().isoformat()
    require(status == "CERTIFIED_BROKER_SERVER_RULE", "Unsupported/unresolved timezone; no inference")
    intervals = validate_intervals(tz)
    candidates = set()
    for item in intervals:
        require(tz["mt5_epoch_semantics"] in {"ALREADY_UTC", "LOCAL_EPOCH_REQUIRES_RULE"}, "Epoch rule unresolved")
        offset = item["offset_minutes"] if tz["mt5_epoch_semantics"] == "LOCAL_EPOCH_REQUIRES_RULE" else 0
        candidate = raw - timedelta(minutes=offset)
        if utc(item["start_utc"]) <= candidate < utc(item["end_utc"]):
            candidates.add(candidate)
    require(len(candidates) == 1, "Ambiguous/nonexistent/out-of-coverage broker timestamp")
    return candidates.pop().isoformat()


def summarize(raw, source, tz):
    columns = FIELDS + source["volume_fields_if_present"]
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
    require(reader.fieldnames == columns, "Raw schema or strategy fields rejected")
    rows = list(reader)
    require(rows, "Empty snapshot")
    previous = None
    identities = set()
    frames = set()
    nonpositive = 0
    for row in rows:
        require(None not in row and all(v is not None for v in row.values()), "Malformed CSV")
        require(row["SOURCE_ID"] == source["source_id"] and row["SYMBOL"] == source["symbol"], "Wrong source or symbol")
        require(row["TIMEFRAME"] in TIMEFRAMES, "Timeframe unsupported")
        frames.add(row["TIMEFRAME"])
        stamp = utc(row["SOURCE_TIMESTAMP"])
        require(utc(normalize_epoch(row["ORIGINAL_SOURCE_TIMESTAMP"], tz)) == stamp, "Normalized timestamp mismatch")
        identity = stable_bar_identity(row)
        require(identity not in identities, "Duplicate bar")
        require(previous is None or stamp > previous, "Timestamp reversal")
        previous = stamp
        identities.add(identity)
        prices = [float(row[k]) for k in ("OPEN", "HIGH", "LOW", "CLOSE")]
        require(all(math.isfinite(x) and x > 0 for x in prices) and prices[2] <= min(prices) and prices[1] >= max(prices), "Invalid raw OHLC")
        spread = float(row["SPREAD"])
        require(math.isfinite(spread), "Missing spread")
        nonpositive += int(spread <= 0)
        for field in source["volume_fields_if_present"]:
            require(math.isfinite(float(row[field])) and float(row[field]) >= 0, "Invalid volume")
    require(len(frames) == 1, "Mixed timeframe snapshot")
    return {"row_count": len(rows), "first_source_timestamp": rows[0]["SOURCE_TIMESTAMP"],
        "last_source_timestamp": rows[-1]["SOURCE_TIMESTAMP"], "timeframe": rows[0]["TIMEFRAME"],
        "raw_sha256": sha(raw), "schema_sha256": sha(encode({"format": "gold_canonical_bars_v1", "encoding": "UTF-8", "delimiter": ",", "columns": columns})),
        "duplicate_timestamps": 0, "non_monotonic_timestamps": 0, "invalid_timestamp_rows": 0,
        "invalid_price_rows": 0, "spread_missing_rows": 0, "spread_nonpositive_rows": nonpositive}


def record_revision(prior, summary):
    for key in ("timeframe", "first_source_timestamp", "last_source_timestamp", "row_count"):
        require(prior[key] == summary[key], "Revision interval must exactly match original")
    require(prior["raw_sha256"] != summary["raw_sha256"], "Identical revision is a duplicate")
    return prior["snapshot_id"]


def binding(root):
    source, tz, activation = [load(file_under(root, name)) for name in
        ("source_attestation.json", "timezone_attestation.json", "activation.json")]
    require(source["template"] is False and tz["template"] is False and activation["template"] is False, "Incomplete attestations")
    require(source["attestation_version"] == "gold_future_capture_source_attestation_v2"
            and tz["attestation_version"] == "gold_future_capture_timezone_attestation_v2"
            and activation["activation_version"] == "gold_future_capture_activation_v2", "Attestation version mismatch")
    require(source["symbol"] == source["symbol_exact_case"] == "GOLD#" and source["source_id"] == "XMGlobal-MT5-6_GOLD", "Exact v2 source/symbol binding required")
    require(source["broker_name"] == "XM Global Limited" and source["broker_server_name"] == "XMGlobal-MT5 6", "Broker identity drift")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", source["source_id"]), "Source identity violates frozen schema")
    require(source["source_account_environment"] in ("demo", "live"), "Environment unresolved")
    require(source["source_timestamp_semantics"] == tz["bar_timestamp_semantics"] == "BAR_OPEN", "BAR_OPEN required")
    require(source["source_timezone"] == tz["timezone"] and source["dst_policy"] == tz["dst_policy"], "Timezone declarations conflict")
    require(tz["source_id"] == source["source_id"] and tz["source_attestation_sha256"] == sha(encode(source)), "Source/timezone linkage")
    require(activation["source_attestation_sha256"] == sha(encode(source)) and activation["timezone_attestation_sha256"] == sha(encode(tz)), "Activation linkage")
    require(Path(activation["capture_root"]).resolve() == root.resolve(), "Capture root mismatch")
    require(activation["collector_version"] == VERSION, "Collector version requires recertification")
    require(activation["status"] == "ACTIVE_UNVERIFIED" and activation["protocol_freeze_commit"], "Activation and protocol freeze required")
    require(activation["higher_timeframe_policy"] == "NATIVE_20TF_REQUIRED" and activation["prefix_policy"] in {"EXACT_CONTINUOUS_CERTIFIED_PREFIX", "PREDECLARED_CAUSAL_REINITIALIZATION"}, "Pipeline and prefix unresolved")
    for doc in (source, tz, activation):
        require(doc["production_promotion"] is False and doc["historical_2025_plus_is_untouched"] is False
                and doc["outcome_inspection_prohibited"] is True and doc["strategy_execution_prohibited"] is True, "Policy substitution")
    require(tz["timezone_status"] in {"CERTIFIED_UTC", "CERTIFIED_FIXED_OFFSET", "CERTIFIED_IANA_WITH_DST", "CERTIFIED_BROKER_SERVER_RULE"}, "Epoch path uncertified")
    if tz["timezone_status"] == "CERTIFIED_BROKER_SERVER_RULE":
        validate_intervals(tz, activation["coverage_start_utc"], activation["coverage_end_utc"])
    return source, tz, activation


def recover_chain(root, schema, audit_quarantine=False):
    """Verify every sealed byte. Quarantine anomalies and stop; no auto repair."""
    source, tz, activation = binding(root)
    entries, previous, seen, paths, last = [], None, {}, set(), {}
    for number, path in enumerate(sorted(file_under(root, "manifests").iterdir())):
        try:
            require(path.name == f"{number:012d}.json", "Duplicate or missing sequence")
            item = load(file_under(root, path.relative_to(root)))
            validate_schema(item, schema)
            require(item["sequence"] == number, "Sequence mismatch")
            verify_previous_hash(item["previous_manifest_sha256"], previous)
            require(item["snapshot_id"] not in seen and item["snapshot_path"] not in paths, "Duplicate snapshot")
            for key, value in {"source_attestation_sha256": sha(encode(source)), "timezone_attestation_sha256": sha(encode(tz)),
                               "activation_sha256": sha(encode(activation)), "source_id": source["source_id"], "symbol": source["symbol"],
                               "collector_commit": activation["collector_commit"], "collector_version": VERSION}.items():
                require(item[key] == value, "Binding changed; new attestation/recertification required")
            snapshot = file_under(root, item["snapshot_path"])
            summary = summarize(snapshot.read_bytes(), source, tz)
            require(all(summary[key] == item[key] for key in summary), "Snapshot hash/metadata mismatch")
            if entries:
                require(utc(item["capture_started_at_utc"]) > utc(entries[-1]["capture_finished_at_utc"])
                        and utc(item["manifest_created_at_utc"]) > utc(entries[-1]["manifest_created_at_utc"]), "Clock reversal")
            require(utc(item["capture_started_at_utc"]) >= utc(activation["activation_effective_at_utc"]), "Capture predates activation")
            revision = item["revision_of_snapshot_id"]
            if revision:
                require(revision in seen, "Orphan revision")
                record_revision(seen[revision], summary)
            else:
                require(item["timeframe"] not in last or utc(item["first_source_timestamp"]) > last[item["timeframe"]], "Duplicate/reversed interval")
                last[item["timeframe"]] = utc(item["last_source_timestamp"])
            seen[item["snapshot_id"]] = item
            paths.add(item["snapshot_path"])
            previous = sha(encode(item))
            entries.append(item)
        except (ValueError, OSError, KeyError, TypeError):
            if audit_quarantine:
                quarantine_partial(root, [path], "Invalid/orphan manifest; manual review and new root required")
            raise
    orphans = [path for path in file_under(root, "snapshots").iterdir() if path.relative_to(root).as_posix() not in paths]
    if orphans:
        if audit_quarantine:
            quarantine_partial(root, orphans, "Orphan snapshot; no automatic chain adoption")
        raise ValueError("Orphan snapshot quarantined; new reviewed root required")
    # Pending writes are copies/hard links, not chain members. Preserve anomalies.
    accepted_hashes = {item["raw_sha256"] for item in entries} | {sha(encode(item)) for item in entries}
    pending = [p for p in file_under(root, "pending").iterdir() if p.is_file() and sha(p.read_bytes()) not in accepted_hashes]
    if pending and audit_quarantine:
        quarantine_partial(root, pending, "Uncommitted pending files; never promoted automatically")
    tip_path = file_under(root, "chain_tip.json")
    expected = None if not entries else {"manifest_sequence": len(entries) - 1, "manifest_sha256": previous,
                                         "updated_at_utc": entries[-1]["manifest_created_at_utc"]}
    try:
        tip = load(tip_path) if tip_path.exists() else None
    except (ValueError, OSError):
        tip = {"invalid": True}
    return entries, expected, assess_tip(entries, tip)


def assess_tip(entries, tip):
    if tip is None:
        return {"status": "MISSING", "chain_valid": True, "readiness_pass": True, "warning": "Non-authoritative index absent"}
    good = isinstance(tip, dict) and set(tip) == {"manifest_sequence", "manifest_sha256", "updated_at_utc"}
    if good:
        number = tip["manifest_sequence"]
        good = type(number) is int and 0 <= number < len(entries)
        if good:
            try:
                good = tip["manifest_sha256"] == sha(encode(entries[number])) and utc(tip["updated_at_utc"]) == utc(entries[number]["manifest_created_at_utc"])
            except (ValueError, TypeError):
                good = False
    if not good:
        return {"status": "CONFLICTING", "chain_valid": True, "readiness_pass": False, "warning": "Rebuild index after full verification"}
    stale = number != len(entries) - 1
    return {"status": "STALE" if stale else "MATCH", "chain_valid": True, "readiness_pass": True,
            "warning": "Index lags verified chain" if stale else None}


@contextmanager
def writer_lock(root):
    lock = file_under(root, "writer.lock")
    lock.mkdir()  # Exclusive directory creation. Never steal a possibly live lock.
    try:
        yield
    finally:
        lock.rmdir()


def publish_tip(root, schema):
    """Caller holds writer_lock; contents are derived only by full verification."""
    require(file_under(root, "writer.lock").is_dir(), "Writer lock required")
    entries, tip, state = recover_chain(root, schema)
    require(entries and tip is not None, "No immutable chain to index")
    temporary = file_under(root, "pending") / (uuid.uuid4().hex + ".tip")
    with temporary.open("xb") as handle:
        handle.write(encode(tip))
        handle.flush()
        os.fsync(handle.fileno())
    require(load(temporary) == tip, "Index bytes mismatch")
    os.replace(temporary, file_under(root, "chain_tip.json"))


def rebuild_tip(root, schema):
    with writer_lock(root):
        recover_chain(root, schema)
        previous = file_under(root, "chain_tip.json")
        if previous.exists():
            quarantine_partial(root, [previous], "Explicit non-authoritative index rebuild receipt")
        publish_tip(root, schema)
    return recover_chain(root, schema)[2]


def append_snapshot(root, raw, schema, started, finished, revision_of=None, stage_hook=lambda stage: None):
    with writer_lock(root):
        entries, _, tip_state = recover_chain(root, schema, audit_quarantine=True)
        require(tip_state["readiness_pass"], "Conflicting index requires explicit rebuild")
        source, tz, activation = binding(root)
        try:
            summary = summarize(raw, source, tz)
            if entries:
                require(utc(started) > utc(entries[-1]["capture_finished_at_utc"]), "Capture timestamp reversal")
            require(utc(started) >= utc(activation["activation_effective_at_utc"]), "Capture predates activation")
            if revision_of:
                prior = next((e for e in entries if e["snapshot_id"] == revision_of), None)
                require(prior is not None, "Unknown revision parent")
                record_revision(prior, summary)
            else:
                same = [e for e in entries if e["timeframe"] == summary["timeframe"] and e["revision_of_snapshot_id"] is None]
                require(not same or utc(summary["first_source_timestamp"]) > utc(same[-1]["last_source_timestamp"]), "Duplicate/reversed bars; explicit revision required")
            number = len(entries)
            identity = f"{number:012d}-" + uuid.uuid4().hex
            item = {"manifest_version": "gold_future_capture_manifest_v2", "sequence": number, "snapshot_sequence": number,
                "previous_manifest_sha256": sha(encode(entries[-1])) if entries else None,
                "manifest_created_at_utc": now(), "snapshot_id": identity, "snapshot_path": "snapshots/" + identity + ".csv",
                **summary, "source_attestation_sha256": sha(encode(source)), "timezone_attestation_sha256": sha(encode(tz)),
                "activation_sha256": sha(encode(activation)), "source_id": source["source_id"], "symbol": source["symbol"],
                "capture_started_at_utc": started, "capture_finished_at_utc": finished,
                "collector_commit": activation["collector_commit"], "collector_version": VERSION,
                "revision_of_snapshot_id": revision_of, "partial_snapshot": False, "sealed": True, "notes": "Raw data only"}
            validate_schema(item, schema)
        except (ValueError, TypeError, KeyError):
            rejected = root / "pending" / (uuid.uuid4().hex + ".rejected")
            with rejected.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            quarantine_partial(root, [rejected], "Input rejected before chain publication")
            raise
        sealed_write(root, root / item["snapshot_path"], raw, lambda stage: stage_hook("snapshot_" + stage))
        stage_hook("snapshot_sealed")
        item["manifest_created_at_utc"] = now()
        validate_schema(item, schema)
        sealed_write(root, root / "manifests" / f"{number:012d}.json", encode(item), lambda stage: stage_hook("manifest_" + stage))
        stage_hook("manifest_sealed")
        publish_tip(root, schema)
        recover_chain(root, schema)
        return item


def verify_executable_binding(activation):
    raw = Path(__file__).read_bytes()
    commit = activation["collector_commit"]
    blob = subprocess.check_output(["git", "cat-file", "blob", commit + ":gold_future_capture_collector_v2.py"], cwd=ROOT)
    require(raw.replace(b"\r\n", b"\n") == blob.replace(b"\r\n", b"\n"), "Collector commit bytes changed")
    require(sha(raw) == activation["collector_review"]["collector_sha256"], "Code bytes changed; recertification required")


def capture_once(root, timeframe, count, certification):
    require(root.resolve().is_relative_to((ROOT / "future_holdout_capture").resolve()), "Research capture root required")
    enablement = load(ROOT / "gold_future_capture_protocol_v2.json")
    require(enablement["activation_permitted"] is True, "Enablement protocol has not authorized activation")
    require(1 <= count <= 1000 and timeframe in TIMEFRAMES, "Bounded native request required")
    source, tz, activation = binding(root)
    report = load(certification)
    require(report["readiness"]["status"] == "READY_FOR_CAPTURE_ACTIVATION"
            and report["collector_static_review"]["static_status"] == "STATICALLY_CONFORMANT", "Certification gate closed")
    require(report["collector_static_review"]["collector_sha256"] == sha(Path(__file__).read_bytes()), "Certification source mismatch")
    require(report["bindings"]["source_attestation_sha256"] == sha(encode(source))
            and report["bindings"]["timezone_attestation_sha256"] == sha(encode(tz))
            and report["bindings"]["activation_sha256"] == sha(encode(activation)), "Certification document mismatch")
    verify_executable_binding(activation)
    schema = load(ROOT / "gold_future_capture_manifest_schema_v2.json")
    require(sha(encode(schema)) == SCHEMA_SHA, "Frozen manifest schema changed")
    recover_chain(root, schema)
    import MetaTrader5 as mt5
    require(mt5.__version__ == enablement["MetaTrader5_version"], "Dependency change requires recertification")
    require(mt5.initialize(source["terminal_path_or_feed_reference"], timeout=10000), "MT5 initialize failed")
    try:
        info = mt5.symbol_info("GOLD#")
        require(info is not None and info.name == "GOLD#" and info.digits == source["price_digits"]
                and info.point == source["point_size"], "Live symbol binding mismatch")
        # Sensitive structure exists in memory only, never serialized or printed.
        identity = mt5.account_info()
        require(identity is not None and identity.server == source["broker_server_name"]
                and identity.company == source["broker_name"], "Live source binding mismatch")
        environment = "demo" if identity.trade_mode == 0 else "live" if identity.trade_mode == 2 else "unknown"
        del identity
        require(environment == source["source_account_environment"], "Live environment changed")
        started = now()
        rates = mt5.copy_rates_from_pos("GOLD#", TIMEFRAMES[timeframe], 1, count)
        finished = now()
        require(rates is not None and len(rates) > 0, "No raw bars")
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=FIELDS + source["volume_fields_if_present"], lineterminator="\n")
        writer.writeheader()
        for rate in rates:
            row = dict(zip(FIELDS, [normalize_epoch(rate["time"], tz), str(int(rate["time"])), source["source_id"], "GOLD#", timeframe,
                       float(rate["open"]), float(rate["high"]), float(rate["low"]), float(rate["close"]), int(rate["spread"])]))
            for field in source["volume_fields_if_present"]:
                row[field] = int(rate[field.lower()])
            writer.writerow(row)
        return append_snapshot(root, output.getvalue().encode(), schema, started, finished)
    finally:
        mt5.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--certification", type=Path, required=True)
    parser.add_argument("--timeframe", choices=TIMEFRAMES, default="M1")
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args()
    try:
        item = capture_once(args.capture_root.resolve(), args.timeframe, args.count, args.certification)
        print(json.dumps({"sequence": item["sequence"], "row_count": item["row_count"]}))
        return 0
    except Exception as error:
        print("CAPTURE_BLOCKED:" + type(error).__name__, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
