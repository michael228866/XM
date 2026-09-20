"""Read-only future GOLD data foundation; no strategy execution or collection."""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import math
import ntpath
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPEC_PATH = ROOT / "execution_spec_gold_future_holdout_data_foundation_v1.json"
SPEC_SHA256 = '993e40b207d2085ab40895f345d658a9c3d27a581ab27d1c5e55b1a0545651ae'
BANNED_MODULES = {"xgboost", "lightgbm", "sklearn", "torch", "stable_baselines3", "MetaTrader5"}
RAW_COLUMNS = {"OPEN", "HIGH", "LOW", "CLOSE"}
FORBIDDEN_FIELDS = {"win_rate", "profit_factor", "pnl", "mean_r", "drawdown", "trades_per_day",
                    "trade_count", "wins", "losses", "prediction", "probability", "signal_score"}
OUTCOME_PATH = re.compile(r"model|result|ledger|prediction|signal|trade_history|metrics|performance|paired_oof|secondary_evidence", re.I)
RAW_SYMBOL = re.compile(r"GOLD#?|XAUUSD", re.I)
TEXT_SUFFIXES = {".py", ".ps1", ".md", ".json", ".toml", ".ini", ".yaml", ".yml", ".txt"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_file(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


READ_ONLY_GIT_COMMANDS = {"rev-parse", "status", "rev-list", "cat-file"}


def is_allowed_git_subprocess(executable, argv):
    """Recognize direct Git argv, including CPython's Windows command line.

    Only simple whole-token quoting is supported; escaped/embedded quotes and
    shell metacharacters fail closed. No shell-language parser is involved.
    """
    if isinstance(argv, str):
        token = r'(?:[^\s"]+|"[^"\r\n]*")'
        if not re.fullmatch(token + r'(?:[ \t]+' + token + r')*', argv):
            return False
        tokens = re.findall(token, argv)
        # A backslash immediately before a closing quote has Windows escape
        # semantics; none of the audit's Git arguments need that ambiguity.
        if any(value.startswith('"') and value.endswith('\\"') for value in tokens):
            return False
        argv = [value[1:-1] if value.startswith('"') else value for value in tokens]
    if not isinstance(argv, (list, tuple)) or len(argv) < 2:
        return False
    values = [*argv, *([] if executable is None else [executable])]
    if any(not isinstance(value, str) or not value or
           any(ord(char) < 32 or char in '&|;<>"' for char in value)
           for value in values):
        return False
    if ntpath.basename(argv[0]).lower() not in {"git", "git.exe"}:
        return False
    if executable is not None and ntpath.basename(executable).lower() not in {"git", "git.exe"}:
        return False
    return argv[1] in READ_ONLY_GIT_COMMANDS


def git_read(repo, *args):
    require(args and args[0] in READ_ONLY_GIT_COMMANDS, "Read-only Git command required")
    return subprocess.check_output(["git", *args], cwd=repo, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})


class ImportGuard:
    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split(".")[0]
        if top == Path(__file__).stem:
            return None
        if top in BANNED_MODULES or top.startswith(("gold_", "validate_gold", "barrier_", "drl_")) or top == "gemini":
            raise RuntimeError("Audit cannot import research/model code: " + fullname)
        return None


def install_guards(output):
    """Defense in depth: process-wide no network, no writes except one output."""
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0, ImportGuard())
    require(not BANNED_MODULES.intersection(sys.modules), "Model package already imported")
    destination = Path(output).resolve() if output else None

    def guard(event, args):
        if event == "open":
            path, mode, flags = args
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if writing:
                require(isinstance(path, (str, bytes, os.PathLike)) and destination is not None
                        and Path(os.fsdecode(path)).resolve() == destination, "Audit write outside explicit output")
        elif event in {"os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.chmod", "os.truncate", "os.link", "os.symlink", "os.utime", "os.system"}:
            raise RuntimeError("Filesystem mutation prohibited: " + event)
        elif event.startswith("socket."):
            raise RuntimeError("Audit network access prohibited")
        elif event == "subprocess.Popen":
            # CPython audit layout: (executable, args, cwd, env). On Windows
            # args is already serialized by subprocess.list2cmdline().
            require(len(args) == 4 and is_allowed_git_subprocess(args[0], args[1]),
                    "Only read-only Git subprocesses permitted")

    sys.addaudithook(guard)


def parse_timestamp(value):
    value = value.strip()
    value = value[:10].replace(".", "-").replace("/", "-") + value[10:]
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def freshness(last, now, comparable, spec):
    if not last or not comparable:
        return {"stale_age_seconds_if_derivable": None, "freshness_status": "UNKNOWN",
                "reason": "Timestamp offset/source conversion is not established"}
    value = parse_timestamp(last)
    if value.tzinfo is None or now.tzinfo is None:
        return freshness(None, now, False, spec)
    age = (now - value).total_seconds()
    if age < 0:
        return {"stale_age_seconds_if_derivable": None, "freshness_status": "UNKNOWN",
                "reason": "Timestamp lies in the future; source clock requires review"}
    limits = spec["freshness_seconds"]
    return {"stale_age_seconds_if_derivable": age,
            "freshness_status": "CURRENT" if age <= limits["current_max"] else
            "RECENT" if age <= limits["recent_max"] else "STALE",
            "reason": "Data operations age from explicit timestamp offset, not broker-clock certification"}


def classify_source(path, columns=()):
    lowered = {c.lower() for c in columns}
    if OUTCOME_PATH.search(str(path)) or lowered & FORBIDDEN_FIELDS or any(
            re.search(r"target|label|score|probab|predict|profit|pnl|realized_wr|net_r|signal|trade|entry|exit|reward", c) for c in lowered):
        return "RESULT_OR_OUTCOME_ARTIFACT"
    if re.search(r"derived|resampl|aggregat", str(path), re.I) or Path(path).suffix.lower() in {".npz", ".npy", ".parquet", ".pkl"}:
        return "DERIVED_SOURCE"
    if RAW_COLUMNS <= set(columns) or {"BID", "ASK"} <= set(columns):
        return "VERIFIED_LOCAL_RAW_SOURCE"
    return "POTENTIAL_RAW_SOURCE"


def inspect_csv(binary, path, now, spec):
    """Stream permitted raw columns only; reject result schemas at the header."""
    prefix = binary.read(4)
    binary.seek(0)
    encoding = "utf-16" if prefix.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    stream = io.TextIOWrapper(binary, encoding=encoding, newline="")
    header = stream.readline(16385)
    require(len(header) <= 16384, "CSV header exceeds limit")
    delimiter = max((",", "\t", ";"), key=header.count)
    stream.seek(0)
    reader = csv.DictReader(stream, delimiter=delimiter)
    names = {name: name.strip().strip("<>").upper() for name in reader.fieldnames or []}
    require(len(set(names.values())) == len(names), "Duplicate normalized column names")
    columns = set(names.values())
    classification = classify_source(path, columns)
    base = {"path": str(path), "classification": classification, "encoding": encoding}
    if classification == "RESULT_OR_OUTCOME_ARTIFACT":
        return {**base, "inspection": "header only; body not read"}
    if not (RAW_COLUMNS <= columns or {"BID", "ASK"} <= columns):
        return {**base, "inspection": "unrecognized raw schema; body not read"}
    row_count = duplicates = reversals = invalid = nonfinite = bad_prices = spread_good = 0
    seen, days, symbols = set(), set(), set()
    previous = first = last = None
    aware = None
    mixed_clock = False
    gaps = []
    gap_count = 0
    timeframe = infer_timeframe(path, spec["required_timeframes"])
    interval = nominal_interval(timeframe)
    for raw_row in reader:
        row_count += 1
        row = {names[k]: v for k, v in raw_row.items() if k in names}
        text = next((row.get(k) for k in ("BAR_OPEN_UTC", "TIMESTAMP_UTC", "TIME_DT", "TIMESTAMP") if row.get(k)), None)
        if text is None:
            text = (row.get("DATE") or "") + (" " + row["TIME"] if row.get("TIME") else "")
        try:
            timestamp = parse_timestamp(text)
            this_aware = timestamp.tzinfo is not None
            if aware is None:
                aware = this_aware
            if aware != this_aware:
                mixed_clock = True
                raise ValueError("Mixed naive/offset clocks")
            duplicates += int(timestamp in seen)
            reversals += int(previous is not None and timestamp < previous)
            if previous is not None and interval and (timestamp - previous).total_seconds() > interval:
                gap_count += 1
                if len(gaps) < 20:
                    gaps.append({"after": previous.isoformat(), "before": timestamp.isoformat(),
                                 "interpretation": "OBSERVED_PATTERN_NOT_PROVEN"})
            seen.add(timestamp)
            days.add(timestamp.date())
            previous = timestamp
            first = timestamp if first is None else min(first, timestamp)
            last = timestamp if last is None else max(last, timestamp)
        except (ValueError, TypeError, OverflowError):
            invalid += 1
        prices = {}
        for name in RAW_COLUMNS if RAW_COLUMNS <= columns else ("BID", "ASK"):
            try:
                number = float(row.get(name, ""))
                if math.isfinite(number):
                    prices[name] = number
            except (TypeError, ValueError):
                pass
        required = RAW_COLUMNS if RAW_COLUMNS <= columns else {"BID", "ASK"}
        nonfinite += int(set(prices) != required)
        if set(prices) == required:
            valid = all(v > 0 for v in prices.values())
            if required == RAW_COLUMNS:
                valid &= prices["LOW"] <= min(prices.values()) and prices["HIGH"] >= max(prices.values())
            else:
                valid &= prices["ASK"] >= prices["BID"]
            bad_prices += int(not valid)
        try:
            spread = float(row.get("SPREAD", row.get("SPREAD_POINTS", "nan")))
            spread_good += int(math.isfinite(spread) and spread > 0)
        except (TypeError, ValueError):
            pass
        symbol = row.get("SOURCE_SYMBOL")
        if symbol and re.fullmatch(r"GOLD#?|XAUUSD", symbol, re.I):
            symbols.add(symbol)
    # The existing collector transforms broker epochs using an empirical rule.
    # A UTC-looking output column alone does not certify that transformation.
    transformed = "RAW_BROKER_EPOCH_SECONDS" in columns or "RAW_BROKER_TIME_MSC" in columns
    comparable = bool(aware and not mixed_clock and not invalid and not transformed)
    return {**base, "timeframe": timeframe, "columns": sorted(columns), "row_count": row_count,
            "first_timestamp": first.isoformat() if first else None,
            "last_timestamp": last.isoformat() if last else None,
            "duplicate_timestamps": duplicates, "non_monotonic_timestamps": reversals,
            "invalid_timestamps": invalid, "nonfinite_rows": nonfinite, "invalid_price_rows": bad_prices,
            "has_spread": "SPREAD" in columns, "has_spread_alias": "SPREAD_POINTS" in columns,
            "finite_positive_spread_rate": spread_good / row_count if row_count else None,
            "spread_fallback_rows": row_count - spread_good,
            "spread_alias_requires_adapter": "SPREAD_POINTS" in columns and "SPREAD" not in columns,
            "observed_dates": len(days), "symbols": sorted(symbols), "gap_count": gap_count,
            "gap_examples": gaps, "gap_examples_limited": gap_count > len(gaps),
            "timestamp_status": "INVALID" if invalid or mixed_clock else "OFFSET_DECLARED" if aware else "NAIVE_CLOCK",
            "broker_conversion_unverified": transformed,
            **freshness(last.isoformat() if last else None, now, comparable, spec)}


def infer_timeframe(path, timeframes):
    tokens = re.split(r"[\\/_.\-]+", str(path))
    for tf in timeframes:
        if tf.lower() in [token.lower() for token in tokens]:
            return tf
    return None


def nominal_interval(tf):
    if tf and re.fullmatch(r"[MH]\d+", tf):
        return int(tf[1:]) * (60 if tf[0] == "M" else 3600)
    return {"Daily": 86400, "Weekly": 604800, "Monthly": 2678400}.get(tf)


def walk_files(root, spec, limitations):
    def on_error(error):
        limitations.append("Unreadable directory: " + str(error.filename))
    for directory, children, filenames in os.walk(root, followlinks=False, onerror=on_error):
        children[:] = sorted(name for name in children if name not in spec["scan_excludes"]
                             and not Path(directory, name).is_symlink()
                             and Path(directory, name).resolve().is_relative_to(root.resolve()))
        for name in sorted(filenames):
            path = Path(directory, name)
            if not path.is_symlink():
                yield path


def text_evidence(path, spec):
    """Extract categories and literal local paths, never execute source/config."""
    content = path.read_bytes()
    encoding = "utf-16" if content.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    source = content.decode(encoding)
    categories = {
        "live_api_reference": r"MetaTrader5|copy_rates|copy_ticks|broker.api",
        "collector_or_export": r"collect|download|export|to_csv|append_rows",
        "timezone_or_dst": r"timezone|UTC.offset|EET|EEST|DST|broker_epoch|server.time",
        "bar_or_calendar": r"bar.open|bar.close|rollover|weekend|holiday|session",
        "capture_integrity": r"sha256|manifest|append.only|restart|retention|fsync",
        "schedule_or_terminal": r"terminal64|scheduled|startup|monitor|XM_TERMINAL_PATH",
    }
    evidence = []
    total = 0
    for number, line in enumerate(source.splitlines(), 1):
        kinds = [key for key, pattern in categories.items() if re.search(pattern, line, re.I)]
        if kinds:
            total += 1
            if len(evidence) < spec["evidence_limit_per_file"]:
                evidence.append({"path": str(path), "line": number, "categories": kinds,
                                 "classification": "CONFIG_REFERENCE", "authority": "STATIC_REFERENCE_ONLY"})
    literals = []
    if path.suffix.lower() == ".py":
        try:
            tree = ast.parse(source)
            literals = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        except SyntaxError:
            pass
    # For non-Python configs, recognize only literal absolute Windows paths.
    literals += [value.replace("\\\\", "\\") for value in re.findall(r'[A-Za-z]:[\\/][^\r\n"<>]+', source)]
    references = sorted({value for value in literals if re.fullmatch(r'[A-Za-z]:[\\/][^\r\n"<>]+', value)})
    return evidence, references, total > len(evidence)


def validate_manifest_chain(entries, fields):
    """Validate proposed metadata linkage only; this does not certify raw files."""
    previous_hash = None
    previous_time = None
    for number, entry in enumerate(entries):
        require(set(fields) <= set(entry), "Missing manifest fields")
        require(type(entry["sequence"]) is int and entry["sequence"] == number, "Manifest sequence")
        require(entry["previous_manifest_sha256"] == previous_hash, "Manifest linkage")
        for key in ("raw_sha256", "schema_sha256"):
            require(isinstance(entry[key], str) and re.fullmatch(r"[0-9a-f]{64}", entry[key]), "Manifest SHA256")
        require(re.fullmatch(r"[0-9a-f]{40,64}", entry["collector_commit"]), "Collector commit")
        captured = parse_timestamp(entry["captured_at_utc"])
        first = parse_timestamp(entry["first_source_timestamp"])
        last = parse_timestamp(entry["last_source_timestamp"])
        require(captured.tzinfo is not None and captured.utcoffset() == timedelta(0), "Capture clock must be UTC")
        require(first.tzinfo is not None and last.tzinfo is not None and first <= last <= captured, "Source chronology")
        require(previous_time is None or captured > previous_time, "Capture time order")
        for key in ("row_count", "duplicate_timestamps", "non_monotonic_timestamps"):
            require(type(entry[key]) is int and entry[key] >= 0, "Manifest count")
        require(entry["row_count"] > 0 and entry["source_identity"] and entry["source_path"], "Manifest identity")
        previous_hash, previous_time = json_hash(entry), captured
    return bool(entries)


def capture_design(spec, collector_verified):
    return {"recommended_capture_mode": spec["capture_design"]["mode"],
            "append_only_possible": True, "hash_chain_possible": True, "manifest_possible": True,
            "raw_file_hashing_possible": True, "snapshot_interval": spec["capture_design"]["snapshot_interval"],
            "mutation_detection_possible": True, "restart_recovery_possible": True, "data_retention_possible": True,
            "capability_basis": "DESIGN_FEASIBILITY_ONLY; not activated or operationally verified",
            "existing_collector_source_verified": collector_verified,
            "existing_collector": "M1/ticks append; rewritable state/manifest; not a linked immutable raw snapshot chain" if collector_verified else "UNVERIFIED",
            "required_manifest_fields": spec["capture_design"]["manifest_fields"],
            "restart_rule": "Seal and fsync snapshot before linked manifest; recover only from last verified link; quarantine partial/orphan snapshots; deduplicate stable source bar identities",
            "retention_rule": "Retain original raw bytes and every linked manifest; restrict writes after seal; externally anchor the chain tip",
            "blockers": ["No activated sealed-snapshot hash chain verified", "Collector ACLs, storage capacity, durable writes and recovery require operational certification"]}


def readiness(blockers):
    codes = sorted({code for code, _ in blockers})
    return {"status": "READY_TO_FREEZE_PROTOCOL" if not codes else codes[0] if len(codes) == 1 else "NOT_READY_MULTIPLE_BLOCKERS",
            "blockers": [{"status": code, "reason": reason} for code, reason in blockers],
            "warnings": ["Local references do not prove a live service; this audit neither creates nor certifies a holdout"]}


def holdout_boundary(spec):
    return {"historical_start_allowed": False, "historical_2025_plus_allowed": False,
            "boundary_rule": spec["future_boundary_rule"], "earliest_possible_future_start": None,
            "boundary_is_currently_certified": False, "holdout_start": None,
            "blockers": spec["boundary_prerequisites"]}


def assert_report_schema(value):
    if isinstance(value, dict):
        require(not {str(key).lower() for key in value} & FORBIDDEN_FIELDS, "Forbidden report field")
        for child in value.values():
            assert_report_schema(child)
    elif isinstance(value, list):
        for child in value:
            assert_report_schema(child)


def build_report(spec, repo, commit, dirty, now, inventory, references, configs, limitations, trace):
    raw = [row for row in inventory if row["classification"] in {"VERIFIED_LOCAL_RAW_SOURCE", "DERIVED_SOURCE"}
           and "row_count" in row]
    m1_sources = [row for row in raw if row.get("timeframe") == "M1"]
    canonical = [row for row in m1_sources if Path(row["path"]).parent == repo and Path(row["path"]).name.startswith("GOLD#_")]
    selected = canonical[0] if len(canonical) == 1 else m1_sources[0] if len(m1_sources) == 1 else None
    m1 = dict(selected) if selected else {"source_present": False, "first_timestamp": None, "last_timestamp": None,
        "row_count": 0, "columns": [], "has_spread": False, "duplicate_timestamps": None,
        "non_monotonic_timestamps": None, "invalid_timestamps": None, "nonfinite_rows": None,
        "stale_age_seconds_if_derivable": None, "freshness_status": "UNKNOWN"}
    m1["source_present"] = bool(m1_sources)
    m1["selection_basis"] = "unique historical canonical M1, otherwise unique M1 candidate; no merging or symbol substitution"
    m1["source_candidates"] = [row["path"] for row in m1_sources]
    blockers = []
    if not m1_sources:
        blockers.append(("NOT_READY_NO_FRESH_SOURCE", "No schema-verified M1 raw source found"))
    elif not selected:
        blockers.append(("NOT_READY_PROVENANCE", "Multiple M1 sources; no unambiguous source binding"))
    if selected and selected["freshness_status"] == "STALE":
        blockers.append(("NOT_READY_STALE_SOURCE", "Selected M1 source exceeds operational freshness threshold"))
    # Static references can establish implementation capabilities, not liveness.
    blockers.append(("NOT_READY_NO_FRESH_SOURCE", "No continuing live source was certified by this offline inspection"))
    rows = []
    for tf in spec["required_timeframes"]:
        sources = [row for row in raw if row.get("timeframe") == tf]
        native = [row for row in sources if row["classification"] == "VERIFIED_LOCAL_RAW_SOURCE"]
        item = sources[0] if len(sources) == 1 else {}
        derived = bool(sources and not native)
        rows.append({"timeframe": tf, "source_mode": "DERIVED" if derived else "FILE_SOURCE" if sources else "MISSING",
            "source_path": [r["path"] for r in sources], "first_timestamp": item.get("first_timestamp"),
            "last_timestamp": item.get("last_timestamp"), "row_count": item.get("row_count"),
            "columns": item.get("columns", []), "timestamp_status": item.get("timestamp_status", "UNRESOLVED"),
            "freshness_status": item.get("freshness_status", "UNKNOWN"),
            "native_or_derived": "DERIVED" if derived else "NATIVE_ORIGIN_UNATTESTED" if sources else "UNKNOWN",
            "pipeline_compatibility": "PROTOCOL_CHANGE_REQUIRED" if derived else "STRUCTURAL_ONLY" if len(sources) == 1 else "UNRESOLVED"})
        if not sources:
            blockers.append(("NOT_READY_INCOMPLETE_TIMEFRAMES", tf + " is absent"))
        elif derived:
            blockers.append(("NOT_READY_PIPELINE_MISMATCH", tf + " resampling requires a new protocol"))
        elif len(sources) != 1:
            blockers.append(("NOT_READY_PROVENANCE", tf + " partition/duplicate-file relationship needs a source manifest"))
    structural = all(len([r for r in raw if r.get("timeframe") == tf and RAW_COLUMNS <= set(r.get("columns", []))
                         and r.get("row_count", 0) >= (254 if tf == "M1" else 21)
                         and not any(r.get(k) for k in ("invalid_timestamps", "duplicate_timestamps", "non_monotonic_timestamps", "nonfinite_rows", "invalid_price_rows"))]) == 1
                     for tf in spec["required_timeframes"])
    execution_structural = bool(selected and RAW_COLUMNS <= set(selected.get("columns", [])) and
                                selected.get("has_spread") and selected.get("row_count", 0) >= 15 and
                                not any(selected.get(k) for k in ("invalid_timestamps", "duplicate_timestamps", "non_monotonic_timestamps", "nonfinite_rows", "invalid_price_rows")))
    tz_evidence = [r for r in references if "timezone_or_dst" in r.get("categories", [])]
    timezone_info = {"declared_timezone": "Existing collector declares UTC converted from empirical EET/EEST" if any(
        "gold_data_foundation" in r["path"] for r in tz_evidence) else None,
        "evidence": tz_evidence, "dst_policy": "Source-level empirical +2/+3 rule; no authoritative broker attestation verified",
        "timestamp_is_bar_open_or_close": "BAR_OPEN_UTC is a declaration only; other CSV labels unresolved",
        "ambiguity": ["Broker versus API epoch interpretation", "Daily rollover, weekend/holiday calendar and native bar boundaries"],
        "status": "UNRESOLVED"}
    blockers.append(("NOT_READY_TIMEZONE", "No authoritative source-specific timezone/DST/bar-calendar certification"))
    prefix = {"recursive_ewm_requires_prefix": True, "prefix_available": bool(selected and selected.get("row_count", 0) >= 254),
              "prefix_identity_attested": False, "context_start": m1.get("first_timestamp"), "holdout_start": None,
              "blockers": ["254 M1 rows cover finite windows, not exact recursive EMA state",
                           "Preserve and attest original source prefix or predeclared state initialization; context is never test evidence"]}
    collector_verified = any(row["path"] == "gold_data_foundation_forward_collector.py" and row["matches"] for row in trace)
    capture = capture_design(spec, collector_verified)
    blockers.extend(("NOT_READY_CAPTURE_INTEGRITY", reason) for reason in capture["blockers"])
    blockers.append(("NOT_READY_PROVENANCE", "Native source identity and full recursive prefix are not attested"))
    if limitations or any(not r["matches"] for r in trace):
        blockers.append(("NOT_READY_PROVENANCE", "Source drift or scan limitations require review"))
    if not structural or not execution_structural:
        blockers.append(("NOT_READY_PIPELINE_MISMATCH", "Raw structural requirements are missing, ambiguous or invalid"))
    stale_end = parse_timestamp(spec["historical_canonical_end"])
    for item in inventory:
        latest = item.get("last_timestamp")
        value = parse_timestamp(latest) if latest else None
        # Only unchanged historical canonical filenames establish a comparable
        # naive clock. Do not compare a collector's UTC conversion to that clock.
        same_clock = Path(item["path"]).parent == repo and Path(item["path"]).name.startswith("GOLD#_")
        item["extends_stale_canonical_clock"] = value > stale_end if value and value.tzinfo is None and same_clock else None
    return {
        "audit_version": spec["experiment_name"], "repo_commit": commit, "repo_dirty": dirty,
        "generated_at_utc": now.isoformat(),
        "historical_context": {"previous_untouched_status": "NOT_READY_CONTAMINATED",
            "previous_canonical_last_available": spec["historical_canonical_end"],
            "historical_data_is_promotion_untouched": False, "basis": "User-supplied previous audit conclusion; existing report not opened"},
        "source_discovery": {"repository_sources": [r for r in inventory if Path(r["path"]).is_relative_to(repo)],
            "external_local_sources": [r for r in inventory if not Path(r["path"]).is_relative_to(repo)],
            "source_references": references, "source_configs": configs, "unresolved_sources": limitations,
            "source_identity_checks": trace},
        "canonical_source_candidate": {"source_type": "LOCAL_CSV" if selected else "UNRESOLVED",
            "symbol": selected.get("symbols") or "GOLD#/XAUUSD filename only; exact broker binding unresolved" if selected else None,
            "path_or_reference": selected["path"] if selected else None, "provenance_status": "UNATTESTED_SOURCE_ORIGIN",
            "append_mode": "UNKNOWN", "writable_by_collector": None, "immutable_after_capture": None,
            "can_continue_forward": None, "latest_observed_timestamp": m1.get("last_timestamp"),
            "earliest_observed_timestamp": m1.get("first_timestamp")},
        "symbols_found": sorted({symbol for row in raw for symbol in row.get("symbols", [])} |
                                {symbol.upper() for row in inventory for symbol in RAW_SYMBOL.findall(row["path"])}),
        "timezone_provenance": timezone_info, "m1_status": m1, "required_timeframes": rows,
        "frozen_pipeline_requirements": {"required_features": spec["required_features"],
            "required_execution_inputs": spec["required_execution_inputs"], "required_timeframes": spec["required_timeframes"],
            "historical_native_timeframe_semantics": "Separate GOLD# timeframe CSVs; lexical duplicate overwrite; shifted native trend/backward asof/global M1 lag",
            "deterministic_resampling_possible": "CONDITIONAL on certified broker bar boundaries/calendar and complete M1",
            "deterministic_resampling_would_change_protocol": True, "resampling_status": "PROTOCOL_CHANGE_REQUIRED"},
        "feature_reconstruction": {"structurally_possible": structural, "exact_reconstruction_possible": False,
            "blockers": ["Native bar provenance/calendar and exact recursive prefix are not certified"]},
        "execution_reconstruction": {"structurally_possible": execution_structural, "exact_reconstruction_possible": False,
            "blockers": ["Need certified M1 clock, SPREAD units and closed-bar semantics; SPREAD_POINTS needs an explicit reviewed adapter"]},
        "prefix_state": prefix, "future_capture_design": capture, "holdout_boundary": holdout_boundary(spec),
        "future_holdout_readiness": readiness(blockers), "production_promotion": False, "production_change": False,
    }


def audit(repo, source_roots, spec):
    now = datetime.now(timezone.utc)
    limitations, references, configs, inventory, trace = [], [], [], [], []
    commit = git_read(repo, "rev-parse", "HEAD").decode("ascii").strip()
    dirty = bool(git_read(repo, "status", "--porcelain"))
    files = set(walk_files(repo, spec, limitations))
    external = set(source_roots)
    for name, expected in spec["source_sha256"].items():
        path = repo / name
        actual = digest_file(path) if path.is_file() else None
        trace.append({"path": name, "sha256": actual, "expected_sha256": expected, "matches": actual == expected})
    for path in sorted(files):
        if path.name in {Path(__file__).name, SPEC_PATH.name}:
            continue
        # Outcome/model reports are metadata-only, including archived JSONs.
        safe_text = path.suffix.lower() in {".py", ".ps1", ".ini", ".toml", ".yaml", ".yml"} or path.name == "AGENTS.md" or (
            path.suffix.lower() == ".json" and "config" in path.name.lower())
        if not safe_text or OUTCOME_PATH.search(str(path)):
            continue
        try:
            if path.stat().st_size > spec["text_scan_max_bytes"]:
                limitations.append("Text scan size limit: " + str(path))
                continue
            evidence, paths, limited = text_evidence(path, spec)
            references.extend(evidence)
            if limited:
                limitations.append("Evidence line limit: " + str(path))
            for literal in paths:
                candidate = Path(literal)
                exists = candidate.exists()
                configs.append({"path": str(candidate), "referenced_by": str(path), "exists": exists,
                                "classification": "CONFIG_REFERENCE" if exists else "UNRESOLVED_REFERENCE"})
                if exists and candidate.is_dir() and not candidate.resolve().is_relative_to(repo):
                    # Do not crawl a whole drive from an incidental source literal.
                    if len(candidate.parts) > 1:
                        external.add(candidate.resolve())
                elif exists and candidate.suffix.lower() in {".csv", ".gz"}:
                    files.add(candidate.resolve())
                elif not exists:
                    limitations.append("Unresolved literal source: " + str(candidate))
        except (OSError, UnicodeError, ValueError) as error:
            limitations.append("Static scan unavailable: " + str(path) + ": " + type(error).__name__)
    for root in sorted(external):
        if root.is_dir():
            files.update(walk_files(root, spec, limitations))
        elif root.is_file():
            files.add(root)
        else:
            limitations.append("Source root unavailable: " + str(root))
    for path in sorted(files):
        explicitly_supplied = any(path == root or path.is_relative_to(root) for root in source_roots)
        if (not RAW_SYMBOL.search(str(path)) and not explicitly_supplied) or path.suffix.lower() not in {".csv", ".gz", ".json", ".npz", ".npy", ".parquet", ".pkl"}:
            continue
        if path.name == SPEC_PATH.name or "future_holdout_data_foundation_report" in path.name:
            continue
        try:
            before = path.stat()
            row = {"path": str(path), "size_bytes": before.st_size, "classification": classify_source(path),
                   "inspection": "metadata only"}
            if row["classification"] != "RESULT_OR_OUTCOME_ARTIFACT" and (path.suffix.lower() == ".csv" or path.name.lower().endswith(".csv.gz")):
                opener = gzip.open if path.suffix.lower() == ".gz" else open
                with opener(path, "rb") as stream:
                    row.update(inspect_csv(stream, path, now, spec))
                if row["classification"] != "RESULT_OR_OUTCOME_ARTIFACT":
                    row["sha256"] = digest_file(path)
                after = path.stat()
                row["changed_during_inspection"] = (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                if row["changed_during_inspection"]:
                    row["classification"] = "POTENTIAL_RAW_SOURCE"
                    limitations.append("Source changed while reading: " + str(path))
            elif path.suffix.lower() in {".json", ".npz", ".npy", ".parquet", ".pkl"}:
                limitations.append("Body intentionally not inspected: " + str(path))
            inventory.append(row)
        except (OSError, ValueError, UnicodeError, csv.Error) as error:
            limitations.append("Raw inspection unavailable: " + str(path) + ": " + type(error).__name__)
    return build_report(spec, repo, commit, dirty, now, inventory, references, configs, limitations, trace)


def self_test(spec):
    now = datetime(2030, 6, 1, 12, tzinfo=timezone.utc)
    repo = Path("D:/XM/數據")
    path = repo / "GOLD#_M1_fixture.csv"
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        for delimiter in (",", "\t", ";"):
            text = delimiter.join(["TIMESTAMP", "OPEN", "HIGH", "LOW", "CLOSE", "SPREAD"]) + "\n"
            text += delimiter.join(["2030-06-01T11:59:00+00:00", "2", "3", "1", "2", "30"]) + "\n"
            row = inspect_csv(io.BytesIO(text.encode(encoding)), path, now, spec)
            require(row["freshness_status"] == "CURRENT" and row["has_spread"] and row["row_count"] == 1,
                    "CSV encoding/delimiter/current M1")
    for hours, expected in ((6, "CURRENT"), (7, "RECENT"), (72, "RECENT"), (73, "STALE")):
        require(freshness((now - timedelta(hours=hours)).isoformat(), now, True, spec)["freshness_status"] == expected,
                "Frozen freshness boundary")
    require(freshness("2030-06-01T11:59:00", now, False, spec)["stale_age_seconds_if_derivable"] is None, "Unknown clock age")
    require(freshness((now + timedelta(minutes=1)).isoformat(), now, True, spec)["freshness_status"] == "UNKNOWN", "Future timestamp")
    fixture = "DATE,TIME,OPEN,HIGH,LOW,CLOSE\n2030.03.31,01:00:00,2,3,1,2\n2030.03.31,03:00:00,2,3,1,2\n2030.03.31,03:00:00,2,3,1,2\n2030.03.31,02:59:00,2,3,1,2\ninvalid,date,nan,3,1,2\n"
    bad = inspect_csv(io.BytesIO(fixture.encode()), path, now, spec)
    require(bad["duplicate_timestamps"] == 1 and bad["non_monotonic_timestamps"] == 1 and
            bad["invalid_timestamps"] == 1 and bad["nonfinite_rows"] == 1 and not bad["has_spread"], "Raw defects")
    require(bad["gap_examples"][0]["interpretation"] == "OBSERVED_PATTERN_NOT_PROVEN" and
            bad["freshness_status"] == "UNKNOWN", "DST pattern cannot certify timezone")
    require(classify_source("GOLD#_M1_derived.csv", RAW_COLUMNS) == "DERIVED_SOURCE", "Derived source")
    require(classify_source(path, RAW_COLUMNS) == "VERIFIED_LOCAL_RAW_SOURCE", "Raw schema")
    require(classify_source("GOLD_trade_history.csv", RAW_COLUMNS) == "RESULT_OR_OUTCOME_ARTIFACT", "Outcome path")
    # Explicit forbidden-field fixtures; these names never enter a report.
    for field in FORBIDDEN_FIELDS:
        rejected = inspect_csv(io.BytesIO(("OPEN,HIGH,LOW,CLOSE," + field + "\n").encode()), path, now, spec)
        require(rejected["classification"] == "RESULT_OR_OUTCOME_ARTIFACT" and "row_count" not in rejected, "Outcome schema blocked")
        try:
            assert_report_schema({field: None})
        except ValueError:
            pass
        else:
            raise AssertionError("Forbidden report field accepted")
    capture = capture_design(spec, True)
    require(capture["append_only_possible"] and capture["hash_chain_possible"] and capture["blockers"], "Design is not activation")
    entry = {"sequence": 0, "previous_manifest_sha256": None, "source_identity": "synthetic GOLD#",
             "captured_at_utc": now.isoformat(), "source_path": str(path), "raw_sha256": "a" * 64,
             "first_source_timestamp": (now - timedelta(minutes=1)).isoformat(),
             "last_source_timestamp": (now - timedelta(minutes=1)).isoformat(), "row_count": 1,
             "schema_sha256": "b" * 64, "duplicate_timestamps": 0, "non_monotonic_timestamps": 0,
             "collector_commit": "c" * 40}
    second = {**entry, "sequence": 1, "previous_manifest_sha256": json_hash(entry),
              "captured_at_utc": (now + timedelta(minutes=1)).isoformat()}
    fields = spec["capture_design"]["manifest_fields"]
    require(validate_manifest_chain([entry, second], fields), "Manifest chain")
    for change in ({"sequence": 3}, {"previous_manifest_sha256": "f" * 64}, {"raw_sha256": "bad"},
                   {"captured_at_utc": now.isoformat()}, {"row_count": -1}):
        try:
            validate_manifest_chain([entry, {**second, **change}], fields)
        except ValueError:
            pass
        else:
            raise AssertionError("Broken manifest accepted")
    require(readiness([])["status"] == "READY_TO_FREEZE_PROTOCOL", "Ready branch")
    for code in ("NOT_READY_NO_FRESH_SOURCE", "NOT_READY_STALE_SOURCE", "NOT_READY_TIMEZONE",
                 "NOT_READY_INCOMPLETE_TIMEFRAMES", "NOT_READY_PROVENANCE", "NOT_READY_CAPTURE_INTEGRITY",
                 "NOT_READY_PIPELINE_MISMATCH"):
        require(readiness([(code, "synthetic")])["status"] == code, "Single readiness blocker")
    require(readiness([("NOT_READY_TIMEZONE", "clock"), ("NOT_READY_INCOMPLETE_TIMEFRAMES", "H12")])["status"] ==
            "NOT_READY_MULTIPLE_BLOCKERS", "Multiple blockers take precedence")
    report = build_report(spec, repo, "c" * 40, False, now, [row], [], [], [], [])
    require(not report["historical_context"]["historical_data_is_promotion_untouched"] and
            report["holdout_boundary"]["holdout_start"] is None and
            report["holdout_boundary"]["earliest_possible_future_start"] is None, "No historical or automatic holdout")
    require(report["timezone_provenance"]["status"] == "UNRESOLVED" and any(
        b["status"] == "NOT_READY_INCOMPLETE_TIMEFRAMES" for b in report["future_holdout_readiness"]["blockers"]), "Report gates")
    assert_report_schema(report)
    require("數據" in json.dumps(report, ensure_ascii=False), "Unicode path roundtrip")
    for name in BANNED_MODULES | {"gold_secondary_s4_confirmation_v1", "barrier_final_train", "gemini"}:
        try:
            ImportGuard().find_spec(name)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Forbidden import accepted")
    for event, args in (("socket.connect", (None, ("synthetic", 1))),
                        ("open", (str(path), "w", os.O_WRONLY))):
        try:
            sys.audit(event, *args)
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError("Network/filesystem guard failed")
    for command, tail in (("rev-parse", ["HEAD"]), ("status", ["--short"]),
                          ("rev-list", ["--objects", "--all"]), ("cat-file", ["-p", "abc"])):
        for binary in ("git", "git.exe", r"C:\Program Files\Git\cmd\git.exe"):
            argv = [binary, command, *tail]
            for representation in (argv, tuple(argv), subprocess.list2cmdline(argv)):
                require(is_allowed_git_subprocess(binary, representation), "Windows Git representation")
                sys.audit("subprocess.Popen", None, representation, "synthetic", {})
    rejected = [["git", cmd] for cmd in ("add", "commit", "checkout", "restore", "reset", "clean", "fetch", "pull", "push", "gc", "config")]
    rejected += [["python", "x.py"], ["powershell", "..."], ["cmd", "/c", "git status"]]
    rejected += [["git", "status", symbol, "echo"] for symbol in ("&", "&&", "|", ";")]
    for argv in rejected:
        for representation in (argv, subprocess.list2cmdline(argv)):
            require(not is_allowed_git_subprocess(None, representation), "Forbidden subprocess")
            try:
                sys.audit("subprocess.Popen", None, representation, "synthetic", {})
            except ValueError:
                pass
            else:
                raise AssertionError("Subprocess guard failed")
    print("SELF_TEST_PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--output", type=Path)
    parser.add_argument("--source-root", type=Path, action="append", default=[])
    parser.add_argument("--repo-root", type=Path)
    args = parser.parse_args()
    try:
        spec = json.loads(SPEC_PATH.read_text(encoding="utf-8-sig"))
        require(json_hash(spec) == SPEC_SHA256, "Frozen execution spec identity mismatch")
        if args.self_test:
            require(not args.source_root and args.repo_root is None, "Synthetic self-test cannot inspect roots")
            install_guards(None)
            self_test(spec)
            return 0
        for path in [args.output, args.repo_root or ROOT, *args.source_root]:
            require(not ntpath.splitdrive(str(path))[0].startswith("\\\\"), "Only local paths; no UNC/network shares")
        output = args.output.resolve()
        require(output.suffix.lower() == ".json" and not output.exists() and output.parent.is_dir(), "Explicit output must be new JSON in existing directory")
        require(not any(p.lower() in {"training_runs", "models", ".git", ".venv"} for p in output.parts), "Protected output directory")
        repo = (args.repo_root or ROOT).resolve()
        require(repo.is_dir(), "Repository root does not exist")
        install_guards(output)
        report = audit(repo, [path.resolve() for path in args.source_root], spec)
        assert_report_schema(report)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        print(report["future_holdout_readiness"]["status"])
        return 0
    except (OSError, ValueError, RuntimeError, UnicodeError, subprocess.CalledProcessError) as error:
        print("FOUNDATION_ERROR: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
