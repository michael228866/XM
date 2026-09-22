"""Offline GOLD capture certification; no collection, models or strategy execution."""
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
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from zoneinfo import TZPATH, ZoneInfo, ZoneInfoNotFoundError

ROOT = Path(__file__).resolve().parent
BANNED_MODULES = {"xgboost", "lightgbm", "sklearn", "torch", "stable_baselines3", "MetaTrader5", "urllib", "requests", "http"}
FORBIDDEN_FIELDS = {"win_rate", "profit_factor", "pnl", "mean_r", "drawdown", "trades_per_day", "trade_count", "wins", "losses", "prediction", "probability", "signal_score", "b0_score", "secondary_score", "entry_price", "exit_price", "realized_r", "strategy_return"}
SECRET_KEY = re.compile(r"password|passwd|secret|token|api.?key|login|account|credential", re.I)
SHA = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")
UNKNOWN = {"", "UNKNOWN", "UNRESOLVED", "TODO", "TBD", "N/A", "NULL", "NOT_SET"}
RELEASE_HASHES = {'execution_spec_gold_future_capture_certification_v1.json': 'e9f489af6294076b2986965672993b8941aab7d28ba63557d26ba3f93334a370',
 'gold_future_capture_manifest_schema_v1.json': '26d158735af2fffff3bbebad87cb93c463bb35e89405dec9f92e1f51c6409a38',
 'gold_future_capture_source_attestation_template_v1.json': '7d703729183ca1b98b22bbb7964362d0dacc1eec571c35039f75c53ed84c00bf',
 'gold_future_capture_timezone_attestation_template_v1.json': '225ffa9bf4b8c0e369f739d571488bd06367c5f242544fe13e0eeac4ddbec428',
 'gold_future_capture_activation_template_v1.json': 'c629d28a8e9a511a13fbb263538043bb7372c32a703a834318ce9e0b72718251'}


class InvalidCertification(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidCertification(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


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
    # pathlib preloads the inert urllib.parse module on CPython 3.11. Permit
    # that existing parser only; network clients and all future urllib imports
    # remain blocked by the finder and socket audit hook.
    require(not (BANNED_MODULES - {"urllib"}).intersection(sys.modules) and
            "urllib.request" not in sys.modules and "http.client" not in sys.modules,
            "Model or network client already imported")
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


def timestamp(value):
    require(isinstance(value, str), "Timestamp must be ISO text")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "Explicit timestamp offset required")
    return parsed


def utc(value):
    parsed = timestamp(value)
    require(parsed.utcoffset() == timedelta(0), "UTC timestamp required")
    return parsed


def resolved(value):
    return isinstance(value, str) and value.strip().upper() not in UNKNOWN and not re.fullmatch(r"<.*>", value.strip())


def safe_keys(value, report=False):
    if isinstance(value, dict):
        for key, child in value.items():
            require(isinstance(key, str) and key.lower() not in FORBIDDEN_FIELDS, "Forbidden document/report field")
            if key == "source_account_environment":
                require(child in {"demo", "live", "unknown"}, "Invalid environment enum")
            elif not (report and key == "credentials_handling"):
                require(not SECRET_KEY.search(key), "Secret-bearing keys are prohibited")
            safe_keys(child, report)
    elif isinstance(value, list):
        for child in value:
            safe_keys(child, report)


def load_json(path, limit=1048576):
    require(path.is_file() and path.stat().st_size <= limit, "Required JSON unavailable or too large")
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def constant(_):
        raise ValueError("Nonfinite JSON value")
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=pairs, parse_constant=constant)


def local_path(value):
    require(isinstance(value, (str, Path)), "Local path required")
    require(not ntpath.splitdrive(str(value))[0].startswith("\\\\"), "UNC/network paths prohibited")
    return Path(value).resolve()


def evidence_reader(base):
    def read(reference):
        path = local_path(base / reference)
        require(path.suffix.lower() in {".pdf", ".md", ".txt", ".json"}, "Evidence must be a local document")
        require(not re.search(r"model|ledger|prediction|trade_history", path.name, re.I), "Result/model evidence prohibited")
        return file_hash(path)
    return read


def evidence(items, reader, now):
    require(isinstance(items, list) and items, "Document evidence required")
    for item in items:
        require(isinstance(item, dict) and set(item) == {"authority", "local_path", "sha256", "reviewed_by", "reviewed_at_utc"}, "Evidence shape")
        require(item["authority"] in {"BROKER_DOCUMENT", "FEED_DOCUMENT", "SIGNED_ATTESTATION", "INDEPENDENT_REVIEW"}, "Authoritative/reviewed evidence required; observed gaps are insufficient")
        require(resolved(item["reviewed_by"]) and resolved(item["local_path"]), "Evidence review identity")
        require(isinstance(item["sha256"], str) and SHA.fullmatch(item["sha256"]), "Evidence hash")
        require(utc(item["reviewed_at_utc"]) <= now, "Evidence review is in the future")
        require(reader(item["local_path"]) == item["sha256"], "Evidence document hash mismatch")


def completed(document, template):
    require(isinstance(document, dict), "Attestation object required")
    safe_keys(document)
    require(set(document) == set(template), "Attestation fields must match the frozen template")
    version = "activation_version" if "activation_version" in template else "attestation_version"
    require(document[version] == template[version] and document["template"] is False, "Template is not a completed attestation")
    for key in ("outcome_inspection_prohibited", "strategy_execution_prohibited"):
        require(document[key] is True, "No-outcome policy required")
    for key in ("historical_2025_plus_is_untouched", "production_promotion"):
        require(document[key] is False, "Historical promotion claim prohibited")
    require(document["holdout_start"] is None and document["earliest_possible_holdout_start"] is None, "No holdout timestamp may be assigned")
    require(isinstance(document["notes"], str), "Notes must be text")


def source_binding(doc, template, reader, now):
    completed(doc, template)
    excluded = {"template", "price_digits", "point_size", "volume_fields_if_present", "evidence", "notes",
                "outcome_inspection_prohibited", "strategy_execution_prohibited", "historical_2025_plus_is_untouched",
                "holdout_start", "earliest_possible_holdout_start", "production_promotion"}
    require(all(resolved(value) for key, value in doc.items() if key not in excluded), "Unresolved source binding field")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", doc["source_id"]), "Invalid source identity")
    require(doc["symbol"] == doc["symbol_exact_case"] and re.fullmatch(r"(?:GOLD|XAUUSD)[A-Za-z0-9#._-]*", doc["symbol"]), "Exact GOLD symbol binding required")
    require(doc["source_data_type"] in {"BAR_M1", "TICK"}, "Source data type")
    require(doc["source_account_environment"] in {"demo", "live"}, "Account environment unresolved")
    require(doc["source_timestamp_semantics"] in {"BAR_OPEN", "BAR_CLOSE"}, "Explicit bar label semantics required")
    require(doc["spread_units"] == "POINTS", "Frozen SPREAD requires attested point units")
    require(type(doc["price_digits"]) is int and 0 <= doc["price_digits"] <= 12, "Price precision")
    require(type(doc["point_size"]) in {int, float} and math.isfinite(doc["point_size"]) and
            doc["point_size"] > 0 and math.isclose(doc["point_size"], 10 ** -doc["price_digits"], rel_tol=1e-12), "Point size/precision consistency")
    volumes = doc["volume_fields_if_present"]
    require(isinstance(volumes, list) and len(set(volumes)) == len(volumes) and
            all(v in {"TICK_VOLUME", "REAL_VOLUME", "VOLUME"} for v in volumes), "Volume schema")
    require(utc(doc["attested_at_utc"]) <= now, "Source attestation is in the future")
    evidence(doc["evidence"], reader, now)
    return {"status": "CERTIFIED_DOCUMENT_BINDING", "source_id": doc["source_id"], "broker": doc["broker_name"],
            "server": doc["broker_server_name"], "symbol": doc["symbol"], "source_data_type": doc["source_data_type"], "blockers": []}


@lru_cache(maxsize=16)
def attested_iana_zone(name, expected_sha):
    require(isinstance(expected_sha, str) and SHA.fullmatch(expected_sha), "IANA identity must be the actual TZif file SHA256")
    require(re.fullmatch(r"[A-Za-z0-9_+-]+(?:/[A-Za-z0-9_+-]+)*", name), "Invalid IANA zone name")
    for directory in TZPATH:
        path = Path(directory) / name
        if path.is_file():
            raw = path.read_bytes()
            require(hashlib.sha256(raw).hexdigest() == expected_sha, "Installed IANA TZif identity mismatch")
            return ZoneInfo.from_file(io.BytesIO(raw), key=name)
    # Fail closed on systems without a local TZif database; no download or
    # timezone-package installation is attempted by certification.
    raise InvalidCertification("Attested local IANA TZif database unavailable")


def certify_timezone(doc, template, source, reader, now):
    completed(doc, template)
    require(doc["source_attestation_sha256"] == canonical_hash(source) and doc["source_id"] == source["source_id"], "Timezone/source binding mismatch")
    require(doc["status"] in {"CERTIFIED_UTC", "CERTIFIED_FIXED_OFFSET", "CERTIFIED_IANA_WITH_DST", "CERTIFIED_BROKER_SERVER_RULE"}, "Timezone unresolved")
    for key in ("timezone", "timezone_authority", "dst_policy", "historical_bars_clock", "session_rollover", "weekend_policy", "holiday_calendar_source", "attested_by"):
        require(resolved(doc[key]), "Missing timezone authority/rule")
    require(doc["timestamp_semantics"] in {"BAR_OPEN", "BAR_CLOSE"} and doc["timestamp_semantics"] == source["source_timestamp_semantics"], "Bar semantics mismatch")
    for key, source_key in (("timezone", "source_timezone"), ("dst_policy", "dst_policy"), ("session_rollover", "session_rollover"),
                            ("weekend_policy", "weekend_policy"), ("holiday_calendar_source", "holiday_policy_source")):
        require(doc[key] == source[source_key], "Timezone/source declarations conflict")
    require(doc["source_timestamp_encoding"] in {"ISO8601", "EPOCH_SECONDS"}, "Source timestamp encoding")
    require(doc["mt5_epoch_semantics"] in {"ALREADY_UTC", "LOCAL_EPOCH_REQUIRES_RULE", "NOT_APPLICABLE"}, "MT5 epoch semantics unresolved")
    require(doc["source_timestamp_encoding"] != "EPOCH_SECONDS" or doc["mt5_epoch_semantics"] != "NOT_APPLICABLE", "Epoch interpretation required")
    if doc["status"] == "CERTIFIED_UTC":
        require(doc["timezone"] == "UTC" and doc["utc_offset_minutes"] == 0 and doc["dst_policy"] == "NONE", "UTC rule conflict")
    elif doc["status"] == "CERTIFIED_FIXED_OFFSET":
        require(type(doc["utc_offset_minutes"]) is int and -840 <= doc["utc_offset_minutes"] <= 840 and doc["dst_policy"] == "NONE", "Fixed offset rule")
    elif doc["status"] == "CERTIFIED_IANA_WITH_DST":
        require(resolved(doc["iana_tzdb_identity"]) and doc["dst_policy"] != "NONE", "IANA database and DST evidence required")
        attested_iana_zone(doc["timezone"], doc["iana_tzdb_identity"])
    else:
        require(resolved(doc["broker_transition_rule"]) and doc["broker_offset_intervals"], "Explicit broker transition rule/table required")
        previous = None
        for interval in doc["broker_offset_intervals"]:
            require(set(interval) == {"start_utc", "end_utc", "offset_minutes"}, "Broker interval schema")
            a, b = utc(interval["start_utc"]), utc(interval["end_utc"])
            require(a < b and (previous is None or previous == a), "Broker rule intervals must be contiguous and ordered")
            require(type(interval["offset_minutes"]) is int and -840 <= interval["offset_minutes"] <= 840, "Broker UTC offset")
            previous = b
    require(utc(doc["attested_at_utc"]) <= now, "Timezone attestation is in the future")
    evidence(doc["evidence"], reader, now)
    return {"status": doc["status"], "timezone": doc["timezone"], "dst_policy": doc["dst_policy"],
            "timestamp_semantics": doc["timestamp_semantics"], "session_rollover": doc["session_rollover"],
            "holiday_calendar_source": doc["holiday_calendar_source"], "blockers": []}


def source_utc(text, tz):
    if tz["source_timestamp_encoding"] == "EPOCH_SECONDS":
        number = float(text)
        require(math.isfinite(number), "Invalid epoch")
        value = datetime.fromtimestamp(number, timezone.utc)
        if tz["mt5_epoch_semantics"] == "ALREADY_UTC":
            return value
        value = value.replace(tzinfo=None)
    else:
        value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if tz["status"] in {"CERTIFIED_UTC", "CERTIFIED_FIXED_OFFSET"}:
        zone = timezone(timedelta(minutes=tz["utc_offset_minutes"]))
        if value.tzinfo is not None:
            require(value.utcoffset() == zone.utcoffset(None), "Raw timestamp offset conflicts with attestation")
        return value.replace(tzinfo=zone).astimezone(timezone.utc)
    if tz["status"] == "CERTIFIED_IANA_WITH_DST":
        zone = attested_iana_zone(tz["timezone"], tz["iana_tzdb_identity"])
        if value.tzinfo is not None:
            require(value.utcoffset() == value.astimezone(zone).utcoffset(), "Raw IANA offset mismatch")
            return value.astimezone(timezone.utc)
        candidates = {value.replace(tzinfo=zone, fold=fold).astimezone(timezone.utc) for fold in (0, 1)
                      if value.replace(tzinfo=zone, fold=fold).astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None) == value}
    else:
        candidates = set()
        for interval in tz["broker_offset_intervals"]:
            offset = timedelta(minutes=interval["offset_minutes"])
            candidate = value.astimezone(timezone.utc) if value.tzinfo else (value - offset).replace(tzinfo=timezone.utc)
            if utc(interval["start_utc"]) <= candidate < utc(interval["end_utc"]) and (value.tzinfo is None or value.utcoffset() == offset):
                candidates.add(candidate)
    require(len(candidates) == 1, "Ambiguous, nonexistent or uncovered source timestamp")
    return next(iter(candidates))


def pipeline_policies(activation, source, tz, spec, reader, now):
    policy = activation["higher_timeframe_policy"]
    native = spec["required_timeframes"]
    if policy == "M1_DETERMINISTIC_RESAMPLE_PROPOSED":
        require(activation["protocol_change_required"] is True, "Resampling must declare protocol change")
        proposal = activation["resampling_proposal"]
        require(isinstance(proposal, dict) and set(proposal) == {"boundaries", "closed_bar_rule", "calendar", "missing_bar_rule", "immutable_approval_required"} and
                all(resolved(proposal[k]) for k in ("boundaries", "closed_bar_rule", "calendar", "missing_bar_rule")) and
                proposal["immutable_approval_required"] is True, "Exact resampling proposal required")
        return {"status": "PROTOCOL_CHANGE_REQUIRED", "policy": policy, "native_required_timeframes": native,
                "protocol_change_required": True, "blockers": ["Later immutable protocol approval required; not historical-equivalent"]}
    require(policy == "NATIVE_20TF_REQUIRED" and activation["protocol_change_required"] is False, "Higher timeframe policy unresolved")
    require(source["source_data_type"] == "BAR_M1", "Native bar source required; ticks are not silently aggregated")
    entries = activation["native_timeframes"]
    require(isinstance(entries, list) and len(entries) == len(native) and {e["timeframe"] for e in entries} == set(native), "Every unique native timeframe required")
    for entry in entries:
        require(set(entry) == {"timeframe", "source_id", "symbol", "source_mode", "timestamp_semantics", "schema_sha256", "freshness_policy", "continuity_policy", "evidence"}, "Native timeframe record schema")
        require(entry["source_id"] == source["source_id"] and entry["symbol"] == source["symbol"] and
                entry["source_mode"] == "NATIVE" and entry["timestamp_semantics"] == tz["timestamp_semantics"], "Native identity/semantics mismatch")
        require(entry["schema_sha256"] == raw_schema_hash(source, spec), "Native schema binding mismatch")
        require(resolved(entry["freshness_policy"]) and resolved(entry["continuity_policy"]), "Native freshness/continuity evidence required")
        evidence(entry["evidence"], reader, now)
    return {"status": "CERTIFIED_DOCUMENT_POLICY", "policy": policy, "native_required_timeframes": native,
            "protocol_change_required": False, "blockers": []}


def certify_prefix(activation, reader, now):
    policy = activation["prefix_policy"]
    if policy == "REINITIALIZATION_PROTOCOL_CHANGE_REQUIRED":
        return {"status": "PROTOCOL_CHANGE_REQUIRED", "policy": policy, "blockers": ["Predeclared reinitialization needs later immutable approval"]}
    require(policy == "EXACT_CONTINUOUS_CERTIFIED_PREFIX", "Prefix policy unresolved")
    require(utc(activation["prefix_context_start"]) < utc(activation["prefix_context_end"]), "Continuous prefix interval required")
    if activation["protocol_freeze_effective_at_utc"]:
        require(utc(activation["prefix_context_end"]) <= utc(activation["protocol_freeze_effective_at_utc"]), "Context cannot extend past freeze")
    evidence(activation["prefix_evidence"], reader, now)
    return {"status": "CERTIFIED_DOCUMENT_POLICY", "policy": policy, "blockers": []}


def certify_activation(doc, template, source, tz, now):
    completed(doc, template)
    require(doc["source_attestation_sha256"] == canonical_hash(source) and doc["timezone_attestation_sha256"] == canonical_hash(tz), "Activation attestation hash mismatch")
    require(doc["status"] in {"NOT_ACTIVATED", "ACTIVATION_ARTIFACT_READY", "ACTIVE_UNVERIFIED"}, "Activation status unsupported/unverified")
    require(doc["historical_rows_context_only"] is True and doc["first_eligible_bar_rule"] == template["first_eligible_bar_rule"], "Activation boundary policy drift")
    require(resolved(doc["capture_root"]) and isinstance(doc["collector_commit"], str) and COMMIT.fullmatch(doc["collector_commit"]) and resolved(doc["collector_version"]), "Collector/destination binding required")
    requested = utc(doc["activation_requested_at_utc"])
    require(max(utc(source["attested_at_utc"]), utc(tz["attested_at_utc"])) <= requested <= now, "Activation request chronology")
    if doc["protocol_freeze_commit"] is not None:
        require(COMMIT.fullmatch(doc["protocol_freeze_commit"]) and utc(doc["protocol_freeze_effective_at_utc"]) <= requested, "Protocol freeze identity/time")
    else:
        require(doc["protocol_freeze_effective_at_utc"] is None, "Freeze time without commit")
    if doc["status"] == "ACTIVE_UNVERIFIED":
        require(doc["protocol_freeze_commit"] is not None and requested <= utc(doc["activation_effective_at_utc"]) <= now, "Effective activation requires freeze and chronology")
    else:
        require(doc["activation_effective_at_utc"] is None, "Pre-activation cannot have effective time")
    return doc["status"]


def inspect_collector(source_text, source_sha, commit, review, spec, reader, now):
    """AST risk screen plus independent review bound to these exact bytes.

    Presence of a named operation is not proof of correct control flow; the
    hash-bound review must cover control flow, dependencies and recovery.
    """
    tree = ast.parse(source_text)
    risks, imports, calls, functions = [], set(), [], set()
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                imports.add(item.name)
                aliases[item.asname or item.name] = item.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.add(module)
            for item in node.names:
                aliases[item.asname or item.name] = module + "." + item.name
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if isinstance(node.value, ast.Constant) and node.value.value not in (None, "", False):
                if any(isinstance(t, ast.Name) and SECRET_KEY.search(t.id) for t in targets):
                    risks.append("Credential-like literal assignment at line " + str(node.lineno))
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and isinstance(key.value, str) and SECRET_KEY.search(key.value) and isinstance(value, ast.Constant) and value.value not in (None, "", False):
                    risks.append("Credential-like dictionary literal at line " + str(node.lineno))
    def dotted(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return dotted(node.value) + "." + node.attr
        return "<dynamic>"
    exclusive = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = dotted(node.func)
        calls.append(name)
        leaf = name.rsplit(".", 1)[-1]
        if leaf in {"fit", "predict", "predict_proba", "exec", "eval", "__import__", "getattr", "setattr"}:
            risks.append("Dynamic/strategy call at line " + str(node.lineno))
        if re.search(r"(?:guess|infer|detect).*timezone|detect_dst", leaf, re.I):
            risks.append("Runtime timezone guess at line " + str(node.lineno))
        if leaf in {"write_text", "write_bytes", "replace", "rename", "unlink", "remove", "rmtree", "truncate", "ftruncate"}:
            risks.append("Mutable storage operation at line " + str(node.lineno))
        if leaf in {"open", "fdopen"}:
            position = 1 if name in {"open", "builtins.open", "io.open", "os.fdopen"} else 0
            mode = next((kw.value for kw in node.keywords if kw.arg == "mode"), node.args[position] if len(node.args) > position else ast.Constant("r"))
            if not isinstance(mode, ast.Constant) or not isinstance(mode.value, str):
                risks.append("Dynamic file mode cannot be certified")
            elif any(c in mode.value for c in "wa+"):
                risks.append("Mutable file mode at line " + str(node.lineno))
            elif "x" in mode.value:
                exclusive = True
        if name == "os.open":
            risks.append("Low-level open requires separate reviewed implementation; v1 fails closed")
    if any(module.split(".")[0] in {"xgboost", "lightgbm", "sklearn", "torch", "stable_baselines3"} or
           module.startswith(("gold_", "validate_gold", "barrier_", "drl_", "gemini")) for module in imports):
        risks.append("Strategy/model or unreviewed repository module import")
    required_functions = {"verify_previous_hash", "recover_chain", "quarantine_partial", "stable_bar_identity", "record_revision"}
    if not exclusive or "os.fsync" not in calls or "os.link" not in calls or not required_functions <= functions:
        risks.append("Exclusive creation, fsync, no-replace atomic publication or recovery interfaces absent")
    reviewed = False
    try:
        require(isinstance(review, dict) and set(review) == {"collector_sha256", "collector_git_commit", "reviewed_by", "reviewed_at_utc", "guarantees", "recovery_cases", "timezone_conversion_location", "evidence"}, "Collector review schema")
        require(review["collector_sha256"] == source_sha and review["collector_git_commit"] == commit and COMMIT.fullmatch(commit), "Review must bind collector bytes and commit")
        require(resolved(review["reviewed_by"]) and utc(review["reviewed_at_utc"]) <= now and resolved(review["timezone_conversion_location"]), "Collector review identity/time")
        require(set(review["guarantees"]) == set(spec["review_guarantees"]) and all(v is True for v in review["guarantees"].values()), "Every collector guarantee must be independently reviewed")
        require(set(review["recovery_cases"]) == set(spec["recovery_cases"]) and all(resolved(v) for v in review["recovery_cases"].values()), "Every recovery case needs a documented procedure")
        evidence(review["evidence"], reader, now)
        reviewed = True
    except (ValueError, TypeError, KeyError, OSError):
        risks.append("Missing or invalid hash-bound independent collector review")
    ok = reviewed and not risks
    categories = {"NOT_CERTIFIED_MUTABLE_STORAGE" if "Mutable" in risk else
                  "NOT_CERTIFIED_TIMEZONE" if "timezone guess" in risk else
                  "NOT_CERTIFIED_RECOVERY" if "recovery interfaces" in risk else
                  "NOT_CERTIFIED_SOURCE_BINDING" if "review" in risk or "Credential" in risk else
                  "NOT_CERTIFIED_SCHEMA" for risk in risks}
    certification_status = "CERTIFIED_FOR_ACTIVATION" if ok else next(iter(categories)) if len(categories) == 1 else "NOT_CERTIFIED_MULTIPLE_BLOCKERS"
    return {"static_status": "STATICALLY_CONFORMANT" if ok else "NOT_STATICALLY_CONFORMANT",
            "operational_status": "NOT_ACTIVATED", "certification_status": certification_status,
            "collector_sha256": source_sha, "collector_commit": commit,
            "writes_append_only": ok, "overwrites_existing_snapshot": any("Mutable" in r for r in risks),
            "uses_atomic_write": "os.link" in calls, "fsync_or_equivalent": "os.fsync" in calls,
            "manifest_written_after_snapshot": ok, "previous_hash_verified_before_append": ok,
            "restart_recovery_defined": ok, "partial_snapshot_quarantine": ok,
            "duplicate_bar_policy": "Reviewed stable source/symbol/timeframe/timestamp identity" if ok else "UNRESOLVED",
            "revision_policy": "Reviewed new revision snapshots; never rewrite old bytes" if ok else "UNRESOLVED",
            "timezone_conversion_location": review.get("timezone_conversion_location") if ok else None,
            "network_or_broker_dependency": sorted(imports & {"MetaTrader5", "requests", "urllib", "http.client", "socket"}),
            "native_timeframes_supported": spec["required_timeframes"] if ok else [], "spread_preserved": ok,
            "credentials_handling": "NO_SECRET_VALUES_REPORTED", "blockers": risks}


def raw_schema_hash(source, spec):
    return canonical_hash({"format": spec["raw_format"], "encoding": "UTF-8", "delimiter": ",",
                           "columns": spec["raw_columns"] + source["volume_fields_if_present"]})


def inspect_snapshot(binary, source, tz, spec):
    """Read predeclared raw-only CSV fields; never model/result artifacts."""
    stream = io.TextIOWrapper(binary, encoding="utf-8", newline="")
    reader = csv.DictReader(stream)
    columns = spec["raw_columns"] + source["volume_fields_if_present"]
    require(reader.fieldnames == columns, "Snapshot schema differs from canonical raw format")
    safe_keys({name.lower(): None for name in columns})
    counts = {name: 0 for name in ("row_count", "duplicate_timestamps", "non_monotonic_timestamps", "invalid_timestamp_rows",
                                  "invalid_price_rows", "spread_missing_rows", "spread_nonpositive_rows")}
    first = last = previous = None
    seen, timeframes = set(), set()
    for row in reader:
        require(None not in row and all(value is not None for value in row.values()), "Malformed raw CSV row")
        require(row["SOURCE_ID"] == source["source_id"] and row["SYMBOL"] == source["symbol"], "Snapshot source/symbol substitution")
        require(row["TIMEFRAME"] in spec["required_timeframes"], "Snapshot timeframe")
        timeframes.add(row["TIMEFRAME"])
        counts["row_count"] += 1
        try:
            current = utc(row["SOURCE_TIMESTAMP"])
            require(source_utc(row["ORIGINAL_SOURCE_TIMESTAMP"], tz) == current, "Raw/normalized timestamp mismatch")
            counts["duplicate_timestamps"] += int(current in seen)
            counts["non_monotonic_timestamps"] += int(previous is not None and current < previous)
            seen.add(current)
            previous = current
            first = current if first is None else min(first, current)
            last = current if last is None else max(last, current)
        except (ValueError, OverflowError):
            counts["invalid_timestamp_rows"] += 1
        try:
            prices = [float(row[k]) for k in ("OPEN", "HIGH", "LOW", "CLOSE")]
            valid = all(math.isfinite(v) and v > 0 for v in prices) and prices[2] <= min(prices) and prices[1] >= max(prices)
        except ValueError:
            valid = False
        counts["invalid_price_rows"] += int(not valid)
        try:
            spread = float(row["SPREAD"])
            if not math.isfinite(spread):
                counts["spread_missing_rows"] += 1
            else:
                counts["spread_nonpositive_rows"] += int(spread <= 0)
        except ValueError:
            counts["spread_missing_rows"] += 1
    require(counts["row_count"] > 0 and first is not None and len(timeframes) == 1, "Empty/invalid or mixed-timeframe snapshot")
    return {**counts, "first_source_timestamp": first.isoformat(), "last_source_timestamp": last.isoformat(),
            "schema_sha256": raw_schema_hash(source, spec), "timeframe": next(iter(timeframes))}


def validate_manifest(item, schema):
    require(isinstance(item, dict) and set(item) == set(schema["required"]), "Manifest has missing/extra fields")
    safe_keys(item)
    for key, rule in schema["properties"].items():
        value = item[key]
        if "const" in rule:
            require(type(value) is type(rule["const"]) and value == rule["const"], "Manifest constant violation")
        if "enum" in rule:
            require(value in rule["enum"], "Manifest enum violation")
        if "type" in rule:
            types = rule["type"] if isinstance(rule["type"], list) else [rule["type"]]
            require(any((kind == "null" and value is None) or (kind == "string" and isinstance(value, str)) or
                        (kind == "integer" and type(value) is int) for kind in types), "Manifest field type violation")
        if value is not None:
            if "pattern" in rule:
                require(re.fullmatch(rule["pattern"], value), "Manifest identifier/hash/path format")
            if "minimum" in rule:
                require(value >= rule["minimum"], "Manifest numeric range")
            if "minLength" in rule:
                require(len(value) >= rule["minLength"], "Manifest empty text")
            if rule.get("format") == "date-time":
                utc(value)
    require(item["snapshot_sequence"] == item["sequence"], "Snapshot/manifest sequence mismatch")
    require((item["previous_manifest_sha256"] is None) == (item["sequence"] == 0), "Genesis predecessor rule")
    require(utc(item["first_source_timestamp"]) <= utc(item["last_source_timestamp"]) <= utc(item["capture_finished_at_utc"]), "Source timestamp range")
    require(utc(item["capture_started_at_utc"]) <= utc(item["capture_finished_at_utc"]) <= utc(item["manifest_created_at_utc"]), "Capture/seal chronology")


def verify_chain(entries, tip, snapshot_reader, source, tz, activation, schema):
    require(entries, "No manifests to verify")
    hashes = {"source_attestation_sha256": canonical_hash(source), "timezone_attestation_sha256": canonical_hash(tz),
              "activation_sha256": canonical_hash(activation)}
    previous_hash = None
    previous_created = previous_finished = None
    identifiers, paths, last_by_tf = {}, set(), {}
    for number, item in enumerate(entries):
        validate_manifest(item, schema)
        require(item["sequence"] == number and item["previous_manifest_sha256"] == previous_hash, "Broken or missing manifest sequence/link")
        require(all(item[k] == value for k, value in hashes.items()), "Attestation hash substitution")
        require(item["source_id"] == source["source_id"] and item["symbol"] == source["symbol"], "Manifest source substitution")
        require(item["collector_commit"] == activation["collector_commit"] and item["collector_version"] == activation["collector_version"], "Collector change requires recertification")
        require(item["snapshot_id"] not in identifiers and item["snapshot_path"] not in paths, "Duplicate snapshot id/path")
        created = utc(item["manifest_created_at_utc"])
        started, finished = utc(item["capture_started_at_utc"]), utc(item["capture_finished_at_utc"])
        require(previous_created is None or created > previous_created, "Manifest clock reversal")
        require(previous_finished is None or started > previous_finished, "Capture clock reversal/overlap")
        summary = snapshot_reader(item)
        require(summary["raw_sha256"] == item["raw_sha256"], "Snapshot SHA mismatch")
        for key in ("schema_sha256", "row_count", "duplicate_timestamps", "non_monotonic_timestamps", "invalid_timestamp_rows", "invalid_price_rows", "spread_missing_rows", "spread_nonpositive_rows", "timeframe"):
            require(summary[key] == item[key], "Snapshot metadata/counters mismatch")
        for key in ("first_source_timestamp", "last_source_timestamp"):
            require(utc(summary[key]) == utc(item[key]), "Snapshot timestamp bounds mismatch")
        require(not any(item[k] for k in ("duplicate_timestamps", "non_monotonic_timestamps", "invalid_timestamp_rows", "invalid_price_rows", "spread_missing_rows")), "Raw quality/ordering defect")
        revision = item["revision_of_snapshot_id"]
        if revision is not None:
            require(revision in identifiers, "Revision must refer to an earlier sealed snapshot")
            prior = identifiers[revision]
            require(all(item[k] == prior[k] for k in ("timeframe", "first_source_timestamp", "last_source_timestamp", "row_count")), "Revision must preserve the referenced bar interval")
        else:
            require(item["timeframe"] not in last_by_tf or utc(item["first_source_timestamp"]) > last_by_tf[item["timeframe"]], "Duplicate/reversed bars across snapshots")
            last_by_tf[item["timeframe"]] = utc(item["last_source_timestamp"])
        identifiers[item["snapshot_id"]] = item
        paths.add(item["snapshot_path"])
        previous_hash, previous_created, previous_finished = canonical_hash(item), created, finished
    require(isinstance(tip, dict) and set(tip) == {"manifest_sequence", "manifest_sha256", "updated_at_utc"}, "Chain tip schema")
    require(type(tip["manifest_sequence"]) is int and tip["manifest_sequence"] == len(entries) - 1 and tip["manifest_sha256"] == previous_hash and utc(tip["updated_at_utc"]) >= previous_created, "Chain tip mismatch")
    return {"manifest_schema_valid": True, "chain_valid": True, "snapshot_hashes_valid": True,
            "attestation_hashes_valid": True, "verified_manifests": len(entries), "chain_tip_sha256": previous_hash}


def verify_capture_root(root, source, tz, activation, schema, spec):
    require(root.is_dir(), "Capture root missing; nothing is created automatically")
    require(local_path(activation["capture_root"]) == root, "Activation/capture root mismatch")
    for name, document in (("source_attestation.json", source), ("timezone_attestation.json", tz), ("activation.json", activation)):
        require(canonical_hash(load_json(root / name)) == canonical_hash(document), "Capture-root attestation bytes/identity mismatch")
    manifests, snapshots = root / "manifests", root / "snapshots"
    require(manifests.is_dir() and snapshots.is_dir() and not manifests.is_symlink() and not snapshots.is_symlink()
            and manifests.resolve().is_relative_to(root) and snapshots.resolve().is_relative_to(root), "Capture layout missing/redirected")
    files = sorted(manifests.iterdir())
    require(len(files) <= spec["max_manifests"], "Manifest scan limit reached")
    require(all(path.is_file() and not path.is_symlink() and re.fullmatch(r"[0-9]{12}\.json", path.name) for path in files), "Partial/unrecognized manifest requires quarantine")
    if not files:
        require(activation["status"] in {"NOT_ACTIVATED", "ACTIVATION_ARTIFACT_READY"} and
                not any(snapshots.iterdir()) and not (root / "chain_tip.json").exists(), "Empty/partial active chain or orphan data")
        return {"manifest_schema_valid": True, "chain_valid": None, "snapshot_hashes_valid": None,
                "attestation_hashes_valid": True, "verified_manifests": 0, "prepared_empty_layout": True}
    require(activation["status"] == "ACTIVE_UNVERIFIED", "Existing capture chain needs an explicit effective activation artifact")
    entries = []
    for number, path in enumerate(files):
        require(path.name == f"{number:012d}.json", "Manifest filename sequence gap")
        entries.append(load_json(path, spec["text_limit_bytes"]))
    require(set(path.name for path in snapshots.iterdir()) == {Path(entry["snapshot_path"]).name for entry in entries}, "Unlinked/partial snapshot requires quarantine")
    def read_snapshot(item):
        path = root / item["snapshot_path"]
        require(not path.is_symlink() and path.resolve().is_relative_to(snapshots.resolve()) and path.is_file(), "Missing or redirected snapshot")
        before = path.stat()
        raw_hash = file_hash(path)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as binary:
            result = inspect_snapshot(binary, source, tz, spec)
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "Snapshot changed while inspecting")
        return {**result, "raw_sha256": raw_hash}
    result = verify_chain(entries, load_json(root / "chain_tip.json"), read_snapshot, source, tz, activation, schema)
    require(all(utc(entry["capture_started_at_utc"]) >= utc(activation["activation_effective_at_utc"]) for entry in entries), "Capture predates activation")
    # Old/context source timestamps can remain in sealed evidence, but this
    # verifier never declares them eligible or computes a first holdout bar.
    return result


def validate_output(output, capture_root=None, exists=None):
    path = local_path(output)
    require(path.suffix.lower() == ".json", "Explicit JSON output required")
    require(not any(part.lower() in {"training_runs", "models", ".git", ".venv", "future_holdout_capture"} for part in path.parts), "Protected output directory")
    require(capture_root is None or not path.is_relative_to(local_path(capture_root)), "Report cannot be inside capture root")
    require(not (path.exists() if exists is None else exists), "Output already exists; overwrite prohibited")
    return path


def readiness(blockers):
    codes = {code for code, _ in blockers}
    return {"status": "READY_FOR_CAPTURE_ACTIVATION" if not codes else next(iter(codes)) if len(codes) == 1 else "NOT_READY_MULTIPLE_BLOCKERS",
            "blockers": [{"status": code, "reason": reason} for code, reason in blockers],
            "warnings": ["Document and static conformity is not operational certification; external chain anchoring is recommended"]}


def build_report(spec, now, commit, dirty, results, blockers):
    report = {"certification_version": spec["experiment_name"], "generated_at_utc": now.isoformat(),
        "repo_commit": commit, "repo_dirty": dirty,
        "historical_context": {"historical_2025_plus_is_untouched": False,
            "previous_foundation_status": spec["previous_foundation_status"], "historical_canonical_end": spec["historical_canonical_end"]},
        **results,
        "holdout_boundary": {"historical_start_allowed": False, "historical_2025_plus_allowed": False,
            "holdout_start": None, "earliest_possible_holdout_start": None, "boundary_rule": spec["boundary_rule"],
            "prerequisites": ["Immutable protocol freeze", "Effective certified activation", "Source/timezone/pipeline/prefix certification", "Valid sealed manifest chain", "No outcome inspection before one-shot unlock"]},
        "readiness": readiness(blockers), "production_promotion": False, "production_change": False}
    safe_keys(report, report=True)
    return report


def certify(args, documents):
    spec, schema, source_template, tz_template, activation_template = documents
    repo = local_path(args.repo_root or ROOT)
    now = datetime.now(timezone.utc)
    commit = git_read(repo, "rev-parse", "HEAD").decode("ascii").strip()
    dirty = bool(git_read(repo, "status", "--porcelain"))
    root = local_path(args.capture_root) if args.capture_root else None
    blockers, results = [], {}
    source = tz = activation = None
    def attempt(code, function):
        try:
            return function()
        except (ValueError, TypeError, KeyError, OSError, SyntaxError, ZoneInfoNotFoundError, subprocess.CalledProcessError) as error:
            # Never echo arbitrary JSON values, source literals or credentials.
            reason = str(error) if isinstance(error, InvalidCertification) else "Input could not be verified (" + type(error).__name__ + ")"
            blockers.append((code, reason))
            return None
    def document(path, basename):
        selected = local_path(path) if path else root / basename if root else None
        require(selected is not None, "Explicit attestation or capture root required")
        value = load_json(selected, spec["text_limit_bytes"])
        safe_keys(value)
        return value, evidence_reader(selected.parent)
    source_input = attempt("NOT_READY_SOURCE_BINDING", lambda: document(args.source_attestation, "source_attestation.json"))
    source_result = None
    if source_input:
        source, source_reader = source_input
        source_result = attempt("NOT_READY_SOURCE_BINDING", lambda: source_binding(source, source_template, source_reader, now))
    results["source_binding"] = source_result or {"status": "NOT_CERTIFIED", "source_id": None, "broker": None, "server": None, "symbol": None, "source_data_type": None, "blockers": ["Source attestation not certified"]}
    tz_input = attempt("NOT_READY_TIMEZONE", lambda: document(args.timezone_attestation, "timezone_attestation.json"))
    tz_result = None
    if tz_input and source_result:
        tz, tz_reader = tz_input
        tz_result = attempt("NOT_READY_TIMEZONE", lambda: certify_timezone(tz, tz_template, source, tz_reader, now))
    if not tz_result:
        blockers.append(("NOT_READY_TIMEZONE", "Timezone/source linkage not certified"))
    results["timezone_certification"] = tz_result or {"status": "UNRESOLVED", "timezone": None, "dst_policy": None, "timestamp_semantics": None, "session_rollover": None, "holiday_calendar_source": None, "blockers": ["Timezone evidence incomplete"]}
    activation_input = attempt("NOT_READY_CAPTURE_INTEGRITY", lambda: document(args.activation, "activation.json"))
    activation_status = None
    if activation_input and isinstance(activation_input[0].get("capture_root"), str):
        validate_output(args.output, activation_input[0]["capture_root"])
    if activation_input and source_result and tz_result:
        activation, activation_reader = activation_input
        activation_status = attempt("NOT_READY_CAPTURE_INTEGRITY", lambda: certify_activation(activation, activation_template, source, tz, now))
        # Check a capture root declared inside the supplied activation even when
        # --capture-root was omitted. No report may enter that directory.
        if isinstance(activation.get("capture_root"), str):
            validate_output(args.output, activation["capture_root"])
    for section, code, function in (
        ("higher_timeframe_policy", "NOT_READY_HIGHER_TIMEFRAMES", lambda: pipeline_policies(activation, source, tz, spec, activation_reader, now)),
        ("prefix_policy", "NOT_READY_PREFIX_POLICY", lambda: certify_prefix(activation, activation_reader, now))):
        result = attempt(code, function) if activation_status else None
        if result is None or result["blockers"]:
            blockers.append((code, "Policy unresolved or protocol change requires separate approval"))
        results[section] = result or {"status": "UNRESOLVED", "policy": "UNRESOLVED", "blockers": ["Completed activation policy required"]}
    results["higher_timeframe_policy"].setdefault("native_required_timeframes", spec["required_timeframes"])
    results["higher_timeframe_policy"].setdefault("protocol_change_required", False)
    collector = None
    def collector_check():
        require(args.collector is not None and activation_status is not None, "Collector source and validated activation review required")
        path = local_path(args.collector)
        require(path.is_file() and path.suffix == ".py" and path.stat().st_size <= spec["text_limit_bytes"], "Python collector missing/too large")
        raw = path.read_bytes()
        require(path.is_relative_to(repo), "Collector must belong to the attested Git checkout")
        sha = hashlib.sha256(raw).hexdigest()
        bound_commit = activation["collector_commit"]
        require(git_read(repo, "cat-file", "-t", bound_commit).strip() == b"commit", "Collector commit missing")
        blob = git_read(repo, "cat-file", "blob", bound_commit + ":" + path.relative_to(repo).as_posix())
        require(raw.replace(b"\r\n", b"\n") == blob.replace(b"\r\n", b"\n"), "Collector source not bound to specified Git commit")
        result = inspect_collector(raw.decode("utf-8-sig"), sha, bound_commit, activation["collector_review"], spec, activation_reader, now)
        result.update(collector_path=str(path), collector_git_commit=bound_commit, collector_version=activation["collector_version"])
        result["operational_status"] = activation_status
        return result
    collector = attempt("NOT_READY_COLLECTOR", collector_check)
    if not collector or collector["blockers"]:
        blockers.append(("NOT_READY_COLLECTOR", "Collector not statically conformant with a bound review"))
    results["collector_certification"] = collector or {"static_status": "NOT_STATICALLY_CONFORMANT", "operational_status": "NOT_ACTIVATED",
        "certification_status": "NOT_CERTIFIED_SOURCE_BINDING", "collector_path": None, "collector_sha256": None,
        "collector_commit": None, "collector_git_commit": None, "writes_append_only": None,
        "overwrites_existing_snapshot": None, "uses_atomic_write": None, "fsync_or_equivalent": None,
        "manifest_written_after_snapshot": None, "previous_hash_verified_before_append": None,
        "restart_recovery_defined": None, "partial_snapshot_quarantine": None, "duplicate_bar_policy": None,
        "revision_policy": None, "timezone_conversion_location": None, "network_or_broker_dependency": None,
        "native_timeframes_supported": [], "spread_preserved": None, "credentials_handling": "NO_SECRET_VALUES_REPORTED",
        "blockers": ["Collector unavailable or not certified"]}
    integrity = None
    if activation_status and root:
        integrity = attempt("NOT_READY_CAPTURE_INTEGRITY", lambda: verify_capture_root(root, source, tz, activation, schema, spec))
    if not integrity:
        blockers.append(("NOT_READY_CAPTURE_INTEGRITY", "Capture layout/attestation/chain not verified"))
    results["capture_integrity"] = {"activation_status": activation_status or "NOT_ACTIVATED", "capture_root": str(root) if root else None,
        "manifest_schema_valid": True, "chain_valid": None, "snapshot_hashes_valid": None, "attestation_hashes_valid": False,
        "restart_recovery_certified": False, "append_only_certified": False,
        "blockers": [] if integrity else ["Capture evidence missing or rejected"], **(integrity or {})}
    def freeze_check():
        require(activation_status and activation["protocol_freeze_commit"], "Immutable protocol freeze not bound")
        require(git_read(repo, "cat-file", "-t", activation["protocol_freeze_commit"]).strip() == b"commit", "Protocol freeze commit unavailable")
        return True
    attempt("NOT_READY_PROTOCOL_FREEZE", freeze_check)
    return build_report(spec, now, commit, dirty, results, blockers)


def self_test(documents):
    spec, schema, source_template, tz_template, activation_template = documents
    now = datetime(2035, 6, 1, 12, tzinfo=timezone.utc)
    def rejected(function):
        try:
            function()
        except (ValueError, TypeError, KeyError):
            return
        raise AssertionError("Invalid synthetic fixture accepted")
    reader = lambda _: "e" * 64
    proof = [{"authority": "BROKER_DOCUMENT", "local_path": "synthetic.txt", "sha256": "e" * 64,
              "reviewed_by": "synthetic reviewer", "reviewed_at_utc": (now - timedelta(days=4)).isoformat()}]
    source = deepcopy(source_template)
    for key, value in source.items():
        if value is None and key not in {"holdout_start", "earliest_possible_holdout_start"}:
            source[key] = "ATTESTED"
    source.update(template=False, source_id="synthetic", broker_name="synthetic broker", broker_server_name="synthetic server",
                  symbol="GOLD#", symbol_exact_case="GOLD#", source_account_environment="demo", source_data_type="BAR_M1",
                  source_timestamp_semantics="BAR_OPEN", source_timezone="UTC", dst_policy="NONE", spread_units="POINTS",
                  price_digits=2, point_size=0.01, evidence=proof, attested_at_utc=(now - timedelta(days=3)).isoformat())
    source_result = source_binding(source, source_template, reader, now)
    rejected(lambda: source_binding(source_template, source_template, reader, now))
    for key in ("broker_name", "broker_server_name", "symbol"):
        rejected(lambda key=key: source_binding({**source, key: None}, source_template, reader, now))
    for key in ("password", "api_key", "account_number", "login", "token", "credential"):
        rejected(lambda key=key: source_binding({**source, key: "synthetic-secret-fixture"}, source_template, reader, now))
    rejected(lambda: source_binding({**source, "historical_2025_plus_is_untouched": True}, source_template, reader, now))
    tz = deepcopy(tz_template)
    tz.update(template=False, status="CERTIFIED_UTC", source_id=source["source_id"], source_attestation_sha256=canonical_hash(source),
              timezone="UTC", timezone_authority="synthetic authority", dst_policy="NONE", timestamp_semantics="BAR_OPEN",
              source_timestamp_encoding="ISO8601", mt5_epoch_semantics="NOT_APPLICABLE", historical_bars_clock="UTC",
              session_rollover=source["session_rollover"], weekend_policy=source["weekend_policy"],
              holiday_calendar_source=source["holiday_policy_source"], utc_offset_minutes=0,
              attested_by="synthetic reviewer", attested_at_utc=(now - timedelta(days=2)).isoformat(), evidence=proof)
    tz_result = certify_timezone(tz, tz_template, source, reader, now)
    rejected(lambda: certify_timezone({**tz, "status": "UNRESOLVED"}, tz_template, source, reader, now))
    rejected(lambda: certify_timezone({**tz, "dst_policy": None}, tz_template, source, reader, now))
    rejected(lambda: certify_timezone({**tz, "evidence": [{**proof[0], "authority": "OBSERVED_GAPS"}]}, tz_template, source, reader, now))
    closed_source = {**source, "source_timestamp_semantics": "BAR_CLOSE"}
    closed_tz = {**tz, "timestamp_semantics": "BAR_CLOSE", "source_attestation_sha256": canonical_hash(closed_source)}
    require(certify_timezone(closed_tz, tz_template, closed_source, reader, now)["timestamp_semantics"] == "BAR_CLOSE", "Explicit close semantics")
    rejected(lambda: certify_timezone(closed_tz, tz_template, source, reader, now))
    activation = deepcopy(activation_template)
    require(activation["status"] == "NOT_ACTIVATED" and activation["activation_effective_at_utc"] is None, "Inert activation template")
    activation.update(template=False, source_attestation_sha256=canonical_hash(source), timezone_attestation_sha256=canonical_hash(tz),
        capture_root="D:/synthetic_capture", collector_commit="c" * 40, collector_version="synthetic-v1",
        protocol_freeze_commit="d" * 40, protocol_freeze_effective_at_utc=(now - timedelta(hours=36)).isoformat(),
        activation_requested_at_utc=(now - timedelta(days=1)).isoformat(), higher_timeframe_policy="NATIVE_20TF_REQUIRED",
        prefix_policy="EXACT_CONTINUOUS_CERTIFIED_PREFIX", prefix_context_start=(now - timedelta(days=100)).isoformat(),
        prefix_context_end=(now - timedelta(days=5)).isoformat(), prefix_evidence=proof)
    activation["native_timeframes"] = [{"timeframe": tf, "source_id": source["source_id"], "symbol": source["symbol"],
        "source_mode": "NATIVE", "timestamp_semantics": "BAR_OPEN", "schema_sha256": raw_schema_hash(source, spec),
        "freshness_policy": "reviewed closed-bar schedule", "continuity_policy": "reviewed gap handling", "evidence": proof}
        for tf in spec["required_timeframes"]]
    require(certify_activation(activation, activation_template, source, tz, now) == "NOT_ACTIVATED", "Completed pre-activation artifact")
    rejected(lambda: certify_activation({**activation, "holdout_start": now.isoformat()}, activation_template, source, tz, now))
    higher = pipeline_policies(activation, source, tz, spec, reader, now)
    rejected(lambda: pipeline_policies({**activation, "native_timeframes": activation["native_timeframes"][:-1]}, source, tz, spec, reader, now))
    rejected(lambda: pipeline_policies({**activation, "higher_timeframe_policy": "UNRESOLVED"}, source, tz, spec, reader, now))
    proposal = {**activation, "higher_timeframe_policy": "M1_DETERMINISTIC_RESAMPLE_PROPOSED", "protocol_change_required": True,
                "resampling_proposal": {"boundaries": "UTC synthetic", "closed_bar_rule": "only closed bars", "calendar": "synthetic",
                                        "missing_bar_rule": "fail closed", "immutable_approval_required": True}}
    require(pipeline_policies(proposal, source, tz, spec, reader, now)["protocol_change_required"] is True, "Resampling is protocol change")
    rejected(lambda: pipeline_policies({**proposal, "protocol_change_required": False}, source, tz, spec, reader, now))
    prefix = certify_prefix(activation, reader, now)
    rejected(lambda: certify_prefix({**activation, "prefix_policy": "UNRESOLVED"}, reader, now))

    # Collector text is an inert AST fixture: none of these operations runs.
    collector_text = '''import os
def verify_previous_hash(expected, actual):
    if expected != actual:
        raise ValueError("chain mismatch")
    return actual
def recover_chain(pairs):
    return [verify_previous_hash(a, b) for a, b in pairs]
def quarantine_partial(path):
    return {"quarantined": str(path)}
def stable_bar_identity(source, symbol, timeframe, stamp):
    return (source, symbol, timeframe, stamp)
def record_revision(prior, current):
    if prior == current:
        raise ValueError("revision needs new identity")
    return {"prior": prior, "current": current}
def publish(pending, sealed, data):
    with open(pending, "xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(pending, sealed)
'''
    collector_sha = hashlib.sha256(collector_text.encode()).hexdigest()
    review = {"collector_sha256": collector_sha, "collector_git_commit": "c" * 40,
              "reviewed_by": "synthetic independent reviewer", "reviewed_at_utc": (now - timedelta(hours=20)).isoformat(),
              "guarantees": dict.fromkeys(spec["review_guarantees"], True),
              "recovery_cases": dict.fromkeys(spec["recovery_cases"], "synthetic reviewed procedure"),
              "timezone_conversion_location": "synthetic reviewed conversion", "evidence": proof}
    collector = inspect_collector(collector_text, collector_sha, "c" * 40, review, spec, reader, now)
    require(collector["static_status"] == "STATICALLY_CONFORMANT" and collector["operational_status"] == "NOT_ACTIVATED", "Static is not operational")
    for unsafe in ('open("fixed", "w")', 'path.write_text("fixed")', 'os.remove("fixed")', 'handle.truncate()',
                   'import xgboost', 'password = "synthetic-literal-fixture"'):
        text = collector_text + "\n" + unsafe + "\n"
        sha = hashlib.sha256(text.encode()).hexdigest()
        require(inspect_collector(text, sha, "c" * 40, {**review, "collector_sha256": sha}, spec, reader, now)["static_status"] != "STATICALLY_CONFORMANT", "Unsafe AST accepted")
    activation["collector_review"] = review
    active = {**activation, "status": "ACTIVE_UNVERIFIED", "activation_effective_at_utc": (now - timedelta(hours=12)).isoformat()}
    snapshots, entries = {}, []
    def new_manifest(number, bar_number=None, revision=None):
        bar = now - timedelta(minutes=10) + timedelta(minutes=number if bar_number is None else bar_number)
        raw = (",".join(spec["raw_columns"]) + "\n" + ",".join([bar.isoformat(), bar.isoformat(), source["source_id"], source["symbol"], "M1", "2000", "2001", "1999", "2000", "30"]) + "\n").encode()
        summary = inspect_snapshot(io.BytesIO(raw), source, tz, spec)
        summary["raw_sha256"] = hashlib.sha256(raw).hexdigest()
        started = now - timedelta(minutes=3) + timedelta(seconds=number * 4)
        item = {"manifest_version": "gold_future_capture_manifest_v1", "sequence": number, "snapshot_sequence": number,
            "previous_manifest_sha256": canonical_hash(entries[-1]) if entries else None,
            "manifest_created_at_utc": (started + timedelta(seconds=2)).isoformat(), "snapshot_id": "snapshot" + str(number),
            "snapshot_path": f"snapshots/{number:012d}.csv", "source_attestation_sha256": canonical_hash(source),
            "timezone_attestation_sha256": canonical_hash(tz), "activation_sha256": canonical_hash(active),
            "source_id": source["source_id"], "symbol": source["symbol"],
            "capture_started_at_utc": started.isoformat(), "capture_finished_at_utc": (started + timedelta(seconds=1)).isoformat(),
            "collector_commit": active["collector_commit"], "collector_version": active["collector_version"],
            "revision_of_snapshot_id": revision, "partial_snapshot": False, "sealed": True, "notes": "", **summary}
        snapshots[item["snapshot_id"]] = summary
        return item
    entries.append(new_manifest(0))
    validate_manifest(entries[0], schema)
    entries.append(new_manifest(1))
    def tip(items):
        return {"manifest_sequence": len(items) - 1, "manifest_sha256": canonical_hash(items[-1]), "updated_at_utc": now.isoformat()}
    read_snapshot = lambda item: snapshots[item["snapshot_id"]]
    def chain(items, chain_tip=None, read=None):
        return verify_chain(items, chain_tip or tip(items), read or read_snapshot, source, tz, active, schema)
    require(chain(entries)["chain_valid"], "Two-manifest chain")
    for change in ({"previous_manifest_sha256": "f" * 64}, {"sequence": 3}, {"sequence": 0},
                   {"raw_sha256": "INVALID"}, {"sealed": False}, {"partial_snapshot": True},
                   {"snapshot_id": "snapshot0"}, {"manifest_created_at_utc": entries[0]["manifest_created_at_utc"]},
                   {"first_source_timestamp": now.isoformat()}, {"revision_of_snapshot_id": "absent"},
                   {"source_attestation_sha256": "f" * 64}, {"row_count": True}):
        altered = [entries[0], {**entries[1], **change}]
        rejected(lambda altered=altered: chain(altered))
    rejected(lambda: chain(entries[1:]))
    rejected(lambda: chain(entries, {**tip(entries), "manifest_sha256": "f" * 64}))
    rejected(lambda: chain(entries, read=lambda item: {**read_snapshot(item), "raw_sha256": "f" * 64}))
    entries.append(new_manifest(2, bar_number=0, revision="snapshot0"))
    require(chain(entries)["chain_valid"], "Retained revision reference")
    unicode_path = Path("D:/合成/數據/report.json")
    require("數據" in str(validate_output(unicode_path, exists=False)), "Unicode output path")
    rejected(lambda: validate_output(unicode_path, exists=True))
    rejected(lambda: validate_output("D:/capture/report.json", "D:/capture", exists=False))
    rejected(lambda: validate_output("D:/future_holdout_capture/report.json", exists=False))
    for code in ("NOT_READY_SOURCE_BINDING", "NOT_READY_TIMEZONE", "NOT_READY_HIGHER_TIMEFRAMES", "NOT_READY_PREFIX_POLICY",
                 "NOT_READY_COLLECTOR", "NOT_READY_CAPTURE_INTEGRITY", "NOT_READY_PROTOCOL_FREEZE"):
        require(readiness([(code, "synthetic")])["status"] == code, "Single blocker precedence")
    require(readiness([("NOT_READY_TIMEZONE", "clock"), ("NOT_READY_COLLECTOR", "code")])["status"] == "NOT_READY_MULTIPLE_BLOCKERS", "Multiple blockers")
    results = {"source_binding": source_result, "timezone_certification": tz_result, "higher_timeframe_policy": higher,
               "prefix_policy": prefix, "collector_certification": collector, "capture_integrity": chain(entries)}
    report = build_report(spec, now, "c" * 40, False, results, [])
    require(report["readiness"]["status"] == "READY_FOR_CAPTURE_ACTIVATION" and report["holdout_boundary"]["holdout_start"] is None
            and report["historical_context"]["historical_2025_plus_is_untouched"] is False, "Ready does not start a holdout")
    # Explicit forbidden-field fixtures; these never enter actual reports.
    for key in FORBIDDEN_FIELDS:
        rejected(lambda key=key: safe_keys({key: None}, report=True))
    safe_keys(report, report=True)
    for module in BANNED_MODULES | {"gold_secondary_s4_confirmation_v1", "gemini"}:
        try:
            ImportGuard().find_spec(module)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Forbidden import allowed")
    try:
        sys.audit("socket.connect", None, ("synthetic", 1))
    except RuntimeError:
        pass
    else:
        raise AssertionError("Network guard")
    for subcommand in spec["allowed_git_subcommands"]:
        for executable in ("git", "git.exe", r"C:\Program Files\Git\cmd\git.exe"):
            argv = [executable, subcommand, "synthetic"]
            for representation in (argv, tuple(argv), subprocess.list2cmdline(argv)):
                require(is_allowed_git_subprocess(executable, representation), "Windows Git representation")
                sys.audit("subprocess.Popen", None, representation, "synthetic", {})
    denied = [["git", cmd] for cmd in ("add", "commit", "push", "pull", "fetch", "checkout", "restore", "reset", "clean", "config", "gc")]
    denied += [[binary, "synthetic"] for binary in ("python", "powershell", "cmd", "terminal64.exe")]
    denied += [["git", "status", separator, "echo"] for separator in ("&", "&&", "|", ";")]
    for argv in denied:
        for representation in (argv, subprocess.list2cmdline(argv)):
            require(not is_allowed_git_subprocess(None, representation), "Forbidden subprocess")
            rejected(lambda representation=representation: sys.audit("subprocess.Popen", None, representation, "synthetic", {}))
    own_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    require(not any(isinstance(node, (ast.Import, ast.ImportFrom)) and any(
        item.name.split(".")[0] in {"xgboost", "MetaTrader5", "requests"} for item in node.names)
        for node in ast.walk(own_tree)), "No executable model/broker imports")
    require(not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
                    node.func.attr in {"fit", "predict", "predict_proba", "initialize", "login", "copy_rates_from_pos", "copy_ticks_range"}
                    for node in ast.walk(own_tree)), "No active model/broker execution paths")
    print("SELF_TEST_PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--output", type=Path)
    for name in ("repo-root", "collector", "source-attestation", "timezone-attestation", "activation", "capture-root"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    try:
        documents = []
        for name, expected in RELEASE_HASHES.items():
            document = load_json(ROOT / name)
            require(canonical_hash(document) == expected, "Certification release file identity mismatch")
            documents.append(document)
        if args.self_test:
            require(not any(getattr(args, name) for name in ("repo_root", "collector", "source_attestation", "timezone_attestation", "activation", "capture_root")), "Self-test cannot inspect supplied artifacts")
            install_guards(None)
            self_test(documents)
            return 0
        output = validate_output(args.output, args.capture_root)
        require(output.parent.is_dir(), "Output parent must already exist")
        install_guards(output)
        report = certify(args, documents)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        print(report["readiness"]["status"])
        return 0
    except (ValueError, OSError, RuntimeError, TypeError, KeyError, subprocess.CalledProcessError) as error:
        reason = str(error) if isinstance(error, InvalidCertification) else type(error).__name__
        print("CERTIFICATION_ERROR: " + reason, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
