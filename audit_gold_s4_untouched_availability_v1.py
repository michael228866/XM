"""Read-only metadata/provenance audit. No fitting, prediction, labels or replay."""
from __future__ import annotations

import argparse
import bisect
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
import zipfile
from array import array
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parent
AUDIT_VERSION = "gold_s4_untouched_availability_audit_v1"
CANDIDATE_START = datetime(2025, 1, 1)
EPOCH = datetime(1970, 1, 1)
TIMEFRAMES = ('M1',
 'Daily',
 'H12',
 'H8',
 'H6',
 'H4',
 'H3',
 'H2',
 'H1',
 'M30',
 'M20',
 'M15',
 'M12',
 'M10',
 'M6',
 'M5',
 'M4',
 'M3',
 'M2',
 'Weekly',
 'Monthly')
RAW_COLUMNS = ("OPEN", "HIGH", "LOW", "CLOSE")
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules"}
TEXT_EXTENSIONS = {".py", ".md", ".json", ".txt", ".ps1", ".yaml", ".yml"}
MAX_TEXT_BYTES = 4 * 1024 * 1024
MAX_EVIDENCE_EXAMPLES = 200
DATE_PATTERN = re.compile(r"20(?:2[5-9]|[3-9][0-9])[-/.][01][0-9][-/.][0-3][0-9](?:[T ][0-2][0-9]:[0-5][0-9](?::[0-5][0-9])?(?:Z|[+-][0-2][0-9]:[0-5][0-9])?)?")
BANNED_MODULES = {"xgboost", "lightgbm", "sklearn", "torch", "MetaTrader5", "gymnasium", "stable_baselines3"}
FEATURES = ['M1_RSI',
 'ATR',
 'MACD_HIST',
 'BB_WIDTH',
 'BIAS_20',
 'BODY_PCT',
 'ROC_5',
 'VOLA_RATIO',
 'HOUR_SIN',
 'HOUR_COS',
 'DAY_OF_WEEK',
 'Daily_TREND',
 'H12_TREND',
 'H1_TREND',
 'H2_TREND',
 'H3_TREND',
 'H4_TREND',
 'H6_TREND',
 'H8_TREND',
 'M10_TREND',
 'M12_TREND',
 'M15_TREND',
 'M20_TREND',
 'M2_TREND',
 'M30_TREND',
 'M3_TREND',
 'M4_TREND',
 'M5_TREND',
 'M6_TREND',
 'Monthly_TREND',
 'Weekly_TREND']
FROZEN_SOURCE_HASHES = {'training_runs/20260903T071729Z_gemini_execution_aligned_label_v1/training_script.py': '9e436e4fe723eb8ad354f5f27a342b60d051c5c71d533f523871b381fc7012c6',
 'barrier_research_suite.py': 'e9b0f84e0aedff532b0d24c5845191d724a7dc44ade15d928fe0a24b7c1341af',
 'drl_trading_v2.py': 'b129d90d4a4a02cbf6d77903351ed7c3273af3cb3d4464b6a146aa25153ef767',
 'drl_train_candidate.py': '8452fe54921dad32d9af0f77b203b78d67bed765eb4e832f39f2c8ed171fd762',
 'barrier_final_train.py': 'a221bda6c61fa89bde0592453de7a8ac84efb80f06b74e7686b872484e2edee5',
 'barrier_classifier_strategy.py': 'a743d15b90321e4d933e4657abd185922bb280b36ec48612b6f1688afb909174',
 'strategy_grid_search.py': 'ad0c49c9a605718b90df49d0afb565b66af7bd3f3e9e68153dbd386be6663403',
 'gold_gemini_execution_semantics_v1.py': 'fc540b959a9fdec6fafed9acce58d70b0e57261b53d6f57c81b2cb7c3e976757',
 'gold_recent_walk_forward.py': 'cf14bbd97e88632ae53cadaabf5e3341f844f5096e8dd0493f8f96a4bf810334',
 'gemini.py': '0ccb4a66c54981e3b207e0f20db1ca64a3f8d76ebe8a74784d1b9b6102fc4b07',
 'gold_gemini_incumbent_robustness_v1.py': 'adb83d97bc4bf7a44140da30f14b469425de3266b93544c6dd43e92f4ed0ae5f',
 'gold_generation11_execution_aligned.py': 'adabda7a5ed4fdd4a7c12a5f95e83835df191da102639295510cf93ebf0344fe',
 'gold_short_rule_research.py': 'fdf9262784a60a3a284b06a4fdc00b90ffe0a01d926fc552c562d7936015f7bb',
 'gold_regime_experts_walk_forward.py': '905c5a2397e8cabc2b1d538c91ed60d97f73cb4b55578af95b996d065fc51694',
 'gold_long_model_optimization.py': '3c91e18776f38822dd0b5ebc8e031718019aeb318888e2dcee09c83e0d89ef75',
 'gold_long_recent_walk_forward.py': '575b56a8a6d526a0102904dc873bac3a70d7569ac09c656720bfd76def768280',
 'gold_regime_experts_iterative.py': '914a1404a99a7a52def92d0fe3bca35ba9da6a1864d156368aec2b9a5b124a21',
 'gold_rolling_champion.py': '1bfe529c44ca59019cc864a998ec128931c121d577c43bc3befa7e743ba560da',
 'gold_generation8_residual_walk_forward.py': '470b0085b820d6ef53f80cd149fdc3bc5afcf2a3996207c68af5363df3a42f83',
 'gold_expected_r_walk_forward.py': '7cd87b364644ab88441055662e18bc1dbe19ef4fb5ee9ebfdef121aa2fe37e2e',
 'gold_expected_r_champion.py': 'ae91fed8968b0d29b963d51ccf9b5f04eff63e4847f4bfbd50989b4b10c74ff9',
 'gold_gemini_core_gate_v1.py': 'ae31c09af94466bf5deeafae4800b2e3a580dfeb31564b1123be69887f57932c'}
ARCHIVES = {'discovery': ('20260915T151723Z_gold_independent_secondary_classifier_v1',
               '522b9a8f477e23a79a9200905667b386747bb4fc9e7e40c66a2bdadd4ba19f4f'),
 'confirmation': ('20260919T124853Z_gold_secondary_s4_confirmation_v1',
                  'e075573581250b97bc9108d2ae0679b547d88fcd1663c3ca19f3b88265b091ac')}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def parse_time(text):
    text = str(text).strip()
    if len(text) >= 10:
        text = text[:10].replace(".", "-").replace("/", "-") + text[10:]
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def ns(value):
    require(value.tzinfo is None, "Canonical broker/API timestamps must be naive; timezone conversion requires provenance")
    delta = value - EPOCH
    return (delta.days * 86400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1000


def stamp(value):
    return (EPOCH + timedelta(microseconds=int(value) // 1000)).isoformat() if value is not None else None


def time_text(value):
    return value.isoformat() if value is not None else None


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


def inventory_csv(path, boundary=CANDIDATE_START, interval_seconds=60):
    """Stream raw fields and timestamps only; no indicator or outcome computation."""
    synthetic = isinstance(path, io.StringIO)
    if not synthetic:
        path = Path(path)
    result = {"path": "synthetic" if synthetic else str(path.resolve()),
              "size_bytes": None if synthetic else path.stat().st_size,
              "sha256": None if synthetic else file_hash(path),
              "columns": [], "missing_required_columns": [], "rows": 0, "row_count_after_2025_01_01": 0,
              "duplicate_timestamps": 0, "non_monotonic_timestamps": 0, "invalid_timestamps": 0,
              "nonfinite_required_inputs": 0, "invalid_price_rows": 0, "spread_fallback_rows": 0,
              "data_gaps": [], "gap_count": 0, "observed_trading_days": 0, "errors": []}
    times, valid = array("q"), array("b")
    if synthetic:
        context = nullcontext(path)
    else:
        opener = gzip.open if path.suffix.lower() == ".gz" else open
        with opener(path, "rb") as raw:
            prefix = raw.read(4)
        encoding = "utf-16" if prefix.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        context = opener(path, "rt", encoding=encoding, newline="")
    with context as stream:
        sample = stream.read(8192)
        stream.seek(0)
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
        except csv.Error:
            delimiter = "\t" if "\t" in sample.partition("\n")[0] else ","
        reader = csv.DictReader(stream, delimiter=delimiter)
        names = {name: name.replace("<", "").replace(">", "").upper() for name in (reader.fieldnames or [])}
        result["columns"] = list(names.values())
        result["missing_required_columns"] = sorted(set(RAW_COLUMNS) - set(names.values()))
        require(len(set(names.values())) == len(names), "Duplicate normalized CSV columns")
        if "DATE" not in names.values():
            result["missing_required_columns"].append("DATE")
        result["delimiter"] = repr(delimiter)
        previous, seen, days = None, set(), set()
        for raw_row in reader:
            result["rows"] += 1
            row = {names[k]: v for k, v in raw_row.items() if k in names}
            try:
                value = parse_time((row.get("DATE") or "") + (" " + row["TIME"] if row.get("TIME") else ""))
                key = ns(value)
            except (ValueError, TypeError, OverflowError):
                result["invalid_timestamps"] += 1
                continue
            post = value >= boundary
            prices = []
            good = True
            for column in RAW_COLUMNS:
                try:
                    number = float(row.get(column, ""))
                    require(math.isfinite(number) and 0 < number < 1e30, "Nonfinite/invalid price")
                    prices.append(number)
                except (ValueError, TypeError):
                    good = False
            if post and not good:
                result["nonfinite_required_inputs"] += 1
            if good and (prices[1] < max(prices[0], prices[2], prices[3]) or prices[2] > min(prices[0], prices[1], prices[3])):
                good = False
                if post:
                    result["invalid_price_rows"] += 1
            if post:
                result["row_count_after_2025_01_01"] += 1
                days.add(value.date().isoformat())
                result["duplicate_timestamps"] += int(key in seen)
                result["non_monotonic_timestamps"] += int(previous is not None and key < previous)
                try:
                    spread = float(row.get("SPREAD", "nan"))
                except (TypeError, ValueError):
                    spread = math.nan
                result["spread_fallback_rows"] += int(not math.isfinite(spread) or spread <= 0)
                if previous is not None and key - previous > interval_seconds * 1_000_000_000:
                    result["gap_count"] += 1
                    if len(result["data_gaps"]) < MAX_EVIDENCE_EXAMPLES:
                        result["data_gaps"].append({"after": stamp(previous), "before": stamp(key),
                            "elapsed_seconds": (key-previous)/1e9, "classification": "UNCLASSIFIED_INTERVAL_NOT_PROVEN_MISSING_BAR"})
            seen.add(key)
            previous = key
            times.append(key)
            valid.append(int(good))
    post_times = [t for t in times if t >= ns(boundary)]
    result.update(actual_first_available_timestamp=stamp(min(post_times)) if post_times else None,
                  actual_last_available_timestamp=stamp(max(post_times)) if post_times else None,
                  observed_trading_days=len(days), timestamp_semantics="naive source broker/API clock; not asserted UTC",
                  gap_examples_truncated=result["gap_count"] > len(result["data_gaps"]))
    return result, times, valid


def duration_classification(days):
    return ("INSUFFICIENT_DURATION" if days < 90 else "LIMITED_DURATION" if days < 180 else
            "MODERATE_DURATION" if days < 365 else "SUBSTANTIAL_DURATION")


def safe_start(raw, features, execution, contamination_cutoff, unbounded=False):
    if features is None or execution is None or unbounded:
        return None
    return max(v for v in (raw, features, execution, contamination_cutoff) if v is not None)


def verdict(missing, contaminated, provenance_complete, days):
    if contaminated:
        return "NOT_READY_CONTAMINATED"
    if missing:
        return "NOT_READY_MISSING_DATA"
    if not provenance_complete:
        return "NOT_READY_INCOMPLETE_PROVENANCE"
    if days < 90:
        return "NOT_READY_INSUFFICIENT_DURATION"
    return "READY_FOR_PROTOCOL_DESIGN"


def classify_text(text, path):
    """Conservative leads, not an inference that source code was executed."""
    found = []
    lines = text.splitlines()
    symbol_context = bool(re.search(r"gold|xauusd|gemini", path + " " + text[:10000], re.I))
    for i, line in enumerate(lines):
        context = " ".join(lines[max(0, i-2):i+3])
        policy = "2014" in line and "2026" in line and "inspected" in line.lower()
        forward = "contaminated_for_future_gate_selection" in line
        dates = DATE_PATTERN.findall(context)
        if not symbol_context or not (dates or policy or forward):
            continue
        if (policy or forward) and ("agents.md" in path.lower() or not path.lower().endswith(".py")):
            classification, reason = "CONFIRMED_CONTAMINATION", "Explicit repository policy declares prior inspection; no bounded clean successor established"
        elif re.search(r"fit|train|optimi[sz]|threshold|evaluat|backtest|performance|selection|confirmation", context, re.I):
            classification, reason = "POTENTIAL_CONTAMINATION", "Dated research reference; source existence alone does not prove execution or a clean cutoff"
        elif re.search(r"ingest|schema|warmup|raw.*download|availability", context, re.I):
            classification, reason = "SAFE_CONTEXT", "Ingestion/schema/warmup reference only; not proof that other use was absent"
        else:
            classification, reason = "POTENTIAL_CONTAMINATION", "Unresolved dated GOLD reference"
        found.append({"path": path, "line": i+1, "classification": classification, "dates": sorted(set(dates)),
                      "evidence": line[:500], "reason": reason, "contamination_end_exclusive": None})
    return found


def walk_files(root):
    for directory, children, files in os.walk(root, followlinks=False):
        children[:] = sorted(d for d in children if d not in SKIP_DIRS and not Path(directory, d).is_symlink()
                             and Path(directory, d).resolve().is_relative_to(Path(root).resolve()))
        for name in sorted(files):
            path = Path(directory, name)
            if not path.is_symlink():
                yield path


def research_text(path):
    return (path.suffix.lower() in TEXT_EXTENSIONS and "/models/" not in path.as_posix()
            and not re.search(r"(?:xgb|lgb|model)\.(?:json|txt)$", path.name, re.I))


def contamination_scan(repo, paths):
    findings, limitations, references = [], [], set()
    this_file = Path(__file__).resolve()
    for path in paths:
        if not research_text(path) or path.resolve() == this_file or "UNTOUCHED_AVAILABILITY_AUDIT" in path.name:
            continue
        if path.stat().st_size > MAX_TEXT_BYTES:
            limitations.append("Text exceeds scan size limit: " + str(path))
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as error:
            limitations.append(str(path) + ": " + str(error))
            continue
        findings.extend(classify_text(text, str(path)))
        for literal in re.findall(r"[A-Za-z]:[\\/][^\r\n\"'<>]+", text):
            candidate = Path(literal.replace("\\\\", "\\"))
            if candidate.is_dir() or candidate.suffix.lower() in {".csv", ".gz", ".npz", ".parquet"}:
                references.add(str(candidate.resolve()))
    seen = set()
    try:
        objects = git_read(repo, "rev-list", "--objects", "--all").decode("utf-8").splitlines()
        for line in objects:
            oid, separator, name = line.partition(" ")
            if not separator or not research_text(Path(name)) or "untouched_availability" in name.lower():
                continue
            if oid in seen:
                continue
            seen.add(oid)
            size = int(git_read(repo, "cat-file", "-s", oid))
            if size > MAX_TEXT_BYTES:
                limitations.append("History blob exceeds limit: " + oid + ":" + name)
                continue
            text = git_read(repo, "cat-file", "blob", oid).decode("utf-8-sig")
            findings.extend(classify_text(text, "git-blob:" + oid + ":" + name))
    except (OSError, UnicodeError, subprocess.CalledProcessError, ValueError) as error:
        limitations.append("Git history coverage incomplete: " + str(error))
    return findings, limitations, sorted(references), len(seen)


def source_trace(repo):
    rows, problems = [], []
    for name, expected in FROZEN_SOURCE_HASHES.items():
        path = repo / name
        actual = file_hash(path) if path.is_file() else None
        rows.append({"path": name, "sha256": actual, "expected_sha256": expected, "identity_matches": actual == expected})
        if actual != expected:
            problems.append("Frozen source identity mismatch: " + name)
    for role, filename in (("discovery", "gold_independent_secondary_classifier_v1.py"),
                           ("confirmation", "gold_secondary_s4_confirmation_v1.py")):
        run = repo / "training_runs" / ARCHIVES[role][0]
        try:
            require(file_hash(run / "FINALIZED.json") == ARCHIVES[role][1], "Archive identity")
            final = read_json(run / "FINALIZED.json")
            snapshot = run / "training_script.py"
            expected = final["file_sha256"]["training_script.py"]
            actual = file_hash(repo / filename)
            require(file_hash(snapshot) == expected, "Archived source identity")
            rows.append({"path": filename, "sha256": actual, "expected_sha256": expected,
                         "archived_path": str(snapshot), "identity_matches": actual == expected})
            if actual != expected:
                problems.append("Research source identity mismatch: " + filename)
        except (OSError, ValueError, KeyError) as error:
            problems.append(role + " source provenance: " + str(error))
    run = repo / "training_runs" / "20260913T141817Z_gemini_cftc_gold_cot_b0_b1_v1"
    try:
        require(file_hash(run / "FINALIZED.json") == "592b68e98c1a0e9dd6f5e5af1d6aeadbca86df5072f2cbe85a064adf06c5ebfe", "B0 helper archive identity")
        final = read_json(run / "FINALIZED.json")
        actual = file_hash(run / "training_script.py")
        expected = final["file_sha256"]["training_script.py"]
        rows.append({"path": str(run / "training_script.py"), "sha256": actual,
                     "expected_sha256": expected, "identity_matches": actual == expected})
        require(actual == expected, "B0 helper source identity")
    except (OSError, ValueError, KeyError) as error:
        problems.append("B0 helper provenance: " + str(error))
    return rows, problems


def model_overlap(repo):
    """Only model JSON structure and whitelisted integer NPZ arrays are read."""
    import numpy as np
    rows, problems = [], []
    for role, (run_id, final_hash) in ARCHIVES.items():
        run = repo / "training_runs" / run_id
        try:
            require(file_hash(run / "FINALIZED.json") == final_hash, "Finalized archive identity")
            final = read_json(run / "FINALIZED.json")
            for name in ("execution_spec.json", "model_inventory.json", "secondary_evidence.npz"):
                require(file_hash(run / name) == final["file_sha256"][name], "Archived metadata identity: " + name)
            spec = read_json(run / "execution_spec.json")
            inventory = read_json(run / "model_inventory.json")
            require(len(inventory) == 3, "Exactly three frozen secondary models required")
            with zipfile.ZipFile(run / "secondary_evidence.npz") as archive:
                for number, item in enumerate(inventory, 1):
                    arrays = {}
                    for suffix in ("train_indices", "subset_indices", "train_ns"):
                        name = f"fold{number}_{suffix}.npy"
                        with archive.open(name) as stream:
                            values = np.lib.format.read_array(stream, allow_pickle=False)
                        require(values.ndim == 1 and values.dtype == np.int64, "Integer provenance arrays only")
                        arrays[suffix] = values
                    train, subset, times = (arrays[k] for k in ("train_indices", "subset_indices", "train_ns"))
                    require(len(train) == len(times) and len(subset) and np.all(np.diff(train) > 0) and np.all(np.diff(times) > 0), "Training index identity")
                    slots = np.searchsorted(train, subset)
                    require(np.all(slots < len(train)) and np.array_equal(train[slots], subset), "Subset membership")
                    digest = hashlib.sha256(str(subset.dtype).encode("ascii") + np.asarray(subset.shape, dtype=np.int64).tobytes() + subset.tobytes()).hexdigest()
                    require(digest == item["train_indices_sha256"], "Subset hash")
                    path = (run / item["path"]).resolve()
                    require(path.is_relative_to(run.resolve()) and file_hash(path) == item["sha256"] == final["file_sha256"][item["path"]], "Model hash")
                    learner = read_json(path)["learner"]
                    require(learner["feature_names"] == FEATURES and int(learner["learner_model_param"]["num_feature"]) == 31
                            and learner["objective"]["name"] == "binary:logistic", "Scorable model structure")
                    rows.append({"run": run_id, "role": role, "fold": item["fold"], "model_path": str(path), "sha256": item["sha256"],
                                 "training_subset_rows": len(subset), "latest_training_timestamp": stamp(int(times[slots].max())),
                                 "first_training_timestamp": stamp(int(times[slots].min())), "overlaps_2025": bool(np.any(times[slots] >= ns(CANDIDATE_START))),
                                 "frozen_model_structurally_sufficient": True})
            # Check B0 structure without constructing any XGBoost object.
            discovery_spec = read_json(repo / "training_runs" / ARCHIVES["discovery"][0] / "execution_spec.json")
            for item in spec["b0_models"]:
                path = repo / "training_runs" / discovery_spec["reference_run"] / item["path"]
                require(file_hash(path) == item["sha256"] and read_json(path)["learner"]["feature_names"] == FEATURES, "B0 model structure/identity")
        except (OSError, ValueError, KeyError, IndexError, zipfile.BadZipFile) as error:
            problems.append(role + " training/model provenance: " + str(error))
    return rows, problems


def warmup_bounds(blocks):
    """Determine structural validity from complete causal input rows, not features."""
    if any(tf not in blocks for tf in TIMEFRAMES):
        return None, None, ["Missing required timeframe file"]
    m1, good = blocks["M1"]
    if len(m1) < 254 or not all(good) or any(m1[i] <= m1[i-1] for i in range(1, len(m1))):
        return None, None, ["M1 insufficient warmup or invalid/duplicate/unordered raw context"]
    first = m1[253]  # ATR(14) -> rolling(240) -> one-row feature lag.
    problems = []
    for tf in TIMEFRAMES[1:]:
        times, valid = blocks[tf]
        if len(times) < 21 or not all(valid) or any(times[i] <= times[i-1] for i in range(1, len(times))):
            problems.append(tf + ": missing 20 genuine closed-bar lookback plus publication row, or invalid context")
            continue
        slot = bisect.bisect_left(m1, times[20]) + 1
        if slot >= len(m1):
            problems.append(tf + ": no later M1 feature-lag row")
        else:
            first = max(first, m1[slot])
    return (None if problems else first), m1[14], problems


def complete_coverage(times, valid, start, schedule):
    """Only an explicitly supplied broker schedule can certify missing bars/days."""
    if not schedule or start is None:
        return {"days": 0, "months": 0, "sessions": None, "eligible_rows": 0, "end": None, "issues": ["No verified market schedule or safe start"]}
    begin, end = parse_time(schedule["start"]), parse_time(schedule["end_exclusive"])
    require(begin.tzinfo is None and end.tzinfo is None and begin < end, "Schedule clock/range")
    begin = max(begin, start)
    present = {t for t, ok in zip(times, valid) if ok and ns(begin) <= t < ns(end)}
    expected, sessions = set(), 0
    last = None
    for entry in schedule["sessions"]:
        a, b = parse_time(entry["start"]), parse_time(entry["end_exclusive"])
        require(a < b and a.second == b.second == a.microsecond == b.microsecond == 0, "Minute-aligned schedule")
        require(last is None or a >= last, "Schedule sessions must not overlap")
        last = b
        if begin <= a and b <= end:
            sessions += 1
        lo, hi = max(a, begin), min(b, end)
        if lo < hi:
            expected.update(range(ns(lo), ns(hi), 60_000_000_000))
    require(expected and times and ns(end) <= max(times) + 60_000_000_000, "Schedule must contain bars and stay within observed coverage")
    missing, extra = expected - present, present - expected
    # A incomplete day ends the certifiable contiguous interval; do not bridge gaps.
    bad = sorted(missing | extra)
    complete_end = min(end, (EPOCH + timedelta(microseconds=bad[0] // 1000)).replace(hour=0, minute=0, second=0, microsecond=0)) if bad else end
    day_start = begin.replace(hour=0, minute=0, second=0, microsecond=0)
    if day_start < begin:
        day_start += timedelta(days=1)
    day_end = complete_end.replace(hour=0, minute=0, second=0, microsecond=0)
    days = max(0, (day_end - day_start).days)
    months = 0
    cursor = day_start.replace(day=1)
    if cursor < day_start:
        cursor = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
    while cursor < day_end:
        following = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
        if following > day_end:
            break
        months += 1
        cursor = following
    return {"days": days, "months": months, "sessions": sessions if not bad else None,
            "eligible_rows": sum(ns(day_start) <= t < ns(day_end) for t in expected & present),
            "start": time_text(day_start) if days else None,
            "end": time_text(day_end) if days else None, "missing_bars": len(missing), "unexpected_bars": len(extra),
            "issues": ["Schedule/observed bar mismatch"] if bad else []}


def timeframe_interval(name):
    if re.fullmatch(r"M\d+", name):
        return int(name[1:]) * 60
    if re.fullmatch(r"H\d+", name):
        return int(name[1:]) * 3600
    return {"Daily": 86400, "Weekly": 604800, "Monthly": 31 * 86400}.get(name, 60)


def audit(repo, extra_roots):
    """Inventory evidence only. No repository modules are imported or executed."""
    commit = git_read(repo, "rev-parse", "HEAD").decode().strip()
    paths = list(walk_files(repo))
    findings, limitations, references, history_count = contamination_scan(repo, paths)
    chain, issues = source_trace(repo)
    models, model_issues = model_overlap(repo)
    issues.extend(model_issues)
    roots = set(extra_roots)
    for reference in references:
        path = Path(reference)
        if not path.is_relative_to(repo) or not path.exists():
            roots.add(path)
    candidates = set(paths)
    for root in sorted(roots):
        if root.is_dir():
            candidates.update(walk_files(root))
        elif root.is_file():
            candidates.add(root)
        else:
            limitations.append("Referenced source unavailable: " + str(root))
    canonical = {}
    for path in sorted(repo.glob("GOLD#_*.csv")):
        tf = path.name.split("_")[1]
        canonical.setdefault(tf, []).append(path.resolve())
    blocks, inventory, m1_info = {}, [], None
    missing = [tf + ": source file" for tf in TIMEFRAMES if tf not in canonical]
    for tf, files in canonical.items():
        if len(files) != 1:
            issues.append(tf + ": ambiguous source files; frozen loader overwrites in lexical order")
    for path in sorted(candidates):
        if not re.search(r"gold|xauusd", str(path), re.I) or path.suffix.lower() not in {".csv", ".gz", ".npz", ".parquet", ".npy", ".pkl"}:
            continue
        path = path.resolve()
        match = next((tf for tf, files in canonical.items() if path in files), None)
        # Reports/derived caches are inventoried by identity; outcomes never parsed.
        raw_csv = (path.suffix.lower() == ".csv" or path.name.lower().endswith(".csv.gz")) and (
            match is not None or re.search(r"(?:GOLD#|XAUUSD)[_.-](?:M\d+|H\d+|Daily|Weekly|Monthly)[_.-]", path.name, re.I))
        if not raw_csv:
            inventory.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": file_hash(path),
                              "kind": "derived_or_unrecognized_dataset", "timestamp_coverage": None,
                              "reason": "Metadata only; outcome caches and ledgers are not parsed"})
            continue
        try:
            row, times, valid = inventory_csv(path, interval_seconds=timeframe_interval(match or "M1"))
            row.update(kind="canonical_raw" if match else "alternative_raw_not_substituted", timeframe=match,
                       canonical_symbol="GOLD#" if match else "UNVERIFIED_ALTERNATIVE")
            inventory.append(row)
            if match in TIMEFRAMES and len(canonical[match]) == 1:
                blocks[match] = (times, valid)
                if row["invalid_timestamps"]:
                    issues.append(match + ": invalid timestamp rows cannot be reconstructed")
                if not row["row_count_after_2025_01_01"]:
                    missing.append(match + ": no post-boundary bars")
                missing.extend(match + ": " + field for field in row["missing_required_columns"])
                if match == "M1":
                    m1_info = row
                    if "SPREAD" not in row["columns"]:
                        missing.append("M1: SPREAD")
                    if "TIME" not in row["columns"]:
                        issues.append("M1 DATE-only timestamps cannot establish minute coverage")
        except (OSError, ValueError, UnicodeError, csv.Error) as error:
            issues.append("Cannot inventory " + str(path) + ": " + str(error))
            inventory.append({"path": str(path), "error": str(error)})
    feature_bound, execution_bound, warmup_issues = warmup_bounds(blocks)
    issues.extend(warmup_issues)
    # Loader semantics provide a structural bound, but not broker calendar,
    # closed-bar provenance or an attested original EWM source prefix.
    issues.extend([
        "Broker market calendar (holidays, closures and partial sessions) is not defined by frozen session gates",
        "Source timestamps are naive: actual timezone/DST and bar open/close labeling require provenance",
        "Exact recursive EWM initialization needs the attested original source prefix; finite warmup alone is insufficient",
        "Higher-timeframe final-bar closure and ongoing freshness require broker calendar/provenance",
    ])
    confirmed = any(f["classification"] == "CONFIRMED_CONTAMINATION" for f in findings)
    if any(m["overlaps_2025"] for m in models):
        confirmed = True
        for model in models:
            if model["overlaps_2025"]:
                findings.append({"classification": "CONFIRMED_CONTAMINATION", "path": model["model_path"],
                                 "reason": "Verified frozen secondary fitting indices overlap candidate period",
                                 "latest_training_timestamp": model["latest_training_timestamp"]})
    if confirmed:
        issues.append("Confirmed candidate-period contamination; no verified bounded clean successor")
    if any(f["classification"] == "POTENTIAL_CONTAMINATION" for f in findings):
        issues.append("Dated research leads need review; source existence does not prove execution")
    issues.append("Repository search cannot establish absence of off-repository/manual outcome inspection")
    issues.extend(limitations)
    issues.extend("Missing " + item for item in missing)
    first_feature = first_execution = None
    safe = safe_start(CANDIDATE_START, first_feature, first_execution, None, unbounded=confirmed)
    coverage = complete_coverage(*blocks.get("M1", ([], [])), safe, None)
    base = m1_info or {}
    first, last = base.get("actual_first_available_timestamp"), base.get("actual_last_available_timestamp")
    structural_models = len(models) == 6 and not model_issues
    return {
        "audit_version": AUDIT_VERSION, "repo_commit": commit,
        "repo_status": git_read(repo, "status", "--short").decode("utf-8"),
        "source_files": inventory, "source_chain": chain,
        "canonical_symbol": "GOLD#", "alternative_symbols": "XAUUSD is inventoried but never silently substituted",
        "timezone_semantics": "Naive broker/API source clock; research offset=0; true UTC/DST not established",
        "candidate_untouched_start": time_text(CANDIDATE_START), "raw_candidate_start": time_text(CANDIDATE_START),
        "actual_first_available_timestamp": first, "actual_last_available_timestamp": last,
        "complete_start_timestamp": None, "complete_end_timestamp": None,
        "available_days": None, "available_months": None,
        "certified_untouched_days": 0, "certified_untouched_months": 0,
        "duration_classification": duration_classification(0), "duration_basis": "certified complete untouched days only; actual completeness unknown",
        "raw_elapsed_days": (parse_time(last) - parse_time(first)).total_seconds() / 86400 if first and last else None,
        "observed_trading_days": base.get("observed_trading_days", 0),
        "eligible_m1_rows": None, "complete_trading_sessions": None,
        "bar_frequency": "M1 base; separate native timeframe CSVs; no resampling",
        "required_timeframes": list(TIMEFRAMES), "row_count_after_2025_01_01": base.get("row_count_after_2025_01_01", 0),
        "required_columns": {"M1": ["DATE", "TIME", *RAW_COLUMNS, "SPREAD"], "other_timeframes": ["DATE", *RAW_COLUMNS],
                             "S5_derived": ["TIME_DT", *RAW_COLUMNS, "ATR", "M1_RSI", "SPREAD"]},
        "missing_required_columns": missing, "frozen_features": FEATURES,
        "feature_pipeline_reconstructable": False, "execution_pipeline_reconstructable": False,
        "structural_feature_lower_bound": stamp(feature_bound), "structural_execution_lower_bound": stamp(execution_bound),
        "earliest_feature_valid_start": None, "earliest_execution_valid_start": None,
        "warmup": {"m1_minimum_rows": 254, "formula": "ATR rolling14 -> ATR mean rolling240 -> lag1",
                   "ema": "MACD EMA12/26, signal EMA9 adjust=False: exact state requires original full prefix",
                   "higher_timeframe": "20 genuine prior closes, trend shift1, backward asof, then M1 lag1",
                   "initial_trend_warning": "np.where emits -1 before rolling20 exists; not genuine complete lookback",
                   "pre_2025": "context only, never test evidence"},
        "execution_semantics": {"source": "gold_gemini_execution_semantics_v1.py (identity checked)",
            "point": 0.01, "spread": "SPREAD required; nonfinite/nonpositive values use frozen 30-point fallback",
            "extra_cost_points": 5, "stress_extra_points": 10,
            "session": "Frozen hour/weekday entry gates are not a complete market-data calendar",
            "rsi_atr": "causal lagged M1 RSI14 and ATR14; no indicator arrays generated by this audit",
            "entry": "next-row open; entry-bar high/low inspected stop-first by S5, never replayed here",
            "timeout_wallclock_minutes": 90,
            "state": "Frozen cooldown, occupancy, sizing and TP/SL retained by source identity; not executed"},
        "data_gaps": [{"path": r["path"], "gap_count": r.get("gap_count"), "examples": r.get("data_gaps", [])}
                      for r in inventory if "gap_count" in r],
        "duplicate_timestamps": base.get("duplicate_timestamps"), "non_monotonic_timestamps": base.get("non_monotonic_timestamps"),
        "provenance_findings": {"scan_limitations": limitations, "historical_blobs_inspected": history_count,
                                "local_references": references, "coverage": coverage,
                                "absence_of_matches_is_not_untouched_evidence": True},
        "contamination_findings": findings, "training_overlap": models,
        "test_designs": {
            "A_strict_frozen": {"model_artifacts_structurally_sufficient": structural_models,
                "feasible_as_untouched_now": False, "retraining_required": False,
                "condition": "Features and untouched provenance must exist; predeclare which frozen fold/model combination is used"},
            "B_retrained": {"feasibility": "CONDITIONAL_NEW_PREDECLARED_PROTOCOL_REQUIRED", "retraining_required": True,
                "condition": "Fit strictly before untouched boundary; preserve boundary and label horizon; do not use test outcomes"},
            "chosen_design": None, "future_final_model_requires_retraining": "Depends on A versus B; no design selected"},
        "untouched_status": verdict(bool(missing), confirmed, not issues, 0),
        "earliest_safe_untouched_start": time_text(safe), "latest_safe_complete_end": None,
        "blocking_issues": sorted(set(issues)), "production_promotion": False, "production_change": False,
    }


def self_test():
    allowed = [["git", "rev-parse", "HEAD"], ["git", "status", "--short"],
               ["git", "rev-list", "--objects", "--all"], ["git", "cat-file", "-p", "abc123"]]
    for command in allowed:
        for binary in ("git", "git.exe", r"C:\Program Files\Git\cmd\git.exe", "/usr/bin/git"):
            argv = [binary, *command[1:]]
            for executable in (None, binary, r"C:\Program Files\Git\cmd\git.exe"):
                for representation in (argv, tuple(argv), subprocess.list2cmdline(argv)):
                    require(is_allowed_git_subprocess(executable, representation), "Read-only Git representation")
                    # Exercise the installed hook with the real Windows event
                    # shape, including executable=None and serialized argv.
                    # sys.audit emits an event only; it launches no process.
                    sys.audit("subprocess.Popen", executable, representation, "synthetic", {})
    rejected = [["git", name] for name in (
        "add", "commit", "checkout", "restore", "reset", "clean", "fetch",
        "pull", "push", "gc", "config")]
    rejected += [["git", "add", "."], ["git", "commit", "-m", "x"], ["git", "clean", "-fd"],
                 ["python", "x.py"], ["powershell", "..."], ["cmd", "/c", "git status"],
                 ["git.cmd", "status"], ["git", "-c", "alias.x=status", "x"],
                 [], ["git"], ["git", "STATUS"], ["git", "status", "bad\nargument"]]
    rejected += [["git", "status", symbol, "echo"] for symbol in ("&", "&&", "|", ";")]
    for argv in rejected:
        for representation in (argv, tuple(argv), subprocess.list2cmdline(argv)):
            require(not is_allowed_git_subprocess(None, representation), "Forbidden subprocess")
            try:
                sys.audit("subprocess.Popen", None, representation, "synthetic", {})
            except ValueError:
                pass
            else:
                raise AssertionError("Hook accepted forbidden subprocess")
    for executable, argv in [("python", ["git", "status"]), ("cmd.exe", "git status"),
                             ("git.exe", ["python", "status"]), (None, "echo arbitrary command"),
                             (None, '"git status"'), (None, 'git "status'),
                             (None, 'git sta"tus"'), (None, 'git status "trailing\\"'),
                             (None, "git status&&echo bad"), (None, "git status;echo bad"),
                             (None, "git status | echo bad"), (None, "git status & echo bad"),
                             (None, b"git status"), (None, ["git", "status", None])]:
        require(not is_allowed_git_subprocess(executable, argv), "Ambiguous/non-Git subprocess")
    header = "DATE,TIME,OPEN,HIGH,LOW,CLOSE,SPREAD\n"
    fixture = header + "2024.12.31,23:59:00,2,3,1,2,30\n2025.01.01,00:00:00,2,3,1,2,30\n2025.01.01,00:02:00,2,3,1,2,0\n2025.01.01,00:02:00,2,3,1,2,30\n2025.01.01,00:01:00,2,3,1,2,30\n"
    row, times, good = inventory_csv(io.StringIO(fixture))
    require(row["row_count_after_2025_01_01"] == 4 and row["duplicate_timestamps"] == 1
            and row["non_monotonic_timestamps"] == 1 and row["gap_count"] == 1, "Boundary/duplicate/order/gap")
    require(times[0] < ns(CANDIDATE_START) and len(good) == 5 and row["spread_fallback_rows"] == 1, "Warmup retained; no test counting")
    row, _, _ = inventory_csv(io.StringIO("DATE,TIME,OPEN,HIGH,LOW\n2025.01.01,00:00:00,2,3,1\n"))
    require("CLOSE" in row["missing_required_columns"], "Missing column")
    row, _, good = inventory_csv(io.StringIO(header + "2025.01.01,00:00:00,nan,3,1,2,30\n"))
    require(row["nonfinite_required_inputs"] == 1 and not good[0], "NaN raw input")
    for text, expected in [("GOLD raw ingestion schema 2025-01-01", "SAFE_CONTEXT"),
                           ("GOLD threshold tuning 2025-01-01", "POTENTIAL_CONTAMINATION"),
                           ("GOLD historical 2014-2026 inspected repeatedly", "CONFIRMED_CONTAMINATION")]:
        require(classify_text(text, "AGENTS.md")[0]["classification"] == expected, "Contamination classification")
    warm = datetime(2024, 12, 1)
    later = datetime(2025, 2, 1)
    require(safe_start(CANDIDATE_START, warm, warm, None) == CANDIDATE_START, "Warmup not test evidence")
    require(safe_start(CANDIDATE_START, warm, later, None) == later, "Maximum safe boundary")
    require(safe_start(CANDIDATE_START, warm, warm, later) == later, "Contamination cutoff")
    require(safe_start(CANDIDATE_START, warm, warm, None, True) is None
            and safe_start(CANDIDATE_START, None, warm, None) is None, "Unknown boundary stays unknown")
    for days, label in [(0, "INSUFFICIENT"), (89, "INSUFFICIENT"), (90, "LIMITED"), (179, "LIMITED"),
                        (180, "MODERATE"), (364, "MODERATE"), (365, "SUBSTANTIAL")]:
        require(duration_classification(days) == label + "_DURATION", "Frozen duration threshold")
    for args, expected in [((False, False, True, 90), "READY_FOR_PROTOCOL_DESIGN"),
                           ((True, False, True, 365), "NOT_READY_MISSING_DATA"),
                           ((False, True, True, 365), "NOT_READY_CONTAMINATED"),
                           ((False, False, False, 365), "NOT_READY_INCOMPLETE_PROVENANCE"),
                           ((False, False, True, 89), "NOT_READY_INSUFFICIENT_DURATION")]:
        require(verdict(*args) == expected, "Verdict")
    minute = 60_000_000_000
    m1 = array("q", (ns(warm) + i * minute for i in range(300)))
    blocks = {tf: (m1, array("b", [1] * 300)) for tf in TIMEFRAMES}
    feature, execution, problems = warmup_bounds(blocks)
    require(feature == m1[253] and execution == m1[14] and not problems, "Causal structural warmup")
    begin = CANDIDATE_START
    end = begin + timedelta(days=1)
    full = array("q", range(ns(begin), ns(end), minute))
    schedule = {"start": time_text(begin), "end_exclusive": time_text(end),
                "sessions": [{"start": time_text(begin), "end_exclusive": time_text(end)}]}
    require(complete_coverage(full, [1] * len(full), begin, schedule)["days"] == 1, "Complete synthetic day")
    validity = [1] * len(full)
    validity[10] = 0
    require(complete_coverage(full, validity, begin, schedule)["days"] == 0, "Gap prevents complete day")
    require(complete_coverage(full, validity, begin, None)["end"] is None, "No guessed schedule")
    try:
        ImportGuard().find_spec("xgboost")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Model import guard")
    print("SELF_TEST_PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true", help="Synthetic in-memory checks only")
    mode.add_argument("--output", type=Path, help="Explicit new JSON output; parent must already exist")
    parser.add_argument("--source-root", type=Path, action="append", default=[], help="Additional read-only local dataset source")
    args = parser.parse_args()
    try:
        if args.self_test:
            require(not args.source_root, "Self-test does not inspect source roots")
            install_guards(None)
            self_test()
            return 0
        output = args.output.resolve()
        require(output.suffix.lower() == ".json" and not output.exists() and output.parent.is_dir(), "Output must be a new JSON in an existing directory")
        require(not any(part.lower() in {"training_runs", ".git", "models", ".venv"} for part in output.parts), "Output cannot be inside protected run/model/Git paths")
        install_guards(output)
        report = audit(ROOT, [p.resolve() for p in args.source_root])
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        print(report["untouched_status"])
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print("AUDIT_ERROR: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
