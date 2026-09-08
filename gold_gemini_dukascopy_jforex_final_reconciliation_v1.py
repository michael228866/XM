from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import gold_gemini_dukascopy_fx_tick_gap_adjudication_v1 as parent_code
import gold_gemini_usd_fx_pressure_source_reconciliation_v1 as matrix_code
import training_run_history as archive


ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "training_runs/20260907T114431Z_gemini_dukascopy_fx_tick_gap_adjudication_v1"
SOURCE_PARENT = ROOT / "training_runs/20260906T180930Z_gemini_usd_fx_pressure_source_reconciliation_v1"
JAVA_PROJECT = ROOT / "jforex_final_reconciliation"
TOOLS = ROOT / ".research_tools"
JAVA = TOOLS / "liberica-jdk8-full/jdk8u472-full/bin/java.exe"
MAVEN = TOOLS / "apache-maven-3.9.16/bin/mvn.cmd"
M2 = TOOLS / "m2"
INSTRUMENTS = ("EUR/USD", "GBP/USD", "USD/JPY")
CODES = {"EUR/USD": "EURUSD", "GBP/USD": "GBPUSD", "USD/JPY": "USDJPY"}
POINTS = {"EUR/USD": 1e-5, "GBP/USD": 1e-5, "USD/JPY": 1e-3}
FIELDS = ("open", "high", "low", "close")
FEATURES = parent_code.FEATURES
MINUTE_NS = 60_000_000_000
HOUR_NS = 60 * MINUTE_NS
MISMATCH_PAIR = "GBP/USD"
MISMATCH_NS = pd.Timestamp("2023-12-10T22:59:00Z").value
EXPECTED_TIMESTAMP_HASHES = parent_code.EXPECTED_TIMESTAMP_HASHES
JFOREX_JNLP_URL = "https://platform.dukascopy.com/demo_3/jforex_3.jnlp"
JFOREX_JNLP_SHA256 = "4e5adcbb29116e7f17b3babfc4aa47590d06baca50a98745d300d4824a1a70e9"
JFOREX_SDK_SHA256 = "a7a2fb6c070f800145adf5d88a7de9ed37e7544878b12b4312ea006365947016"
JFOREX_API_SOURCE_SHA256 = "259dfe7f4edf9424a70077a4b223a3dd1541cb1b920cfe48f17c62256a062df0"
LIBERICA_SHA1 = "b57a92d0d3288e73496f65594e20e8771a9dbdf8"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def digest(values: np.ndarray) -> str:
    return parent_code.digest(values)


def logical_hash(*arrays: np.ndarray) -> str:
    return parent_code.logical_hash(*arrays)


def tree_hash(path: Path) -> str:
    result = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        relative = item.relative_to(path).as_posix().encode("utf-8")
        result.update(len(relative).to_bytes(8, "big"))
        result.update(relative)
        result.update(bytes.fromhex(archive.file_sha256(item)))
    return result.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def load_native() -> dict[str, dict[str, np.ndarray]]:
    return {
        pair: load_npz(PARENT / f"fx_source_DUKASCOPY_{CODES[pair]}.npz")
        for pair in INSTRUMENTS
    }


def row_at(source: dict[str, np.ndarray], minute_ns: int) -> dict[str, float] | None:
    position = int(np.searchsorted(source["open_utc_ns"], minute_ns))
    if position >= len(source["open_utc_ns"]):
        return None
    if int(source["open_utc_ns"][position]) != minute_ns:
        return None
    return {field: float(source[field][position]) for field in FIELDS}


def bars_agree(
    left: dict[str, float] | None,
    right: dict[str, float] | None,
    tolerance: float,
) -> bool:
    return bool(
        left is not None
        and right is not None
        and all(abs(left[field] - right[field]) <= tolerance + 1e-12 for field in FIELDS)
    )


def canonical_ticks(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["time_ms", "bid", "ask", "bid_volume", "ask_volume"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    result = frame[columns].copy()
    return result.sort_values(columns, kind="stable").drop_duplicates(columns).reset_index(drop=True)


def frame_hash(frame: pd.DataFrame) -> str:
    result = hashlib.sha256()
    for row in frame.itertuples(index=False):
        result.update(np.asarray([int(row.time_ms)], dtype=">i8").tobytes())
        result.update(np.asarray(
            [row.bid, row.ask, row.bid_volume, row.ask_volume], dtype=">f8"
        ).tobytes())
    return result.hexdigest()


def aggregate_ticks(frame: pd.DataFrame) -> dict[int, dict[str, float]]:
    if frame.empty:
        return {}
    ticks = canonical_ticks(frame).sort_values(
        ["time_ms", "bid", "ask", "bid_volume", "ask_volume"], kind="stable"
    )
    ticks["minute_ms"] = (ticks["time_ms"].astype("int64") // 60_000) * 60_000
    result: dict[int, dict[str, float]] = {}
    for minute, group in ticks.groupby("minute_ms", sort=True):
        bids = group["bid"].to_numpy(float)
        result[int(minute) * 1_000_000] = {
            "open": float(bids[0]),
            "high": float(bids.max()),
            "low": float(bids.min()),
            "close": float(bids[-1]),
        }
    return result


def exact_times(run: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    shutil.copy2(PARENT / "exact_timestamps.npz", run / "exact_timestamps.npz")
    blocks: dict[str, np.ndarray] = {}
    pieces = []
    with np.load(run / "exact_timestamps.npz", allow_pickle=False) as source:
        for block, expected in EXPECTED_TIMESTAMP_HASHES.items():
            broker = source[f"{block}_broker_ns"].astype(np.int64)
            utc = source[f"{block}_utc_ns"].astype(np.int64)
            if digest(broker) != expected:
                raise RuntimeError(f"timestamp hash mismatch: {block}")
            blocks[block] = utc
            pieces.append(utc)
    return blocks, np.unique(np.concatenate(pieces))


def parent_state(
    native: dict[str, dict[str, np.ndarray]], times: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, list[tuple[int, int]]]]:
    all_gaps = pd.read_csv(SOURCE_PARENT / "source_gap_audit.csv")
    classes = pd.read_csv(PARENT / "gap_minute_classification.csv.gz")
    closures = {pair: [] for pair in INSTRUMENTS}
    for gap in all_gaps[
        ~all_gaps["classification"].eq("unexplained_source_gap")
    ].itertuples(index=False):
        closures[gap.instrument].append((
            pd.Timestamp(gap.gap_start_utc).value + MINUTE_NS,
            pd.Timestamp(gap.gap_end_utc).value + MINUTE_NS,
        ))
    verified = classes[
        classes["classification"].eq("verified_no_tick_interval")
    ].copy()
    verified["minute_ns"] = np.asarray(
        [pd.Timestamp(value).value for value in verified["minute_open_utc"]],
        dtype=np.int64,
    )
    for pair in INSTRUMENTS:
        values = np.sort(
            verified.loc[verified["instrument"].eq(pair), "minute_ns"].to_numpy(np.int64)
        )
        if len(values):
            cuts = np.flatnonzero(np.diff(values) > MINUTE_NS) + 1
            for group in np.split(values, cuts):
                closures[pair].append((
                    int(group[0] + MINUTE_NS), int(group[-1] + 2 * MINUTE_NS)
                ))
        closures[pair] = parent_code.merge_intervals(closures[pair])
    matrix, unknown, _ = matrix_code.build_matrix(times, native, closures)
    if int(unknown.any(axis=1).sum()) != 6_931:
        raise RuntimeError("previous_remaining_unresolved_rows is not 6931")
    return matrix, unknown, closures


def write_request_manifest(run: Path) -> pd.DataFrame:
    audit = pd.read_csv(PARENT / "tick_request_audit.csv")
    errors = audit[audit["state"].eq("retrieval_error")].copy()
    if len(errors) != 783:
        raise RuntimeError(f"expected 783 retrieval errors, found {len(errors)}")
    errors = errors.sort_values(["instrument", "hour_utc"], ignore_index=True)
    rows = []
    for index, row in errors.iterrows():
        hour = pd.Timestamp(row["hour_utc"])
        rows.append({
            "request_id": f"R{index:04d}",
            "pair": row["instrument"],
            "from_ms": int(hour.value // 1_000_000),
            "to_ms": int(hour.value // 1_000_000 + 3_600_000),
            "kind": "retrieval_error_hour",
        })
    rows.append({
        "request_id": "MISMATCH_GBPUSD_20231210T2259Z",
        "pair": MISMATCH_PAIR,
        "from_ms": int((MISMATCH_NS - 2 * MINUTE_NS) // 1_000_000),
        "to_ms": int((MISMATCH_NS + 3 * MINUTE_NS) // 1_000_000),
        "kind": "mismatch",
    })
    result = pd.DataFrame(rows)
    result.to_csv(run / "jforex_request_manifest.tsv", sep="\t", index=False)
    return result


def preregister(run: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    if archive.validate_run(PARENT):
        raise RuntimeError("parent finalized run failed provenance validation")
    if archive.validate_run(SOURCE_PARENT):
        raise RuntimeError("source parent finalized run failed provenance validation")
    manifest = archive.read_json(run / "manifest.json")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/main")
    status_lines = git("status", "--porcelain").splitlines()
    run_prefix = f"training_runs/{run.name}/"
    unexpected_dirty = [
        line for line in status_lines
        if run_prefix not in line.replace("\\", "/")
    ]
    if manifest["git_dirty"] or unexpected_dirty or head != remote or head != manifest["git_commit"]:
        raise RuntimeError("clean pushed pre-run Git state required")
    if archive.file_sha256(Path(__file__)) != manifest["training_script_sha256"]:
        raise RuntimeError("executed script differs from immutable snapshot")
    if not JAVA.exists() or not MAVEN.exists() or not M2.exists():
        raise RuntimeError("repository-local JForex runtime is incomplete")
    if os.getenv("DUKASCOPY_USERNAME", "").strip() == "" or os.getenv(
        "DUKASCOPY_PASSWORD", ""
    ).strip() == "":
        raise RuntimeError("Dukascopy credentials are not available in process environment")

    parent_before = tree_hash(PARENT)
    source_parent_before = tree_hash(SOURCE_PARENT)
    blocks, times = exact_times(run)
    native = load_native()
    _matrix, old_unknown, _closures = parent_state(native, times)
    old_rows = times[old_unknown.any(axis=1)]
    if len(old_rows) != 6_931:
        raise RuntimeError("frozen unresolved row identity failure")
    np.savez_compressed(
        run / "preserved_defect_universe.npz",
        unresolved_utc_ns=old_rows,
        unresolved_feature_mask=old_unknown[old_unknown.any(axis=1)],
        feature_names=np.asarray(FEATURES),
    )
    requests = write_request_manifest(run)
    shutil.copy2(
        ROOT / "gold_gemini_dukascopy_jforex_final_reconciliation_v1_validator.py",
        run / "validator_script.py",
    )
    shutil.copy2(
        JAVA_PROJECT / "src/main/java/local/xm/JForexFinalCollector.java",
        run / "jforex_collector.java",
    )
    shutil.copy2(JAVA_PROJECT / "pom.xml", run / "jforex_pom.xml")
    defect_manifest = {
        "parent_run": PARENT.name,
        "parent_tree_sha256_before": parent_before,
        "source_parent_tree_sha256_before": source_parent_before,
        "original_gap_intervals": 554,
        "original_retrieval_error_hours": 783,
        "previous_remaining_unresolved_rows": 6_931,
        "previous_remaining_unresolved_timestamp_sha256": digest(old_rows),
        "original_mismatch": {
            "pair": MISMATCH_PAIR,
            "minute_open_utc": pd.Timestamp(MISMATCH_NS, tz="UTC").isoformat(),
            "tolerance": POINTS[MISMATCH_PAIR],
        },
        "six_timestamp_hashes": EXPECTED_TIMESTAMP_HASHES,
        "request_manifest_sha256": archive.file_sha256(run / "jforex_request_manifest.tsv"),
        "request_count": len(requests),
    }
    archive.write_json(run / "cohort_manifest.json", defect_manifest)
    manifest["pre_run_git"] = {
        "pre_run_git_commit": head,
        "pre_run_git_dirty": False,
        "head_sha": head,
        "origin_main_sha": remote,
        "head_equals_origin_main": True,
    }
    manifest["operational_hashes_before"] = {
        name: archive.file_sha256(ROOT / name)
        for name in ("gemini.py", "gold_long_recent_candidate_xgb.json")
    }
    manifest["protected_runs_before"] = {
        PARENT.name: parent_before,
        SOURCE_PARENT.name: source_parent_before,
    }
    manifest["foundation_specification"] = {
        "provider": "Dukascopy only",
        "instruments": list(INSTRUMENTS),
        "features": list(FEATURES),
        "horizons_minutes": [1, 5, 15, 60],
        "usd_orientation": [-1.0, -1.0, 1.0],
        "equal_weighting": True,
        "dispersion_ddof": 1,
        "staleness_minutes": 5,
        "tolerance": POINTS,
        "tick_api_bounds": "inclusive from and to",
        "aggregation_bins": "minute_open <= tick_time < minute_open + 60 seconds",
        "bar_api": "Period.ONE_MIN, OfferSide.BID, Filter.NO_FILTER",
        "jnlp_url": JFOREX_JNLP_URL,
        "jnlp_sha256": JFOREX_JNLP_SHA256,
        "sdk_zip_sha256": JFOREX_SDK_SHA256,
        "api_source_jar_sha256": JFOREX_API_SOURCE_SHA256,
        "java_runtime": "Liberica Full JDK 8u472+9 repository-local",
        "java_zip_server_etag_sha1": LIBERICA_SHA1,
        "credentials_retained": False,
    }
    manifest["model"].update({
        "trained": False,
        "model_type": "not_applicable_data_only",
        "features": list(FEATURES),
        "feature_count": len(FEATURES),
        "not_applicable_reason": "JForex historical data reconciliation only",
    })
    manifest["search"].update({
        "performed": False,
        "predefined_search_space": {},
        "candidate_results_file": "not_applicable_data_only",
        "not_applicable_reason": "Frozen defect universe and arbitration rules",
    })
    manifest["promotion"].update({
        "requested": False,
        "gate_result": "not_applicable_data_only",
        "replacement_authorized": False,
        "operational_artifact_changed": False,
    })
    archive.write_json(run / "manifest.json", manifest)
    return manifest, blocks, times, old_unknown


def run_collector(run: Path) -> None:
    compile_command = [
        str(MAVEN), "-o", "-f", str(JAVA_PROJECT / "pom.xml"),
        f"-Dmaven.repo.local={M2}", "-DskipTests", "compile",
    ]
    subprocess.run(compile_command, cwd=ROOT, check=True)
    jars = sorted(
        path for path in M2.rglob("*.jar")
        if not path.name.endswith("-sources.jar") and not path.name.endswith("-javadoc.jar")
    )
    classpath = os.pathsep.join(
        [str(JAVA_PROJECT / "target/classes"), *(str(path) for path in jars)]
    )
    command = [
        str(JAVA), "-cp", classpath, "local.xm.JForexFinalCollector",
        str(run / "jforex_request_manifest.tsv"), str(run),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def read_jforex(run: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit = pd.read_csv(run / "jforex_request_audit.tsv", sep="\t", keep_default_na=False)
    ticks = pd.read_csv(run / "jforex_ticks.tsv.gz", sep="\t")
    bars = pd.read_csv(run / "jforex_bars.tsv.gz", sep="\t")
    if len(audit) != 784 * 3:
        raise RuntimeError(f"expected 2352 JForex audit rows, found {len(audit)}")
    return audit, ticks, bars


def request_tick_state(
    request_id: str, audit: pd.DataFrame, ticks: pd.DataFrame
) -> tuple[str, pd.DataFrame, dict[str, Any]]:
    rows = audit[(audit["request_id"] == request_id) & audit["mechanism"].isin(
        ["getTicks", "readTicks"]
    )].set_index("mechanism")
    frames = {
        mechanism: canonical_ticks(ticks[
            (ticks["request_id"] == request_id) & (ticks["mechanism"] == mechanism)
        ])
        for mechanism in ("getTicks", "readTicks")
    }
    success = {
        mechanism: rows.loc[mechanism, "status"] == "success"
        for mechanism in ("getTicks", "readTicks")
    }
    normalized_hashes = {mechanism: frame_hash(frames[mechanism]) for mechanism in frames}
    if success["getTicks"] and success["readTicks"]:
        if normalized_hashes["getTicks"] != normalized_hashes["readTicks"]:
            state = "inconsistent_provider_history"
            canonical = pd.DataFrame(columns=frames["getTicks"].columns)
        else:
            canonical = frames["getTicks"]
            state = "ticks" if len(canonical) else "independent_zero_ticks"
    elif success["getTicks"] ^ success["readTicks"]:
        mechanism = "getTicks" if success["getTicks"] else "readTicks"
        canonical = frames[mechanism]
        state = "retrieval_mechanism_failure_resolved"
    else:
        canonical = pd.DataFrame(columns=frames["getTicks"].columns)
        state = "both_failed"
    details = {
        "getTicks_success": bool(success["getTicks"]),
        "readTicks_success": bool(success["readTicks"]),
        "getTicks_count": len(frames["getTicks"]),
        "readTicks_count": len(frames["readTicks"]),
        "getTicks_normalized_sha256": normalized_hashes["getTicks"],
        "readTicks_normalized_sha256": normalized_hashes["readTicks"],
    }
    return state, canonical, details


def bar_map(frame: pd.DataFrame) -> dict[int, dict[str, float]]:
    result = {}
    for row in frame.itertuples(index=False):
        result[int(row.bar_time_ms) * 1_000_000] = {
            field: float(getattr(row, field)) for field in FIELDS
        }
    return result


def classify_minutes(
    run: Path,
    requests: pd.DataFrame,
    audit: pd.DataFrame,
    ticks: pd.DataFrame,
    bars: pd.DataFrame,
    native: dict[str, dict[str, np.ndarray]],
) -> tuple[pd.DataFrame, dict[str, Any], dict[tuple[str, int], dict[str, float]]]:
    previous = pd.read_csv(PARENT / "gap_minute_classification.csv.gz")
    targets = previous[previous["classification"].eq("retrieval_error")][
        ["instrument", "minute_open_utc"]
    ].drop_duplicates()
    request_by_key = {
        (row.pair, int(row.from_ms) * 1_000_000): row.request_id
        for row in requests[requests["kind"].eq("retrieval_error_hour")].itertuples(index=False)
    }
    states: dict[str, tuple[str, pd.DataFrame, dict[str, Any]]] = {}
    tick_bars: dict[str, dict[int, dict[str, float]]] = {}
    official_bars: dict[str, dict[int, dict[str, float]]] = {}
    dispositions = []
    for request in requests[requests["kind"].eq("retrieval_error_hour")].itertuples(index=False):
        state, canonical, details = request_tick_state(request.request_id, audit, ticks)
        states[request.request_id] = (state, canonical, details)
        tick_bars[request.request_id] = aggregate_ticks(canonical)
        official = bars[bars["request_id"] == request.request_id]
        official_bars[request.request_id] = bar_map(official)
        if state in {"ticks", "retrieval_mechanism_failure_resolved"} and len(canonical):
            disposition = "recovered_ticks"
        elif state == "independent_zero_ticks" and official.empty:
            disposition = "verified_zero_tick"
        elif not official.empty and state in {"both_failed", "independent_zero_ticks"}:
            disposition = "jforex_bar_only_evidence"
        elif state == "both_failed":
            disposition = "continued_api_failure"
        else:
            disposition = "inconsistent_provider_evidence"
        dispositions.append({
            "request_id": request.request_id,
            "pair": request.pair,
            "hour_utc": pd.Timestamp(int(request.from_ms), unit="ms", tz="UTC").isoformat(),
            "state": state,
            "disposition": disposition,
            "jforex_bar_count": len(official),
            **details,
        })
    disposition_frame = pd.DataFrame(dispositions)
    disposition_frame.to_csv(run / "retrieval_error_disposition.csv", index=False)

    rows = []
    repairs: dict[tuple[str, int], dict[str, float]] = {}
    for target in targets.itertuples(index=False):
        pair = target.instrument
        minute_ns = pd.Timestamp(target.minute_open_utc).value
        hour_ns = (minute_ns // HOUR_NS) * HOUR_NS
        request_id = request_by_key[(pair, hour_ns)]
        state, _canonical, details = states[request_id]
        a = row_at(native[pair], minute_ns)
        b = tick_bars[request_id].get(minute_ns)
        c = official_bars[request_id].get(minute_ns)
        tolerance = POINTS[pair]
        if bars_agree(b, c, tolerance) and (a is None or not bars_agree(a, b, tolerance)):
            classification = (
                "retrieval_error_resolved_same_provider" if a is None
                else "native_endpoint_representation_defect"
            )
            repairs[(pair, minute_ns)] = c
        elif bars_agree(a, c, tolerance) and not bars_agree(b, c, tolerance):
            classification = "tick_reconstruction_or_tick_feed_inconsistency"
        elif bars_agree(a, b, tolerance) and not bars_agree(c, b, tolerance):
            classification = "jforex_bar_api_inconsistency"
        elif (
            a is None and b is None and c is None
            and state == "independent_zero_ticks"
        ):
            classification = "verified_no_quote_minute"
        else:
            classification = "inconsistent_provider_history"
        rows.append({
            "pair": pair,
            "minute_open_utc": pd.Timestamp(minute_ns, tz="UTC").isoformat(),
            "request_id": request_id,
            "tick_state": state,
            "classification": classification,
            "a_present": a is not None,
            "b_present": b is not None,
            "c_present": c is not None,
            "a_ohlc": json.dumps(a, sort_keys=True) if a else "",
            "b_ohlc": json.dumps(b, sort_keys=True) if b else "",
            "c_ohlc": json.dumps(c, sort_keys=True) if c else "",
            **details,
        })
    minute_frame = pd.DataFrame(rows)
    minute_frame.to_csv(
        run / "three_way_arbitration.csv.gz", index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    return minute_frame, {
        "hours": disposition_frame,
        "states": states,
        "tick_bars": tick_bars,
        "official_bars": official_bars,
    }, repairs


def prior_http_ticks_for_mismatch() -> pd.DataFrame:
    hour = pd.Timestamp(MISMATCH_NS, tz="UTC").floor("h")
    database = sqlite3.connect(PARENT / "dukascopy_tick_responses.sqlite")
    records = list(database.execute(
        "SELECT acquisition,status,body FROM responses WHERE pair=? AND hour_utc=? "
        "ORDER BY acquisition",
        (MISMATCH_PAIR, hour.isoformat()),
    ))
    database.close()
    pieces = []
    for acquisition, status, body in records:
        if status != 200:
            continue
        times, bids = parent_code.decode_bi5(body, MISMATCH_PAIR, hour.value)
        pieces.append(pd.DataFrame({
            "acquisition": acquisition,
            "time_ms": times // 1_000_000,
            "bid": bids,
        }))
    if not pieces:
        return pd.DataFrame(columns=["acquisition", "time_ms", "bid"])
    return pd.concat(pieces, ignore_index=True)


def mismatch_root_cause(
    run: Path,
    audit: pd.DataFrame,
    ticks: pd.DataFrame,
    bars: pd.DataFrame,
    native: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    request_id = "MISMATCH_GBPUSD_20231210T2259Z"
    state, canonical, details = request_tick_state(request_id, audit, ticks)
    jf_tick_bars = aggregate_ticks(canonical)
    jf_bars = bar_map(bars[bars["request_id"] == request_id])
    previous_tick = load_npz(PARENT / "tick_derived_m1_GBPUSD.npz")
    a = row_at(native[MISMATCH_PAIR], MISMATCH_NS)
    b = row_at(previous_tick, MISMATCH_NS)
    c = jf_bars.get(MISMATCH_NS)
    d = jf_tick_bars.get(MISMATCH_NS)
    tolerance = POINTS[MISMATCH_PAIR]
    if bars_agree(a, c, tolerance) and not bars_agree(b, c, tolerance):
        classification = "tick_reconstruction_or_tick_feed_inconsistency"
    elif bars_agree(b, c, tolerance) and not bars_agree(a, c, tolerance):
        classification = "native_endpoint_representation_defect"
    elif bars_agree(a, b, tolerance) and not bars_agree(c, b, tolerance):
        classification = "jforex_bar_api_inconsistency"
    else:
        classification = "inconsistent_provider_history"

    window_start_ms = int((MISMATCH_NS - 2 * MINUTE_NS) // 1_000_000)
    window_end_ms = int((MISMATCH_NS + 3 * MINUTE_NS) // 1_000_000)
    previous_raw = prior_http_ticks_for_mismatch()
    previous_raw = previous_raw[
        previous_raw["time_ms"].between(window_start_ms, window_end_ms, inclusive="both")
    ]
    previous_raw.to_csv(
        run / "gbpusd_previous_http_ticks.csv.gz", index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    minute_ticks = canonical[
        (canonical["time_ms"] >= MISMATCH_NS // 1_000_000)
        & (canonical["time_ms"] < (MISMATCH_NS + MINUTE_NS) // 1_000_000)
    ].sort_values(["time_ms", "bid"], kind="stable")

    def tick_record(row: pd.Series | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {"time_ms": int(row["time_ms"]), "bid": float(row["bid"])}

    high_tick = None if minute_ticks.empty else minute_ticks.loc[minute_ticks["bid"].idxmax()]
    low_tick = None if minute_ticks.empty else minute_ticks.loc[minute_ticks["bid"].idxmin()]
    first_tick = None if minute_ticks.empty else minute_ticks.iloc[0]
    last_tick = None if minute_ticks.empty else minute_ticks.iloc[-1]
    differing = []
    if a and b:
        differing = [field for field in FIELDS if abs(a[field] - b[field]) > tolerance + 1e-12]
    result = {
        "pair": MISMATCH_PAIR,
        "minute_open_utc": pd.Timestamp(MISMATCH_NS, tz="UTC").isoformat(),
        "original_tolerance": tolerance,
        "original_tolerance_preserved": True,
        "a_native_endpoint_ohlc": a,
        "b_previous_tick_derived_ohlc": b,
        "c_jforex_native_ohlc": c,
        "jforex_tick_reconstructed_ohlc": d,
        "jforex_tick_state": state,
        "jforex_retrieval": details,
        "root_cause_classification": classification,
        "differing_ohlc_fields_a_vs_b": differing,
        "difference_magnitude_a_vs_b": {
            field: abs(a[field] - b[field]) for field in FIELDS
        } if a and b else None,
        "exact_tick_producing_high": tick_record(high_tick),
        "exact_tick_producing_low": tick_record(low_tick),
        "first_tick": tick_record(first_tick),
        "last_tick": tick_record(last_tick),
        "ticks_exactly_at_minute_open": int(
            (canonical["time_ms"] == MISMATCH_NS // 1_000_000).sum()
        ),
        "ticks_exactly_at_next_minute_boundary": int(
            (canonical["time_ms"] == (MISMATCH_NS + MINUTE_NS) // 1_000_000).sum()
        ),
        "exact_duplicate_jforex_ticks": int(len(canonical) - len(canonical.drop_duplicates())),
        "out_of_order_jforex_ticks_in_serialized_response": int(
            (ticks.loc[ticks["request_id"] == request_id, "time_ms"].diff() < 0).sum()
        ),
        "bid_side_used": True,
        "price_precision": 1e-5,
        "rounding_before_aggregation": False,
        "previous_http_tick_rows_in_window": len(previous_raw),
    }
    archive.write_json(run / "gbpusd_single_mismatch_root_cause.json", result)
    return result


def add_minute_closures(
    base: dict[str, list[tuple[int, int]]], frame: pd.DataFrame
) -> dict[str, list[tuple[int, int]]]:
    result = {pair: list(base[pair]) for pair in INSTRUMENTS}
    verified = frame[frame["classification"].eq("verified_no_quote_minute")]
    for row in verified.itertuples(index=False):
        minute = pd.Timestamp(row.minute_open_utc).value
        result[row.pair].append((minute + MINUTE_NS, minute + 2 * MINUTE_NS))
    return {pair: parent_code.merge_intervals(values) for pair, values in result.items()}


def apply_repairs(
    native: dict[str, dict[str, np.ndarray]],
    repairs: dict[tuple[str, int], dict[str, float]],
) -> tuple[dict[str, dict[str, np.ndarray]], pd.DataFrame]:
    result = {}
    provenance = []
    for pair in INSTRUMENTS:
        source = {key: value.copy() for key, value in native[pair].items()}
        pair_repairs = sorted(
            (minute, bar) for (economic, minute), bar in repairs.items() if economic == pair
        )
        for minute, bar in pair_repairs:
            position = int(np.searchsorted(source["open_utc_ns"], minute))
            exists = position < len(source["open_utc_ns"]) and int(
                source["open_utc_ns"][position]
            ) == minute
            if exists:
                for field in FIELDS:
                    source[field][position] = bar[field]
                action = "replace"
            else:
                source["open_utc_ns"] = np.insert(source["open_utc_ns"], position, minute)
                for field in FIELDS:
                    source[field] = np.insert(source[field], position, bar[field])
                action = "insert"
            provenance.append({
                "pair": pair,
                "minute_open_utc": pd.Timestamp(minute, tz="UTC").isoformat(),
                "action": action,
                "support": "JForex getTicks/readTicks plus getBars BID NO_FILTER",
                **bar,
            })
        result[pair] = source
    return result, pd.DataFrame(provenance)


def new_overlap_equivalence(
    run: Path,
    requests: pd.DataFrame,
    state_bundle: dict[str, Any],
    native: dict[str, dict[str, np.ndarray]],
    mismatch: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    seen = set()
    for request in requests[requests["kind"].eq("retrieval_error_hour")].itertuples(index=False):
        state, _canonical, _details = state_bundle["states"][request.request_id]
        if state not in {"ticks", "retrieval_mechanism_failure_resolved"}:
            continue
        for minute, tick_bar in state_bundle["tick_bars"][request.request_id].items():
            key = (request.pair, minute)
            if key in seen:
                continue
            seen.add(key)
            native_bar = row_at(native[request.pair], minute)
            if native_bar is None:
                continue
            jf_bar = state_bundle["official_bars"][request.request_id].get(minute)
            tolerance = POINTS[request.pair]
            resolved = bars_agree(native_bar, tick_bar, tolerance) or (
                jf_bar is not None
                and (bars_agree(native_bar, jf_bar, tolerance)
                     or bars_agree(tick_bar, jf_bar, tolerance))
            )
            maximum = max(abs(native_bar[field] - tick_bar[field]) for field in FIELDS)
            rows.append({
                "pair": request.pair,
                "minute_open_utc": pd.Timestamp(minute, tz="UTC").isoformat(),
                "max_abs_native_tick_difference": maximum,
                "within_tolerance": maximum <= tolerance + 1e-12,
                "three_way_resolved": resolved,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(run / "new_native_tick_equivalence.csv", index=False)
    root_resolved = mismatch["root_cause_classification"] in {
        "tick_reconstruction_or_tick_feed_inconsistency",
        "native_endpoint_representation_defect",
    }
    unresolved_new = int((~frame["three_way_resolved"]).sum()) if len(frame) else 0
    unresolved = (0 if root_resolved else 1) + unresolved_new
    result = {
        "previous_overlap_bars": 57_365,
        "newly_evaluable_overlap_bars": len(frame),
        "total_compared": 57_365 + len(frame),
        "previous_mismatch_bars": 1,
        "new_raw_mismatch_bars": int((~frame["within_tolerance"]).sum()) if len(frame) else 0,
        "root_caused_previous_mismatch": bool(root_resolved),
        "unresolved_mismatches": unresolved,
        "mismatch_rate": unresolved / (57_365 + len(frame)),
        "maximum_new_mismatch": (
            float(frame["max_abs_native_tick_difference"].max()) if len(frame) else None
        ),
        "original_tolerance_preserved": True,
        "gate": "PASS" if unresolved == 0 else "FAIL",
    }
    archive.write_json(run / "native_tick_jforex_equivalence.json", result)
    return result


def reconcile(run: Path, manifest: dict[str, Any], blocks: dict[str, np.ndarray],
              times: np.ndarray, old_unknown: np.ndarray) -> dict[str, Any]:
    native = load_native()
    _old_matrix, verified_old_unknown, base_closures = parent_state(native, times)
    if not np.array_equal(old_unknown, verified_old_unknown):
        raise RuntimeError("old unresolved mask changed during acquisition")
    requests = pd.read_csv(run / "jforex_request_manifest.tsv", sep="\t")
    audit, ticks, bars = read_jforex(run)
    minute_frame, state_bundle, repairs = classify_minutes(
        run, requests, audit, ticks, bars, native
    )
    mismatch = mismatch_root_cause(run, audit, ticks, bars, native)
    if mismatch["root_cause_classification"] == "native_endpoint_representation_defect":
        c = mismatch["c_jforex_native_ohlc"]
        if c is not None:
            repairs[(MISMATCH_PAIR, MISMATCH_NS)] = c
    equivalence = new_overlap_equivalence(
        run, requests, state_bundle, native, mismatch
    )

    closures = add_minute_closures(base_closures, minute_frame)
    no_quote_matrix, no_quote_unknown, _ = matrix_code.build_matrix(times, native, closures)
    repaired, provenance = apply_repairs(native, repairs)
    matrix, unknown, legitimate = matrix_code.build_matrix(times, repaired, closures)
    old_rows = old_unknown.any(axis=1)
    resolved_no_quote = old_rows & ~no_quote_unknown.any(axis=1)
    resolved_repairs = old_rows & no_quote_unknown.any(axis=1) & ~unknown.any(axis=1)
    remaining = old_rows & unknown.any(axis=1)
    new_unresolved = ~old_rows & unknown.any(axis=1)

    unresolved_classes = minute_frame[minute_frame["classification"].isin([
        "jforex_bar_api_inconsistency", "inconsistent_provider_history"
    ])]
    unresolved_keys = set(
        (row.pair, (pd.Timestamp(row.minute_open_utc).value // HOUR_NS) * HOUR_NS)
        for row in unresolved_classes.itertuples(index=False)
    )
    hours = state_bundle["hours"]
    recovered_get = int(((hours["getTicks_success"]) & (hours["getTicks_count"] > 0)).sum())
    recovered_read = int(((hours["readTicks_success"]) & (hours["readTicks_count"] > 0)).sum())
    verified_zero = int((hours["disposition"] == "verified_zero_tick").sum())
    recovered_bars = int((hours["jforex_bar_count"] > 0).sum())
    continued = int(hours["disposition"].isin([
        "continued_api_failure", "inconsistent_provider_evidence", "jforex_bar_only_evidence"
    ]).sum())
    affecting_remaining = len(unresolved_keys) if remaining.any() else 0
    all_rows_resolved = int(remaining.sum()) == 0
    ready_pre_validator = bool(
        equivalence["gate"] == "PASS"
        and affecting_remaining == 0
        and all_rows_resolved
        and int(new_unresolved.sum()) == 0
    )

    provenance.to_csv(
        run / "dukascopy_certified_bar_manifest.csv.gz", index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    source_paths = []
    matrix_hash = None
    per_block_hashes: dict[str, str] = {}
    if ready_pre_validator:
        for pair in INSTRUMENTS:
            path = run / f"dukascopy_certified_{CODES[pair]}.npz"
            np.savez_compressed(path, **repaired[pair])
            source_paths.append(path)
        matrix_path = run / "dukascopy_final_feature_matrix.npz"
        np.savez_compressed(
            matrix_path,
            utc_ns=times,
            features=matrix,
            feature_names=np.asarray(FEATURES),
            unknown_source_mask=unknown,
            legitimate_nan_mask=legitimate,
        )
        source_paths.append(matrix_path)
        matrix_hash = logical_hash(matrix)
        for block, block_times in blocks.items():
            indices = np.searchsorted(times, block_times)
            per_block_hashes[block] = logical_hash(matrix[indices])

    source_hash = logical_hash(*(
        repaired[pair][field]
        for pair in INSTRUMENTS
        for field in ("open_utc_ns", *FIELDS)
    ))
    finite = {
        feature: float(np.isfinite(matrix[:, index]).mean() * 100)
        for index, feature in enumerate(FEATURES)
    }
    classification_counts = minute_frame["classification"].value_counts().to_dict()
    result = {
        "run_id": run.name,
        "run_status": "pending_validator",
        "original_retrieval_error_hours": 783,
        "recovered_by_getTicks": recovered_get,
        "recovered_by_readTicks": recovered_read,
        "verified_zero_tick_hours": verified_zero,
        "recovered_native_m1_bar_hours": recovered_bars,
        "retrieval_error_hours_remaining": continued,
        "retrieval_error_hours_affecting_required_rows_remaining": affecting_remaining,
        "original_mismatch_bars": 1,
        "mismatch_pair": MISMATCH_PAIR,
        "mismatch_timestamp": pd.Timestamp(MISMATCH_NS, tz="UTC").isoformat(),
        "mismatch_differing_ohlc_fields": mismatch["differing_ohlc_fields_a_vs_b"],
        "mismatch_a_native": mismatch["a_native_endpoint_ohlc"],
        "mismatch_b_tick_derived": mismatch["b_previous_tick_derived_ohlc"],
        "mismatch_c_jforex_native": mismatch["c_jforex_native_ohlc"],
        "mismatch_root_cause_classification": mismatch["root_cause_classification"],
        "original_tolerance_preserved": True,
        "equivalence": equivalence,
        "equivalence_gate": equivalence["gate"],
        "previous_unresolved_rows": 6_931,
        "resolved_by_verified_no_quote": int(resolved_no_quote.sum()),
        "resolved_by_recovered_jforex_ticks": int(resolved_repairs.sum()),
        "resolved_by_jforex_native_m1_bars": 0,
        "resolved_by_representation_arbitration": 0,
        "proven_feature_impact_neutral": max(0, continued - affecting_remaining),
        "remaining_unresolved_rows": int(remaining.sum()),
        "new_unresolved_rows": int(new_unresolved.sum()),
        "minute_arbitration_counts": {key: int(value) for key, value in classification_counts.items()},
        "feature_finite_percentage": finite,
        "all_six_timestamp_hashes_matched": True,
        "source_dataset_sha256": source_hash,
        "per_block_feature_matrix_sha256": per_block_hashes,
        "dukascopy_final_feature_matrix_sha256": matrix_hash,
        "data_foundation_ready_pre_validator": ready_pre_validator,
        "data_foundation_ready": False,
        "family_testable_to_provenance_standard": ready_pre_validator,
        "model_training_performed": False,
        "strategy_evaluation_performed": False,
        "gemini_py_changed": False,
        "operational_model_changed": False,
        "single_next_action": (
            "Run independent validation; even on PASS, await explicit B0/B1 authorization."
            if ready_pre_validator
            else "Stop USD FX PRESSURE research; move to a cleaner external information family."
        ),
    }
    archive.write_json(run / "metrics.json", result)
    archive.write_json(run / "feature_matrix_manifest.json", {
        "created": ready_pre_validator,
        "provider": "Dukascopy",
        "representations": ["native historical M1", "JForex M1", "JForex tick-derived M1"],
        "source_dataset_sha256": source_hash,
        "dukascopy_final_feature_matrix_sha256": matrix_hash,
        "feature_names": list(FEATURES),
        "dtype": str(matrix.dtype),
        "shape": list(matrix.shape),
        "finite_percentage": finite,
        "per_block_feature_matrix_sha256": per_block_hashes,
        "timestamp_hashes": EXPECTED_TIMESTAMP_HASHES,
    })

    manifest["data"].update({
        "symbols": list(INSTRUMENTS),
        "data_sources": [
            "Dukascopy native M1 BID", "Dukascopy JForex IHistory.getTicks",
            "Dukascopy JForex IHistory.readTicks", "Dukascopy JForex IHistory.getBars",
        ],
        "source_files": [{
            "path": path.name,
            "sha256": archive.file_sha256(path),
            "retention_status": "stored_in_finalized_run_git_lfs_archival_pending",
        } for path in [
            run / "jforex_ticks.tsv.gz", run / "jforex_bars.tsv.gz",
            run / "jforex_request_audit.tsv", *source_paths,
        ]],
        "timezone": "UTC millisecond ticks; UTC M1 bar-open; completed bars only",
        "validation_rows": len(times),
        "train_rows": 0,
        "test_rows": 0,
        "purge_details": "not applicable: no labels or fitting",
        "embargo_details": "not applicable: no model or strategy evaluation",
        "raw_snapshot_retained": True,
        "reproducibility_claim": "Raw JForex tick/bar responses and request audit retained",
        "mt5_fetch": {"used": False, "not_applicable_reason": "Dukascopy-only"},
    })
    manifest["registry"].update({
        "parent_or_incumbent": PARENT.name,
        "selected_configuration": "not applicable; frozen data reconciliation",
        "trades_per_day": "not_applicable_data_only",
        "realized_win_rate": "not_applicable_data_only",
        "pf": "not_applicable_data_only",
        "mean_r": "not_applicable_data_only",
        "pnl": "not_applicable_data_only",
        "max_dd": "not_applicable_data_only",
        "validator_result": "pending",
    })
    manifest["operational_hashes_after"] = {
        name: archive.file_sha256(ROOT / name)
        for name in manifest["operational_hashes_before"]
    }
    manifest["protected_runs_after"] = {
        PARENT.name: tree_hash(PARENT),
        SOURCE_PARENT.name: tree_hash(SOURCE_PARENT),
    }
    if manifest["protected_runs_after"] != manifest["protected_runs_before"]:
        raise RuntimeError("a protected finalized run changed")
    result["gemini_py_changed"] = (
        manifest["operational_hashes_before"]["gemini.py"]
        != manifest["operational_hashes_after"]["gemini.py"]
    )
    result["operational_model_changed"] = (
        manifest["operational_hashes_before"]["gold_long_recent_candidate_xgb.json"]
        != manifest["operational_hashes_after"]["gold_long_recent_candidate_xgb.json"]
    )
    archive.write_json(run / "metrics.json", result)
    manifest["artifacts"] = []
    for path in sorted(run.iterdir()):
        if path.is_file() and path.name not in {"manifest.json", "environment.txt", "stdout.log"}:
            manifest["artifacts"].append({
                "kind": "data_only_evidence",
                "path": path.name,
                "sha256": archive.file_sha256(path),
                "retention_status": "stored_in_finalized_run_git_archival_pending",
            })
    archive.write_json(run / "manifest.json", manifest)
    report = [
        "# GEMINI DUKASCOPY JFOREX FINAL RECONCILIATION V1",
        "",
        "Data-only. No label, prediction, trade, return, or strategy metric was loaded.",
        "",
        f"Pre-validator foundation ready: **{'YES' if ready_pre_validator else 'NO'}**",
        "",
        "```json",
        json.dumps(result, indent=2),
        "```",
        "",
    ]
    (run / "report.md").write_text("\n".join(report), encoding="utf-8")
    return result


def main(run: Path) -> None:
    manifest, blocks, times, old_unknown = preregister(run)
    run_collector(run)
    result = reconcile(run, manifest, blocks, times, old_unknown)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: gold_gemini_dukascopy_jforex_final_reconciliation_v1.py RUN_DIR")
    main(Path(sys.argv[1]).resolve())
