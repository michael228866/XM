"""Adjudicate frozen Dukascopy native-M1 gaps with Dukascopy BID ticks only."""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import lzma
import re
import shutil
import sqlite3
import struct
import subprocess
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import gold_gemini_usd_fx_pressure_source_reconciliation_v1 as prior_code
import training_run_history as archive


ROOT = Path(__file__).resolve().parent
PREVIOUS = (
    ROOT
    / "training_runs/20260906T180930Z_gemini_usd_fx_pressure_source_reconciliation_v1"
)
EARLIER = ROOT / "training_runs/20260906T104638Z_gemini_usd_fx_pressure_foundation_v1"
TICK_BASE = "https://datafeed.dukascopy.com/datafeed"
INSTRUMENTS = ("EUR/USD", "GBP/USD", "USD/JPY")
CODES = {"EUR/USD": "EURUSD", "GBP/USD": "GBPUSD", "USD/JPY": "USDJPY"}
POINTS = {"EUR/USD": 1e-5, "GBP/USD": 1e-5, "USD/JPY": 1e-3}
FEATURES = (
    "USD_PRESSURE_1M",
    "USD_PRESSURE_5M",
    "USD_PRESSURE_15M",
    "USD_PRESSURE_60M",
    "USD_DISPERSION_15M",
)
EXPECTED_TIMESTAMP_HASHES = {
    "fold1_train": "6a7405a13f30c54e6863cf6e80ea1b2e9ee93a90902e5edd37fe4305d471aab6",
    "fold1_score": "47086d0837f09e86d8874874de302e96e7573efb29236f24720f4a14e0e33c94",
    "fold2_train": "691d8d2b01c3829f010ab559b1daa07c1899f8425a5b8b23b3007bf58467df44",
    "fold2_score": "1fc6500ff642dbf7172328f71f13f38712d11f6c0ce873c38524745710c608b8",
    "fold3_train": "3cf7f68ba2cf994d23d4db97c2e7ec10b2ed772245e3d4b776272ed20ee06b1b",
    "fold3_score": "1451d2069b1d087dc8b4bb7b3ed5840a7b2c03d4adfb548eadc611ed07803449",
}
PREVIOUS_HASHES = {
    "FINALIZED.json": "8e3ae9ae084ea05075dd5d4704190b7e4574e73e352f72b40bc50dcd6ca2a0e1",
    "source_gap_audit.csv": "406876878b0d6019dbe9dd376db0668f5edc41f623896b14415cbb95902af3d0",
    "metrics.json": "e5ff7a4a4454ccac0c22126d05fe8da25c96cfcb26cc6cde9ffbe6f6b50bce47",
    "usd_fx_reconciled_feature_matrix.npz": "4768cb9f34cd21afa5666c3d22f8e526bcb508c435ef9dd3759019b7b22ada89",
    "fx_source_DUKASCOPY_EURUSD.npz": "c07d8ae4eef3712989a398fe66d80bf5f8f12fdc7fa284ffa202c6743ff47191",
    "fx_source_DUKASCOPY_GBPUSD.npz": "2cf18b635ebb316e264aa3bb301e4475b1f67987ce30f687e805bf877177ba67",
    "fx_source_DUKASCOPY_USDJPY.npz": "64c87d7bcebe91ca016f30bf376d24a22560678972f3c13c046e60bfa8d2c3ba",
    "exact_timestamps.npz": "2f04a9e01d32ae796565cd453c69c0f0de8141e50cb4e62dd72838898e18efa9",
}
MINUTE_NS = 60_000_000_000
HOUR_NS = 60 * MINUTE_NS
HORIZONS = (1, 5, 15, 60)
ORIENTATION = (-1.0, -1.0, 1.0)
TICK_RECORD = struct.Struct(">IIIff")
MIN_EQUIVALENCE_BARS = 1_000
HTTP_WORKERS = 24
HTTP_LOCAL = threading.local()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def digest(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    result = hashlib.sha256()
    result.update(str(values.dtype).encode())
    result.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    result.update(values.tobytes())
    return result.hexdigest()


def logical_hash(*arrays: np.ndarray) -> str:
    result = hashlib.sha256()
    for values in arrays:
        values = np.ascontiguousarray(values)
        result.update(str(values.dtype).encode())
        result.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        result.update(values.tobytes())
    return result.hexdigest()


def inventory(path: Path) -> dict[str, str]:
    return {
        item.relative_to(path).as_posix(): archive.file_sha256(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def tick_url(pair: str, hour: pd.Timestamp) -> str:
    return (
        f"{TICK_BASE}/{CODES[pair]}/{hour.year:04d}/{hour.month - 1:02d}/"
        f"{hour.day:02d}/{hour.hour:02d}h_ticks.bi5"
    )


def decode_bi5(body: bytes, pair: str, hour_ns: int) -> tuple[np.ndarray, np.ndarray]:
    if not body:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    raw = lzma.decompress(body)
    if len(raw) % TICK_RECORD.size:
        raise ValueError(f"BI5 record alignment: {len(raw)}")
    count = len(raw) // TICK_RECORD.size
    times = np.empty(count, dtype=np.int64)
    bids = np.empty(count, dtype=np.float64)
    scale = round(1.0 / POINTS[pair])
    for index, (offset_ms, _ask, bid, _ask_volume, _bid_volume) in enumerate(
        TICK_RECORD.iter_unpack(raw)
    ):
        if offset_ms >= 3_600_000:
            raise ValueError(f"tick offset outside hour: {offset_ms}")
        times[index] = hour_ns + int(offset_ms) * 1_000_000
        bids[index] = bid / scale
    if len(times) and np.any(np.diff(times) < 0):
        raise ValueError("non-monotonic tick order")
    if len(times):
        key = np.rec.fromarrays([times, bids], names="time,bid")
        _, keep = np.unique(key, return_index=True)
        keep.sort()
        times, bids = times[keep], bids[keep]
    return times, bids


def aggregate_m1(times: np.ndarray, bids: np.ndarray) -> dict[str, np.ndarray]:
    if not len(times):
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float64)
        return {
            "open_utc_ns": empty_i,
            "open": empty_f,
            "high": empty_f,
            "low": empty_f,
            "close": empty_f,
            "tick_count": empty_i.copy(),
        }
    minute = (times // MINUTE_NS) * MINUTE_NS
    starts = np.r_[0, np.flatnonzero(np.diff(minute)) + 1]
    ends = np.r_[starts[1:], len(minute)]
    return {
        "open_utc_ns": minute[starts],
        "open": bids[starts],
        "high": np.maximum.reduceat(bids, starts),
        "low": np.minimum.reduceat(bids, starts),
        "close": bids[ends - 1],
        "tick_count": ends - starts,
    }


def reset_http_connection() -> None:
    connection = getattr(HTTP_LOCAL, "connection", None)
    if connection is not None:
        connection.close()
    HTTP_LOCAL.connection = None


def http_get(url: str, retries: int = 4) -> tuple[int, bytes, int, str | None]:
    last_status, last_body, last_error = 0, b"", None
    path = urllib.parse.urlsplit(url).path
    for attempt in range(1, retries + 1):
        try:
            connection = getattr(HTTP_LOCAL, "connection", None)
            if connection is None:
                connection = http.client.HTTPSConnection(
                    "datafeed.dukascopy.com", timeout=35
                )
                HTTP_LOCAL.connection = connection
            connection.request(
                "GET",
                path,
                headers={
                    "User-Agent": "XM-GOLD-Dukascopy-tick-gap-audit/1.0",
                    "Accept-Encoding": "identity",
                    "Connection": "keep-alive",
                },
            )
            response = connection.getresponse()
            last_status, last_body = int(response.status), response.read()
            if response.getheader("Connection", "").lower() == "close":
                reset_http_connection()
            if last_status == 404:
                return last_status, last_body, attempt, "HTTP 404"
            if last_status == 200:
                return last_status, last_body, attempt, None
            last_error = f"HTTP {last_status}"
            if last_status >= 500:
                reset_http_connection()
        except (OSError, http.client.HTTPException) as exc:
            last_status, last_body = 0, b""
            last_error = f"{type(exc).__name__}: {exc}"
            reset_http_connection()
        if attempt < retries:
            time.sleep(min(8, 2 ** (attempt - 1)))
    return last_status, last_body, retries, last_error


def init_db(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=DELETE")
    db.execute(
        "CREATE TABLE IF NOT EXISTS responses("
        "pair TEXT,hour_utc TEXT,acquisition INTEGER,url TEXT,status INTEGER,"
        "body BLOB,body_sha256 TEXT,fetched_at_utc TEXT,http_attempts INTEGER,"
        "error TEXT,PRIMARY KEY(pair,hour_utc,acquisition))"
    )
    return db


def acquire_ticks(run: Path, requests: pd.DataFrame) -> None:
    db_path = run / "dukascopy_tick_responses.sqlite"
    db = init_db(db_path)
    for acquisition in (1, 2):
        existing = {
            (row[0], row[1])
            for row in db.execute(
                "SELECT pair,hour_utc FROM responses WHERE acquisition=?",
                (acquisition,),
            )
        }
        jobs = [
            (row.instrument, row.hour_utc, row.url)
            for row in requests.itertuples(index=False)
            if (row.instrument, row.hour_utc) not in existing
        ]

        def fetch(job: tuple[str, str, str]) -> tuple[Any, ...]:
            pair, hour_utc, url = job
            status, body, attempts, error = http_get(url)
            return (
                pair,
                hour_utc,
                acquisition,
                url,
                status,
                sqlite3.Binary(body),
                hashlib.sha256(body).hexdigest(),
                datetime.now(timezone.utc).isoformat(),
                attempts,
                error,
            )

        completed = 0
        with ThreadPoolExecutor(max_workers=HTTP_WORKERS) as pool:
            futures = [pool.submit(fetch, job) for job in jobs]
            for future in as_completed(futures):
                db.execute("INSERT OR REPLACE INTO responses VALUES(?,?,?,?,?,?,?,?,?,?)", future.result())
                completed += 1
                if completed % 100 == 0:
                    db.commit()
                    print(
                        f"Dukascopy tick acquisition {acquisition}: "
                        f"{completed}/{len(jobs)}",
                        flush=True,
                    )
        db.commit()
    db.execute("VACUUM")
    db.close()


def request_hours(gaps: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for gap in gaps.itertuples(index=False):
        start = pd.Timestamp(gap.gap_start_utc) - pd.Timedelta(minutes=65)
        end = pd.Timestamp(gap.gap_end_utc) + pd.Timedelta(minutes=5)
        for hour in pd.date_range(
            start.floor("h"),
            (end - pd.Timedelta(nanoseconds=1)).floor("h"),
            freq="h",
        ):
            rows.append(
                {
                    "instrument": gap.instrument,
                    "hour_utc": hour.isoformat(),
                    "url": tick_url(gap.instrument, hour),
                }
            )
    return pd.DataFrame(rows).drop_duplicates().sort_values(
        ["instrument", "hour_utc"], ignore_index=True
    )


def load_tick_states(
    run: Path, requests: pd.DataFrame
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    db = sqlite3.connect(run / "dukascopy_tick_responses.sqlite")
    states: dict[tuple[str, int], dict[str, Any]] = {}
    audit_rows = []
    pair_ticks: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        pair: [] for pair in INSTRUMENTS
    }
    for request in requests.itertuples(index=False):
        hour = pd.Timestamp(request.hour_utc)
        hour_ns = hour.value
        records = list(
            db.execute(
                "SELECT acquisition,status,body,body_sha256,http_attempts,error "
                "FROM responses WHERE pair=? AND hour_utc=? ORDER BY acquisition",
                (request.instrument, request.hour_utc),
            )
        )
        state, logical = "retrieval_error", [None, None]
        decoded: list[tuple[np.ndarray, np.ndarray] | None] = [None, None]
        decode_errors: list[str | None] = [None, None]
        if len(records) == 2:
            for index, record in enumerate(records):
                if record[1] == 200:
                    try:
                        decoded[index] = decode_bi5(record[2], request.instrument, hour_ns)
                        logical[index] = logical_hash(*decoded[index])
                    except (lzma.LZMAError, ValueError) as exc:
                        decode_errors[index] = f"{type(exc).__name__}: {exc}"
            statuses = [record[1] for record in records]
            if statuses == [404, 404]:
                state = "verified_no_tick_hour"
            elif statuses == [200, 200] and not any(decode_errors):
                if logical[0] == logical[1]:
                    assert decoded[0] is not None
                    state = "ticks" if len(decoded[0][0]) else "verified_no_tick_hour"
                    if state == "ticks":
                        pair_ticks[request.instrument].append(decoded[0])
                else:
                    state = "inconsistent_provider_history"
        states[(request.instrument, hour_ns)] = {
            "state": state,
            "records": records,
            "logical_hashes": logical,
            "decode_errors": decode_errors,
        }
        audit_rows.append(
            {
                "instrument": request.instrument,
                "hour_utc": request.hour_utc,
                "url": request.url,
                "acquisition_1_status": records[0][1] if len(records) > 0 else None,
                "acquisition_2_status": records[1][1] if len(records) > 1 else None,
                "acquisition_1_body_sha256": records[0][3] if len(records) > 0 else None,
                "acquisition_2_body_sha256": records[1][3] if len(records) > 1 else None,
                "acquisition_1_tick_sha256": logical[0],
                "acquisition_2_tick_sha256": logical[1],
                "acquisition_1_ticks": len(decoded[0][0]) if decoded[0] else None,
                "acquisition_2_ticks": len(decoded[1][0]) if decoded[1] else None,
                "state": state,
                "decode_error_1": decode_errors[0],
                "decode_error_2": decode_errors[1],
            }
        )
    db.close()
    pd.DataFrame(audit_rows).to_csv(run / "tick_request_audit.csv", index=False)
    derived = {}
    for pair, pieces in pair_ticks.items():
        if pieces:
            times = np.concatenate([item[0] for item in pieces])
            bids = np.concatenate([item[1] for item in pieces])
            order = np.argsort(times, kind="stable")
            times, bids = times[order], bids[order]
            key = np.rec.fromarrays([times, bids], names="time,bid")
            _, keep = np.unique(key, return_index=True)
            keep.sort()
            times, bids = times[keep], bids[keep]
        else:
            times = np.empty(0, dtype=np.int64)
            bids = np.empty(0, dtype=np.float64)
        if len(times) and np.any(np.diff(times) < 0):
            raise RuntimeError(f"non-monotonic combined ticks: {pair}")
        derived[pair] = aggregate_m1(times, bids)
        np.savez_compressed(
            run / f"tick_derived_m1_{CODES[pair]}.npz", **derived[pair]
        )
    return states, derived


def load_native(run: Path) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for pair in INSTRUMENTS:
        name = f"fx_source_DUKASCOPY_{CODES[pair]}.npz"
        shutil.copy2(PREVIOUS / name, run / name)
        with np.load(run / name, allow_pickle=False) as source:
            result[pair] = {key: source[key].copy() for key in source.files}
    return result


def enrich_gap_edges(
    gaps: pd.DataFrame, native: dict[str, dict[str, np.ndarray]]
) -> pd.DataFrame:
    result = gaps.copy()
    edge_rows = []
    for gap in result.itertuples(index=False):
        source = native[gap.instrument]
        start = pd.Timestamp(gap.gap_start_utc).value
        position = int(np.searchsorted(source["open_utc_ns"], start))
        before, after = position - 1, position
        values: dict[str, Any] = {}
        for label, index in (("before", before), ("after", after)):
            valid = 0 <= index < len(source["open_utc_ns"])
            values[f"native_{label}_open_utc"] = (
                pd.Timestamp(source["open_utc_ns"][index], tz="UTC").isoformat()
                if valid
                else None
            )
            for field in ("open", "high", "low", "close"):
                values[f"native_{label}_{field}"] = (
                    float(source[field][index]) if valid else None
                )
        edge_rows.append(values)
    return pd.concat([result, pd.DataFrame(edge_rows)], axis=1)


def record_affected_exact_rows(
    run: Path,
    gaps: pd.DataFrame,
    times: np.ndarray,
    old_unknown_mask: np.ndarray,
) -> None:
    feature_horizons = (1, 5, 15, 60, 15)
    rows = []
    for gap_id, gap in enumerate(gaps.itertuples(index=False)):
        information_start = pd.Timestamp(gap.gap_start_utc).value + MINUTE_NS
        information_end = pd.Timestamp(gap.gap_end_utc).value + MINUTE_NS
        for feature_index, (feature, horizon) in enumerate(
            zip(FEATURES, feature_horizons)
        ):
            current = (times >= information_start) & (times < information_end)
            past = (
                (times - horizon * MINUTE_NS >= information_start)
                & (times - horizon * MINUTE_NS < information_end)
            )
            affected = np.flatnonzero((current | past) & old_unknown_mask[:, feature_index])
            for index in affected:
                rows.append(
                    {
                        "gap_id": gap_id,
                        "instrument": gap.instrument,
                        "gold_timestamp_utc": pd.Timestamp(times[index], tz="UTC").isoformat(),
                        "feature": feature,
                        "horizon_minutes": horizon,
                        "affected_anchor": (
                            "both" if current[index] and past[index]
                            else "current" if current[index]
                            else "past"
                        ),
                    }
                )
    pd.DataFrame(rows).to_csv(
        run / "gap_affected_exact_rows.csv.gz",
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )


def equivalence_test(
    run: Path,
    requests: pd.DataFrame,
    states: dict[tuple[str, int], dict[str, Any]],
    native: dict[str, dict[str, np.ndarray]],
    derived: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    rows = []
    pair_summary = {}
    for pair in INSTRUMENTS:
        usable_hours = {
            pd.Timestamp(row.hour_utc).value
            for row in requests[requests["instrument"].eq(pair)].itertuples(index=False)
            if states[(pair, pd.Timestamp(row.hour_utc).value)]["state"]
            in {"ticks", "verified_no_tick_hour"}
        }
        native_values = native[pair]
        native_hours = (native_values["open_utc_ns"] // HOUR_NS) * HOUR_NS
        expected_indices = np.flatnonzero(np.isin(native_hours, list(usable_hours)))
        tick_values = derived[pair]
        positions = np.searchsorted(
            tick_values["open_utc_ns"], native_values["open_utc_ns"][expected_indices]
        )
        found = (positions < len(tick_values["open_utc_ns"]))
        safe = np.minimum(positions, max(len(tick_values["open_utc_ns"]) - 1, 0))
        if len(tick_values["open_utc_ns"]):
            found &= (
                tick_values["open_utc_ns"][safe]
                == native_values["open_utc_ns"][expected_indices]
            )
        else:
            found[:] = False
        mismatch = 0
        magnitudes = []
        for local_index, native_index in enumerate(expected_indices):
            timestamp = int(native_values["open_utc_ns"][native_index])
            if not found[local_index]:
                rows.append(
                    {
                        "instrument": pair,
                        "minute_open_utc": pd.Timestamp(timestamp, tz="UTC").isoformat(),
                        "timestamp_match": False,
                        "open_abs_diff": None,
                        "high_abs_diff": None,
                        "low_abs_diff": None,
                        "close_abs_diff": None,
                        "mismatch": True,
                    }
                )
                mismatch += 1
                continue
            tick_index = positions[local_index]
            differences = [
                abs(
                    float(native_values[field][native_index])
                    - float(tick_values[field][tick_index])
                )
                for field in ("open", "high", "low", "close")
            ]
            bad = any(value > POINTS[pair] + 1e-12 for value in differences)
            mismatch += int(bad)
            magnitudes.extend(differences)
            rows.append(
                {
                    "instrument": pair,
                    "minute_open_utc": pd.Timestamp(timestamp, tz="UTC").isoformat(),
                    "timestamp_match": True,
                    "open_abs_diff": differences[0],
                    "high_abs_diff": differences[1],
                    "low_abs_diff": differences[2],
                    "close_abs_diff": differences[3],
                    "mismatch": bad,
                }
            )
        expected = len(expected_indices)
        compared = int(found.sum())
        pair_summary[pair] = {
            "native_overlap_expected": expected,
            "compared_bars": compared,
            "timestamp_identity_rate": compared / expected if expected else 0.0,
            "mismatch_bars": mismatch,
            "tolerance": POINTS[pair],
            "max_abs_ohlc_difference": max(magnitudes, default=None),
        }
    frame = pd.DataFrame(rows)
    frame.to_csv(
        run / "native_tick_equivalence.csv.gz",
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    expected = sum(item["native_overlap_expected"] for item in pair_summary.values())
    compared = sum(item["compared_bars"] for item in pair_summary.values())
    mismatches = sum(item["mismatch_bars"] for item in pair_summary.values())
    magnitudes = frame[
        ["open_abs_diff", "high_abs_diff", "low_abs_diff", "close_abs_diff"]
    ].to_numpy(dtype=float).ravel()
    magnitudes = magnitudes[np.isfinite(magnitudes)]
    passed = (
        compared >= MIN_EQUIVALENCE_BARS
        and compared == expected
        and mismatches == 0
    )
    return {
        "gate": "PASS" if passed else "FAIL",
        "pre_registered_tolerance": POINTS,
        "minimum_compared_bars": MIN_EQUIVALENCE_BARS,
        "native_overlap_expected": expected,
        "compared_bars": compared,
        "timestamp_identity_rate": compared / expected if expected else 0.0,
        "mismatch_bars": mismatches,
        "mismatch_rate": mismatches / expected if expected else 1.0,
        "absolute_difference_p50": float(np.quantile(magnitudes, 0.5)) if len(magnitudes) else None,
        "absolute_difference_p95": float(np.quantile(magnitudes, 0.95)) if len(magnitudes) else None,
        "absolute_difference_p99": float(np.quantile(magnitudes, 0.99)) if len(magnitudes) else None,
        "absolute_difference_max": float(magnitudes.max()) if len(magnitudes) else None,
        "pairs": pair_summary,
    }


def classify_gap_minutes(
    run: Path,
    gaps: pd.DataFrame,
    states: dict[tuple[str, int], dict[str, Any]],
    derived: dict[str, dict[str, np.ndarray]],
    equivalence_passed: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = []
    derived_sets = {
        pair: set(values["open_utc_ns"].tolist()) for pair, values in derived.items()
    }
    for gap_id, gap in enumerate(gaps.itertuples(index=False)):
        start = pd.Timestamp(gap.gap_start_utc).value
        end = pd.Timestamp(gap.gap_end_utc).value
        minutes = np.arange(start, end, MINUTE_NS, dtype=np.int64)
        classes = []
        for minute in minutes:
            state = states.get((gap.instrument, (int(minute) // HOUR_NS) * HOUR_NS), {}).get(
                "state", "retrieval_error"
            )
            if state == "retrieval_error":
                classification = "retrieval_error"
            elif state == "inconsistent_provider_history":
                classification = "inconsistent_provider_history"
            elif int(minute) in derived_sets[gap.instrument]:
                classification = (
                    "native_m1_missing_but_ticks_present"
                    if equivalence_passed
                    else "inconsistent_provider_history"
                )
            else:
                classification = "verified_no_tick_interval"
            classes.append(classification)
        frames.append(
            pd.DataFrame(
                {
                    "gap_id": gap_id,
                    "instrument": gap.instrument,
                    "minute_open_utc": pd.to_datetime(minutes, utc=True),
                    "classification": classes,
                }
            )
        )
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(
        run / "gap_minute_classification.csv.gz",
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    classes = (
        "verified_no_tick_interval",
        "native_m1_missing_but_ticks_present",
        "retrieval_error",
        "inconsistent_provider_history",
    )
    summary = {
        name: {
            "intervals": int(result.loc[result["classification"].eq(name), "gap_id"].nunique()),
            "minutes": int(result["classification"].eq(name).sum()),
        }
        for name in classes
    }
    archive.write_json(run / "gap_classification_summary.json", summary)
    return result, summary


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def closure_intervals(
    all_gaps: pd.DataFrame, minute_classes: pd.DataFrame
) -> dict[str, list[tuple[int, int]]]:
    result = {pair: [] for pair in INSTRUMENTS}
    for gap in all_gaps[~all_gaps["classification"].eq("unexplained_source_gap")].itertuples(index=False):
        start = pd.Timestamp(gap.gap_start_utc).value
        end = pd.Timestamp(gap.gap_end_utc).value
        result[gap.instrument].append((start + MINUTE_NS, end + MINUTE_NS))
    verified = minute_classes[
        minute_classes["classification"].eq("verified_no_tick_interval")
    ].copy()
    verified["minute_ns"] = pd.to_datetime(verified["minute_open_utc"], utc=True).astype("int64")
    for pair in INSTRUMENTS:
        values = np.sort(verified.loc[verified["instrument"].eq(pair), "minute_ns"].to_numpy(np.int64))
        if not len(values):
            continue
        cuts = np.flatnonzero(np.diff(values) > MINUTE_NS) + 1
        for group in np.split(values, cuts):
            result[pair].append((int(group[0] + MINUTE_NS), int(group[-1] + 2 * MINUTE_NS)))
        result[pair] = merge_intervals(result[pair])
    return result


def add_repairs(
    native: dict[str, dict[str, np.ndarray]],
    derived: dict[str, dict[str, np.ndarray]],
    minute_classes: pd.DataFrame,
    allowed: bool,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, int]]:
    result, counts = {}, {}
    for pair in INSTRUMENTS:
        base = {key: values.copy() for key, values in native[pair].items()}
        repair_times = set()
        if allowed:
            repair_times = set(
                pd.to_datetime(
                    minute_classes.loc[
                        minute_classes["instrument"].eq(pair)
                        & minute_classes["classification"].eq(
                            "native_m1_missing_but_ticks_present"
                        ),
                        "minute_open_utc",
                    ],
                    utc=True,
                ).astype("int64")
            )
        tick = derived[pair]
        take = np.asarray(
            [value in repair_times for value in tick["open_utc_ns"]], dtype=bool
        )
        counts[pair] = int(take.sum())
        arrays = {}
        for field in ("open_utc_ns", "open", "high", "low", "close"):
            arrays[field] = np.concatenate([base[field], tick[field][take]])
        order = np.argsort(arrays["open_utc_ns"], kind="stable")
        arrays = {key: values[order] for key, values in arrays.items()}
        if len(np.unique(arrays["open_utc_ns"])) != len(arrays["open_utc_ns"]):
            raise RuntimeError(f"duplicate repaired M1 timestamp: {pair}")
        result[pair] = arrays
    return result, counts


def exact_times(run: Path) -> tuple[dict[str, np.ndarray], np.ndarray, bool]:
    shutil.copy2(PREVIOUS / "exact_timestamps.npz", run / "exact_timestamps.npz")
    blocks, pieces, valid = {}, [], True
    with np.load(run / "exact_timestamps.npz", allow_pickle=False) as source:
        for block, expected in EXPECTED_TIMESTAMP_HASHES.items():
            broker = source[f"{block}_broker_ns"].astype(np.int64)
            utc = source[f"{block}_utc_ns"].astype(np.int64)
            valid &= digest(broker) == expected
            blocks[block] = utc
            pieces.append(utc)
    return blocks, np.unique(np.concatenate(pieces)), valid


def preregister(run: Path) -> dict[str, Any]:
    manifest = archive.read_json(run / "manifest.json")
    head, remote = git("rev-parse", "HEAD"), git("rev-parse", "origin/main")
    if manifest["git_dirty"] or head != remote or head != manifest["git_commit"]:
        raise RuntimeError("clean pushed pre-run Git state required")
    if archive.file_sha256(Path(__file__)) != manifest["training_script_sha256"]:
        raise RuntimeError("executed script differs from immutable snapshot")
    for source in (PREVIOUS, EARLIER):
        errors = archive.validate_run(source)
        if errors:
            raise RuntimeError(f"input run does not validate: {source.name}: {errors}")
    for name, expected in PREVIOUS_HASHES.items():
        if archive.file_sha256(PREVIOUS / name) != expected:
            raise RuntimeError(f"previous input hash mismatch: {name}")
    manifest["pre_run_git"] = {
        "pre_run_git_commit": head,
        "pre_run_git_dirty": False,
        "head_sha": head,
        "origin_main_sha": remote,
        "head_equals_origin_main": True,
    }
    manifest["protected_runs_before"] = {
        source.name: inventory(source) for source in (EARLIER, PREVIOUS)
    }
    manifest["operational_hashes_before"] = {
        name: archive.file_sha256(ROOT / name)
        for name in ("gemini.py", "gold_long_recent_candidate_xgb.json")
    }
    manifest["foundation_specification"] = {
        "economic_data_provider": "Dukascopy",
        "representation_sources": "native_m1 + tick_reconstructed_m1",
        "instruments": list(INSTRUMENTS),
        "features": list(FEATURES),
        "horizons_minutes": list(HORIZONS),
        "orientation": list(ORIENTATION),
        "max_staleness_minutes": 5,
        "native_convention": "Dukascopy native M1 BID OHLC",
        "tick_convention": "Dukascopy BID ticks; UTC exact instant",
        "bar_availability": "M1 open plus 1 minute",
        "tick_archive_url_template": (
            "https://datafeed.dukascopy.com/datafeed/{PAIR}/"
            "{YYYY}/{zero_based_MM}/{DD}/{HH}h_ticks.bi5"
        ),
        "tick_window": "gap_start-65m through gap_end+5m",
        "equivalence_tolerance": POINTS,
        "equivalence_minimum_bars": MIN_EQUIVALENCE_BARS,
        "equivalence_mismatch_allowed": 0,
        "interpolation": False,
        "provider_splicing": False,
    }
    manifest["model"].update(
        {
            "trained": False,
            "model_type": "not_applicable_data_only",
            "features": list(FEATURES),
            "feature_count": len(FEATURES),
            "not_applicable_reason": "Dukascopy tick-gap data adjudication only.",
        }
    )
    manifest["search"].update(
        {
            "performed": False,
            "predefined_search_space": {},
            "candidate_results_file": "not_applicable_data_only",
            "not_applicable_reason": "One frozen source, gap universe, aggregation, and tolerance.",
        }
    )
    manifest["promotion"].update(
        {
            "requested": False,
            "gate_result": "not_applicable_data_only",
            "replacement_authorized": False,
            "operational_artifact_changed": False,
        }
    )
    archive.write_json(run / "manifest.json", manifest)
    shutil.copy2(
        ROOT / "gold_gemini_dukascopy_fx_tick_gap_adjudication_v1_validator.py",
        run / "validator_script.py",
    )
    shutil.copy2(
        ROOT / "gold_gemini_usd_fx_pressure_source_reconciliation_v1.py",
        run / "source_reconciliation_dependency.py",
    )
    return manifest


def main(run: Path) -> None:
    manifest = preregister(run)
    blocks, times, hashes_match = exact_times(run)
    all_gaps = pd.read_csv(PREVIOUS / "source_gap_audit.csv")
    gaps = all_gaps[
        all_gaps["classification"].eq("unexplained_source_gap")
    ].copy().reset_index(drop=True)
    previous_metrics = archive.read_json(PREVIOUS / "metrics.json")
    if len(gaps) != 554 or previous_metrics["unresolved_source_rows_final"] != 8476:
        raise RuntimeError("frozen prior gap universe does not match 554 / 8476")
    native = load_native(run)
    gaps = enrich_gap_edges(gaps, native)
    gaps.to_csv(run / "gap_universe.csv", index=False)
    requests = request_hours(gaps)
    requests.to_csv(run / "tick_request_manifest.csv", index=False)
    archive.write_json(
        run / "gap_universe_manifest.json",
        {
            "source_run": PREVIOUS.name,
            "source_gap_audit_sha256": PREVIOUS_HASHES["source_gap_audit.csv"],
            "gap_intervals": len(gaps),
            "gap_minutes": int(gaps["missing_minutes"].sum()),
            "previous_unresolved_exact_rows": 8476,
            "requested_unique_pair_hours": len(requests),
            "requested_acquisitions_per_hour": 2,
        },
    )
    with np.load(PREVIOUS / "usd_fx_reconciled_feature_matrix.npz", allow_pickle=False) as previous:
        old_unknown_mask = previous["unknown_source_mask"].astype(bool)
        previous_times = previous["utc_ns"].astype(np.int64)
    if not np.array_equal(previous_times, times) or int(old_unknown_mask.any(axis=1).sum()) != 8476:
        raise RuntimeError("previous exact unresolved-row identity mismatch")
    np.savez_compressed(
        run / "affected_exact_rows.npz",
        utc_ns=times[old_unknown_mask.any(axis=1)],
        feature_unknown_mask=old_unknown_mask[old_unknown_mask.any(axis=1)],
        feature_names=np.asarray(FEATURES),
    )
    record_affected_exact_rows(run, gaps, times, old_unknown_mask)
    acquire_ticks(run, requests)
    states, derived = load_tick_states(run, requests)
    equivalence = equivalence_test(run, requests, states, native, derived)
    archive.write_json(run / "native_tick_equivalence.json", equivalence)
    minute_classes, class_summary = classify_gap_minutes(
        run, gaps, states, derived, equivalence["gate"] == "PASS"
    )
    closures = closure_intervals(all_gaps, minute_classes)
    a_matrix, a_unknown, _a_legitimate = prior_code.build_matrix(times, native, closures)
    repaired, repair_counts = add_repairs(
        native,
        derived,
        minute_classes,
        equivalence["gate"] == "PASS",
    )
    matrix, unknown, legitimate = prior_code.build_matrix(times, repaired, closures)
    old_unknown = old_unknown_mask.any(axis=1)
    a_resolved = old_unknown & ~a_unknown.any(axis=1)
    b_resolved = old_unknown & a_unknown.any(axis=1) & ~unknown.any(axis=1)
    remaining = old_unknown & unknown.any(axis=1)
    new_unknown = ~old_unknown & unknown.any(axis=1)
    retrieval_pass = all(
        value["state"] in {"ticks", "verified_no_tick_hour"}
        for value in states.values()
    )
    all_classified = not minute_classes["classification"].isin(
        ["retrieval_error", "inconsistent_provider_history"]
    ).any()
    ready = bool(
        hashes_match
        and retrieval_pass
        and equivalence["gate"] == "PASS"
        and all_classified
        and not remaining.any()
        and not new_unknown.any()
    )
    source_files = [run / "dukascopy_tick_responses.sqlite"]
    for pair in INSTRUMENTS:
        repaired_path = run / f"dukascopy_tick_reconciled_{CODES[pair]}.npz"
        np.savez_compressed(repaired_path, **repaired[pair])
        source_files.extend(
            [run / f"fx_source_DUKASCOPY_{CODES[pair]}.npz", repaired_path]
        )
    matrix_hash = None
    block_hashes = {}
    if ready:
        matrix_path = run / "dukascopy_tick_reconciled_feature_matrix.npz"
        np.savez_compressed(
            matrix_path,
            utc_ns=times,
            features=matrix,
            feature_names=np.asarray(FEATURES),
            unknown_source_mask=unknown,
            legitimate_nan_mask=legitimate,
        )
        matrix_hash = logical_hash(matrix)
        source_files.append(matrix_path)
        for block, block_times in blocks.items():
            indices = np.searchsorted(times, block_times)
            block_hashes[block] = logical_hash(matrix[indices])
    feature_coverage = {
        feature: float(np.isfinite(matrix[:, index]).mean() * 100)
        for index, feature in enumerate(FEATURES)
    }
    source_hash = logical_hash(
        *(
            repaired[pair][field]
            for pair in INSTRUMENTS
            for field in ("open_utc_ns", "open", "high", "low", "close")
        )
    )
    archive.write_json(
        run / "feature_matrix_manifest.json",
        {
            "created": ready,
            "economic_data_provider": "Dukascopy",
            "representation_sources": "native_m1 + tick_reconstructed_m1",
            "source_provider_count": 1,
            "source_dataset_sha256": source_hash,
            "matrix_sha256": matrix_hash,
            "feature_names": list(FEATURES),
            "dtype": str(matrix.dtype),
            "shape": list(matrix.shape),
            "per_block_feature_hashes": block_hashes,
        },
    )
    results = {
        "run_id": run.name,
        "run_status": "pending_validator",
        "dukascopy_tick_retrieval": "PASS" if retrieval_pass else "FAIL",
        "gap_intervals_audited": len(gaps),
        "requested_unique_pair_hours": len(requests),
        "native_overlap_bars_compared": equivalence["compared_bars"],
        "native_tick_ohlc_equivalence": equivalence["gate"],
        "mismatch_bars": equivalence["mismatch_bars"],
        "mismatch_rate": equivalence["mismatch_rate"],
        "gap_classification": class_summary,
        "previous_unresolved_rows": 8476,
        "resolved_by_verified_no_tick": int(a_resolved.sum()),
        "resolved_by_tick_reconstruction": int(b_resolved.sum()),
        "remaining_unresolved_rows": int(remaining.sum()),
        "new_unresolved_rows": int(new_unknown.sum()),
        "reconstructed_native_m1_minutes": repair_counts,
        "feature_coverage": feature_coverage,
        "all_six_timestamp_hashes_matched": hashes_match,
        "source_provider_count": 1,
        "economic_data_provider": "Dukascopy",
        "representation_sources": "native_m1 + tick_reconstructed_m1",
        "source_dataset_sha256": source_hash,
        "dukascopy_tick_reconciled_feature_matrix_sha256": matrix_hash,
        "data_foundation_ready": ready,
        "model_training_performed": False,
        "strategy_evaluation_performed": False,
        "gemini_py_changed": False,
        "operational_model_changed": False,
        "single_next_action": (
            "Await explicit authorization for a frozen B0/B1 information study."
            if ready
            else "Resolve only the preserved retrieval/equivalence defects; do not train."
        ),
    }
    archive.write_json(run / "metrics.json", results)
    manifest["data"].update(
        {
            "symbols": list(INSTRUMENTS),
            "data_sources": [
                "Dukascopy native M1 BID OHLC",
                "Dukascopy official historical BID tick archive",
            ],
            "source_files": [
                {
                    "path": path.relative_to(run).as_posix(),
                    "sha256": archive.file_sha256(path),
                    "retention_status": "stored_in_finalized_run_git_lfs_archival_pending",
                }
                for path in source_files
            ],
            "timezone": (
                "UTC tick instants and UTC M1 bar-open timestamps; completed M1 "
                "information becomes usable at bar-open plus one minute"
            ),
            "data_start_utc": "2016-06-30T19:59:00+00:00",
            "data_end_utc": "2024-12-31T18:02:00+00:00",
            "train_start_utc": "not_applicable_data_only",
            "train_end_utc": "not_applicable_data_only",
            "train_rows": 0,
            "validation_start_utc": "2016-06-30T21:00:00+00:00",
            "validation_end_utc": "2024-12-31T18:00:00+00:00",
            "validation_rows": len(times),
            "test_start_utc": "not_applicable_data_only",
            "test_end_utc": "not_applicable_data_only",
            "test_rows": 0,
            "purge_details": "not applicable: no labels or fitting",
            "embargo_details": "not applicable: no model or strategy evaluation",
            "raw_snapshot_retained": True,
            "reproducibility_claim": "Raw Dukascopy tick responses and immutable native M1 inputs retained.",
            "mt5_fetch": {
                "used": False,
                "not_applicable_reason": "Dukascopy-only adjudication; XM is not queried.",
            },
        }
    )
    manifest["operational_hashes_after"] = {
        name: archive.file_sha256(ROOT / name)
        for name in manifest["operational_hashes_before"]
    }
    manifest["registry"].update(
        {
            "parent_or_incumbent": PREVIOUS.name,
            "selected_configuration": "Dukascopy BID tick gap adjudication",
            "trades_per_day": "not_applicable_data_only",
            "realized_win_rate": "not_applicable_data_only",
            "pf": "not_applicable_data_only",
            "mean_r": "not_applicable_data_only",
            "pnl": "not_applicable_data_only",
            "max_dd": "not_applicable_data_only",
            "validator_result": "pending",
        }
    )
    manifest["artifacts"] = []
    for path in sorted(run.iterdir()):
        if path.is_file() and path.name not in {"manifest.json", "environment.txt", "stdout.log"}:
            manifest["artifacts"].append(
                {
                    "kind": "data_only_evidence",
                    "path": path.name,
                    "sha256": archive.file_sha256(path),
                    "retention_status": "stored_in_finalized_run_git_archival_pending",
                }
            )
    archive.write_json(run / "manifest.json", manifest)
    report = [
        "# GEMINI DUKASCOPY FX TICK GAP ADJUDICATION V1",
        "",
        "Data-only. No model, label, prediction, target, trade, or strategy metric was accessed.",
        "",
        f"Preliminary foundation ready: **{'YES' if ready else 'NO'}**",
        "",
        "```json",
        json.dumps(results, indent=2),
        "```",
        "",
    ]
    (run / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


def self_check() -> None:
    raw = b"".join(
        [
            TICK_RECORD.pack(1_000, 110_001, 110_000, 1.0, 1.0),
            TICK_RECORD.pack(30_000, 110_003, 110_002, 1.0, 1.0),
        ]
    )
    body = lzma.compress(raw)
    times, bids = decode_bi5(body, "EUR/USD", 0)
    bars = aggregate_m1(times, bids)
    assert bars["open"].tolist() == [1.1]
    assert bars["high"].tolist() == [1.10002]
    assert tick_url("EUR/USD", pd.Timestamp("2020-01-02T12:00:00Z")).endswith(
        "/EURUSD/2020/00/02/12h_ticks.bi5"
    )
    assert len(EXPECTED_TIMESTAMP_HASHES["fold3_train"]) == 64
    print("DUKASCOPY_TICK_GAP_ADJUDICATION_SELF_CHECK_PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, nargs="?")
    parser.add_argument("--self-check", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_check:
        self_check()
    elif arguments.run_dir is None:
        parser.error("run_dir required")
    else:
        main(arguments.run_dir.resolve())
