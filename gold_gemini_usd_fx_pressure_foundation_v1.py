"""Build the frozen, data-only USD FX pressure information foundation."""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import math
import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import training_run_history as archive

ROOT = Path(__file__).resolve().parent
TERMINAL = Path(r"D:\XM2\terminal64.exe")
TIMESTAMP_FOUNDATION = ROOT / "training_runs/20260905T171629Z_gemini_macro_event_integration_foundation_v1"
PARENT = ROOT / "training_runs/20260906T060148Z_gemini_macro_event_timing_b0_b1_v1"
INSTRUMENTS = ("EUR/USD", "GBP/USD", "USD/JPY")
FEATURES = (
    "USD_PRESSURE_1M",
    "USD_PRESSURE_5M",
    "USD_PRESSURE_15M",
    "USD_PRESSURE_60M",
    "USD_DISPERSION_15M",
)
HORIZONS = (1, 5, 15, 60)
ORIENTATION = np.array([-1.0, -1.0, 1.0], dtype=np.float64)
MAX_STALENESS_MINUTES = 5
MINUTE_NS = 60_000_000_000
EXPECTED_TIMESTAMP_HASHES = {
    "fold1_train": "6a7405a13f30c54e6863cf6e80ea1b2e9ee93a90902e5edd37fe4305d471aab6",
    "fold1_score": "47086d0837f09e86d8874874de302e96e7573efb29236f24720f4a14e0e33c94",
    "fold2_train": "691d8d2b01c3829f010ab559b1daa07c1899f8425a5b8b23b3007bf58467df44",
    "fold2_score": "1fc6500ff642dbf7172328f71f13f38712d11f6c0ce873c38524745710c608b8",
    "fold3_train": "3cf7f68ba2cf994d23d4db97c2e7ec10b2ed772245e3d4b776272ed20ee06b1b",
    "fold3_score": "1451d2069b1d087dc8b4bb7b3ed5840a7b2c03d4adfb548eadc611ed07803449",
}


def file_hash(path: Path) -> str:
    return archive.file_sha256(path)


def array_hash(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    payload = str(values.dtype).encode() + np.asarray(values.shape, dtype=np.int64).tobytes() + values.tobytes()
    return hashlib.sha256(payload).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def json_write(path: Path, value: Any) -> None:
    archive.write_json(path, value)


def csv_write(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    pd.DataFrame(rows, columns=fields).to_csv(path, index=False)


def inventory(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): file_hash(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def last_sunday(year: int, month: int) -> datetime:
    day = calendar.monthrange(year, month)[1]
    value = datetime(year, month, day, 1, tzinfo=timezone.utc)
    return value - timedelta(days=(value.weekday() + 1) % 7)


def server_offset_seconds(actual_utc: datetime) -> int:
    start, end = last_sunday(actual_utc.year, 3), last_sunday(actual_utc.year, 10)
    return 3 * 3600 if start <= actual_utc < end else 2 * 3600


def broker_epoch_to_utc_ns(epoch_seconds: np.ndarray) -> np.ndarray:
    wall = pd.to_datetime(epoch_seconds.astype(np.int64), unit="s")
    return pd.DatetimeIndex(wall).tz_localize("Europe/Helsinki", ambiguous="raise", nonexistent="raise").tz_convert("UTC").asi8


def utc_to_broker_query(value: datetime) -> datetime:
    return value + timedelta(seconds=server_offset_seconds(value))


def add_artifact(manifest: dict[str, Any], run: Path, path: Path, kind: str) -> None:
    relative = path.relative_to(run).as_posix()
    manifest.setdefault("artifacts", []).append(
        {
            "kind": kind,
            "path": relative,
            "sha256": file_hash(path),
            "retention_status": "stored_in_finalized_run_git_archival_pending",
        }
    )


def preregister(run: Path) -> dict[str, Any]:
    manifest = archive.read_json(run / "manifest.json")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/main")
    if manifest["git_dirty"] or head != remote or head != manifest["git_commit"]:
        raise RuntimeError("Clean pushed pre-run Git state required")
    if file_hash(Path(__file__)) != manifest["training_script_sha256"]:
        raise RuntimeError("Executed script differs from immutable run snapshot")
    if archive.validate_run(TIMESTAMP_FOUNDATION) or archive.validate_run(PARENT):
        raise RuntimeError("Required finalized input run does not validate")
    manifest["pre_run_git"] = {
        "pre_run_git_commit": head,
        "pre_run_git_dirty": False,
        "head_sha": head,
        "origin_main_sha": remote,
        "head_equals_origin_main": True,
    }
    manifest["protected_runs_before"] = {
        item.parent.name: inventory(item.parent)
        for item in sorted((ROOT / "training_runs").glob("*/FINALIZED.json"))
    }
    manifest["operational_hashes_before"] = {
        name: file_hash(ROOT / name)
        for name in ("gemini.py", "gold_long_recent_candidate_xgb.json")
    }
    manifest["foundation_specification"] = {
        "family": "USD FX PRESSURE",
        "economic_instruments": list(INSTRUMENTS),
        "horizons_minutes": list(HORIZONS),
        "features": list(FEATURES),
        "orientation": {"EUR/USD": -1, "GBP/USD": -1, "USD/JPY": 1},
        "aggregation": "equal arithmetic mean; sample standard deviation ddof=1 for 15m dispersion",
        "max_external_bar_staleness_minutes": MAX_STALENESS_MINUTES,
        "bar_semantics": "MT5 M1 time is server-wall bar OPEN; information time is converted UTC open plus one minute",
        "asof_semantics": "latest completed close <= anchor; both t and t-h endpoints <=5 wall-clock minutes stale",
        "normalization": "none",
        "feature_or_horizon_search": False,
        "model_training": False,
        "strategy_evaluation": False,
    }
    manifest["search"].update(
        {
            "performed": False,
            "predefined_search_space": {},
            "candidate_results_file": "not_applicable_data_only",
            "not_applicable_reason": "Fixed five-feature data foundation; no candidates or outcomes.",
        }
    )
    manifest["model"].update(
        {
            "trained": False,
            "model_type": "not_applicable_data_only",
            "features": list(FEATURES),
            "feature_count": 5,
            "not_applicable_reason": "No labels, predictions, outcomes, fitting, or strategy evaluation loaded.",
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
    json_write(run / "manifest.json", manifest)
    shutil.copyfile(ROOT / "gold_gemini_usd_fx_pressure_foundation_v1_validator.py", run / "validator_script.py")
    shutil.copyfile(TIMESTAMP_FOUNDATION / "exact_timestamps.npz", run / "exact_timestamps.npz")
    shutil.copyfile(TIMESTAMP_FOUNDATION / "timestamp_universe.json", run / "inherited_timestamp_universe.json")
    return manifest


def resolve_symbols(mt5: Any) -> tuple[dict[str, str], dict[str, Any]]:
    all_symbols = list(mt5.symbols_get() or ())
    gold_suffix = "#" if mt5.symbol_info("GOLD#") is not None else ""
    resolutions: dict[str, Any] = {}
    selected: dict[str, str] = {}
    for economic in INSTRUMENTS:
        base_ccy, quote_ccy = economic.split("/")
        candidates = []
        for item in all_symbols:
            if str(getattr(item, "currency_base", "")).upper() != base_ccy:
                continue
            if str(getattr(item, "currency_profit", "")).upper() != quote_ccy:
                continue
            candidates.append(
                {
                    "symbol": item.name,
                    "description": item.description,
                    "path": item.path,
                    "visible": bool(item.visible),
                    "trade_mode": int(item.trade_mode),
                    "currency_base": item.currency_base,
                    "currency_profit": item.currency_profit,
                    "digits": int(item.digits),
                    "point": float(item.point),
                }
            )
        active = [row for row in candidates if row["trade_mode"] != 0]
        matching_suffix = [row for row in active if gold_suffix and row["symbol"].endswith(gold_suffix)]
        exact = [row for row in active if row["symbol"].upper() == (base_ccy + quote_ccy)]
        chosen = matching_suffix if len(matching_suffix) == 1 else exact if len(exact) == 1 else active
        status = "resolved" if len(chosen) == 1 else "missing" if not chosen else "ambiguous"
        resolutions[economic] = {
            "status": status,
            "resolution_rule": "exact currency_base/profit; unique operational GOLD suffix, then unique exact base name, then unique active contract",
            "gold_suffix_preference": gold_suffix,
            "candidates": candidates,
            "selected_symbol": chosen[0]["symbol"] if status == "resolved" else None,
        }
        if status == "resolved":
            selected[economic] = chosen[0]["symbol"]
    return selected, resolutions


def month_ranges(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    current = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    ranges = []
    while current < end:
        next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
        ranges.append((max(start, current), min(end, next_month)))
        current = next_month
    return ranges


def acquire_symbol(mt5: Any, symbol: str, start: datetime, end: datetime) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    chunks, records = [], []
    mt5.symbol_select(symbol, True)
    for chunk_start, chunk_end in month_ranges(start, end):
        query_start = utc_to_broker_query(chunk_start - timedelta(minutes=5))
        query_end = utc_to_broker_query(chunk_end + timedelta(minutes=5))
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, query_start, query_end)
        error = list(mt5.last_error())
        row = {
            "requested_start_utc": chunk_start.isoformat(),
            "requested_end_utc_exclusive": chunk_end.isoformat(),
            "query_start_broker_epoch_clock": query_start.isoformat(),
            "query_end_broker_epoch_clock": query_end.isoformat(),
            "status": "ok" if rates is not None else "error",
            "returned_rows_before_filter": 0 if rates is None else int(len(rates)),
            "mt5_last_error": error,
        }
        if rates is not None and len(rates):
            raw = rates["time"].astype(np.int64)
            actual = broker_epoch_to_utc_ns(raw)
            mask = (actual >= int(chunk_start.timestamp() * 1e9)) & (actual < int(chunk_end.timestamp() * 1e9))
            if mask.any():
                chunks.append(
                    np.rec.fromarrays(
                        [raw[mask], actual[mask], rates["open"][mask], rates["high"][mask], rates["low"][mask], rates["close"][mask]],
                        names="broker_epoch,open_utc_ns,open,high,low,close",
                    )
                )
            row["returned_rows_after_filter"] = int(mask.sum())
        else:
            row["returned_rows_after_filter"] = 0
        records.append(row)
        print(f"{symbol} {chunk_start:%Y-%m}: {row['returned_rows_after_filter']:,}", flush=True)
    if chunks:
        joined = np.concatenate(chunks)
        order = np.argsort(joined["open_utc_ns"], kind="stable")
        joined = joined[order]
        unique, first = np.unique(joined["open_utc_ns"], return_index=True)
        data = {name: joined[name][first] for name in joined.dtype.names or ()}
        data["duplicate_rows_removed"] = np.asarray([len(joined) - len(unique)], dtype=np.int64)
    else:
        data = {
            "broker_epoch": np.array([], dtype=np.int64),
            "open_utc_ns": np.array([], dtype=np.int64),
            "open": np.array([], dtype=np.float64),
            "high": np.array([], dtype=np.float64),
            "low": np.array([], dtype=np.float64),
            "close": np.array([], dtype=np.float64),
            "duplicate_rows_removed": np.array([0], dtype=np.int64),
        }
    return data, records


def exact_blocks(run: Path) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], np.ndarray]:
    loaded = np.load(run / "exact_timestamps.npz", allow_pickle=False)
    arrays = {name: loaded[name] for name in loaded.files}
    rows = []
    for block, expected in EXPECTED_TIMESTAMP_HASHES.items():
        broker = arrays[block + "_broker_ns"].astype(np.int64)
        utc = arrays[block + "_utc_ns"].astype(np.int64)
        if array_hash(broker) != expected:
            raise RuntimeError(f"Frozen timestamp mismatch: {block}")
        rows.append(
            {
                "block": block,
                "rows": len(broker),
                "first_broker": str(pd.Timestamp(broker[0])),
                "last_broker": str(pd.Timestamp(broker[-1])),
                "first_utc": pd.Timestamp(utc[0], tz="UTC").isoformat(),
                "last_utc": pd.Timestamp(utc[-1], tz="UTC").isoformat(),
                "broker_timestamp_sha256": array_hash(broker),
                "utc_timestamp_sha256": array_hash(utc),
            }
        )
    union = np.unique(np.concatenate([arrays[name + "_utc_ns"] for name in EXPECTED_TIMESTAMP_HASHES])).astype(np.int64)
    csv_write(run / "exact_timestamp_blocks.csv", rows)
    return rows, arrays, union


def valid_ohlc(data: dict[str, np.ndarray]) -> np.ndarray:
    if not len(data["close"]):
        return np.array([], dtype=bool)
    values = np.column_stack([data[name] for name in ("open", "high", "low", "close")])
    return (
        np.isfinite(values).all(axis=1)
        & (values > 0).all(axis=1)
        & (data["high"] >= np.maximum(data["open"], data["close"]))
        & (data["low"] <= np.minimum(data["open"], data["close"]))
    )


def gap_rows(all_data: dict[str, dict[str, np.ndarray]], symbols: dict[str, str]) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    raw: dict[str, list[tuple[int, int, int]]] = {}
    for economic, data in all_data.items():
        times = data["open_utc_ns"]
        positions = np.flatnonzero(np.diff(times) > MINUTE_NS)
        raw[economic] = [(int(times[i] + MINUTE_NS), int(times[i + 1]), int((times[i + 1] - times[i]) // MINUTE_NS - 1)) for i in positions]
    common = []
    for start, end, missing in raw.get(INSTRUMENTS[0], []):
        if missing <= MAX_STALENESS_MINUTES:
            continue
        matches = []
        for other in INSTRUMENTS[1:]:
            matches.append(any(abs(start - a) <= 5 * MINUTE_NS and abs(end - b) <= 5 * MINUTE_NS for a, b, count in raw.get(other, []) if count > MAX_STALENESS_MINUTES))
        if all(matches):
            common.append((start, end))
    rows = []
    for economic, gaps in raw.items():
        for start, end, missing in gaps:
            classification = "minor_quote_gap"
            if missing > MAX_STALENESS_MINUTES:
                classification = "confirmed_common_broker_quote_closure" if any(abs(start-a)<=5*MINUTE_NS and abs(end-b)<=5*MINUTE_NS for a,b in common) else "unexplained_source_gap"
            rows.append(
                {
                    "instrument": economic,
                    "broker_symbol": symbols.get(economic),
                    "gap_start_utc": pd.Timestamp(start, tz="UTC").isoformat(),
                    "gap_end_utc": pd.Timestamp(end, tz="UTC").isoformat(),
                    "missing_minutes": missing,
                    "classification": classification,
                }
            )
    return rows, common


def asof_price(data: dict[str, np.ndarray], anchors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    closes = data["open_utc_ns"].astype(np.int64) + MINUTE_NS
    prices = data["close"].astype(np.float64)
    if not len(closes):
        return (
            np.full(len(anchors), np.nan, dtype=np.float64),
            np.full(len(anchors), np.iinfo(np.int64).min, dtype=np.int64),
            np.zeros(len(anchors), dtype=bool),
        )
    index = np.searchsorted(closes, anchors, side="right") - 1
    exists = index >= 0
    safe = np.maximum(index, 0)
    selected_time = np.where(exists, closes[safe], np.iinfo(np.int64).min)
    age = np.where(exists, anchors - selected_time, np.iinfo(np.int64).max)
    fresh = exists & (age >= 0) & (age <= MAX_STALENESS_MINUTES * MINUTE_NS)
    selected_price = np.where(fresh, prices[safe], np.nan)
    return selected_price, selected_time, fresh


def inside_intervals(values: np.ndarray, intervals: list[tuple[int, int]]) -> np.ndarray:
    if not intervals:
        return np.zeros(len(values), dtype=bool)
    ordered = sorted(intervals)
    merged: list[list[int]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    starts = np.asarray([item[0] for item in merged], dtype=np.int64)
    ends = np.asarray([item[1] for item in merged], dtype=np.int64)
    index = np.searchsorted(starts, values, side="right") - 1
    valid = index >= 0
    result = np.zeros(len(values), dtype=bool)
    result[valid] = values[valid] < ends[index[valid]]
    return result


def build_features(union: np.ndarray, all_data: dict[str, dict[str, np.ndarray]], common: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, dict[int, dict[str, np.ndarray]]]]:
    endpoints: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    oriented = np.full((len(union), len(INSTRUMENTS), len(HORIZONS)), np.nan, dtype=np.float64)
    unknown_component = np.zeros_like(oriented, dtype=bool)
    legitimate_component = np.zeros_like(oriented, dtype=bool)
    for instrument_index, economic in enumerate(INSTRUMENTS):
        endpoints[economic] = {}
        data = all_data.get(economic, {"open_utc_ns": np.array([], dtype=np.int64), "close": np.array([], dtype=np.float64)})
        current_price, current_time, current_fresh = asof_price(data, union)
        for horizon_index, horizon in enumerate(HORIZONS):
            past_anchor = union - horizon * MINUTE_NS
            past_price, past_time, past_fresh = asof_price(data, past_anchor)
            valid = current_fresh & past_fresh
            oriented[:, instrument_index, horizon_index] = ORIENTATION[instrument_index] * np.log(current_price / past_price)
            missing_current_closure = (~current_fresh) & inside_intervals(union, common)
            missing_past_closure = (~past_fresh) & inside_intervals(past_anchor, common)
            legitimate = (~valid) & ((current_fresh | missing_current_closure) & (past_fresh | missing_past_closure))
            legitimate_component[:, instrument_index, horizon_index] = legitimate
            unknown_component[:, instrument_index, horizon_index] = (~valid) & ~legitimate
            endpoints[economic][horizon] = {
                "current_time": current_time,
                "past_time": past_time,
                "current_fresh": current_fresh,
                "past_fresh": past_fresh,
                "valid": valid,
            }
    features = np.full((len(union), len(FEATURES)), np.nan, dtype=np.float64)
    unknown = np.zeros_like(features, dtype=bool)
    legitimate = np.zeros_like(features, dtype=bool)
    for feature_index, horizon in enumerate(HORIZONS):
        source = oriented[:, :, feature_index]
        finite = np.isfinite(source).all(axis=1)
        features[finite, feature_index] = source[finite].mean(axis=1)
        unknown[:, feature_index] = unknown_component[:, :, feature_index].any(axis=1)
        legitimate[:, feature_index] = (~finite) & ~unknown[:, feature_index]
    source = oriented[:, :, HORIZONS.index(15)]
    finite = np.isfinite(source).all(axis=1)
    features[finite, 4] = source[finite].std(axis=1, ddof=1)
    unknown[:, 4] = unknown_component[:, :, HORIZONS.index(15)].any(axis=1)
    legitimate[:, 4] = (~finite) & ~unknown[:, 4]
    return features, unknown, legitimate, endpoints


def main_run(run: Path) -> None:
    manifest = preregister(run)
    definition = {
        "family": "USD FX PRESSURE",
        "instruments": list(INSTRUMENTS),
        "feature_names_order": list(FEATURES),
        "formula": "mean([-r_EURUSD(h), -r_GBPUSD(h), +r_USDJPY(h)]); sample std at h=15",
        "horizons_minutes": list(HORIZONS),
        "max_staleness_minutes": MAX_STALENESS_MINUTES,
        "completed_bar_rule": "latest M1 close with open_utc + 1 minute <= decision/anchor",
        "normalization": "none",
        "prohibited": ["models", "labels", "outcomes", "thresholds", "alternative instruments", "alternative horizons", "forward fill"],
    }
    json_write(run / "fx_family_definition.json", definition)
    blocks, timestamp_arrays, union = exact_blocks(run)
    start = datetime.fromtimestamp((int(union.min()) - 66 * MINUTE_NS) / 1e9, tz=timezone.utc)
    end = datetime.fromtimestamp((int(union.max()) + 2 * MINUTE_NS) / 1e9, tz=timezone.utc)
    all_data: dict[str, dict[str, np.ndarray]] = {}
    acquisition: dict[str, Any] = {}
    resolution: dict[str, Any] = {}
    terminal_record: dict[str, Any] = {"path": str(TERMINAL)}
    try:
        import MetaTrader5 as mt5

        if not TERMINAL.is_file() or not mt5.initialize(path=str(TERMINAL), timeout=20_000):
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
        terminal_info, account_info = mt5.terminal_info(), mt5.account_info()
        terminal_record.update(
            {
                "initialized": True,
                "connected": bool(terminal_info.connected) if terminal_info else False,
                "build": int(terminal_info.build) if terminal_info else None,
                "maxbars": int(terminal_info.maxbars) if terminal_info else None,
                "version": list(mt5.version() or ()),
                "server": account_info.server if account_info else None,
                "login": int(account_info.login) if account_info else None,
                "timestamp_semantics": "MT5 rate time treated as XM EET/EEST server-wall bar-open epoch per repository audit; converted per date to UTC",
            }
        )
        selected, resolution = resolve_symbols(mt5)
        for economic in INSTRUMENTS:
            if economic not in selected:
                continue
            symbol = selected[economic]
            try:
                data, chunks = acquire_symbol(mt5, symbol, start, end)
                all_data[economic] = data
                target = run / f"fx_source_{re.sub('[^A-Z]', '', economic)}.npz"
                np.savez_compressed(target, **data)
                acquisition[economic] = {
                    "status": "ok",
                    "broker_symbol": symbol,
                    "requested_start_utc": start.isoformat(),
                    "requested_end_utc": end.isoformat(),
                    "chunks": chunks,
                    "archive_path": target.name,
                    "archive_sha256": file_hash(target),
                    "logical_open_utc_sha256": array_hash(data["open_utc_ns"]),
                    "logical_close_sha256": array_hash(data["close"]),
                }
            except Exception as exc:
                acquisition[economic] = {
                    "status": "error", "broker_symbol": symbol,
                    "requested_start_utc": start.isoformat(), "requested_end_utc": end.isoformat(),
                    "chunks": [], "error": repr(exc),
                }
    except Exception as exc:
        terminal_record.update({"initialized": False, "error": repr(exc)})
    finally:
        try:
            mt5.shutdown()
        except (NameError, Exception):
            pass
    json_write(run / "broker_symbol_resolution.json", {"terminal": terminal_record, "resolutions": resolution})
    json_write(run / "fx_source_provenance.json", {"terminal": terminal_record, "acquisition": acquisition})

    empty = {
        "broker_epoch": np.array([], dtype=np.int64), "open_utc_ns": np.array([], dtype=np.int64),
        "open": np.array([], dtype=np.float64), "high": np.array([], dtype=np.float64),
        "low": np.array([], dtype=np.float64), "close": np.array([], dtype=np.float64),
        "duplicate_rows_removed": np.array([0], dtype=np.int64),
    }
    for economic in INSTRUMENTS:
        all_data.setdefault(economic, empty)
    symbols = {economic: resolution.get(economic, {}).get("selected_symbol") for economic in INSTRUMENTS}
    gaps, common_closures = gap_rows(all_data, symbols)
    csv_write(run / "fx_gap_audit.csv", gaps, ["instrument", "broker_symbol", "gap_start_utc", "gap_end_utc", "missing_minutes", "classification"])
    features, unknown, legitimate, endpoints = build_features(union, all_data, common_closures)
    coverage_rows, alignment_rows = [], []
    source_ready = len(resolution) == 3
    for economic in INSTRUMENTS:
        data = all_data[economic]
        times = data["open_utc_ns"]
        ohlc = valid_ohlc(data)
        chunk_errors = sum(row["status"] != "ok" for row in acquisition.get(economic, {}).get("chunks", []))
        unexplained = [row for row in gaps if row["instrument"] == economic and row["classification"] == "unexplained_source_gap"]
        coverage = {
            "instrument": economic,
            "broker_symbol": symbols.get(economic),
            "first_m1_open_utc": pd.Timestamp(times[0], tz="UTC").isoformat() if len(times) else None,
            "last_m1_open_utc": pd.Timestamp(times[-1], tz="UTC").isoformat() if len(times) else None,
            "total_rows": len(times),
            "duplicate_rows_removed": int(data["duplicate_rows_removed"][0]),
            "non_monotonic_after_canonicalization": int(np.sum(np.diff(times) <= 0)) if len(times) else 0,
            "invalid_ohlc_rows": int((~ohlc).sum()),
            "zero_or_invalid_prices": int(sum(np.sum(~np.isfinite(data[name]) | (data[name] <= 0)) for name in ("open", "high", "low", "close"))),
            "acquisition_chunk_errors": chunk_errors,
            "unexplained_gap_count": len(unexplained),
            "maximum_unexplained_gap_minutes": max((int(row["missing_minutes"]) for row in unexplained), default=0),
            "spans_required_prehistory": bool(len(times) and times[0] <= union.min() - 60 * MINUTE_NS),
            "spans_required_end": bool(len(times) and asof_price(data, np.asarray([union.max()], dtype=np.int64))[2][0]),
        }
        coverage_rows.append(coverage)
        source_ready &= bool(symbols.get(economic)) and not chunk_errors and not coverage["invalid_ohlc_rows"] and coverage["spans_required_prehistory"] and coverage["spans_required_end"]
        current = endpoints[economic][1]
        for block in blocks:
            indices = np.searchsorted(union, timestamp_arrays[block["block"] + "_utc_ns"])
            anchors = union[indices]
            fresh = current["current_fresh"][indices]
            closure = (~fresh) & inside_intervals(anchors, common_closures)
            unresolved = (~fresh) & ~closure
            selected = current["current_time"][indices]
            valid_selected = selected[fresh]
            alignment_rows.append(
                {
                    "block": block["block"],
                    "record_type": "instrument",
                    "instrument_or_feature": economic,
                    "exact_gold_timestamps": len(indices),
                    "fresh_causal_fx_close": int(fresh.sum()),
                    "fresh_percentage": float(fresh.mean() * 100),
                    "stale_gt_5m": int((~fresh).sum()),
                    "inside_confirmed_fx_closure": int(closure.sum()),
                    "unresolved_source_gap_rows": int(unresolved.sum()),
                    "maximum_unexplained_gap_minutes": max(
                        (
                            int(row["missing_minutes"])
                            for row in unexplained
                            if pd.Timestamp(row["gap_start_utc"]).value <= anchors[-1]
                            and pd.Timestamp(row["gap_end_utc"]).value >= anchors[0] - 60 * MINUTE_NS
                        ),
                        default=0,
                    ),
                    "maximum_timestamp_mismatch_minutes": float(np.max((anchors[fresh] - valid_selected) / MINUTE_NS)) if fresh.any() else None,
                    "first_usable_timestamp": pd.Timestamp(anchors[fresh][0], tz="UTC").isoformat() if fresh.any() else None,
                    "last_usable_timestamp": pd.Timestamp(anchors[fresh][-1], tz="UTC").isoformat() if fresh.any() else None,
                }
            )
    csv_write(run / "fx_coverage.csv", coverage_rows)
    for block in blocks:
        indices = np.searchsorted(union, timestamp_arrays[block["block"] + "_utc_ns"])
        for column, name in enumerate(FEATURES):
            finite = np.isfinite(features[indices, column])
            alignment_rows.append(
                {
                    "block": block["block"], "record_type": "feature", "instrument_or_feature": name,
                    "exact_gold_timestamps": len(indices), "fresh_causal_fx_close": int(finite.sum()),
                    "fresh_percentage": float(finite.mean() * 100), "stale_gt_5m": int((~finite).sum()),
                    "inside_confirmed_fx_closure": int(legitimate[indices, column].sum()),
                    "unresolved_source_gap_rows": int(unknown[indices, column].sum()),
                    "maximum_unexplained_gap_minutes": None,
                    "maximum_timestamp_mismatch_minutes": None,
                    "first_usable_timestamp": pd.Timestamp(union[indices][finite][0], tz="UTC").isoformat() if finite.any() else None,
                    "last_usable_timestamp": pd.Timestamp(union[indices][finite][-1], tz="UTC").isoformat() if finite.any() else None,
                }
            )
    csv_write(run / "fx_alignment_audit.csv", alignment_rows)
    np.savez_compressed(
        run / "fx_feature_matrix.npz",
        utc_ns=union,
        features=features,
        feature_names=np.asarray(FEATURES),
        unknown_source_mask=unknown,
        legitimate_nan_mask=legitimate,
    )
    block_manifest = []
    for block in blocks:
        indices = np.searchsorted(union, timestamp_arrays[block["block"] + "_utc_ns"])
        block_manifest.append(
            {
                "block": block["block"], "rows": len(indices),
                "timestamp_sha256": block["broker_timestamp_sha256"],
                "utc_timestamp_sha256": block["utc_timestamp_sha256"],
                "feature_matrix_sha256": array_hash(features[indices]),
                "unknown_source_rows": int(unknown[indices].any(axis=1).sum()),
                "legitimate_nan_rows": int(legitimate[indices].any(axis=1).sum()),
            }
        )
    source_identity = {
        economic: {key: acquisition.get(economic, {}).get(key) for key in ("broker_symbol", "archive_sha256", "logical_open_utc_sha256", "logical_close_sha256")}
        for economic in INSTRUMENTS
    }
    source_dataset_sha = hashlib.sha256(json.dumps(source_identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    matrix_manifest = {
        "feature_names_order": list(FEATURES), "dtype": str(features.dtype), "shape": list(features.shape),
        "union_timestamp_sha256": array_hash(union), "usd_fx_feature_matrix_sha256": array_hash(features),
        "feature_archive_sha256": file_hash(run / "fx_feature_matrix.npz"),
        "source_dataset_sha256": source_dataset_sha, "source_identity": source_identity,
        "blocks": block_manifest,
    }
    json_write(run / "fx_feature_matrix_manifest.json", matrix_manifest)
    exact_hashes = all(row["broker_timestamp_sha256"] == EXPECTED_TIMESTAMP_HASHES[row["block"]] for row in blocks)
    unresolved_rows = int(unknown.any(axis=1).sum())
    preliminary_ready = bool(source_ready and exact_hashes and not unresolved_rows and len(resolution) == 3)
    finite = {
        name: {"finite_rows": int(np.isfinite(features[:, index]).sum()), "nan_rows": int(np.isnan(features[:, index]).sum()), "finite_percentage": float(np.isfinite(features[:, index]).mean() * 100)}
        for index, name in enumerate(FEATURES)
    }
    metrics = {
        "run_id": run.name, "run_status": "pending_independent_validator",
        "preliminary_data_readiness": preliminary_ready, "data_foundation_ready": False,
        "symbols": symbols, "coverage": coverage_rows, "feature_coverage": finite,
        "all_six_timestamp_hashes_matched": exact_hashes,
        "usd_fx_feature_matrix_sha256": matrix_manifest["usd_fx_feature_matrix_sha256"],
        "fx_source_dataset_sha256": source_dataset_sha,
        "unknown_source_rows_union": unresolved_rows,
        "legitimate_nan_rows_union": int(legitimate.any(axis=1).sum()),
        "model_training_performed": False, "strategy_evaluation_performed": False,
        "labels_loaded": False, "predictions_loaded": False, "outcomes_loaded": False,
        "gemini_py_changed": False, "operational_model_changed": False,
        "single_next_action": "Run the fixed B0/B1 USD FX pressure information-family experiment only after explicit authorization and only if independent validator passes." if preliminary_ready else "Resolve documented FX source/alignment defects in a new data-only run; do not train.",
    }
    json_write(run / "metrics.json", metrics)
    report = [
        "# GEMINI USD FX PRESSURE FOUNDATION V1", "", "Data-only. No model, label, prediction, outcome, or strategy metric was accessed.", "",
        "## Preliminary decision", "", f"USD FX PRESSURE DATA FOUNDATION READY pending validator: **{'YES' if preliminary_ready else 'NO'}**", "",
        "## Metrics", "", "```json", json.dumps(metrics, indent=2), "```",
    ]
    (run / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if any(inventory(ROOT / "training_runs" / name) != value for name, value in manifest["protected_runs_before"].items()):
        raise RuntimeError("A previous finalized run changed")
    operational_after = {name: file_hash(ROOT / name) for name in manifest["operational_hashes_before"]}
    if operational_after != manifest["operational_hashes_before"]:
        raise RuntimeError("Operational artifact changed")
    manifest = archive.read_json(run / "manifest.json")
    manifest["operational_hashes_after"] = operational_after
    manifest["data"].update(
        {
            "symbols": [symbols.get(name) or f"UNRESOLVED:{name}" for name in INSTRUMENTS],
            "data_sources": ["XM MT5 M1 OHLC via D:\\XM2\\terminal64.exe", "immutable exact timestamp foundation"],
            "source_files": [
                {"path": value["archive_path"], "sha256": value["archive_sha256"], "retention_status": "raw MT5 M1 snapshot retained in run"}
                for value in acquisition.values() if value.get("archive_path")
            ],
            "timezone": "XM EET/EEST broker wall time converted per-date to UTC; M1 open + 1 minute is availability time",
            "data_start_utc": start.isoformat(), "data_end_utc": end.isoformat(),
            "train_start_utc": "not_applicable_data_only", "train_end_utc": "not_applicable_data_only", "train_rows": 0,
            "validation_start_utc": pd.Timestamp(union[0], tz="UTC").isoformat(),
            "validation_end_utc": pd.Timestamp(union[-1], tz="UTC").isoformat(), "validation_rows": len(union),
            "test_start_utc": "not_applicable_data_only", "test_end_utc": "not_applicable_data_only", "test_rows": 0,
            "purge_details": "not applicable: exact timestamp-only data construction; no labels",
            "embargo_details": "not applicable: no fitting or evaluation",
            "raw_snapshot_retained": bool(acquisition),
            "reproducibility_claim": "Raw broker M1 OHLC snapshots, exact timestamp arrays, fixed feature matrix, hashes, and code retained in the finalized run.",
            "mt5_fetch": terminal_record,
        }
    )
    manifest["registry"].update({key: "not_applicable_data_only" for key in archive.REGISTRY_FIELDS})
    manifest["registry"].update(
        {
            "parent_or_incumbent": PARENT.name,
            "selected_configuration": "USD FX pressure fixed five-feature data foundation",
            "validator_result": "PENDING",
        }
    )
    manifest["promotion"].update(
        {"gemini_py_changed": False, "operational_model_changed": False, "operational_artifact_changed": False}
    )
    for path in sorted(run.iterdir()):
        if path.is_file() and path.name not in {"manifest.json", "environment.txt", "stdout.log"}:
            add_artifact(manifest, run, path, "data_foundation_evidence")
    for path in sorted(run.glob("fx_source_*.npz")):
        if not any(item["path"] == path.name for item in manifest["artifacts"]):
            add_artifact(manifest, run, path, "raw_mt5_m1_snapshot")
    json_write(run / "manifest.json", manifest)
    print(json.dumps(metrics, indent=2), flush=True)


def self_check() -> None:
    anchors = np.array([100, 101, 105, 106], dtype=np.int64) * MINUTE_NS
    data = {
        "open_utc_ns": np.array([98, 99, 100, 104], dtype=np.int64) * MINUTE_NS,
        "close": np.array([1.0, 2.0, 4.0, 8.0]),
    }
    price, selected, fresh = asof_price(data, anchors)
    assert price.tolist() == [2.0, 4.0, 8.0, 8.0]
    assert np.all(selected <= anchors) and fresh.all()
    sample = np.array([[1.0, 2.0, 4.0]])
    assert math.isclose(sample.std(axis=1, ddof=1)[0], math.sqrt(7 / 3))
    assert list(FEATURES) == ["USD_PRESSURE_1M", "USD_PRESSURE_5M", "USD_PRESSURE_15M", "USD_PRESSURE_60M", "USD_DISPERSION_15M"]
    print("SELF_CHECK_PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "self-check"))
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.action == "self-check":
        self_check()
    else:
        if args.run_dir is None:
            parser.error("--run-dir is required")
        run_dir = args.run_dir.resolve()
        if run_dir.parent != ROOT / "training_runs" or (run_dir / "FINALIZED.json").exists():
            raise RuntimeError("Invalid or finalized run directory")
        main_run(run_dir)
