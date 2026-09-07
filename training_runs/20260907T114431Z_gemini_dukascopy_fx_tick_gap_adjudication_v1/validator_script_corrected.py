"""Independent DATA-only validator for Dukascopy BID tick gap adjudication."""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import sqlite3
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import training_run_history as archive


ROOT = Path(__file__).resolve().parent
PREVIOUS = (
    ROOT
    / "training_runs/20260906T180930Z_gemini_usd_fx_pressure_source_reconciliation_v1"
)
EARLIER = ROOT / "training_runs/20260906T104638Z_gemini_usd_fx_pressure_foundation_v1"
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
HORIZONS = (1, 5, 15, 60)
ORIENTATION = (-1.0, -1.0, 1.0)
EXPECTED_TIMESTAMP_HASHES = {
    "fold1_train": "6a7405a13f30c54e6863cf6e80ea1b2e9ee93a90902e5edd37fe4305d471aab6",
    "fold1_score": "47086d0837f09e86d8874874de302e96e7573efb29236f24720f4a14e0e33c94",
    "fold2_train": "691d8d2b01c3829f010ab559b1daa07c1899f8425a5b8b23b3007bf58467df44",
    "fold2_score": "1fc6500ff642dbf7172328f71f13f38712d11f6c0ce873c38524745710c608b8",
    "fold3_train": "3cf7f68ba2cf994d23d4db97c2e7ec10b2ed772245e3d4b776272ed20ee06b1b",
    "fold3_score": "1451d2069b1d087dc8b4bb7b3ed5840a7b2c03d4adfb548eadc611ed07803449",
}
MINUTE_NS = 60_000_000_000
HOUR_NS = 60 * MINUTE_NS
TICK_RECORD = struct.Struct(">IIIff")


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


def decode(body: bytes, pair: str, hour_ns: int) -> tuple[np.ndarray, np.ndarray]:
    if not body:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    raw = lzma.decompress(body)
    if len(raw) % TICK_RECORD.size:
        raise ValueError("unaligned BI5 payload")
    scale = round(1.0 / POINTS[pair])
    rows = list(TICK_RECORD.iter_unpack(raw))
    times = np.asarray([hour_ns + int(row[0]) * 1_000_000 for row in rows], dtype=np.int64)
    bids = np.asarray([row[2] / scale for row in rows], dtype=np.float64)
    if len(times) and (np.any(np.diff(times) < 0) or np.any(times >= hour_ns + HOUR_NS)):
        raise ValueError("invalid tick timestamp order")
    if len(times):
        key = np.rec.fromarrays([times, bids], names="time,bid")
        _, keep = np.unique(key, return_index=True)
        keep.sort()
        times, bids = times[keep], bids[keep]
    return times, bids


def aggregate(times: np.ndarray, bids: np.ndarray) -> dict[str, np.ndarray]:
    if not len(times):
        return {
            "open_utc_ns": np.empty(0, dtype=np.int64),
            **{
                field: np.empty(0, dtype=np.float64)
                for field in ("open", "high", "low", "close")
            },
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
    }


def rebuild_tick_state(
    run: Path, requests: pd.DataFrame
) -> tuple[
    dict[tuple[str, int], str],
    dict[str, dict[str, np.ndarray]],
    bool,
    list[str],
]:
    database = (run / "dukascopy_tick_responses.sqlite").resolve().as_uri()
    db = sqlite3.connect(f"{database}?mode=ro", uri=True)
    states, pieces = {}, {pair: [] for pair in INSTRUMENTS}
    errors: list[str] = []
    complete = True
    for row in requests.itertuples(index=False):
        hour_ns = pd.Timestamp(row.hour_utc).value
        records = list(
            db.execute(
                "SELECT acquisition,status,body,body_sha256 FROM responses "
                "WHERE pair=? AND hour_utc=? ORDER BY acquisition",
                (row.instrument, row.hour_utc),
            )
        )
        state = "retrieval_error"
        if len(records) != 2:
            complete = False
            errors.append(f"{row.instrument} {row.hour_utc}: {len(records)} acquisitions")
        elif [item[1] for item in records] == [404, 404]:
            state = "verified_no_tick_hour"
        elif [item[1] for item in records] == [200, 200]:
            decoded = []
            try:
                decoded = [decode(item[2], row.instrument, hour_ns) for item in records]
            except (lzma.LZMAError, ValueError) as exc:
                errors.append(f"{row.instrument} {row.hour_utc}: {exc}")
            if decoded:
                hashes = [logical_hash(*item) for item in decoded]
                if hashes[0] == hashes[1]:
                    state = "ticks" if len(decoded[0][0]) else "verified_no_tick_hour"
                    if state == "ticks":
                        pieces[row.instrument].append(decoded[0])
                else:
                    state = "inconsistent_provider_history"
        states[(row.instrument, hour_ns)] = state
    invalid_rows = list(
        db.execute(
            "SELECT pair,hour_utc,acquisition,url FROM responses "
            "WHERE url NOT LIKE 'https://datafeed.dukascopy.com/datafeed/%'"
        )
    )
    db.close()
    if invalid_rows:
        complete = False
        errors.append(f"non-Dukascopy URLs: {invalid_rows[:3]}")
    derived = {}
    for pair in INSTRUMENTS:
        if pieces[pair]:
            times = np.concatenate([item[0] for item in pieces[pair]])
            bids = np.concatenate([item[1] for item in pieces[pair]])
            order = np.argsort(times, kind="stable")
            times, bids = times[order], bids[order]
            key = np.rec.fromarrays([times, bids], names="time,bid")
            _, keep = np.unique(key, return_index=True)
            keep.sort()
            times, bids = times[keep], bids[keep]
        else:
            times = np.empty(0, dtype=np.int64)
            bids = np.empty(0, dtype=np.float64)
        derived[pair] = aggregate(times, bids)
    return states, derived, complete, errors


def load_native(run: Path) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for pair in INSTRUMENTS:
        with np.load(
            run / f"fx_source_DUKASCOPY_{CODES[pair]}.npz", allow_pickle=False
        ) as source:
            result[pair] = {key: source[key].copy() for key in source.files}
    return result


def recompute_equivalence(
    requests: pd.DataFrame,
    states: dict[tuple[str, int], str],
    native: dict[str, dict[str, np.ndarray]],
    derived: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    expected_total = compared_total = mismatch_total = 0
    for pair in INSTRUMENTS:
        usable_hours = {
            pd.Timestamp(row.hour_utc).value
            for row in requests[requests["instrument"].eq(pair)].itertuples(index=False)
            if states[(pair, pd.Timestamp(row.hour_utc).value)]
            in {"ticks", "verified_no_tick_hour"}
        }
        source, tick = native[pair], derived[pair]
        native_hours = (source["open_utc_ns"] // HOUR_NS) * HOUR_NS
        native_indices = np.flatnonzero(np.isin(native_hours, list(usable_hours)))
        positions = np.searchsorted(tick["open_utc_ns"], source["open_utc_ns"][native_indices])
        found = positions < len(tick["open_utc_ns"])
        if len(tick["open_utc_ns"]):
            safe = np.minimum(positions, len(tick["open_utc_ns"]) - 1)
            found &= tick["open_utc_ns"][safe] == source["open_utc_ns"][native_indices]
        else:
            found[:] = False
        mismatch = int((~found).sum())
        for local, native_index in enumerate(native_indices[found]):
            tick_index = positions[np.flatnonzero(found)[local]]
            mismatch += int(
                any(
                    abs(float(source[field][native_index]) - float(tick[field][tick_index]))
                    > POINTS[pair] + 1e-12
                    for field in ("open", "high", "low", "close")
                )
            )
        expected_total += len(native_indices)
        compared_total += int(found.sum())
        mismatch_total += mismatch
    return {
        "gate": (
            "PASS"
            if compared_total >= 1_000
            and compared_total == expected_total
            and mismatch_total == 0
            else "FAIL"
        ),
        "native_overlap_expected": expected_total,
        "compared_bars": compared_total,
        "mismatch_bars": mismatch_total,
        "mismatch_rate": mismatch_total / expected_total if expected_total else 1.0,
    }


def verify_classifications(
    gaps: pd.DataFrame,
    saved: pd.DataFrame,
    states: dict[tuple[str, int], str],
    derived: dict[str, dict[str, np.ndarray]],
    equivalence_passed: bool,
) -> tuple[bool, dict[str, dict[str, int]]]:
    expected_rows = int(gaps["missing_minutes"].sum())
    if len(saved) != expected_rows:
        return False, {}
    tick_minutes = {
        pair: set(values["open_utc_ns"].tolist()) for pair, values in derived.items()
    }
    valid = True
    for row in saved.itertuples(index=False):
        minute = pd.Timestamp(row.minute_open_utc).value
        state = states.get((row.instrument, (minute // HOUR_NS) * HOUR_NS), "retrieval_error")
        if state == "retrieval_error":
            expected = "retrieval_error"
        elif state == "inconsistent_provider_history":
            expected = "inconsistent_provider_history"
        elif minute in tick_minutes[row.instrument]:
            expected = (
                "native_m1_missing_but_ticks_present"
                if equivalence_passed
                else "inconsistent_provider_history"
            )
        else:
            expected = "verified_no_tick_interval"
        valid &= row.classification == expected
    names = (
        "verified_no_tick_interval",
        "native_m1_missing_but_ticks_present",
        "retrieval_error",
        "inconsistent_provider_history",
    )
    summary = {
        name: {
            "intervals": int(saved.loc[saved["classification"].eq(name), "gap_id"].nunique()),
            "minutes": int(saved["classification"].eq(name).sum()),
        }
        for name in names
    }
    return valid, summary


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[list[int]] = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return [(start, end) for start, end in result]


def closures(all_gaps: pd.DataFrame, classes: pd.DataFrame) -> dict[str, list[tuple[int, int]]]:
    result = {pair: [] for pair in INSTRUMENTS}
    for gap in all_gaps[~all_gaps["classification"].eq("unexplained_source_gap")].itertuples(index=False):
        result[gap.instrument].append(
            (
                pd.Timestamp(gap.gap_start_utc).value + MINUTE_NS,
                pd.Timestamp(gap.gap_end_utc).value + MINUTE_NS,
            )
        )
    verified = classes[classes["classification"].eq("verified_no_tick_interval")].copy()
    verified["minute_ns"] = np.asarray(
        [pd.Timestamp(value).value for value in verified["minute_open_utc"]],
        dtype=np.int64,
    )
    for pair in INSTRUMENTS:
        values = np.sort(verified.loc[verified["instrument"].eq(pair), "minute_ns"].to_numpy(np.int64))
        if len(values):
            cuts = np.flatnonzero(np.diff(values) > MINUTE_NS) + 1
            for group in np.split(values, cuts):
                result[pair].append((int(group[0] + MINUTE_NS), int(group[-1] + 2 * MINUTE_NS)))
        result[pair] = merge_intervals(result[pair])
    return result


def inside(values: np.ndarray, intervals: list[tuple[int, int]]) -> np.ndarray:
    if not intervals:
        return np.zeros(len(values), dtype=bool)
    starts = np.asarray([item[0] for item in intervals], dtype=np.int64)
    ends = np.asarray([item[1] for item in intervals], dtype=np.int64)
    index = np.searchsorted(starts, values, side="right") - 1
    valid = index >= 0
    result = np.zeros(len(values), dtype=bool)
    result[valid] = values[valid] < ends[index[valid]]
    return result


def asof(source: dict[str, np.ndarray], anchors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    available = source["open_utc_ns"] + MINUTE_NS
    index = np.searchsorted(available, anchors, side="right") - 1
    exists = index >= 0
    safe = np.maximum(index, 0)
    age = np.where(exists, anchors - available[safe], np.iinfo(np.int64).max)
    fresh = exists & (age >= 0) & (age <= 5 * MINUTE_NS)
    return np.where(fresh, source["close"][safe], np.nan), fresh


def matrix(
    times: np.ndarray,
    sources: dict[str, dict[str, np.ndarray]],
    closure_map: dict[str, list[tuple[int, int]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    component = np.full((len(times), 3, 4), np.nan)
    unknown_component = np.zeros_like(component, dtype=bool)
    for pair_index, pair in enumerate(INSTRUMENTS):
        current, current_ok = asof(sources[pair], times)
        for horizon_index, horizon in enumerate(HORIZONS):
            anchor = times - horizon * MINUTE_NS
            past, past_ok = asof(sources[pair], anchor)
            valid = current_ok & past_ok
            component[:, pair_index, horizon_index] = ORIENTATION[pair_index] * np.log(
                current / past
            )
            legitimate = (~valid) & (
                (current_ok | inside(times, closure_map[pair]))
                & (past_ok | inside(anchor, closure_map[pair]))
            )
            unknown_component[:, pair_index, horizon_index] = (~valid) & ~legitimate
    values = np.full((len(times), 5), np.nan)
    unknown = np.zeros_like(values, dtype=bool)
    legitimate = np.zeros_like(values, dtype=bool)
    for index in range(4):
        finite = np.isfinite(component[:, :, index]).all(axis=1)
        values[finite, index] = component[finite, :, index].mean(axis=1)
        unknown[:, index] = unknown_component[:, :, index].any(axis=1)
        legitimate[:, index] = (~finite) & ~unknown[:, index]
    finite = np.isfinite(component[:, :, 2]).all(axis=1)
    values[finite, 4] = component[finite, :, 2].std(axis=1, ddof=1)
    unknown[:, 4] = unknown_component[:, :, 2].any(axis=1)
    legitimate[:, 4] = (~finite) & ~unknown[:, 4]
    return values, unknown, legitimate


def repaired_sources(
    native: dict[str, dict[str, np.ndarray]],
    derived: dict[str, dict[str, np.ndarray]],
    classes: pd.DataFrame,
    allowed: bool,
) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for pair in INSTRUMENTS:
        repair = set()
        if allowed:
            repair = {
                pd.Timestamp(value).value
                for value in classes.loc[
                    classes["instrument"].eq(pair)
                    & classes["classification"].eq("native_m1_missing_but_ticks_present"),
                    "minute_open_utc",
                ]
            }
        take = np.asarray(
            [time in repair for time in derived[pair]["open_utc_ns"]], dtype=bool
        )
        arrays = {
            field: np.concatenate([native[pair][field], derived[pair][field][take]])
            for field in ("open_utc_ns", "open", "high", "low", "close")
        }
        order = np.argsort(arrays["open_utc_ns"], kind="stable")
        result[pair] = {field: values[order] for field, values in arrays.items()}
    return result


def expected_affected_rows(
    gaps: pd.DataFrame, times: np.ndarray, old_unknown: np.ndarray
) -> pd.DataFrame:
    rows = []
    feature_horizons = (1, 5, 15, 60, 15)
    for gap_id, gap in enumerate(gaps.itertuples(index=False)):
        start = pd.Timestamp(gap.gap_start_utc).value + MINUTE_NS
        end = pd.Timestamp(gap.gap_end_utc).value + MINUTE_NS
        for feature_index, (feature, horizon) in enumerate(zip(FEATURES, feature_horizons)):
            current = (times >= start) & (times < end)
            past = (times - horizon * MINUTE_NS >= start) & (
                times - horizon * MINUTE_NS < end
            )
            for index in np.flatnonzero((current | past) & old_unknown[:, feature_index]):
                rows.append(
                    {
                        "gap_id": gap_id,
                        "instrument": gap.instrument,
                        "gold_timestamp_utc": pd.Timestamp(times[index], tz="UTC").isoformat(),
                        "feature": feature,
                        "horizon_minutes": horizon,
                        "affected_anchor": (
                            "both"
                            if current[index] and past[index]
                            else "current"
                            if current[index]
                            else "past"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def validate(run: Path) -> int:
    manifest = archive.read_json(run / "manifest.json")
    metrics = archive.read_json(run / "metrics.json")
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, evidence: Any, category: str = "methodology") -> None:
        checks.append(
            {
                "check": name,
                "verdict": "PASS" if condition else "FAIL",
                "evidence": evidence,
                "category": category,
            }
        )

    protected = manifest["protected_runs_before"]
    old_ok = all(
        source.name in protected and inventory(source) == protected[source.name]
        for source in (EARLIER, PREVIOUS)
    )
    check("old failed runs byte-identical", old_ok, list(protected))
    all_gaps = pd.read_csv(PREVIOUS / "source_gap_audit.csv")
    expected_gaps = all_gaps[
        all_gaps["classification"].eq("unexplained_source_gap")
    ].reset_index(drop=True)
    saved_gaps = pd.read_csv(run / "gap_universe.csv")
    base_gap_columns = expected_gaps.columns.tolist()
    gap_ok = (
        len(saved_gaps) == 554
        and saved_gaps[base_gap_columns].equals(expected_gaps)
        and saved_gaps["native_before_open_utc"].notna().all()
        and saved_gaps["native_after_open_utc"].notna().all()
    )
    check(
        "exact 554 input gap universe",
        gap_ok,
        {"intervals": len(saved_gaps), "minutes": int(saved_gaps["missing_minutes"].sum())},
    )
    requests = pd.read_csv(run / "tick_request_manifest.csv")
    source_ok = (
        set(requests["instrument"]) == set(INSTRUMENTS)
        and requests["url"].str.startswith(
            "https://datafeed.dukascopy.com/datafeed/"
        ).all()
        and manifest["foundation_specification"]["economic_data_provider"] == "Dukascopy"
    )
    check("Dukascopy-only sourcing", source_ok, {"requests": len(requests)})
    states, derived, acquisition_complete, decode_errors = rebuild_tick_state(run, requests)
    check(
        "repeated acquisition evidence",
        acquisition_complete,
        {"pair_hours": len(requests), "errors": decode_errors[:10]},
    )
    tick_semantics = all(
        not len(values["open_utc_ns"])
        or (
            np.all(np.diff(values["open_utc_ns"]) > 0)
            and np.all(values["high"] >= values["open"])
            and np.all(values["high"] >= values["close"])
            and np.all(values["low"] <= values["open"])
            and np.all(values["low"] <= values["close"])
        )
        for values in derived.values()
    )
    check("BID tick timestamps and aggregation", tick_semantics, "UTC floor-minute OHLC")
    native = load_native(run)
    equivalence = recompute_equivalence(requests, states, native, derived)
    saved_equivalence = archive.read_json(run / "native_tick_equivalence.json")
    equivalence_reproduced = all(
        saved_equivalence[key] == equivalence[key]
        for key in ("gate", "native_overlap_expected", "compared_bars", "mismatch_bars")
    )
    check("native-M1 equivalence recomputed", equivalence_reproduced, equivalence)
    check(
        "native-M1 equivalence gate",
        equivalence["gate"] == "PASS",
        equivalence,
        "data",
    )
    classes = pd.read_csv(run / "gap_minute_classification.csv.gz")
    classifications_ok, class_summary = verify_classifications(
        saved_gaps,
        classes,
        states,
        derived,
        equivalence["gate"] == "PASS",
    )
    check("gap classifications reproduce raw ticks", classifications_ok, class_summary)
    retrieval_pass = all(
        state in {"ticks", "verified_no_tick_hour"} for state in states.values()
    )
    check(
        "Dukascopy tick retrieval",
        retrieval_pass,
        pd.Series(list(states.values())).value_counts().to_dict(),
        "data",
    )
    with np.load(run / "exact_timestamps.npz", allow_pickle=False) as exact:
        pieces, timestamp_ok = [], True
        timestamp_hashes = {}
        for block, expected in EXPECTED_TIMESTAMP_HASHES.items():
            broker = exact[f"{block}_broker_ns"].astype(np.int64)
            utc = exact[f"{block}_utc_ns"].astype(np.int64)
            timestamp_hashes[block] = digest(broker)
            timestamp_ok &= timestamp_hashes[block] == expected
            pieces.append(utc)
        times = np.unique(np.concatenate(pieces))
    check("six GOLD timestamp hashes", timestamp_ok, timestamp_hashes)
    closure_map = closures(all_gaps, classes)
    a_values, a_unknown, _ = matrix(times, native, closure_map)
    repaired = repaired_sources(
        native, derived, classes, equivalence["gate"] == "PASS"
    )
    values, unknown, legitimate = matrix(times, repaired, closure_map)
    with np.load(PREVIOUS / "usd_fx_reconciled_feature_matrix.npz", allow_pickle=False) as old:
        old_unknown_features = old["unknown_source_mask"].astype(bool)
        old_unknown = old_unknown_features.any(axis=1)
    affected_expected = expected_affected_rows(saved_gaps, times, old_unknown_features)
    affected_saved = pd.read_csv(run / "gap_affected_exact_rows.csv.gz")
    affected_ok = affected_saved.equals(affected_expected)
    check(
        "exact affected GOLD timestamps and horizons",
        affected_ok,
        {"records": len(affected_saved), "unique_exact_rows": affected_saved["gold_timestamp_utc"].nunique()},
    )
    resolved_a = old_unknown & ~a_unknown.any(axis=1)
    resolved_b = old_unknown & a_unknown.any(axis=1) & ~unknown.any(axis=1)
    remaining = old_unknown & unknown.any(axis=1)
    no_new_unknown = not np.any(~old_unknown & unknown.any(axis=1))
    row_counts_match = (
        int(old_unknown.sum()) == 8476
        and int(resolved_a.sum()) == metrics["resolved_by_verified_no_tick"]
        and int(resolved_b.sum()) == metrics["resolved_by_tick_reconstruction"]
        and int(remaining.sum()) == metrics["remaining_unresolved_rows"]
        and no_new_unknown
    )
    check(
        "exact 8476-row reconciliation",
        row_counts_match,
        {
            "previous": int(old_unknown.sum()),
            "verified_no_tick": int(resolved_a.sum()),
            "tick_reconstruction": int(resolved_b.sum()),
            "remaining": int(remaining.sum()),
        },
    )
    sources_match = True
    for pair in INSTRUMENTS:
        with np.load(
            run / f"dukascopy_tick_reconciled_{CODES[pair]}.npz",
            allow_pickle=False,
        ) as saved:
            sources_match &= all(
                np.array_equal(saved[field], repaired[pair][field])
                for field in ("open_utc_ns", "open", "high", "low", "close")
            )
    check("same-provider repaired source reconstruction", sources_match, list(INSTRUMENTS))
    matrix_path = run / "dukascopy_tick_reconciled_feature_matrix.npz"
    if matrix_path.is_file():
        with np.load(matrix_path, allow_pickle=False) as saved:
            matrix_match = (
                np.array_equal(saved["utc_ns"], times)
                and saved["feature_names"].astype(str).tolist() == list(FEATURES)
                and np.array_equal(saved["features"], values, equal_nan=True)
                and np.array_equal(saved["unknown_source_mask"], unknown)
                and np.array_equal(saved["legitimate_nan_mask"], legitimate)
            )
    else:
        matrix_match = metrics["dukascopy_tick_reconciled_feature_matrix_sha256"] is None
    check("5-minute causal matrix reconstruction", matrix_match, {"matrix_exists": matrix_path.is_file()})
    data_gate = (
        retrieval_pass
        and equivalence["gate"] == "PASS"
        and int(remaining.sum()) == 0
        and matrix_path.is_file()
        and matrix_match
    )
    check("remaining unresolved rows are zero", int(remaining.sum()) == 0, int(remaining.sum()), "data")
    check("one Dukascopy provider", metrics["source_provider_count"] == 1, metrics["source_provider_count"])
    source_text = (run / manifest["training_script_snapshot"]).read_text(encoding="utf-8").lower()
    forbidden = [
        token
        for token in ("import xgboost", "import sklearn", "predict_proba", "realized_r", "trade_ledger")
        if token in source_text
    ]
    check(
        "no model or outcome access",
        not forbidden
        and not metrics["model_training_performed"]
        and not metrics["strategy_evaluation_performed"],
        forbidden,
    )
    current = {
        name: archive.file_sha256(ROOT / name)
        for name in manifest["operational_hashes_before"]
    }
    operational_ok = (
        current
        == manifest["operational_hashes_before"]
        == manifest["operational_hashes_after"]
    )
    check("production unchanged", operational_ok, current)
    methodology_pass = all(
        item["verdict"] == "PASS"
        for item in checks
        if item["category"] == "methodology"
    )
    ready = bool(methodology_pass and data_gate and metrics["data_foundation_ready"])
    generic = [
        ("chronology", tick_semantics and matrix_match, "UTC completed-bar causality"),
        ("feature leakage", matrix_match, "fixed causal feature reconstruction"),
        ("label maturity", True, "not applicable: no labels"),
        ("OOF predictions", True, "not applicable: no predictions"),
        ("calibration", True, "not applicable: no calibration"),
        ("threshold selection", True, "not applicable: no selection"),
        ("purge/embargo", True, "not applicable: no fitting"),
        ("holdout contamination", True, "no outcomes accessed"),
        ("recent-period reuse", True, "no strategy evidence claimed"),
        ("execution alignment", True, "not applicable: no strategy execution"),
        ("cost assumptions", True, "not applicable: no economics"),
        ("multiple-testing risk", True, "one frozen data hypothesis"),
    ]
    result = {
        "overall": "PASS" if methodology_pass else "FAIL",
        "internal_methodology": "PASS" if methodology_pass else "FAIL",
        "data_certification": "PASS" if ready else "FAIL",
        "data_foundation_ready": ready,
        "checks": checks,
        "walk_forward_checks": [
            {"check": name, "verdict": "PASS" if passed else "FAIL", "evidence": evidence}
            for name, passed, evidence in generic
        ],
        "model_training_performed": False,
        "strategy_evaluation_performed": False,
    }
    archive.write_json(run / "validator.json", result)
    lines = [
        "# Independent DATA-only validator",
        "",
        f"Overall: **{result['overall']}**",
        "",
        f"Internal methodology: **{result['internal_methodology']}**",
        "",
        f"Data certification: **{result['data_certification']}**",
        "",
        f"Foundation ready: **{'YES' if ready else 'NO'}**",
        "",
        "| Check | Verdict | Evidence |",
        "|---|---|---|",
        *[
            f"| {item['check']} | {item['verdict']} | {str(item['evidence']).replace('|', '/')} |"
            for item in checks
        ],
        "",
        "## Walk-forward-validator contract",
        "",
        "| Check | Verdict | Evidence | Failure or reason for pass | Required validation correction |",
        "|---|---|---|---|---|",
        *[
            f"| {name} | {'PASS' if passed else 'FAIL'} | {evidence} | "
            f"{'Applicable evidence reproduced' if passed else 'Not reproducible'} | "
            f"{'None' if passed else 'Correct data-only methodology before reuse'} |"
            for name, passed, evidence in generic
        ],
        "",
        "No OOS strategy metrics exist in this data-only run. The 60% WR target and economic guardrails are not applicable.",
        "",
        f"Submitted data-foundation claim: **{'valid' if ready else 'invalid'}**.",
        "",
    ]
    (run / "validator.md").write_text("\n".join(lines), encoding="utf-8")
    metrics.update(
        {
            "validator_internal_methodology": result["internal_methodology"],
            "validator_data_certification": result["data_certification"],
            "data_foundation_ready": ready,
            "run_status": "pass" if ready else "fail",
        }
    )
    archive.write_json(run / "metrics.json", metrics)
    report = (run / "report.md").read_text(encoding="utf-8").split(
        "\n## Independent validation\n", 1
    )[0].rstrip()
    report += (
        "\n\n## Independent validation\n\n"
        f"Internal methodology: **{result['internal_methodology']}**. "
        f"Data certification: **{result['data_certification']}**. "
        f"Foundation ready: **{'YES' if ready else 'NO'}**.\n"
    )
    (run / "report.md").write_text(report, encoding="utf-8")
    manifest["registry"]["validator_result"] = (
        f"internal {result['internal_methodology']}; "
        f"data certification {result['data_certification']}"
    )
    for name in (
        "metrics.json",
        "report.md",
        "validator.json",
        "validator.md",
        "validator_script.py",
    ):
        manifest["artifacts"] = [
            item for item in manifest["artifacts"] if item.get("path") != name
        ]
        manifest["artifacts"].append(
            {
                "kind": "validator_evidence",
                "path": name,
                "sha256": archive.file_sha256(run / name),
                "retention_status": "stored_in_finalized_run_git_archival_pending",
            }
        )
    archive.write_json(run / "manifest.json", manifest)
    print(json.dumps(result, indent=2))
    return 0 if methodology_pass else 1


def self_check() -> None:
    raw = TICK_RECORD.pack(1_000, 110_001, 110_000, 1.0, 1.0)
    times, bids = decode(lzma.compress(raw), "EUR/USD", 0)
    bars = aggregate(times, bids)
    assert bars["open"].tolist() == [1.1]
    assert bars["open_utc_ns"].tolist() == [0]
    print("DUKASCOPY_TICK_GAP_VALIDATOR_SELF_CHECK_PASS")


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
        raise SystemExit(validate(arguments.run_dir.resolve()))
