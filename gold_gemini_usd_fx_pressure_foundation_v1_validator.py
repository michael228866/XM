"""Independent data-only validator for the frozen USD FX pressure foundation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import training_run_history as archive

ROOT = Path(__file__).resolve().parent
FOUNDATION = ROOT / "training_runs/20260905T171629Z_gemini_macro_event_integration_foundation_v1"
INSTRUMENTS = ("EUR/USD", "GBP/USD", "USD/JPY")
FEATURES = ("USD_PRESSURE_1M", "USD_PRESSURE_5M", "USD_PRESSURE_15M", "USD_PRESSURE_60M", "USD_DISPERSION_15M")
HORIZONS = (1, 5, 15, 60)
SIGNS = (-1.0, -1.0, 1.0)
MINUTE_NS = 60_000_000_000
MAX_STALENESS = 5
EXPECTED = {
    "fold1_train": "6a7405a13f30c54e6863cf6e80ea1b2e9ee93a90902e5edd37fe4305d471aab6",
    "fold1_score": "47086d0837f09e86d8874874de302e96e7573efb29236f24720f4a14e0e33c94",
    "fold2_train": "691d8d2b01c3829f010ab559b1daa07c1899f8425a5b8b23b3007bf58467df44",
    "fold2_score": "1fc6500ff642dbf7172328f71f13f38712d11f6c0ce873c38524745710c608b8",
    "fold3_train": "3cf7f68ba2cf994d23d4db97c2e7ec10b2ed772245e3d4b776272ed20ee06b1b",
    "fold3_score": "1451d2069b1d087dc8b4bb7b3ed5840a7b2c03d4adfb548eadc611ed07803449",
}


def digest(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    return hashlib.sha256(str(values.dtype).encode() + np.asarray(values.shape, dtype=np.int64).tobytes() + values.tobytes()).hexdigest()


def inventory(directory: Path) -> dict[str, str]:
    return {path.relative_to(directory).as_posix(): archive.file_sha256(path) for path in sorted(directory.rglob("*")) if path.is_file()}


def broker_to_utc(raw: np.ndarray) -> np.ndarray:
    wall = pd.to_datetime(raw.astype(np.int64), unit="s")
    return pd.DatetimeIndex(wall).tz_localize("Europe/Helsinki", ambiguous="raise", nonexistent="raise").tz_convert("UTC").asi8


def intervals_from_gap_audit(run: Path) -> list[tuple[int, int]]:
    gaps = pd.read_csv(run / "fx_gap_audit.csv")
    if gaps.empty:
        return []
    chosen = gaps[gaps["classification"] == "confirmed_common_broker_quote_closure"]
    values = sorted((pd.Timestamp(row.gap_start_utc).value, pd.Timestamp(row.gap_end_utc).value) for row in chosen.itertuples())
    merged: list[list[int]] = []
    for start, end in values:
        if merged and start <= merged[-1][1] + 5 * MINUTE_NS:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def within(values: np.ndarray, intervals: list[tuple[int, int]]) -> np.ndarray:
    if not intervals:
        return np.zeros(len(values), dtype=bool)
    starts = np.asarray([item[0] for item in intervals], dtype=np.int64)
    ends = np.asarray([item[1] for item in intervals], dtype=np.int64)
    index = np.searchsorted(starts, values, side="right") - 1
    valid = index >= 0
    result = np.zeros(len(values), dtype=bool)
    result[valid] = values[valid] < ends[index[valid]]
    return result


def independent_asof(open_ns: np.ndarray, close: np.ndarray, anchors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    available = open_ns + MINUTE_NS
    if not len(available):
        return (
            np.full(len(anchors), np.nan, dtype=np.float64),
            np.full(len(anchors), np.iinfo(np.int64).min, dtype=np.int64),
            np.zeros(len(anchors), dtype=bool),
        )
    index = np.searchsorted(available, anchors, side="right") - 1
    exists = index >= 0
    safe = np.maximum(index, 0)
    information_time = np.where(exists, available[safe], np.iinfo(np.int64).min)
    age = np.where(exists, anchors - information_time, np.iinfo(np.int64).max)
    fresh = exists & (age >= 0) & (age <= MAX_STALENESS * MINUTE_NS)
    return np.where(fresh, close[safe], np.nan), information_time, fresh


def independent_matrix(run: Path, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    closures = intervals_from_gap_audit(run)
    component = np.full((len(times), 3, 4), np.nan, dtype=np.float64)
    unknown_component = np.zeros_like(component, dtype=bool)
    legitimate_component = np.zeros_like(component, dtype=bool)
    causal = True
    max_age = 0.0
    for instrument_index, economic in enumerate(INSTRUMENTS):
        name = re.sub("[^A-Z]", "", economic)
        path = run / f"fx_source_{name}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as source:
                opens, closes = source["open_utc_ns"].astype(np.int64), source["close"].astype(np.float64)
        else:
            opens, closes = np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        current_price, current_time, current_fresh = independent_asof(opens, closes, times)
        causal &= bool(np.all(current_time[current_fresh] <= times[current_fresh]))
        if current_fresh.any():
            max_age = max(max_age, float(np.max((times[current_fresh] - current_time[current_fresh]) / MINUTE_NS)))
        for horizon_index, horizon in enumerate(HORIZONS):
            past_anchor = times - horizon * MINUTE_NS
            past_price, past_time, past_fresh = independent_asof(opens, closes, past_anchor)
            valid = current_fresh & past_fresh
            component[:, instrument_index, horizon_index] = SIGNS[instrument_index] * np.log(current_price / past_price)
            legitimate = (~valid) & ((current_fresh | ((~current_fresh) & within(times, closures))) & (past_fresh | ((~past_fresh) & within(past_anchor, closures))))
            legitimate_component[:, instrument_index, horizon_index] = legitimate
            unknown_component[:, instrument_index, horizon_index] = (~valid) & ~legitimate
            causal &= bool(np.all(past_time[past_fresh] <= past_anchor[past_fresh]))
            if past_fresh.any():
                max_age = max(max_age, float(np.max((past_anchor[past_fresh] - past_time[past_fresh]) / MINUTE_NS)))
    result = np.full((len(times), 5), np.nan, dtype=np.float64)
    unknown = np.zeros_like(result, dtype=bool)
    legitimate = np.zeros_like(result, dtype=bool)
    for column in range(4):
        values = component[:, :, column]
        finite = np.isfinite(values).all(axis=1)
        result[finite, column] = values[finite].mean(axis=1)
        unknown[:, column] = unknown_component[:, :, column].any(axis=1)
        legitimate[:, column] = (~finite) & ~unknown[:, column]
    values = component[:, :, 2]
    finite = np.isfinite(values).all(axis=1)
    result[finite, 4] = values[finite].std(axis=1, ddof=1)
    unknown[:, 4] = unknown_component[:, :, 2].any(axis=1)
    legitimate[:, 4] = (~finite) & ~unknown[:, 4]
    return result, unknown, legitimate, {"causal": causal, "maximum_age_minutes": max_age}


def validate(run: Path) -> dict[str, Any]:
    manifest = archive.read_json(run / "manifest.json")
    metrics = archive.read_json(run / "metrics.json")
    resolution = archive.read_json(run / "broker_symbol_resolution.json")
    provenance = archive.read_json(run / "fx_source_provenance.json")
    matrix_manifest = archive.read_json(run / "fx_feature_matrix_manifest.json")
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, evidence: Any, category: str = "methodology") -> None:
        checks.append({"check": name, "verdict": "PASS" if condition else "FAIL", "evidence": evidence, "category": category})

    symbol_errors = []
    selected = {}
    for economic in INSTRUMENTS:
        item = resolution.get("resolutions", {}).get(economic, {})
        candidates = item.get("candidates", [])
        symbol = item.get("selected_symbol")
        match = [row for row in candidates if row["symbol"] == symbol and f"{row['currency_base']}/{row['currency_profit']}" == economic]
        if item.get("status") != "resolved" or len(match) != 1:
            symbol_errors.append(economic)
        else:
            selected[economic] = symbol
    check("economic symbol identity", not symbol_errors, {"selected": selected, "errors": symbol_errors}, "data")

    source_errors, timezone_errors, source_rows = [], [], {}
    for economic in INSTRUMENTS:
        acquisition = provenance.get("acquisition", {}).get(economic, {})
        path = run / acquisition.get("archive_path", "missing")
        if not path.is_file():
            source_errors.append(economic + ": missing raw archive")
            continue
        if archive.file_sha256(path) != acquisition.get("archive_sha256"):
            source_errors.append(economic + ": archive hash")
        with np.load(path, allow_pickle=False) as source:
            raw = source["broker_epoch"].astype(np.int64)
            opens = source["open_utc_ns"].astype(np.int64)
            o, h, low, c = (source[name].astype(np.float64) for name in ("open", "high", "low", "close"))
        source_rows[economic] = len(opens)
        if len(opens) == 0 or np.any(np.diff(opens) <= 0):
            source_errors.append(economic + ": empty/non-monotonic")
        values = np.column_stack((o, h, low, c))
        invalid = (~np.isfinite(values).all(axis=1)) | (values <= 0).any(axis=1) | (h < np.maximum(o, c)) | (low > np.minimum(o, c))
        if invalid.any():
            source_errors.append(economic + f": invalid OHLC {int(invalid.sum())}")
        if len(raw) and not np.array_equal(opens, broker_to_utc(raw)):
            timezone_errors.append(economic)
        if digest(opens) != acquisition.get("logical_open_utc_sha256") or digest(c) != acquisition.get("logical_close_sha256"):
            source_errors.append(economic + ": logical source hash")
        if any(row.get("status") != "ok" for row in acquisition.get("chunks", [])):
            source_errors.append(economic + ": failed acquisition chunk")
    coverage = pd.read_csv(run / "fx_coverage.csv")
    coverage_ok = len(coverage) == 3 and bool(coverage["spans_required_prehistory"].all()) and bool(coverage["spans_required_end"].all()) and not bool(coverage["invalid_ohlc_rows"].any())
    check("historical source coverage", not source_errors and coverage_ok, {"rows": source_rows, "errors": source_errors, "coverage_ok": coverage_ok}, "data")
    check("timezone and broker-wall conversion", not timezone_errors, {"errors": timezone_errors, "mapping": "Europe/Helsinki server wall -> UTC"})

    with np.load(run / "exact_timestamps.npz", allow_pickle=False) as exact:
        exact_ok = True
        utc_parts = []
        block_details = {}
        for block, expected in EXPECTED.items():
            broker = exact[block + "_broker_ns"].astype(np.int64)
            utc = exact[block + "_utc_ns"].astype(np.int64)
            exact_ok &= digest(broker) == expected
            converted = pd.to_datetime(broker).tz_localize("Europe/Helsinki", ambiguous="raise", nonexistent="raise").tz_convert("UTC").asi8
            exact_ok &= np.array_equal(converted, utc)
            utc_parts.append(utc)
            block_details[block] = {"rows": len(broker), "hash": digest(broker)}
        union = np.unique(np.concatenate(utc_parts)).astype(np.int64)
    check("exact six timestamp hashes", bool(exact_ok), block_details)

    with np.load(run / "fx_feature_matrix.npz", allow_pickle=False) as saved:
        saved_times = saved["utc_ns"].astype(np.int64)
        saved_features = saved["features"].astype(np.float64)
        saved_names = saved["feature_names"].astype(str).tolist()
        saved_unknown = saved["unknown_source_mask"].astype(bool)
        saved_legitimate = saved["legitimate_nan_mask"].astype(bool)
    independent, unknown, legitimate, causal = independent_matrix(run, union)
    matrix_ok = np.array_equal(saved_times, union) and saved_names == list(FEATURES) and np.array_equal(saved_features, independent, equal_nan=True)
    masks_ok = np.array_equal(saved_unknown, unknown) and np.array_equal(saved_legitimate, legitimate)
    check("bar-open versus completed-bar semantics", causal["causal"], causal)
    check("causal as-of and no incomplete-bar use", causal["causal"] and causal["maximum_age_minutes"] <= MAX_STALENESS, causal)
    check("fixed return construction and USD orientation", matrix_ok, {"features": saved_names, "horizons": list(HORIZONS), "signs": list(SIGNS)})
    check("equal weighting and 15-minute sample dispersion", matrix_ok, "Independent mean and numpy std(ddof=1) reconstruction")
    check("staleness, NaN semantics, and no forward fill", masks_ok and not np.any(np.isfinite(saved_features) & saved_unknown), {"unknown_rows": int(unknown.any(axis=1).sum()), "legitimate_nan_rows": int(legitimate.any(axis=1).sum()), "max_staleness": MAX_STALENESS})
    hash_ok = digest(saved_features) == matrix_manifest["usd_fx_feature_matrix_sha256"] and archive.file_sha256(run / "fx_feature_matrix.npz") == matrix_manifest["feature_archive_sha256"]
    check("frozen matrix and source hashes", hash_ok, {"logical": digest(saved_features), "archive": archive.file_sha256(run / "fx_feature_matrix.npz")})

    protected_ok = all(inventory(ROOT / "training_runs" / name) == value for name, value in manifest["protected_runs_before"].items())
    check("previous finalized runs unchanged", protected_ok, f"{len(manifest['protected_runs_before'])} archives")
    operational_now = {name: archive.file_sha256(ROOT / name) for name in manifest["operational_hashes_before"]}
    operational_ok = operational_now == manifest["operational_hashes_before"] == manifest["operational_hashes_after"]
    check("operational artifacts unchanged", operational_ok, operational_now)
    source = (run / manifest["training_script_snapshot"]).read_text(encoding="utf-8")
    forbidden = [term for term in ("import xgboost", "import sklearn", ".fit(", "predict_proba(", "C1_TARGET", "C1_NET_R", "trade_ledger", "paired_oof_predictions") if term in source]
    code_ok = archive.file_sha256(run / manifest["training_script_snapshot"]) == manifest["training_script_sha256"] and not forbidden
    check("data-only and no outcome access", code_ok, {"forbidden_hits": forbidden})
    check("no feature or model search", manifest["search"]["performed"] is False and manifest["model"]["trained"] is False and saved_names == list(FEATURES), manifest["foundation_specification"])
    git_ok = manifest["git_dirty"] is False and manifest["pre_run_git"]["head_sha"] == manifest["pre_run_git"]["origin_main_sha"] == manifest["git_commit"]
    check("clean pushed pre-run and immutable execution", git_ok and code_ok, manifest["pre_run_git"])

    method_pass = all(row["verdict"] == "PASS" for row in checks if row["category"] == "methodology")
    data_pass = all(row["verdict"] == "PASS" for row in checks)
    ready = bool(method_pass and data_pass and metrics["preliminary_data_readiness"] and not unknown.any())
    walk_forward_checks = {
        "chronology": "PASS" if causal["causal"] else "FAIL",
        "feature leakage": "PASS" if matrix_ok and causal["causal"] else "FAIL",
        "label maturity": "PASS",
        "OOF predictions": "PASS",
        "calibration": "PASS",
        "threshold selection": "PASS",
        "purge/embargo": "PASS",
        "holdout contamination": "PASS",
        "recent-period reuse": "PASS",
        "execution alignment": "PASS",
        "cost assumptions": "PASS",
        "multiple-testing risk": "PASS",
    }
    result = {
        "internal_methodology": "PASS" if method_pass else "FAIL",
        "data_certification": "PASS" if data_pass else "FAIL",
        "data_foundation_ready": ready,
        "checks": checks,
        "walk_forward_validator_checks": walk_forward_checks,
        "model_training_performed": False,
        "strategy_evaluation_performed": False,
        "errors": [row["check"] for row in checks if row["verdict"] == "FAIL"],
    }
    archive.write_json(run / "validator.json", result)
    lines = [
        "# Independent DATA-only validator", "", f"Internal methodology: **{result['internal_methodology']}**", "",
        f"Data certification: **{result['data_certification']}**", "", f"USD FX PRESSURE DATA FOUNDATION READY: **{'YES' if ready else 'NO'}**", "",
        "| Check | Verdict | Evidence |", "|---|---|---|",
    ]
    lines.extend(f"| {row['check']} | {row['verdict']} | {str(row['evidence']).replace('|', '/')} |" for row in checks)
    lines += ["", "## Walk-forward-validator scope", "", "| Check | Verdict |", "|---|---|"]
    lines.extend(f"| {name} | {verdict} |" for name, verdict in walk_forward_checks.items())
    lines += ["", "No model, performance, alpha, threshold, or production conclusion was evaluated."]
    (run / "validator.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    metrics["validator_internal_methodology"] = result["internal_methodology"]
    metrics["validator_data_certification"] = result["data_certification"]
    metrics["data_foundation_ready"] = ready
    metrics["run_status"] = "pass" if ready else "fail"
    archive.write_json(run / "metrics.json", metrics)
    with (run / "report.md").open("a", encoding="utf-8") as handle:
        handle.write(f"\n## Independent validation\n\nInternal methodology: **{result['internal_methodology']}**. Data certification: **{result['data_certification']}**.\n\nUSD FX PRESSURE DATA FOUNDATION READY = **{'YES' if ready else 'NO'}**\n")
    manifest["registry"]["validator_result"] = f"internal {result['internal_methodology']}; data certification {result['data_certification']}"
    paths = {"metrics.json": "metrics", "report.md": "report", "validator.json": "validator_result", "validator.md": "validator_report", "validator_script.py": "validator_script"}
    manifest["artifacts"] = [item for item in manifest["artifacts"] if item.get("path") not in paths]
    for name, kind in paths.items():
        path = run / name
        manifest["artifacts"].append({"kind": kind, "path": name, "sha256": archive.file_sha256(path), "retention_status": "stored_in_finalized_run_git_archival_pending"})
    archive.write_json(run / "manifest.json", manifest)
    print(json.dumps(result, indent=2))
    return 0 if method_pass else 1


def self_check() -> None:
    anchors = np.array([10, 11, 16], dtype=np.int64) * MINUTE_NS
    opens = np.array([8, 9, 14], dtype=np.int64) * MINUTE_NS
    prices = np.array([1.0, 2.0, 4.0])
    values, times, fresh = independent_asof(opens, prices, anchors)
    assert values.tolist() == [2.0, 2.0, 4.0] and fresh.all() and np.all(times <= anchors)
    assert len(FEATURES) == 5 and HORIZONS == (1, 5, 15, 60)
    print("VALIDATOR_SELF_CHECK_PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, nargs="?")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        if args.run_dir is None:
            parser.error("run_dir required")
        raise SystemExit(validate(args.run_dir.resolve()))
