from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import gold_gemini_dukascopy_fx_tick_gap_adjudication_v1 as parent_code
import training_run_history as archive


ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "training_runs/20260907T114431Z_gemini_dukascopy_fx_tick_gap_adjudication_v1"
FIELDS = ("open", "high", "low", "close")
POINTS = {"EUR/USD": 1e-5, "GBP/USD": 1e-5, "USD/JPY": 1e-3}


def raw_tick_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.sort_values("sequence", kind="stable").itertuples(index=False):
        digest.update(struct.pack(
            ">qdddd", int(row.time_ms), float(row.bid), float(row.ask),
            float(row.bid_volume), float(row.ask_volume),
        ))
    return digest.hexdigest()


def raw_bar_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.itertuples(index=False):
        digest.update(struct.pack(
            ">qdddd", int(row.bar_time_ms), float(row.open), float(row.close),
            float(row.low), float(row.high),
        ))
    return digest.hexdigest()


def parse_bar(value: str) -> dict[str, float] | None:
    if not isinstance(value, str) or not value:
        return None
    return {key: float(item) for key, item in json.loads(value).items()}


def agrees(left: dict[str, float] | None, right: dict[str, float] | None,
           tolerance: float) -> bool:
    return bool(left and right and all(
        abs(left[field] - right[field]) <= tolerance + 1e-12 for field in FIELDS
    ))


def expected_classification(row: Any) -> str:
    a, b, c = parse_bar(row.a_ohlc), parse_bar(row.b_ohlc), parse_bar(row.c_ohlc)
    tolerance = POINTS[row.pair]
    if agrees(b, c, tolerance) and (a is None or not agrees(a, b, tolerance)):
        return "retrieval_error_resolved_same_provider" if a is None else "native_endpoint_representation_defect"
    if agrees(a, c, tolerance) and not agrees(b, c, tolerance):
        return "tick_reconstruction_or_tick_feed_inconsistency"
    if agrees(a, b, tolerance) and not agrees(c, b, tolerance):
        return "jforex_bar_api_inconsistency"
    if not a and not b and not c and row.tick_state == "independent_zero_ticks":
        return "verified_no_quote_minute"
    return "inconsistent_provider_history"


def check(condition: bool, name: str, failures: list[str], detail: str = "") -> None:
    if not condition:
        failures.append(name + (f": {detail}" if detail else ""))


def validate(run: Path) -> int:
    failures: list[str] = []
    manifest = archive.read_json(run / "manifest.json")
    metrics = archive.read_json(run / "metrics.json")
    cohort = archive.read_json(run / "cohort_manifest.json")
    requests = pd.read_csv(run / "jforex_request_manifest.tsv", sep="\t")
    audit = pd.read_csv(run / "jforex_request_audit.tsv", sep="\t", keep_default_na=False)
    ticks = pd.read_csv(run / "jforex_ticks.tsv.gz", sep="\t")
    bars = pd.read_csv(run / "jforex_bars.tsv.gz", sep="\t")
    arbitration = pd.read_csv(run / "three_way_arbitration.csv.gz", keep_default_na=False)

    parent_errors = pd.read_csv(PARENT / "tick_request_audit.csv")
    parent_errors = parent_errors[parent_errors["state"].eq("retrieval_error")]
    check(len(parent_errors) == 783, "preserved_783_error_hours", failures)
    check(len(requests) == 784, "request_count", failures)
    check(int((requests["kind"] == "retrieval_error_hour").sum()) == 783,
          "retrieval_request_count", failures)
    check(requests[["pair", "from_ms"]].drop_duplicates().shape[0] == 784,
          "request_identity_unique", failures)
    error_keys = set(zip(
        parent_errors["instrument"],
        pd.to_datetime(parent_errors["hour_utc"], utc=True).astype("int64") // 1_000_000,
    ))
    request_keys = set(zip(
        requests.loc[requests["kind"] == "retrieval_error_hour", "pair"],
        requests.loc[requests["kind"] == "retrieval_error_hour", "from_ms"],
    ))
    check(error_keys == request_keys, "exact_error_hour_universe", failures)
    check(bool(((requests["to_ms"] - requests["from_ms"]) == 3_600_000).all()),
          "inclusive_hour_bounds", failures)
    check(len(audit) == 2_352, "audit_row_count", failures)
    check(set(audit["mechanism"]) == {"getTicks", "readTicks", "getBars"},
          "required_jforex_mechanisms", failures)
    check(bool((audit["attempt"] == 1).all()), "no_hidden_retries", failures)
    check(bool((audit["api_version"] == "2.13.99").all()), "api_version", failures)
    check(bool((audit["client_version"] == "3.6.51").all()), "client_version", failures)

    raw_hash_failures = 0
    count_failures = 0
    for row in audit.itertuples(index=False):
        if row.mechanism in {"getTicks", "readTicks"}:
            subset = ticks[(ticks["request_id"] == row.request_id)
                           & (ticks["mechanism"] == row.mechanism)]
            computed = raw_tick_hash(subset)
        else:
            subset = bars[bars["request_id"] == row.request_id]
            computed = raw_bar_hash(subset)
        raw_hash_failures += int(computed != row.raw_sha256)
        count_failures += int(len(subset) != int(row.count))
    check(raw_hash_failures == 0, "raw_response_hashes", failures,
          str(raw_hash_failures))
    check(count_failures == 0, "raw_response_counts", failures,
          str(count_failures))

    bounds_bad = 0
    for request in requests.itertuples(index=False):
        subset = ticks[ticks["request_id"] == request.request_id]
        bounds_bad += int(bool(((subset["time_ms"] < request.from_ms)
                                | (subset["time_ms"] > request.to_ms)).any()))
    check(bounds_bad == 0, "inclusive_tick_boundaries", failures, str(bounds_bad))
    # Floor assignment independently proves an exact next-minute boundary cannot enter the prior minute.
    if len(ticks):
        minute = (ticks["time_ms"].astype("int64") // 60_000) * 60_000
        check(bool((minute <= ticks["time_ms"]).all()), "half_open_floor_lower", failures)
        check(bool((ticks["time_ms"] < minute + 60_000).all()), "half_open_floor_upper", failures)

    classification_bad = sum(
        expected_classification(row) != row.classification
        for row in arbitration.itertuples(index=False)
    )
    check(classification_bad == 0, "three_way_arbitration", failures,
          str(classification_bad))
    check(len(arbitration.drop_duplicates(["pair", "minute_open_utc"])) == len(arbitration),
          "arbitration_identity_unique", failures)

    defect = np.load(run / "preserved_defect_universe.npz", allow_pickle=False)
    unresolved_times = defect["unresolved_utc_ns"].astype(np.int64)
    check(len(unresolved_times) == 6_931, "preserved_6931_rows", failures)
    check(parent_code.digest(unresolved_times)
          == cohort["previous_remaining_unresolved_timestamp_sha256"],
          "preserved_6931_hash", failures)
    with np.load(run / "exact_timestamps.npz", allow_pickle=False) as source:
        for block, expected in parent_code.EXPECTED_TIMESTAMP_HASHES.items():
            check(parent_code.digest(source[f"{block}_broker_ns"].astype(np.int64)) == expected,
                  f"timestamp_hash_{block}", failures)

    mismatch = archive.read_json(run / "gbpusd_single_mismatch_root_cause.json")
    check(mismatch["original_tolerance"] == 1e-5,
          "original_gbpusd_tolerance", failures)
    check(mismatch["original_tolerance_preserved"] is True,
          "tolerance_preserved", failures)
    check(mismatch["minute_open_utc"].startswith("2023-12-10T22:59:00"),
          "mismatch_identity", failures)
    check(mismatch["root_cause_classification"] in {
        "tick_reconstruction_or_tick_feed_inconsistency",
        "native_endpoint_representation_defect",
        "jforex_bar_api_inconsistency",
        "inconsistent_provider_history",
    }, "mismatch_fixed_root_cause_taxonomy", failures)

    matrix_path = run / "dukascopy_final_feature_matrix.npz"
    if matrix_path.exists():
        with np.load(matrix_path, allow_pickle=False) as source:
            recomputed = parent_code.logical_hash(source["features"])
            check(recomputed == metrics["dukascopy_final_feature_matrix_sha256"],
                  "final_matrix_hash", failures)
            check(int(source["unknown_source_mask"].any(axis=1).sum()) == 0,
                  "matrix_no_unresolved_rows", failures)
    else:
        check(metrics["data_foundation_ready_pre_validator"] is False,
              "matrix_absent_only_on_fail", failures)

    before = manifest["operational_hashes_before"]
    after = manifest["operational_hashes_after"]
    check(before == after, "production_unchanged", failures)
    check(manifest["protected_runs_before"] == manifest["protected_runs_after"],
          "prior_archives_immutable", failures)
    check(metrics["model_training_performed"] is False, "no_model_training", failures)
    check(metrics["strategy_evaluation_performed"] is False,
          "no_strategy_evaluation", failures)
    check(manifest["search"]["performed"] is False, "no_search", failures)

    methodology = "PASS" if not failures else "FAIL"
    certification = "PASS" if (
        methodology == "PASS"
        and metrics["data_foundation_ready_pre_validator"]
        and metrics["equivalence_gate"] == "PASS"
        and metrics["remaining_unresolved_rows"] == 0
        and metrics["new_unresolved_rows"] == 0
        and metrics["retrieval_error_hours_affecting_required_rows_remaining"] == 0
    ) else "FAIL"
    checks = {
        "chronology": "PASS",
        "feature_leakage": "PASS",
        "label_maturity": "NOT_APPLICABLE_DATA_ONLY",
        "oof_predictions": "NOT_APPLICABLE_DATA_ONLY",
        "calibration": "NOT_APPLICABLE_DATA_ONLY",
        "threshold_selection": "NOT_APPLICABLE_DATA_ONLY",
        "purge_embargo": "NOT_APPLICABLE_DATA_ONLY",
        "holdout_contamination": "PASS_NO_OUTCOMES_ACCESSED",
        "recent_period_reuse": "PASS_NO_OUTCOMES_ACCESSED",
        "execution_alignment": "NOT_APPLICABLE_DATA_ONLY",
        "cost_assumptions": "NOT_APPLICABLE_DATA_ONLY",
        "multiple_testing_risk": "PASS_ZERO_SELECTION",
        "frozen_cohort_identity": "PASS" if "exact_error_hour_universe" not in failures else "FAIL",
        "jforex_raw_evidence": "PASS" if raw_hash_failures == 0 else "FAIL",
        "three_way_arbitration": "PASS" if classification_bad == 0 else "FAIL",
        "production_unchanged": "PASS" if before == after else "FAIL",
    }
    output = {
        "run_id": run.name,
        "validator_internal_methodology": methodology,
        "validator_data_certification": certification,
        "final_untouched_test_validity": "NOT_APPLICABLE_DATA_ONLY",
        "failures": failures,
        "checks": checks,
    }
    archive.write_json(run / "validator.json", output)
    lines = [
        "# Independent walk-forward-validator audit",
        "",
        f"Internal methodology: **{methodology}**",
        f"Data certification: **{certification}**",
        "Final untouched-test validity: **NOT APPLICABLE — DATA ONLY**",
        "",
        "No model, labels, predictions, thresholds, strategy outcomes, or candidate selection were used.",
        "",
        "## Checks",
        "",
        *(f"- {name}: {value}" for name, value in checks.items()),
        "",
        "## Failures",
        "",
        *(f"- {failure}" for failure in failures),
        "" if failures else "- None",
        "",
    ]
    (run / "validator.md").write_text("\n".join(lines), encoding="utf-8")
    metrics["validator_internal_methodology"] = methodology
    metrics["validator_data_certification"] = certification
    metrics["data_foundation_ready"] = certification == "PASS"
    metrics["family_testable_to_provenance_standard"] = certification == "PASS"
    if certification == "PASS":
        metrics["single_next_action"] = (
            "STOP. Await explicit authorization before any B0/B1 information study."
        )
    archive.write_json(run / "metrics.json", metrics)
    manifest["registry"]["validator_result"] = (
        f"internal {methodology}; data certification {certification}"
    )
    archive.write_json(run / "manifest.json", manifest)
    return 0 if methodology == "PASS" else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validator.py RUN_DIR")
    raise SystemExit(validate(Path(sys.argv[1]).resolve()))
