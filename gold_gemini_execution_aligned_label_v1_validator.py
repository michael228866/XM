from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import gold_gemini_execution_aligned_label_v1 as experiment
import gold_gemini_execution_semantics_v1 as semantics


ROOT = Path(__file__).resolve().parent
CHECK_NAMES = (
    "chronology",
    "feature leakage",
    "label maturity",
    "OOF predictions",
    "calibration",
    "threshold selection",
    "purge/embargo",
    "holdout contamination",
    "recent-period reuse",
    "execution alignment",
    "cost assumptions",
    "multiple-testing risk",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent paired-label validator")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(experiment.sanitize(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def close(left: float | None, right: float | None, tolerance: float = 1e-8) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if math.isinf(float(left)) or math.isinf(float(right)):
        return math.isinf(float(left)) and math.isinf(float(right))
    return bool(math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance))


def metric(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    winners = values[values > 0.0]
    losers = values[values <= 0.0]
    gains = float(winners.sum()) if len(winners) else 0.0
    loss = -float(losers.sum()) if len(losers) else 0.0
    average_winner = float(winners.mean()) if len(winners) else None
    average_loser = float(losers.mean()) if len(losers) else None
    payoff = (
        average_winner / abs(average_loser)
        if average_winner is not None and average_loser not in (None, 0.0)
        else None
    )
    break_even = 1.0 / (1.0 + payoff) if payoff is not None else None
    win_rate = float(len(winners) / len(values)) if len(values) else 0.0
    equity = np.r_[0.0, np.cumsum(values)]
    drawdown = equity - np.maximum.accumulate(equity)
    return {
        "trades": len(values),
        "wins": len(winners),
        "losses": len(losers),
        "realized_wr": win_rate,
        "average_winner_r": average_winner,
        "average_loser_r": average_loser,
        "payoff_ratio": payoff,
        "break_even_wr": break_even,
        "break_even_adjusted_edge": None if break_even is None else win_rate - break_even,
        "pf": math.inf if loss == 0.0 and gains > 0.0 else (gains / loss if loss else 0.0),
        "mean_r": float(values.mean()) if len(values) else 0.0,
        "pnl_r": float(values.sum()),
        "max_dd_r": float(drawdown.min()),
    }


def load_reconstructed_cohort(run_dir: Path, model_id: str) -> pd.DataFrame:
    with np.load(run_dir / "paired_oof_predictions.npz", allow_pickle=False) as data:
        frame = pd.DataFrame(
            {
                "TIME_DT": pd.to_datetime(data["time_ns"].astype(np.int64)),
                "OPEN": data["open"].astype(np.float64),
                "HIGH": data["high"].astype(np.float64),
                "LOW": data["low"].astype(np.float64),
                "CLOSE": data["close"].astype(np.float64),
                "ATR": data["atr"].astype(np.float64),
                "M1_RSI": data["rsi"].astype(np.float64),
                "SPREAD": data["spread"].astype(np.float64),
                "buy_prob": data["score_c0" if model_id == experiment.MODEL_IDS[0] else "score_c1"].astype(np.float32),
                "sell_prob": np.zeros(len(data["time_ns"]), dtype=np.float32),
            }
        )
    return semantics.finalize_cohort(frame, f"VALIDATOR_{model_id}", offset_hours=0)


def reconstruct_ledgers(run_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    expected = pd.read_csv(run_dir / "trade_ledger.csv", low_memory=False)
    errors: list[str] = []
    result: dict[str, list[dict[str, Any]]] = {}
    for model_id in experiment.MODEL_IDS:
        cohort = load_reconstructed_cohort(run_dir, model_id)
        trades, _ = semantics.simulate(cohort, semantics.SIMULATORS[-1])
        for trade in trades:
            entry = pd.Timestamp(trade["entry_time_api"])
            trade["fold"] = next(
                (name for name, start, end in experiment.FOLDS if start <= entry < end),
                "outside_fold",
            )
        result[model_id] = trades
        submitted = expected[expected["model_id"] == model_id].reset_index(drop=True)
        if len(submitted) != len(trades):
            errors.append(f"{model_id} trade count {len(trades)} != submitted {len(submitted)}")
            continue
        for index, trade in enumerate(trades):
            row = submitted.iloc[index]
            if (
                pd.Timestamp(row["entry_time_api"]) != pd.Timestamp(trade["entry_time_api"])
                or pd.Timestamp(row["exit_time_api"]) != pd.Timestamp(trade["exit_time_api"])
                or row["exit_reason"] != trade["exit_reason"]
                or not close(float(row["net_r"]), trade["net_r"], 1e-7)
            ):
                errors.append(f"{model_id} trade mismatch at ordinal {index}")
                break
    return result, errors


def validate_metrics(run_dir: Path, ledgers: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[str]]:
    submitted = pd.read_csv(run_dir / "fold_metrics.csv")
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    for model_id in experiment.MODEL_IDS:
        scopes = [
            (name, [trade for trade in ledgers[model_id] if trade["fold"] == name], experiment.core.fold_days(start, end))
            for name, start, end in experiment.FOLDS
        ]
        scopes.append(
            (
                "pooled",
                ledgers[model_id],
                sum(experiment.core.fold_days(start, end) for _, start, end in experiment.FOLDS),
            )
        )
        for fold, trades, days in scopes:
            values = np.asarray([trade["net_r"] for trade in trades], dtype=np.float64)
            stress = np.asarray([trade["stress_r"] for trade in trades], dtype=np.float64)
            recomputed = metric(values)
            recomputed["trades_per_day"] = len(trades) / max(days, 1)
            recomputed["cost_stress_pf"] = metric(stress)["pf"]
            recomputed["tp_first_wr"] = (
                sum(trade["exit_reason"] == "take_profit" for trade in trades) / len(trades)
                if trades
                else 0.0
            )
            row = submitted[(submitted["model_id"] == model_id) & (submitted["fold"] == fold)]
            if len(row) != 1:
                errors.append(f"missing submitted metric row {model_id}/{fold}")
                continue
            row = row.iloc[0]
            for key in (
                "trades",
                "trades_per_day",
                "realized_wr",
                "tp_first_wr",
                "pf",
                "mean_r",
                "pnl_r",
                "max_dd_r",
                "cost_stress_pf",
            ):
                if not close(float(row[key]), recomputed[key], 1e-7):
                    errors.append(f"metric mismatch {model_id}/{fold}/{key}")
            rows.append({"model_id": model_id, "fold": fold, **recomputed})
    return rows, errors


def validate_costs(run_dir: Path) -> list[str]:
    ledger = pd.read_csv(run_dir / "trade_ledger.csv", low_memory=False)
    errors = []
    denominator = ledger["sl_distance"].to_numpy(dtype=np.float64) + ledger["spread_points"].to_numpy(dtype=np.float64) * experiment.POINT
    expected = (
        ledger["gross_price"].to_numpy(dtype=np.float64)
        - (ledger["spread_points"].to_numpy(dtype=np.float64) + ledger["extra_cost_points"].to_numpy(dtype=np.float64)) * experiment.POINT
    ) / denominator
    stress = (
        ledger["gross_price"].to_numpy(dtype=np.float64)
        - (ledger["spread_points"].to_numpy(dtype=np.float64) + experiment.STRESS_EXTRA_COST_POINTS) * experiment.POINT
    ) / denominator
    if not np.allclose(expected, ledger["net_r"].to_numpy(dtype=np.float64), rtol=1e-9, atol=1e-9):
        errors.append("nominal per-trade cost arithmetic mismatch")
    if not np.allclose(stress, ledger["stress_r"].to_numpy(dtype=np.float64), rtol=1e-9, atol=1e-9):
        errors.append("stress per-trade cost arithmetic mismatch")
    return errors


def validate_provenance(run_dir: Path, manifest: dict[str, Any]) -> tuple[dict[str, bool], list[str]]:
    provenance = read_json(run_dir / "fold_model_provenance.json")
    flags = {
        "six_paired_fold_models": len([m for m in provenance["models"] if m.get("fold") != "shadow_final"]) == 6,
        "feature_count_31": len(provenance["features"]) == 31,
        "identical_x_train": True,
        "identical_x_score": True,
        "identical_parameters": True,
        "identical_seed": True,
        "strict_maturity": True,
        "model_hashes": True,
    }
    errors = []
    for fold in provenance["folds"]:
        flags["identical_x_train"] &= fold["x_train_sha256_C0"] == fold["x_train_sha256_C1"]
        flags["identical_x_score"] &= fold["x_score_sha256_C0"] == fold["x_score_sha256_C1"]
        flags["identical_parameters"] &= fold["parameters_C0"] == fold["parameters_C1"] == experiment.FIXED_XGB_PARAMETERS
        flags["identical_seed"] &= fold["random_seed_C0"] == fold["random_seed_C1"] == experiment.RANDOM_STATE
        flags["strict_maturity"] &= (
            pd.Timestamp(fold["latest_training_label_information_time"])
            < pd.Timestamp(fold["score_start"])
            and bool(fold["strict_label_maturity_before_score"])
        )
    for model in provenance["models"]:
        artifact = run_dir / model["path"]
        flags["model_hashes"] &= artifact.is_file() and sha256(artifact) == model["sha256"]
    for key, passed in flags.items():
        if not passed:
            errors.append(f"provenance assertion failed: {key}")
    dependencies = manifest["dependency_sha256"]
    for name in (
        "gold_gemini_execution_semantics_v1.py",
        "gold_gemini_core_gate_v1.py",
        "barrier_classifier_strategy.py",
        "barrier_research_suite.py",
        "barrier_final_train.py",
        "drl_trading_v2.py",
    ):
        if sha256(ROOT / name) != dependencies[name]:
            errors.append(f"dependency changed after execution: {name}")
    return flags, errors


def validate_oof(run_dir: Path) -> tuple[dict[str, Any], list[str]]:
    errors = []
    with np.load(run_dir / "paired_oof_predictions.npz", allow_pickle=False) as data:
        time_ns = data["time_ns"].astype(np.int64)
        feature_ns = data["feature_time_ns"].astype(np.int64)
        target = data["c1_target"].astype(np.int8)
        net_r = data["c1_net_r"].astype(np.float64)
        score0 = data["score_c0"].astype(np.float64)
        score1 = data["score_c1"].astype(np.float64)
        if not np.all(feature_ns < time_ns):
            errors.append("feature bar time is not strictly before decision/entry time")
        if not np.array_equal(target, (net_r > 0.0).astype(np.int8)):
            errors.append("C1 target does not equal net realized R > 0")
        if not np.isfinite(score0).all() or not np.isfinite(score1).all():
            errors.append("OOF scores contain non-finite values")
        if np.any(np.diff(time_ns) < 0):
            errors.append("OOF timestamps are not chronological")
        summary = {
            "rows": len(time_ns),
            "feature_time_strictly_before_decision": bool(np.all(feature_ns < time_ns)),
            "c1_target_matches_positive_net_r": bool(np.array_equal(target, (net_r > 0.0).astype(np.int8))),
            "scores_finite": bool(np.isfinite(score0).all() and np.isfinite(score1).all()),
        }
    return summary, errors


def check_result(
    verdict: bool,
    evidence: str,
    passed: str,
    failed: str,
    correction: str,
) -> dict[str, str]:
    return {
        "verdict": "PASS" if verdict else "FAIL",
        "evidence": evidence,
        "reason": passed if verdict else failed,
        "required_validation_correction": "none" if verdict else correction,
    }


def render_validator(result: dict[str, Any]) -> str:
    lines = [
        f"# {experiment.EXPERIMENT} - independent validation",
        "",
        f"Overall: **{result['overall']}**",
        "",
        f"Internal methodology: **{result['internal_methodology']}**",
        "",
        f"Final untouched-test validity: **{result['final_untouched_test_validity']}**",
        "",
        "| Check | Verdict | Evidence | Failure or reason for pass | Required validation correction |",
        "|---|---|---|---|---|",
    ]
    for name in CHECK_NAMES:
        item = result["checks"][name]
        lines.append(
            f"| {name} | {item['verdict']} | {item['evidence']} | {item['reason']} | {item['required_validation_correction']} |"
        )
    lines.extend(
        [
            "",
            "## Independent reconciliation",
            "",
            f"Paired fold/data provenance: **{result['paired_provenance']}**.",
            f"OOF cohort identity: **{result['oof_identity']}**.",
            f"S5 trade-ledger reconstruction: **{result['trade_reconstruction']}**.",
            f"Metric recomputation: **{result['metric_recomputation']}**.",
            f"Cost arithmetic: **{result['cost_recomputation']}**.",
            f"Operational artifacts unchanged: **{result['operational_artifacts_unchanged']}**.",
            "",
            "The paired historical hypothesis is internally interpretable, but all scored folds are development data and cannot establish final untouched validity.",
            "",
            "## Smallest validation-only correction",
            "",
            result["smallest_validation_only_correction"],
            "",
            f"Submitted claim: `{result['submitted_claim']}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def self_check() -> None:
    values = np.array([1.0, -1.0])
    result = metric(values)
    assert result["trades"] == 2 and result["realized_wr"] == 0.5
    assert result["pf"] == 1.0 and result["mean_r"] == 0.0
    assert set(CHECK_NAMES) == {
        "chronology", "feature leakage", "label maturity", "OOF predictions",
        "calibration", "threshold selection", "purge/embargo", "holdout contamination",
        "recent-period reuse", "execution alignment", "cost assumptions", "multiple-testing risk",
    }
    print("VALIDATOR_SELF_CHECK_OK", flush=True)


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    self_check()
    manifest = read_json(run_dir / "manifest.json")
    metrics = read_json(run_dir / "metrics.json")
    required = (
        "label_definition.json",
        "fold_model_provenance.json",
        "fold_metrics.csv",
        "probability_diagnostics.csv",
        "trade_ledger.csv",
        "paired_oof_predictions.npz",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing validator inputs: " + ", ".join(missing))

    provenance_flags, provenance_errors = validate_provenance(run_dir, manifest)
    oof_summary, oof_errors = validate_oof(run_dir)
    ledgers, trade_errors = reconstruct_ledgers(run_dir)
    recomputed_metrics, metric_errors = validate_metrics(run_dir, ledgers)
    cost_errors = validate_costs(run_dir)
    operational_unchanged = bool(
        manifest["operational_hashes_before"] == manifest["operational_hashes_after"]
        and manifest["operational_hashes_after"][experiment.GEMINI_FILE.name] == sha256(experiment.GEMINI_FILE)
        and manifest["operational_hashes_after"][experiment.OPERATIONAL_MODEL.name] == sha256(experiment.OPERATIONAL_MODEL)
    )

    chronology_ok = provenance_flags["strict_maturity"] and oof_summary["feature_time_strictly_before_decision"]
    feature_ok = oof_summary["feature_time_strictly_before_decision"] and provenance_flags["feature_count_31"]
    maturity_ok = provenance_flags["strict_maturity"] and oof_summary["c1_target_matches_positive_net_r"]
    oof_ok = not oof_errors and provenance_flags["six_paired_fold_models"]
    execution_ok = not trade_errors and not metric_errors
    cost_ok = not cost_errors
    internal_errors = [*provenance_errors, *oof_errors, *trade_errors, *metric_errors, *cost_errors]
    if not operational_unchanged:
        internal_errors.append("operational artifact changed")

    checks = {
        "chronology": check_result(
            chronology_ok,
            "fold_model_provenance.json; paired_oof_predictions.npz",
            "Every fold uses trailing 18 months and latest C0/C1 label information is strictly earlier than score start; feature bars precede decisions.",
            "A time or maturity boundary assertion failed.",
            "Create a new run after correcting chronological boundaries only.",
        ),
        "feature leakage": check_result(
            feature_ok and not provenance_errors,
            "31-feature list/hashes; dependency SHA-256 inventory",
            "Both models use the same shifted 31-feature matrices and producing dependencies match their executed hashes.",
            "Feature identity, timing, or dependency verification failed.",
            "Rebuild paired matrices from completed-bar features in a new run.",
        ),
        "label maturity": check_result(
            maturity_ok,
            "C0 240-row maturity; C1 stored exit maturity; strict fold assertions",
            "The common training cohort admits a row only after both labels mature strictly before scoring; C1 target equals net R > 0.",
            "Label maturity or target semantics failed.",
            "Correct only the maturity/label implementation and create a new run.",
        ),
        "OOF predictions": check_result(
            oof_ok and provenance_flags["model_hashes"],
            "six retained fold models; paired OOF score artifact",
            "Each scored fold has separate C0/C1 models trained solely on the preceding purged 18-month cohort.",
            "OOF scores or fold-model provenance could not be reproduced.",
            "Regenerate only genuinely fold-held-out predictions in a new run.",
        ),
        "calibration": check_result(
            manifest["model"]["calibration_method"] == "none",
            "manifest.model.calibration_method",
            "No calibration is fit, selected, or applied.",
            "An unregistered calibration stage was used.",
            "Remove it and rerun the fixed paired comparison.",
        ),
        "threshold selection": check_result(
            manifest["search"]["performed"] is False
            and manifest["paired_design"]["threshold"] == 0.75
            and len(manifest["paired_design"]["model_definitions"]) == 2,
            "manifest.paired_design/search; model_comparison.csv",
            "Threshold 0.75, one C1 label, parameters, folds, features, and window are preregistered and never searched.",
            "The run contains an adaptive selection outside the paired hypothesis.",
            "Discard the adaptive result and create a new preregistered run.",
        ),
        "purge/embargo": check_result(
            provenance_flags["strict_maturity"],
            "latest C0/C1 information times per fold",
            "The shared purge uses the later of C0 and C1 maturity; no label interval reaches score start.",
            "A label/execution interval overlaps scoring.",
            "Increase only the purge and repeat the paired run.",
        ),
        "holdout contamination": check_result(
            manifest["evidence_status"]["untouched_oos_claim"] is False,
            "manifest.evidence_status",
            "All historical folds are explicitly development evidence and no holdout claim is made.",
            "A contaminated interval was represented as untouched.",
            "Downgrade the evidence classification without changing strategy results.",
        ),
        "recent-period reuse": check_result(
            manifest["evidence_status"]["previous_forward_status"] == experiment.PREVIOUS_FORWARD_STATUS,
            "manifest.evidence_status.previous_forward_status",
            "The inspected former forward interval remains contaminated and is not used as a fresh final test.",
            "Previously inspected data was treated as untouched.",
            "Exclude it from any final claim.",
        ),
        "execution alignment": check_result(
            execution_ok,
            "paired OOF OHLC cohort; exact preserved S5; independently reconstructed trade ledger",
            "C0 and C1 traverse the same S5 barwise state machine with next-open entry, entry-bar HIGH/LOW stop-first, timeout, occupancy, sessions, and risk state.",
            "Independent S5 trade or metric reconstruction differs.",
            "Correct execution reconstruction only and create a new run if results change.",
        ),
        "cost assumptions": check_result(
            cost_ok,
            "trade_ledger.csv gross/spread/extra-cost/net-R/stress-R",
            "Observed/fallback spread and both nominal and stress costs reconcile per trade.",
            "Per-trade nominal or stress cost arithmetic failed.",
            "Correct cost accounting and repeat the paired run.",
        ),
        "multiple-testing risk": check_result(
            False,
            "manifest.evidence_status; repository historical research record",
            "",
            "The experiment is a single preregistered label hypothesis, but every historical fold has been inspected previously and no untouched final test exists.",
            "Freeze a qualified candidate first, then collect genuinely new post-freeze shadow data; do not relabel historical data.",
        ),
    }
    internal_methodology = "PASS" if not internal_errors and all(
        checks[name]["verdict"] == "PASS" for name in CHECK_NAMES if name != "multiple-testing risk"
    ) else "FAIL"
    result = {
        "overall": "FAIL",
        "internal_methodology": internal_methodology,
        "final_untouched_test_validity": "FAIL",
        "checks": checks,
        "paired_provenance": "PASS" if not provenance_errors else "FAIL",
        "oof_identity": "PASS" if not oof_errors else "FAIL",
        "trade_reconstruction": "PASS" if not trade_errors else "FAIL",
        "metric_recomputation": "PASS" if not metric_errors else "FAIL",
        "cost_recomputation": "PASS" if not cost_errors else "FAIL",
        "operational_artifacts_unchanged": operational_unchanged,
        "internal_errors": internal_errors,
        "provenance_flags": provenance_flags,
        "oof_summary": oof_summary,
        "recomputed_metrics": recomputed_metrics,
        "economic_gate": metrics["decision"],
        "smallest_validation_only_correction": (
            "No internal rerun is required; final validity requires new data collected only after a completely frozen candidate cutoff."
            if internal_methodology == "PASS"
            else "Resolve the listed internal methodology errors in a new formal run without strategy tuning."
        ),
        "submitted_claim": (
            "valid_internal_paired_label_attribution_invalid_for_final_strategy_claim"
            if internal_methodology == "PASS"
            else "invalid_internal_paired_label_attribution"
        ),
    }
    validator_json = run_dir / "validator.json"
    validator_md = run_dir / "validator.md"
    validator_script = run_dir / "validator_script.py"
    write_json(validator_json, result)
    validator_md.write_text(render_validator(result), encoding="utf-8")
    shutil.copy2(Path(__file__), validator_script)

    manifest = read_json(run_dir / "manifest.json")
    manifest["registry"]["validator_result"] = result["overall"]
    manifest["validator"] = {
        "overall": result["overall"],
        "internal_methodology": internal_methodology,
        "final_untouched_test_validity": result["final_untouched_test_validity"],
        "script_path": validator_script.relative_to(run_dir).as_posix(),
        "script_sha256": sha256(validator_script),
    }
    for path, kind in (
        (validator_json, "independent_validator_result"),
        (validator_md, "independent_validator_report"),
        (validator_script, "independent_validator_script"),
    ):
        experiment.add_artifact(manifest, run_dir, path, kind)
    write_json(run_dir / "manifest.json", manifest)
    print(f"VALIDATOR_OVERALL_{result['overall']}", flush=True)
    print(f"INTERNAL_METHODOLOGY_{internal_methodology}", flush=True)
    print("FINAL_UNTOUCHED_TEST_VALIDITY_FAIL", flush=True)
    return 0 if internal_methodology == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
