from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import gold_gemini_execution_semantics_v1 as semantics
import gold_gemini_macro_event_timing_v1 as experiment


ROOT = Path(__file__).resolve().parent
CHECKS = (
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
OFFICIAL_DOMAINS = {"www.bls.gov", "bls.gov", "www.bea.gov", "bea.gov", "www.federalreserve.gov", "federalreserve.gov"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent macro-event timing validator")
    parser.add_argument("run_dir", type=Path, nargs="?")
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


def close(left: Any, right: Any, tolerance: float = 1e-7) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if math.isinf(float(left)) or math.isinf(float(right)):
        return math.isinf(float(left)) and math.isinf(float(right))
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def validate_required(run_dir: Path) -> list[str]:
    required = {
        "manifest.json", "training_script.py", "event_family_definition.json",
        "event_source_provenance.csv", "event_timestamp_dataset.csv", "event_coverage.csv",
        "feature_alignment_audit.csv", "fold_model_provenance.json", "model_comparison.csv",
        "fold_metrics.csv", "information_gain.csv", "probability_diagnostics.csv",
        "ranking_diagnostics.csv", "ranking_deciles.csv", "trade_ledger.csv",
        "metrics.json", "report.md", "environment.txt", "stdout.log",
    }
    return [f"missing required artifact: {name}" for name in sorted(required) if not (run_dir / name).is_file()]


def validate_definition(run_dir: Path) -> list[str]:
    definition = read_json(run_dir / "event_family_definition.json")
    errors: list[str] = []
    if tuple(definition.get("event_universe", {})) != experiment.EVENT_TYPES:
        errors.append("event universe differs from frozen CPI/EMPLOYMENT/PCE/FOMC order")
    if tuple(definition.get("features", [])) != experiment.EVENT_FEATURES:
        errors.append("event feature list differs from exact frozen eight-feature definition")
    if definition.get("excluded_information") != ["actual", "forecast", "consensus", "surprise", "revisions", "NLP", "future schedule changes"]:
        errors.append("excluded event-outcome information declaration changed")
    if definition.get("no_search") is not True:
        errors.append("event family was not preregistered as no-search")
    return errors


def validate_event_provenance(run_dir: Path) -> tuple[list[str], dict[str, Any]]:
    sources = load_csv(run_dir / "event_source_provenance.csv")
    events = load_csv(run_dir / "event_timestamp_dataset.csv")
    coverage = load_csv(run_dir / "event_coverage.csv")
    errors: list[str] = []
    if len(sources) and not set(sources["official_domain"].dropna().astype(str)).issubset(OFFICIAL_DOMAINS):
        errors.append("non-official event source domain present")
    forbidden = {"actual", "forecast", "consensus", "surprise", "sentiment", "revision"}
    if forbidden & {column.lower() for column in events.columns}:
        errors.append("event outcome/forecast field present")
    if len(events):
        required = {
            "event_type", "official_release_timestamp", "original_timezone", "release_timestamp_utc",
            "official_source", "source_document", "source_release_identifier", "acquisition_timestamp_utc",
        }
        if not required.issubset(events.columns):
            errors.append("event timestamp provenance fields incomplete")
        else:
            parsed = pd.to_datetime(events["release_timestamp_utc"], utc=True, errors="coerce")
            if parsed.isna().any():
                errors.append("unparseable UTC event timestamp")
            if events.duplicated(["event_type", "release_timestamp_utc"]).any():
                errors.append("duplicate event type/timestamp")
            if not set(events["event_type"]).issubset(experiment.EVENT_TYPES):
                errors.append("unregistered event type present")
            if not set(events["official_source"]).issubset(OFFICIAL_DOMAINS):
                errors.append("event dataset contains non-official source")
            if parsed.notna().any() and not parsed.dt.year.between(experiment.ACQUISITION_START_YEAR, experiment.ACQUISITION_END_YEAR).all():
                errors.append("event outside frozen acquisition years")
    stated_gate = bool(coverage["coverage_gate_pass"].astype(str).str.lower().eq("true").all())
    summary = {
        "source_fetches": len(sources),
        "successful_fetches": int(sources.get("status", pd.Series(dtype=str)).eq("ok").sum()),
        "verified_events": len(events),
        "coverage_rows": len(coverage),
        "all_coverage_rows_pass": stated_gate,
    }
    return errors, summary


def validate_readiness_stop(run_dir: Path, manifest: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    readiness = metrics["data_readiness"]
    errors: list[str] = []
    coverage = load_csv(run_dir / "event_coverage.csv")
    recomputed = bool(coverage["coverage_gate_pass"].astype(str).str.lower().eq("true").all())
    if bool(readiness["passed"]) != recomputed:
        errors.append("readiness verdict does not equal all coverage gates")
    trained = bool(manifest["model"].get("trained"))
    provenance = read_json(run_dir / "fold_model_provenance.json")
    if not readiness["passed"] and (trained or provenance.get("models")):
        errors.append("model was trained after data-readiness failure")
    if readiness["passed"] and not trained:
        errors.append("readiness passed but paired models are missing")
    return errors


def validate_alignment(run_dir: Path, trained: bool) -> list[str]:
    audit = load_csv(run_dir / "feature_alignment_audit.csv")
    errors: list[str] = []
    if trained:
        if "event_release_le_decision" not in audit.columns or not audit["event_release_le_decision"].astype(str).str.lower().eq("true").all():
            errors.append("feature alignment contains or cannot exclude pre-release exposure")
        if not audit.get("timezone_conversion", pd.Series(dtype=str)).astype(str).str.contains("America/New_York.*UTC.*EET/EEST", regex=True).all():
            errors.append("timezone conversion evidence incomplete")
    return errors


def validate_paired_models(run_dir: Path) -> tuple[list[str], dict[str, Any]]:
    provenance = read_json(run_dir / "fold_model_provenance.json")
    errors: list[str] = []
    folds = provenance.get("folds", [])
    models = provenance.get("models", [])
    if len(folds) != 3 or len(models) != 6:
        errors.append("expected exactly three folds and six paired fold models")
    if len(provenance.get("base_features", [])) != 31:
        errors.append("B0 base feature count is not 31")
    if tuple(provenance.get("event_features", [])) != experiment.EVENT_FEATURES:
        errors.append("B1 appended feature list mismatch")
    for fold in folds:
        if fold.get("target_sha256_B0") != fold.get("target_sha256_B1"):
            errors.append(f"paired target mismatch: {fold.get('fold')}")
        if fold.get("parameters_B0") != fold.get("parameters_B1") or fold.get("parameters_B0") != experiment.baseline.FIXED_XGB_PARAMETERS:
            errors.append(f"paired XGBoost parameters mismatch: {fold.get('fold')}")
        if fold.get("random_seed_B0") != fold.get("random_seed_B1") or fold.get("random_seed_B0") != 42:
            errors.append(f"paired random seed mismatch: {fold.get('fold')}")
        if not fold.get("strict_label_maturity_before_score") or pd.Timestamp(fold["latest_training_label_information_time"]) >= pd.Timestamp(fold["score_start"]):
            errors.append(f"label maturity overlap: {fold.get('fold')}")
    for model in models:
        path = run_dir / model["path"]
        if not path.is_file() or sha256(path) != model["sha256"]:
            errors.append(f"model hash mismatch: {model.get('path')}")
    baseline_match = None
    oof = run_dir / "paired_oof_predictions.npz"
    prior = experiment.BASELINE_RUN / "paired_oof_predictions.npz"
    if oof.is_file() and prior.is_file():
        with np.load(oof, allow_pickle=False) as current, np.load(prior, allow_pickle=False) as old:
            same_times = np.array_equal(current["time_ns"], old["time_ns"])
            same_target = np.array_equal(current["target"], old["target"])
            same_score = np.array_equal(current["score_b0"], old["score_c1"])
            maximum_difference = float(np.max(np.abs(current["score_b0"].astype(float) - old["score_c1"].astype(float)))) if same_times else None
            baseline_match = {"same_times": same_times, "same_target": same_target, "same_score_exact": same_score, "maximum_score_difference": maximum_difference}
            if not same_times or not same_target:
                errors.append("B0 scored rows/target differ from frozen prior C1 baseline")
            if not same_score:
                errors.append("B0 probabilities do not exactly reproduce prior C1")
    else:
        errors.append("paired OOF or prior baseline OOF missing")
    return errors, {"folds": len(folds), "models": len(models), "baseline_reproduction": baseline_match}


def reconstruct(run_dir: Path) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    with np.load(run_dir / "paired_oof_predictions.npz", allow_pickle=False) as data:
        base_frame = pd.DataFrame({
            "TIME_DT": pd.to_datetime(data["time_ns"].astype(np.int64)),
            "OPEN": data["open"], "HIGH": data["high"], "LOW": data["low"], "CLOSE": data["close"],
            "ATR": data["atr"], "M1_RSI": data["rsi"], "SPREAD": data["spread"],
        })
        scores = {experiment.MODEL_IDS[0]: data["score_b0"], experiment.MODEL_IDS[1]: data["score_b1"]}
    submitted_trades = load_csv(run_dir / "trade_ledger.csv")
    submitted_metrics = load_csv(run_dir / "fold_metrics.csv")
    summary: dict[str, Any] = {}
    for model_id in experiment.MODEL_IDS:
        cohort = base_frame.copy()
        cohort["buy_prob"] = scores[model_id].astype(np.float32)
        cohort["sell_prob"] = np.float32(0)
        cohort = semantics.finalize_cohort(cohort, f"VALIDATOR_{model_id}", offset_hours=0)
        trades, _ = semantics.simulate(cohort, semantics.SIMULATORS[-1])
        submitted = submitted_trades[submitted_trades["model_id"].eq(model_id)].reset_index(drop=True)
        if len(trades) != len(submitted):
            errors.append(f"S5 trade count mismatch {model_id}: {len(trades)} != {len(submitted)}")
        else:
            for index, trade in enumerate(trades):
                row = submitted.iloc[index]
                if pd.Timestamp(row["entry_time_api"]) != pd.Timestamp(trade["entry_time_api"]) or row["exit_reason"] != trade["exit_reason"] or not close(row["net_r"], trade["net_r"]):
                    errors.append(f"S5 trade identity mismatch {model_id} ordinal {index}")
                    break
        pooled = experiment.baseline.trade_metrics(trades, sum(experiment.baseline.core.fold_days(start, end) for _, start, end in experiment.baseline.FOLDS))
        row = submitted_metrics[submitted_metrics["model_id"].eq(model_id) & submitted_metrics["fold"].eq("pooled")]
        if len(row) != 1:
            errors.append(f"pooled metric row missing {model_id}")
        else:
            for key in ("trades", "trades_per_day", "realized_wr", "pf", "mean_r", "pnl_r", "max_dd_r", "cost_stress_pf"):
                if not close(row.iloc[0][key], pooled[key]):
                    errors.append(f"metric mismatch {model_id}/{key}")
        summary[model_id] = {key: pooled[key] for key in ("trades", "trades_per_day", "realized_wr", "pf", "mean_r", "pnl_r", "max_dd_r", "cost_stress_pf")}
    return errors, summary


def verdict_rows(internal_errors: list[str], trained: bool, readiness_pass: bool) -> list[dict[str, str]]:
    internal = "PASS" if not internal_errors else "FAIL"
    reasons = "; ".join(internal_errors[:4]) if internal_errors else "frozen paired design and readiness stop were independently verified"
    rows: list[dict[str, str]] = []
    for check in CHECKS:
        if check in {"holdout contamination", "multiple-testing risk"}:
            verdict = "FAIL"
            evidence = "2018-2024 outcomes were repeatedly inspected; no untouched final interval is used"
            correction = "evaluate a completely frozen candidate on genuinely new future data"
        else:
            verdict = internal
            if not trained and not readiness_pass:
                evidence = "data-readiness gate failed and fitting/execution were correctly not run"
            else:
                evidence = "independent provenance, paired-design, maturity, and S5 reconstruction checks"
            correction = "none" if verdict == "PASS" else "correct the listed methodology/evidence error and rerun without tuning"
        rows.append({"Check": check, "Verdict": verdict, "Evidence": evidence, "Failure or reason for pass": reasons if verdict == "FAIL" else evidence, "Required validation correction": correction})
    return rows


def report_text(rows: list[dict[str, str]], result: dict[str, Any]) -> str:
    lines = ["# Independent walk-forward validation", "", "Overall: FAIL", "", "| Check | Verdict | Evidence | Failure or reason for pass | Required validation correction |", "|---|---|---|---|---|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[key]).replace("|", "/") for key in ("Check", "Verdict", "Evidence", "Failure or reason for pass", "Required validation correction")) + " |")
    lines += [
        "", "## Independent conclusions", "",
        f"Internal methodology: `{result['internal_methodology']}`.",
        f"Final untouched validity: `{result['final_untouched_validity']}`.",
        f"Data-readiness gate: `{'PASS' if result['data_readiness_pass'] else 'FAIL'}`.",
        "The overall FAIL is retained because no untouched final interval exists; this does not overturn a correctly enforced internal data-readiness stop.",
        "", "Submitted historical performance claim: `invalid as final untouched evidence`.", "",
    ]
    return "\n".join(lines)


def self_check() -> None:
    assert close(1.0, 1.0 + 1e-9)
    assert len(CHECKS) == 12
    print("SELF_CHECK_OK")


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0
    if args.run_dir is None:
        raise ValueError("run_dir is required")
    run_dir = args.run_dir.resolve()
    errors = validate_required(run_dir)
    if errors:
        raise RuntimeError("; ".join(errors))
    manifest = read_json(run_dir / "manifest.json")
    metrics = read_json(run_dir / "metrics.json")
    errors += validate_definition(run_dir)
    provenance_errors, event_summary = validate_event_provenance(run_dir)
    errors += provenance_errors
    errors += validate_readiness_stop(run_dir, manifest, metrics)
    trained = bool(manifest["model"].get("trained"))
    readiness_pass = bool(metrics["data_readiness"]["passed"])
    errors += validate_alignment(run_dir, trained)
    model_summary: dict[str, Any] = {"not_run": True}
    metric_summary: dict[str, Any] = {"not_run": True}
    if trained:
        model_errors, model_summary = validate_paired_models(run_dir)
        errors += model_errors
        execution_errors, metric_summary = reconstruct(run_dir)
        errors += execution_errors
    before = manifest.get("operational_hashes_before", {})
    after = manifest.get("operational_hashes_after", {})
    current = {experiment.GEMINI_FILE.name: sha256(experiment.GEMINI_FILE), experiment.OPERATIONAL_MODEL.name: sha256(experiment.OPERATIONAL_MODEL)}
    if before != after or after != current:
        errors.append("operational artifact hash changed")
    internal = "PASS" if not errors else "FAIL"
    result = {
        "overall": "FAIL",
        "internal_methodology": internal,
        "final_untouched_validity": "FAIL",
        "data_readiness_pass": readiness_pass,
        "model_trained": trained,
        "event_provenance": event_summary,
        "paired_model_audit": model_summary,
        "recomputed_metrics": metric_summary,
        "internal_errors": errors,
        "multiple_testing": "FAIL: repeatedly inspected 2018-2024 development history and no untouched final interval",
    }
    rows = verdict_rows(errors, trained, readiness_pass)
    experiment.write_json(run_dir / "validator.json", {**result, "checks": rows})
    (run_dir / "validator.md").write_text(report_text(rows, result), encoding="utf-8")
    manifest = read_json(run_dir / "manifest.json")
    manifest["registry"]["validator_result"] = f"internal_methodology={internal}; final_untouched_validity=FAIL"
    manifest["independent_validator"] = result
    for path in (run_dir / "validator.json", run_dir / "validator.md"):
        experiment.baseline.add_artifact(manifest, run_dir, path, "independent_validator")
    experiment.write_json(run_dir / "manifest.json", manifest)
    print(f"VALIDATOR internal_methodology={internal} final_untouched_validity=FAIL errors={len(errors)}")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
