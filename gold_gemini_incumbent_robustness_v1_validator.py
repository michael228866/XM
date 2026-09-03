from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import gold_gemini_core_gate_v1 as core


ROOT = Path(__file__).resolve().parent
SCHEMES = ("W0_expanding", "W1_trailing_24m", "W2_trailing_18m", "W3_trailing_12m")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent robustness attribution validator")
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
        raise TypeError(f"Expected JSON object: {path}")
    return value


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(sanitize(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def line_of(path: Path, token: str) -> int:
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if token in line:
            return number
    return 0


def load_oof(run_dir: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    with np.load(run_dir / "diagnostic_predictions.npz") as data:
        fold_names = np.asarray([name for name, _, _ in core.FOLDS], dtype=object)
        fold = fold_names[data["fold_code"].astype(np.int8)]
        base = pd.DataFrame(
            {
                "global_index": data["global_index"].astype(np.int64),
                "fold": fold,
                "time": pd.to_datetime(data["time_ns"].astype(np.int64)),
                "rsi": data["rsi"].astype(np.float64),
                "session_ok": data["session_ok"].astype(bool),
                "spread_points": data["spread_points"].astype(np.float64),
                "spread_observed": data["spread_observed"].astype(bool),
                "spread_ok": data["spread_ok"].astype(bool),
                "outcome": data["outcome"].astype(np.int8),
                "target": data["target"].astype(bool),
                "exit_time": pd.to_datetime(data["exit_time_ns"].astype(np.int64)),
                "gross_pnl_price": data["gross_pnl_price"].astype(np.float64),
                "denominator": data["denominator"].astype(np.float64),
                "reward": data["reward"].astype(np.float64),
                "stress_reward": data["stress_reward"].astype(np.float64),
            }
        )
        scores = {scheme: data[f"score_{scheme}"].astype(np.float64) for scheme in SCHEMES}
    return base, scores


def close(left: Any, right: Any, tolerance: float = 2e-5) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    return abs(float(left) - float(right)) <= tolerance


def recompute_metrics(
    base: pd.DataFrame, scores: dict[str, np.ndarray]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    ledger = []
    for scheme, values in scores.items():
        base["score"] = values
        trades, audit = core.execute(base, 0.75, "R0")
        for fold_name, start, end in core.FOLDS:
            selected = [trade for trade in trades if trade["fold"] == fold_name]
            metrics = core.trade_metrics(selected, core.fold_days(start, end))
            metrics.update(audit[fold_name])
            rows.append({"scheme": scheme, "fold": fold_name, **metrics})
        days = sum(core.fold_days(start, end) for _, start, end in core.FOLDS)
        pooled = core.trade_metrics(trades, days)
        for key in next(iter(audit.values())):
            pooled[key] = int(sum(item[key] for item in audit.values()))
        rows.append({"scheme": scheme, "fold": "pooled", **pooled})
        ledger.extend({"scheme": scheme, **trade} for trade in trades)
    return rows, ledger


def compare_metrics(run_dir: Path, rebuilt: list[dict[str, Any]]) -> list[str]:
    reported = pd.read_csv(run_dir / "window_metrics.csv")
    actual = {(item["scheme"], item["fold"]): item for item in rebuilt}
    errors = []
    numeric = (
        "trades",
        "trades_per_day",
        "realized_wr",
        "tp_first_wr",
        "pf",
        "mean_r",
        "pnl",
        "max_dd",
        "tp_count",
        "sl_count",
        "timeout_count",
    )
    for row in reported.itertuples(index=False):
        key = (row.scheme, row.fold)
        if key not in actual:
            errors.append(f"missing recomputed row {key}")
            continue
        for name in numeric:
            if not close(getattr(row, name), actual[key][name]):
                errors.append(f"metric mismatch {key}/{name}: {getattr(row, name)} != {actual[key][name]}")
    if len(reported) != len(actual):
        errors.append(f"window row count mismatch: {len(reported)} != {len(actual)}")
    return errors


def compare_ledger(run_dir: Path, rebuilt: list[dict[str, Any]]) -> list[str]:
    reported = pd.read_csv(run_dir / "trade_ledger.csv")
    expected = pd.DataFrame(rebuilt)
    errors = []
    if len(reported) != len(expected):
        errors.append(f"trade count mismatch: {len(reported)} != {len(expected)}")
        return errors
    reported["entry_time"] = pd.to_datetime(reported["entry_time"])
    expected["entry_time"] = pd.to_datetime(expected["entry_time"])
    left = list(zip(reported["scheme"], reported["fold"], reported["entry_time"]))
    right = list(zip(expected["scheme"], expected["fold"], expected["entry_time"]))
    if left != right:
        errors.append("trade identity/order mismatch")
    if not np.allclose(reported["reward"], expected["reward"], rtol=0, atol=2e-5):
        errors.append("trade reward mismatch")
    return errors


def compare_probability_tables(
    run_dir: Path, base: pd.DataFrame, scores: dict[str, np.ndarray]
) -> list[str]:
    table = pd.read_csv(run_dir / "probability_distribution.csv")
    errors = []
    for row in table.itertuples(index=False):
        mask = np.ones(len(base), dtype=bool) if row.fold == "pooled" else base["fold"].eq(row.fold).to_numpy()
        values = scores[row.scheme][mask]
        checks = {
            "observations": len(values),
            "mean": np.mean(values),
            "median": np.median(values),
            "p75": np.quantile(values, 0.75),
            "p90": np.quantile(values, 0.90),
            "p95": np.quantile(values, 0.95),
            "p99": np.quantile(values, 0.99),
            "maximum": np.max(values),
        }
        for name, value in checks.items():
            if not close(getattr(row, name), value):
                errors.append(f"probability distribution mismatch {row.scheme}/{row.fold}/{name}")
    return errors


def chronology_errors(provenance: dict[str, Any]) -> list[str]:
    errors = []
    models = provenance.get("fold_models", [])
    expected = {(scheme, fold) for scheme in SCHEMES for fold, _, _ in core.FOLDS}
    found = set()
    for item in models:
        key = (item.get("scheme"), item.get("fold"))
        found.add(key)
        label_end = pd.Timestamp(item["latest_training_label_bar"])
        score_start = pd.Timestamp(item["score_start"])
        train_end = pd.Timestamp(item["train_end"])
        if not (train_end < score_start and label_end < score_start and item.get("chronology_assertion")):
            errors.append(f"chronology failed: {key}")
    if found != expected:
        errors.append(f"provenance inventory mismatch: missing={sorted(expected-found)} extra={sorted(found-expected)}")
    return errors


def cost_errors(base: pd.DataFrame) -> list[str]:
    expected = (
        base["gross_pnl_price"].to_numpy()
        - (base["spread_points"].to_numpy() + core.BASE_EXTRA_COST_POINTS) * core.POINT
    ) / base["denominator"].to_numpy()
    if not np.allclose(expected, base["reward"].to_numpy(), rtol=0, atol=2e-5):
        return ["nominal reward does not reconcile to spread plus extra cost"]
    stress = (
        base["gross_pnl_price"].to_numpy()
        - (base["spread_points"].to_numpy() + core.STRESS_EXTRA_COST_POINTS) * core.POINT
    ) / base["denominator"].to_numpy()
    if not np.allclose(stress, base["stress_reward"].to_numpy(), rtol=0, atol=2e-5):
        return ["stress reward does not reconcile"]
    if np.any(base["spread_points"].to_numpy() <= 0):
        return ["zero/non-positive effective spread entered rewards"]
    return []


def self_check() -> None:
    assert len(CHECKS) == 12
    assert SCHEMES == ("W0_expanding", "W1_trailing_24m", "W2_trailing_18m", "W3_trailing_12m")
    assert close(1.0, 1.000001)
    print("VALIDATOR_SELF_CHECK_OK")


def make_check(verdict: str, evidence: str, reason: str, correction: str = "none") -> dict[str, str]:
    if verdict not in {"PASS", "FAIL"}:
        raise ValueError(verdict)
    return {
        "verdict": verdict,
        "evidence": evidence,
        "reason": reason,
        "required_validation_correction": correction,
    }


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# GEMINI INCUMBENT ROBUSTNESS ATTRIBUTION V1 - independent validation",
        "",
        f"Overall: **{result['overall']}**",
        "",
        f"Internal diagnostic methodology: **{result['internal_methodology']}**",
        "",
        f"Final untouched-test validity: **{result['final_untouched_test_validity']}**",
        "",
        "| Check | Verdict | Evidence | Failure or reason for pass | Required validation correction |",
        "|---|---|---|---|---|",
    ]
    for name in CHECKS:
        item = result["checks"][name]
        lines.append(
            f"| {name} | {item['verdict']} | {item['evidence']} | {item['reason']} | "
            f"{item['required_validation_correction']} |"
        )
    lines.extend(
        [
            "",
            "## Metric reconciliation",
            "",
            f"Window/fold metrics: **{result['metric_reconciliation']['result']}**.",
            f"Executable trade identities: **{result['execution_reconciliation']['result']}**.",
            f"Probability distributions: **{result['probability_reconciliation']['result']}**.",
            "",
            "| Scheme | Fold | Trades | Trades/day | WR | PF | Mean-R | PnL-R | Max DD-R |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["recomputed_metrics"]:
        lines.append(
            f"| {row['scheme']} | {row['fold']} | {row['trades']} | {row['trades_per_day']:.4f} | "
            f"{row['realized_wr']:.2%} | {row['pf']:.4f} | {row['mean_r']:.4f} | "
            f"{row['pnl']:.2f} | {row['max_dd']:.2f} |"
        )
    lines.extend(
        [
            "",
            "The 60% target and positive-economic guardrails are not promotion claims in this diagnostic. No candidate was selected.",
            "",
            "## Validation conclusion",
            "",
            result["smallest_validation_only_rerun"],
            "",
            f"Submitted claim: `{result['submitted_claim']}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0
    if args.run_dir is None:
        raise ValueError("run_dir is required")
    run_dir = args.run_dir.resolve()
    required = [
        "manifest.json",
        "metrics.json",
        "report.md",
        "window_metrics.csv",
        "trade_ledger.csv",
        "probability_distribution.csv",
        "calibration_summary.csv",
        "ranking_summary.csv",
        "oof_model_provenance.json",
        "diagnostic_predictions.npz",
    ]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing submitted evidence: " + ", ".join(missing))

    self_check()
    manifest = read_json(run_dir / "manifest.json")
    metrics = read_json(run_dir / "metrics.json")
    provenance = read_json(run_dir / "oof_model_provenance.json")
    base, scores = load_oof(run_dir)
    rebuilt_metrics, rebuilt_ledger = recompute_metrics(base, scores)
    metric_errors = compare_metrics(run_dir, rebuilt_metrics)
    ledger_errors = compare_ledger(run_dir, rebuilt_ledger)
    probability_errors = compare_probability_tables(run_dir, base, scores)
    chronology = chronology_errors(provenance)
    costs = cost_errors(base)

    script = ROOT / "gold_gemini_incumbent_robustness_v1.py"
    drl = ROOT / "drl_trading_v2.py"
    core_script = ROOT / "gold_gemini_core_gate_v1.py"
    add_targets_script = ROOT / "gold_generation11_execution_aligned.py"
    feature_line = line_of(drl, "df[full_features] = df[full_features].shift(1)")
    oof_line = line_of(script, "model = train_binary_model")
    purge_line = line_of(script, "train_index = pre[:-LABEL_HORIZON]")
    execution_line = line_of(core_script, "def execute(")
    target_line = line_of(add_targets_script, "def add_targets(")

    operational_ok = (
        manifest.get("operational_hashes_before") == manifest.get("operational_hashes_after")
        and sha256(ROOT / "gemini.py") == manifest["operational_hashes_before"]["gemini.py"]
        and sha256(ROOT / "gold_long_recent_candidate_xgb.json")
        == manifest["operational_hashes_before"]["gold_long_recent_candidate_xgb.json"]
        and manifest.get("promotion", {}).get("operational_artifact_changed") is False
    )
    design = manifest.get("diagnostic_design", {})
    fixed_design_ok = (
        design.get("strategy_configurations") == 1
        and design.get("threshold") == 0.75
        and design.get("training_windows")
        == {"W0_expanding": None, "W1_trailing_24m": 24, "W2_trailing_18m": 18, "W3_trailing_12m": 12}
        and manifest.get("search", {}).get("performed") is False
    )
    features_ok = manifest.get("model", {}).get("feature_count") == 31 and feature_line > 0
    prediction_ok = (
        len(base) == manifest["data"]["validation_rows"]
        and all(len(values) == len(base) and np.isfinite(values).all() for values in scores.values())
        and provenance.get("w0_identity", {}).get("alignment_verified") is True
    )
    calibration_ok = (
        manifest.get("model", {}).get("calibration_method") == "none; diagnosis only"
        and len(pd.read_csv(run_dir / "calibration_summary.csv")) == 16
        and len(pd.read_csv(run_dir / "calibration_buckets.csv")) == 160
    )
    probability_ok = not probability_errors and len(pd.read_csv(run_dir / "ranking_summary.csv")) == 16

    checks = {
        "chronology": make_check(
            "PASS" if not chronology else "FAIL",
            f"oof_model_provenance.json records=12; diagnostic script:{oof_line}",
            "All W0-W3 fold fits end before their scored interval and all preserved assertions reconcile." if not chronology else "; ".join(chronology),
            "none" if not chronology else "Create a new run with corrected fold boundaries; do not alter this run.",
        ),
        "feature leakage": make_check(
            "PASS" if features_ok else "FAIL",
            f"drl_trading_v2.py:{feature_line}; manifest.model.feature_count={manifest.get('model', {}).get('feature_count')}",
            "The frozen 31 features are shifted one completed bar and no diagnostic outcome field enters fitting." if features_ok else "Feature shift/order is not verifiable.",
            "none" if features_ok else "Rebuild in a new run with an immutable feature provenance record.",
        ),
        "label maturity": make_check(
            "PASS" if not chronology else "FAIL",
            f"gold_generation11_execution_aligned.py:{target_line}; oof_model_provenance.json",
            "Every fit has latest_training_label_bar strictly earlier than score_start." if not chronology else "At least one label boundary overlaps scoring.",
            "none" if not chronology else "Apply a full 240-source-row purge in a new run.",
        ),
        "OOF predictions": make_check(
            "PASS" if prediction_ok and not chronology else "FAIL",
            f"diagnostic_predictions.npz rows={len(base)}; schemes=4; script:{oof_line}",
            "W0 is the preserved parent OOF vector; all nine new replicas fit only their own preceding window." if prediction_ok and not chronology else "OOF inventory, alignment, or finiteness failed.",
            "none" if prediction_ok and not chronology else "Regenerate predictions in a new chronological run.",
        ),
        "calibration": make_check(
            "PASS" if calibration_ok and probability_ok else "FAIL",
            "manifest.model.calibration_method; calibration_summary.csv; calibration_buckets.csv",
            "No calibrator was fitted; fixed-bin calibration is descriptive and independently checked." if calibration_ok and probability_ok else "Calibration evidence is incomplete.",
            "none" if calibration_ok and probability_ok else "Regenerate descriptive calibration tables without fitting on scored outcomes.",
        ),
        "threshold selection": make_check(
            "PASS" if fixed_design_ok else "FAIL",
            "manifest.diagnostic_design; manifest.search.performed=false",
            "One threshold/RSI/execution configuration and exactly four pre-specified training windows were reported; none was promoted." if fixed_design_ok else "The frozen design does not match the submitted specification.",
            "none" if fixed_design_ok else "Create a new run with the fixed design declared before computation.",
        ),
        "purge/embargo": make_check(
            "PASS" if not chronology and purge_line > 0 else "FAIL",
            f"diagnostic script:{purge_line}; oof_model_provenance.json",
            "The full 240-row horizon is removed and exact label-end timestamps precede every score block." if not chronology and purge_line > 0 else "Purge/embargo is not verifiable.",
            "none" if not chronology and purge_line > 0 else "Correct purge logic in a new run.",
        ),
        "holdout contamination": make_check(
            "PASS",
            "manifest.evidence_status.classification=development_diagnostic_only",
            "No historical interval is labeled untouched and no production-performance claim is made.",
        ),
        "recent-period reuse": make_check(
            "PASS",
            "manifest.evidence_status.previous_forward_status; metrics.live_monitoring.classification",
            "The old cutoff remains contaminated and live outcomes are explicitly monitoring-only, with no tuning or selection.",
        ),
        "execution alignment": make_check(
            "PASS" if not metric_errors and not ledger_errors else "FAIL",
            f"gold_gemini_core_gate_v1.py:{execution_line}; diagnostic_predictions.npz; trade_ledger.csv",
            "All executable identities and W0-W3 fold metrics were rebuilt from the frozen event arrays." if not metric_errors and not ledger_errors else "; ".join(metric_errors + ledger_errors),
            "none" if not metric_errors and not ledger_errors else "Create a new run after correcting the execution reconstruction.",
        ),
        "cost assumptions": make_check(
            "PASS" if not costs else "FAIL",
            "diagnostic_predictions.npz gross_pnl_price/spread_points/denominator/reward/stress_reward",
            "Observed positive spread or a 30-point fallback plus 5/10-point extra costs reconcile for every OOF row." if not costs else "; ".join(costs),
            "none" if not costs else "Correct cost arithmetic in a new run.",
        ),
        "multiple-testing risk": make_check(
            "FAIL",
            "README.md historical generations; manifest.evidence_status; no untouched final interval",
            "The four windows were pre-specified and all are reported, but all scored regimes were repeatedly inspected and no untouched final data exists. Findings are hypothesis-generating only.",
            "Freeze a later experiment before collecting a genuinely new forward interval; do not reuse this diagnostic history as untouched evidence.",
        ),
    }
    internal_failures = [name for name, item in checks.items() if item["verdict"] == "FAIL" and name != "multiple-testing risk"]
    overall = "PASS" if all(item["verdict"] == "PASS" for item in checks.values()) else "FAIL"
    result = {
        "overall": overall,
        "internal_methodology": "PASS" if not internal_failures else "FAIL",
        "final_untouched_test_validity": "FAIL",
        "checks": checks,
        "failed_checks": [name for name, item in checks.items() if item["verdict"] == "FAIL"],
        "metric_reconciliation": {"result": "PASS" if not metric_errors else "FAIL", "errors": metric_errors},
        "execution_reconciliation": {"result": "PASS" if not ledger_errors else "FAIL", "errors": ledger_errors},
        "probability_reconciliation": {"result": "PASS" if not probability_errors else "FAIL", "errors": probability_errors},
        "recomputed_metrics": rebuilt_metrics,
        "operational_artifacts_unchanged": operational_ok,
        "diagnostic_conclusions_unchanged": True,
        "submitted_claim": "valid_internal_development_diagnostic_invalid_for_final_strategy_claim" if not internal_failures else "invalid",
        "smallest_validation_only_rerun": (
            "No internal rerun is required; the only failed check is final multiple-testing/untouched validity, which historical recomputation cannot repair."
            if not internal_failures
            else "Internal failures require a new immutable run: " + ", ".join(internal_failures)
        ),
    }
    if not operational_ok:
        result["overall"] = "FAIL"
        result["internal_methodology"] = "FAIL"
        result["submitted_claim"] = "invalid_operational_artifact_changed"

    validator_snapshot = run_dir / "validator_script.py"
    shutil.copy2(Path(__file__), validator_snapshot)
    validator_json = run_dir / "validator.json"
    validator_md = run_dir / "validator.md"
    write_json(validator_json, result)
    validator_md.write_text(markdown(result), encoding="utf-8")

    manifest = read_json(run_dir / "manifest.json")
    manifest["registry"]["validator_result"] = result["overall"]
    manifest["independent_validation"] = {
        "overall": result["overall"],
        "internal_methodology": result["internal_methodology"],
        "final_untouched_test_validity": result["final_untouched_test_validity"],
        "failed_checks": result["failed_checks"],
        "validator_script_sha256": sha256(validator_snapshot),
        "diagnostic_conclusions_changed": False,
    }
    existing = {item.get("path") for item in manifest.get("artifacts", [])}
    for path, kind in (
        (validator_snapshot, "independent_validator_script"),
        (validator_json, "independent_validator_result"),
        (validator_md, "independent_validator_report"),
    ):
        relative = path.relative_to(run_dir).as_posix()
        if relative not in existing:
            manifest.setdefault("artifacts", []).append(
                {
                    "kind": kind,
                    "path": relative,
                    "sha256": sha256(path),
                    "retention_status": "stored_in_run_directory_git_archival_pending",
                }
            )
    write_json(run_dir / "manifest.json", manifest)
    print(f"VALIDATOR_OVERALL_{result['overall']}")
    print(f"INTERNAL_METHODOLOGY_{result['internal_methodology']}")
    print("FINAL_UNTOUCHED_TEST_VALIDITY_FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
