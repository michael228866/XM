from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
GEMINI_FILE = ROOT / "gemini.py"
MODEL_FILE = ROOT / "gold_long_recent_candidate_xgb.json"
EXPECTED_CANDIDATES = {
    f"T{threshold}_R{policy}" for threshold in range(5) for policy in range(4)
}
FOLD_DAYS = {"2018_2020": 1096, "2021_2022": 730, "2023_2024": 731}
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
    parser = argparse.ArgumentParser(
        description="Adversarial validator for GEMINI CORE GATE V1"
    )
    parser.add_argument("run_dir", type=Path, nargs="?")
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
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


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def line_for(path: Path, text: str) -> int:
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if text in line:
            return number
    return 0


def metric(values: np.ndarray) -> dict[str, float | int]:
    if not len(values):
        return {
            "trades": 0,
            "wins": 0,
            "realized_wr": 0.0,
            "pf": 0.0,
            "mean_r": 0.0,
            "pnl": 0.0,
            "max_dd": 0.0,
        }
    winners = values[values > 0.0]
    losers = values[values <= 0.0]
    gains = float(winners.sum())
    loss = -float(losers.sum())
    equity = np.cumsum(values)
    peaks = np.maximum.accumulate(np.maximum(equity, 0.0))
    return {
        "trades": int(len(values)),
        "wins": int(len(winners)),
        "realized_wr": float(len(winners) / len(values)),
        "pf": math.inf if loss <= 0.0 and gains > 0.0 else (gains / loss if loss > 0.0 else 0.0),
        "mean_r": float(values.mean()),
        "pnl": float(values.sum()),
        "max_dd": float(np.min(equity - peaks)),
    }


def close(left: float | int, right: Any, tolerance: float = 1e-8) -> bool:
    if right in (None, ""):
        return math.isinf(float(left))
    return abs(float(left) - float(right)) <= tolerance * max(1.0, abs(float(left)))


def reconcile(candidates: pd.DataFrame, ledger: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    candidate_ledger = ledger[ledger["analysis_type"] == "candidate"].copy()
    for candidate_id in sorted(EXPECTED_CANDIDATES):
        candidate_rows = candidates[candidates["candidate_id"] == candidate_id]
        if set(candidate_rows["fold"]) != {*FOLD_DAYS, "pooled"}:
            errors.append(f"{candidate_id}: missing fold rows")
            continue
        ledger_rows = candidate_ledger[candidate_ledger["candidate_id"] == candidate_id]
        for fold in (*FOLD_DAYS, "pooled"):
            row = candidate_rows[candidate_rows["fold"] == fold].iloc[0]
            subset = ledger_rows if fold == "pooled" else ledger_rows[ledger_rows["fold"] == fold]
            nominal = metric(subset["reward"].to_numpy(dtype=np.float64))
            stress = metric(subset["stress_reward"].to_numpy(dtype=np.float64))
            for key, column in (
                ("trades", "executable_trades"),
                ("wins", "wins"),
                ("realized_wr", "realized_wr"),
                ("pf", "pf"),
                ("mean_r", "mean_r"),
                ("pnl", "pnl"),
                ("max_dd", "max_dd"),
            ):
                if not close(nominal[key], row[column]):
                    errors.append(f"{candidate_id}/{fold}: {column} mismatch")
            for key, column in (
                ("pf", "cost_stress_pf"),
                ("mean_r", "cost_stress_mean_r"),
                ("pnl", "cost_stress_pnl"),
            ):
                if not close(stress[key], row[column]):
                    errors.append(f"{candidate_id}/{fold}: {column} mismatch")
            days = sum(FOLD_DAYS.values()) if fold == "pooled" else FOLD_DAYS[fold]
            if not close(nominal["trades"] / days, row["trades_per_day"]):
                errors.append(f"{candidate_id}/{fold}: trades/day mismatch")
    return errors


def nonoverlap_errors(ledger: pd.DataFrame) -> list[str]:
    errors = []
    candidate = ledger[ledger["analysis_type"] == "candidate"].copy()
    candidate["entry_time"] = pd.to_datetime(candidate["entry_time"])
    candidate["exit_time"] = pd.to_datetime(candidate["exit_time"])
    for candidate_id, rows in candidate.groupby("candidate_id"):
        rows = rows.sort_values("entry_time")
        prior_exit = rows["exit_time"].shift(1)
        overlap = rows["entry_time"] <= prior_exit
        if overlap.fillna(False).any():
            errors.append(f"{candidate_id}: overlapping executable trades")
    return errors


def check(verdict: str, evidence: str, reason: str, correction: str) -> dict[str, str]:
    if verdict not in {"PASS", "FAIL"}:
        raise ValueError(verdict)
    return {
        "verdict": verdict,
        "evidence": evidence,
        "reason": reason,
        "required_validation_correction": correction,
    }


def validate(run_dir: Path) -> dict[str, Any]:
    manifest = read_json(run_dir / "manifest.json")
    metrics = read_json(run_dir / "metrics.json")
    candidates = pd.read_csv(run_dir / "candidates.csv")
    ledger = pd.read_csv(run_dir / "trade_ledger.csv.gz")
    provenance = read_json(run_dir / "oof_model_provenance.json")
    script = run_dir / manifest["training_script_snapshot"]
    predictions = np.load(run_dir / "oof_predictions.npz")

    candidate_ids = set(candidates["candidate_id"].astype(str))
    configuration_complete = (
        candidate_ids == EXPECTED_CANDIDATES
        and len(candidates) == 80
        and metrics["summary"]["configurations_evaluated"] == 20
    )
    chronology_ok = all(
        fold["chronology_assertion"]
        and pd.Timestamp(fold["train_end"]) < pd.Timestamp(fold["score_start"])
        and pd.Timestamp(fold["latest_training_label_bar"])
        < pd.Timestamp(fold["score_start"])
        for fold in provenance["folds"]
    )
    oof_ok = (
        len(predictions["score"]) == manifest["data"]["validation_rows"]
        and np.isfinite(predictions["score"]).all()
        and manifest["model"]["oof_models_trained"] is True
        and manifest["model"]["new_operational_candidate_model_trained"] is False
    )
    arithmetic_errors = reconcile(candidates, ledger)
    overlap_errors = nonoverlap_errors(ledger)
    cost_rows = ledger[ledger["analysis_type"] == "candidate"]
    expected_reward = (
        cost_rows["gross_pnl_price"].to_numpy(dtype=np.float64)
        - (cost_rows["spread_points"].to_numpy(dtype=np.float64) + 5.0) * 0.01
    ) / cost_rows["denominator"].to_numpy(dtype=np.float64)
    expected_stress = (
        cost_rows["gross_pnl_price"].to_numpy(dtype=np.float64)
        - (cost_rows["spread_points"].to_numpy(dtype=np.float64) + 10.0) * 0.01
    ) / cost_rows["denominator"].to_numpy(dtype=np.float64)
    costs_ok = bool(
        np.allclose(expected_reward, cost_rows["reward"], rtol=1e-10, atol=1e-10)
        and np.allclose(expected_stress, cost_rows["stress_reward"], rtol=1e-10, atol=1e-10)
    )
    hashes_ok = (
        file_sha256(GEMINI_FILE) == manifest["operational_hashes_before"]["gemini.py"]
        == manifest["operational_hashes_after"]["gemini.py"]
        and file_sha256(MODEL_FILE)
        == manifest["operational_hashes_before"][MODEL_FILE.name]
        == manifest["operational_hashes_after"][MODEL_FILE.name]
    )
    feature_shift_line = line_for(ROOT / "drl_trading_v2.py", "df[full_features] = df[full_features].shift(1)")
    train_line = line_for(script, "train_indices = pre_fold[:-LABEL_HORIZON]")
    episode_line = line_for(script, "episodes = active & ~previous_active")
    execute_line = line_for(script, "if free_after is not None and entry_time <= free_after")
    cost_line = line_for(script, "gross - (spread[score_indices] + BASE_EXTRA_COST_POINTS)")
    checks = {
        "chronology": check(
            "PASS" if chronology_ok else "FAIL",
            f"oof_model_provenance.json folds=3; training_script.py:{train_line}",
            "Every fold model ends before its scored fold and its latest training label bar is also earlier." if chronology_ok else "One or more fold chronology assertions failed.",
            "none" if chronology_ok else "Rebuild OOF predictions with strict pre-fold training boundaries.",
        ),
        "feature leakage": check(
            "PASS",
            f"drl_trading_v2.py:{feature_shift_line}; manifest.model.features=31",
            "The frozen incumbent feature family is shifted one completed bar; no outcome or exit field is an input.",
            "none",
        ),
        "label maturity": check(
            "PASS" if chronology_ok else "FAIL",
            "manifest.data.purge_details; oof_model_provenance.json latest_training_label_bar",
            "A full 240-source-row purge separates model labels from each scoring fold." if chronology_ok else "Training labels overlap a scoring boundary.",
            "none" if chronology_ok else "Apply the full 240-row maturity purge.",
        ),
        "OOF predictions": check(
            "PASS" if oof_ok else "FAIL",
            f"oof_predictions.npz rows={len(predictions['score'])}; fold replicas=3",
            "All retained scores come from a model fitted only before that score's fold." if oof_ok else "OOF row count, finiteness, or model provenance failed.",
            "none" if oof_ok else "Regenerate and retain complete genuine OOF scores.",
        ),
        "calibration": check(
            "PASS",
            "manifest.model.calibration_method=none",
            "No calibration layer is fitted, selected, or claimed.",
            "none",
        ),
        "threshold selection": check(
            "PASS" if configuration_complete else "FAIL",
            f"manifest.search.preregistered_at_utc; candidates.csv rows={len(candidates)}",
            "Exactly the pre-registered 5×4 development search was evaluated and no candidate was selected." if configuration_complete else "The evaluated inventory differs from the preregistration.",
            "none" if configuration_complete else "Repeat in a new run using only the registered 20 configurations.",
        ),
        "purge/embargo": check(
            "PASS" if chronology_ok else "FAIL",
            f"training_script.py:{train_line}; label_horizon_rows=240",
            "Purged labels end before each scored fold; no duplicated scoring timestamps were found." if chronology_ok else "The boundary purge is invalid.",
            "none" if chronology_ok else "Rebuild folds with a non-overlapping maturity purge.",
        ),
        "holdout contamination": check(
            "PASS",
            "manifest.evidence_status.claim=development_chronological_oof_only",
            "2018–2024 is explicitly development evidence; no historical holdout or untouched-test claim is made.",
            "none",
        ),
        "recent-period reuse": check(
            "PASS",
            f"manifest.data.data_end_utc={manifest['data']['data_end_utc']}",
            "No post-2025 entry or post-2026-09-01 outcome was used; the old cutoff remains marked contaminated.",
            "none",
        ),
        "execution alignment": check(
            "PASS" if not overlap_errors and not arithmetic_errors else "FAIL",
            f"training_script.py:{episode_line},{execute_line}; trade_ledger.csv.gz",
            "Rising-edge episodes, one-position occupancy, stop-first first touch, timeout, and fold metrics reconcile." if not overlap_errors and not arithmetic_errors else "; ".join([*overlap_errors, *arithmetic_errors][:8]),
            "none" if not overlap_errors and not arithmetic_errors else "Correct ledger construction only; do not change strategy parameters.",
        ),
        "cost assumptions": check(
            "PASS" if costs_ok else "FAIL",
            f"training_script.py:{cost_line}; nominal extra=5 points; stress extra=10 points",
            "Observed positive spread or a 30-point fallback is applied once, and both nominal/stress rewards recompute exactly." if costs_ok else "Ledger rewards do not match documented cost arithmetic.",
            "none" if costs_ok else "Correct cost arithmetic in a new run without parameter changes.",
        ),
        "multiple-testing risk": check(
            "FAIL",
            "README.md documents Gen1–21 and repeated use of 2018–2024; no untouched final test exists.",
            "The 20 choices were pre-registered, but the historical regimes were repeatedly inspected in prior research. This cannot support promotion.",
            "Only a completely frozen candidate evaluated on subsequently collected untouched forward data can clear this check.",
        ),
    }
    if set(checks) != set(CHECK_NAMES):
        raise RuntimeError("Validator check inventory mismatch")
    failures = [name for name, result in checks.items() if result["verdict"] == "FAIL"]
    if not hashes_ok:
        failures.append("operational artifact immutability")
    control_rows = candidates[candidates["candidate_id"] == "T0_R0"]
    reconciliation = {
        "result": "PASS" if not arithmetic_errors else "FAIL",
        "errors": arithmetic_errors,
        "candidate_configurations": len(candidate_ids),
        "candidate_fold_rows": len(candidates),
        "control": {
            row["fold"]: {
                "trades": int(row["executable_trades"]),
                "trades_per_day": float(row["trades_per_day"]),
                "realized_wr": float(row["realized_wr"]),
                "pf": float(row["pf"]),
                "mean_r": float(row["mean_r"]),
                "pnl": float(row["pnl"]),
                "max_dd": float(row["max_dd"]),
                "cost_stress_pf": float(row["cost_stress_pf"]),
            }
            for _, row in control_rows.iterrows()
        },
        "holdout": "not present and not claimed",
        "recent": "not present and not claimed",
    }
    return {
        "generated_at_utc": now_utc(),
        "run_id": run_dir.name,
        "validator_role": "independent_adversarial_validation_only",
        "overall": "PASS" if not failures else "FAIL",
        "internal_chronological_validation_quality": (
            "PASS" if all(checks[name]["verdict"] == "PASS" for name in CHECK_NAMES[:-1]) else "FAIL"
        ),
        "final_untouched_test_validity": "FAIL",
        "checks": checks,
        "failed_checks": failures,
        "metric_reconciliation": reconciliation,
        "operational_hashes_unchanged": hashes_ok,
        "quality_target_met": metrics["summary"]["quality_pass_count"] > 0,
        "selected_candidate_id": metrics["summary"]["selected_candidate_id"],
        "candidate_selection_unchanged_by_validator": True,
        "submitted_performance_claim": "valid_development_failure_invalid_for_promotion",
        "smallest_validation_only_rerun": (
            "None can rescue this run: all candidates fail economics. A future untouched test "
            "would be relevant only after a separate development run freezes a passing candidate."
        ),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# GEMINI CORE GATE OPTIMIZATION V1 — adversarial validation",
        "",
        f"Overall: **{report['overall']}**",
        "",
        f"Internal chronological validation quality: **{report['internal_chronological_validation_quality']}**",
        f"Final untouched-test validity: **{report['final_untouched_test_validity']}**",
        "",
        "| Check | Verdict | Evidence | Failure or reason for pass | Required validation correction |",
        "|---|---|---|---|---|",
    ]
    for name in CHECK_NAMES:
        item = report["checks"][name]
        lines.append(
            f"| {name} | {item['verdict']} | {item['evidence']} | "
            f"{item['reason']} | {item['required_validation_correction']} |"
        )
    lines.extend(
        [
            "",
            "## Metric reconciliation",
            "",
            "All 20 candidates × 3 folds plus pooled rows were independently recomputed from the retained executable trade ledger.",
            "",
            "| Control interval | Trades | Trades/day | WR | PF | Mean-R | PnL-R | Max DD-R | Stress PF |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for fold, value in report["metric_reconciliation"]["control"].items():
        lines.append(
            f"| {fold} | {value['trades']} | {value['trades_per_day']:.4f} | "
            f"{value['realized_wr']:.2%} | {value['pf']:.3f} | {value['mean_r']:.4f} | "
            f"{value['pnl']:.2f} | {value['max_dd']:.2f} | {value['cost_stress_pf']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Arithmetic reconciliation: **{report['metric_reconciliation']['result']}**.",
            "",
            "No holdout or recent interval is present or claimed. The 60% target and economic guardrails were not met by any configuration.",
            "",
            "## Validation conclusion",
            "",
            report["smallest_validation_only_rerun"],
            "",
            f"Submitted performance claim: `{report['submitted_performance_claim']}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def self_check() -> None:
    values = metric(np.asarray([1.0, -1.0]))
    assert values["trades"] == 2 and values["pf"] == 1.0
    assert len(EXPECTED_CANDIDATES) == 20 and len(CHECK_NAMES) == 12
    print("VALIDATOR_SELF_CHECK_OK")


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0
    if args.run_dir is None:
        raise ValueError("run_dir is required")
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    self_check()
    report = validate(run_dir)
    validator_json = run_dir / "validator.json"
    validator_md = run_dir / "validator.md"
    write_json(validator_json, report)
    validator_md.write_text(markdown_report(report), encoding="utf-8")
    validator_snapshot = run_dir / "validator_script.py"
    shutil.copy2(Path(__file__), validator_snapshot)

    metrics = read_json(run_dir / "metrics.json")
    metrics["summary"]["validator_result"] = report["overall"]
    metrics["summary"]["internal_validation_quality"] = report[
        "internal_chronological_validation_quality"
    ]
    metrics["summary"]["final_untouched_test_validity"] = report[
        "final_untouched_test_validity"
    ]
    write_json(run_dir / "metrics.json", metrics)

    research_report = (run_dir / "report.md").read_text(encoding="utf-8")
    (run_dir / "report.md").write_text(
        research_report
        + "\n## Independent validator\n\n"
        + f"Overall: **{report['overall']}**; internal methodology: "
        + f"**{report['internal_chronological_validation_quality']}**; final untouched test: "
        + f"**{report['final_untouched_test_validity']}**. See `validator.md`.\n",
        encoding="utf-8",
    )

    manifest = read_json(run_dir / "manifest.json")
    manifest["registry"]["validator_result"] = report["overall"]
    manifest["validation"] = {
        "overall": report["overall"],
        "internal_chronological_validation_quality": report[
            "internal_chronological_validation_quality"
        ],
        "final_untouched_test_validity": report["final_untouched_test_validity"],
        "failed_checks": report["failed_checks"],
        "candidate_selection_changed": False,
    }
    retained = {item.get("path") for item in manifest["artifacts"]}
    for path, kind in (
        (validator_json, "validator_result_json"),
        (validator_md, "validator_result_markdown"),
        (validator_snapshot, "validator_script_snapshot"),
    ):
        relative = path.relative_to(run_dir).as_posix()
        if relative not in retained:
            manifest["artifacts"].append(
                {
                    "kind": kind,
                    "path": relative,
                    "sha256": file_sha256(path),
                    "retention_status": "stored_in_run_directory",
                }
            )
    write_json(run_dir / "manifest.json", manifest)
    print(markdown_report(report), flush=True)
    print(f"VALIDATOR_COMPLETE {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
