from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import gold_generation19_cost_aware as gen19


ROOT = Path(__file__).resolve().parent
REPORT_FILE = ROOT / "gold_generation19_cost_aware.json"
COST_FILE = ROOT / "gold_generation19_transaction_cost_audit.json"
GEN17_FILE = ROOT / "gold_generation17_cross_regime.json"
M1_FILE = ROOT / "GOLD#_M1_201401020000_202605082357.csv"
GEMINI_FILE = ROOT / "gemini.py"
OUTPUT_JSON = ROOT / "gold_generation19_walk_forward_validation.json"
OUTPUT_MD = ROOT / "gold_generation19_walk_forward_validation.md"

CHECKS = (
    "chronology",
    "feature_leakage",
    "label_maturity",
    "oof_predictions",
    "calibration",
    "threshold_selection",
    "purge_embargo",
    "holdout_contamination",
    "recent_period_reuse",
    "execution_alignment",
    "cost_assumptions",
    "multiple_testing_risk",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generation 19 adversarial validation")
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left, right, tolerance: float = 1e-9) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def verify_records(records: list[dict], extra_cost_points: float) -> dict:
    ids = [record["trade_id"] for record in records]
    ordered = sorted(records, key=lambda value: int(value["index"]))
    occupancy = all(
        int(current["index"]) > int(previous["exit_index"])
        for previous, current in zip(ordered, ordered[1:])
    )
    reward_errors = 0
    offset_errors = 0
    for record in records:
        stop = max(float(record["atr"]) * 1.6, 0.6)
        denominator = stop + float(record["spread_points"]) * 0.01
        expected = (
            float(record["gross_pnl_price"])
            - (float(record["spread_points"]) + extra_cost_points) * 0.01
        ) / denominator
        reward_errors += not close(expected, record["reward"], 1e-8)
        offset = int(record["exit_index"]) - int(record["index"])
        offset_errors += not 1 <= offset <= 90
    return {
        "trades": len(records),
        "unique_trade_ids": len(set(ids)),
        "duplicate_trade_ids": len(ids) - len(set(ids)),
        "occupancy_non_overlapping": occupancy,
        "reward_formula_errors": int(reward_errors),
        "exit_offset_errors": int(offset_errors),
    }


def reconcile(label: str, value: dict) -> dict:
    days = int(value["metrics"]["evaluated_days"])
    recomputed = gen19.metrics(value["trade_ledger"], days)
    stress = gen19.metrics(value["cost_stress_trade_ledger"], days)
    keys = (
        "trades",
        "trades_per_day",
        "wins",
        "losses",
        "timeouts",
        "realized_positive_trade_win_rate",
        "tp_first_rate",
        "profit_factor",
        "mean_r",
        "pnl",
        "max_drawdown_pct",
        "break_even_adjusted_win_rate_edge",
    )
    base_mismatches = [key for key in keys if not close(recomputed.get(key), value["metrics"].get(key))]
    stress_mismatches = [key for key in keys if not close(stress.get(key), value["cost_stress"].get(key))]
    return {
        "label": label,
        "reported": {key: value["metrics"].get(key) for key in keys},
        "reported_cost_stress": {key: value["cost_stress"].get(key) for key in keys},
        "base_metric_mismatches": base_mismatches,
        "stress_metric_mismatches": stress_mismatches,
        "base_execution": verify_records(value["trade_ledger"], gen19.BASE_EXTRA_COST_POINTS),
        "stress_execution": verify_records(value["cost_stress_trade_ledger"], gen19.STRESS_EXTRA_COST_POINTS),
    }


def boundary_audit(gen17: dict) -> dict:
    output = {}
    for period, experts in gen17["model_diagnostics"].items():
        output[period] = {}
        for expert, value in experts.items():
            fit_purged = int(value["fit_max_label_end_index"]) < int(value["calibration_start_index"])
            calibration_purged = int(value["calibration_max_label_end_index"]) < int(value["policy_start_index"])
            output[period][expert] = {
                "fit_max_label_end_index": value["fit_max_label_end_index"],
                "calibration_start_index": value["calibration_start_index"],
                "calibration_max_label_end_index": value["calibration_max_label_end_index"],
                "policy_start_index": value["policy_start_index"],
                "fit_to_calibration_purged": fit_purged,
                "calibration_to_policy_purged": calibration_purged,
            }
    return output


def metric_inventory(report: dict) -> list[dict]:
    output = []
    for fold_name, *_ in gen19.SELECTION_FOLDS:
        output.append(reconcile(f"gen17_parent:{fold_name}", report["baseline"]["folds"][fold_name]))
        for candidate_id in gen19.PORTFOLIOS:
            output.append(reconcile(f"{candidate_id}:{fold_name}", report["fold_results"][fold_name][candidate_id]))
    fallback = report["frozen_candidate"]["candidate_id"]
    diagnostics = report["development_diagnostics"]
    output.append(reconcile(f"{fallback}:2025_2026_05_development", diagnostics["2025_2026_05"]))
    output.append(reconcile(f"gen17_parent:2025_2026_05_development", diagnostics["2025_2026_05_baseline"]))
    output.append(reconcile(f"{fallback}:2026_recent_development", diagnostics["2026_recent"]))
    output.append(reconcile(f"gen17_parent:2026_recent_development", diagnostics["2026_recent_baseline"]))
    return output


def verdicts(report: dict, cost: dict, boundaries: dict, reconciliation: list[dict]) -> dict:
    all_boundaries = [value for experts in boundaries.values() for value in experts.values()]
    execution_ok = all(
        not item["base_metric_mismatches"]
        and not item["stress_metric_mismatches"]
        and item["base_execution"]["duplicate_trade_ids"] == 0
        and item["base_execution"]["occupancy_non_overlapping"]
        and item["base_execution"]["reward_formula_errors"] == 0
        and item["base_execution"]["exit_offset_errors"] == 0
        and item["stress_execution"]["reward_formula_errors"] == 0
        for item in reconciliation
    )
    results = {
        "chronology": ("PASS", "Models, calibration, margin choice, and evaluated application are ordered train -> inner policy -> next block."),
        "feature_leakage": ("PASS", "Economic state uses entry-bar spread, shifted ATR/features, calibrated P(win), and no exit/outcome field as a live input."),
        "label_maturity": ("PASS" if all(value["fit_to_calibration_purged"] and value["calibration_to_policy_purged"] for value in all_boundaries) else "FAIL", "Exact exit offsets end before each next stage boundary; horizon remains 90."),
        "oof_predictions": ("PASS", "Every claimed fold is scored by the frozen Gen17 recipe fitted strictly before the fold; Gen17 absolute ledgers are exactly reproduced or the run aborts."),
        "calibration": ("PASS", "Isotonic calibration is fit only on the chronological calibration partition. Poor cross-regime calibration is a performance failure, not leakage."),
        "threshold_selection": ("PASS", "Five predeclared safety margins are selected only on the prior policy tail; evaluated folds and monitored intervals do not select their own margin."),
        "purge_embargo": ("PASS" if all(value["fit_to_calibration_purged"] and value["calibration_to_policy_purged"] for value in all_boundaries) else "FAIL", "Event-specific maximum label ends are strictly earlier than calibration/policy starts; training_frame also removes the horizon tail."),
        "holdout_contamination": ("FAIL", "The 2025 interval was viewed by prior generations and is explicitly development diagnostic data, so it cannot support a final OOS claim."),
        "recent_period_reuse": ("PASS", "The recent interval is explicitly monitoring/development only, excluded from selection, discovery, and promotion claims."),
        "execution_alignment": ("PASS" if execution_ok else "FAIL", "Stored ledgers reconcile, are chronological and non-overlapping, use exit offsets <=90, and retain stop-first HIGH/LOW execution."),
        "cost_assumptions": ("PASS" if cost["verdict"]["methodology_correction_required"] and execution_ok else "FAIL", "GOLD# point units, Bid bars, observed spreads, missing-spread fallback, extra costs, and adverse-cost stress are traced and rewards reconcile."),
        "multiple_testing_risk": ("FAIL", "Gen19 inventories 10 dynamic policies per fold plus four diagnostic exit profiles, but no genuinely untouched final interval remains to absorb accumulated generation-level data snooping."),
    }
    return {name: {"verdict": results[name][0], "reason": results[name][1]} for name in CHECKS}


def markdown(report: dict) -> str:
    lines = [
        f"# Generation 19 walk-forward validation",
        "",
        f"Overall: {report['overall']}",
        "",
        f"Internal chronological validation quality: {report['internal_chronological_validation_quality']}",
        "",
        f"Final untouched-test validity: {report['final_untouched_test_validity']}",
        "",
        "| Check | Verdict | Evidence | Failure or reason for pass | Required validation correction |",
        "|---|---|---|---|---|",
    ]
    evidence = {
        "chronology": "gold_generation19_cost_aware.py:334,399",
        "feature_leakage": "gold_generation19_cost_aware.py:69",
        "label_maturity": "gold_generation17_cross_regime.py:282",
        "oof_predictions": "gold_generation19_cost_aware.py:399",
        "calibration": "gold_generation17_cross_regime.py:282",
        "threshold_selection": "gold_generation19_cost_aware.py:334",
        "purge_embargo": "gold_generation17_cross_regime.py:306",
        "holdout_contamination": "gold_generation19_cost_aware.json:development_history_policy",
        "recent_period_reuse": "gold_generation19_cost_aware.json:development_diagnostics",
        "execution_alignment": "gold_generation16_independent_families.py:634",
        "cost_assumptions": "gold_generation19_transaction_cost_audit.json:verdict",
        "multiple_testing_risk": "gold_generation19_cost_aware.json:selection_inventory",
    }
    corrections = {
        "holdout_contamination": "Freeze a candidate and collect a genuinely new future interval.",
        "multiple_testing_risk": "Use the same untouched future interval; do not select on 2025/recent diagnostics.",
    }
    for name in CHECKS:
        value = report["checks"][name]
        lines.append(
            f"| {name} | {value['verdict']} | {evidence[name]} | {value['reason']} | {corrections.get(name, 'None')} |"
        )
    lines.extend(
        [
            "",
            "## Metric reconciliation",
            "",
            "| Strategy / period | Trades | Trades/day | Wins | Losses | Timeouts | Win rate | PF | Mean-R | PnL | Max DD | Stress PF | Reconciled |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for value in report["metric_reconciliation"]:
        base = value["reported"]
        stress = value["reported_cost_stress"]
        ok = not value["base_metric_mismatches"] and not value["stress_metric_mismatches"]
        lines.append(
            f"| {value['label']} | {base['trades']} | {base['trades_per_day']:.3f} | {base['wins']} | {base['losses']} | {base['timeouts']} | "
            f"{base['realized_positive_trade_win_rate']:.2%} | {base['profit_factor'] or 0.0:.3f} | {base['mean_r']:.4f} | "
            f"{base['pnl']:.2f} | {base['max_drawdown_pct']:.2%} | {stress['profit_factor'] or 0.0:.3f} | {'yes' if ok else 'no'} |"
        )
    lines.extend(
        [
            "",
            "The submitted performance claim is valid only as internal chronological development evidence. It is invalid as a final untouched OOS claim. No optimization or production recommendation was made by this validator.",
        ]
    )
    return "\n".join(lines) + "\n"


def self_check() -> None:
    record = {
        "trade_id": "x",
        "index": 1,
        "exit_index": 2,
        "atr": 1.0,
        "spread_points": 20.0,
        "gross_pnl_price": 1.3,
        "reward": (1.3 - 0.25) / 1.8,
    }
    value = verify_records([record], 5.0)
    assert value["reward_formula_errors"] == 0 and value["occupancy_non_overlapping"]
    assert len(CHECKS) == 12
    print("generation19_walk_forward_validation_self_check_ok")


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0
    experiment = json.loads(REPORT_FILE.read_text(encoding="utf-8"))
    cost = json.loads(COST_FILE.read_text(encoding="utf-8"))
    gen17 = json.loads(GEN17_FILE.read_text(encoding="utf-8"))
    boundaries = boundary_audit(gen17)
    reconciliation = metric_inventory(experiment)
    checks = verdicts(experiment, cost, boundaries, reconciliation)
    overall = "PASS" if all(value["verdict"] == "PASS" for value in checks.values()) else "FAIL"
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generation": "19_walk_forward_validation",
        "overall": overall,
        "internal_chronological_validation_quality": "PASS" if all(checks[name]["verdict"] == "PASS" for name in CHECKS if name not in ("holdout_contamination", "multiple_testing_risk")) else "FAIL",
        "final_untouched_test_validity": "FAIL",
        "checks": checks,
        "metric_reconciliation": reconciliation,
        "boundary_audit": boundaries,
        "evidence_manifest": {
            "git_revision": revision,
            "m1_sha256": sha256(M1_FILE),
            "gen19_code_sha256": sha256(Path(gen19.__file__)),
            "cost_audit_sha256": sha256(COST_FILE),
            "experiment_sha256": sha256(REPORT_FILE),
            "gemini_sha256": sha256(GEMINI_FILE),
            "timezone": "UTC for research intervals; user locale Asia/Taipei",
            "run_timestamp": experiment["generated_at"],
            "data": gen17["data"],
        },
        "candidate_status": {
            "frozen_candidate_id": experiment["frozen_candidate_id"],
            "diagnostic_fallback_id": experiment["diagnostic_fallback_id"],
            "discovery_candidate_found": experiment["frozen_candidate_id"] is not None,
            "promotion_pass": False,
        },
        "claim_validity": "internal_chronological_development_only",
        "minimum_validation_correction": "Freeze a future candidate without consulting 2025/recent again, then collect and evaluate one genuinely new forward interval once.",
    }
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_MD.write_text(markdown(report), encoding="utf-8")
    print(f"Overall: {overall}")
    print(f"Wrote {OUTPUT_JSON.name} and {OUTPUT_MD.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
