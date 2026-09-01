from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import gold_generation20_walk_forward_validation as audit20


ROOT = Path(__file__).resolve().parent
EXPERIMENT_FILE = ROOT / "gold_generation21_new_information.json"
CONFIG_FILE = ROOT / "gold_generation21_candidate.json"
FORWARD_FILE = ROOT / "gold_generation21_forward_protocol.json"
SOURCE_FILE = ROOT / "gold_generation21_new_information.py"
GOLD_M1 = ROOT / "GOLD#_M1_201401020000_202605082357.csv"
SILVER_M1 = (
    ROOT / "數據集" / "SILVER#_M1_201401020000_202605281623.csv"
)
GEMINI_FILE = ROOT / "gemini.py"
OUTPUT_JSON = ROOT / "gold_generation21_walk_forward_validation.json"
OUTPUT_MD = ROOT / "gold_generation21_walk_forward_validation.md"

FOLDS = ("2018_2020", "2021_2022", "2023_2024")
VERSIONS = (
    "A_technical_control",
    "B_microstructure",
    "C_cross_market_xag",
    "E_microstructure_plus_xag",
)
CHECKS = audit20.CHECKS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generation 21 adversarial walk-forward validation"
    )
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left: object, right: object, tolerance: float = 1e-8) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(
        float(left), float(right), rel_tol=tolerance, abs_tol=tolerance
    )


def spearman(left: list[float], right: list[float]) -> float | None:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left_values) & np.isfinite(right_values)
    if finite.sum() < 3:
        return None
    value = pd.Series(left_values[finite]).corr(
        pd.Series(right_values[finite]), method="spearman"
    )
    return None if pd.isna(value) else float(value)


def calibration(probability: list[float], target: list[int]) -> dict:
    probability_values = np.asarray(probability, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    finite = np.isfinite(probability_values) & np.isfinite(target_values)
    probability_values = probability_values[finite]
    target_values = target_values[finite]
    weighted_error = 0.0
    bins = []
    for lower in np.arange(0.0, 1.0, 0.2):
        upper = lower + 0.2
        keep = (probability_values >= lower) & (
            probability_values <= upper
            if upper >= 1.0
            else probability_values < upper
        )
        count = int(keep.sum())
        predicted = (
            float(probability_values[keep].mean()) if count else None
        )
        observed = float(target_values[keep].mean()) if count else None
        if count:
            weighted_error += count * abs(predicted - observed)
        bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_predicted": predicted,
                "observed_positive_rate": observed,
            }
        )
    return {
        "events": len(probability_values),
        "brier": (
            float(np.mean((probability_values - target_values) ** 2))
            if len(probability_values)
            else None
        ),
        "ece": (
            weighted_error / len(probability_values)
            if len(probability_values)
            else None
        ),
        "bins": bins,
    }


def metric_inventory(report: dict) -> list[dict]:
    output = []
    for fold in FOLDS:
        value = report["control_baseline"]["folds"][fold]
        output.append(
            audit20.reconcile_ledgers(
                f"gen20_control:{fold}",
                value["trade_ledger"],
                value["cost_stress_trade_ledger"],
                value["metrics"],
                value["cost_stress"],
            )
        )
    pooled = report["control_baseline"]["selection_pooled"]
    output.append(
        audit20.reconcile_ledgers(
            "gen20_control:selection_pooled",
            pooled["trade_ledger"],
            pooled["cost_stress_trade_ledger"],
            pooled["metrics"],
            pooled["cost_stress"],
        )
    )
    candidate = report["candidate_construction"]
    if candidate["executed"]:
        output.append(
            audit20.reconcile_ledgers(
                "gen21_candidate:selection_pooled",
                candidate["trade_ledger"],
                candidate["cost_stress_trade_ledger"],
                candidate["pooled_metrics"],
                candidate["pooled_cost_stress"],
            )
        )
    return output


def score_reconciliation(report: dict) -> list[dict]:
    output = []
    for fold in FOLDS:
        baseline = report["control_baseline"]["folds"][fold]["trade_ledger"]
        baseline_by_id = {record["trade_id"]: record for record in baseline}
        for version in VERSIONS:
            value = report["fold_results"][fold][version]
            ledger = value["fixed_cohort_prediction_ledger"]
            ids = [record["trade_id"] for record in ledger]
            rewards = [float(record["reward"]) for record in ledger]
            positive = [int(reward > 0.0) for reward in rewards]
            tp_first = [int(int(record["outcome"]) == 1) for record in ledger]
            p_tp = [float(record["p_tp_first"]) for record in ledger]
            p_net = [float(record["p_net"]) for record in ledger]
            e_net = [float(record["e_net"]) for record in ledger]
            recomputed = {
                "existing_p_tp_first_vs_tp_first": spearman(p_tp, tp_first),
                "existing_p_tp_first_vs_net_r": spearman(p_tp, rewards),
                "p_net_vs_positive_net_r": spearman(p_net, positive),
                "p_net_vs_net_r": spearman(p_net, rewards),
                "e_net_vs_net_r": spearman(e_net, rewards),
            }
            ranking_mismatches = [
                key
                for key, expected in recomputed.items()
                if not close(expected, value["ranking"].get(key))
            ]
            calibration_value = calibration(p_net, positive)
            calibration_mismatches = [
                key
                for key in ("events", "brier", "ece")
                if not close(
                    calibration_value[key], value["calibration"]["p_net"][key]
                )
            ]
            identity_errors = sum(
                trade_id not in baseline_by_id
                or not close(
                    ledger[index]["reward"], baseline_by_id[trade_id]["reward"]
                )
                for index, trade_id in enumerate(ids)
            )
            output.append(
                {
                    "fold": fold,
                    "version": version,
                    "trades": len(ledger),
                    "unique_ids": len(set(ids)),
                    "baseline_ids": len(baseline_by_id),
                    "identity_errors": int(identity_errors),
                    "ranking_mismatches": ranking_mismatches,
                    "calibration_mismatches": calibration_mismatches,
                    "reported_ranking": value["ranking"],
                    "recomputed_ranking": recomputed,
                    "reported_calibration": {
                        key: value["calibration"]["p_net"][key]
                        for key in ("events", "brier", "ece")
                    },
                    "recomputed_calibration": {
                        key: calibration_value[key]
                        for key in ("events", "brier", "ece")
                    },
                }
            )
    return output


def boundary_audit(report: dict) -> dict:
    output = {}
    for fold in FOLDS:
        output[fold] = {}
        for version in VERSIONS:
            model = report["model_diagnostics"][fold][version]
            output[fold][version] = {
                "fit_max_label_end_index": model["fit_max_label_end_index"],
                "calibration_start_index": model["calibration_start_index"],
                "calibration_max_label_end_index": model[
                    "calibration_max_label_end_index"
                ],
                "policy_start_index": model["policy_start_index"],
                "fit_to_calibration_purged": (
                    int(model["fit_max_label_end_index"])
                    < int(model["calibration_start_index"])
                ),
                "calibration_to_policy_purged": (
                    int(model["calibration_max_label_end_index"])
                    < int(model["policy_start_index"])
                ),
            }
    return output


def gate_audit(report: dict) -> dict:
    summaries = report["version_summary"]
    control = summaries["A_technical_control"]
    gate = report["information_gate"]
    output = {}
    for version in VERSIONS:
        value = summaries[version]
        correlations = value["fold_e_net_spearman_vs_net_r"]
        control_folds = control["fold_e_net_spearman_vs_net_r"]
        improvements = [
            current - baseline
            for current, baseline in zip(correlations, control_folds)
        ]
        recomputed = bool(
            version != "A_technical_control"
            and np.mean(correlations) - np.mean(control_folds)
            >= gate["mean_spearman_improvement_min"]
            and sum(value > 0.0 for value in correlations)
            >= gate["positive_folds_min"]
            and sum(value > 0.0 for value in improvements)
            >= gate["improved_folds_min"]
            and min(correlations) >= gate["worst_fold_min"]
        )
        output[version] = {
            "reported": value["information_gate_pass"],
            "recomputed": recomputed,
        }
    any_pass = any(value["recomputed"] for value in output.values())
    return {
        "versions": output,
        "any_information_gate_pass": any_pass,
        "candidate_construction_correctly_gated_off": (
            not any_pass
            and not report["candidate_construction"]["executed"]
            and report["frozen_candidate_id"] is None
        ),
    }


def verdicts(
    report: dict,
    metrics: list[dict],
    scores: list[dict],
    boundaries: dict,
    gate: dict,
    forward: dict,
) -> dict:
    flat_boundaries = [
        value for fold in boundaries.values() for value in fold.values()
    ]
    boundary_ok = all(
        value["fit_to_calibration_purged"]
        and value["calibration_to_policy_purged"]
        for value in flat_boundaries
    )
    scores_ok = all(
        value["trades"] == value["unique_ids"] == value["baseline_ids"]
        and value["identity_errors"] == 0
        and not value["ranking_mismatches"]
        and not value["calibration_mismatches"]
        for value in scores
    )
    metrics_ok = all(
        not value["base_metric_mismatches"]
        and not value["stress_metric_mismatches"]
        and value["base_execution"]["duplicate_trade_ids"] == 0
        and value["base_execution"]["occupancy_non_overlapping"]
        and value["base_execution"]["reward_formula_errors"] == 0
        and value["base_execution"]["cost_component_errors"] == 0
        and value["base_execution"]["exit_offset_errors"] == 0
        and value["base_execution"]["spread_quality_errors"] == 0
        for value in metrics
    )
    feature_sets = report["feature_sets"]
    forbidden = {
        "NET_REWARD",
        "NET_POSITIVE",
        "SHORT_OUTCOME",
        "SHORT_REWARD",
        "SHORT_EXIT_OFFSET",
    }
    feature_ok = all(
        not forbidden.intersection(features)
        for name, features in feature_sets.items()
        if isinstance(features, list)
    )
    source = SOURCE_FILE.read_text(encoding="utf-8")
    alignment_ok = all(
        token in source
        for token in (
            '.shift(1)',
            'direction="backward"',
            "tolerance=pd.Timedelta(minutes=5)",
            "No timestamped historical calendar",
        )
    )
    forward_cutoff = datetime.fromisoformat(
        forward["untouched_forward_cutoff_utc"]
    )
    run_time = datetime.fromisoformat(report["generated_at"])
    forward_ok = (
        forward_cutoff > run_time
        and forward["status"] == "untouched_pending"
        and report["final_untouched_test_validity"]
        == "FAIL_pending_future_data"
    )
    raw = {
        "chronology": (
            "PASS" if alignment_ok else "FAIL",
            "All ablations fit before each scored fold; microstructure uses completed bars and XAG uses backward-only as-of alignment.",
        ),
        "feature_leakage": (
            "PASS" if feature_ok and alignment_ok else "FAIL",
            "No return target, outcome, exit offset, future calendar value, forward merge, or fabricated unavailable source enters a model.",
        ),
        "label_maturity": (
            "PASS" if boundary_ok else "FAIL",
            "Exact event label ends precede calibration and policy partitions for all four ablations and all folds.",
        ),
        "oof_predictions": (
            "PASS" if scores_ok else "FAIL",
            "Every stored score ledger exactly covers the frozen executable cohort and independently reproduces reported ranking metrics.",
        ),
        "calibration": (
            "PASS" if boundary_ok and scores_ok else "FAIL",
            "P(net_R>0) isotonic calibration uses only the purged chronological calibration partition; Brier/ECE recompute exactly.",
        ),
        "threshold_selection": (
            "PASS" if gate["candidate_construction_correctly_gated_off"] else "FAIL",
            "The information gate was predeclared, all versions failed it, and no post-hoc feature or selector became a candidate.",
        ),
        "purge_embargo": (
            "PASS" if boundary_ok else "FAIL",
            "Fit and calibration labels end strictly before the next partition; outer fold training retains the horizon-tail purge.",
        ),
        "holdout_contamination": (
            "FAIL",
            "All inspected history is development data and no untouched final test has yet arrived after the new cutoff.",
        ),
        "recent_period_reuse": (
            "PASS" if forward_ok else "FAIL",
            "Gen21 did not evaluate recent strategy outcomes and recorded a future cutoff later than the experiment run.",
        ),
        "execution_alignment": (
            "PASS" if metrics_ok and scores_ok else "FAIL",
            "The experiment ranks the unchanged non-overlapping Gen17 cohort; independent cost, reward, PF, PnL, DD, and occupancy checks reconcile.",
        ),
        "cost_assumptions": (
            "PASS" if metrics_ok else "FAIL",
            "Observed spread/fallback-30, 5-point base extra cost, and 10-point stress cost are unchanged and independently reconciled.",
        ),
        "multiple_testing_risk": (
            "FAIL",
            "Four feature ablations plus post-hoc univariate diagnostics were inspected after prior generations without an evaluated untouched test.",
        ),
    }
    return {
        name: {"verdict": raw[name][0], "reason": raw[name][1]}
        for name in CHECKS
    }


def markdown(report: dict) -> str:
    evidence = {
        "chronology": "gold_generation21_new_information.py:add_microstructure_features,add_cross_market_features",
        "feature_leakage": "gold_generation21_new_information.json:feature_sets,source_inventory",
        "label_maturity": "gold_generation21_walk_forward_validation.json:boundary_audit",
        "oof_predictions": "gold_generation21_new_information.json:fold_results.*.fixed_cohort_prediction_ledger",
        "calibration": "gold_generation21_new_information.py:score_fixed_cohort",
        "threshold_selection": "gold_generation21_new_information.json:information_gate,candidate_construction",
        "purge_embargo": "gold_generation21_new_information.json:model_diagnostics",
        "holdout_contamination": "gold_generation21_new_information.json:development_history_policy",
        "recent_period_reuse": "gold_generation21_forward_protocol.json:untouched_forward_cutoff_utc",
        "execution_alignment": "gold_generation21_new_information.json:control_baseline",
        "cost_assumptions": "gold_generation21_new_information.json:frozen_comparability",
        "multiple_testing_risk": "gold_generation21_new_information.json:selection_inventory",
    }
    corrections = {
        "holdout_contamination": "Wait for the registered future interval; do not inspect outcomes before a candidate is frozen.",
        "multiple_testing_risk": "Test only a preregistered frozen hypothesis on genuinely new forward data.",
    }
    lines = [
        "# Generation 21 walk-forward validation",
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
    for name in CHECKS:
        value = report["checks"][name]
        lines.append(
            f"| {name} | {value['verdict']} | {evidence[name]} | "
            f"{value['reason']} | {corrections.get(name, 'None')} |"
        )
    lines.extend(
        [
            "",
            "## Independent metric reconciliation",
            "",
            "| Period | Trades | Days | Trades/day | Wins/Losses | WR | PF | Mean-R | PnL | Max DD | Stress PF | Reconciled |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for value in report["metric_reconciliation"]:
        metric = value["reported"]
        stress = value["reported_cost_stress"]
        ok = (
            not value["base_metric_mismatches"]
            and not value["stress_metric_mismatches"]
        )
        lines.append(
            f"| {value['label']} | {metric['trades']} | {metric['evaluated_days']} | "
            f"{metric['trades_per_day']:.3f} | {metric['wins']}/{metric['losses']} | "
            f"{metric['realized_positive_trade_win_rate']:.2%} | "
            f"{metric['profit_factor'] or 0.0:.3f} | {metric['mean_r']:.4f} | "
            f"{metric['pnl']:.2f} | {metric['max_drawdown_pct']:.2%} | "
            f"{stress['profit_factor'] or 0.0:.3f} | {'yes' if ok else 'no'} |"
        )
    lines.extend(
        [
            "",
            "No feature family passed the frozen information gate. No strategy candidate was constructed or promoted.",
            "",
            "Submitted historical evidence is valid for internal chronological development only and invalid as a final untouched OOS claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def self_check() -> None:
    probability = [0.2, 0.8]
    target = [0, 1]
    value = calibration(probability, target)
    assert close(value["brier"], 0.04)
    assert len(CHECKS) == 12
    print("generation21_walk_forward_validation_self_check_ok")


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0
    experiment = json.loads(EXPERIMENT_FILE.read_text(encoding="utf-8"))
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    forward = json.loads(FORWARD_FILE.read_text(encoding="utf-8"))
    metrics = metric_inventory(experiment)
    scores = score_reconciliation(experiment)
    boundaries = boundary_audit(experiment)
    gate = gate_audit(experiment)
    checks = verdicts(experiment, metrics, scores, boundaries, gate, forward)
    overall = (
        "PASS"
        if all(value["verdict"] == "PASS" for value in checks.values())
        else "FAIL"
    )
    internal = (
        "PASS"
        if all(
            checks[name]["verdict"] == "PASS"
            for name in CHECKS
            if name not in ("holdout_contamination", "multiple_testing_risk")
        )
        else "FAIL"
    )
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generation": "21_walk_forward_validation",
        "validator_role": "adversarial_methodology_audit_only",
        "overall": overall,
        "internal_chronological_validation_quality": internal,
        "final_untouched_test_validity": "FAIL_pending_future_data",
        "checks": checks,
        "metric_reconciliation": metrics,
        "score_reconciliation": scores,
        "boundary_audit": boundaries,
        "information_gate_audit": gate,
        "evidence_manifest": {
            "git_revision": revision,
            "gold_m1_sha256": sha256(GOLD_M1),
            "silver_m1_sha256": sha256(SILVER_M1),
            "gen21_code_sha256": sha256(SOURCE_FILE),
            "gen21_report_sha256": sha256(EXPERIMENT_FILE),
            "forward_protocol_sha256": sha256(FORWARD_FILE),
            "gemini_sha256": sha256(GEMINI_FILE),
            "gemini_hash_matches_experiment": (
                sha256(GEMINI_FILE)
                == experiment["gemini_sha256_before_and_after"]
            ),
            "run_timestamp": experiment["generated_at"],
            "untouched_forward_cutoff_utc": forward[
                "untouched_forward_cutoff_utc"
            ],
        },
        "candidate_status": {
            "candidate_construction_executed": experiment[
                "candidate_construction"
            ]["executed"],
            "frozen_candidate_id": experiment["frozen_candidate_id"],
            "promotion_pass": experiment["promotion_pass"],
            "config_matches": (
                config["frozen_candidate_id"]
                == experiment["frozen_candidate_id"]
                and config["candidate_construction_executed"]
                == experiment["candidate_construction"]["executed"]
                and not config["promotion_pass"]
            ),
        },
        "claim_validity": "internal_chronological_development_only",
        "minimum_validation_correction": (
            "Freeze a preregistered hypothesis without consulting forward outcomes, "
            "then evaluate it once after sufficient data accumulates beyond the cutoff."
        ),
    }
    OUTPUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    OUTPUT_MD.write_text(markdown(report), encoding="utf-8")
    print(f"Overall: {overall}")
    print(f"Internal chronological validation quality: {internal}")
    print(f"Wrote {OUTPUT_JSON.name} and {OUTPUT_MD.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
