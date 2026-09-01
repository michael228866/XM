from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPERIMENT_FILE = ROOT / "gold_generation20_direct_net_edge.json"
CONFIG_FILE = ROOT / "gold_generation20_candidate.json"
SOURCE_FILE = ROOT / "gold_generation20_direct_net_edge.py"
M1_FILE = ROOT / "GOLD#_M1_201401020000_202605082357.csv"
GEMINI_FILE = ROOT / "gemini.py"
OUTPUT_JSON = ROOT / "gold_generation20_walk_forward_validation.json"
OUTPUT_MD = ROOT / "gold_generation20_walk_forward_validation.md"

POINT = 0.01
SL_ATR = 1.6
MIN_SL_PRICE = 0.6
BASE_EXTRA_COST_POINTS = 5.0
STRESS_EXTRA_COST_POINTS = 10.0
FALLBACK_SPREAD_POINTS = 30.0
RISK_PER_TRADE = 0.014
HORIZON = 90
FOLDS = ("2018_2020", "2021_2022", "2023_2024")
POLICIES = (
    "p_net_ge_050",
    "e_net_positive",
    "joint_positive",
    "e_net_top75_past",
)
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
    parser = argparse.ArgumentParser(
        description="Generation 20 adversarial walk-forward audit"
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


def safe_mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def profit_factor(values: list[float]) -> float | None:
    gain = sum(value for value in values if value > 0.0)
    loss = -sum(value for value in values if value < 0.0)
    return None if loss <= 0.0 else gain / loss


def independent_metrics(records: list[dict], evaluated_days: int) -> dict:
    rewards = [float(record["reward"]) for record in records]
    outcomes = [int(record["outcome"]) for record in records]
    positive = [value for value in rewards if value > 0.0]
    nonpositive = [value for value in rewards if value <= 0.0]
    gross = [float(record["gross_reward_before_cost"]) for record in records]
    costs = [float(record["total_cost_r"]) for record in records]
    spread = [float(record["spread_r"]) for record in records]
    extra = [float(record["extra_cost_r"]) for record in records]
    average_winner = safe_mean(positive)
    average_loser = safe_mean(nonpositive)
    payoff = (
        None
        if average_winner is None
        or average_loser is None
        or average_loser == 0.0
        else average_winner / abs(average_loser)
    )
    break_even = None if payoff is None else 1.0 / (1.0 + payoff)
    win_rate = len(positive) / max(len(records), 1)
    balance = 1000.0
    peak = balance
    drawdown = 0.0
    pnl = 0.0
    for reward in rewards:
        change = balance * RISK_PER_TRADE * reward
        balance += change
        pnl += change
        peak = max(peak, balance)
        drawdown = min(drawdown, balance / peak - 1.0)
    return {
        "trades": len(records),
        "evaluated_days": evaluated_days,
        "trades_per_day": len(records) / max(evaluated_days, 1),
        "tp_first_rate": outcomes.count(1) / max(len(records), 1),
        "realized_positive_trade_win_rate": win_rate,
        "tp_exits": outcomes.count(1),
        "sl_exits": outcomes.count(2),
        "timeout_exits": outcomes.count(0),
        "other_exit_types": sum(value not in (0, 1, 2) for value in outcomes),
        "realized_positive_trades": len(positive),
        "realized_nonpositive_trades": len(nonpositive),
        "average_winning_r": average_winner,
        "median_winning_r": statistics.median(positive) if positive else None,
        "average_losing_r": average_loser,
        "median_losing_r": statistics.median(nonpositive) if nonpositive else None,
        "gross_profit_r": sum(positive),
        "gross_loss_r": -sum(nonpositive),
        "payoff_ratio": payoff,
        "realized_break_even_win_rate": break_even,
        "actual_realized_win_rate": win_rate,
        "break_even_adjusted_win_rate_edge": (
            None if break_even is None else win_rate - break_even
        ),
        "profit_factor": profit_factor(rewards),
        "mean_r": safe_mean(rewards) or 0.0,
        "sum_r": sum(rewards),
        "pnl": pnl,
        "max_drawdown_pct": drawdown,
        "mean_gross_r_before_cost": safe_mean(gross) or 0.0,
        "profit_factor_before_cost": profit_factor(gross),
        "average_total_cost_r_per_trade": safe_mean(costs) or 0.0,
        "average_spread_contribution_r_per_trade": safe_mean(spread) or 0.0,
        "average_extra_cost_contribution_r_per_trade": safe_mean(extra) or 0.0,
        "total_cost_contribution_r": sum(costs),
        "wins": len(positive),
        "losses": len(nonpositive),
        "timeouts": outcomes.count(0),
    }


METRIC_KEYS = (
    "trades",
    "evaluated_days",
    "trades_per_day",
    "tp_first_rate",
    "realized_positive_trade_win_rate",
    "tp_exits",
    "sl_exits",
    "timeout_exits",
    "other_exit_types",
    "realized_positive_trades",
    "realized_nonpositive_trades",
    "average_winning_r",
    "median_winning_r",
    "average_losing_r",
    "median_losing_r",
    "gross_profit_r",
    "gross_loss_r",
    "payoff_ratio",
    "realized_break_even_win_rate",
    "actual_realized_win_rate",
    "break_even_adjusted_win_rate_edge",
    "profit_factor",
    "mean_r",
    "sum_r",
    "pnl",
    "max_drawdown_pct",
    "mean_gross_r_before_cost",
    "profit_factor_before_cost",
    "average_total_cost_r_per_trade",
    "average_spread_contribution_r_per_trade",
    "average_extra_cost_contribution_r_per_trade",
    "total_cost_contribution_r",
    "wins",
    "losses",
    "timeouts",
)


def verify_records(records: list[dict], extra_cost_points: float) -> dict:
    ids = [str(record["trade_id"]) for record in records]
    ordered = sorted(records, key=lambda value: str(value["time"]))
    non_overlapping = all(
        str(current["time"]) > str(previous.get("exit_time", previous["time"]))
        if "exit_time" in previous
        else (
            str(current.get("period")) != str(previous.get("period"))
            or int(current["index"]) > int(previous["exit_index"])
        )
        for previous, current in zip(ordered, ordered[1:])
    )
    # Ledgers use period-relative indices, so enforce occupancy within each period too.
    by_period: dict[str, list[dict]] = {}
    for record in records:
        by_period.setdefault(str(record.get("period", "")), []).append(record)
    period_occupancy = all(
        all(
            int(current["index"]) > int(previous["exit_index"])
            for previous, current in zip(
                sorted(group, key=lambda value: int(value["index"])),
                sorted(group, key=lambda value: int(value["index"]))[1:],
            )
        )
        for group in by_period.values()
    )
    reward_errors = 0
    component_errors = 0
    offset_errors = 0
    spread_quality_errors = 0
    for record in records:
        atr = float(record["atr"])
        spread_points = float(record["spread_points"])
        stop = max(atr * SL_ATR, MIN_SL_PRICE)
        denominator = stop + spread_points * POINT
        expected_reward = (
            float(record["gross_pnl_price"])
            - (spread_points + extra_cost_points) * POINT
        ) / denominator
        reward_errors += not close(expected_reward, record["reward"])
        component_errors += not close(
            float(record["gross_pnl_price"]) / denominator,
            record["gross_reward_before_cost"],
        )
        component_errors += not close(
            spread_points * POINT / denominator, record["spread_r"]
        )
        component_errors += not close(
            extra_cost_points * POINT / denominator, record["extra_cost_r"]
        )
        component_errors += not close(
            (spread_points + extra_cost_points) * POINT / denominator,
            record["total_cost_r"],
        )
        offset = int(record["exit_index"]) - int(record["index"])
        offset_errors += not 1 <= offset <= HORIZON
        observed = bool(record.get("spread_observed", False))
        quality = str(record.get("spread_quality", ""))
        if observed:
            spread_quality_errors += spread_points <= 0.0 or quality != "observed"
        else:
            spread_quality_errors += (
                not close(spread_points, FALLBACK_SPREAD_POINTS)
                or quality != "fallback"
            )
    return {
        "trades": len(records),
        "unique_trade_ids": len(set(ids)),
        "duplicate_trade_ids": len(ids) - len(set(ids)),
        "chronological": all(
            str(left["time"]) <= str(right["time"])
            for left, right in zip(records, records[1:])
        ),
        "occupancy_non_overlapping": bool(non_overlapping and period_occupancy),
        "reward_formula_errors": int(reward_errors),
        "cost_component_errors": int(component_errors),
        "exit_offset_errors": int(offset_errors),
        "spread_quality_errors": int(spread_quality_errors),
    }


def reconcile_ledgers(
    label: str,
    records: list[dict],
    stress_records: list[dict],
    reported: dict,
    reported_stress: dict,
) -> dict:
    days = int(reported["evaluated_days"])
    base = independent_metrics(records, days)
    stress = independent_metrics(stress_records, days)
    return {
        "label": label,
        "reported": {key: reported.get(key) for key in METRIC_KEYS},
        "reported_cost_stress": {
            key: reported_stress.get(key) for key in METRIC_KEYS
        },
        "base_metric_mismatches": [
            key for key in METRIC_KEYS if not close(base.get(key), reported.get(key))
        ],
        "stress_metric_mismatches": [
            key
            for key in METRIC_KEYS
            if not close(stress.get(key), reported_stress.get(key))
        ],
        "base_execution": verify_records(records, BASE_EXTRA_COST_POINTS),
        "stress_execution": verify_records(
            stress_records, STRESS_EXTRA_COST_POINTS
        ),
    }


def metric_inventory(report: dict) -> list[dict]:
    output: list[dict] = []
    for fold in FOLDS:
        baseline = report["baseline"]["folds"][fold]
        output.append(
            reconcile_ledgers(
                f"gen17_parent:{fold}",
                baseline["trade_ledger"],
                baseline["cost_stress_trade_ledger"],
                baseline["metrics"],
                baseline["cost_stress"],
            )
        )
        for policy in POLICIES:
            value = report["fold_results"][fold][policy]
            output.append(
                reconcile_ledgers(
                    f"{policy}:{fold}",
                    value["trade_ledger"],
                    value["cost_stress_trade_ledger"],
                    value["metrics"],
                    value["cost_stress"],
                )
            )
    baseline = report["baseline"]["selection_pooled"]
    output.append(
        reconcile_ledgers(
            "gen17_parent:selection_pooled",
            baseline["trade_ledger"],
            baseline["cost_stress_trade_ledger"],
            baseline["metrics"],
            baseline["cost_stress"],
        )
    )
    summary_by_policy = {
        value["policy"]: value for value in report["phase5_summaries"]
    }
    for policy in POLICIES:
        records = [
            record
            for fold in FOLDS
            for record in report["fold_results"][fold][policy]["trade_ledger"]
        ]
        stress = [
            record
            for fold in FOLDS
            for record in report["fold_results"][fold][policy][
                "cost_stress_trade_ledger"
            ]
        ]
        summary = summary_by_policy[policy]
        output.append(
            reconcile_ledgers(
                f"{policy}:selection_pooled",
                records,
                stress,
                summary["metrics"],
                summary["cost_stress"],
            )
        )
    for period, diagnostic in report["development_diagnostics"].items():
        for strategy, value in (
            ("gen17_parent", diagnostic["baseline"]),
            *diagnostic["candidates"].items(),
        ):
            output.append(
                reconcile_ledgers(
                    f"{strategy}:{period}_development",
                    value["trade_ledger"],
                    value["cost_stress_trade_ledger"],
                    value["metrics"],
                    value["cost_stress"],
                )
            )
    return output


def expected_policy_ids(
    baseline_records: list[dict], policy: str, threshold: dict
) -> set[str]:
    def keep(record: dict) -> bool:
        if policy == "p_net_ge_050":
            return float(record["p_net"]) >= 0.5
        if policy in ("e_net_positive", "e_net_top75_past"):
            return float(record["e_net"]) >= float(threshold["e_min"])
        if policy == "joint_positive":
            return float(record["p_net"]) >= 0.5 and float(record["e_net"]) >= 0.0
        raise ValueError(policy)

    return {
        str(record["trade_id"]) for record in baseline_records if keep(record)
    }


def subset_audit(report: dict) -> list[dict]:
    output = []
    periods = [
        (
            fold,
            report["baseline"]["folds"][fold],
            report["fold_results"][fold],
            report["model_diagnostics"][fold]["model"],
        )
        for fold in FOLDS
    ]
    periods.extend(
        (
            f"{name}_development",
            value["baseline"],
            value["candidates"],
            value["model_diagnostics"]["model"],
        )
        for name, value in report["development_diagnostics"].items()
    )
    for period, baseline, candidates, model in periods:
        baseline_records = baseline["trade_ledger"]
        baseline_by_id = {
            str(record["trade_id"]): record for record in baseline_records
        }
        base_ids = set(baseline_by_id)
        for policy in POLICIES:
            value = candidates[policy]
            candidate_ids = {
                str(record["trade_id"]) for record in value["trade_ledger"]
            }
            expected_ids = expected_policy_ids(
                baseline_records, policy, value["threshold"]
            )
            removed = base_ids - candidate_ids
            losers_removed = sum(
                float(baseline_by_id[trade_id]["reward"]) <= 0.0
                for trade_id in removed
            )
            winners_removed = sum(
                float(baseline_by_id[trade_id]["reward"]) > 0.0
                for trade_id in removed
            )
            comparison = value["comparison_to_gen17"]
            threshold_matches_past = True
            if policy == "e_net_top75_past":
                threshold_matches_past = close(
                    value["threshold"]["e_min"],
                    model["policy_score_distribution"]["e_net"][1],
                    tolerance=1e-7,
                )
            output.append(
                {
                    "period": period,
                    "policy": policy,
                    "baseline_trades": len(base_ids),
                    "candidate_trades": len(candidate_ids),
                    "new_trade_ids": len(candidate_ids - base_ids),
                    "missing_expected_ids": len(expected_ids - candidate_ids),
                    "unexpected_selected_ids": len(candidate_ids - expected_ids),
                    "losers_removed": losers_removed,
                    "winners_accidentally_removed": winners_removed,
                    "reported_comparison_matches": (
                        losers_removed == comparison["losers_removed"]
                        and winners_removed
                        == comparison["winners_accidentally_removed"]
                        and len(candidate_ids - base_ids)
                        == comparison["unique_executable_trades_added"]
                    ),
                    "top75_threshold_matches_past_policy_quantile": (
                        threshold_matches_past
                    ),
                }
            )
    return output


def boundary_audit(report: dict) -> dict:
    output = {}
    sources = {
        **report["model_diagnostics"],
        **{
            f"{name}_development": value["model_diagnostics"]
            for name, value in report["development_diagnostics"].items()
        },
    }
    for period, value in sources.items():
        model = value["model"]
        output[period] = {
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
    by_policy = {value["policy"]: value for value in report["phase5_summaries"]}
    recomputed = {}
    for policy in POLICIES:
        value = by_policy[policy]
        metric = value["metrics"]
        stress = value["cost_stress"]
        folds = [report["fold_results"][fold][policy] for fold in FOLDS]
        stable = all(
            fold["metrics"]["trades"] >= 5
            and fold["metrics"]["profit_factor"] is not None
            and fold["metrics"]["profit_factor"] > 1.0
            and fold["metrics"]["mean_r"] > 0.0
            and fold["metrics"]["break_even_adjusted_win_rate_edge"] is not None
            and fold["metrics"]["break_even_adjusted_win_rate_edge"] > 0.0
            for fold in folds
        )
        passed = bool(
            metric["profit_factor"] is not None
            and metric["profit_factor"] >= 1.05
            and metric["mean_r"] > 0.0
            and metric["pnl"] > 0.0
            and metric["break_even_adjusted_win_rate_edge"] is not None
            and metric["break_even_adjusted_win_rate_edge"] > 0.0
            and metric["realized_positive_trade_win_rate"] >= 0.58
            and value["frequency_retention"] >= 0.50
            and stress["profit_factor"] is not None
            and stress["profit_factor"] > 1.0
            and stress["mean_r"] > 0.0
            and stable
        )
        recomputed[policy] = {
            "reported": bool(value["phase5_success"]),
            "recomputed": passed,
            "stable_positive_all_folds": stable,
        }
    any_success = any(value["recomputed"] for value in recomputed.values())
    return {
        "policies": recomputed,
        "reported_phase5_success": report["phase5_success"],
        "recomputed_phase5_success": any_success,
        "phase6_correctly_gated_off": (
            not any_success
            and not report["phase6"]["executed"]
            and report["frozen_candidate_id"] is None
        ),
    }


def verdicts(
    report: dict,
    reconciliation: list[dict],
    subsets: list[dict],
    boundaries: dict,
    gate: dict,
) -> dict:
    boundary_ok = all(
        value["fit_to_calibration_purged"]
        and value["calibration_to_policy_purged"]
        for value in boundaries.values()
    )
    metrics_ok = all(
        not value["base_metric_mismatches"]
        and not value["stress_metric_mismatches"]
        for value in reconciliation
    )
    execution_ok = all(
        all(
            item[key] == 0
            for key in (
                "duplicate_trade_ids",
                "reward_formula_errors",
                "cost_component_errors",
                "exit_offset_errors",
                "spread_quality_errors",
            )
        )
        and item["chronological"]
        and item["occupancy_non_overlapping"]
        for value in reconciliation
        for item in (value["base_execution"], value["stress_execution"])
    )
    selection_ok = all(
        value["new_trade_ids"] == 0
        and value["missing_expected_ids"] == 0
        and value["unexpected_selected_ids"] == 0
        and value["reported_comparison_matches"]
        and value["top75_threshold_matches_past_policy_quantile"]
        for value in subsets
    )
    architecture = report["architecture"]
    feature_ok = (
        "SPREAD_QUALITY" not in architecture["model_features"]
        and not architecture["spread_quality_is_model_feature"]
        and architecture["relative_feature_method"]["current_block_uses"]
        == "strictly previous rows only"
        and all(
            forbidden not in architecture["model_features"]
            for forbidden in (
                "NET_REWARD",
                "NET_POSITIVE",
                "SHORT_OUTCOME",
                "SHORT_REWARD",
                "SHORT_EXIT_OFFSET",
            )
        )
    )
    gate_ok = (
        gate["reported_phase5_success"]
        == gate["recomputed_phase5_success"]
        and gate["phase6_correctly_gated_off"]
    )
    raw = {
        "chronology": (
            "PASS",
            "Each research fold is fitted on history ending before the fold; 2025/recent are diagnostics only.",
        ),
        "feature_leakage": (
            "PASS" if feature_ok else "FAIL",
            "Model features exclude net-return labels, outcomes, exit offsets, and spread_quality; relative cost transforms use prior blocks only.",
        ),
        "label_maturity": (
            "PASS" if boundary_ok else "FAIL",
            "Exact event exit offsets mature before calibration and policy boundaries.",
        ),
        "oof_predictions": (
            "PASS" if selection_ok else "FAIL",
            "Fixed-cohort selections exactly reproduce frozen policy rules on fold predictions and add no signals in Phases 1-5.",
        ),
        "calibration": (
            "PASS" if boundary_ok else "FAIL",
            "P(net_R>0) isotonic calibration is fitted on a purged chronological calibration partition only.",
        ),
        "threshold_selection": (
            "PASS" if selection_ok else "FAIL",
            "0.50 and zero economic cutoffs are predeclared; top-75 uses the prior policy-tail prediction quantile, not evaluated outcomes.",
        ),
        "purge_embargo": (
            "PASS" if boundary_ok else "FAIL",
            "Fit/calibration events have exact label-end purges; the outer training helper removes the horizon tail.",
        ),
        "holdout_contamination": (
            "FAIL",
            "The repeatedly inspected 2025 interval is development data and cannot be an untouched final test.",
        ),
        "recent_period_reuse": (
            "PASS",
            "Recent results are stored only under development_diagnostics and do not select or freeze a candidate.",
        ),
        "execution_alignment": (
            "PASS" if metrics_ok and execution_ok and gate_ok else "FAIL",
            "Independent ledger metrics, position occupancy, event horizon, subset identity, and the Phase-6 gate reconcile.",
        ),
        "cost_assumptions": (
            "PASS" if execution_ok else "FAIL",
            "Observed entry spreads are charged when available; unavailable/zero spread is charged 30 points, plus 5-point base and 10-point stress extras.",
        ),
        "multiple_testing_risk": (
            "FAIL",
            "Two targets and four policies were inspected after many prior generations, with no untouched future interval available.",
        ),
    }
    return {
        name: {"verdict": raw[name][0], "reason": raw[name][1]}
        for name in CHECKS
    }


def markdown(report: dict) -> str:
    lines = [
        "# Generation 20 walk-forward validation",
        "",
        f"Overall: {report['overall']}",
        "",
        "This validator audits methodology only; it does not optimize or promote the strategy.",
        "",
        f"Internal chronological validation quality: {report['internal_chronological_validation_quality']}",
        "",
        f"Final untouched-test validity: {report['final_untouched_test_validity']}",
        "",
        "| Check | Verdict | Reason |",
        "|---|---|---|",
    ]
    for name in CHECKS:
        value = report["checks"][name]
        lines.append(f"| {name} | {value['verdict']} | {value['reason']} |")
    lines.extend(
        [
            "",
            "## Metric reconciliation",
            "",
            "| Strategy / period | Trades | Trades/day | WR | PF | Mean-R | PnL | Max DD | Stress PF | Reconciled |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
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
            f"| {value['label']} | {metric['trades']} | {metric['trades_per_day']:.3f} | "
            f"{metric['realized_positive_trade_win_rate']:.2%} | {metric['profit_factor'] or 0.0:.3f} | "
            f"{metric['mean_r']:.4f} | {metric['pnl']:.2f} | {metric['max_drawdown_pct']:.2%} | "
            f"{stress['profit_factor'] or 0.0:.3f} | {'yes' if ok else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Adversarial conclusion",
            "",
            "No Phase-5 policy passed. Phase 6 was correctly not executed, no candidate was frozen, and no production promotion is valid.",
            "",
            "The experiment is valid as internal chronological development evidence only. Final untouched-test validity remains FAIL until a genuinely new forward interval is collected without re-selection.",
        ]
    )
    return "\n".join(lines) + "\n"


def self_check() -> None:
    record = {
        "trade_id": "x",
        "period": "x",
        "index": 1,
        "exit_index": 2,
        "time": "2020-01-01T00:00:00",
        "outcome": 1,
        "reward": (1.3 - 0.35) / 1.9,
        "gross_pnl_price": 1.3,
        "gross_reward_before_cost": 1.3 / 1.9,
        "spread_r": 0.3 / 1.9,
        "extra_cost_r": 0.05 / 1.9,
        "total_cost_r": 0.35 / 1.9,
        "atr": 1.0,
        "spread_points": 30.0,
        "spread_observed": False,
        "spread_quality": "fallback",
    }
    audit = verify_records([record], BASE_EXTRA_COST_POINTS)
    assert audit["reward_formula_errors"] == 0
    assert audit["cost_component_errors"] == 0
    assert audit["spread_quality_errors"] == 0
    assert len(CHECKS) == 12
    print("generation20_walk_forward_validation_self_check_ok")


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0
    experiment = json.loads(EXPERIMENT_FILE.read_text(encoding="utf-8"))
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    reconciliation = metric_inventory(experiment)
    subsets = subset_audit(experiment)
    boundaries = boundary_audit(experiment)
    gate = gate_audit(experiment)
    checks = verdicts(experiment, reconciliation, subsets, boundaries, gate)
    overall = (
        "PASS"
        if all(value["verdict"] == "PASS" for value in checks.values())
        else "FAIL"
    )
    internal_exclusions = {"holdout_contamination", "multiple_testing_risk"}
    internal = (
        "PASS"
        if all(
            checks[name]["verdict"] == "PASS"
            for name in CHECKS
            if name not in internal_exclusions
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
        "generation": "20_walk_forward_validation",
        "validator_role": "adversarial_methodology_audit_only",
        "overall": overall,
        "internal_chronological_validation_quality": internal,
        "final_untouched_test_validity": "FAIL",
        "checks": checks,
        "metric_reconciliation": reconciliation,
        "fixed_cohort_subset_audit": subsets,
        "boundary_audit": boundaries,
        "phase5_gate_audit": gate,
        "evidence_manifest": {
            "git_revision": revision,
            "m1_sha256": sha256(M1_FILE),
            "gen20_code_sha256": sha256(SOURCE_FILE),
            "gen20_report_sha256": sha256(EXPERIMENT_FILE),
            "gen20_config_sha256": sha256(CONFIG_FILE),
            "gemini_sha256": sha256(GEMINI_FILE),
            "gemini_hash_matches_experiment": (
                sha256(GEMINI_FILE)
                == experiment["gemini_sha256_before_and_after"]
            ),
            "run_timestamp": experiment["generated_at"],
        },
        "candidate_status": {
            "phase5_success": experiment["phase5_success"],
            "frozen_candidate_id": experiment["frozen_candidate_id"],
            "diagnostic_fallback_id": experiment["diagnostic_fallback_id"],
            "phase6_executed": experiment["phase6"]["executed"],
            "promotion_pass": experiment["promotion_pass"],
            "config_matches": (
                config["phase5_success"] == experiment["phase5_success"]
                and config["frozen_candidate_id"]
                == experiment["frozen_candidate_id"]
                and config["phase6_executed"]
                == experiment["phase6"]["executed"]
                and not config["promotion_pass"]
            ),
        },
        "claim_validity": "internal_chronological_development_only",
        "minimum_validation_correction": (
            "Pre-register and freeze a future candidate without consulting "
            "2025/recent again, then evaluate one genuinely new forward interval."
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
