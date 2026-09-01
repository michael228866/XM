from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT = PROJECT_ROOT / "gold_generation18_payoff_alignment.json"
CALIBRATION = PROJECT_ROOT / "gold_generation18_calibration_drift.json"
PAYOFF_AUDIT = PROJECT_ROOT / "gold_generation18_payoff_audit.json"
GEN17_REPORT = PROJECT_ROOT / "gold_generation17_cross_regime.json"
EXPERIMENT_SOURCE = PROJECT_ROOT / "gold_generation18_payoff_alignment.py"
PAYOFF_SOURCE = PROJECT_ROOT / "gold_generation18_payoff_audit.py"
GEN17_SOURCE = PROJECT_ROOT / "gold_generation17_cross_regime.py"
TRAINING_SOURCE = PROJECT_ROOT / "gold_regime_experts_iterative.py"
HISTORICAL_FEATURE_SOURCE = PROJECT_ROOT / "drl_trading_v2.py"
RECENT_FEATURE_SOURCE = PROJECT_ROOT / "gold_recent_walk_forward.py"
EXECUTION_SOURCE = PROJECT_ROOT / "gold_generation16_independent_families.py"
FIRST_TOUCH_SOURCE = PROJECT_ROOT / "barrier_classifier_strategy.py"
REPORT_JSON = PROJECT_ROOT / "gold_generation18_walk_forward_validation.json"
REPORT_MD = PROJECT_ROOT / "gold_generation18_walk_forward_validation.md"

RISK_PER_TRADE = 0.014
TOLERANCE = 1e-8
SELECTION_FOLDS = ("2018_2020", "2021_2022", "2023_2024")
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


def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profit_factor(rewards: np.ndarray) -> float | None:
    gains = float(rewards[rewards > 0.0].sum())
    losses = float(-rewards[rewards < 0.0].sum())
    return None if losses <= 0.0 else gains / losses


def recompute(ledger: list[dict], evaluated_days: int) -> dict:
    outcomes = np.asarray(
        [int(record["outcome"]) for record in ledger], dtype=np.int8
    )
    rewards = np.asarray(
        [float(record["reward"]) for record in ledger], dtype=np.float64
    )
    positive = rewards[rewards > 0.0]
    nonpositive = rewards[rewards <= 0.0]
    average_winner = float(positive.mean()) if len(positive) else None
    average_loser = float(nonpositive.mean()) if len(nonpositive) else None
    payoff_ratio = (
        None
        if average_winner is None or average_loser is None or average_loser == 0.0
        else average_winner / abs(average_loser)
    )
    break_even = None if payoff_ratio is None else 1.0 / (1.0 + payoff_ratio)
    trades = len(ledger)
    realized_win = int((rewards > 0.0).sum()) / max(trades, 1)
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
        "trades": trades,
        "evaluated_days": evaluated_days,
        "trades_per_day": trades / max(evaluated_days, 1),
        "tp_first_rate": int((outcomes == 1).sum()) / max(trades, 1),
        "realized_positive_trade_win_rate": realized_win,
        "tp_exits": int((outcomes == 1).sum()),
        "sl_exits": int((outcomes == 2).sum()),
        "timeout_exits": int((outcomes == 0).sum()),
        "average_winning_r": average_winner,
        "average_losing_r": average_loser,
        "payoff_ratio": payoff_ratio,
        "realized_break_even_win_rate": break_even,
        "break_even_adjusted_win_rate_edge": (
            None if break_even is None else realized_win - break_even
        ),
        "profit_factor": profit_factor(rewards),
        "mean_r": float(rewards.mean()) if trades else 0.0,
        "sum_r": float(rewards.sum()) if trades else 0.0,
        "pnl": pnl,
        "max_drawdown_pct": drawdown,
    }


def equal(left, right) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(
            float(left), float(right), rel_tol=1e-8, abs_tol=TOLERANCE
        )
    return left == right


def reconcile(claimed: dict, ledger: list[dict]) -> tuple[bool, dict]:
    value = recompute(ledger, int(claimed["evaluated_days"]))
    differences = {
        key: {"claimed": claimed.get(key), "recomputed": result}
        for key, result in value.items()
        if not equal(claimed.get(key), result)
    }
    return not differences, {
        "claimed": {key: claimed.get(key) for key in value},
        "recomputed": value,
        "differences": differences,
    }


def non_overlapping(ledger: list[dict]) -> bool:
    ordered = sorted(ledger, key=lambda record: int(record["index"]))
    return all(
        int(current["index"]) > int(previous["exit_index"])
        for previous, current in zip(ordered, ordered[1:])
    )


def check_boundaries(groups: dict) -> tuple[bool, list[str]]:
    failures = []
    for period, experts in groups.items():
        for expert, value in experts.items():
            if value["fit_max_label_end_index"] >= value["calibration_start_index"]:
                failures.append(f"{period}/{expert}: fit crosses calibration")
            if value["calibration_max_label_end_index"] >= value["policy_start_index"]:
                failures.append(f"{period}/{expert}: calibration crosses policy")
    return not failures, failures


def current_dataset_manifest() -> list[dict]:
    output = []
    for path in sorted(PROJECT_ROOT.glob("GOLD#_*.csv")):
        stat = path.stat()
        output.append(
            {
                "file": path.name,
                "size_bytes": stat.st_size,
                "modified_utc": datetime.fromtimestamp(
                    stat.st_mtime, timezone.utc
                ).isoformat(),
            }
        )
    return output


def verdict(status: str, evidence: str, correction: str = "None.") -> dict:
    if status not in {"PASS", "FAIL"}:
        raise ValueError(status)
    return {
        "verdict": status,
        "evidence": evidence,
        "failure_or_reason_for_pass": evidence,
        "required_validation_correction": correction,
    }


def markdown(report: dict) -> str:
    lines = [
        "# Generation 18 walk-forward adversarial validation",
        "",
        f"Overall: **{report['overall_verdict']}**",
        "",
        "| Validation scope | Verdict |",
        "|---|---|",
        f"| Internal chronological validation quality | {report['internal_chronological_validation_quality']} |",
        f"| Final untouched-test validity | {report['final_untouched_test_validity']} |",
        "",
        "| Check | Verdict | Evidence | Failure or reason for pass | Required validation correction |",
        "|---|---|---|---|---|",
    ]
    for name in CHECK_NAMES:
        item = report["checks"][name]
        lines.append(
            f"| {name} | {item['verdict']} | {item['evidence']} | "
            f"{item['failure_or_reason_for_pass']} | "
            f"{item['required_validation_correction']} |"
        )
    lines.extend(
        [
            "",
            "## Frozen diagnostic reconciliation",
            "",
            "| Period | Profile | Trades | Trades/day | TP-first | Realized win | Payoff | Break-even | Edge | PF | Mean-R | PnL | Max DD | Match |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for period, profiles in report["metric_reconciliation"].items():
        for profile, item in profiles.items():
            value = item["recomputed"]
            lines.append(
                f"| {period} | {profile} | {value['trades']} | "
                f"{value['trades_per_day']:.3f} | "
                f"{value['tp_first_rate']:.2%} | "
                f"{value['realized_positive_trade_win_rate']:.2%} | "
                f"{value['payoff_ratio'] or 0.0:.4f} | "
                f"{value['realized_break_even_win_rate'] or 0.0:.2%} | "
                f"{value['break_even_adjusted_win_rate_edge'] or 0.0:.2%} | "
                f"{value['profit_factor'] or 0.0:.2f} | "
                f"{value['mean_r']:.4f} | {value['pnl']:.2f} | "
                f"{value['max_drawdown_pct']:.2%} | "
                f"{'PASS' if not item['differences'] else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            "The validator did not optimize, rank, retrain, or alter the submitted strategy.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    experiment = json.loads(load_text(EXPERIMENT))
    calibration = json.loads(load_text(CALIBRATION))
    payoff_audit = json.loads(load_text(PAYOFF_AUDIT))
    gen17_report = json.loads(load_text(GEN17_REPORT))
    source = load_text(EXPERIMENT_SOURCE)
    payoff_source = load_text(PAYOFF_SOURCE)
    gen17_source = load_text(GEN17_SOURCE)
    training_source = load_text(TRAINING_SOURCE)
    historical_features = load_text(HISTORICAL_FEATURE_SOURCE)
    recent_features = load_text(RECENT_FEATURE_SOURCE)
    execution_source = load_text(EXECUTION_SOURCE)
    first_touch_source = load_text(FIRST_TOUCH_SOURCE)

    reconciliation = {}
    ledger_failures = []
    metric_failures = []
    all_results = []
    for fold in SELECTION_FOLDS:
        all_results.extend(
            (f"{fold}/{candidate_id}", value)
            for candidate_id, value in experiment["selection"][
                "candidate_fold_results"
            ][fold].items()
        )
    all_results.extend(experiment["selected"]["results"].items())
    seen = set()
    for label, value in all_results:
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        ok, detail = reconcile(value["payoff"], value["trade_ledger"])
        if not ok:
            metric_failures.append(label)
        if not non_overlapping(value["trade_ledger"]):
            ledger_failures.append(f"{label}/base overlap")
        if not non_overlapping(value["cost_stress_trade_ledger"]):
            ledger_failures.append(f"{label}/stress overlap")
        if [row["trade_id"] for row in value["trade_ledger"]] != [
            row["trade_id"] for row in value["cost_stress_trade_ledger"]
        ]:
            ledger_failures.append(f"{label}: stress changed entries")

    for period, value in experiment["selected"]["results"].items():
        ok, base = reconcile(value["payoff"], value["trade_ledger"])
        stress_claim = recompute(
            value["cost_stress_trade_ledger"],
            int(value["cost_stress"]["evaluated_days"]),
        )
        for key in (
            "trades",
            "evaluated_days",
            "trades_per_day",
            "profit_factor",
            "pnl",
            "sum_r",
            "mean_r",
            "max_drawdown_pct",
        ):
            stress_claim[key] = value["cost_stress"][key]
        stress_ok, stress = reconcile(
            stress_claim, value["cost_stress_trade_ledger"]
        )
        reconciliation[period] = {"base": base, "cost_stress": stress}
        if not ok or not stress_ok:
            metric_failures.append(f"selected/{period}")

    all_boundaries = dict(experiment["model_diagnostics"])
    all_boundaries.update(experiment["diagnostic_model_diagnostics"])
    boundary_ok, boundary_failures = check_boundaries(all_boundaries)
    policy_inventory_ok = all(
        value["candidate_count"] == 12
        and len(value["all_candidates"]) == 12
        and value["selected"] in value["all_candidates"]
        for fold in SELECTION_FOLDS
        for value in experiment["inner_policy_selection"][fold].values()
    )
    selection_before_diagnostics = (
        source.index("ranked = aggregate_candidates")
        < source.index("holdout = history[")
        < source.index("mt5.initialize")
    )
    block_past_only = (
        source.index("ranks = rank_values(block, history, mode)")
        < source.index("history = pd.concat(")
    )
    absolute_reproduced = all(
        calibration["absolute_gen17_ledger_reproduced"].values()
    )
    dataset_match = (
        experiment["data"]["historical_dataset_manifest"]
        == current_dataset_manifest()
    )
    payoff_key = payoff_audit["strategies"][
        "gen17_short_trend_diagnostic"
    ]["selection_pooled"]
    gen17_ledger = [
        record
        for fold in SELECTION_FOLDS
        for record in gen17_report["selected"]["results"][fold]["trade_ledger"]
    ]
    payoff_recomputed = recompute(gen17_ledger, payoff_key["evaluated_days"])
    payoff_semantics_ok = all(
        equal(payoff_key[key], payoff_recomputed[key])
        for key in payoff_recomputed
        if key in payoff_key
    )

    checks = {
        "chronology": verdict(
            "PASS" if selection_before_diagnostics and block_past_only else "FAIL",
            "Each outer fold trains before fold_start; inner rank policy is chosen in prior policy data; block ranks are computed before the current block is appended; 2025/recent run after frozen candidate ranking.",
            "Move every adaptive choice before its scored block." if not block_past_only else "None.",
        ),
        "feature leakage": verdict(
            "PASS"
            if "df[full_features] = df[full_features].shift(1)" in historical_features
            and "frame[features] = frame[features].shift(1)" in recent_features
            and "maturity = indices + exits + 1" in gen17_source
            else "FAIL",
            "M1/MTF inputs are shifted, Gen17 normalized features remain entry-time known, and past family outcomes enter only after exit maturity+1. Outcomes in calibration reports are diagnostic labels, not rank inputs.",
        ),
        "label maturity": verdict(
            "PASS" if boundary_ok else "FAIL",
            "Fit label ends precede calibration starts and calibration label ends precede policy starts in every selection and diagnostic model."
            if boundary_ok
            else "; ".join(boundary_failures),
            "Purge every crossing label and regenerate frozen scores." if not boundary_ok else "None.",
        ),
        "OOF predictions": verdict(
            "PASS" if boundary_ok and absolute_reproduced else "FAIL",
            "Outer scores come from models fit/calibrated before each fold; inner rank policies use the disjoint policy segment. All three reconstructed absolute short-trend ledgers exactly match Gen17.",
        ),
        "calibration": verdict(
            "PASS" if boundary_ok else "FAIL",
            "Isotonic calibration is fit only on the chronological calibration partition. Reliability curves and deciles use outer outcomes for diagnosis only and never update a candidate cutoff.",
        ),
        "threshold selection": verdict(
            "PASS"
            if policy_inventory_ok
            and selection_before_diagnostics
            and experiment["architecture"]["absolute_threshold_sweep"] is False
            else "FAIL",
            "Exactly 12 predeclared inner policies per expert (3 rank modes x top 1/2/5/10%) are selected before the next outer fold; four fixed expert portfolios are compared only on 2018-2024 development folds.",
            "Regenerate a complete predeclared inner inventory." if not policy_inventory_ok else "None.",
        ),
        "purge/embargo": verdict(
            "PASS" if ".iloc[:-HORIZON]" in training_source and boundary_ok else "FAIL",
            "Outer training drops H=90 rows and inner fit/calibration boundaries use exact exit label ends.",
        ),
        "holdout contamination": verdict(
            "FAIL",
            "The experiment explicitly retains 2025 as repeatedly inspected development/diagnostic data; no untouched historical outer test exists.",
            "Freeze a future qualified candidate before collecting a genuinely new embargoed forward interval.",
        ),
        "recent-period reuse": verdict(
            "PASS" if selection_before_diagnostics else "FAIL",
            "Recent data are evaluated only after selection and explicitly excluded from candidate choice and promotion claims.",
        ),
        "execution alignment": verdict(
            "PASS"
            if not ledger_failures
            and not metric_failures
            and "for offset in range(1, horizon + 1)" in first_touch_source
            and "take_event = active & take_hit & ~stop_hit" in first_touch_source
            and "free_index = index + exit_offset + 1" in execution_source
            else "FAIL",
            "All four candidate paths and the frozen diagnostic reconcile from executable ledgers; HIGH/LOW first-touch starts after entry, ties are stop-first, and single-position occupancy is enforced."
            if not ledger_failures and not metric_failures
            else "; ".join(ledger_failures + metric_failures),
            "Correct ledger semantics only; do not retune." if ledger_failures or metric_failures else "None.",
        ),
        "cost assumptions": verdict(
            "PASS" if payoff_semantics_ok and not metric_failures else "FAIL",
            "Phase 1 maps each ledger entry back to ATR, outcome, reward, and exit offset; base reward applies 30 spread + 5 extra points, stress applies 10 extra points with identical entries. TP-first and positive-net-return rates are separately reconciled.",
            "Repair payoff semantics and rerun the same frozen ledgers." if not payoff_semantics_ok else "None.",
        ),
        "multiple-testing risk": verdict(
            "FAIL",
            "Twelve inner rank policies per expert, four portfolios, prior generations, reused 2025, and monitored recent data have all been inspected; there is no untouched final test or selection-aware final claim.",
            "Pre-register one qualified candidate and obtain a new untouched forward outer interval without further tuning on it.",
        ),
    }
    internal_checks = (
        "chronology",
        "feature leakage",
        "label maturity",
        "OOF predictions",
        "calibration",
        "threshold selection",
        "purge/embargo",
        "recent-period reuse",
        "execution alignment",
        "cost assumptions",
    )
    internal_quality = (
        "PASS"
        if dataset_match
        and all(checks[name]["verdict"] == "PASS" for name in internal_checks)
        else "FAIL"
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT.name,
        "submitted_candidate_status": experiment["selected"]["status"],
        "discovery_qualified_count": experiment["selection"]["qualified_count"],
        "internal_chronological_validation_quality": internal_quality,
        "final_untouched_test_validity": "FAIL",
        "overall_verdict": "FAIL",
        "performance_claim": "invalid_no_untouched_test_and_no_qualified_candidate",
        "checks": checks,
        "metric_reconciliation": reconciliation,
        "dataset_manifest_matches_run": dataset_match,
        "payoff_semantics_reconciled": payoff_semantics_ok,
        "absolute_gen17_ledger_reproduced": absolute_reproduced,
        "ledger_failures": ledger_failures,
        "metric_failures": metric_failures,
        "source_hashes": {
            path.name: sha256(path)
            for path in (
                EXPERIMENT_SOURCE,
                PAYOFF_SOURCE,
                GEN17_SOURCE,
                TRAINING_SOURCE,
                HISTORICAL_FEATURE_SOURCE,
                RECENT_FEATURE_SOURCE,
                EXECUTION_SOURCE,
                FIRST_TOUCH_SOURCE,
            )
        },
        "validator_scope": "validation_only_no_optimization",
    }
    if set(checks) != set(CHECK_NAMES):
        raise RuntimeError("Validator check inventory differs from required checks")
    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REPORT_MD.write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    print(f"Saved {REPORT_JSON.name}, {REPORT_MD.name}")
    return 0 if not ledger_failures and not metric_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
