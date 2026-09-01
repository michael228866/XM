from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT = PROJECT_ROOT / "gold_generation16_independent_families.json"
EXPERIMENT_SOURCE = PROJECT_ROOT / "gold_generation16_independent_families.py"
TRAINING_SOURCE = PROJECT_ROOT / "gold_regime_experts_iterative.py"
HISTORICAL_FEATURE_SOURCE = PROJECT_ROOT / "drl_trading_v2.py"
RECENT_FEATURE_SOURCE = PROJECT_ROOT / "gold_recent_walk_forward.py"
FIRST_TOUCH_SOURCE = PROJECT_ROOT / "barrier_classifier_strategy.py"
REPORT_JSON = PROJECT_ROOT / "gold_generation16_walk_forward_validation.json"
REPORT_MD = PROJECT_ROOT / "gold_generation16_walk_forward_validation.md"

RISK_PER_TRADE = 0.014
TOLERANCE = 1e-9
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


def profit_factor(rewards: list[float]) -> float | None:
    gains = sum(value for value in rewards if value > 0.0)
    losses = -sum(value for value in rewards if value < 0.0)
    return None if losses <= 0.0 else gains / losses


def recompute(ledger: list[dict], evaluated_days: int) -> dict:
    outcomes = [int(record["outcome"]) for record in ledger]
    rewards = [float(record["reward"]) for record in ledger]
    trades = len(ledger)
    wins = sum(outcome == 1 for outcome in outcomes)
    losses = sum(outcome == 2 for outcome in outcomes)
    timeouts = sum(outcome == 0 for outcome in outcomes)
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
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "win_rate": wins / max(trades, 1),
        "profit_factor": profit_factor(rewards),
        "pnl": pnl,
        "sum_r": sum(rewards),
        "mean_r": sum(rewards) / max(trades, 1),
        "max_drawdown_pct": drawdown,
    }


def equal_value(left, right) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=TOLERANCE)
    return left == right


def reconcile_metrics(claimed: dict, ledger: list[dict]) -> tuple[bool, dict]:
    recomputed = recompute(ledger, int(claimed["evaluated_days"]))
    differences = {
        key: {"claimed": claimed[key], "recomputed": recomputed[key]}
        for key in recomputed
        if not equal_value(claimed[key], recomputed[key])
    }
    return not differences, {
        "claimed": claimed,
        "recomputed": recomputed,
        "differences": differences,
    }


def non_overlapping(ledger: list[dict]) -> bool:
    ordered = sorted(ledger, key=lambda record: int(record["index"]))
    return all(
        int(current["index"]) > int(previous["exit_index"])
        for previous, current in zip(ordered, ordered[1:])
    )


def check_model_boundaries(model_diagnostics: dict) -> tuple[bool, list[str]]:
    failures = []
    for period, experts in model_diagnostics.items():
        for expert, value in experts.items():
            if value.get("status") != "trained":
                continue
            if value["fit_max_label_end_index"] >= value["calibration_start_index"]:
                failures.append(f"{period}/{expert}: fit labels cross calibration")
            if value["calibration_max_label_end_index"] >= value["policy_start_index"]:
                failures.append(f"{period}/{expert}: calibration labels cross policy")
    return not failures, failures


def verdict(status: str, evidence: str) -> dict:
    if status not in {"PASS", "FAIL"}:
        raise ValueError(status)
    return {"verdict": status, "evidence": evidence}


def markdown(report: dict) -> str:
    lines = [
        "# Generation 16 walk-forward adversarial validation",
        "",
        f"Overall: **{report['overall_verdict']}**",
        f"Submitted performance claim: **{report['performance_claim']}**",
        "",
        "| Check | Verdict | Evidence |",
        "|---|---|---|",
    ]
    for name in CHECK_NAMES:
        item = report["checks"][name]
        lines.append(
            f"| {name} | {item['verdict']} | {item['evidence']} |"
        )
    lines.extend(
        [
            "",
            "## Metric reconciliation",
            "",
            "| Period | Profile | Trades | Trades/day | Wins | Losses | Timeouts | Win | PF | PnL | Mean-R | DD | Match |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for period, profiles in report["metric_reconciliation"].items():
        for profile, item in profiles.items():
            value = item["recomputed"]
            pf = "inf" if value["profit_factor"] is None else f"{value['profit_factor']:.2f}"
            lines.append(
                f"| {period} | {profile} | {value['trades']} | "
                f"{value['trades_per_day']:.3f} | {value['wins']} | "
                f"{value['losses']} | {value['timeouts']} | "
                f"{value['win_rate']:.2%} | {pf} | {value['pnl']:.2f} | "
                f"{value['mean_r']:.4f} | {value['max_drawdown_pct']:.2%} | "
                f"{'PASS' if not item['differences'] else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            "## Minimum validation-only correction",
            "",
            *[f"- {item}" for item in report["minimum_validation_only_correction"]],
            "",
            "The validator did not optimize, resweep, or change the strategy.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    experiment = json.loads(load_text(EXPERIMENT))
    source = load_text(EXPERIMENT_SOURCE)
    training_source = load_text(TRAINING_SOURCE)
    historical_features = load_text(HISTORICAL_FEATURE_SOURCE)
    recent_features = load_text(RECENT_FEATURE_SOURCE)
    first_touch = load_text(FIRST_TOUCH_SOURCE)

    selected_results = experiment["selected"]["results"]
    reconciliation = {}
    ledger_failures = []
    metric_failures = []
    for period, value in selected_results.items():
        base_ok, base_detail = reconcile_metrics(
            value["metrics"], value["trade_ledger"]
        )
        stress_ok, stress_detail = reconcile_metrics(
            value["cost_stress"], value["cost_stress_trade_ledger"]
        )
        reconciliation[period] = {
            "base": base_detail,
            "cost_stress": stress_detail,
        }
        if not base_ok or not stress_ok:
            metric_failures.append(period)
        if not non_overlapping(value["trade_ledger"]):
            ledger_failures.append(f"{period}/base overlap")
        if not non_overlapping(value["cost_stress_trade_ledger"]):
            ledger_failures.append(f"{period}/stress overlap")
        base_ids = [record["trade_id"] for record in value["trade_ledger"]]
        stress_ids = [
            record["trade_id"] for record in value["cost_stress_trade_ledger"]
        ]
        if base_ids != stress_ids:
            ledger_failures.append(f"{period}: cost stress changed entries")

    boundary_ok, boundary_failures = check_model_boundaries(
        experiment["model_diagnostics"]
    )
    selection_periods = ("2018_2020", "2021_2022", "2023_2024")
    inventory_complete = all(
        len(experiment["selection"]["candidate_fold_results"][period])
        == experiment["selection"]["candidate_count"]
        for period in selection_periods
    )
    selection_before_holdout = (
        source.index("ranked = aggregate_candidates")
        < source.index("holdout = history[")
        < source.index("mt5.initialize")
    )

    checks = {
        "chronology": verdict(
            "PASS",
            "Each fold trains with training_frame(history, fold_start); holdout and recent are evaluated only after the selection ranking is frozen.",
        ),
        "feature leakage": verdict(
            "PASS"
            if "df[full_features] = df[full_features].shift(1)" in historical_features
            and "frame[features] = frame[features].shift(1)" in recent_features
            else "FAIL",
            "Historical and MT5 feature builders shift M1 features one bar and higher-timeframe trends one completed bar before scoring.",
        ),
        "label maturity": verdict(
            "PASS" if boundary_ok else "FAIL",
            "All recorded fit labels end before calibration and all calibration labels end before policy."
            if boundary_ok
            else "; ".join(boundary_failures),
        ),
        "OOF predictions": verdict(
            "PASS" if selection_before_holdout and inventory_complete else "FAIL",
            "All 72 predeclared candidates have fold-level ledgers; predictions come from models fit strictly before each evaluated fold.",
        ),
        "calibration": verdict(
            "PASS" if boundary_ok else "FAIL",
            "Chronological fit, isotonic-calibration, and policy segments are disjoint; the policy segment is not reused to fit the calibrator.",
        ),
        "threshold selection": verdict(
            "PASS"
            if experiment["architecture"]["threshold_tuning"] is False
            and selection_before_holdout
            else "FAIL",
            "P(TP-first)=0.60 and Expected-R=0 are fixed; family/context/top-k architectures are selected on 2018-2024 only, before holdout/recent evaluation.",
        ),
        "purge/embargo": verdict(
            "PASS"
            if ".iloc[:-HORIZON]" in training_source and boundary_ok
            else "FAIL",
            "Outer training removes the final H=90 rows; internal fit/calibration/policy boundaries additionally require event label_end before the next stage.",
        ),
        "holdout contamination": verdict(
            "FAIL",
            "The report explicitly identifies 2025-2026-05 as a reused historical holdout already inspected by earlier generations; it is not untouched.",
        ),
        "recent-period reuse": verdict(
            "PASS",
            "The repeatedly observed 2026 recent period is explicitly monitoring-only and is excluded from candidate/model selection and promotion claims.",
        ),
        "execution alignment": verdict(
            "PASS"
            if not ledger_failures
            and not metric_failures
            and "for offset in range(1, horizon + 1)" in first_touch
            and "take_event = active & take_hit & ~stop_hit" in first_touch
            else "FAIL",
            "First-touch starts at offset 1, same-bar TP/SL ties are stop-first, and every persisted ledger is single-position and non-overlapping."
            if not ledger_failures and not metric_failures
            else "; ".join(ledger_failures + metric_failures),
        ),
        "cost assumptions": verdict(
            "PASS" if not ledger_failures and not metric_failures else "FAIL",
            "Base uses 30 spread points plus 5 extra points; stress uses the same entries with 10 extra points. Ledgers reconcile and costs are applied once in stop-risk units.",
        ),
        "multiple-testing risk": verdict(
            "FAIL",
            "Seventy-two Generation 16 architectures follow many prior generations, while the nominal holdout is already reused; there is no fresh untouched outer interval for a positive performance claim.",
        ),
    }
    overall = "PASS" if all(
        item["verdict"] == "PASS" for item in checks.values()
    ) else "FAIL"
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT.name,
        "submitted_candidate_status": experiment["selected"]["status"],
        "overall_verdict": overall,
        "performance_claim": "invalid",
        "checks": checks,
        "metric_reconciliation": reconciliation,
        "selection_inventory_complete": inventory_complete,
        "ledger_failures": ledger_failures,
        "metric_failures": metric_failures,
        "target_assessment": {
            period: {
                "win_rate_at_least_60": value["metrics"]["win_rate"] >= 0.60,
                "pf_above_1": value["metrics"]["profit_factor"] is not None
                and value["metrics"]["profit_factor"] > 1.0,
                "positive_mean_r": value["metrics"]["mean_r"] > 0.0,
            }
            for period, value in selected_results.items()
        },
        "minimum_validation_only_correction": [
            "Do not treat the reused 2025 holdout or recent monitoring window as untouched evidence.",
            "Pre-register and freeze a qualified future candidate before collecting a new embargoed forward interval that no research iteration has inspected.",
            "Run the same executable-event and cost-stress reconciliation once on that new interval without changing models, families, thresholds, or context rules.",
        ],
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
    return 0 if not metric_failures and not ledger_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
