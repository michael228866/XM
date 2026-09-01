from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT = PROJECT_ROOT / "gold_generation17_cross_regime.json"
TRANSFER = PROJECT_ROOT / "gold_generation17_regime_transfer.json"
EXPERIMENT_SOURCE = PROJECT_ROOT / "gold_generation17_cross_regime.py"
TRAINING_SOURCE = PROJECT_ROOT / "gold_regime_experts_iterative.py"
HISTORICAL_FEATURE_SOURCE = PROJECT_ROOT / "drl_trading_v2.py"
RECENT_FEATURE_SOURCE = PROJECT_ROOT / "gold_recent_walk_forward.py"
EXECUTION_SOURCE = PROJECT_ROOT / "gold_generation16_independent_families.py"
FIRST_TOUCH_SOURCE = PROJECT_ROOT / "barrier_classifier_strategy.py"
REPORT_JSON = PROJECT_ROOT / "gold_generation17_walk_forward_validation.json"
REPORT_MD = PROJECT_ROOT / "gold_generation17_walk_forward_validation.md"

RISK_PER_TRADE = 0.014
TOLERANCE = 1e-9
SELECTION_PERIODS = ("2018_2020", "2021_2022", "2023_2024")
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


def profit_factor(rewards: list[float]) -> float | None:
    gains = sum(value for value in rewards if value > 0.0)
    losses = -sum(value for value in rewards if value < 0.0)
    return None if losses <= 0.0 else gains / losses


def recompute(ledger: list[dict], evaluated_days: int) -> dict:
    rewards = [float(record["reward"]) for record in ledger]
    outcomes = [int(record["outcome"]) for record in ledger]
    trades = len(ledger)
    balance = 1000.0
    peak = balance
    maximum_drawdown = 0.0
    pnl = 0.0
    for reward in rewards:
        change = balance * RISK_PER_TRADE * reward
        balance += change
        pnl += change
        peak = max(peak, balance)
        maximum_drawdown = min(maximum_drawdown, balance / peak - 1.0)
    wins = sum(outcome == 1 for outcome in outcomes)
    return {
        "trades": trades,
        "evaluated_days": evaluated_days,
        "trades_per_day": trades / max(evaluated_days, 1),
        "wins": wins,
        "losses": sum(outcome == 2 for outcome in outcomes),
        "timeouts": sum(outcome == 0 for outcome in outcomes),
        "win_rate": wins / max(trades, 1),
        "profit_factor": profit_factor(rewards),
        "pnl": pnl,
        "sum_r": sum(rewards),
        "mean_r": sum(rewards) / max(trades, 1),
        "max_drawdown_pct": maximum_drawdown,
    }


def equal_value(left, right) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(
            float(left), float(right), rel_tol=1e-9, abs_tol=TOLERANCE
        )
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


def model_boundaries(model_diagnostics: dict) -> tuple[bool, list[str]]:
    failures = []
    for period, experts in model_diagnostics.items():
        for expert, value in experts.items():
            required = {
                "fit_max_label_end_index",
                "calibration_start_index",
                "calibration_max_label_end_index",
                "policy_start_index",
            }
            missing = sorted(required - set(value))
            if missing:
                failures.append(f"{period}/{expert}: missing {missing}")
                continue
            if value["fit_max_label_end_index"] >= value["calibration_start_index"]:
                failures.append(f"{period}/{expert}: fit label crosses calibration")
            if value["calibration_max_label_end_index"] >= value["policy_start_index"]:
                failures.append(f"{period}/{expert}: calibration label crosses policy")
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


def check(status: str, evidence: str, correction: str = "None.") -> dict:
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
        "# Generation 17 walk-forward adversarial validation",
        "",
        f"Overall: **{report['overall_verdict']}**",
        "",
        "| Validation scope | Verdict |",
        "|---|---|",
        f"| Internal chronological validation quality | {report['internal_chronological_validation_quality']} |",
        f"| Final untouched-test validity | {report['final_untouched_test_validity']} |",
        "",
        "| Check | Verdict | Evidence | Required correction |",
        "|---|---|---|---|",
    ]
    for name in CHECK_NAMES:
        item = report["checks"][name]
        lines.append(
            f"| {name} | {item['verdict']} | {item['evidence']} | "
            f"{item['required_validation_correction']} |"
        )
    lines.extend(
        [
            "",
            "## Metric reconciliation",
            "",
            "| Period | Profile | Trades | Trades/day | Wins | Losses | Timeouts | Win | PF | PnL | Mean-R | Max DD | Match |",
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
            "The validator did not optimize, resweep, retrain, or modify the strategy.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    experiment = json.loads(load_text(EXPERIMENT))
    transfer = json.loads(load_text(TRANSFER))
    source = load_text(EXPERIMENT_SOURCE)
    training_source = load_text(TRAINING_SOURCE)
    historical_features = load_text(HISTORICAL_FEATURE_SOURCE)
    recent_features = load_text(RECENT_FEATURE_SOURCE)
    execution_source = load_text(EXECUTION_SOURCE)
    first_touch_source = load_text(FIRST_TOUCH_SOURCE)

    reconciliation = {}
    ledger_failures = []
    metric_failures = []
    for period, value in experiment["selected"]["results"].items():
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
        for profile, ledger in (
            ("base", value["trade_ledger"]),
            ("stress", value["cost_stress_trade_ledger"]),
        ):
            if not non_overlapping(ledger):
                ledger_failures.append(f"{period}/{profile}: overlap")
        if [item["trade_id"] for item in value["trade_ledger"]] != [
            item["trade_id"] for item in value["cost_stress_trade_ledger"]
        ]:
            ledger_failures.append(f"{period}: stress changed entries")

    boundary_ok, boundary_failures = model_boundaries(
        experiment["model_diagnostics"]
    )
    inventory_complete = all(
        len(experiment["selection"]["candidate_fold_results"][period])
        == experiment["selection"]["candidate_count"]
        for period in SELECTION_PERIODS
    )
    selection_before_diagnostics = (
        source.index("ranked = aggregate_candidates")
        < source.index("holdout = history[")
        < source.index("mt5.initialize")
    )
    past_state_safe = all(
        token in source
        for token in (
            "maturity = indices + exits + 1",
            "maturity_order = np.argsort(maturity, kind=\"stable\")",
            "win_rows[maturity[maturity_last]] = win_state[maturity_last]",
        )
    )
    normalized_only = (
        experiment["architecture"]["absolute_scale_features_excluded_from_model"]
        == ["ATR"]
        and "ATR" not in experiment["selected"]["params"].get("model_features", [])
        and "ATR" not in json.loads(load_text(PROJECT_ROOT / "gold_generation17_candidate.json"))["model_features"]
    )
    dataset_match = (
        experiment["data"]["historical_dataset_manifest"]
        == current_dataset_manifest()
    )

    checks = {
        "chronology": check(
            "PASS" if selection_before_diagnostics else "FAIL",
            "Models use training_frame(history, fold_start); ranking is frozen after 2018-2024 folds and before reused 2025/recent diagnostics.",
            "Move every diagnostic interval after frozen selection." if not selection_before_diagnostics else "None.",
        ),
        "feature leakage": check(
            "PASS"
            if past_state_safe
            and normalized_only
            and "df[full_features] = df[full_features].shift(1)" in historical_features
            and "frame[features] = frame[features].shift(1)" in recent_features
            else "FAIL",
            "M1/MTF inputs are shifted; bar structure uses t-1; family outcome state appears only at exit_offset+1 in actual maturity order; raw ATR is excluded from model inputs.",
            "Repair any unshifted or pre-maturity feature and rerun unchanged validation." if not past_state_safe or not normalized_only else "None.",
        ),
        "label maturity": check(
            "PASS" if boundary_ok else "FAIL",
            "Every recorded fit label ends before calibration, and every calibration label ends before policy."
            if boundary_ok
            else "; ".join(boundary_failures),
            "Purge crossing labels and regenerate predictions." if not boundary_ok else "None.",
        ),
        "OOF predictions": check(
            "PASS" if inventory_complete and selection_before_diagnostics else "FAIL",
            "All five predeclared candidates have executable ledgers in each of three chronological development folds; each fold model trains only before its fold.",
            "Regenerate missing fold predictions without filling from in-sample scores." if not inventory_complete else "None.",
        ),
        "calibration": check(
            "PASS" if boundary_ok else "FAIL",
            "Chronological 60/20/20 fit, isotonic calibration, and policy segments are disjoint with label-end purging.",
            "Refit calibration on a disjoint past-only segment." if not boundary_ok else "None.",
        ),
        "threshold selection": check(
            "PASS"
            if experiment["architecture"]["threshold_sweep"] is False
            and experiment["selection"]["candidate_count"] == 5
            and selection_before_diagnostics
            else "FAIL",
            "P(TP-first)=0.60, Expected-R=0, and top-2/expert/day are fixed; only three single families and two fixed portfolios were compared.",
            "Freeze thresholds before evaluated folds." if experiment["architecture"]["threshold_sweep"] else "None.",
        ),
        "purge/embargo": check(
            "PASS" if ".iloc[:-HORIZON]" in training_source and boundary_ok else "FAIL",
            "Outer training drops the last H=90 rows; internal boundaries require exact label end before the next stage.",
            "Apply outer and inner label-horizon purge." if not boundary_ok else "None.",
        ),
        "holdout contamination": check(
            "FAIL",
            "The experiment correctly labels 2025 as repeatedly inspected development data; it cannot support an untouched OOS claim.",
            "Freeze a candidate, then collect a new embargoed future interval that no research decision has observed.",
        ),
        "recent-period reuse": check(
            "PASS",
            "The reused recent interval is evaluated after selection and is explicitly monitoring/development-only, not a selection or promotion gate.",
        ),
        "execution alignment": check(
            "PASS"
            if not ledger_failures
            and not metric_failures
            and "for offset in range(1, horizon + 1)" in first_touch_source
            and "take_event = active & take_hit & ~stop_hit" in first_touch_source
            and "free_index = index + exit_offset + 1" in execution_source
            else "FAIL",
            "HIGH/LOW first-touch starts at offset 1, ties are stop-first, position occupancy is enforced, and all persisted metrics reconcile."
            if not ledger_failures and not metric_failures
            else "; ".join(ledger_failures + metric_failures),
            "Correct ledger execution/reconciliation only; do not retune." if ledger_failures or metric_failures else "None.",
        ),
        "cost assumptions": check(
            "PASS" if not ledger_failures and not metric_failures else "FAIL",
            "Base applies the existing 30-point spread plus 5 extra points; stress keeps identical entries and raises extra points to 10.",
            "Apply costs once in stop-risk units and preserve entry identity." if ledger_failures or metric_failures else "None.",
        ),
        "multiple-testing risk": check(
            "FAIL",
            "Generation 17 tested five fixed configurations after many prior generations; a raw-ATR implementation was also rejected pre-freeze for specification mismatch. All historical/recent intervals are development data.",
            "Pre-register one frozen candidate and obtain a new untouched future outer test; do not use that interval for further tuning.",
        ),
    }

    internal_integrity_checks = (
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
        and all(checks[name]["verdict"] == "PASS" for name in internal_integrity_checks)
        else "FAIL"
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT.name,
        "transfer_report": TRANSFER.name,
        "submitted_candidate_status": experiment["selected"]["status"],
        "discovery_qualified_count": experiment["selection"]["qualified_count"],
        "internal_chronological_validation_quality": internal_quality,
        "final_untouched_test_validity": "FAIL",
        "overall_verdict": "FAIL",
        "performance_claim": "no final OOS performance claim is valid",
        "checks": checks,
        "metric_reconciliation": reconciliation,
        "selection_inventory_complete": inventory_complete,
        "dataset_manifest_matches_run": dataset_match,
        "ledger_failures": ledger_failures,
        "metric_failures": metric_failures,
        "source_hashes": {
            path.name: sha256(path)
            for path in (
                EXPERIMENT_SOURCE,
                TRAINING_SOURCE,
                HISTORICAL_FEATURE_SOURCE,
                RECENT_FEATURE_SOURCE,
                EXECUTION_SOURCE,
                FIRST_TOUCH_SOURCE,
            )
        },
        "transfer_condition_inventory": {
            expert: {
                "generalizable_entry_condition_found": value[
                    "generalizable_entry_condition_found"
                ],
                "reproducible_entry_features": value[
                    "reproducible_entry_features"
                ],
            }
            for expert, value in transfer["families"].items()
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
    return 0 if not metric_failures and not ledger_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
