from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import xgboost as xgb

from barrier_classifier_strategy import HORIZON, evaluate
from barrier_final_train import (
    FINAL_PARAMS,
    MODEL_PATH,
    TRAIN_END_RATIO,
    prepare_barrier_data,
)
from barrier_research_suite import (
    make_direction_probs,
    predict_positive,
    train_binary_model,
)


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_JSON = PROJECT_ROOT / "gold_short_rule_research.json"
REPORT_MD = PROJECT_ROOT / "gold_short_rule_research.md"
CANDIDATE_MODEL = PROJECT_ROOT / "gold_short_candidate_xgb.json"

THRESHOLDS = (0.45, 0.50, 0.525, 0.55, 0.60)
DEDICATED_THRESHOLDS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
EDGE_THRESHOLDS = (0.0, 0.10, 0.20)
TP_ATR_VALUES = (0.9, 1.1, 1.3)
SL_ATR_VALUES = (1.6, 2.0)
MAX_HOLD_VALUES = (90, 180)
SHORT_RISK_PER_TRADE = 0.014

CURRENT_HOURS = tuple(FINAL_PARAMS["allowed_entry_hours"])
CURRENT_WEEKDAYS = tuple(FINAL_PARAMS["allowed_entry_weekdays"])
SESSION_PROFILES = {
    "current": (CURRENT_HOURS, CURRENT_WEEKDAYS),
    "current_all_weekdays": (CURRENT_HOURS, (0, 1, 2, 3, 4)),
    "expanded": (tuple(sorted(set(CURRENT_HOURS) | {2, 4, 18})), (0, 1, 2, 3, 4)),
    "all_weekdays_all_hours": (tuple(range(24)), (0, 1, 2, 3, 4)),
}


def evaluate_frame(params: dict, frame, probs: np.ndarray) -> dict:
    return evaluate(
        params,
        frame["CLOSE"].to_numpy(dtype=np.float64),
        frame["ATR"].to_numpy(dtype=np.float64),
        probs,
        hours=frame["TIME_DT"].dt.hour.to_numpy(dtype=np.int16),
        weekdays=frame["TIME_DT"].dt.dayofweek.to_numpy(dtype=np.int8),
        dates=frame["TIME_DT"].dt.date.to_numpy(),
        rsi_values=frame["M1_RSI"].to_numpy(dtype=np.float64),
    )


def compact_stats(stats: dict) -> dict:
    keys = (
        "pnl",
        "trades",
        "win_rate",
        "profit_factor",
        "max_drawdown_pct",
        "max_consecutive_losses",
        "take_profit_exits",
        "stop_loss_exits",
        "timeout_exits",
        "stopped_out",
    )
    result = {}
    for key in keys:
        value = stats[key]
        if isinstance(value, (bool, np.bool_)):
            result[key] = bool(value)
        elif isinstance(value, (int, np.integer)):
            result[key] = int(value)
        elif np.isfinite(value):
            result[key] = round(float(value), 6)
        else:
            result[key] = None
    return result


def passes_development(stats: dict) -> bool:
    return bool(
        not stats["stopped_out"]
        and stats["trades"] >= 15
        and stats["pnl"] > 0
        and stats["win_rate"] >= 0.58
        and stats["profit_factor"] >= 1.08
    )


def passes_promotion_fold(stats: dict) -> bool:
    profit_factor = stats["profit_factor"]
    return bool(
        not stats["stopped_out"]
        and stats["trades"] >= 10
        and stats["pnl"] > 0
        and stats["win_rate"] >= 0.60
        and (profit_factor is None or profit_factor >= 1.15)
        and stats["max_drawdown_pct"] >= -0.15
    )


def candidate_score(stats: dict) -> float:
    if not passes_development(stats):
        return -1e12 + float(stats["pnl"])
    return (
        float(stats["pnl"])
        + float(stats["trades"]) * 2.0
        + float(stats["win_rate"]) * 250.0
        + min(float(stats["profit_factor"]), 3.0) * 40.0
        + float(stats["max_drawdown_pct"]) * 300.0
    )


def make_params(
    threshold: float,
    edge_threshold: float,
    tp_atr: float,
    sl_atr: float,
    max_hold: int,
    session_profile: str,
) -> dict:
    hours, weekdays = SESSION_PROFILES[session_profile]
    params = dict(FINAL_PARAMS)
    params.update(
        {
            "threshold": threshold,
            "edge_threshold": edge_threshold,
            "tp_atr": tp_atr,
            "sl_atr": sl_atr,
            "max_hold": max_hold,
            "direction_mode": "short",
            "risk_per_trade": SHORT_RISK_PER_TRADE,
            "allowed_entry_hours": list(hours),
            "allowed_entry_weekdays": list(weekdays),
            "excluded_rsi_ranges": [],
        }
    )
    return params


def public_params(params: dict, session_profile: str) -> dict:
    keys = (
        "threshold",
        "edge_threshold",
        "tp_atr",
        "sl_atr",
        "max_hold",
        "direction_mode",
        "risk_per_trade",
        "allowed_entry_hours",
        "allowed_entry_weekdays",
    )
    result = {key: params[key] for key in keys}
    result["session_profile"] = session_profile
    return result


def markdown_report(report: dict) -> str:
    lines = [
        "# GOLD short-direction research",
        "",
        "Research-only: the live runner and production model are unchanged.",
        "",
        "| Fold | Trades | Win | PF | PnL | DD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("development", "validation", "test", "test_cost_10"):
        stats = report["selected"][name]
        pf = "inf" if stats["profit_factor"] is None else f"{stats['profit_factor']:.2f}"
        lines.append(
            f"| {name} | {stats['trades']} | {stats['win_rate']:.2%} | "
            f"{pf} | {stats['pnl']:.2f} | {stats['max_drawdown_pct']:.2%} |"
        )
    lines.extend(
        [
            "",
            f"Promotion gate: `{'PASS' if report['promotion_pass'] else 'FAIL'}`",
            "",
            "Promotion requires validation and untouched test folds to have at least "
            "10 trades, positive PnL, win rate >= 60%, PF >= 1.15, DD <= 15%, "
            "and the 10-point cost stress test to remain profitable with PF >= 1.05.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dedicated-model",
        action="store_true",
        help="Train and test a research-only binary short model.",
    )
    parser.add_argument("--estimators", type=int, default=180)
    args = parser.parse_args()
    if args.estimators < 50:
        parser.error("--estimators must be at least 50")
    return args


def main() -> int:
    args = parse_args()
    frame, features = prepare_barrier_data()
    test_start = int(len(frame) * TRAIN_END_RATIO)
    out_of_sample = frame.iloc[test_start:].copy().reset_index(drop=True)
    first_cut = int(len(out_of_sample) * 0.40)
    second_cut = int(len(out_of_sample) * 0.70)
    development = out_of_sample.iloc[:first_cut].copy().reset_index(drop=True)
    validation = out_of_sample.iloc[first_cut:second_cut].copy().reset_index(drop=True)
    test = out_of_sample.iloc[second_cut:].copy().reset_index(drop=True)

    if args.dedicated_model:
        train = frame.iloc[: max(0, test_start - HORIZON)].copy()
        print(
            f"Training dedicated short model: rows={len(train):,} "
            f"estimators={args.estimators}",
            flush=True,
        )
        model = train_binary_model(train, features, 2, args.estimators)
        model.save_model(CANDIDATE_MODEL)
        development_probs = make_direction_probs(
            predict_positive(model, development, features), "short"
        )
        validation_probs = make_direction_probs(
            predict_positive(model, validation, features), "short"
        )
        test_probs = make_direction_probs(
            predict_positive(model, test, features), "short"
        )
        model_name = CANDIDATE_MODEL.name
        thresholds = DEDICATED_THRESHOLDS
    else:
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        model.set_params(device="cpu")
        development_probs = model.predict_proba(development[features]).astype(np.float32)
        validation_probs = model.predict_proba(validation[features]).astype(np.float32)
        test_probs = model.predict_proba(test[features]).astype(np.float32)
        model_name = Path(MODEL_PATH).name
        thresholds = THRESHOLDS

    print(
        "Rows | "
        f"development={len(development):,} validation={len(validation):,} "
        f"test={len(test):,}",
        flush=True,
    )

    candidates = []
    combinations = product(
        thresholds,
        EDGE_THRESHOLDS,
        TP_ATR_VALUES,
        SL_ATR_VALUES,
        MAX_HOLD_VALUES,
        SESSION_PROFILES,
    )
    for index, values in enumerate(combinations, start=1):
        threshold, edge, tp_atr, sl_atr, max_hold, session_profile = values
        params = make_params(
            threshold,
            edge,
            tp_atr,
            sl_atr,
            max_hold,
            session_profile,
        )
        stats = evaluate_frame(params, development, development_probs)
        candidates.append(
            {
                "score": candidate_score(stats),
                "params": params,
                "session_profile": session_profile,
                "development": compact_stats(stats),
            }
        )
        if index % 120 == 0:
            print(f"Evaluated {index} development candidates", flush=True)

    candidates.sort(key=lambda item: item["score"], reverse=True)
    qualified = [
        item
        for item in candidates
        if item["development"]["trades"] >= 15
        and item["development"]["pnl"] > 0
        and item["development"]["win_rate"] >= 0.58
        and (
            item["development"]["profit_factor"] is None
            or item["development"]["profit_factor"] >= 1.08
        )
    ]
    finalists = []
    finalist_pool = qualified[:20] if qualified else candidates[:20]
    for candidate in finalist_pool:
        validation_stats = evaluate_frame(
            candidate["params"], validation, validation_probs
        )
        candidate["validation"] = compact_stats(validation_stats)
        candidate["validation_score"] = candidate_score(validation_stats)
        finalists.append(candidate)
    finalists.sort(key=lambda item: item["validation_score"], reverse=True)
    selected = finalists[0]

    test_stats = evaluate_frame(selected["params"], test, test_probs)
    cost_params = dict(selected["params"])
    cost_params["extra_cost_points"] = 10.0
    cost_stats = evaluate_frame(cost_params, test, test_probs)
    compact_test = compact_stats(test_stats)
    compact_cost = compact_stats(cost_stats)
    promotion_pass = bool(
        qualified
        and passes_promotion_fold(selected["validation"])
        and passes_promotion_fold(compact_test)
        and compact_cost["pnl"] > 0
        and (
            compact_cost["profit_factor"] is None
            or compact_cost["profit_factor"] >= 1.05
        )
    )

    report = {
        "status": "research_only",
        "model": model_name,
        "development_qualified_count": len(qualified),
        "data": {
            "development": [
                development["TIME_DT"].iloc[0].isoformat(),
                development["TIME_DT"].iloc[-1].isoformat(),
            ],
            "validation": [
                validation["TIME_DT"].iloc[0].isoformat(),
                validation["TIME_DT"].iloc[-1].isoformat(),
            ],
            "test": [
                test["TIME_DT"].iloc[0].isoformat(),
                test["TIME_DT"].iloc[-1].isoformat(),
            ],
        },
        "selected": {
            "params": public_params(selected["params"], selected["session_profile"]),
            "development": selected["development"],
            "validation": selected["validation"],
            "test": compact_test,
            "test_cost_10": compact_cost,
        },
        "promotion_pass": promotion_pass,
        "finalists": [
            {
                "params": public_params(item["params"], item["session_profile"]),
                "development": item["development"],
                "validation": item["validation"],
            }
            for item in finalists[:10]
        ],
    }
    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REPORT_MD.write_text(markdown_report(report), encoding="utf-8")
    print(markdown_report(report), flush=True)
    print(f"Selected params: {report['selected']['params']}", flush=True)
    print(f"Saved {REPORT_JSON.name} and {REPORT_MD.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
