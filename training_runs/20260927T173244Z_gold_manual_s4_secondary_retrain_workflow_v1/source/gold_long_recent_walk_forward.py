from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import MetaTrader5 as mt5

from barrier_classifier_strategy import HORIZON
from barrier_final_train import FINAL_PARAMS
from barrier_research_suite import (
    make_direction_probs,
    predict_positive,
    train_binary_model,
)
from gold_recent_walk_forward import (
    DEFAULT_START,
    DEFAULT_TERMINAL,
    DEFAULT_TEST_START,
    DEFAULT_VALIDATION_START,
    build_feature_frame,
)
from gold_short_rule_research import compact_stats, evaluate_frame


PROJECT_ROOT = Path(__file__).resolve().parent
CANDIDATE_MODEL = PROJECT_ROOT / "gold_long_recent_candidate_xgb.json"
REPORT_JSON = PROJECT_ROOT / "gold_long_recent_walk_forward.json"
REPORT_MD = PROJECT_ROOT / "gold_long_recent_walk_forward.md"
N_ESTIMATORS = 220

THRESHOLDS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
TP_ATR_VALUES = (0.9, 1.1, 1.3)
SL_ATR_VALUES = (1.6, 2.0)
MAX_HOLD_VALUES = (90, 180)
CURRENT_HOURS = tuple(FINAL_PARAMS["allowed_entry_hours"])
CURRENT_WEEKDAYS = tuple(FINAL_PARAMS["allowed_entry_weekdays"])
SESSION_PROFILES = {
    "current": (CURRENT_HOURS, CURRENT_WEEKDAYS),
    "current_all_weekdays": (CURRENT_HOURS, (0, 1, 2, 3, 4)),
    "expanded": (
        tuple(sorted(set(CURRENT_HOURS) | {2, 4, 18})),
        (0, 1, 2, 3, 4),
    ),
    "all_weekdays_all_hours": (tuple(range(24)), (0, 1, 2, 3, 4)),
}


def make_params(
    threshold: float,
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
            "edge_threshold": 0.0,
            "tp_atr": tp_atr,
            "sl_atr": sl_atr,
            "max_hold": max_hold,
            "direction_mode": "long",
            "risk_per_trade": 0.014,
            "allowed_entry_hours": list(hours),
            "allowed_entry_weekdays": list(weekdays),
            "excluded_rsi_ranges": [(0.0, 22.0), (35.0, 45.0)],
        }
    )
    return params


def public_params(params: dict, session_profile: str) -> dict:
    return {
        "threshold": params["threshold"],
        "tp_atr": params["tp_atr"],
        "sl_atr": params["sl_atr"],
        "max_hold": params["max_hold"],
        "risk_per_trade": params["risk_per_trade"],
        "allowed_entry_hours": params["allowed_entry_hours"],
        "allowed_entry_weekdays": params["allowed_entry_weekdays"],
        "session_profile": session_profile,
    }


def is_qualified(stats: dict, min_trades: int) -> bool:
    profit_factor = stats["profit_factor"]
    return bool(
        not stats["stopped_out"]
        and stats["trades"] >= min_trades
        and stats["pnl"] > 0
        and stats["win_rate"] >= 0.60
        and (profit_factor is None or profit_factor >= 1.15)
        and stats["max_drawdown_pct"] >= -0.20
    )


def score(stats: dict) -> float:
    if not is_qualified(stats, 15):
        return -1e12 + float(stats["pnl"])
    return (
        float(stats["pnl"])
        + float(stats["trades"]) * 4.0
        + float(stats["win_rate"]) * 600.0
        + min(float(stats["profit_factor"]), 3.0) * 100.0
        + float(stats["max_drawdown_pct"]) * 400.0
    )


def markdown_report(report: dict) -> str:
    lines = [
        "# GOLD recent dedicated-long walk-forward",
        "",
        "Research-only unless the promotion gate passes.",
        "",
        "| Fold | Trades | Win | PF | PnL | DD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("validation", "test", "test_cost_10"):
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
            "The gate requires validation and untouched test to be profitable with "
            "win rate >= 60%, PF >= 1.15, DD <= 20%, and the 10-point cost "
            "stress test to remain profitable with PF >= 1.05.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    if not DEFAULT_TERMINAL.exists():
        raise FileNotFoundError(f"MT5 terminal not found: {DEFAULT_TERMINAL}")
    if not mt5.initialize(path=str(DEFAULT_TERMINAL), timeout=10_000):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        tick = mt5.symbol_info_tick("GOLD#")
        if tick is None:
            raise RuntimeError(f"No GOLD# tick: {mt5.last_error()}")
        end = datetime.fromtimestamp(tick.time, tz=timezone.utc)
        frame, features = build_feature_frame(DEFAULT_START, end)
    finally:
        mt5.shutdown()

    validation_start = DEFAULT_VALIDATION_START.replace(tzinfo=None)
    test_start = DEFAULT_TEST_START.replace(tzinfo=None)
    train_validation = frame[frame["TIME_DT"] < validation_start].iloc[:-HORIZON]
    validation = frame[
        (frame["TIME_DT"] >= validation_start) & (frame["TIME_DT"] < test_start)
    ].copy().reset_index(drop=True)
    train_test = frame[frame["TIME_DT"] < test_start].iloc[:-HORIZON]
    test = frame[frame["TIME_DT"] >= test_start].copy().reset_index(drop=True)
    if min(map(len, (train_validation, validation, train_test, test))) == 0:
        raise RuntimeError("One or more walk-forward folds are empty")

    print(
        "Rows | "
        f"train_validation={len(train_validation):,} validation={len(validation):,} "
        f"train_test={len(train_test):,} test={len(test):,}",
        flush=True,
    )
    validation_model = train_binary_model(
        train_validation, features, 1, N_ESTIMATORS
    )
    validation_probs = make_direction_probs(
        predict_positive(validation_model, validation, features), "long"
    )

    candidates = []
    combinations = product(
        THRESHOLDS,
        TP_ATR_VALUES,
        SL_ATR_VALUES,
        MAX_HOLD_VALUES,
        SESSION_PROFILES,
    )
    for threshold, tp_atr, sl_atr, max_hold, session_profile in combinations:
        params = make_params(
            threshold, tp_atr, sl_atr, max_hold, session_profile
        )
        stats = evaluate_frame(params, validation, validation_probs)
        candidates.append(
            {
                "score": score(stats),
                "params": params,
                "session_profile": session_profile,
                "validation": compact_stats(stats),
                "qualified": is_qualified(stats, 15),
            }
        )
    qualified = sorted(
        (item for item in candidates if item["qualified"]),
        key=lambda item: item["score"],
        reverse=True,
    )
    selected = qualified[0] if qualified else max(
        candidates, key=lambda item: item["score"]
    )

    test_model = train_binary_model(train_test, features, 1, N_ESTIMATORS)
    test_model.save_model(CANDIDATE_MODEL)
    test_probs = make_direction_probs(
        predict_positive(test_model, test, features), "long"
    )
    test_stats = evaluate_frame(selected["params"], test, test_probs)
    cost_params = dict(selected["params"])
    cost_params["extra_cost_points"] = 10.0
    cost_stats = evaluate_frame(cost_params, test, test_probs)
    compact_test = compact_stats(test_stats)
    compact_cost = compact_stats(cost_stats)
    promotion_pass = bool(
        qualified
        and is_qualified(selected["validation"], 15)
        and is_qualified(compact_test, 20)
        and compact_cost["pnl"] > 0
        and (
            compact_cost["profit_factor"] is None
            or compact_cost["profit_factor"] >= 1.05
        )
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "promotion_pass" if promotion_pass else "research_only",
        "model": CANDIDATE_MODEL.name,
        "data": {
            "start": DEFAULT_START.isoformat(),
            "validation_start": DEFAULT_VALIDATION_START.isoformat(),
            "test_start": DEFAULT_TEST_START.isoformat(),
            "end": frame["TIME_DT"].iloc[-1].isoformat(),
            "validation_rows": len(validation),
            "test_rows": len(test),
        },
        "qualified_count": len(qualified),
        "selected": {
            "params": public_params(selected["params"], selected["session_profile"]),
            "validation": selected["validation"],
            "test": compact_test,
            "test_cost_10": compact_cost,
        },
        "promotion_pass": promotion_pass,
    }
    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REPORT_MD.write_text(markdown_report(report), encoding="utf-8")
    print(markdown_report(report), flush=True)
    print(f"Selected params: {report['selected']['params']}", flush=True)
    print(f"Saved {CANDIDATE_MODEL.name}, {REPORT_JSON.name}, {REPORT_MD.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
