from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import MetaTrader5 as mt5

from barrier_classifier_strategy import HORIZON
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
from gold_short_rule_research import (
    DEDICATED_THRESHOLDS,
    EDGE_THRESHOLDS,
    MAX_HOLD_VALUES,
    SESSION_PROFILES,
    SL_ATR_VALUES,
    TP_ATR_VALUES,
    candidate_score,
    compact_stats,
    evaluate_frame,
    make_params,
    passes_promotion_fold,
    public_params,
)


PROJECT_ROOT = Path(__file__).resolve().parent
CANDIDATE_MODEL = PROJECT_ROOT / "gold_short_recent_candidate_xgb.json"
REPORT_JSON = PROJECT_ROOT / "gold_short_recent_walk_forward.json"
REPORT_MD = PROJECT_ROOT / "gold_short_recent_walk_forward.md"
N_ESTIMATORS = 220


def is_qualified(stats: dict) -> bool:
    return bool(
        not stats["stopped_out"]
        and stats["trades"] >= 10
        and stats["pnl"] > 0
        and stats["win_rate"] >= 0.58
        and stats["profit_factor"] >= 1.08
    )


def markdown_report(report: dict) -> str:
    lines = [
        "# GOLD recent short walk-forward",
        "",
        "Research-only: the live runner and production models are unchanged.",
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
            "Promotion requires validation and untouched recent test to have at "
            "least 10 trades, positive PnL, win rate >= 60%, PF >= 1.15, "
            "DD <= 15%, and a profitable 10-point cost stress test with PF >= 1.05.",
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
        train_validation, features, 2, N_ESTIMATORS
    )
    validation_probs = make_direction_probs(
        predict_positive(validation_model, validation, features), "short"
    )

    candidates = []
    combinations = product(
        DEDICATED_THRESHOLDS,
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
        stats = evaluate_frame(params, validation, validation_probs)
        candidates.append(
            {
                "score": candidate_score(stats),
                "params": params,
                "session_profile": session_profile,
                "validation": compact_stats(stats),
                "qualified": is_qualified(stats),
            }
        )
        if index % 120 == 0:
            print(f"Evaluated {index} validation candidates", flush=True)

    qualified = sorted(
        (item for item in candidates if item["qualified"]),
        key=lambda item: item["score"],
        reverse=True,
    )
    selected = qualified[0] if qualified else max(
        candidates, key=lambda item: item["score"]
    )

    test_model = train_binary_model(train_test, features, 2, N_ESTIMATORS)
    test_model.save_model(CANDIDATE_MODEL)
    test_probs = make_direction_probs(
        predict_positive(test_model, test, features), "short"
    )
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
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_only",
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
        "top_validation_candidates": [
            {
                "params": public_params(item["params"], item["session_profile"]),
                "validation": item["validation"],
            }
            for item in qualified[:10]
        ],
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
