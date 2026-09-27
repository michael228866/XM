from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import xgboost as xgb

from barrier_classifier_strategy import build_long_first_touch_target
from barrier_research_suite import class_weight, make_direction_probs, predict_positive
from gold_long_recent_walk_forward import make_params
from gold_recent_walk_forward import (
    DEFAULT_START,
    DEFAULT_TERMINAL,
    DEFAULT_TEST_START,
    DEFAULT_VALIDATION_START,
    build_feature_frame,
)
from gold_short_rule_research import compact_stats, evaluate_frame


PROJECT_ROOT = Path(__file__).resolve().parent
CANDIDATE_MODEL = PROJECT_ROOT / "gold_long_aligned_candidate_xgb.json"
REPORT_JSON = PROJECT_ROOT / "gold_long_aligned_model_optimization.json"
REPORT_MD = PROJECT_ROOT / "gold_long_aligned_model_optimization.md"

ALIGNED_HORIZON = 90
ALIGNED_TP_ATR = 1.3
ALIGNED_SL_ATR = 1.6
ALIGNED_MIN_TP = 1.5
ALIGNED_MIN_SL = 0.6

TRAIN_STARTS = (
    datetime(2025, 1, 1),
    datetime(2025, 7, 1),
    datetime(2026, 1, 1),
)
MODEL_PROFILES = {
    "baseline": (220, 0.05, 4, 80),
    "shallow": (300, 0.04, 3, 50),
    "regularized": (320, 0.03, 4, 120),
    "flexible": (260, 0.04, 5, 40),
}
POSITIVE_MULTIPLIERS = (0.85, 1.0, 1.15)
THRESHOLDS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80)


def build_aligned_target(frame):
    return build_long_first_touch_target(
        frame,
        horizon=ALIGNED_HORIZON,
        tp_atr=ALIGNED_TP_ATR,
        sl_atr=ALIGNED_SL_ATR,
        min_tp_price=ALIGNED_MIN_TP,
        min_sl_price=ALIGNED_MIN_SL,
    )


def train_model(frame, features, profile_name: str, positive_multiplier: float):
    estimators, learning_rate, max_depth, min_child_weight = MODEL_PROFILES[
        profile_name
    ]
    target = frame["BARRIER_TARGET"].to_numpy(dtype=np.int8)
    binary = (target == 1).astype(np.int8)
    weights = class_weight(target, 1).astype(np.float64)
    weights[binary == 1] *= positive_multiplier
    invalid = ~np.isfinite(weights) | (weights <= 0)
    if invalid.any():
        print(
            f"Replacing {int(invalid.sum()):,} invalid weights with 1.0",
            flush=True,
        )
        weights[invalid] = 1.0
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        device="cpu",
        n_estimators=estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=max(1, (os.cpu_count() or 2) - 1),
        verbosity=0,
    )
    model.fit(frame[features].astype(np.float32), binary, sample_weight=weights)
    return model


def validation_pass(stats: dict) -> bool:
    profit_factor = stats["profit_factor"]
    return bool(
        not stats["stopped_out"]
        and stats["trades"] >= 20
        and stats["pnl"] > 0
        and stats["win_rate"] >= 0.62
        and (profit_factor is None or profit_factor >= 1.20)
        and stats["max_drawdown_pct"] >= -0.15
    )


def promotion_fold_pass(stats: dict, min_trades: int) -> bool:
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
    if not validation_pass(stats):
        return -1e12 + float(stats["pnl"])
    profit_factor = stats["profit_factor"]
    finite_pf = 3.0 if profit_factor is None else min(float(profit_factor), 3.0)
    return (
        float(stats["pnl"])
        + float(stats["trades"]) * 5.0
        + float(stats["win_rate"]) * 800.0
        + finite_pf * 120.0
        + float(stats["max_drawdown_pct"]) * 400.0
    )


def evaluate_probs(frame, probabilities, threshold: float, extra_cost_points=5.0):
    params = make_params(threshold, 1.3, 1.6, 90, "expanded")
    params["extra_cost_points"] = extra_cost_points
    probs = make_direction_probs(probabilities, "long")
    return evaluate_frame(params, frame, probs)


def markdown_report(report: dict) -> str:
    lines = [
        "# GOLD execution-aligned long model optimization",
        "",
        "Research-only unless the promotion gate passes.",
        "",
        "| Fold | Trades | Win | PF | PnL | DD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("validation", "test", "june_july", "august", "test_cost_10"):
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
            f"Selected model: `{json.dumps(report['selected']['model'])}`",
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

    frame["BARRIER_TARGET"] = build_aligned_target(frame)
    frame = frame.iloc[:-ALIGNED_HORIZON].copy().reset_index(drop=True)
    validation_start = DEFAULT_VALIDATION_START.replace(tzinfo=None)
    test_start = DEFAULT_TEST_START.replace(tzinfo=None)
    august_start = datetime(2026, 8, 1)
    validation = frame[
        (frame["TIME_DT"] >= validation_start) & (frame["TIME_DT"] < test_start)
    ].copy().reset_index(drop=True)
    test = frame[frame["TIME_DT"] >= test_start].copy().reset_index(drop=True)
    june_july = test[test["TIME_DT"] < august_start].copy().reset_index(drop=True)
    august = test[test["TIME_DT"] >= august_start].copy().reset_index(drop=True)
    if min(map(len, (validation, test, june_july, august))) == 0:
        raise RuntimeError("One or more model-optimization folds are empty")

    candidates = []
    combinations = list(product(TRAIN_STARTS, MODEL_PROFILES, POSITIVE_MULTIPLIERS))
    for index, (train_start, profile_name, positive_multiplier) in enumerate(
        combinations, start=1
    ):
        train = frame[
            (frame["TIME_DT"] >= train_start)
            & (frame["TIME_DT"] < validation_start)
        ].iloc[:-ALIGNED_HORIZON]
        if len(train) < 20_000:
            continue
        model = train_model(train, features, profile_name, positive_multiplier)
        validation_probability = predict_positive(model, validation, features)
        for threshold in THRESHOLDS:
            stats = evaluate_probs(validation, validation_probability, threshold)
            candidates.append(
                {
                    "score": score(stats),
                    "qualified": validation_pass(stats),
                    "train_start": train_start,
                    "profile": profile_name,
                    "positive_multiplier": positive_multiplier,
                    "threshold": threshold,
                    "validation": compact_stats(stats),
                }
            )
        print(f"Trained validation model {index}/{len(combinations)}", flush=True)

    qualified = sorted(
        (candidate for candidate in candidates if candidate["qualified"]),
        key=lambda candidate: candidate["score"],
        reverse=True,
    )
    selected = qualified[0] if qualified else max(
        candidates, key=lambda candidate: candidate["score"]
    )
    final_train = frame[
        (frame["TIME_DT"] >= selected["train_start"])
        & (frame["TIME_DT"] < test_start)
    ].iloc[:-ALIGNED_HORIZON]
    final_model = train_model(
        final_train,
        features,
        selected["profile"],
        selected["positive_multiplier"],
    )
    final_model.save_model(CANDIDATE_MODEL)
    test_probability = predict_positive(final_model, test, features)
    test_stats = evaluate_probs(test, test_probability, selected["threshold"])
    split = len(june_july)
    june_july_stats = evaluate_probs(
        june_july, test_probability[:split], selected["threshold"]
    )
    august_stats = evaluate_probs(
        august, test_probability[split:], selected["threshold"]
    )
    cost_stats = evaluate_probs(
        test, test_probability, selected["threshold"], extra_cost_points=10.0
    )
    compact_test = compact_stats(test_stats)
    compact_june_july = compact_stats(june_july_stats)
    compact_august = compact_stats(august_stats)
    compact_cost = compact_stats(cost_stats)
    promotion_pass = bool(
        qualified
        and promotion_fold_pass(compact_test, 20)
        and compact_cost["pnl"] > 0
        and (
            compact_cost["profit_factor"] is None
            or compact_cost["profit_factor"] >= 1.05
        )
    )

    selected_model = {
        "train_start": selected["train_start"].isoformat(),
        "profile": selected["profile"],
        "positive_multiplier": selected["positive_multiplier"],
        "threshold": selected["threshold"],
        "tp_atr": 1.3,
        "sl_atr": 1.6,
        "max_hold": 90,
        "session_profile": "expanded",
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "promotion_pass" if promotion_pass else "research_only",
        "model": CANDIDATE_MODEL.name,
        "searched_models": len(combinations),
        "qualified_count": len(qualified),
        "data": {
            "validation_start": DEFAULT_VALIDATION_START.isoformat(),
            "test_start": DEFAULT_TEST_START.isoformat(),
            "august_start": august_start.isoformat(),
            "end": frame["TIME_DT"].iloc[-1].isoformat(),
            "label": {
                "horizon": ALIGNED_HORIZON,
                "tp_atr": ALIGNED_TP_ATR,
                "sl_atr": ALIGNED_SL_ATR,
                "min_tp": ALIGNED_MIN_TP,
                "min_sl": ALIGNED_MIN_SL,
            },
        },
        "selected": {
            "model": selected_model,
            "validation": selected["validation"],
            "test": compact_test,
            "june_july": compact_june_july,
            "august": compact_august,
            "test_cost_10": compact_cost,
        },
        "promotion_pass": promotion_pass,
        "top_validation_candidates": [
            {
                "model": {
                    "train_start": candidate["train_start"].isoformat(),
                    "profile": candidate["profile"],
                    "positive_multiplier": candidate["positive_multiplier"],
                    "threshold": candidate["threshold"],
                },
                "validation": candidate["validation"],
            }
            for candidate in qualified[:10]
        ],
    }
    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REPORT_MD.write_text(markdown_report(report), encoding="utf-8")
    print(markdown_report(report), flush=True)
    print(f"Saved {CANDIDATE_MODEL.name}, {REPORT_JSON.name}, {REPORT_MD.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
