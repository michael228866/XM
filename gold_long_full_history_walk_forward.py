from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import xgboost as xgb

from barrier_classifier_strategy import (
    build_long_first_touch_target,
    build_long_reward_target,
    evaluate,
)
from barrier_final_train import prepare_barrier_data
from barrier_research_suite import make_direction_probs, predict_positive
from gold_long_model_optimization import train_model
from gold_long_recent_walk_forward import make_params
from gold_recent_walk_forward import (
    DEFAULT_TERMINAL,
    DEFAULT_TEST_START,
    build_feature_frame,
)
from gold_short_rule_research import compact_stats


PROJECT_ROOT = Path(__file__).resolve().parent
CANDIDATE_MODEL = PROJECT_ROOT / "gold_long_full_history_first_touch_candidate_xgb.json"
REPORT_JSON = PROJECT_ROOT / "gold_long_full_history_first_touch_walk_forward.json"
REPORT_MD = PROJECT_ROOT / "gold_long_full_history_first_touch_walk_forward.md"

MODEL_HORIZON = 90
MODEL_TP_ATR = 1.3
MODEL_SL_ATR = 1.6
MODEL_MIN_TP_PRICE = 1.5
MODEL_MIN_SL_PRICE = 0.6

FAMILIES = {
    "full_history": {
        "years": None,
        "profile": "baseline",
        "label_mode": "first_touch",
    },
    "rolling_4y": {
        "years": 4,
        "profile": "shallow",
        "label_mode": "first_touch",
    },
    "rolling_2y": {
        "years": 2,
        "profile": "regularized",
        "label_mode": "first_touch",
    },
    "full_history_reward": {
        "years": None,
        "profile": "baseline",
        "label_mode": "reward_weighted",
    },
    "rolling_4y_reward": {
        "years": 4,
        "profile": "shallow",
        "label_mode": "reward_weighted",
    },
    "rolling_2y_reward": {
        "years": 2,
        "profile": "regularized",
        "label_mode": "reward_weighted",
    },
}
THRESHOLDS = (
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
)
SELECTION_FOLDS = (
    ("2018_2020", datetime(2018, 1, 1), datetime(2021, 1, 1)),
    ("2021_2022", datetime(2021, 1, 1), datetime(2023, 1, 1)),
    ("2023_2024", datetime(2023, 1, 1), datetime(2025, 1, 1)),
)
HISTORICAL_HOLDOUT_START = datetime(2025, 1, 1)


def training_frame(frame, cutoff: datetime, family: dict):
    mask = frame["TIME_DT"] < cutoff
    years = family["years"]
    if years is not None:
        mask &= frame["TIME_DT"] >= cutoff - timedelta(days=365 * years)
    train = frame[mask].iloc[:-MODEL_HORIZON]
    if len(train) < 100_000:
        raise RuntimeError(
            f"Training fold too small: cutoff={cutoff.date()} rows={len(train):,}"
        )
    return train


def strategy_stats(frame, probability, threshold: float, extra_cost_points=5.0):
    params = make_params(
        threshold,
        MODEL_TP_ATR,
        MODEL_SL_ATR,
        MODEL_HORIZON,
        "expanded",
    )
    params["extra_cost_points"] = extra_cost_points
    return evaluate(
        params,
        frame["CLOSE"].to_numpy(dtype=np.float64),
        frame["ATR"].to_numpy(dtype=np.float64),
        make_direction_probs(probability, "long"),
        hours=frame["TIME_DT"].dt.hour.to_numpy(dtype=np.int16),
        weekdays=frame["TIME_DT"].dt.dayofweek.to_numpy(dtype=np.int8),
        dates=frame["TIME_DT"].dt.date.to_numpy(),
        rsi_values=frame["M1_RSI"].to_numpy(dtype=np.float64),
        highs=frame["HIGH"].to_numpy(dtype=np.float64),
        lows=frame["LOW"].to_numpy(dtype=np.float64),
    )


def train_family_model(frame, features, family):
    if family["label_mode"] == "first_touch":
        return train_model(frame, features, family["profile"], 1.0)

    target = frame["LONG_REWARD_TARGET"].to_numpy(dtype=np.float64)
    if not np.isfinite(target).all():
        raise RuntimeError("Training reward target contains non-finite values")
    binary = (target > 0.0).astype(np.int8)
    weights = np.clip(np.abs(target), 0.05, 3.0)
    profile = {
        "baseline": (220, 0.05, 4, 80),
        "shallow": (300, 0.04, 3, 50),
        "regularized": (320, 0.03, 4, 120),
    }[family["profile"]]
    estimators, learning_rate, max_depth, min_child_weight = profile
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
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(frame[features].astype(np.float32), binary, sample_weight=weights)
    return model


def selection_fold_pass(stats: dict) -> bool:
    profit_factor = stats["profit_factor"]
    return bool(
        not stats["stopped_out"]
        and stats["trades"] >= 40
        and stats["pnl"] > 0
        and stats["win_rate"] >= 0.58
        and (profit_factor is None or profit_factor >= 1.10)
        and stats["max_drawdown_pct"] >= -0.25
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


def aggregate_score(folds: list[dict]) -> float:
    if not all(stats["trades"] >= 40 for stats in folds):
        return -2e12 + sum(float(stats["trades"]) for stats in folds)
    if not all(selection_fold_pass(stats) for stats in folds):
        return -1e12 + sum(
            float(stats["pnl"])
            + float(stats["win_rate"]) * 200.0
            + min(float(stats["profit_factor"]), 3.0) * 100.0
            + float(stats["max_drawdown_pct"]) * 200.0
            for stats in folds
        )
    return sum(
        float(stats["pnl"])
        + float(stats["trades"]) * 2.0
        + float(stats["win_rate"]) * 500.0
        + min(float(stats["profit_factor"]), 3.0) * 100.0
        + float(stats["max_drawdown_pct"]) * 300.0
        for stats in folds
    )


def markdown_report(report: dict) -> str:
    lines = [
        "# GOLD full-history long first-touch and reward-weighted walk-forward",
        "",
        "Local history: 2014-02 through 2026-05. Recent MT5 data is the final gate.",
        "First-touch and payoff-weighted labels are compared across all folds.",
        "",
        "| Fold | Trades | Win | PF | PnL | DD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, stats in report["selected"]["folds"].items():
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
            f"Selected: `{json.dumps(report['selected']['model'])}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    history, features = prepare_barrier_data()
    history["BARRIER_TARGET"] = build_long_first_touch_target(
        history,
        horizon=MODEL_HORIZON,
        tp_atr=MODEL_TP_ATR,
        sl_atr=MODEL_SL_ATR,
        min_tp_price=MODEL_MIN_TP_PRICE,
        min_sl_price=MODEL_MIN_SL_PRICE,
    )
    history["LONG_REWARD_TARGET"] = build_long_reward_target(
        history,
        horizon=MODEL_HORIZON,
        tp_atr=MODEL_TP_ATR,
        sl_atr=MODEL_SL_ATR,
        min_tp_price=MODEL_MIN_TP_PRICE,
        min_sl_price=MODEL_MIN_SL_PRICE,
        extra_cost_points=5.0,
    )
    print(
        f"Full history: rows={len(history):,} "
        f"{history['TIME_DT'].iloc[0]} -> {history['TIME_DT'].iloc[-1]} "
        f"long_targets={int(history['BARRIER_TARGET'].sum()):,}",
        flush=True,
    )

    fold_results = {
        (family_name, threshold): {}
        for family_name in FAMILIES
        for threshold in THRESHOLDS
    }
    for fold_name, fold_start, fold_end in SELECTION_FOLDS:
        validation = history[
            (history["TIME_DT"] >= fold_start) & (history["TIME_DT"] < fold_end)
        ].copy().reset_index(drop=True)
        if validation.empty:
            raise RuntimeError(f"Empty selection fold: {fold_name}")
        for family_name, family in FAMILIES.items():
            train = training_frame(history, fold_start, family)
            model = train_family_model(train, features, family)
            probability = predict_positive(model, validation, features)
            for threshold in THRESHOLDS:
                fold_results[(family_name, threshold)][fold_name] = strategy_stats(
                    validation, probability, threshold
                )
            print(
                f"Trained {family_name} for {fold_name}: "
                f"train={len(train):,} validation={len(validation):,} "
                f"p99={float(np.quantile(probability, 0.99)):.3f} "
                f"pmax={float(probability.max()):.3f}",
                flush=True,
            )

    ranked = []
    for (family_name, threshold), folds in fold_results.items():
        stats = list(folds.values())
        ranked.append(
            {
                "family": family_name,
                "threshold": threshold,
                "score": aggregate_score(stats),
                "qualified": all(selection_fold_pass(item) for item in stats),
                "folds": {name: compact_stats(item) for name, item in folds.items()},
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    qualified = [item for item in ranked if item["qualified"]]
    covered = [
        item
        for item in ranked
        if all(stats["trades"] >= 40 for stats in item["folds"].values())
    ]
    selected = qualified[0] if qualified else covered[0] if covered else ranked[0]
    family = FAMILIES[selected["family"]]

    holdout = history[
        history["TIME_DT"] >= HISTORICAL_HOLDOUT_START
    ].copy().reset_index(drop=True)
    holdout_train = training_frame(history, HISTORICAL_HOLDOUT_START, family)
    holdout_model = train_family_model(holdout_train, features, family)
    holdout_probability = predict_positive(holdout_model, holdout, features)
    holdout_stats = strategy_stats(holdout, holdout_probability, selected["threshold"])

    if not DEFAULT_TERMINAL.exists():
        raise FileNotFoundError(f"MT5 terminal not found: {DEFAULT_TERMINAL}")
    if not mt5.initialize(path=str(DEFAULT_TERMINAL), timeout=10_000):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        tick = mt5.symbol_info_tick("GOLD#")
        if tick is None:
            raise RuntimeError(f"No GOLD# tick: {mt5.last_error()}")
        recent_end = datetime.fromtimestamp(tick.time, tz=timezone.utc)
        recent, recent_features = build_feature_frame(DEFAULT_TEST_START, recent_end)
    finally:
        mt5.shutdown()
    missing = [feature for feature in features if feature not in recent.columns]
    if missing:
        raise RuntimeError(f"Recent frame missing features: {missing}")
    if set(features) != set(recent_features):
        raise RuntimeError("Historical and recent feature sets differ")

    final_cutoff = history["TIME_DT"].iloc[-1] + timedelta(minutes=1)
    final_train = training_frame(history, final_cutoff, family)
    final_model = train_family_model(final_train, features, family)
    final_model.save_model(CANDIDATE_MODEL)
    recent_probability = predict_positive(final_model, recent, features)
    recent_stats = strategy_stats(recent, recent_probability, selected["threshold"])
    recent_cost = strategy_stats(
        recent, recent_probability, selected["threshold"], extra_cost_points=10.0
    )

    compact_holdout = compact_stats(holdout_stats)
    compact_recent = compact_stats(recent_stats)
    compact_cost = compact_stats(recent_cost)
    promotion_pass = bool(
        qualified
        and promotion_fold_pass(compact_holdout, 30)
        and promotion_fold_pass(compact_recent, 20)
        and compact_cost["pnl"] > 0
        and (
            compact_cost["profit_factor"] is None
            or compact_cost["profit_factor"] >= 1.05
        )
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "promotion_pass" if promotion_pass else "research_only",
        "model_file": CANDIDATE_MODEL.name,
        "data": {
            "history_rows": len(history),
            "history_start": history["TIME_DT"].iloc[0].isoformat(),
            "history_end": history["TIME_DT"].iloc[-1].isoformat(),
            "recent_start": recent["TIME_DT"].iloc[0].isoformat(),
            "recent_end": recent["TIME_DT"].iloc[-1].isoformat(),
            "features": len(features),
        },
        "qualified_count": len(qualified),
        "selected": {
            "model": {
                "family": selected["family"],
                "profile": family["profile"],
                "training_years": family["years"],
                "label_mode": family["label_mode"],
                "threshold": selected["threshold"],
                "tp_atr": MODEL_TP_ATR,
                "sl_atr": MODEL_SL_ATR,
                "max_hold": MODEL_HORIZON,
                "session_profile": "expanded",
            },
            "folds": {
                **selected["folds"],
                "2025_2026_05_holdout": compact_holdout,
                "2026_06_recent": compact_recent,
                "2026_06_recent_cost_10": compact_cost,
            },
        },
        "promotion_pass": promotion_pass,
        "ranked_selection": ranked,
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
