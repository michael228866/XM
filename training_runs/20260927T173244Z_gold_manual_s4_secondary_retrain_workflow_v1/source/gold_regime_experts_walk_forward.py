from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import xgboost as xgb

from barrier_classifier_strategy import (
    build_long_first_touch_target,
    build_short_first_touch_target,
    evaluate,
)
from barrier_final_train import FINAL_PARAMS, prepare_barrier_data
from barrier_research_suite import predict_positive
from gold_long_model_optimization import train_model
from gold_recent_walk_forward import DEFAULT_TERMINAL, build_feature_frame
from gold_short_rule_research import compact_stats


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_JSON = PROJECT_ROOT / "gold_regime_experts_walk_forward.json"
REPORT_MD = PROJECT_ROOT / "gold_regime_experts_walk_forward.md"
CONFIG_FILE = PROJECT_ROOT / "gold_regime_experts_candidate.json"
MODEL_FILES = {
    "long_trend": PROJECT_ROOT / "gold_regime_long_trend_xgb.json",
    "long_pullback": PROJECT_ROOT / "gold_regime_long_pullback_xgb.json",
    "short_trend": PROJECT_ROOT / "gold_regime_short_trend_xgb.json",
    "short_pullback": PROJECT_ROOT / "gold_regime_short_pullback_xgb.json",
}

CURRENT_MODEL_FILE = PROJECT_ROOT / "gold_long_recent_candidate_xgb.json"
MAY_MODEL_FILE = PROJECT_ROOT / "gold_barrier_final_xgb.json"
RECENT_START = datetime(2026, 5, 18, tzinfo=timezone.utc)
HISTORICAL_HOLDOUT_START = datetime(2025, 1, 1)
SELECTION_FOLDS = (
    ("2018_2020", datetime(2018, 1, 1), datetime(2021, 1, 1)),
    ("2021_2022", datetime(2021, 1, 1), datetime(2023, 1, 1)),
    ("2023_2024", datetime(2023, 1, 1), datetime(2025, 1, 1)),
)

HORIZON = 180
TP_ATR = 1.1
SL_ATR = 2.0
MIN_TP_PRICE = 1.5
MIN_SL_PRICE = 0.6
RISK_PER_TRADE = 0.014

MAY_HOURS = (0, 1, 3, 8, 9, 11, 12, 17, 19, 20, 22, 23)
MAY_WEEKDAYS = (0, 1, 2, 4)
SESSION_PROFILES = {
    "may_baseline": (MAY_HOURS, MAY_WEEKDAYS),
    "controlled_expanded": (
        tuple(sorted(set(MAY_HOURS) | {2, 4})),
        (0, 1, 2, 3, 4),
    ),
}
THRESHOLDS = (0.50, 0.525, 0.55, 0.575, 0.60, 0.65)
MIN_TREND_STRENGTHS = (0.0, 0.2, 0.4)
HIGHER_TF_PREFIXES = (
    "H1_",
    "H2_",
    "H3_",
    "H4_",
    "H6_",
    "H8_",
    "H12_",
    "Daily_",
    "Weekly_",
    "Monthly_",
)


def route_arrays(frame, features):
    trend_columns = [
        name
        for name in features
        if name.endswith("_TREND") and name.startswith(HIGHER_TF_PREFIXES)
    ]
    if not trend_columns:
        raise RuntimeError("No higher-timeframe trend features were found")
    trend_score = frame[trend_columns].to_numpy(dtype=np.float32).mean(axis=1)
    bias = frame["BIAS_20"].to_numpy(dtype=np.float32)
    long_regime = trend_score >= 0.0
    short_regime = ~long_regime
    masks = {
        "long_trend": long_regime & (bias >= 0.0),
        "long_pullback": long_regime & (bias < 0.0),
        "short_trend": short_regime & (bias <= 0.0),
        "short_pullback": short_regime & (bias > 0.0),
    }
    return trend_score, masks


def training_frame(frame, cutoff):
    train = frame[frame["TIME_DT"] < cutoff].iloc[:-HORIZON].copy()
    if len(train) < 100_000:
        raise RuntimeError(
            f"Training fold too small: cutoff={cutoff.date()} rows={len(train):,}"
        )
    return train


def train_experts(frame, features):
    _, masks = route_arrays(frame, features)
    models = {}
    for name, mask in masks.items():
        subset = frame.loc[mask].copy()
        target_column = "LONG_TARGET" if name.startswith("long_") else "SHORT_TARGET"
        subset["BARRIER_TARGET"] = subset[target_column].to_numpy(dtype=np.int8)
        counts = subset["BARRIER_TARGET"].value_counts().to_dict()
        if len(subset) < 20_000 or len(counts) < 2:
            raise RuntimeError(
                f"Expert training data is insufficient: {name} rows={len(subset):,} "
                f"classes={counts}"
            )
        models[name] = train_model(subset, features, "shallow", 1.0)
        print(f"  {name}: rows={len(subset):,} classes={counts}", flush=True)
    return models


def predict_experts(models, frame, features):
    trend_score, masks = route_arrays(frame, features)
    raw_probs = np.zeros((len(frame), 3), dtype=np.float32)
    for name, mask in masks.items():
        if not mask.any():
            continue
        probability = predict_positive(models[name], frame.loc[mask], features)
        direction = 1 if name.startswith("long_") else 2
        raw_probs[mask, direction] = probability
    raw_probs[:, 0] = 1.0 - np.maximum(raw_probs[:, 1], raw_probs[:, 2])
    return np.clip(raw_probs, 0.0, 1.0), np.abs(trend_score)


def filtered_probs(raw_probs, trend_strength, minimum):
    probs = raw_probs.copy()
    weak = trend_strength < minimum
    probs[weak, 1:] = 0.0
    probs[weak, 0] = 1.0
    return probs


def make_params(threshold, session_profile, extra_cost_points=5.0):
    hours, weekdays = SESSION_PROFILES[session_profile]
    params = dict(FINAL_PARAMS)
    params.update(
        {
            "threshold": threshold,
            "edge_threshold": 0.0,
            "tp_atr": TP_ATR,
            "sl_atr": SL_ATR,
            "min_tp_price": MIN_TP_PRICE,
            "min_sl_price": MIN_SL_PRICE,
            "max_hold": HORIZON,
            "direction_mode": "both",
            "risk_per_trade": RISK_PER_TRADE,
            "allowed_entry_hours": list(hours),
            "allowed_entry_weekdays": list(weekdays),
            "excluded_rsi_ranges": [(35.0, 45.0)],
            "extra_cost_points": extra_cost_points,
        }
    )
    return params


def evaluate_frame(params, frame, probs):
    return evaluate(
        params,
        frame["CLOSE"].to_numpy(dtype=np.float64),
        frame["ATR"].to_numpy(dtype=np.float64),
        probs,
        hours=frame["TIME_DT"].dt.hour.to_numpy(dtype=np.int16),
        weekdays=frame["TIME_DT"].dt.dayofweek.to_numpy(dtype=np.int8),
        dates=frame["TIME_DT"].dt.date.to_numpy(),
        rsi_values=frame["M1_RSI"].to_numpy(dtype=np.float64),
        highs=frame["HIGH"].to_numpy(dtype=np.float64),
        lows=frame["LOW"].to_numpy(dtype=np.float64),
    )


def candidate_stats(frame, raw_probs, trend_strength, candidate, cost=5.0):
    probs = filtered_probs(raw_probs, trend_strength, candidate["min_trend_strength"])
    params = make_params(candidate["threshold"], candidate["session_profile"], cost)
    return evaluate_frame(params, frame, probs)


def fold_pass(stats, min_trades):
    profit_factor = stats["profit_factor"]
    return bool(
        not stats["stopped_out"]
        and stats["trades"] >= min_trades
        and stats["pnl"] > 0
        and stats["win_rate"] >= 0.60
        and (profit_factor is None or profit_factor >= 1.15)
        and stats["max_drawdown_pct"] >= -0.20
    )


def profit_factor_value(stats):
    value = stats["profit_factor"]
    return 3.0 if value is None else min(float(value), 3.0)


def aggregate_score(folds):
    if not all(stats["trades"] >= 100 for stats in folds):
        return -2e12 + sum(float(stats["trades"]) for stats in folds)
    if not all(fold_pass(stats, 100) for stats in folds):
        return -1e12 + sum(
            float(stats["pnl"])
            + float(stats["trades"]) * 0.2
            + float(stats["win_rate"]) * 300.0
            + profit_factor_value(stats) * 150.0
            + float(stats["max_drawdown_pct"]) * 300.0
            for stats in folds
        )
    return sum(
        float(stats["pnl"])
        + float(stats["trades"]) * 2.0
        + float(stats["win_rate"]) * 500.0
        + profit_factor_value(stats) * 250.0
        + float(stats["max_drawdown_pct"]) * 300.0
        for stats in folds
    )


def load_model(path):
    if not path.exists():
        raise FileNotFoundError(path)
    model = xgb.XGBClassifier()
    model.load_model(path)
    model.set_params(device="cpu")
    return model


def benchmark_current(frame, features):
    model = load_model(CURRENT_MODEL_FILE)
    probability = predict_positive(model, frame, features)
    probs = np.zeros((len(frame), 3), dtype=np.float32)
    probs[:, 1] = probability
    probs[:, 0] = 1.0 - probability
    params = dict(FINAL_PARAMS)
    params.update(
        {
            "threshold": 0.75,
            "edge_threshold": 0.0,
            "tp_atr": 1.3,
            "sl_atr": 1.6,
            "max_hold": 90,
            "direction_mode": "long",
            "risk_per_trade": RISK_PER_TRADE,
            "allowed_entry_hours": [0, 1, 2, 3, 4, 8, 9, 11, 12, 17, 18, 19, 20, 22, 23],
            "allowed_entry_weekdays": [0, 1, 2, 3, 4],
            "excluded_rsi_ranges": [(0.0, 22.0), (35.0, 45.0)],
            "extra_cost_points": 5.0,
        }
    )
    return evaluate_frame(params, frame, probs)


def benchmark_may(frame, features):
    model = load_model(MAY_MODEL_FILE)
    probs = model.predict_proba(frame[features]).astype(np.float32)
    params = make_params(0.525, "may_baseline", 5.0)
    params["direction_mode"] = "long"
    return evaluate_frame(params, frame, probs)


def public_candidate(candidate):
    return {
        "threshold": candidate["threshold"],
        "min_trend_strength": candidate["min_trend_strength"],
        "session_profile": candidate["session_profile"],
        "tp_atr": TP_ATR,
        "sl_atr": SL_ATR,
        "max_hold": HORIZON,
        "risk_per_trade": RISK_PER_TRADE,
        "model_files": {name: path.name for name, path in MODEL_FILES.items()},
    }


def markdown_report(report):
    lines = [
        "# GOLD regime experts walk-forward",
        "",
        "Four separate long/short trend/pullback experts. Research-only unless promotion passes.",
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
            f"Current recent benchmark: `{json.dumps(report['benchmarks']['current'])}`",
            "",
            f"May-18 recent benchmark: `{json.dumps(report['benchmarks']['may_18'])}`",
            "",
            f"Promotion gate: `{'PASS' if report['promotion_pass'] else 'FAIL'}`",
            "",
            f"Selected: `{json.dumps(report['selected']['params'])}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    history, features = prepare_barrier_data()
    history["LONG_TARGET"] = build_long_first_touch_target(
        history, HORIZON, TP_ATR, SL_ATR, MIN_TP_PRICE, MIN_SL_PRICE
    )
    history["SHORT_TARGET"] = build_short_first_touch_target(
        history, HORIZON, TP_ATR, SL_ATR, MIN_TP_PRICE, MIN_SL_PRICE
    )
    print(
        f"History rows={len(history):,} {history['TIME_DT'].iloc[0]} -> "
        f"{history['TIME_DT'].iloc[-1]} long={int(history['LONG_TARGET'].sum()):,} "
        f"short={int(history['SHORT_TARGET'].sum()):,}",
        flush=True,
    )

    candidates = [
        {
            "threshold": threshold,
            "min_trend_strength": strength,
            "session_profile": session,
        }
        for threshold in THRESHOLDS
        for strength in MIN_TREND_STRENGTHS
        for session in SESSION_PROFILES
    ]
    fold_results = {index: {} for index in range(len(candidates))}
    for fold_name, fold_start, fold_end in SELECTION_FOLDS:
        train = training_frame(history, fold_start)
        validation = history[
            (history["TIME_DT"] >= fold_start) & (history["TIME_DT"] < fold_end)
        ].copy().reset_index(drop=True)
        models = train_experts(train, features)
        raw_probs, trend_strength = predict_experts(models, validation, features)
        for index, candidate in enumerate(candidates):
            fold_results[index][fold_name] = candidate_stats(
                validation, raw_probs, trend_strength, candidate
            )
        print(
            f"Fold {fold_name}: train={len(train):,} validation={len(validation):,} "
            f"long_p99={np.quantile(raw_probs[:, 1], 0.99):.3f} "
            f"short_p99={np.quantile(raw_probs[:, 2], 0.99):.3f}",
            flush=True,
        )

    ranked = []
    for index, candidate in enumerate(candidates):
        folds = fold_results[index]
        fold_list = list(folds.values())
        ranked.append(
            {
                **candidate,
                "qualified": all(fold_pass(stats, 100) for stats in fold_list),
                "score": aggregate_score(fold_list),
                "folds": {name: compact_stats(stats) for name, stats in folds.items()},
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    qualified = [item for item in ranked if item["qualified"]]
    covered = [
        item
        for item in ranked
        if all(stats["trades"] >= 100 for stats in item["folds"].values())
    ]
    selected = qualified[0] if qualified else covered[0] if covered else ranked[0]

    holdout_train = training_frame(history, HISTORICAL_HOLDOUT_START)
    holdout = history[
        history["TIME_DT"] >= HISTORICAL_HOLDOUT_START
    ].copy().reset_index(drop=True)
    holdout_models = train_experts(holdout_train, features)
    holdout_raw, holdout_strength = predict_experts(holdout_models, holdout, features)
    holdout_stats = candidate_stats(
        holdout, holdout_raw, holdout_strength, selected
    )

    if not DEFAULT_TERMINAL.exists():
        raise FileNotFoundError(DEFAULT_TERMINAL)
    if not mt5.initialize(path=str(DEFAULT_TERMINAL), timeout=10_000):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        tick = mt5.symbol_info_tick("GOLD#")
        if tick is None:
            raise RuntimeError(f"No GOLD# tick: {mt5.last_error()}")
        recent_end = datetime.fromtimestamp(tick.time, tz=timezone.utc)
        recent, recent_features = build_feature_frame(RECENT_START, recent_end)
    finally:
        mt5.shutdown()
    if set(features) != set(recent_features):
        raise RuntimeError("Historical and recent feature sets differ")

    final_cutoff = history["TIME_DT"].iloc[-1] + timedelta(minutes=1)
    final_train = training_frame(history, final_cutoff)
    final_models = train_experts(final_train, features)
    for name, model in final_models.items():
        model.save_model(MODEL_FILES[name])
    recent_raw, recent_strength = predict_experts(final_models, recent, features)
    recent_stats = candidate_stats(recent, recent_raw, recent_strength, selected)
    recent_cost = candidate_stats(
        recent, recent_raw, recent_strength, selected, cost=10.0
    )
    current_stats = benchmark_current(recent, features)
    may_stats = benchmark_may(recent, features)

    compact_holdout = compact_stats(holdout_stats)
    compact_recent = compact_stats(recent_stats)
    compact_cost = compact_stats(recent_cost)
    compact_current = compact_stats(current_stats)
    compact_may = compact_stats(may_stats)
    promotion_pass = bool(
        qualified
        and fold_pass(holdout_stats, 60)
        and fold_pass(recent_stats, 60)
        and compact_recent["trades"] > compact_current["trades"]
        and compact_recent["win_rate"] > compact_current["win_rate"]
        and compact_cost["pnl"] > 0
        and (
            compact_cost["profit_factor"] is None
            or compact_cost["profit_factor"] >= 1.05
        )
    )

    selected_params = public_candidate(selected)
    config = {
        **selected_params,
        "features": features,
        "higher_tf_prefixes": HIGHER_TF_PREFIXES,
        "promotion_pass": promotion_pass,
    }
    CONFIG_FILE.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "promotion_pass" if promotion_pass else "research_only",
        "data": {
            "history_rows": len(history),
            "history_start": history["TIME_DT"].iloc[0].isoformat(),
            "history_end": history["TIME_DT"].iloc[-1].isoformat(),
            "recent_rows": len(recent),
            "recent_start": recent["TIME_DT"].iloc[0].isoformat(),
            "recent_end": recent["TIME_DT"].iloc[-1].isoformat(),
        },
        "qualified_count": len(qualified),
        "selected": {
            "params": selected_params,
            "folds": {
                **selected["folds"],
                "2025_2026_05_holdout": compact_holdout,
                "2026_recent": compact_recent,
                "2026_recent_cost_10": compact_cost,
            },
        },
        "benchmarks": {"current": compact_current, "may_18": compact_may},
        "promotion_pass": promotion_pass,
        "ranked_selection": ranked,
    }
    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REPORT_MD.write_text(markdown_report(report), encoding="utf-8")
    print(markdown_report(report), flush=True)
    print(
        f"Saved {REPORT_JSON.name}, {REPORT_MD.name}, {CONFIG_FILE.name}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
