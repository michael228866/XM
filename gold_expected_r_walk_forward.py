from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import xgboost as xgb

from barrier_classifier_strategy import (
    build_long_reward_target,
    build_short_reward_target,
)
from barrier_final_train import prepare_barrier_data
from gold_expected_r_champion import (
    EXPERT_DIRECTIONS,
    rolling_top_k_champion_signals,
)
from gold_recent_walk_forward import DEFAULT_TERMINAL, build_feature_frame
from gold_regime_experts_walk_forward import (
    MAY_HOURS,
    MAY_WEEKDAYS,
    RECENT_START,
    SELECTION_FOLDS,
    benchmark_current,
    benchmark_may,
    evaluate_frame,
    route_arrays,
)
from gold_short_rule_research import compact_stats


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_JSON = PROJECT_ROOT / "gold_expected_r_walk_forward.json"
REPORT_MD = PROJECT_ROOT / "gold_expected_r_walk_forward.md"
CONFIG_FILE = PROJECT_ROOT / "gold_expected_r_candidate.json"
MODEL_FILES = {
    name: PROJECT_ROOT / f"gold_expected_r_{name}_xgb.json"
    for name in EXPERT_DIRECTIONS
}

HORIZON = 90
TP_ATR = 1.3
SL_ATR = 1.6
MIN_TP_PRICE = 1.5
MIN_SL_PRICE = 0.6
RISK_PER_TRADE = 0.014
HISTORICAL_HOLDOUT_START = datetime(2025, 1, 1)
EXTRA_COST_POINTS = 5.0

SESSION_PROFILES = {
    "may_baseline": (MAY_HOURS, MAY_WEEKDAYS),
    "controlled_expanded": (
        tuple(sorted(set(MAY_HOURS) | {2, 4, 18})),
        (0, 1, 2, 3, 4),
    ),
}
MODEL_PROFILE = {
    "objective": "reg:pseudohubererror",
    "n_estimators": 320,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 120,
    "recency_half_life_days": 730.0,
    "target_clip": 2.0,
}
CHAMPION_CONFIG = {
    "maturity_rows": HORIZON,
    "window_rows": 60_000,
    "min_rows": 20_000,
    "block_rows": 10_080,
    "champion_min_trades": 10,
    "switch_margin": 0.05,
    "confirm_blocks": 2,
}


def training_frame(frame, cutoff):
    train = frame[frame["TIME_DT"] < cutoff]
    if len(train) <= HORIZON:
        raise RuntimeError(f"No purged training rows before {cutoff}")
    train = train.iloc[:-HORIZON].copy()
    if len(train) < 100_000:
        raise RuntimeError(
            f"Training fold too small: cutoff={cutoff.date()} rows={len(train):,}"
        )
    return train


def expected_r_sample_weight(frame, target):
    latest = frame["TIME_DT"].iloc[-1]
    age_days = (
        (latest - frame["TIME_DT"]).dt.total_seconds().to_numpy(dtype=np.float64)
        / 86_400.0
    )
    half_life = MODEL_PROFILE["recency_half_life_days"]
    recency = 0.15 + 0.85 * np.exp(-math.log(2.0) * age_days / half_life)
    magnitude = np.clip(np.abs(target), 0.25, 2.0)
    weights = recency * magnitude
    invalid = ~np.isfinite(weights) | (weights <= 0.0)
    if invalid.any():
        weights[invalid] = 1.0
    return weights


def train_expected_r_experts(frame, features):
    _, masks = route_arrays(frame, features)
    models = {}
    for name, direction in EXPERT_DIRECTIONS.items():
        target_column = "LONG_REWARD" if direction == 1 else "SHORT_REWARD"
        valid = masks[name] & np.isfinite(frame[target_column].to_numpy())
        subset = frame.loc[valid].copy()
        target = subset[target_column].to_numpy(dtype=np.float64)
        if len(subset) < 20_000:
            raise RuntimeError(f"Expected-R expert too small: {name}={len(subset):,}")
        clipped_target = np.clip(
            target,
            -MODEL_PROFILE["target_clip"],
            MODEL_PROFILE["target_clip"],
        )
        model = xgb.XGBRegressor(
            objective=MODEL_PROFILE["objective"],
            tree_method="hist",
            device="cpu",
            n_estimators=MODEL_PROFILE["n_estimators"],
            learning_rate=MODEL_PROFILE["learning_rate"],
            max_depth=MODEL_PROFILE["max_depth"],
            min_child_weight=MODEL_PROFILE["min_child_weight"],
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=max(1, (os.cpu_count() or 2) - 1),
            verbosity=0,
        )
        model.fit(
            subset[features].astype(np.float32),
            clipped_target,
            sample_weight=expected_r_sample_weight(subset, clipped_target),
        )
        models[name] = model
        print(
            f"  {name}: rows={len(subset):,} mean_R={target.mean():.4f} "
            f"positive={np.mean(target > 0.0):.2%}",
            flush=True,
        )
    return models


def predict_expected_r(models, frame, features):
    _, masks = route_arrays(frame, features)
    predictions = {}
    for name in EXPERT_DIRECTIONS:
        values = np.full(len(frame), np.nan, dtype=np.float32)
        mask = masks[name]
        if mask.any():
            values[mask] = models[name].predict(
                frame.loc[mask, features].astype(np.float32)
            ).astype(np.float32)
        predictions[name] = values
    return predictions


def session_mask(frame, profile):
    hours, weekdays = SESSION_PROFILES[profile]
    time = frame["TIME_DT"]
    allowed = time.dt.hour.isin(hours) & time.dt.dayofweek.isin(weekdays)
    rsi = frame["M1_RSI"].to_numpy(dtype=np.float64)
    return allowed.to_numpy() & ~((rsi >= 35.0) & (rsi <= 45.0))


def make_params(candidate, extra_cost_points=EXTRA_COST_POINTS):
    hours, weekdays = SESSION_PROFILES[candidate["session_profile"]]
    return {
        "threshold": 0.5,
        "edge_threshold": 0.0,
        "tp_atr": TP_ATR,
        "sl_atr": SL_ATR,
        "min_tp_price": MIN_TP_PRICE,
        "min_sl_price": MIN_SL_PRICE,
        "max_hold": HORIZON,
        "cooldown_ticks": 0,
        "close_on_opposite": False,
        "direction_mode": "both",
        "initial_balance": 1000,
        "stop_out_balance": 0,
        "risk_per_trade": RISK_PER_TRADE,
        "allowed_entry_hours": list(hours),
        "allowed_entry_weekdays": list(weekdays),
        "excluded_rsi_ranges": [(35.0, 45.0)],
        "max_daily_loss_pct": 0.05,
        "max_daily_trades": None,
        "extra_cost_points": extra_cost_points,
        "drawdown_guard_start_pct": 0.08,
        "drawdown_guard_full_pct": 0.35,
        "drawdown_guard_min_risk_mult": 0.50,
        "loss_streak_threshold": 3,
        "loss_streak_risk_mult": 0.55,
        "loss_streak_pause_threshold": 3,
        "loss_streak_pause_ticks": 120,
        "rolling_guard_window": 30,
        "rolling_guard_min_trades": 18,
        "rolling_guard_min_profit_factor": 1.15,
        "rolling_guard_min_win_rate": None,
        "rolling_guard_risk_mult": 0.50,
        "rolling_guard_pause_ticks": 0,
    }


def candidate_signals(frame, predictions, candidate):
    return rolling_top_k_champion_signals(
        predictions,
        frame["LONG_REWARD"].to_numpy(dtype=np.float32),
        frame["SHORT_REWARD"].to_numpy(dtype=np.float32),
        frame["TIME_DT"].dt.date.to_numpy(),
        session_mask(frame, candidate["session_profile"]),
        top_k_per_day=candidate["top_k_per_day"],
        minimum_expected_r=candidate["minimum_expected_r"],
        **CHAMPION_CONFIG,
    )


def candidate_stats(frame, predictions, candidate, cost=EXTRA_COST_POINTS):
    probs, trace = candidate_signals(frame, predictions, candidate)
    stats = evaluate_frame(make_params(candidate, cost), frame, probs)
    return stats, trace


def fold_pass(stats, min_trades):
    profit_factor = stats["profit_factor"]
    return bool(
        not stats["stopped_out"]
        and stats["trades"] >= min_trades
        and stats["pnl"] > 0.0
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
        positive_folds = sum(float(stats["pnl"]) > 0.0 for stats in folds)
        return (
            -1e12
            + positive_folds * 1_000_000.0
            + min(profit_factor_value(stats) for stats in folds) * 100_000.0
            + min(float(stats["win_rate"]) for stats in folds) * 10_000.0
            + min(float(stats["max_drawdown_pct"]) for stats in folds) * 1_000.0
            + sum(float(stats["pnl"]) for stats in folds)
        )
    return sum(
        float(stats["pnl"])
        + float(stats["trades"]) * 2.0
        + float(stats["win_rate"]) * 500.0
        + profit_factor_value(stats) * 250.0
        + float(stats["max_drawdown_pct"]) * 300.0
        for stats in folds
    )


def public_candidate(candidate):
    return {
        "generation": candidate["generation"],
        "top_k_per_day": candidate["top_k_per_day"],
        "minimum_expected_r": candidate["minimum_expected_r"],
        "session_profile": candidate["session_profile"],
        "tp_atr": TP_ATR,
        "sl_atr": SL_ATR,
        "max_hold": HORIZON,
        "direction_mode": "both",
        "risk_per_trade": RISK_PER_TRADE,
        "model_files": {name: path.name for name, path in MODEL_FILES.items()},
        "model_profile": MODEL_PROFILE,
        "champion_config": CHAMPION_CONFIG,
    }


def markdown_report(report):
    lines = [
        "# GOLD generation 6 Expected-R walk-forward",
        "",
        "Four independent Expected-R experts with no-lookahead rolling top-k and realized-R champions.",
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
            f"Promotion gate: `{'PASS' if report['promotion_pass'] else 'FAIL'}`",
            "",
            f"Selected: `{json.dumps(report['selected']['params'])}`",
        ]
    )
    return "\n".join(lines) + "\n"


def add_reward_targets(frame):
    frame = frame.copy()
    frame["LONG_REWARD"] = build_long_reward_target(
        frame,
        HORIZON,
        TP_ATR,
        SL_ATR,
        MIN_TP_PRICE,
        MIN_SL_PRICE,
        EXTRA_COST_POINTS,
    )
    frame["SHORT_REWARD"] = build_short_reward_target(
        frame,
        HORIZON,
        TP_ATR,
        SL_ATR,
        MIN_TP_PRICE,
        MIN_SL_PRICE,
        EXTRA_COST_POINTS,
    )
    return frame


def main():
    history, features = prepare_barrier_data()
    history = add_reward_targets(history)
    print(
        f"History rows={len(history):,} {history['TIME_DT'].iloc[0]} -> "
        f"{history['TIME_DT'].iloc[-1]} long_R={np.nanmean(history['LONG_REWARD']):.4f} "
        f"short_R={np.nanmean(history['SHORT_REWARD']):.4f}",
        flush=True,
    )

    candidates = [
        {
            "generation": "6_expected_r",
            "top_k_per_day": top_k,
            "minimum_expected_r": minimum_r,
            "session_profile": session,
        }
        for top_k, minimum_r, session in product(
            (1, 2, 3, 4),
            (-0.30, -0.20, -0.10, 0.0),
            SESSION_PROFILES,
        )
    ]
    fold_results = {index: {} for index in range(len(candidates))}
    fold_records = []
    for fold_name, fold_start, fold_end in SELECTION_FOLDS:
        train = training_frame(history, fold_start)
        validation = history[
            (history["TIME_DT"] >= fold_start) & (history["TIME_DT"] < fold_end)
        ].copy().reset_index(drop=True)
        models = train_expected_r_experts(train, features)
        predictions = predict_expected_r(models, validation, features)
        for index, candidate in enumerate(candidates):
            stats, _ = candidate_stats(validation, predictions, candidate)
            fold_results[index][fold_name] = stats
        fold_records.append((fold_name, validation, predictions))
        print(
            f"Fold {fold_name}: train={len(train):,} validation={len(validation):,}",
            flush=True,
        )

    ranked = []
    for index, candidate in enumerate(candidates):
        folds = fold_results[index]
        fold_values = list(folds.values())
        ranked.append(
            {
                **candidate,
                "qualified": all(fold_pass(stats, 100) for stats in fold_values),
                "score": aggregate_score(fold_values),
                "folds": {
                    name: compact_stats(stats) for name, stats in folds.items()
                },
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
    selection_traces = {}
    for fold_name, validation, predictions in fold_records:
        _, trace = candidate_signals(validation, predictions, selected)
        selection_traces[fold_name] = trace

    holdout_train = training_frame(history, HISTORICAL_HOLDOUT_START)
    holdout = history[history["TIME_DT"] >= HISTORICAL_HOLDOUT_START].copy()
    holdout_models = train_expected_r_experts(holdout_train, features)
    holdout_predictions = predict_expected_r(holdout_models, holdout, features)
    holdout_stats, holdout_trace = candidate_stats(
        holdout, holdout_predictions, selected
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
    recent = add_reward_targets(recent)

    final_cutoff = history["TIME_DT"].iloc[-1] + timedelta(minutes=1)
    final_train = training_frame(history, final_cutoff)
    final_models = train_expected_r_experts(final_train, features)
    for name, model in final_models.items():
        model.save_model(MODEL_FILES[name])
    recent_predictions = predict_expected_r(final_models, recent, features)
    recent_stats, recent_trace = candidate_stats(recent, recent_predictions, selected)
    recent_cost, _ = candidate_stats(
        recent, recent_predictions, selected, cost=10.0
    )
    current_stats = benchmark_current(recent, features)
    may_stats = benchmark_may(recent, features)

    compact_holdout = compact_stats(holdout_stats)
    compact_recent = compact_stats(recent_stats)
    compact_cost = compact_stats(recent_cost)
    compact_current = compact_stats(current_stats)
    compact_may = compact_stats(may_stats)
    recent_pf = compact_recent["profit_factor"]
    current_pf = compact_current["profit_factor"]
    promotion_pass = bool(
        qualified
        and fold_pass(holdout_stats, 100)
        and fold_pass(recent_stats, 40)
        and compact_recent["trades"] > compact_current["trades"]
        and compact_recent["win_rate"] > compact_current["win_rate"]
        and compact_recent["pnl"] > compact_current["pnl"]
        and recent_pf is not None
        and recent_pf >= 1.15
        and (current_pf is None or recent_pf > current_pf)
        and compact_cost["pnl"] > 0.0
        and compact_cost["profit_factor"] is not None
        and compact_cost["profit_factor"] >= 1.05
    )

    selected_params = public_candidate(selected)
    config = {
        **selected_params,
        "features": features,
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
        "champion_trace": {
            "selection": selection_traces,
            "holdout": holdout_trace,
            "recent": recent_trace,
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
