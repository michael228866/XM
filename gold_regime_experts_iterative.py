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
    build_long_first_touch_target,
    build_short_first_touch_target,
)
from barrier_final_train import prepare_barrier_data
from barrier_research_suite import class_weight, predict_positive
from gold_recent_walk_forward import DEFAULT_TERMINAL, build_feature_frame
from gold_rolling_champion import rolling_champion_probabilities
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
REPORT_JSON = PROJECT_ROOT / "gold_regime_experts_iterative.json"
REPORT_MD = PROJECT_ROOT / "gold_regime_experts_iterative.md"
CONFIG_FILE = PROJECT_ROOT / "gold_regime_experts_iterative_candidate.json"

EXPERT_NAMES = (
    "long_trend",
    "long_pullback",
    "short_trend",
    "short_pullback",
)
MODEL_FILES = {
    generation: {
        name: PROJECT_ROOT / f"gold_iterative_{generation}_{name}_xgb.json"
        for name in EXPERT_NAMES
    }
    for generation in ("balanced", "time_decay")
}

# The previous four-expert run used 1.1 / 2.0 / 180 and produced a recent PF
# below 1.0. This iteration aligns both labels and exits with the more effective
# recent long model before testing model weighting and ensembling.
HORIZON = 90
TP_ATR = 1.3
SL_ATR = 1.6
MIN_TP_PRICE = 1.5
MIN_SL_PRICE = 0.6
RISK_PER_TRADE = 0.014
HISTORICAL_HOLDOUT_START = datetime(2025, 1, 1)

SESSION_PROFILES = {
    "may_baseline": (MAY_HOURS, MAY_WEEKDAYS),
    "controlled_expanded": (
        tuple(sorted(set(MAY_HOURS) | {2, 4, 18})),
        (0, 1, 2, 3, 4),
    ),
}

MODEL_PROFILES = {
    "balanced": {
        "n_estimators": 300,
        "learning_rate": 0.04,
        "max_depth": 3,
        "min_child_weight": 50,
        "recency_half_life_days": None,
    },
    "time_decay": {
        "n_estimators": 320,
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 120,
        "recency_half_life_days": 730.0,
    },
}
ROLLING_CHAMPION_CONFIG = {
    "maturity_rows": HORIZON,
    "window_rows": 60_000,
    "min_rows": 20_000,
    "block_rows": 10_080,
    "switch_margin": 0.002,
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


def build_sample_weight(frame, target, generation):
    weights = class_weight(target, 1).astype(np.float64)
    half_life = MODEL_PROFILES[generation]["recency_half_life_days"]
    if half_life is not None:
        latest = frame["TIME_DT"].iloc[-1]
        age_days = (
            (latest - frame["TIME_DT"]).dt.total_seconds().to_numpy(dtype=np.float64)
            / 86_400.0
        )
        # Older observations remain in training instead of being discarded.
        recency = 0.15 + 0.85 * np.exp(-math.log(2.0) * age_days / half_life)
        weights *= recency
    invalid = ~np.isfinite(weights) | (weights <= 0)
    if invalid.any():
        weights[invalid] = 1.0
    return weights


def train_binary_expert(frame, features, target_column, generation):
    target = frame[target_column].to_numpy(dtype=np.int8)
    counts = np.bincount(target, minlength=2)
    if counts.min() == 0:
        raise RuntimeError(
            f"Expert target has one class: {target_column} counts={counts.tolist()}"
        )
    profile = MODEL_PROFILES[generation]
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        device="cpu",
        n_estimators=profile["n_estimators"],
        learning_rate=profile["learning_rate"],
        max_depth=profile["max_depth"],
        min_child_weight=profile["min_child_weight"],
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=max(1, (os.cpu_count() or 2) - 1),
        verbosity=0,
    )
    weights = build_sample_weight(frame, target, generation)
    model.fit(
        frame[features].astype(np.float32),
        target,
        sample_weight=weights,
    )
    return model


def train_experts(frame, features, generation):
    _, masks = route_arrays(frame, features)
    models = {}
    for name in EXPERT_NAMES:
        subset = frame.loc[masks[name]].copy()
        target_column = "LONG_TARGET" if name.startswith("long_") else "SHORT_TARGET"
        counts = subset[target_column].value_counts().sort_index().to_dict()
        if len(subset) < 20_000 or len(counts) < 2:
            raise RuntimeError(
                f"Insufficient expert data: {name} rows={len(subset):,} classes={counts}"
            )
        models[name] = train_binary_expert(
            subset,
            features,
            target_column,
            generation,
        )
        print(
            f"  {generation}/{name}: rows={len(subset):,} classes={counts}",
            flush=True,
        )
    return models


def predict_experts(models, frame, features):
    trend_score, masks = route_arrays(frame, features)
    probs = np.zeros((len(frame), 3), dtype=np.float32)
    for name in EXPERT_NAMES:
        mask = masks[name]
        if not mask.any():
            continue
        direction = 1 if name.startswith("long_") else 2
        probs[mask, direction] = predict_positive(
            models[name], frame.loc[mask], features
        )
    probs[:, 0] = 1.0 - np.maximum(probs[:, 1], probs[:, 2])
    return np.clip(probs, 0.0, 1.0), np.abs(trend_score)


def make_params(candidate, extra_cost_points=5.0):
    hours, weekdays = SESSION_PROFILES[candidate["session_profile"]]
    threshold = min(
        candidate.get("long_threshold", candidate["threshold"]),
        candidate.get("short_threshold", candidate["threshold"]),
    )
    return {
        "threshold": threshold,
        "edge_threshold": 0.0,
        "tp_atr": TP_ATR,
        "sl_atr": SL_ATR,
        "min_tp_price": MIN_TP_PRICE,
        "min_sl_price": MIN_SL_PRICE,
        "max_hold": HORIZON,
        "cooldown_ticks": 0,
        "close_on_opposite": False,
        "direction_mode": candidate.get("direction_mode", "both"),
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


def candidate_probs(record, candidate):
    if candidate["generation"].startswith("5_"):
        probs = record["champion_probs"].copy()
    else:
        weight = candidate["balanced_weight"]
        probs = (
            weight * record["balanced_probs"]
            + (1.0 - weight) * record["time_decay_probs"]
        ).astype(np.float32)
    long_threshold = candidate.get("long_threshold", candidate["threshold"])
    short_threshold = candidate.get("short_threshold", candidate["threshold"])
    below_long = probs[:, 1] < long_threshold
    below_short = probs[:, 2] < short_threshold
    if below_long.any() or below_short.any():
        probs = probs.copy()
        probs[below_long, 1] = 0.0
        probs[below_short, 2] = 0.0
        probs[:, 0] = 1.0 - np.maximum(probs[:, 1], probs[:, 2])
    weak = record["trend_strength"] < candidate["min_trend_strength"]
    if weak.any():
        probs = probs.copy()
        probs[weak, 1:] = 0.0
        probs[weak, 0] = 1.0
    return probs


def candidate_stats(frame, record, candidate, cost=5.0):
    return evaluate_frame(
        make_params(candidate, cost),
        frame,
        candidate_probs(record, candidate),
    )


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
        positive_folds = sum(float(stats["pnl"]) > 0.0 for stats in folds)
        worst_pf = min(profit_factor_value(stats) for stats in folds)
        worst_win = min(float(stats["win_rate"]) for stats in folds)
        worst_drawdown = min(float(stats["max_drawdown_pct"]) for stats in folds)
        stopped_out_folds = sum(bool(stats["stopped_out"]) for stats in folds)
        # When no candidate passes, move the next generation toward the worst
        # quality fold first. Frequency is only a tie-breaker after PF and win.
        return (
            -1e12
            - stopped_out_folds * 10_000_000.0
            + positive_folds * 1_000_000.0
            + worst_pf * 100_000.0
            + worst_win * 10_000.0
            + worst_drawdown * 1_000.0
            + sum(float(stats["pnl"]) for stats in folds)
            + math.log1p(sum(float(stats["trades"]) for stats in folds))
        )
    return sum(
        float(stats["pnl"])
        + float(stats["trades"]) * 2.0
        + float(stats["win_rate"]) * 500.0
        + profit_factor_value(stats) * 250.0
        + float(stats["max_drawdown_pct"]) * 300.0
        for stats in folds
    )


def evaluate_candidates(candidates, records, history):
    ranked = []
    for index, candidate in enumerate(candidates, start=1):
        folds = {}
        for record in records:
            frame = history.iloc[record["start_index"] : record["end_index"]]
            folds[record["name"]] = candidate_stats(frame, record, candidate)
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
        if index % 12 == 0 or index == len(candidates):
            print(f"  evaluated {index}/{len(candidates)}", flush=True)
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def bounded_values(center, step, lower, upper):
    return tuple(
        sorted(
            {
                round(max(lower, min(upper, center + offset)), 3)
                for offset in (-step, 0.0, step)
            }
        )
    )


def public_candidate(candidate):
    return {
        "generation": candidate["generation"],
        "balanced_weight": candidate["balanced_weight"],
        "time_decay_weight": round(1.0 - candidate["balanced_weight"], 3),
        "threshold": candidate["threshold"],
        "long_threshold": candidate.get("long_threshold", candidate["threshold"]),
        "short_threshold": candidate.get("short_threshold", candidate["threshold"]),
        "min_trend_strength": candidate["min_trend_strength"],
        "session_profile": candidate["session_profile"],
        "tp_atr": TP_ATR,
        "sl_atr": SL_ATR,
        "max_hold": HORIZON,
        "direction_mode": "both",
        "risk_per_trade": RISK_PER_TRADE,
        "allowed_entry_hours": list(
            SESSION_PROFILES[candidate["session_profile"]][0]
        ),
        "allowed_entry_weekdays": list(
            SESSION_PROFILES[candidate["session_profile"]][1]
        ),
        "model_files": {
            generation: {
                name: path.name for name, path in files.items()
            }
            for generation, files in MODEL_FILES.items()
        },
    }


def markdown_report(report):
    lines = [
        "# GOLD iterative regime experts",
        "",
        "Separate long/short trend/pullback models using first-touch labels.",
        "Generation 1 uses balanced full history; generation 2 adds time decay; generation 3 blends both; generation 4 uses independent direction thresholds; generation 5 adds no-lookahead rolling calibration and dynamic champions.",
        "",
        "| Generation | Qualified | Minimum trades | Worst win | Worst PF | Total PnL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in report["generation_summary"].items():
        folds = list(item["folds"].values())
        finite_profit_factors = [
            stats["profit_factor"]
            for stats in folds
            if stats["profit_factor"] is not None
        ]
        worst_pf = min(finite_profit_factors) if finite_profit_factors else float("inf")
        lines.append(
            f"| {name} | {item['qualified']} | "
            f"{min(stats['trades'] for stats in folds)} | "
            f"{min(stats['win_rate'] for stats in folds):.2%} | "
            f"{worst_pf:.2f} | {sum(stats['pnl'] for stats in folds):.2f} |"
        )
    lines.extend([
        "",
        "| Fold | Trades | Win | PF | PnL | DD |",
        "|---|---:|---:|---:|---:|---:|",
    ])
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


def train_generation_pair(train, validation, features):
    predictions = {}
    trend_strength = None
    for generation in MODEL_PROFILES:
        models = train_experts(train, features, generation)
        predictions[generation], generation_strength = predict_experts(
            models, validation, features
        )
        if trend_strength is None:
            trend_strength = generation_strength
        elif not np.allclose(trend_strength, generation_strength):
            raise RuntimeError("Trend routing differs between generations")
    return predictions, trend_strength


def add_rolling_champion(record, frame):
    probabilities, trace = rolling_champion_probabilities(
        {
            "balanced": record["balanced_probs"],
            "time_decay": record["time_decay_probs"],
        },
        frame["LONG_TARGET"].to_numpy(dtype=np.int8),
        frame["SHORT_TARGET"].to_numpy(dtype=np.int8),
        **ROLLING_CHAMPION_CONFIG,
    )
    record["champion_probs"] = probabilities
    record["champion_trace"] = trace
    return record


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

    times = history["TIME_DT"].to_numpy(dtype="datetime64[ns]")
    records = []
    for fold_name, fold_start, fold_end in SELECTION_FOLDS:
        train = training_frame(history, fold_start)
        start_index = int(np.searchsorted(times, np.datetime64(fold_start), side="left"))
        end_index = int(np.searchsorted(times, np.datetime64(fold_end), side="left"))
        validation = history.iloc[start_index:end_index]
        predictions, trend_strength = train_generation_pair(
            train, validation, features
        )
        record = add_rolling_champion(
            {
                "name": fold_name,
                "start_index": start_index,
                "end_index": end_index,
                "balanced_probs": predictions["balanced"],
                "time_decay_probs": predictions["time_decay"],
                "trend_strength": trend_strength,
            },
            validation,
        )
        records.append(record)
        print(
            f"Fold {fold_name}: train={len(train):,} validation={len(validation):,}",
            flush=True,
        )

    generation_one = [
        {
            "generation": "1_balanced",
            "balanced_weight": 1.0,
            "threshold": threshold,
            "min_trend_strength": strength,
            "session_profile": session,
        }
        for threshold, strength, session in product(
            (0.65, 0.75, 0.80),
            (0.0, 0.2),
            SESSION_PROFILES,
        )
    ]
    print("Generation 1: balanced full-history models", flush=True)
    ranked_one = evaluate_candidates(generation_one, records, history)
    center_one = ranked_one[0]

    generation_two = [
        {
            "generation": "2_time_decay",
            "balanced_weight": 0.0,
            "threshold": threshold,
            "min_trend_strength": strength,
            "session_profile": center_one["session_profile"],
        }
        for threshold, strength in product(
            (0.70, 0.75, 0.80),
            (0.0, 0.2, 0.4),
        )
    ]
    print("Generation 2: time-decayed full-history models", flush=True)
    ranked_two = evaluate_candidates(generation_two, records, history)
    center_two = ranked_two[0]

    better_center = max((center_one, center_two), key=lambda item: item["score"])
    generation_three = [
        {
            "generation": "3_probability_blend",
            "balanced_weight": weight,
            "threshold": threshold,
            "min_trend_strength": strength,
            "session_profile": better_center["session_profile"],
        }
        for weight, threshold, strength in product(
            (0.25, 0.50, 0.75),
            bounded_values(better_center["threshold"], 0.025, 0.45, 0.85),
            bounded_values(
                better_center["min_trend_strength"], 0.1, 0.0, 0.5
            ),
        )
    ]
    print("Generation 3: probability blending and local refinement", flush=True)
    ranked_three = evaluate_candidates(generation_three, records, history)

    center_three = ranked_three[0]
    generation_four = [
        {
            "generation": "4_independent_direction_thresholds",
            "balanced_weight": center_three["balanced_weight"],
            "threshold": min(long_threshold, short_threshold),
            "long_threshold": long_threshold,
            "short_threshold": short_threshold,
            "min_trend_strength": center_three["min_trend_strength"],
            "session_profile": center_three["session_profile"],
        }
        for long_threshold, short_threshold in product(
            (0.725, 0.75, 0.775, 0.80),
            (0.725, 0.75, 0.775, 0.80),
        )
    ]
    print("Generation 4: independent long/short thresholds", flush=True)
    ranked_four = evaluate_candidates(generation_four, records, history)

    generation_five_shared = [
        {
            "generation": "5_rolling_champion_shared",
            "balanced_weight": 0.5,
            "threshold": threshold,
            "min_trend_strength": strength,
            "session_profile": session,
        }
        for threshold, strength, session in product(
            (0.55, 0.60, 0.65, 0.70),
            (0.0, 0.2),
            SESSION_PROFILES,
        )
    ]
    print("Generation 5a: rolling calibration and dynamic champion", flush=True)
    ranked_five_shared = evaluate_candidates(
        generation_five_shared, records, history
    )
    center_five = ranked_five_shared[0]
    generation_five_directional = [
        {
            "generation": "5_rolling_champion_directional",
            "balanced_weight": 0.5,
            "threshold": min(long_threshold, short_threshold),
            "long_threshold": long_threshold,
            "short_threshold": short_threshold,
            "min_trend_strength": center_five["min_trend_strength"],
            "session_profile": center_five["session_profile"],
        }
        for long_threshold, short_threshold in product(
            bounded_values(center_five["threshold"], 0.025, 0.50, 0.75),
            bounded_values(center_five["threshold"], 0.025, 0.50, 0.75),
        )
    ]
    print("Generation 5b: independent calibrated direction thresholds", flush=True)
    ranked_five_directional = evaluate_candidates(
        generation_five_directional, records, history
    )

    ranked = sorted(
        ranked_one
        + ranked_two
        + ranked_three
        + ranked_four
        + ranked_five_shared
        + ranked_five_directional,
        key=lambda item: item["score"],
        reverse=True,
    )
    qualified = [item for item in ranked if item["qualified"]]
    covered = [
        item
        for item in ranked
        if all(stats["trades"] >= 100 for stats in item["folds"].values())
    ]
    selected = qualified[0] if qualified else covered[0] if covered else ranked[0]

    holdout_train = training_frame(history, HISTORICAL_HOLDOUT_START)
    holdout = history[history["TIME_DT"] >= HISTORICAL_HOLDOUT_START]
    holdout_predictions, holdout_strength = train_generation_pair(
        holdout_train, holdout, features
    )
    holdout_record = add_rolling_champion(
        {
            "balanced_probs": holdout_predictions["balanced"],
            "time_decay_probs": holdout_predictions["time_decay"],
            "trend_strength": holdout_strength,
        },
        holdout,
    )
    holdout_stats = candidate_stats(holdout, holdout_record, selected)

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
    recent["LONG_TARGET"] = build_long_first_touch_target(
        recent, HORIZON, TP_ATR, SL_ATR, MIN_TP_PRICE, MIN_SL_PRICE
    )
    recent["SHORT_TARGET"] = build_short_first_touch_target(
        recent, HORIZON, TP_ATR, SL_ATR, MIN_TP_PRICE, MIN_SL_PRICE
    )

    final_cutoff = history["TIME_DT"].iloc[-1] + timedelta(minutes=1)
    final_train = training_frame(history, final_cutoff)
    recent_predictions = {}
    recent_strength = None
    for generation in MODEL_PROFILES:
        models = train_experts(final_train, features, generation)
        for name, model in models.items():
            model.save_model(MODEL_FILES[generation][name])
        recent_predictions[generation], generation_strength = predict_experts(
            models, recent, features
        )
        if recent_strength is None:
            recent_strength = generation_strength
    recent_record = add_rolling_champion(
        {
            "balanced_probs": recent_predictions["balanced"],
            "time_decay_probs": recent_predictions["time_decay"],
            "trend_strength": recent_strength,
        },
        recent,
    )
    recent_stats = candidate_stats(recent, recent_record, selected)
    recent_cost = candidate_stats(recent, recent_record, selected, cost=10.0)
    current_stats = benchmark_current(recent, features)
    may_stats = benchmark_may(recent, features)

    compact_holdout = compact_stats(holdout_stats)
    compact_recent = compact_stats(recent_stats)
    compact_cost = compact_stats(recent_cost)
    compact_current = compact_stats(current_stats)
    compact_may = compact_stats(may_stats)
    current_pf = compact_current["profit_factor"]
    recent_pf = compact_recent["profit_factor"]
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
        and compact_cost["pnl"] > 0
        and compact_cost["profit_factor"] is not None
        and compact_cost["profit_factor"] >= 1.05
    )

    selected_params = public_candidate(selected)
    config = {
        **selected_params,
        "features": features,
        "model_profiles": MODEL_PROFILES,
        "rolling_champion": ROLLING_CHAMPION_CONFIG,
        "promotion_pass": promotion_pass,
    }
    CONFIG_FILE.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "promotion_pass" if promotion_pass else "research_only",
        "iteration_reason": (
            "Previous TP=1.1 SL=2.0 H=180 experts had recent PF below 1.0; "
            "this run aligns first-touch labels and exits to TP=1.3 SL=1.6 H=90, "
            "then compares balanced, time-decayed, blended, independently gated, "
            "and rolling-calibrated dynamic champion generations."
        ),
        "data": {
            "history_rows": len(history),
            "history_start": history["TIME_DT"].iloc[0].isoformat(),
            "history_end": history["TIME_DT"].iloc[-1].isoformat(),
            "recent_rows": len(recent),
            "recent_start": recent["TIME_DT"].iloc[0].isoformat(),
            "recent_end": recent["TIME_DT"].iloc[-1].isoformat(),
        },
        "generation_summary": {
            "1_balanced": ranked_one[0],
            "2_time_decay": ranked_two[0],
            "3_probability_blend": ranked_three[0],
            "4_independent_direction_thresholds": ranked_four[0],
            "5_rolling_champion_shared": ranked_five_shared[0],
            "5_rolling_champion_directional": ranked_five_directional[0],
        },
        "rolling_champion_trace": {
            "selection": {
                record["name"]: record["champion_trace"] for record in records
            },
            "holdout": holdout_record["champion_trace"],
            "recent": recent_record["champion_trace"],
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
