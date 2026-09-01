from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import xgboost as xgb

from barrier_final_train import prepare_barrier_data
from gold_expected_r_champion import (
    EXPERT_DIRECTIONS,
    rolling_top_k_champion_signals,
)
from gold_expected_r_walk_forward import (
    CHAMPION_CONFIG,
    EXTRA_COST_POINTS,
    HORIZON,
    HISTORICAL_HOLDOUT_START,
    RISK_PER_TRADE,
    SESSION_PROFILES,
    SL_ATR,
    TP_ATR,
    add_reward_targets,
    aggregate_score,
    benchmark_current,
    benchmark_may,
    evaluate_frame,
    fold_pass,
    make_params,
    session_mask,
    training_frame,
)
from gold_recent_walk_forward import DEFAULT_TERMINAL, build_feature_frame
from gold_regime_experts_walk_forward import RECENT_START, SELECTION_FOLDS
from gold_short_rule_research import compact_stats


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_JSON = PROJECT_ROOT / "gold_event_rank_walk_forward.json"
REPORT_MD = PROJECT_ROOT / "gold_event_rank_walk_forward.md"
CONFIG_FILE = PROJECT_ROOT / "gold_event_rank_candidate.json"
MODEL_FILES = {
    name: PROJECT_ROOT / f"gold_event_binary_{name}_xgb.json"
    for name in EXPERT_DIRECTIONS
}

EVENT_COOLDOWN_MINUTES = 5
MIN_EVENT_ROWS = 2_000
TOP_K_VALUES = (1,)
MINIMUM_EXPECTED_R_VALUES = (-0.30, -0.20, -0.10, 0.0)
MODEL_PROFILE = {
    "objective": "binary:logistic",
    "n_estimators": 180,
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_child_weight": 100,
    "recency_half_life_days": 730.0,
}
CHAMPION_QUALITY_PROFILES = {
    "nonnegative_realized_r": {
        "minimum_champion_mean_r": 0.0,
        "minimum_champion_profit_factor": 1.0,
    },
}
EVENT_PROFILE = {
    "minimum_abs_mtf_trend": 0.20,
    "minimum_breakout_atr": -0.15,
    "minimum_pullback_depth_atr": 0.25,
    "long_pullback_max_rsi": 50.0,
    "short_pullback_min_rsi": 50.0,
}

GEOMETRY_FEATURES = (
    "BREAKOUT_20_ATR",
    "BREAKDOWN_20_ATR",
    "LONG_PULLBACK_DEPTH_ATR",
    "SHORT_PULLBACK_DEPTH_ATR",
    "CLOSE_EMA20_ATR",
    "CLOSE_EMA50_ATR",
    "SIGNED_BODY_ATR",
    "UPPER_WICK_ATR",
    "LOWER_WICK_ATR",
    "CLOSE_LOCATION",
    "RETURN_1_ATR",
    "RETURN_5_ATR",
    "RETURN_15_ATR",
    "RETURN_60_ATR",
    "EMA20_SLOPE_5_ATR",
    "EMA50_SLOPE_15_ATR",
    "RANGE_POSITION_60",
    "RSI_CHANGE_5",
    "MACD_HIST_CHANGE_5",
    "MTF_TREND_SCORE",
    "MTF_TREND_SLOPE",
)
VOLATILITY_FEATURES = (
    "RANGE_ATR",
    "ATR_MEDIAN_RATIO",
    "ATR_EXPANSION",
)
CONTEXT_FEATURES = (
    "TICKVOL_SURGE",
    "SPREAD_TO_ATR",
    "CLOSE_VWAP240_ATR",
)
EVENT_COLUMNS = tuple(f"EVENT_{name.upper()}" for name in EXPERT_DIRECTIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generation 7 event-driven GOLD Expected-R walk-forward."
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run the no-lookahead and ranker smoke checks only.",
    )
    return parser.parse_args()


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0.0, np.nan)


def _deduplicate(mask: np.ndarray, time: pd.Series) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    result = np.zeros(len(mask), dtype=bool)
    candidates = np.flatnonzero(mask)
    if len(candidates) == 0:
        return result
    minutes = time.to_numpy(dtype="datetime64[m]").astype(np.int64)
    last_minute = None
    for index in candidates:
        current = minutes[index]
        if last_minute is None or current - last_minute >= EVENT_COOLDOWN_MINUTES:
            result[index] = True
            last_minute = current
    return result


def add_event_context(frame: pd.DataFrame, base_features: list[str]) -> pd.DataFrame:
    missing = {
        "TIME_DT",
        "OPEN",
        "HIGH",
        "LOW",
        "CLOSE",
        "M1_RSI",
        "BIAS_20",
        "ROC_5",
    } - set(frame.columns)
    if missing:
        raise ValueError(f"Missing event columns: {sorted(missing)}")

    result = frame.copy()
    close = result["CLOSE"].astype(np.float64)
    open_price = result["OPEN"].astype(np.float64)
    high = result["HIGH"].astype(np.float64)
    low = result["LOW"].astype(np.float64)
    previous_close = close.shift(1)
    true_range = pd.concat(
        (
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ),
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(14).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    prior_high = high.shift(1).rolling(20).max()
    prior_low = low.shift(1).rolling(20).min()
    candle_range = (high - low).replace(0.0, np.nan)
    tick_volume = result.get("TICKVOL", pd.Series(0.0, index=result.index)).astype(
        np.float64
    )
    volume_sum = tick_volume.rolling(240).sum()
    vwap240 = _safe_ratio(
        (close * tick_volume).rolling(240).sum(),
        volume_sum,
    )
    spread = result.get("SPREAD", pd.Series(0.0, index=result.index)).astype(
        np.float64
    )

    raw = pd.DataFrame(index=result.index)
    raw["BREAKOUT_20_ATR"] = _safe_ratio(close - prior_high, atr)
    raw["BREAKDOWN_20_ATR"] = _safe_ratio(prior_low - close, atr)
    raw["LONG_PULLBACK_DEPTH_ATR"] = _safe_ratio(ema20 - low, atr)
    raw["SHORT_PULLBACK_DEPTH_ATR"] = _safe_ratio(high - ema20, atr)
    raw["CLOSE_EMA20_ATR"] = _safe_ratio(close - ema20, atr)
    raw["CLOSE_EMA50_ATR"] = _safe_ratio(close - ema50, atr)
    raw["SIGNED_BODY_ATR"] = _safe_ratio(close - open_price, atr)
    raw["UPPER_WICK_ATR"] = _safe_ratio(
        high - pd.concat((open_price, close), axis=1).max(axis=1), atr
    )
    raw["LOWER_WICK_ATR"] = _safe_ratio(
        pd.concat((open_price, close), axis=1).min(axis=1) - low, atr
    )
    raw["CLOSE_LOCATION"] = _safe_ratio(close - low, candle_range)
    raw["RETURN_1_ATR"] = _safe_ratio(close - close.shift(1), atr)
    raw["RETURN_5_ATR"] = _safe_ratio(close - close.shift(5), atr)
    raw["RETURN_15_ATR"] = _safe_ratio(close - close.shift(15), atr)
    raw["RETURN_60_ATR"] = _safe_ratio(close - close.shift(60), atr)
    raw["EMA20_SLOPE_5_ATR"] = _safe_ratio(ema20 - ema20.shift(5), atr)
    raw["EMA50_SLOPE_15_ATR"] = _safe_ratio(ema50 - ema50.shift(15), atr)
    raw["RANGE_POSITION_60"] = _safe_ratio(
        close - low.rolling(60).min(),
        high.rolling(60).max() - low.rolling(60).min(),
    )
    raw["RANGE_ATR"] = _safe_ratio(high - low, atr)
    raw["ATR_MEDIAN_RATIO"] = _safe_ratio(atr, atr.rolling(240).median())
    raw["ATR_EXPANSION"] = _safe_ratio(atr, atr.rolling(60).mean())
    raw["TICKVOL_SURGE"] = _safe_ratio(
        tick_volume, tick_volume.rolling(240).mean()
    )
    raw["SPREAD_TO_ATR"] = _safe_ratio(spread * 0.01, atr)
    raw["CLOSE_VWAP240_ATR"] = _safe_ratio(close - vwap240, atr)
    shifted_raw_columns = [
        name
        for name in (
            *GEOMETRY_FEATURES,
            *VOLATILITY_FEATURES,
            *CONTEXT_FEATURES,
        )
        if name in raw.columns
    ]
    result[shifted_raw_columns] = raw[shifted_raw_columns].shift(1)
    result["RSI_CHANGE_5"] = result["M1_RSI"].diff(5)
    result["MACD_HIST_CHANGE_5"] = result["MACD_HIST"].diff(5)

    trend_columns = [
        name
        for name in base_features
        if name.endswith("_TREND") and name in result.columns
    ]
    if not trend_columns:
        raise RuntimeError("No higher-timeframe trend features were found")
    result["MTF_TREND_SCORE"] = result[trend_columns].mean(axis=1)
    result["MTF_TREND_SLOPE"] = (
        result["MTF_TREND_SCORE"] - result["MTF_TREND_SCORE"].shift(60)
    )

    trend_score = result["MTF_TREND_SCORE"].to_numpy(dtype=np.float64)
    bias = result["BIAS_20"].to_numpy(dtype=np.float64)
    roc = result["ROC_5"].to_numpy(dtype=np.float64)
    body = result["SIGNED_BODY_ATR"].to_numpy(dtype=np.float64)
    close_location = result["CLOSE_LOCATION"].to_numpy(dtype=np.float64)
    breakout = result["BREAKOUT_20_ATR"].to_numpy(dtype=np.float64)
    breakdown = result["BREAKDOWN_20_ATR"].to_numpy(dtype=np.float64)
    long_depth = result["LONG_PULLBACK_DEPTH_ATR"].to_numpy(dtype=np.float64)
    short_depth = result["SHORT_PULLBACK_DEPTH_ATR"].to_numpy(dtype=np.float64)
    return_15 = result["RETURN_15_ATR"].to_numpy(dtype=np.float64)
    rsi = result["M1_RSI"].to_numpy(dtype=np.float64)
    finite = np.isfinite(
        np.column_stack(
            (
                trend_score,
                bias,
                roc,
                body,
                close_location,
                breakout,
                breakdown,
                long_depth,
                short_depth,
                return_15,
                rsi,
            )
        )
    ).all(axis=1)
    raw_masks = {
        "long_trend": (
            finite
            & (trend_score >= EVENT_PROFILE["minimum_abs_mtf_trend"])
            & (bias >= 0.0)
            & (roc > 0.0)
            & (body > 0.0)
            & (return_15 > 0.0)
            & (breakout >= EVENT_PROFILE["minimum_breakout_atr"])
        ),
        "long_pullback": (
            finite
            & (trend_score >= EVENT_PROFILE["minimum_abs_mtf_trend"])
            & (bias < 0.0)
            & (body > 0.0)
            & (long_depth >= EVENT_PROFILE["minimum_pullback_depth_atr"])
            & (close_location >= 0.55)
            & (rsi <= EVENT_PROFILE["long_pullback_max_rsi"])
        ),
        "short_trend": (
            finite
            & (trend_score <= -EVENT_PROFILE["minimum_abs_mtf_trend"])
            & (bias <= 0.0)
            & (roc < 0.0)
            & (body < 0.0)
            & (return_15 < 0.0)
            & (breakdown >= EVENT_PROFILE["minimum_breakout_atr"])
        ),
        "short_pullback": (
            finite
            & (trend_score <= -EVENT_PROFILE["minimum_abs_mtf_trend"])
            & (bias > 0.0)
            & (body < 0.0)
            & (short_depth >= EVENT_PROFILE["minimum_pullback_depth_atr"])
            & (close_location <= 0.45)
            & (rsi >= EVENT_PROFILE["short_pullback_min_rsi"])
        ),
    }
    for name, mask in raw_masks.items():
        result[f"EVENT_{name.upper()}"] = _deduplicate(mask, result["TIME_DT"])
    return result


def event_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    missing = set(EVENT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing event flags: {sorted(missing)}")
    return {
        name: frame[f"EVENT_{name.upper()}"].to_numpy(dtype=bool)
        for name in EXPERT_DIRECTIONS
    }


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def build_feature_profiles(base_features: list[str]) -> dict[str, list[str]]:
    stable_trend = [
        name
        for name in (
            "H1_TREND",
            "H4_TREND",
            "H12_TREND",
            "Daily_TREND",
            "Weekly_TREND",
        )
        if name in base_features
    ]
    core = [
        name
        for name in ("M1_RSI", "ATR", "MACD_HIST", "BIAS_20", "ROC_5")
        if name in base_features
    ]
    geometry = _ordered_unique(core + stable_trend + list(GEOMETRY_FEATURES))
    geometry_volatility = _ordered_unique(
        geometry
        + [name for name in ("BB_WIDTH", "VOLA_RATIO") if name in base_features]
        + list(VOLATILITY_FEATURES)
    )
    full = _ordered_unique(
        geometry_volatility
        + [
            name
            for name in ("BODY_PCT", "HOUR_SIN", "HOUR_COS", "DAY_OF_WEEK")
            if name in base_features
        ]
        + list(CONTEXT_FEATURES)
    )
    return {
        "stable_geometry": geometry,
        "stable_geometry_volatility": geometry_volatility,
        "stable_full_context": full,
    }


def relevance_labels(reward: np.ndarray) -> np.ndarray:
    reward = np.asarray(reward, dtype=np.float64)
    labels = np.zeros(len(reward), dtype=np.int32)
    labels[reward > -0.50] = 1
    labels[reward > 0.0] = 2
    labels[reward > 0.35] = 3
    labels[reward > 0.75] = 4
    return labels


def _interaction_constraints(features: list[str]) -> str:
    core_names = {
        "M1_RSI",
        "ATR",
        "MACD_HIST",
        "BIAS_20",
        "ROC_5",
        "MTF_TREND_SCORE",
    }
    families = (
        core_names | set(GEOMETRY_FEATURES),
        core_names | set(VOLATILITY_FEATURES) | {"BB_WIDTH", "VOLA_RATIO"},
        core_names | set(CONTEXT_FEATURES) | {"HOUR_SIN", "HOUR_COS", "DAY_OF_WEEK"},
        core_names | {name for name in features if name.endswith("_TREND")},
    )
    groups = [
        [index for index, name in enumerate(features) if name in family]
        for family in families
    ]
    groups = [group for group in groups if group]
    return json.dumps(groups)


def _event_training_rows(
    frame: pd.DataFrame,
    features: list[str],
    mask: np.ndarray,
    target_column: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    feature_values = frame[features].to_numpy(dtype=np.float32)
    valid = (
        mask
        & np.isfinite(feature_values).all(axis=1)
        & np.isfinite(frame[target_column].to_numpy(dtype=np.float64))
    )
    subset = frame.loc[valid, ["TIME_DT", target_column] + features].copy()
    subset = subset.sort_values("TIME_DT")
    labels = (subset[target_column].to_numpy(dtype=np.float64) > 0.0).astype(
        np.int32
    )
    if len(subset) < MIN_EVENT_ROWS:
        raise RuntimeError(f"Event expert has only {len(subset):,} usable rows")
    if len(np.unique(labels)) != 2:
        raise RuntimeError("Event expert requires both winning and losing rows")
    return subset, labels, _event_sample_weights(subset)


def _event_sample_weights(subset: pd.DataFrame) -> np.ndarray:
    latest = subset["TIME_DT"].iloc[-1]
    age_days = (
        (latest - subset["TIME_DT"]).dt.total_seconds().to_numpy() / 86_400.0
    )
    half_life = MODEL_PROFILE["recency_half_life_days"]
    weights = 0.15 + 0.85 * np.exp(-np.log(2.0) * age_days / half_life)
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise RuntimeError("Event sample weights must be finite and positive")
    return weights.astype(np.float32)


def _new_model(
    features: list[str], estimators: int | None = None
) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective=MODEL_PROFILE["objective"],
        tree_method="hist",
        device="cpu",
        n_estimators=estimators or MODEL_PROFILE["n_estimators"],
        learning_rate=MODEL_PROFILE["learning_rate"],
        max_depth=MODEL_PROFILE["max_depth"],
        min_child_weight=MODEL_PROFILE["min_child_weight"],
        subsample=0.85,
        colsample_bytree=0.85,
        interaction_constraints=_interaction_constraints(features),
        random_state=42,
        n_jobs=max(1, (os.cpu_count() or 2) - 1),
        verbosity=0,
    )


def train_event_models(
    frame: pd.DataFrame, features: list[str]
) -> dict[str, dict]:
    masks = event_masks(frame)
    eligible_session = session_mask(frame, "controlled_expanded")
    models = {}
    for name, direction in EXPERT_DIRECTIONS.items():
        target_column = "LONG_REWARD" if direction == 1 else "SHORT_REWARD"
        subset, labels, weights = _event_training_rows(
            frame, features, masks[name] & eligible_session, target_column
        )
        model = _new_model(features)
        model.fit(
            subset[features].astype(np.float32),
            labels,
            sample_weight=weights,
            verbose=False,
        )
        reward = subset[target_column].to_numpy(dtype=np.float64)
        mean_win_r = float(reward[labels == 1].mean())
        mean_loss_r = float(reward[labels == 0].mean())
        models[name] = {
            "model": model,
            "mean_win_r": mean_win_r,
            "mean_loss_r": mean_loss_r,
        }
        print(
            f"  {name}: events={len(subset):,} mean_R={reward.mean():.4f} "
            f"positive={np.mean(reward > 0.0):.2%} "
            f"win_R={mean_win_r:.3f} loss_R={mean_loss_r:.3f}",
            flush=True,
        )
    return models


def _predict_expected_r(expert: dict, frame: pd.DataFrame) -> np.ndarray:
    probability = expert["model"].predict_proba(frame.astype(np.float32))[:, 1]
    return (
        probability * expert["mean_win_r"]
        + (1.0 - probability) * expert["mean_loss_r"]
    )


def predict_event_models(
    models: dict[str, dict],
    frame: pd.DataFrame,
    features: list[str],
) -> dict[str, np.ndarray]:
    masks = event_masks(frame)
    finite = np.isfinite(frame[features].to_numpy(dtype=np.float32)).all(axis=1)
    predictions = {}
    for name in EXPERT_DIRECTIONS:
        values = np.full(len(frame), np.nan, dtype=np.float32)
        valid = masks[name] & finite
        if valid.any():
            values[valid] = _predict_expected_r(
                models[name], frame.loc[valid, features]
            ).astype(np.float32)
        predictions[name] = values
    return predictions


def candidate_signals(
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    candidate: dict,
) -> tuple[np.ndarray, dict]:
    return rolling_top_k_champion_signals(
        predictions,
        frame["LONG_REWARD"].to_numpy(dtype=np.float32),
        frame["SHORT_REWARD"].to_numpy(dtype=np.float32),
        frame["TIME_DT"].dt.date.to_numpy(),
        session_mask(frame, candidate["session_profile"]),
        top_k_per_day=candidate["top_k_per_day"],
        minimum_expected_r=candidate["minimum_expected_r"],
        **CHAMPION_CONFIG,
        **CHAMPION_QUALITY_PROFILES[candidate["champion_quality"]],
    )


def candidate_stats(
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    candidate: dict,
    cost: float = EXTRA_COST_POINTS,
) -> tuple[dict, dict]:
    probabilities, trace = candidate_signals(frame, predictions, candidate)
    return evaluate_frame(make_params(candidate, cost), frame, probabilities), trace


def ranking_diagnostics(
    frame: pd.DataFrame, predictions: dict[str, np.ndarray]
) -> dict[str, dict]:
    result = {}
    eligible_session = session_mask(frame, "controlled_expanded")
    for name, direction in EXPERT_DIRECTIONS.items():
        reward_column = "LONG_REWARD" if direction == 1 else "SHORT_REWARD"
        prediction = predictions[name]
        reward = frame[reward_column].to_numpy(dtype=np.float64)
        valid = eligible_session & np.isfinite(prediction) & np.isfinite(reward)
        values = reward[valid]
        scores = prediction[valid]
        if len(values) == 0:
            result[name] = {"events": 0}
            continue
        keep = max(1, int(np.ceil(len(values) * 0.10)))
        top = values[np.argpartition(scores, len(scores) - keep)[-keep:]]
        result[name] = {
            "events": int(len(values)),
            "all_mean_r": round(float(values.mean()), 6),
            "top_decile_mean_r": round(float(top.mean()), 6),
            "top_decile_lift_r": round(float(top.mean() - values.mean()), 6),
            "top_decile_win_rate": round(float(np.mean(top > 0.0)), 6),
        }
    return result


def _prediction_quality(prediction: np.ndarray, reward: np.ndarray) -> float:
    keep = max(1, int(np.ceil(len(reward) * 0.10)))
    top = reward[np.argpartition(prediction, len(prediction) - keep)[-keep:]]
    return float(top.mean() - reward.mean())


def heldout_group_permutation(
    models: dict[str, dict],
    frame: pd.DataFrame,
    features: list[str],
) -> dict[str, dict]:
    rng = np.random.default_rng(42)
    masks = event_masks(frame)
    eligible_session = session_mask(frame, "controlled_expanded")
    families = {
        "geometry": set(GEOMETRY_FEATURES),
        "volatility": set(VOLATILITY_FEATURES) | {"BB_WIDTH", "VOLA_RATIO"},
        "context": set(CONTEXT_FEATURES) | {"HOUR_SIN", "HOUR_COS", "DAY_OF_WEEK"},
        "higher_timeframe": {name for name in features if name.endswith("_TREND")},
    }
    output = {}
    for name, direction in EXPERT_DIRECTIONS.items():
        target_column = "LONG_REWARD" if direction == 1 else "SHORT_REWARD"
        valid = masks[name] & eligible_session & np.isfinite(
            frame[features].to_numpy(dtype=np.float32)
        ).all(axis=1)
        subset = frame.loc[valid, features].copy()
        reward = frame.loc[valid, target_column].to_numpy(dtype=np.float64)
        finite_reward = np.isfinite(reward)
        subset = subset.loc[finite_reward]
        reward = reward[finite_reward]
        if len(subset) > 20_000:
            chosen = np.sort(rng.choice(len(subset), 20_000, replace=False))
            subset = subset.iloc[chosen]
            reward = reward[chosen]
        baseline_prediction = _predict_expected_r(models[name], subset)
        baseline = _prediction_quality(baseline_prediction, reward)
        drops = {}
        for family_name, family in families.items():
            columns = [column for column in features if column in family]
            if not columns:
                continue
            shuffled = subset.copy()
            order = rng.permutation(len(shuffled))
            shuffled.loc[:, columns] = shuffled[columns].to_numpy()[order]
            shuffled_prediction = _predict_expected_r(models[name], shuffled)
            drops[family_name] = round(
                baseline - _prediction_quality(shuffled_prediction, reward), 6
            )
        output[name] = {
            "events": int(len(subset)),
            "baseline_top_decile_lift_r": round(baseline, 6),
            "importance_drop_r": drops,
        }
    return output


def count_events(frame: pd.DataFrame) -> dict[str, int]:
    masks = event_masks(frame)
    return {name: int(mask.sum()) for name, mask in masks.items()}


def public_candidate(
    candidate: dict, features: list[str], models: dict[str, dict]
) -> dict:
    return {
        "generation": "7_event_expected_r",
        "feature_profile": candidate["feature_profile"],
        "top_k_per_day": candidate["top_k_per_day"],
        "minimum_expected_r": candidate["minimum_expected_r"],
        "session_profile": candidate["session_profile"],
        "champion_quality": candidate["champion_quality"],
        "tp_atr": TP_ATR,
        "sl_atr": SL_ATR,
        "max_hold": HORIZON,
        "direction_mode": "both",
        "risk_per_trade": RISK_PER_TRADE,
        "event_cooldown_minutes": EVENT_COOLDOWN_MINUTES,
        "event_profile": EVENT_PROFILE,
        "features": features,
        "model_files": {name: path.name for name, path in MODEL_FILES.items()},
        "payoff_calibration": {
            name: {
                "mean_win_r": expert["mean_win_r"],
                "mean_loss_r": expert["mean_loss_r"],
            }
            for name, expert in models.items()
        },
        "model_profile": MODEL_PROFILE,
        "champion_config": CHAMPION_CONFIG,
    }


def markdown_report(report: dict) -> str:
    lines = [
        "# GOLD generation 7 event Expected-R walk-forward",
        "",
        "Four event-only probability-to-Expected-R experts with rolling top-k and realized-R champions.",
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
            f"Selected feature profile: `{report['selected']['params']['feature_profile']}`",
            f"Qualified selection candidates: `{report['qualified_count']}`",
            f"Current recent benchmark: `{json.dumps(report['benchmarks']['current'])}`",
            f"Promotion gate: `{'PASS' if report['promotion_pass'] else 'FAIL'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def self_check() -> None:
    rng = np.random.default_rng(7)
    rows = 800
    time = pd.date_range("2026-01-01", periods=rows, freq="min")
    close = 2_000.0 + np.cumsum(rng.normal(0.0, 0.4, rows))
    open_price = close + rng.normal(0.0, 0.1, rows)
    high = np.maximum(open_price, close) + rng.uniform(0.05, 0.5, rows)
    low = np.minimum(open_price, close) - rng.uniform(0.05, 0.5, rows)
    frame = pd.DataFrame(
        {
            "TIME_DT": time,
            "OPEN": open_price,
            "HIGH": high,
            "LOW": low,
            "CLOSE": close,
            "TICKVOL": rng.integers(50, 500, rows),
            "SPREAD": rng.integers(20, 45, rows),
            "M1_RSI": rng.uniform(20, 80, rows),
            "ATR": np.full(rows, 0.8),
            "MACD_HIST": rng.normal(0, 0.1, rows),
            "BB_WIDTH": rng.uniform(0.001, 0.01, rows),
            "BIAS_20": rng.normal(0, 0.002, rows),
            "BODY_PCT": rng.uniform(0, 1, rows),
            "ROC_5": rng.normal(0, 0.001, rows),
            "VOLA_RATIO": rng.uniform(0.5, 1.5, rows),
            "HOUR_SIN": np.sin(2 * np.pi * time.hour / 24),
            "HOUR_COS": np.cos(2 * np.pi * time.hour / 24),
            "DAY_OF_WEEK": time.dayofweek / 7.0,
            "H1_TREND": np.where(np.arange(rows) % 200 < 100, 1, -1),
        }
    )
    base_features = [
        "M1_RSI",
        "ATR",
        "MACD_HIST",
        "BB_WIDTH",
        "BIAS_20",
        "BODY_PCT",
        "ROC_5",
        "VOLA_RATIO",
        "HOUR_SIN",
        "HOUR_COS",
        "DAY_OF_WEEK",
        "H1_TREND",
    ]
    prepared = add_event_context(frame, base_features)
    mutated = frame.copy()
    mutated.loc[650:, ["OPEN", "HIGH", "LOW", "CLOSE"]] *= 1.5
    prepared_mutated = add_event_context(mutated, base_features)
    check_columns = list(GEOMETRY_FEATURES) + list(VOLATILITY_FEATURES) + list(
        CONTEXT_FEATURES
    ) + list(EVENT_COLUMNS)
    pd.testing.assert_frame_equal(
        prepared.loc[:649, check_columns],
        prepared_mutated.loc[:649, check_columns],
    )
    labels = relevance_labels(np.array([-1.0, -0.25, 0.1, 0.5, 1.0]))
    assert labels.tolist() == [0, 1, 2, 3, 4]
    features = build_feature_profiles(base_features)["stable_geometry"]
    toy = pd.DataFrame(rng.normal(size=(40, len(features))), columns=features)
    toy_labels = np.tile(np.array([0, 0, 1, 1]), 10)
    model = _new_model(features, estimators=5)
    model.fit(
        toy,
        toy_labels,
        sample_weight=np.ones(len(toy), dtype=np.float32),
        verbose=False,
    )
    expert = {"model": model, "mean_win_r": 0.8, "mean_loss_r": -1.0}
    assert np.isfinite(_predict_expected_r(expert, toy)).all()
    print("event_rank_self_check_ok")


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0

    history, base_features = prepare_barrier_data()
    history = add_event_context(add_reward_targets(history), base_features)
    feature_profiles = build_feature_profiles(base_features)
    print(
        f"History rows={len(history):,} {history['TIME_DT'].iloc[0]} -> "
        f"{history['TIME_DT'].iloc[-1]} events={count_events(history)}",
        flush=True,
    )

    candidates = [
        {
            "generation": "7_event_expected_r",
            "feature_profile": profile,
            "top_k_per_day": top_k,
            "minimum_expected_r": minimum_expected_r,
            "session_profile": session,
            "champion_quality": champion_quality,
        }
        for profile, top_k, minimum_expected_r, session, champion_quality in product(
            feature_profiles,
            TOP_K_VALUES,
            MINIMUM_EXPECTED_R_VALUES,
            SESSION_PROFILES,
            CHAMPION_QUALITY_PROFILES,
        )
    ]
    fold_results = {index: {} for index in range(len(candidates))}
    fold_predictions: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    fold_diagnostics: dict[str, dict[str, dict]] = {
        profile: {} for profile in feature_profiles
    }
    for fold_name, fold_start, fold_end in SELECTION_FOLDS:
        train = training_frame(history, fold_start)
        validation = history[
            (history["TIME_DT"] >= fold_start) & (history["TIME_DT"] < fold_end)
        ].copy().reset_index(drop=True)
        fold_predictions[fold_name] = {}
        for profile, features in feature_profiles.items():
            print(f"Fold {fold_name} profile={profile}", flush=True)
            models = train_event_models(train, features)
            predictions = predict_event_models(models, validation, features)
            fold_predictions[fold_name][profile] = predictions
            fold_diagnostics[profile][fold_name] = ranking_diagnostics(
                validation, predictions
            )
            for index, candidate in enumerate(candidates):
                if candidate["feature_profile"] != profile:
                    continue
                stats, _ = candidate_stats(validation, predictions, candidate)
                fold_results[index][fold_name] = stats
        print(
            f"Fold {fold_name}: train={len(train):,} validation={len(validation):,} "
            f"events={count_events(validation)}",
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
    selected_profile = selected["feature_profile"]
    selected_features = feature_profiles[selected_profile]
    selection_traces = {}
    for fold_name, _, _ in SELECTION_FOLDS:
        validation = history[
            (history["TIME_DT"] >= dict(
                (name, start) for name, start, _ in SELECTION_FOLDS
            )[fold_name])
            & (history["TIME_DT"] < dict(
                (name, end) for name, _, end in SELECTION_FOLDS
            )[fold_name])
        ].copy().reset_index(drop=True)
        _, trace = candidate_signals(
            validation,
            fold_predictions[fold_name][selected_profile],
            selected,
        )
        selection_traces[fold_name] = trace
    del fold_predictions

    holdout_train = training_frame(history, HISTORICAL_HOLDOUT_START)
    holdout = history[history["TIME_DT"] >= HISTORICAL_HOLDOUT_START].copy()
    print(f"Holdout profile={selected_profile}", flush=True)
    holdout_models = train_event_models(holdout_train, selected_features)
    holdout_predictions = predict_event_models(
        holdout_models, holdout, selected_features
    )
    holdout_stats, holdout_trace = candidate_stats(
        holdout, holdout_predictions, selected
    )
    holdout_diagnostics = ranking_diagnostics(holdout, holdout_predictions)
    permutation = heldout_group_permutation(
        holdout_models, holdout, selected_features
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
    if set(base_features) != set(recent_features):
        raise RuntimeError("Historical and recent feature sets differ")
    recent = add_event_context(add_reward_targets(recent), base_features)

    final_cutoff = history["TIME_DT"].iloc[-1] + timedelta(minutes=1)
    final_train = training_frame(history, final_cutoff)
    print(f"Final profile={selected_profile}", flush=True)
    final_models = train_event_models(final_train, selected_features)
    for name, expert in final_models.items():
        expert["model"].save_model(MODEL_FILES[name])
    recent_predictions = predict_event_models(
        final_models, recent, selected_features
    )
    recent_stats, recent_trace = candidate_stats(recent, recent_predictions, selected)
    recent_cost, _ = candidate_stats(
        recent, recent_predictions, selected, cost=10.0
    )
    recent_diagnostics = ranking_diagnostics(recent, recent_predictions)
    current_stats = benchmark_current(recent, base_features)
    may_stats = benchmark_may(recent, base_features)

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

    selected_params = public_candidate(selected, selected_features, final_models)
    config = {**selected_params, "promotion_pass": promotion_pass}
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
            "history_events": count_events(history),
            "recent_rows": len(recent),
            "recent_start": recent["TIME_DT"].iloc[0].isoformat(),
            "recent_end": recent["TIME_DT"].iloc[-1].isoformat(),
            "recent_events": count_events(recent),
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
        "ranking_diagnostics": {
            "selection": fold_diagnostics[selected_profile],
            "holdout": holdout_diagnostics,
            "recent": recent_diagnostics,
        },
        "heldout_group_permutation": permutation,
        "feature_ablation": {
            profile: [
                item for item in ranked if item["feature_profile"] == profile
            ][0]
            for profile in feature_profiles
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
