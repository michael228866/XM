from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from barrier_classifier_strategy import build_first_touch_outcome_and_reward
from barrier_final_train import FINAL_PARAMS, prepare_barrier_data
from barrier_research_suite import predict_positive, train_binary_model
from drl_trading_v2 import add_indicators
from gold_expected_r_champion import (
    EXPERT_DIRECTIONS,
    rolling_top_k_champion_signals,
)
from gold_expected_r_walk_forward import (
    EXTRA_COST_POINTS,
    HISTORICAL_HOLDOUT_START,
    HORIZON,
    MIN_SL_PRICE,
    MIN_TP_PRICE,
    RISK_PER_TRADE,
    SESSION_PROFILES,
    SL_ATR,
    TP_ATR,
    aggregate_score,
    fold_pass,
    session_mask,
)
from gold_recent_walk_forward import DEFAULT_TERMINAL, build_feature_frame, copy_rates
from gold_regime_experts_walk_forward import (
    CURRENT_MODEL_FILE,
    RECENT_START,
    SELECTION_FOLDS,
    benchmark_current,
    evaluate_frame,
    load_model,
)
from gold_short_rule_research import compact_stats


PROJECT_ROOT = Path(__file__).resolve().parent
HISTORICAL_M5 = PROJECT_ROOT / "GOLD#_M5_201401020000_202605082355.csv"
REPORT_JSON = PROJECT_ROOT / "gold_generation8_residual_walk_forward.json"
REPORT_MD = PROJECT_ROOT / "gold_generation8_residual_walk_forward.md"
CONFIG_FILE = PROJECT_ROOT / "gold_generation8_residual_candidate.json"
MODEL_FILES = {
    name: PROJECT_ROOT / f"gold_generation8_{name}_xgb.json"
    for name in EXPERT_DIRECTIONS
}

ANCHOR_THRESHOLD = 0.75
ANCHOR_LABEL_HORIZON = 240
ANCHOR_N_ESTIMATORS = 220
ANCHOR_TRAIN_DAYS = 455
EVENT_COOLDOWN_MINUTES = 5
MIN_EVENT_ROWS = 600
CALIBRATION_RATIO = 0.20
TOP_K_VALUES = (1,)
MINIMUM_EXPECTED_R_VALUES = (0.05, 0.10, 0.20, 0.30, 0.40)
CHAMPION_CONFIG = {
    "maturity_rows": HORIZON,
    "window_rows": 60_000,
    "min_rows": 20_000,
    "block_rows": 10_080,
    "champion_min_trades": 10,
    "switch_margin": 0.05,
    "confirm_blocks": 2,
}
CHAMPION_QUALITY_PROFILES = {
    "quality_115": {
        "minimum_champion_mean_r": 0.05,
        "minimum_champion_profit_factor": 1.15,
    },
    "quality_125": {
        "minimum_champion_mean_r": 0.10,
        "minimum_champion_profit_factor": 1.25,
    },
    "quality_140": {
        "minimum_champion_mean_r": 0.15,
        "minimum_champion_profit_factor": 1.40,
    },
}
MODEL_PROFILE = {
    "objective": "multi:softprob",
    "num_class": 3,
    "n_estimators": 180,
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_child_weight": 60,
    "recency_half_life_days": 730.0,
    "calibration_ratio": CALIBRATION_RATIO,
}
OUTCOME_LABELS = {0: "timeout", 1: "tp_first", 2: "sl_first"}

M5_FEATURES = (
    "M5_RSI",
    "M5_MACD_ATR",
    "M5_BB_WIDTH",
    "M5_BIAS_20",
    "M5_ROC_1",
    "M5_ROC_3",
    "M5_ROC_12",
    "M5_BODY_ATR",
    "M5_UPPER_WICK_ATR",
    "M5_LOWER_WICK_ATR",
    "M5_CLOSE_LOCATION",
    "M5_BREAKOUT_20_ATR",
    "M5_BREAKDOWN_20_ATR",
    "M5_LONG_PULLBACK_ATR",
    "M5_SHORT_PULLBACK_ATR",
    "M5_EMA20_SLOPE_ATR",
    "M5_EMA50_SLOPE_ATR",
    "M5_ATR_RATIO",
    "M5_VOLUME_SURGE",
    "M5_RANGE_POSITION_60",
    "CTX_M15_TREND",
    "CTX_H1_TREND",
    "CTX_H4_TREND",
    "CTX_TREND_SCORE",
    "CTX_M1_RSI",
    "CTX_M1_VOLA_RATIO",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generation 8 M5 residual Expected-R walk-forward."
    )
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0.0, np.nan)


def _load_mt5_export(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep="\t")
    frame.columns = [column.strip("<>").upper() for column in frame.columns]
    required = {"DATE", "TIME", "OPEN", "HIGH", "LOW", "CLOSE"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    frame["TIME_DT"] = pd.to_datetime(
        frame["DATE"].astype(str) + " " + frame["TIME"].astype(str),
        errors="raise",
    )
    return frame.sort_values("TIME_DT").drop_duplicates("TIME_DT").reset_index(drop=True)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    difference = close.diff()
    gain = difference.clip(lower=0.0).rolling(window).mean()
    loss = (-difference.clip(upper=0.0)).rolling(window).mean()
    return 100.0 - 100.0 / (1.0 + gain / (loss + 1e-6))


def add_m5_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = add_indicators(frame).copy()
    close = result["CLOSE"].astype(np.float64)
    open_price = result["OPEN"].astype(np.float64)
    high = result["HIGH"].astype(np.float64)
    low = result["LOW"].astype(np.float64)
    atr = result["ATR"].astype(np.float64)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    candle_range = (high - low).replace(0.0, np.nan)
    prior_high = high.shift(1).rolling(20).max()
    prior_low = low.shift(1).rolling(20).min()
    tick_volume = result.get(
        "TICKVOL", pd.Series(0.0, index=result.index)
    ).astype(np.float64)

    result["SIGNAL_TIME"] = result["TIME_DT"] + pd.Timedelta(minutes=5)
    result["M5_RSI"] = _rsi(close)
    result["M5_MACD_ATR"] = _safe_ratio(result["MACD_HIST"], atr)
    result["M5_BB_WIDTH"] = result["BB_WIDTH"]
    result["M5_BIAS_20"] = result["BIAS_20"]
    result["M5_ROC_1"] = close.pct_change(1)
    result["M5_ROC_3"] = close.pct_change(3)
    result["M5_ROC_12"] = close.pct_change(12)
    result["M5_BODY_ATR"] = _safe_ratio(close - open_price, atr)
    result["M5_UPPER_WICK_ATR"] = _safe_ratio(
        high - pd.concat((open_price, close), axis=1).max(axis=1), atr
    )
    result["M5_LOWER_WICK_ATR"] = _safe_ratio(
        pd.concat((open_price, close), axis=1).min(axis=1) - low, atr
    )
    result["M5_CLOSE_LOCATION"] = _safe_ratio(close - low, candle_range)
    result["M5_BREAKOUT_20_ATR"] = _safe_ratio(close - prior_high, atr)
    result["M5_BREAKDOWN_20_ATR"] = _safe_ratio(prior_low - close, atr)
    result["M5_LONG_PULLBACK_ATR"] = _safe_ratio(ema20 - low, atr)
    result["M5_SHORT_PULLBACK_ATR"] = _safe_ratio(high - ema20, atr)
    result["M5_EMA20_SLOPE_ATR"] = _safe_ratio(ema20 - ema20.shift(3), atr)
    result["M5_EMA50_SLOPE_ATR"] = _safe_ratio(ema50 - ema50.shift(6), atr)
    result["M5_ATR_RATIO"] = _safe_ratio(atr, atr.rolling(240).median())
    result["M5_VOLUME_SURGE"] = _safe_ratio(
        tick_volume, tick_volume.rolling(240).mean()
    )
    result["M5_RANGE_POSITION_60"] = _safe_ratio(
        close - low.rolling(60).min(),
        high.rolling(60).max() - low.rolling(60).min(),
    )
    return result


def add_generation8_targets(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    long_outcome, long_reward = build_first_touch_outcome_and_reward(
        result,
        HORIZON,
        TP_ATR,
        SL_ATR,
        MIN_TP_PRICE,
        MIN_SL_PRICE,
        direction=1,
        extra_cost_points=EXTRA_COST_POINTS,
    )
    short_outcome, short_reward = build_first_touch_outcome_and_reward(
        result,
        HORIZON,
        TP_ATR,
        SL_ATR,
        MIN_TP_PRICE,
        MIN_SL_PRICE,
        direction=2,
        extra_cost_points=EXTRA_COST_POINTS,
    )
    result["LONG_OUTCOME"] = long_outcome
    result["SHORT_OUTCOME"] = short_outcome
    result["LONG_REWARD"] = long_reward
    result["SHORT_REWARD"] = short_reward
    return result


def _deduplicate(mask: np.ndarray, time: pd.Series) -> np.ndarray:
    output = np.zeros(len(mask), dtype=bool)
    minutes = time.to_numpy(dtype="datetime64[m]").astype(np.int64)
    previous = None
    for index in np.flatnonzero(mask):
        if previous is None or minutes[index] - previous >= EVENT_COOLDOWN_MINUTES:
            output[index] = True
            previous = minutes[index]
    return output


def build_m5_event_table(m5: pd.DataFrame, m1: pd.DataFrame) -> pd.DataFrame:
    required_context = ("M15_TREND", "H1_TREND", "H4_TREND", "M1_RSI", "VOLA_RATIO")
    missing = set(required_context) - set(m1.columns)
    if missing:
        raise ValueError(f"M1 frame is missing context columns: {sorted(missing)}")

    result = add_m5_features(m5)
    m1_times = m1["TIME_DT"].to_numpy(dtype="datetime64[ns]")
    signal_times = result["SIGNAL_TIME"].to_numpy(dtype="datetime64[ns]")
    indices = np.searchsorted(m1_times, signal_times, side="left")
    bounded = indices < len(m1)
    safe_indices = np.minimum(indices, max(0, len(m1) - 1))
    delay = np.full(len(result), np.timedelta64(999, "m"), dtype="timedelta64[ns]")
    delay[bounded] = m1_times[safe_indices[bounded]] - signal_times[bounded]
    valid = bounded & (delay >= np.timedelta64(0, "m")) & (delay <= np.timedelta64(2, "m"))
    result = result.loc[valid].copy().reset_index(drop=True)
    mapped = safe_indices[valid]
    result["M1_INDEX"] = mapped
    result["ENTRY_TIME"] = m1.iloc[mapped]["TIME_DT"].to_numpy()
    for source, target in (
        ("M15_TREND", "CTX_M15_TREND"),
        ("H1_TREND", "CTX_H1_TREND"),
        ("H4_TREND", "CTX_H4_TREND"),
        ("M1_RSI", "CTX_M1_RSI"),
        ("VOLA_RATIO", "CTX_M1_VOLA_RATIO"),
        ("LONG_OUTCOME", "LONG_OUTCOME"),
        ("SHORT_OUTCOME", "SHORT_OUTCOME"),
        ("LONG_REWARD", "LONG_REWARD"),
        ("SHORT_REWARD", "SHORT_REWARD"),
    ):
        result[target] = m1.iloc[mapped][source].to_numpy()

    result["CTX_TREND_SCORE"] = result[
        ["CTX_M15_TREND", "CTX_H1_TREND", "CTX_H4_TREND"]
    ].mean(axis=1)
    finite = np.isfinite(result[list(M5_FEATURES)].to_numpy(dtype=np.float64)).all(axis=1)
    trend = result["CTX_TREND_SCORE"].to_numpy(dtype=np.float64)
    bias = result["M5_BIAS_20"].to_numpy(dtype=np.float64)
    body = result["M5_BODY_ATR"].to_numpy(dtype=np.float64)
    return_3 = result["M5_ROC_3"].to_numpy(dtype=np.float64)
    breakout = result["M5_BREAKOUT_20_ATR"].to_numpy(dtype=np.float64)
    breakdown = result["M5_BREAKDOWN_20_ATR"].to_numpy(dtype=np.float64)
    long_depth = result["M5_LONG_PULLBACK_ATR"].to_numpy(dtype=np.float64)
    short_depth = result["M5_SHORT_PULLBACK_ATR"].to_numpy(dtype=np.float64)
    location = result["M5_CLOSE_LOCATION"].to_numpy(dtype=np.float64)
    rsi = result["M5_RSI"].to_numpy(dtype=np.float64)

    raw_masks = {
        "long_trend": finite & (trend >= 1.0 / 3.0) & (bias >= 0.0)
        & (body > 0.0) & (return_3 > 0.0) & (breakout >= -0.30),
        "long_pullback": finite & (trend >= 1.0 / 3.0) & (bias < 0.0)
        & (body > 0.0) & (long_depth >= 0.15) & (location >= 0.55) & (rsi <= 55.0),
        "short_trend": finite & (trend <= -1.0 / 3.0) & (bias <= 0.0)
        & (body < 0.0) & (return_3 < 0.0) & (breakdown >= -0.30),
        "short_pullback": finite & (trend <= -1.0 / 3.0) & (bias > 0.0)
        & (body < 0.0) & (short_depth >= 0.15) & (location <= 0.45) & (rsi >= 45.0),
    }
    for name, mask in raw_masks.items():
        result[f"EVENT_{name.upper()}"] = _deduplicate(mask, result["SIGNAL_TIME"])
    return result


def event_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        name: int(frame[f"EVENT_{name.upper()}"].sum())
        for name in EXPERT_DIRECTIONS
    }


def _sample_weights(frame: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=3)
    balance = len(labels) / (3.0 * np.maximum(counts, 1))
    latest = frame["ENTRY_TIME"].iloc[-1]
    age_days = (
        (latest - frame["ENTRY_TIME"]).dt.total_seconds().to_numpy(dtype=np.float64)
        / 86_400.0
    )
    recency = 0.15 + 0.85 * np.exp(
        -math.log(2.0) * age_days / MODEL_PROFILE["recency_half_life_days"]
    )
    weights = recency * balance[labels]
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise RuntimeError("Generation 8 sample weights must be finite and positive")
    return weights.astype(np.float32)


def _new_model(estimators: int | None = None) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective=MODEL_PROFILE["objective"],
        num_class=MODEL_PROFILE["num_class"],
        tree_method="hist",
        device="cpu",
        n_estimators=estimators or MODEL_PROFILE["n_estimators"],
        learning_rate=MODEL_PROFILE["learning_rate"],
        max_depth=MODEL_PROFILE["max_depth"],
        min_child_weight=MODEL_PROFILE["min_child_weight"],
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=max(1, (os.cpu_count() or 2) - 1),
        verbosity=0,
    )


def train_m5_experts(
    events: pd.DataFrame,
    cutoff_index: int,
    start_index: int = 0,
    skip_incomplete: bool = False,
) -> dict[str, dict]:
    if start_index < 0 or start_index >= cutoff_index:
        raise ValueError("start_index must be non-negative and before cutoff_index")
    models: dict[str, dict] = {}
    mature_index = cutoff_index - HORIZON
    for name, direction in EXPERT_DIRECTIONS.items():
        outcome_column = "LONG_OUTCOME" if direction == 1 else "SHORT_OUTCOME"
        reward_column = "LONG_REWARD" if direction == 1 else "SHORT_REWARD"
        valid = (
            events[f"EVENT_{name.upper()}"].to_numpy(dtype=bool)
            & (events["M1_INDEX"].to_numpy(dtype=np.int64) >= start_index)
            & (events["M1_INDEX"].to_numpy(dtype=np.int64) < mature_index)
            & (events[outcome_column].to_numpy(dtype=np.int8) >= 0)
            & np.isfinite(events[reward_column].to_numpy(dtype=np.float64))
            & np.isfinite(events[list(M5_FEATURES)].to_numpy(dtype=np.float32)).all(axis=1)
        )
        subset = events.loc[valid].sort_values("ENTRY_TIME").reset_index(drop=True)
        if len(subset) < MIN_EVENT_ROWS:
            raise RuntimeError(f"{name} has only {len(subset):,} mature M5 events")
        split = int(len(subset) * (1.0 - CALIBRATION_RATIO))
        calibration_start_index = int(subset["M1_INDEX"].iloc[split])
        fit = subset[
            subset["M1_INDEX"] < calibration_start_index - HORIZON
        ].copy()
        calibration = subset.iloc[split:].copy()
        if len(fit) < 400 or len(calibration) < 100:
            raise RuntimeError(
                f"{name} fit/calibration split too small: {len(fit):,}/{len(calibration):,}"
            )
        fit_labels = fit[outcome_column].to_numpy(dtype=np.int32)
        calibration_labels = calibration[outcome_column].to_numpy(dtype=np.int32)
        if len(np.unique(fit_labels)) != 3:
            if skip_incomplete:
                print(f"  {name}: skipped because the fit lacks an outcome class", flush=True)
                continue
            raise RuntimeError(f"{name} requires all three outcomes in fit")

        model = _new_model()
        model.fit(
            fit[list(M5_FEATURES)].astype(np.float32),
            fit_labels,
            sample_weight=_sample_weights(fit, fit_labels),
            verbose=False,
        )
        raw = model.predict_proba(
            calibration[list(M5_FEATURES)].astype(np.float32)
        )
        calibration_weights = _sample_weights(calibration, calibration_labels)
        calibrators = []
        for outcome in range(3):
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(
                raw[:, outcome],
                (calibration_labels == outcome).astype(np.float32),
                sample_weight=calibration_weights,
            )
            calibrators.append(calibrator)
        reward = calibration[reward_column].to_numpy(dtype=np.float64)
        full_reward = subset[reward_column].to_numpy(dtype=np.float64)
        full_labels = subset[outcome_column].to_numpy(dtype=np.int32)
        payoff = np.array(
            [
                reward[calibration_labels == outcome].mean()
                if np.any(calibration_labels == outcome)
                else full_reward[full_labels == outcome].mean()
                for outcome in range(3)
            ],
            dtype=np.float64,
        )
        models[name] = {
            "model": model,
            "calibrators": calibrators,
            "payoff": payoff,
            "fit_rows": len(fit),
            "calibration_rows": len(calibration),
        }
        print(
            f"  {name}: fit={len(fit):,} calibrate={len(calibration):,} "
            f"payoff(timeout,tp,sl)={np.round(payoff, 3).tolist()}",
            flush=True,
        )
    return models


def _calibrated_probabilities(expert: dict, features: pd.DataFrame) -> np.ndarray:
    raw = expert["model"].predict_proba(features.astype(np.float32))
    calibrated = np.column_stack(
        [
            expert["calibrators"][outcome].predict(raw[:, outcome])
            for outcome in range(3)
        ]
    )
    totals = calibrated.sum(axis=1, keepdims=True)
    invalid = ~np.isfinite(totals[:, 0]) | (totals[:, 0] <= 0.0)
    calibrated[invalid] = raw[invalid]
    totals = calibrated.sum(axis=1, keepdims=True)
    return calibrated / np.maximum(totals, 1e-12)


def predict_expected_r_to_m1(
    models: dict[str, dict], events: pd.DataFrame, m1: pd.DataFrame
) -> dict[str, np.ndarray]:
    target_times = m1["TIME_DT"].to_numpy(dtype="datetime64[ns]")
    predictions = {
        name: np.full(len(m1), np.nan, dtype=np.float32)
        for name in EXPERT_DIRECTIONS
    }
    for name, expert in models.items():
        valid = (
            events[f"EVENT_{name.upper()}"].to_numpy(dtype=bool)
            & np.isfinite(events[list(M5_FEATURES)].to_numpy(dtype=np.float32)).all(axis=1)
        )
        subset = events.loc[valid]
        if subset.empty:
            continue
        indices = np.searchsorted(
            target_times,
            subset["ENTRY_TIME"].to_numpy(dtype="datetime64[ns]"),
            side="left",
        )
        in_range = indices < len(m1)
        indices = indices[in_range]
        subset = subset.iloc[np.flatnonzero(in_range)]
        exact = target_times[indices] == subset["ENTRY_TIME"].to_numpy(dtype="datetime64[ns]")
        if not exact.any():
            continue
        indices = indices[exact]
        subset = subset.iloc[np.flatnonzero(exact)]
        probabilities = _calibrated_probabilities(
            expert, subset[list(M5_FEATURES)]
        )
        predictions[name][indices] = (
            probabilities @ expert["payoff"]
        ).astype(np.float32)
    return predictions


def _anchor_eligible(frame: pd.DataFrame, probability: np.ndarray) -> np.ndarray:
    time = frame["TIME_DT"]
    rsi = frame["M1_RSI"].to_numpy(dtype=np.float64)
    hours = (0, 1, 2, 3, 4, 8, 9, 11, 12, 17, 18, 19, 20, 22, 23)
    return (
        (probability >= ANCHOR_THRESHOLD)
        & time.dt.hour.isin(hours).to_numpy()
        & time.dt.dayofweek.isin((0, 1, 2, 3, 4)).to_numpy()
        & (rsi > 22.0)
        & ~((rsi >= 35.0) & (rsi <= 45.0))
    )


def _anchor_probabilities(model: xgb.XGBClassifier, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return predict_positive(model, frame, features).astype(np.float32)


def _anchor_signal_probs(frame: pd.DataFrame, probability: np.ndarray) -> np.ndarray:
    mask = _anchor_eligible(frame, probability)
    output = np.zeros((len(frame), 3), dtype=np.float32)
    output[:, 0] = 1.0
    output[mask, 0] = 0.0
    output[mask, 1] = 1.0
    return output


def _execution_params(extra_cost_points: float) -> dict:
    params = dict(FINAL_PARAMS)
    params.update(
        {
            "threshold": 0.5,
            "edge_threshold": 0.0,
            "tp_atr": TP_ATR,
            "sl_atr": SL_ATR,
            "min_tp_price": MIN_TP_PRICE,
            "min_sl_price": MIN_SL_PRICE,
            "max_hold": HORIZON,
            "direction_mode": "both",
            "risk_per_trade": RISK_PER_TRADE,
            "allowed_entry_hours": list(range(24)),
            "allowed_entry_weekdays": [0, 1, 2, 3, 4],
            "excluded_rsi_ranges": [],
            "extra_cost_points": extra_cost_points,
        }
    )
    return params


def _combine_anchor_residual(
    anchor: np.ndarray, residual: np.ndarray
) -> np.ndarray:
    if anchor.shape != residual.shape or anchor.ndim != 2 or anchor.shape[1] != 3:
        raise ValueError("Anchor and residual probabilities must have equal Nx3 shapes")
    output = residual.copy()
    anchor_long = anchor[:, 1] >= 0.5
    output[anchor_long] = anchor[anchor_long]
    no_signal = np.maximum(output[:, 1], output[:, 2]) < 0.5
    output[no_signal, 0] = 1.0
    output[~no_signal, 0] = 0.0
    return output


def residual_signals(
    frame: pd.DataFrame, predictions: dict[str, np.ndarray], candidate: dict
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
    anchor_probability: np.ndarray,
    candidate: dict,
    cost: float = EXTRA_COST_POINTS,
) -> tuple[dict, dict]:
    residual, trace = residual_signals(frame, predictions, candidate)
    anchor = _anchor_signal_probs(frame, anchor_probability)
    combined = _combine_anchor_residual(anchor, residual)
    return evaluate_frame(_execution_params(cost), frame, combined), trace


def anchor_stats(
    frame: pd.DataFrame,
    anchor_probability: np.ndarray,
    cost: float = EXTRA_COST_POINTS,
) -> dict:
    return evaluate_frame(
        _execution_params(cost), frame, _anchor_signal_probs(frame, anchor_probability)
    )


def _cutoff_index(frame: pd.DataFrame, cutoff: datetime) -> int:
    times = frame["TIME_DT"].to_numpy(dtype="datetime64[ns]")
    return int(np.searchsorted(times, np.datetime64(cutoff), side="left"))


def train_anchor(
    frame: pd.DataFrame, features: list[str], cutoff_index: int
) -> xgb.XGBClassifier:
    train_end = cutoff_index - ANCHOR_LABEL_HORIZON
    cutoff_time = frame["TIME_DT"].iloc[cutoff_index]
    train_start_time = cutoff_time - timedelta(days=ANCHOR_TRAIN_DAYS)
    train_start = _cutoff_index(frame, train_start_time)
    if train_end - train_start < 100_000:
        raise RuntimeError(
            f"Anchor training fold too small: {train_end - train_start:,}"
        )
    return train_binary_model(
        frame.iloc[train_start:train_end],
        features,
        positive_class=1,
        n_estimators=ANCHOR_N_ESTIMATORS,
    )


def _profit_factor(stats: dict) -> float:
    value = stats["profit_factor"]
    return 3.0 if value is None else float(value)


def public_model_bundle(models: dict[str, dict]) -> dict:
    return {
        name: {
            "file": MODEL_FILES[name].name,
            "fit_rows": expert["fit_rows"],
            "calibration_rows": expert["calibration_rows"],
            "payoff_r": {
                OUTCOME_LABELS[outcome]: float(expert["payoff"][outcome])
                for outcome in range(3)
            },
            "isotonic": {
                OUTCOME_LABELS[outcome]: {
                    "x": expert["calibrators"][outcome].X_thresholds_.tolist(),
                    "y": expert["calibrators"][outcome].y_thresholds_.tolist(),
                }
                for outcome in range(3)
            },
        }
        for name, expert in models.items()
    }


def public_candidate(candidate: dict, models: dict[str, dict]) -> dict:
    return {
        "generation": "8_m5_residual_expected_r",
        "architecture": "incumbent_long_anchor_plus_nonconflicting_m5_residual",
        "outcomes": OUTCOME_LABELS,
        "top_k_per_day": candidate["top_k_per_day"],
        "minimum_expected_r": candidate["minimum_expected_r"],
        "session_profile": candidate["session_profile"],
        "champion_quality": candidate["champion_quality"],
        "anchor_threshold": ANCHOR_THRESHOLD,
        "tp_atr": TP_ATR,
        "sl_atr": SL_ATR,
        "max_hold": HORIZON,
        "risk_per_trade": RISK_PER_TRADE,
        "features": list(M5_FEATURES),
        "model_profile": MODEL_PROFILE,
        "champion_config": CHAMPION_CONFIG,
        "models": public_model_bundle(models),
    }


def markdown_report(report: dict) -> str:
    lines = [
        "# GOLD generation 8 M5 residual walk-forward",
        "",
        "Current long model stays the anchor; four calibrated M5 Expected-R experts may only add non-conflicting trades.",
        "",
        "| Fold | Combined trades | Win | PF | PnL | DD | Anchor trades | Anchor win | Anchor PF |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    anchor_folds = report["anchor_folds"]
    for name, stats in report["selected"]["folds"].items():
        anchor = anchor_folds.get(name, report["benchmarks"]["current"])
        pf = "inf" if stats["profit_factor"] is None else f"{stats['profit_factor']:.2f}"
        anchor_pf = "inf" if anchor["profit_factor"] is None else f"{anchor['profit_factor']:.2f}"
        lines.append(
            f"| {name} | {stats['trades']} | {stats['win_rate']:.2%} | {pf} | "
            f"{stats['pnl']:.2f} | {stats['max_drawdown_pct']:.2%} | "
            f"{anchor['trades']} | {anchor['win_rate']:.2%} | {anchor_pf} |"
        )
    lines.extend(
        [
            "",
            f"Qualified selection candidates: `{report['qualified_count']}`",
            "Selected: `"
            f"min_R={report['selected']['params']['minimum_expected_r']}, "
            f"session={report['selected']['params']['session_profile']}, "
            f"quality={report['selected']['params']['champion_quality']}`",
            f"Promotion gate: `{'PASS' if report['promotion_pass'] else 'FAIL'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def self_check() -> None:
    rng = np.random.default_rng(8)
    rows = 900
    m1_time = pd.date_range("2026-01-01", periods=rows, freq="min")
    close = 2_000.0 + np.cumsum(rng.normal(0.0, 0.25, rows))
    m1 = pd.DataFrame(
        {
            "TIME_DT": m1_time,
            "OPEN": close,
            "HIGH": close + 0.3,
            "LOW": close - 0.3,
            "CLOSE": close,
            "ATR": np.full(rows, 0.5),
            "M1_RSI": rng.uniform(25.0, 75.0, rows),
            "VOLA_RATIO": np.ones(rows),
            "M15_TREND": np.where(np.arange(rows) < 450, 1, -1),
            "H1_TREND": np.where(np.arange(rows) < 450, 1, -1),
            "H4_TREND": np.where(np.arange(rows) < 450, 1, -1),
        }
    )
    m1 = add_generation8_targets(m1)
    m5 = pd.DataFrame(
        {
            "TIME_DT": m1_time[::5],
            "OPEN": close[::5],
            "HIGH": close[::5] + 0.4,
            "LOW": close[::5] - 0.4,
            "CLOSE": close[::5] + rng.normal(0.0, 0.1, len(m1_time[::5])),
            "TICKVOL": rng.integers(50, 500, len(m1_time[::5])),
        }
    )
    prepared = build_m5_event_table(m5, m1)
    assert (prepared["ENTRY_TIME"] >= prepared["SIGNAL_TIME"]).all()
    assert set(M5_FEATURES).issubset(prepared.columns)

    toy = pd.DataFrame(rng.normal(size=(300, len(M5_FEATURES))), columns=M5_FEATURES)
    labels = np.tile(np.array([0, 1, 2], dtype=np.int32), 100)
    model = _new_model(estimators=5)
    model.fit(toy, labels)
    raw = model.predict_proba(toy)
    calibrators = []
    for outcome in range(3):
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw[:, outcome], (labels == outcome).astype(np.float32))
        calibrators.append(calibrator)
    expert = {
        "model": model,
        "calibrators": calibrators,
        "payoff": np.array([-0.1, 0.8, -1.0]),
    }
    calibrated = _calibrated_probabilities(expert, toy)
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-6)

    anchor = np.zeros((4, 3), dtype=np.float32)
    residual = np.zeros((4, 3), dtype=np.float32)
    anchor[:, 0] = residual[:, 0] = 1.0
    anchor[1] = (0.0, 1.0, 0.0)
    residual[1] = residual[2] = (0.0, 0.0, 1.0)
    combined = _combine_anchor_residual(anchor, residual)
    assert combined[1].tolist() == [0.0, 1.0, 0.0]
    assert combined[2].tolist() == [0.0, 0.0, 1.0]

    same_bar = pd.DataFrame(
        {
            "CLOSE": [100.0, 100.0],
            "HIGH": [100.0, 102.0],
            "LOW": [100.0, 98.0],
            "ATR": [1.0, 1.0],
        }
    )
    outcome, _ = build_first_touch_outcome_and_reward(
        same_bar, 1, 1.0, 1.0, 0.0, 0.0, direction=1
    )
    assert outcome[0] == 2
    print("generation8_self_check_ok")


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0

    history, features = prepare_barrier_data()
    history = add_generation8_targets(history)
    historical_m5 = _load_mt5_export(HISTORICAL_M5)
    historical_events = build_m5_event_table(historical_m5, history)
    print(
        f"History rows={len(history):,} {history['TIME_DT'].iloc[0]} -> "
        f"{history['TIME_DT'].iloc[-1]} M5 events={event_counts(historical_events)}",
        flush=True,
    )

    candidates = [
        {
            "generation": "8_m5_residual_expected_r",
            "top_k_per_day": top_k,
            "minimum_expected_r": minimum_r,
            "session_profile": profile,
            "champion_quality": champion_quality,
        }
        for top_k, minimum_r, profile, champion_quality in product(
            TOP_K_VALUES,
            MINIMUM_EXPECTED_R_VALUES,
            ("controlled_expanded",),
            CHAMPION_QUALITY_PROFILES,
        )
    ]
    fold_results = {index: {} for index in range(len(candidates))}
    fold_traces = {index: {} for index in range(len(candidates))}
    anchor_folds = {}
    for fold_name, fold_start, fold_end in SELECTION_FOLDS:
        cutoff_index = _cutoff_index(history, fold_start)
        validation = history[
            (history["TIME_DT"] >= fold_start) & (history["TIME_DT"] < fold_end)
        ].copy().reset_index(drop=True)
        print(f"Fold {fold_name}: train cutoff={cutoff_index:,}", flush=True)
        experts = train_m5_experts(historical_events, cutoff_index)
        predictions = predict_expected_r_to_m1(
            experts, historical_events, validation
        )
        anchor_model = train_anchor(history, features, cutoff_index)
        anchor_probability = _anchor_probabilities(anchor_model, validation, features)
        anchor_folds[fold_name] = compact_stats(
            anchor_stats(validation, anchor_probability)
        )
        for index, candidate in enumerate(candidates):
            stats, trace = candidate_stats(
                validation, predictions, anchor_probability, candidate
            )
            fold_results[index][fold_name] = stats
            fold_traces[index][fold_name] = trace

    ranked = []
    for index, candidate in enumerate(candidates):
        fold_values = list(fold_results[index].values())
        ranked.append(
            {
                **candidate,
                "candidate_index": index,
                "qualified": all(fold_pass(stats, 100) for stats in fold_values),
                "score": aggregate_score(fold_values),
                "folds": {
                    name: compact_stats(stats)
                    for name, stats in fold_results[index].items()
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
    selected_index = selected["candidate_index"]

    holdout_cutoff = _cutoff_index(history, HISTORICAL_HOLDOUT_START)
    holdout = history[history["TIME_DT"] >= HISTORICAL_HOLDOUT_START].copy().reset_index(drop=True)
    print("Historical holdout", flush=True)
    holdout_experts = train_m5_experts(historical_events, holdout_cutoff)
    holdout_predictions = predict_expected_r_to_m1(
        holdout_experts, historical_events, holdout
    )
    holdout_anchor_model = train_anchor(history, features, holdout_cutoff)
    holdout_anchor_probability = _anchor_probabilities(
        holdout_anchor_model, holdout, features
    )
    holdout_stats, holdout_trace = candidate_stats(
        holdout, holdout_predictions, holdout_anchor_probability, selected
    )
    anchor_folds["2025_2026_05_holdout"] = compact_stats(
        anchor_stats(holdout, holdout_anchor_probability)
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
        recent_m5 = copy_rates("M5", RECENT_START - timedelta(days=10), recent_end)
    finally:
        mt5.shutdown()
    if set(features) != set(recent_features):
        raise RuntimeError("Historical and recent feature sets differ")
    recent = add_generation8_targets(recent)
    recent_events = build_m5_event_table(recent_m5, recent)
    print(
        f"Recent rows={len(recent):,} events={event_counts(recent_events)}",
        flush=True,
    )

    final_experts = train_m5_experts(historical_events, len(history))
    for name, expert in final_experts.items():
        expert["model"].save_model(MODEL_FILES[name])
    recent_predictions = predict_expected_r_to_m1(
        final_experts, recent_events, recent
    )
    current_model = load_model(CURRENT_MODEL_FILE)
    current_probability = _anchor_probabilities(current_model, recent, features)
    recent_stats, recent_trace = candidate_stats(
        recent, recent_predictions, current_probability, selected
    )
    recent_cost_stats, _ = candidate_stats(
        recent, recent_predictions, current_probability, selected, cost=10.0
    )
    current_stats = benchmark_current(recent, features)

    compact_holdout = compact_stats(holdout_stats)
    compact_recent = compact_stats(recent_stats)
    compact_recent_cost = compact_stats(recent_cost_stats)
    compact_current = compact_stats(current_stats)
    recent_pf = _profit_factor(compact_recent)
    current_pf = _profit_factor(compact_current)
    promotion_pass = bool(
        qualified
        and fold_pass(holdout_stats, 100)
        and fold_pass(recent_stats, 40)
        and compact_recent["trades"] > compact_current["trades"]
        and compact_recent["win_rate"] > compact_current["win_rate"]
        and compact_recent["pnl"] > compact_current["pnl"]
        and recent_pf >= 1.15
        and recent_pf > current_pf
        and compact_recent_cost["pnl"] > 0.0
        and _profit_factor(compact_recent_cost) >= 1.05
    )

    selected_params = public_candidate(selected, final_experts)
    config = {**selected_params, "promotion_pass": promotion_pass}
    CONFIG_FILE.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    selected_folds = {
        **selected["folds"],
        "2025_2026_05_holdout": compact_holdout,
        "2026_recent": compact_recent,
        "2026_recent_cost_10": compact_recent_cost,
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "promotion_pass" if promotion_pass else "research_only",
        "data": {
            "history_rows": len(history),
            "history_start": history["TIME_DT"].iloc[0].isoformat(),
            "history_end": history["TIME_DT"].iloc[-1].isoformat(),
            "historical_m5_rows": len(historical_m5),
            "historical_events": event_counts(historical_events),
            "recent_rows": len(recent),
            "recent_start": recent["TIME_DT"].iloc[0].isoformat(),
            "recent_end": recent["TIME_DT"].iloc[-1].isoformat(),
            "recent_events": event_counts(recent_events),
        },
        "qualified_count": len(qualified),
        "selected": {"params": selected_params, "folds": selected_folds},
        "anchor_folds": anchor_folds,
        "champion_trace": {
            "selection": fold_traces[selected_index],
            "holdout": holdout_trace,
            "recent": recent_trace,
        },
        "benchmarks": {"current": compact_current},
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
