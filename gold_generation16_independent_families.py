from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from barrier_classifier_strategy import SPREAD_POINTS
from barrier_final_train import prepare_barrier_data
from gold_expected_r_walk_forward import (
    EXTRA_COST_POINTS,
    HISTORICAL_HOLDOUT_START,
    HORIZON,
    MIN_SL_PRICE,
    RISK_PER_TRADE,
    SL_ATR,
)
from gold_generation11_execution_aligned import add_targets
from gold_generation12_executable_events import sequential_event_indices
from gold_recent_walk_forward import DEFAULT_TERMINAL, build_feature_frame
from gold_regime_experts_iterative import training_frame
from gold_regime_experts_walk_forward import (
    RECENT_START,
    SELECTION_FOLDS,
    benchmark_current,
)


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_JSON = PROJECT_ROOT / "gold_generation16_independent_families.json"
REPORT_MD = PROJECT_ROOT / "gold_generation16_independent_families.md"
CONFIG_FILE = PROJECT_ROOT / "gold_generation16_candidate.json"
GEN15_REPORT = PROJECT_ROOT / "gold_generation15_signal_mining.json"

TARGET_WIN_RATE = 0.60
MIN_PROFIT_FACTOR = 1.0
MIN_SELECTION_TRADES = 20
MAX_DRAWDOWN_PCT = -0.20
MIN_PWIN = 0.60
MIN_EXPECTED_R = 0.0
BASE_COST_POINTS = EXTRA_COST_POINTS
STRESS_COST_POINTS = 10.0

MODEL_PROFILE = {
    "n_estimators": 140,
    "learning_rate": 0.035,
    "max_depth": 3,
    "min_child_weight": 16,
    "fit_ratio": 0.60,
    "calibration_ratio": 0.20,
    "recency_half_life_days": 1_095.0,
}

FAMILIES = (
    "trend_continuation",
    "breakout",
    "pullback_reversal",
    "mean_reversion",
    "volatility_expansion",
)
EXPERTS = tuple(
    f"{direction}_{family}"
    for direction in ("long", "short")
    for family in FAMILIES
)
EXPERT_DIRECTION = {
    name: 1 if name.startswith("long_") else 2 for name in EXPERTS
}
MODEL_FILES = {
    (name, kind): PROJECT_ROOT / f"gold_generation16_{name}_{kind}_xgb.json"
    for name in EXPERTS
    for kind in ("win", "mean_r")
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generation 16 independent GOLD signal-family research."
    )
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def higher_timeframe_score(
    frame: pd.DataFrame, features: list[str]
) -> np.ndarray:
    prefixes = (
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
    columns = [
        name
        for name in features
        if name.endswith("_TREND") and name.startswith(prefixes)
    ]
    if not columns:
        raise RuntimeError("No past-only higher-timeframe trend features found")
    return frame[columns].to_numpy(dtype=np.float32).mean(axis=1)


def family_masks(
    frame: pd.DataFrame, features: list[str]
) -> dict[str, np.ndarray]:
    trend = higher_timeframe_score(frame, features)
    rsi = frame["M1_RSI"].to_numpy(dtype=np.float32)
    bias = frame["BIAS_20"].to_numpy(dtype=np.float32)
    roc = frame["ROC_5"].to_numpy(dtype=np.float32)
    macd = frame["MACD_HIST"].to_numpy(dtype=np.float32)
    body = frame["BODY_PCT"].to_numpy(dtype=np.float32)
    vola = frame["VOLA_RATIO"].to_numpy(dtype=np.float32)
    close = frame["CLOSE"].to_numpy(dtype=np.float64)
    atr = frame["ATR"].to_numpy(dtype=np.float64)
    roc_atr = np.abs(roc) * close / np.maximum(atr, 1e-9)
    finite = np.isfinite(
        np.column_stack((trend, rsi, bias, roc, macd, body, vola, roc_atr))
    ).all(axis=1)

    long_aligned = (roc > 0.0) & (macd > 0.0)
    short_aligned = (roc < 0.0) & (macd < 0.0)
    masks = {
        "long_trend_continuation": (
            (trend >= 0.20)
            & (bias >= 0.0)
            & long_aligned
            & (rsi >= 48.0)
            & (rsi <= 72.0)
            & (vola >= 0.70)
            & (vola <= 1.60)
        ),
        "short_trend_continuation": (
            (trend <= -0.20)
            & (bias <= 0.0)
            & short_aligned
            & (rsi >= 28.0)
            & (rsi <= 52.0)
            & (vola >= 0.70)
            & (vola <= 1.60)
        ),
        "long_breakout": (
            (bias >= 0.0)
            & long_aligned
            & (body >= 0.45)
            & (roc_atr >= 0.12)
        ),
        "short_breakout": (
            (bias <= 0.0)
            & short_aligned
            & (body >= 0.45)
            & (roc_atr >= 0.12)
        ),
        "long_pullback_reversal": (
            (trend >= 0.20)
            & (bias < 0.0)
            & (rsi <= 45.0)
            & (roc > 0.0)
        ),
        "short_pullback_reversal": (
            (trend <= -0.20)
            & (bias > 0.0)
            & (rsi >= 55.0)
            & (roc < 0.0)
        ),
        "long_mean_reversion": (
            (bias <= -0.0010)
            & (rsi <= 32.0)
            & (roc > 0.0)
            & (body >= 0.25)
        ),
        "short_mean_reversion": (
            (bias >= 0.0010)
            & (rsi >= 68.0)
            & (roc < 0.0)
            & (body >= 0.25)
        ),
        "long_volatility_expansion": (
            (vola >= 1.20)
            & long_aligned
            & (body >= 0.50)
            & (roc_atr >= 0.10)
        ),
        "short_volatility_expansion": (
            (vola >= 1.20)
            & short_aligned
            & (body >= 0.50)
            & (roc_atr >= 0.10)
        ),
    }
    return {name: finite & masks[name] for name in EXPERTS}


def rising_edges(mask: np.ndarray) -> np.ndarray:
    return mask & ~np.r_[False, mask[:-1]]


def context_codes(frame: pd.DataFrame) -> np.ndarray:
    hour = frame["TIME_DT"].dt.hour.to_numpy(dtype=np.int16)
    session = np.zeros(len(frame), dtype=np.int16)
    session[(hour >= 6) & (hour <= 12)] = 1
    session[(hour >= 13) & (hour <= 18)] = 2
    session[hour >= 19] = 3
    vola = frame["VOLA_RATIO"].to_numpy(dtype=np.float32)
    vola_code = np.ones(len(frame), dtype=np.int16)
    vola_code[vola < 0.85] = 0
    vola_code[vola > 1.20] = 2
    return session * 10 + vola_code


def expert_event_indices(
    frame: pd.DataFrame, features: list[str]
) -> dict[str, np.ndarray]:
    masks = family_masks(frame, features)
    output = {}
    for name, mask in masks.items():
        direction = EXPERT_DIRECTION[name]
        outcome_column = "LONG_OUTCOME" if direction == 1 else "SHORT_OUTCOME"
        reward_column = "LONG_REWARD" if direction == 1 else "SHORT_REWARD"
        exit_column = "LONG_EXIT_OFFSET" if direction == 1 else "SHORT_EXIT_OFFSET"
        eligible = (
            rising_edges(mask)
            & (frame[outcome_column].to_numpy(dtype=np.int8) >= 0)
            & np.isfinite(frame[reward_column].to_numpy(dtype=np.float32))
        )
        output[name] = sequential_event_indices(
            eligible, frame[exit_column].to_numpy(dtype=np.int16)
        )
    return output


def sample_weights(times: pd.Series, targets: np.ndarray) -> np.ndarray:
    counts = np.bincount(targets, minlength=2)
    class_weight = len(targets) / (2.0 * np.maximum(counts, 1))
    latest = times.iloc[-1]
    age_days = (
        (latest - times).dt.total_seconds().to_numpy(dtype=np.float64) / 86_400.0
    )
    recency = 0.25 + 0.75 * np.exp(
        -math.log(2.0)
        * age_days
        / MODEL_PROFILE["recency_half_life_days"]
    )
    weights = recency * class_weight[targets]
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise RuntimeError("Training weights must be finite and positive")
    return weights.astype(np.float32)


def new_classifier() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
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


def new_regressor() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:pseudohubererror",
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


def profit_factor(rewards: np.ndarray) -> float | None:
    gains = float(rewards[rewards > 0.0].sum())
    losses = float(-rewards[rewards < 0.0].sum())
    return None if losses <= 0.0 else gains / losses


def array_stats(outcomes: np.ndarray, rewards: np.ndarray) -> dict:
    trades = len(outcomes)
    wins = int((outcomes == 1).sum())
    losses = int((outcomes == 2).sum())
    timeouts = int((outcomes == 0).sum())
    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "win_rate": wins / max(trades, 1),
        "profit_factor": profit_factor(rewards),
        "sum_r": float(rewards.sum()) if trades else 0.0,
        "mean_r": float(rewards.mean()) if trades else 0.0,
    }


def accepted_contexts(
    frame: pd.DataFrame,
    indices: np.ndarray,
    probability: np.ndarray,
    expected_r: np.ndarray,
    direction: int,
) -> tuple[set[int], dict]:
    outcome_column = "LONG_OUTCOME" if direction == 1 else "SHORT_OUTCOME"
    reward_column = "LONG_REWARD" if direction == 1 else "SHORT_REWARD"
    selected = (
        (probability >= MIN_PWIN)
        & (expected_r >= MIN_EXPECTED_R)
    )
    chosen = indices[selected]
    codes = context_codes(frame)[chosen]
    outcomes = frame[outcome_column].to_numpy(dtype=np.int8)[chosen]
    rewards = frame[reward_column].to_numpy(dtype=np.float32)[chosen]
    accepted = set()
    profile = {}
    for code in sorted(set(codes.tolist())):
        keep = codes == code
        stats = array_stats(outcomes[keep], rewards[keep])
        profile[str(code)] = stats
        pf = stats["profit_factor"]
        if (
            stats["trades"] >= 8
            and stats["win_rate"] >= TARGET_WIN_RATE
            and pf is not None
            and pf > MIN_PROFIT_FACTOR
            and stats["mean_r"] > 0.0
        ):
            accepted.add(int(code))
    return accepted, profile


def train_experts(
    frame: pd.DataFrame, features: list[str]
) -> tuple[dict[str, dict], dict]:
    event_indices = expert_event_indices(frame, features)
    models = {}
    diagnostics = {}
    for name, indices in event_indices.items():
        direction = EXPERT_DIRECTION[name]
        outcome_column = "LONG_OUTCOME" if direction == 1 else "SHORT_OUTCOME"
        reward_column = "LONG_REWARD" if direction == 1 else "SHORT_REWARD"
        exit_column = "LONG_EXIT_OFFSET" if direction == 1 else "SHORT_EXIT_OFFSET"
        if len(indices) < 700:
            diagnostics[name] = {"status": "insufficient", "events": len(indices)}
            continue
        fit_end = int(len(indices) * MODEL_PROFILE["fit_ratio"])
        calibration_end = int(
            len(indices)
            * (MODEL_PROFILE["fit_ratio"] + MODEL_PROFILE["calibration_ratio"])
        )
        calibration_start_index = int(indices[fit_end])
        policy_start_index = int(indices[calibration_end])
        exits = frame[exit_column].to_numpy(dtype=np.int16)
        fit_indices = indices[
            (indices < calibration_start_index)
            & (indices + exits[indices] < calibration_start_index)
        ]
        calibration_indices = indices[
            (indices >= calibration_start_index)
            & (indices < policy_start_index)
            & (indices + exits[indices] < policy_start_index)
        ]
        policy_indices = indices[indices >= policy_start_index]
        if (
            len(fit_indices) < 350
            or len(calibration_indices) < 100
            or len(policy_indices) < 100
        ):
            diagnostics[name] = {
                "status": "split_insufficient",
                "events": len(indices),
                "fit": len(fit_indices),
                "calibration": len(calibration_indices),
                "policy": len(policy_indices),
            }
            continue
        outcomes = frame[outcome_column].to_numpy(dtype=np.int8)
        rewards = frame[reward_column].to_numpy(dtype=np.float32)
        fit_target = (outcomes[fit_indices] == 1).astype(np.int8)
        calibration_target = (outcomes[calibration_indices] == 1).astype(np.int8)
        if np.unique(fit_target).size != 2 or np.unique(calibration_target).size != 2:
            diagnostics[name] = {"status": "class_insufficient", "events": len(indices)}
            continue
        classifier = new_classifier()
        mean_r_model = new_regressor()
        x_fit = frame.iloc[fit_indices][features].astype(np.float32)
        weights = sample_weights(frame["TIME_DT"].iloc[fit_indices], fit_target)
        classifier.fit(x_fit, fit_target, sample_weight=weights)
        mean_r_model.fit(x_fit, rewards[fit_indices], sample_weight=weights)
        raw_calibration = classifier.predict_proba(
            frame.iloc[calibration_indices][features].astype(np.float32)
        )[:, 1]
        calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        )
        calibrator.fit(raw_calibration, calibration_target)
        calibration_rewards = rewards[calibration_indices]
        average_win_r = float(calibration_rewards[calibration_target == 1].mean())
        average_nonwin_r = float(calibration_rewards[calibration_target == 0].mean())

        x_policy = frame.iloc[policy_indices][features].astype(np.float32)
        policy_probability = calibrator.predict(
            classifier.predict_proba(x_policy)[:, 1]
        ).astype(np.float32)
        probability_r = (
            policy_probability * average_win_r
            + (1.0 - policy_probability) * average_nonwin_r
        )
        policy_mean_r = mean_r_model.predict(x_policy).astype(np.float32)
        policy_expected_r = 0.5 * (probability_r + policy_mean_r)
        contexts, context_profile = accepted_contexts(
            frame,
            policy_indices,
            policy_probability,
            policy_expected_r,
            direction,
        )
        models[name] = {
            "win": classifier,
            "mean_r": mean_r_model,
            "calibrator": calibrator,
            "average_win_r": average_win_r,
            "average_nonwin_r": average_nonwin_r,
            "accepted_contexts": contexts,
            "context_profile": context_profile,
        }
        diagnostics[name] = {
            "status": "trained",
            "events": len(indices),
            "fit": len(fit_indices),
            "calibration": len(calibration_indices),
            "policy": len(policy_indices),
            "fit_last_time": frame["TIME_DT"].iat[int(fit_indices[-1])].isoformat(),
            "fit_max_label_end_index": int(
                np.max(fit_indices + exits[fit_indices])
            ),
            "calibration_start_index": calibration_start_index,
            "calibration_start_time": frame["TIME_DT"].iat[
                calibration_start_index
            ].isoformat(),
            "calibration_max_label_end_index": int(
                np.max(calibration_indices + exits[calibration_indices])
            ),
            "policy_start_index": policy_start_index,
            "policy_start_time": frame["TIME_DT"].iat[
                policy_start_index
            ].isoformat(),
            "fit_win_rate": float(fit_target.mean()),
            "calibration_win_rate": float(calibration_target.mean()),
            "accepted_contexts": sorted(contexts),
            "context_profile": context_profile,
        }
        print(
            f"  {name}: events={len(indices):,} fit={len(fit_indices):,} "
            f"cal={len(calibration_indices):,} policy={len(policy_indices):,} "
            f"contexts={len(contexts)}",
            flush=True,
        )
    return models, diagnostics


def empty_entries() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "index": pd.Series(dtype="int64"),
            "direction": pd.Series(dtype="int8"),
            "priority": pd.Series(dtype="float32"),
            "expert": pd.Series(dtype="object"),
            "family": pd.Series(dtype="object"),
            "context": pd.Series(dtype="int16"),
            "p_win": pd.Series(dtype="float32"),
            "expected_r": pd.Series(dtype="float32"),
        }
    )


def score_experts(
    models: dict[str, dict],
    frame: pd.DataFrame,
    features: list[str],
) -> tuple[dict[str, pd.DataFrame], dict]:
    masks = family_masks(frame, features)
    contexts = context_codes(frame)
    output = {}
    diagnostics = {}
    for name in EXPERTS:
        model = models.get(name)
        if model is None:
            output[name] = empty_entries()
            diagnostics[name] = {"stage1_events": 0, "stage2_rows": 0}
            continue
        indices = np.flatnonzero(rising_edges(masks[name]))
        if len(indices) == 0:
            output[name] = empty_entries()
            diagnostics[name] = {"stage1_events": 0, "stage2_rows": 0}
            continue
        x = frame.iloc[indices][features].astype(np.float32)
        probability = model["calibrator"].predict(
            model["win"].predict_proba(x)[:, 1]
        ).astype(np.float32)
        probability_r = (
            probability * model["average_win_r"]
            + (1.0 - probability) * model["average_nonwin_r"]
        )
        mean_r = model["mean_r"].predict(x).astype(np.float32)
        expected_r = 0.5 * (probability_r + mean_r)
        selected = (
            (probability >= MIN_PWIN)
            & (expected_r >= MIN_EXPECTED_R)
        )
        chosen = indices[selected]
        direction = EXPERT_DIRECTION[name]
        family = name.split("_", 1)[1]
        output[name] = pd.DataFrame(
            {
                "index": chosen.astype(np.int64),
                "direction": np.full(len(chosen), direction, dtype=np.int8),
                "priority": (
                    probability[selected] + 0.05 * np.tanh(expected_r[selected])
                ).astype(np.float32),
                "expert": np.full(len(chosen), name, dtype=object),
                "family": np.full(len(chosen), family, dtype=object),
                "context": contexts[chosen],
                "p_win": probability[selected],
                "expected_r": expected_r[selected],
            }
        )
        diagnostics[name] = {
            "stage1_events": len(indices),
            "stage2_rows": len(chosen),
            "p_win_q50_q90_q99": np.quantile(
                probability, (0.50, 0.90, 0.99)
            ).round(4).tolist(),
            "expected_r_q50_q90_q99": np.quantile(
                expected_r, (0.50, 0.90, 0.99)
            ).round(4).tolist(),
        }
    return output, diagnostics


def candidate_grid() -> list[dict]:
    subsets = {name: (name,) for name in EXPERTS}
    for family in FAMILIES:
        subsets[f"both_{family}"] = (
            f"long_{family}",
            f"short_{family}",
        )
    subsets["all_long"] = tuple(name for name in EXPERTS if name.startswith("long_"))
    subsets["all_short"] = tuple(name for name in EXPERTS if name.startswith("short_"))
    subsets["all_families"] = EXPERTS
    return [
        {
            "candidate_id": f"{label}__{context_mode}__top{top_k}",
            "label": label,
            "experts": experts,
            "context_mode": context_mode,
            "top_k_per_expert_day": top_k,
            "minimum_p_win": MIN_PWIN,
            "minimum_expected_r": MIN_EXPECTED_R,
        }
        for label, experts in subsets.items()
        for context_mode in ("none", "calibrated")
        for top_k in (1, 2)
    ]


def candidate_entries(
    frame: pd.DataFrame,
    scored: dict[str, pd.DataFrame],
    models: dict[str, dict],
    candidate: dict,
) -> pd.DataFrame:
    pieces = []
    for name in candidate["experts"]:
        entries = scored.get(name, empty_entries()).copy()
        if entries.empty:
            continue
        if candidate["context_mode"] == "calibrated":
            accepted = models[name]["accepted_contexts"]
            entries = entries.loc[entries["context"].isin(accepted)].copy()
        if entries.empty:
            continue
        entries["date"] = frame["TIME_DT"].iloc[
            entries["index"].to_numpy(dtype=np.int64)
        ].dt.date.to_numpy()
        entries = (
            entries.sort_values(["date", "priority"], ascending=[True, False])
            .groupby("date", sort=False)
            .head(candidate["top_k_per_expert_day"])
            .drop(columns="date")
        )
        pieces.append(entries)
    if not pieces:
        return empty_entries()
    return pd.concat(pieces, ignore_index=True)


def adjusted_reward(
    frame: pd.DataFrame, index: int, direction: int, cost_points: float
) -> float:
    column = "LONG_REWARD" if direction == 1 else "SHORT_REWARD"
    reward = float(frame[column].iat[index])
    if cost_points == BASE_COST_POINTS:
        return reward
    stop_loss = max(float(frame["ATR"].iat[index]) * SL_ATR, MIN_SL_PRICE)
    extra_price = (cost_points - BASE_COST_POINTS) * 0.01
    return reward - extra_price / (stop_loss + SPREAD_POINTS * 0.01)


def execute_entries(
    frame: pd.DataFrame,
    entries: pd.DataFrame,
    period: str,
    cost_points: float = BASE_COST_POINTS,
) -> list[dict]:
    if entries.empty:
        return []
    ordered = entries.sort_values(
        ["index", "priority"], ascending=[True, False]
    ).drop_duplicates(["index", "direction"])
    records = []
    free_index = 0
    for index, same_bar in ordered.groupby("index", sort=True):
        index = int(index)
        if index < free_index:
            continue
        entry = same_bar.iloc[0]
        direction = int(entry["direction"])
        exit_column = "LONG_EXIT_OFFSET" if direction == 1 else "SHORT_EXIT_OFFSET"
        outcome_column = "LONG_OUTCOME" if direction == 1 else "SHORT_OUTCOME"
        exit_offset = int(frame[exit_column].iat[index])
        reward = adjusted_reward(frame, index, direction, cost_points)
        if exit_offset <= 0 or not np.isfinite(reward):
            continue
        timestamp = frame["TIME_DT"].iat[index]
        records.append(
            {
                "trade_id": f"{timestamp.isoformat()}|{direction}",
                "period": period,
                "index": index,
                "exit_index": index + exit_offset,
                "time": timestamp.isoformat(),
                "direction": direction,
                "expert": str(entry["expert"]),
                "family": str(entry["family"]),
                "context": int(entry["context"]),
                "p_win": float(entry["p_win"]),
                "expected_r": float(entry["expected_r"]),
                "outcome": int(frame[outcome_column].iat[index]),
                "reward": reward,
            }
        )
        free_index = index + exit_offset + 1
    return records


def contribution(records: list[dict]) -> dict:
    if not records:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "timeouts": 0,
            "win_rate": 0.0,
            "profit_factor": None,
            "sum_r": 0.0,
            "mean_r": 0.0,
        }
    outcomes = np.asarray([record["outcome"] for record in records], dtype=np.int8)
    rewards = np.asarray([record["reward"] for record in records], dtype=np.float64)
    return array_stats(outcomes, rewards)


def metrics(records: list[dict], frame: pd.DataFrame) -> dict:
    base = contribution(records)
    evaluated_days = int(frame["TIME_DT"].dt.date.nunique())
    balance = 1000.0
    peak = balance
    max_drawdown = 0.0
    pnl_total = 0.0
    for record in records:
        pnl = balance * RISK_PER_TRADE * record["reward"]
        balance += pnl
        pnl_total += pnl
        peak = max(peak, balance)
        max_drawdown = min(max_drawdown, balance / peak - 1.0)
    direction_contribution = {
        label: contribution(
            [record for record in records if record["direction"] == direction]
        )
        for direction, label in ((1, "long"), (2, "short"))
    }
    expert_contribution = {
        name: contribution(
            [record for record in records if record["expert"] == name]
        )
        for name in EXPERTS
    }
    family_contribution = {
        family: contribution(
            [record for record in records if record["family"] == family]
        )
        for family in FAMILIES
    }
    return {
        **base,
        "evaluated_days": evaluated_days,
        "trades_per_day": base["trades"] / max(evaluated_days, 1),
        "pnl": pnl_total,
        "ending_balance": balance,
        "max_drawdown_pct": max_drawdown,
        "direction_contribution": direction_contribution,
        "expert_contribution": expert_contribution,
        "family_contribution": family_contribution,
    }


def compact_metrics(value: dict) -> dict:
    keys = (
        "trades",
        "evaluated_days",
        "trades_per_day",
        "wins",
        "losses",
        "timeouts",
        "win_rate",
        "profit_factor",
        "pnl",
        "sum_r",
        "mean_r",
        "max_drawdown_pct",
    )
    return {key: value[key] for key in keys}


def compact_ledger(records: list[dict]) -> list[dict]:
    return [dict(record) for record in records]


def compare_records(candidate: list[dict], baseline: list[dict]) -> dict:
    candidate_by_id = {record["trade_id"]: record for record in candidate}
    baseline_by_id = {record["trade_id"]: record for record in baseline}
    added = set(candidate_by_id) - set(baseline_by_id)
    removed = set(baseline_by_id) - set(candidate_by_id)
    return {
        "unique_executable_trades_added": len(added),
        "unique_added_winners": sum(
            candidate_by_id[key]["outcome"] == 1 for key in added
        ),
        "unique_added_losers": sum(
            candidate_by_id[key]["outcome"] == 2 for key in added
        ),
        "unique_added_timeouts": sum(
            candidate_by_id[key]["outcome"] == 0 for key in added
        ),
        "baseline_trades_removed_by_filter_or_occupancy": len(removed),
        "losers_removed": sum(
            baseline_by_id[key]["reward"] <= 0.0 for key in removed
        ),
        "winners_accidentally_removed": sum(
            baseline_by_id[key]["reward"] > 0.0 for key in removed
        ),
        "execution_overlap": len(set(candidate_by_id) & set(baseline_by_id)),
    }


def result_record(
    value: dict,
    stress: dict,
    records: list[dict],
    stress_records: list[dict],
    baseline: list[dict],
) -> dict:
    return {
        "metrics": compact_metrics(value),
        "cost_stress": compact_metrics(stress),
        "comparison": compare_records(records, baseline),
        "direction_contribution": value["direction_contribution"],
        "expert_contribution": value["expert_contribution"],
        "family_contribution": value["family_contribution"],
        "trade_ledger": compact_ledger(records),
        "cost_stress_trade_ledger": compact_ledger(stress_records),
    }


def baseline_records(report: dict, period: str) -> list[dict]:
    return [dict(record) for record in report["baseline_trade_ledgers"][period]]


def evaluate_candidates(
    frame: pd.DataFrame,
    period: str,
    models: dict[str, dict],
    scored: dict[str, pd.DataFrame],
    candidates: list[dict],
    baseline: list[dict],
) -> dict:
    results = {}
    for candidate in candidates:
        entries = candidate_entries(frame, scored, models, candidate)
        records = execute_entries(frame, entries, period)
        stress_records = execute_entries(
            frame, entries, f"{period}_cost_10", STRESS_COST_POINTS
        )
        value = metrics(records, frame)
        stress = metrics(stress_records, frame)
        results[candidate["candidate_id"]] = result_record(
            value, stress, records, stress_records, baseline
        )
    return results


def stage_diagnostics(
    frame: pd.DataFrame,
    features: list[str],
    scored: dict[str, pd.DataFrame],
) -> dict:
    event_indices = expert_event_indices(frame, features)
    output = {}
    for name in EXPERTS:
        direction = EXPERT_DIRECTION[name]
        family = name.split("_", 1)[1]
        indices = event_indices[name]
        stage1 = pd.DataFrame(
            {
                "index": indices,
                "direction": np.full(len(indices), direction),
                "priority": np.zeros(len(indices)),
                "expert": np.full(len(indices), name, dtype=object),
                "family": np.full(len(indices), family, dtype=object),
                "context": context_codes(frame)[indices],
                "p_win": np.zeros(len(indices)),
                "expected_r": np.zeros(len(indices)),
            }
        )
        stage1_records = execute_entries(frame, stage1, f"{name}_stage1")
        stage2_records = execute_entries(frame, scored[name], f"{name}_stage2")
        stage2_stress = execute_entries(
            frame, scored[name], f"{name}_stage2_cost_10", STRESS_COST_POINTS
        )
        output[name] = {
            "stage1": compact_metrics(metrics(stage1_records, frame)),
            "stage2": compact_metrics(metrics(stage2_records, frame)),
            "stage2_cost_stress": compact_metrics(metrics(stage2_stress, frame)),
        }
    return output


def fold_pass(value: dict) -> bool:
    stress = value["cost_stress"]
    base = value["metrics"]
    return bool(
        base["trades"] >= MIN_SELECTION_TRADES
        and base["win_rate"] >= TARGET_WIN_RATE
        and base["profit_factor"] is not None
        and base["profit_factor"] > MIN_PROFIT_FACTOR
        and base["sum_r"] > 0.0
        and base["max_drawdown_pct"] >= MAX_DRAWDOWN_PCT
        and stress["win_rate"] >= TARGET_WIN_RATE
        and stress["profit_factor"] is not None
        and stress["profit_factor"] > MIN_PROFIT_FACTOR
        and stress["sum_r"] > 0.0
    )


def aggregate_candidates(
    candidates: list[dict], fold_results: dict
) -> list[dict]:
    ranked = []
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        values = [
            fold_results[name][candidate_id]
            for name, *_ in SELECTION_FOLDS
        ]
        total_trades = sum(value["metrics"]["trades"] for value in values)
        total_wins = sum(value["metrics"]["wins"] for value in values)
        ranked.append(
            {
                **candidate,
                "qualified": all(fold_pass(value) for value in values),
                "folds_passed": sum(fold_pass(value) for value in values),
                "total_trades": total_trades,
                "weighted_win_rate": total_wins / max(total_trades, 1),
                "minimum_win_rate": min(
                    value["metrics"]["win_rate"] for value in values
                ),
                "minimum_cost_stress_pf": min(
                    (
                        value["cost_stress"]["profit_factor"]
                        if value["cost_stress"]["profit_factor"] is not None
                        else float("inf")
                    )
                    for value in values
                ),
                "total_sum_r": sum(
                    value["metrics"]["sum_r"] for value in values
                ),
                "worst_drawdown_pct": min(
                    value["metrics"]["max_drawdown_pct"] for value in values
                ),
            }
        )
    ranked.sort(
        key=lambda item: (
            item["qualified"],
            item["folds_passed"],
            item["minimum_win_rate"],
            item["total_trades"],
            item["weighted_win_rate"],
            item["minimum_cost_stress_pf"],
        ),
        reverse=True,
    )
    return ranked


def pareto_frontier(ranked: list[dict]) -> list[str]:
    qualified = [item for item in ranked if item["qualified"]]
    output = []
    for item in qualified:
        dominated = any(
            other["weighted_win_rate"] >= item["weighted_win_rate"]
            and other["total_trades"] >= item["total_trades"]
            and (
                other["weighted_win_rate"] > item["weighted_win_rate"]
                or other["total_trades"] > item["total_trades"]
            )
            for other in qualified
            if other["candidate_id"] != item["candidate_id"]
        )
        if not dominated:
            output.append(item["candidate_id"])
    return output


def save_models(models: dict[str, dict]) -> dict:
    output = {}
    for name, model in models.items():
        model["win"].save_model(MODEL_FILES[(name, "win")])
        model["mean_r"].save_model(MODEL_FILES[(name, "mean_r")])
        output[name] = {
            "win_file": MODEL_FILES[(name, "win")].name,
            "mean_r_file": MODEL_FILES[(name, "mean_r")].name,
            "isotonic": {
                "x": model["calibrator"].X_thresholds_.tolist(),
                "y": model["calibrator"].y_thresholds_.tolist(),
            },
            "average_win_r": model["average_win_r"],
            "average_nonwin_r": model["average_nonwin_r"],
            "accepted_contexts": sorted(model["accepted_contexts"]),
            "context_profile": model["context_profile"],
        }
    return output


def markdown_report(report: dict) -> str:
    lines = [
        "# GOLD generation 16 independent signal families",
        "",
        "Family-specific long/short Stage-2 models; fixed 60% calibrated probability floor.",
        "",
        "## Selected candidate",
        "",
        f"Status: `{report['selected']['status']}`",
        f"Parameters: `{json.dumps(report['selected']['params'], ensure_ascii=False)}`",
        "",
        "| Period | Trades | Trades/day | Wins | Losses | Timeouts | Win | PF | PnL | Mean-R | DD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for period, value in report["selected"]["results"].items():
        stats = value["metrics"]
        pf = "inf" if stats["profit_factor"] is None else f"{stats['profit_factor']:.2f}"
        lines.append(
            f"| {period} | {stats['trades']} | {stats['trades_per_day']:.3f} | "
            f"{stats['wins']} | {stats['losses']} | {stats['timeouts']} | "
            f"{stats['win_rate']:.2%} | {pf} | {stats['pnl']:.2f} | "
            f"{stats['mean_r']:.4f} | {stats['max_drawdown_pct']:.2%} |"
        )
    lines.extend(
        [
            "",
            f"Qualified candidates: `{report['selection']['qualified_count']}`",
            f"Pareto frontier: `{json.dumps(report['selection']['pareto_frontier'])}`",
            f"Research success: `{report['research_success']}`",
            "Promotion pass: `False`",
            "",
            "This generation is research_only and does not modify gemini.py.",
        ]
    )
    return "\n".join(lines) + "\n"


def self_check() -> None:
    assert len(EXPERTS) == 10
    assert len(candidate_grid()) == 72
    frame = pd.DataFrame(
        {
            "TIME_DT": pd.date_range("2026-01-01", periods=20, freq="min"),
            "ATR": np.ones(20),
            "LONG_REWARD": np.ones(20),
            "SHORT_REWARD": -np.ones(20),
            "LONG_EXIT_OFFSET": np.full(20, 3),
            "SHORT_EXIT_OFFSET": np.full(20, 3),
            "LONG_OUTCOME": np.ones(20),
            "SHORT_OUTCOME": np.full(20, 2),
        }
    )
    entries = pd.DataFrame(
        {
            "index": [0, 2, 5],
            "direction": [1, 1, 1],
            "priority": [0.8, 0.9, 0.7],
            "expert": ["long_breakout"] * 3,
            "family": ["breakout"] * 3,
            "context": [1, 1, 1],
            "p_win": [0.7, 0.7, 0.7],
            "expected_r": [0.1, 0.1, 0.1],
        }
    )
    records = execute_entries(frame, entries, "self_check")
    assert [record["index"] for record in records] == [0, 5]
    assert contribution(records)["wins"] == 2
    print("generation16_self_check_ok")


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0
    if not GEN15_REPORT.exists():
        raise FileNotFoundError(GEN15_REPORT)
    gen15 = json.loads(GEN15_REPORT.read_text(encoding="utf-8"))
    candidates = candidate_grid()
    history, features = prepare_barrier_data()
    history = add_targets(history)
    print(
        f"History rows={len(history):,} {history['TIME_DT'].iloc[0]} -> "
        f"{history['TIME_DT'].iloc[-1]}",
        flush=True,
    )

    fold_results = {}
    fold_model_diagnostics = {}
    fold_score_diagnostics = {}
    fold_family_diagnostics = {}
    for fold_name, fold_start, fold_end in SELECTION_FOLDS:
        train = training_frame(history, fold_start)
        evaluation = history[
            (history["TIME_DT"] >= fold_start)
            & (history["TIME_DT"] < fold_end)
        ].copy().reset_index(drop=True)
        models, model_diagnostics = train_experts(train, features)
        scored, score_diagnostics = score_experts(models, evaluation, features)
        baseline = baseline_records(gen15, fold_name)
        fold_results[fold_name] = evaluate_candidates(
            evaluation,
            fold_name,
            models,
            scored,
            candidates,
            baseline,
        )
        fold_model_diagnostics[fold_name] = model_diagnostics
        fold_score_diagnostics[fold_name] = score_diagnostics
        fold_family_diagnostics[fold_name] = stage_diagnostics(
            evaluation, features, scored
        )
        print(f"Fold {fold_name} complete", flush=True)

    ranked = aggregate_candidates(candidates, fold_results)
    selected_summary = ranked[0]
    selected_id = selected_summary["candidate_id"]
    selected = next(
        item for item in candidates if item["candidate_id"] == selected_id
    )
    qualified_count = sum(item["qualified"] for item in ranked)

    holdout = history[
        history["TIME_DT"] >= HISTORICAL_HOLDOUT_START
    ].copy().reset_index(drop=True)
    holdout_train = training_frame(history, HISTORICAL_HOLDOUT_START)
    holdout_models, holdout_model_diagnostics = train_experts(
        holdout_train, features
    )
    holdout_scored, holdout_score_diagnostics = score_experts(
        holdout_models, holdout, features
    )
    holdout_result = evaluate_candidates(
        holdout,
        "2025_2026_05_holdout",
        holdout_models,
        holdout_scored,
        [selected],
        baseline_records(gen15, "2025_2026_05_holdout"),
    )[selected_id]
    holdout_family_diagnostics = stage_diagnostics(
        holdout, features, holdout_scored
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
    recent = add_targets(recent)
    final_cutoff = history["TIME_DT"].iloc[-1] + pd.Timedelta(minutes=1)
    final_train = training_frame(history, final_cutoff)
    final_models, final_model_diagnostics = train_experts(final_train, features)
    recent_scored, recent_score_diagnostics = score_experts(
        final_models, recent, features
    )
    recent_result = evaluate_candidates(
        recent,
        "2026_recent",
        final_models,
        recent_scored,
        [selected],
        baseline_records(gen15, "2026_recent"),
    )[selected_id]
    recent_family_diagnostics = stage_diagnostics(
        recent, features, recent_scored
    )
    current_stats = benchmark_current(recent, features)
    current_stats["evaluated_days"] = int(recent["TIME_DT"].dt.date.nunique())
    current_stats["trades_per_day"] = current_stats["trades"] / max(
        current_stats["evaluated_days"], 1
    )

    selected_results = {
        name: fold_results[name][selected_id] for name, *_ in SELECTION_FOLDS
    }
    selected_results["2025_2026_05_holdout"] = holdout_result
    selected_results["2026_recent"] = recent_result
    all_periods_pass = all(fold_pass(value) for value in selected_results.values())
    research_success = bool(selected_summary["qualified"] and all_periods_pass)
    model_files = save_models(final_models)

    config = {
        "generation": "16_independent_families",
        "status": "research_only",
        "candidate_status": (
            "frozen_candidate" if selected_summary["qualified"]
            else "diagnostic_fallback"
        ),
        "selected": selected if selected_summary["qualified"] else None,
        "diagnostic_fallback": (
            None if selected_summary["qualified"] else selected
        ),
        "research_success": research_success,
        "model_profile": MODEL_PROFILE,
        "execution": {
            "horizon": HORIZON,
            "tp_atr": 1.3,
            "sl_atr": 1.6,
            "base_extra_cost_points": BASE_COST_POINTS,
            "stress_extra_cost_points": STRESS_COST_POINTS,
            "risk_per_trade": RISK_PER_TRADE,
            "single_position": True,
            "stop_first_on_same_bar": True,
        },
        "model_files": model_files,
        "promotion_pass": False,
    }
    CONFIG_FILE.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_only",
        "objective": "maximize executable trades/day subject to OOS TP-first win rate >=60%",
        "architecture": {
            "stage1": "five independent entry-time family generators per direction",
            "stage2": "family-specific calibrated P(TP-first) plus Mean-R",
            "selection_threshold": MIN_PWIN,
            "threshold_tuning": False,
            "families": list(FAMILIES),
            "experts": list(EXPERTS),
        },
        "data": {
            "history_rows": len(history),
            "history_start": history["TIME_DT"].iloc[0].isoformat(),
            "history_end": history["TIME_DT"].iloc[-1].isoformat(),
            "recent_rows": len(recent),
            "recent_start": recent["TIME_DT"].iloc[0].isoformat(),
            "recent_end": recent["TIME_DT"].iloc[-1].isoformat(),
        },
        "selection": {
            "candidate_count": len(candidates),
            "qualified_count": qualified_count,
            "pareto_frontier": pareto_frontier(ranked),
            "ranked": ranked,
            "candidate_fold_results": fold_results,
        },
        "selected": {
            "status": (
                "frozen_candidate" if selected_summary["qualified"]
                else "diagnostic_fallback"
            ),
            "params": selected,
            "selection_summary": selected_summary,
            "results": selected_results,
        },
        "family_diagnostics": {
            **fold_family_diagnostics,
            "2025_2026_05_holdout": holdout_family_diagnostics,
            "2026_recent": recent_family_diagnostics,
        },
        "model_diagnostics": {
            **fold_model_diagnostics,
            "2025_2026_05_holdout": holdout_model_diagnostics,
            "2026_recent": final_model_diagnostics,
        },
        "score_diagnostics": {
            **fold_score_diagnostics,
            "2025_2026_05_holdout": holdout_score_diagnostics,
            "2026_recent": recent_score_diagnostics,
        },
        "baselines": {
            "generation15_parent": gen15["parent"],
            "legacy": gen15["legacy_baselines"],
            "production_recent": current_stats,
        },
        "validation_roles": {
            "selection_folds": "chronological model selection only",
            "holdout": "reused historical holdout; not claimed untouched",
            "recent": "monitoring only; excluded from model and candidate selection",
        },
        "research_success": research_success,
        "validation_pending": True,
        "promotion_pass": False,
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
