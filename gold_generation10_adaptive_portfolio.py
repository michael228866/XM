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

from barrier_final_train import prepare_barrier_data
from gold_expected_r_champion import _realized_metrics, _top_k_threshold
from gold_expected_r_walk_forward import (
    EXTRA_COST_POINTS,
    HISTORICAL_HOLDOUT_START,
    HORIZON,
    aggregate_score,
    fold_pass,
    session_mask,
)
from gold_generation8_residual_walk_forward import (
    HISTORICAL_M5,
    _anchor_eligible,
    _anchor_probabilities,
    _cutoff_index,
    _execution_params,
    _load_mt5_export,
    add_generation8_targets,
    anchor_stats,
    build_m5_event_table,
    event_counts,
    predict_expected_r_to_m1,
    train_anchor,
    train_m5_experts,
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
REPORT_JSON = PROJECT_ROOT / "gold_generation10_adaptive_portfolio.json"
REPORT_MD = PROJECT_ROOT / "gold_generation10_adaptive_portfolio.md"
CONFIG_FILE = PROJECT_ROOT / "gold_generation10_candidate.json"

WINDOWS = {"slow": None, "medium": 1_095, "fast": 455}
EXPERT_NAMES = ("long_trend", "long_pullback", "short_trend", "short_pullback")
ARM_DIRECTIONS = {
    "anchor_long": 1,
    **{
        f"{window}_{expert}": 1 if expert.startswith("long_") else 2
        for window in WINDOWS
        for expert in EXPERT_NAMES
    },
}
BASE_MODEL_FILES = {
    arm: PROJECT_ROOT / f"gold_generation10_{arm}_xgb.json"
    for arm in ARM_DIRECTIONS
    if arm != "anchor_long"
}
META_MODEL_FILES = {
    (direction, kind): PROJECT_ROOT
    / f"gold_generation10_meta_{direction}_{kind}_xgb.json"
    for direction in ("long", "short")
    for kind in ("win", "mean_r", "q35_r")
}

OOF_START_YEAR = 2016
META_CALIBRATION_RATIO = 0.20
META_MIN_ROWS = 2_000
RANK_QUANTILES = np.linspace(0.0, 1.0, 101, dtype=np.float32)
META_PROFILE = {
    "n_estimators": 180,
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_child_weight": 80,
    "recency_half_life_days": 730.0,
    "quantile_alpha": 0.35,
}
ALLOCATOR_CONFIG = {
    "maturity_rows": HORIZON,
    "window_rows": 60_000,
    "min_rows": 20_000,
    "block_rows": 10_080,
    "champion_min_trades": 10,
    "switch_margin": 0.05,
    "confirm_blocks": 2,
}
QUALITY_PROFILES = {
    "quality_115": {"minimum_mean_r": 0.03, "minimum_profit_factor": 1.15},
    "quality_125": {"minimum_mean_r": 0.08, "minimum_profit_factor": 1.25},
}
META_FEATURES = (
    "BASE_SCORE",
    "IS_ANCHOR",
    "IS_FAST",
    "IS_MEDIUM",
    "IS_PULLBACK",
    "M1_RSI",
    "VOLA_RATIO",
    "BIAS_20",
    "ROC_5",
    "MACD_ATR",
    "BB_WIDTH",
    "REGIME_TREND",
    "HOUR_SIN",
    "HOUR_COS",
    "DAY_OF_WEEK",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generation 10 OOF meta-filtered CASH portfolio."
    )
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def _cutoff_time(frame: pd.DataFrame, cutoff_index: int) -> pd.Timestamp:
    if cutoff_index < len(frame):
        return frame["TIME_DT"].iloc[cutoff_index]
    return frame["TIME_DT"].iloc[-1] + pd.Timedelta(minutes=1)


def train_expert_pool(
    events: pd.DataFrame, history: pd.DataFrame, cutoff_index: int
) -> dict[str, dict]:
    cutoff_time = _cutoff_time(history, cutoff_index)
    pool = {}
    for window, days in WINDOWS.items():
        start_index = 0
        if days is not None:
            start_index = _cutoff_index(history, cutoff_time - timedelta(days=days))
        print(
            f"  pool={window} start={start_index:,} cutoff={cutoff_index:,}",
            flush=True,
        )
        experts = train_m5_experts(
            events,
            cutoff_index,
            start_index=start_index,
            skip_incomplete=True,
        )
        for name in EXPERT_NAMES:
            if name in experts:
                pool[f"{window}_{name}"] = experts[name]
            elif window == "medium":
                pool[f"{window}_{name}"] = pool[f"slow_{name}"]
            elif window == "fast":
                pool[f"{window}_{name}"] = pool[f"medium_{name}"]
            else:
                raise RuntimeError(f"Slow expert {name} is incomplete")
    return pool


def predict_expert_pool(
    pool: dict[str, dict], events: pd.DataFrame, frame: pd.DataFrame
) -> dict[str, np.ndarray]:
    output = {}
    for window in WINDOWS:
        experts = {
            name: pool[f"{window}_{name}"]
            for name in EXPERT_NAMES
        }
        predictions = predict_expected_r_to_m1(experts, events, frame)
        output.update(
            {f"{window}_{name}": values for name, values in predictions.items()}
        )
    return output


def base_arm_scores(
    frame: pd.DataFrame,
    pool_predictions: dict[str, np.ndarray],
    anchor_probability: np.ndarray,
) -> dict[str, np.ndarray]:
    scores = {name: values.copy() for name, values in pool_predictions.items()}
    anchor = np.full(len(frame), np.nan, dtype=np.float32)
    eligible = _anchor_eligible(frame, anchor_probability)
    anchor[eligible] = anchor_probability[eligible]
    scores["anchor_long"] = anchor
    if set(scores) != set(ARM_DIRECTIONS):
        raise RuntimeError("Generation 10 arm set is incomplete")
    return scores


def _arm_flags(arm: str) -> dict[str, float]:
    return {
        "IS_ANCHOR": float(arm == "anchor_long"),
        "IS_FAST": float(arm.startswith("fast_")),
        "IS_MEDIUM": float(arm.startswith("medium_")),
        "IS_PULLBACK": float("pullback" in arm),
    }


def meta_feature_frame(
    frame: pd.DataFrame,
    indices: np.ndarray,
    arm: str,
    base_score: np.ndarray,
) -> pd.DataFrame:
    subset = frame.iloc[indices]
    result = pd.DataFrame(index=np.arange(len(indices)))
    result["BASE_SCORE"] = np.asarray(base_score, dtype=np.float32)
    for name, value in _arm_flags(arm).items():
        result[name] = value
    for name in (
        "M1_RSI",
        "VOLA_RATIO",
        "BIAS_20",
        "ROC_5",
        "BB_WIDTH",
        "HOUR_SIN",
        "HOUR_COS",
        "DAY_OF_WEEK",
    ):
        result[name] = subset[name].to_numpy(dtype=np.float32)
    result["MACD_ATR"] = (
        subset["MACD_HIST"].to_numpy(dtype=np.float32)
        / np.maximum(subset["ATR"].to_numpy(dtype=np.float32), 1e-6)
    )
    result["REGIME_TREND"] = subset[
        ["M15_TREND", "H1_TREND", "H4_TREND"]
    ].to_numpy(dtype=np.float32).mean(axis=1)
    return result[list(META_FEATURES)]


def build_meta_rows(
    frame: pd.DataFrame, arm_scores: dict[str, np.ndarray]
) -> pd.DataFrame:
    records = []
    for arm, score in arm_scores.items():
        valid = np.isfinite(score)
        indices = np.flatnonzero(valid)
        if len(indices) == 0:
            continue
        direction = ARM_DIRECTIONS[arm]
        reward_column = "LONG_REWARD" if direction == 1 else "SHORT_REWARD"
        features = meta_feature_frame(frame, indices, arm, score[indices])
        features["ARM"] = arm
        features["DIRECTION"] = direction
        features["ENTRY_TIME"] = frame.iloc[indices]["TIME_DT"].to_numpy()
        features["M1_INDEX"] = frame.iloc[indices]["GLOBAL_INDEX"].to_numpy(
            dtype=np.int64
        )
        features["REWARD"] = frame.iloc[indices][reward_column].to_numpy(
            dtype=np.float32
        )
        records.append(features)
    if not records:
        raise RuntimeError("No OOF meta rows were generated")
    return pd.concat(records, ignore_index=True).sort_values("ENTRY_TIME")


def build_oof_meta_rows(
    history: pd.DataFrame,
    events: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, dict[int, tuple[dict[str, dict], xgb.XGBClassifier]]]:
    rows = []
    cache = {}
    history_end = history["TIME_DT"].iloc[-1]
    for year in range(OOF_START_YEAR, history_end.year + 1):
        start = datetime(year, 1, 1)
        end = min(datetime(year + 1, 1, 1), history_end + timedelta(minutes=1))
        validation = history[
            (history["TIME_DT"] >= start) & (history["TIME_DT"] < end)
        ].copy().reset_index(drop=True)
        if validation.empty:
            continue
        cutoff_index = _cutoff_index(history, start)
        print(f"OOF {year}: rows={len(validation):,}", flush=True)
        pool = train_expert_pool(events, history, cutoff_index)
        anchor = train_anchor(history, features, cutoff_index)
        predictions = predict_expert_pool(pool, events, validation)
        anchor_probability = _anchor_probabilities(anchor, validation, features)
        scores = base_arm_scores(validation, predictions, anchor_probability)
        rows.append(build_meta_rows(validation, scores))
        if year in {2018, 2021, 2023, 2025}:
            cache[year] = (pool, anchor)
    if not rows:
        raise RuntimeError("No annual OOF folds were available")
    return pd.concat(rows, ignore_index=True).sort_values("ENTRY_TIME"), cache


def _meta_sample_weight(frame: pd.DataFrame, win: np.ndarray) -> np.ndarray:
    positives = max(int(win.sum()), 1)
    negatives = max(len(win) - positives, 1)
    balance = np.where(
        win == 1,
        len(win) / (2.0 * positives),
        len(win) / (2.0 * negatives),
    )
    latest = frame["ENTRY_TIME"].iloc[-1]
    age_days = (
        (latest - frame["ENTRY_TIME"]).dt.total_seconds().to_numpy(dtype=np.float64)
        / 86_400.0
    )
    recency = 0.15 + 0.85 * np.exp(
        -math.log(2.0) * age_days / META_PROFILE["recency_half_life_days"]
    )
    weight = balance * recency
    if not np.isfinite(weight).all() or np.any(weight <= 0.0):
        raise RuntimeError("Meta sample weights must be finite and positive")
    return weight.astype(np.float32)


def _new_meta_classifier(estimators: int | None = None) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        device="cpu",
        n_estimators=estimators or META_PROFILE["n_estimators"],
        learning_rate=META_PROFILE["learning_rate"],
        max_depth=META_PROFILE["max_depth"],
        min_child_weight=META_PROFILE["min_child_weight"],
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=max(1, (os.cpu_count() or 2) - 1),
        verbosity=0,
    )


def _new_meta_regressor(objective: str, estimators: int | None = None) -> xgb.XGBRegressor:
    parameters = {
        "objective": objective,
        "tree_method": "hist",
        "device": "cpu",
        "n_estimators": estimators or META_PROFILE["n_estimators"],
        "learning_rate": META_PROFILE["learning_rate"],
        "max_depth": META_PROFILE["max_depth"],
        "min_child_weight": META_PROFILE["min_child_weight"],
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "n_jobs": max(1, (os.cpu_count() or 2) - 1),
        "verbosity": 0,
    }
    if objective == "reg:quantileerror":
        parameters["quantile_alpha"] = META_PROFILE["quantile_alpha"]
    return xgb.XGBRegressor(**parameters)


def train_meta_models(
    oof_rows: pd.DataFrame, cutoff_index: int
) -> dict[int, dict]:
    output = {}
    for direction in (1, 2):
        subset = oof_rows[
            (oof_rows["DIRECTION"] == direction)
            & (oof_rows["M1_INDEX"] < cutoff_index - HORIZON)
            & np.isfinite(oof_rows["REWARD"])
        ].copy()
        subset = subset.sort_values("ENTRY_TIME").reset_index(drop=True)
        if len(subset) < META_MIN_ROWS:
            raise RuntimeError(
                f"Direction {direction} has only {len(subset):,} OOF meta rows"
            )
        split = int(len(subset) * (1.0 - META_CALIBRATION_RATIO))
        calibration_start = int(subset["M1_INDEX"].iloc[split])
        fit = subset[subset["M1_INDEX"] < calibration_start - HORIZON].copy()
        calibration = subset.iloc[split:].copy()
        win = (fit["REWARD"].to_numpy(dtype=np.float32) > 0.0).astype(np.int8)
        calibration_win = (
            calibration["REWARD"].to_numpy(dtype=np.float32) > 0.0
        ).astype(np.int8)
        if len(fit) < 1_000 or len(np.unique(win)) != 2:
            raise RuntimeError(f"Direction {direction} meta fit is not usable")
        weight = _meta_sample_weight(fit, win)
        classifier = _new_meta_classifier()
        mean_model = _new_meta_regressor("reg:pseudohubererror")
        quantile_model = _new_meta_regressor("reg:quantileerror")
        x_fit = fit[list(META_FEATURES)].astype(np.float32)
        classifier.fit(x_fit, win, sample_weight=weight)
        mean_model.fit(x_fit, fit["REWARD"].to_numpy(dtype=np.float32), sample_weight=weight)
        quantile_model.fit(
            x_fit,
            fit["REWARD"].to_numpy(dtype=np.float32),
            sample_weight=weight,
        )
        raw = classifier.predict_proba(
            calibration[list(META_FEATURES)].astype(np.float32)
        )[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw, calibration_win)
        calibration_features = calibration[list(META_FEATURES)].astype(np.float32)
        calibration_predictions = {
            "win_probability": calibrator.predict(raw).astype(np.float32),
            "mean_r": mean_model.predict(calibration_features).astype(np.float32),
            "q35_r": quantile_model.predict(calibration_features).astype(np.float32),
        }
        rank_knots = {}
        calibration_arms = calibration["ARM"].to_numpy()
        for arm in ARM_DIRECTIONS:
            arm_mask = calibration_arms == arm
            if int(arm_mask.sum()) < 100:
                continue
            rank_knots[arm] = {
                name: np.quantile(values[arm_mask], RANK_QUANTILES).astype(
                    np.float32
                )
                for name, values in calibration_predictions.items()
            }
        output[direction] = {
            "win": classifier,
            "mean_r": mean_model,
            "q35_r": quantile_model,
            "calibrator": calibrator,
            "rank_knots": rank_knots,
            "fit_rows": len(fit),
            "calibration_rows": len(calibration),
        }
        print(
            f"  meta direction={direction}: fit={len(fit):,} "
            f"calibrate={len(calibration):,} win={win.mean():.2%}",
            flush=True,
        )
    return output


def _rank_from_past(values: np.ndarray, knots: np.ndarray) -> np.ndarray:
    ranks = np.searchsorted(knots, values, side="right") / float(len(knots))
    return np.clip(ranks, 0.0, 1.0).astype(np.float32)


def predict_meta_outputs(
    meta_models: dict[int, dict],
    frame: pd.DataFrame,
    arm_scores: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    output = {}
    for arm, score in arm_scores.items():
        direction = ARM_DIRECTIONS[arm]
        model = meta_models[direction]
        valid = np.isfinite(score)
        indices = np.flatnonzero(valid)
        values = {
            name: np.full(len(frame), np.nan, dtype=np.float32)
            for name in (
                "win_probability",
                "mean_r",
                "q35_r",
                "win_rank",
                "mean_rank",
                "q35_rank",
            )
        }
        if len(indices):
            features = meta_feature_frame(frame, indices, arm, score[indices])
            raw = model["win"].predict_proba(features)[:, 1]
            values["win_probability"][indices] = model["calibrator"].predict(
                raw
            ).astype(np.float32)
            values["mean_r"][indices] = model["mean_r"].predict(features).astype(
                np.float32
            )
            values["q35_r"][indices] = model["q35_r"].predict(features).astype(
                np.float32
            )
            knots = model["rank_knots"].get(arm)
            if knots is not None:
                for value_name, rank_name in (
                    ("win_probability", "win_rank"),
                    ("mean_r", "mean_rank"),
                    ("q35_r", "q35_rank"),
                ):
                    values[rank_name][indices] = _rank_from_past(
                        values[value_name][indices], knots[value_name]
                    )
        output[arm] = values
    return output


def meta_output_diagnostics(
    meta_outputs: dict[str, dict[str, np.ndarray]]
) -> dict[str, dict]:
    result = {}
    for direction, label in ((1, "long"), (2, "short")):
        values = [
            meta_outputs[arm]
            for arm, arm_direction in ARM_DIRECTIONS.items()
            if arm_direction == direction
        ]
        probability = np.concatenate(
            [item["win_probability"][np.isfinite(item["win_probability"])] for item in values]
        )
        mean_r = np.concatenate(
            [item["mean_r"][np.isfinite(item["mean_r"])] for item in values]
        )
        q35_r = np.concatenate(
            [item["q35_r"][np.isfinite(item["q35_r"])] for item in values]
        )
        result[label] = {
            "rows": int(len(probability)),
            "win_probability_q50_q90_q99": np.quantile(
                probability, (0.50, 0.90, 0.99)
            ).round(4).tolist(),
            "mean_r_q50_q90_q99": np.quantile(
                mean_r, (0.50, 0.90, 0.99)
            ).round(4).tolist(),
            "q35_r_q50_q90_q99": np.quantile(
                q35_r, (0.50, 0.90, 0.99)
            ).round(4).tolist(),
        }
    return result


def _candidate_arm_scores(
    frame: pd.DataFrame,
    meta_outputs: dict[str, dict[str, np.ndarray]],
    candidate: dict,
) -> dict[str, np.ndarray]:
    allowed = session_mask(frame, "controlled_expanded")
    output = {}
    for arm, values in meta_outputs.items():
        probability = values["win_probability"]
        mean_r = values["mean_r"]
        q35_r = values["q35_r"]
        win_rank = values["win_rank"]
        mean_rank = values["mean_rank"]
        q35_rank = values["q35_rank"]
        joint_rank = 0.45 * win_rank + 0.35 * mean_rank + 0.20 * q35_rank
        valid = (
            allowed
            & np.isfinite(probability)
            & np.isfinite(mean_r)
            & np.isfinite(q35_r)
            & np.isfinite(joint_rank)
            & (joint_rank >= candidate["minimum_joint_rank"])
            & (mean_r >= candidate["minimum_mean_r"])
            & (q35_r >= candidate["minimum_q35_r"])
        )
        score = np.full(len(frame), np.nan, dtype=np.float32)
        score[valid] = joint_rank[valid]
        output[arm] = score
    return output


def rolling_cash_allocator_signals(
    frame: pd.DataFrame,
    meta_outputs: dict[str, dict[str, np.ndarray]],
    candidate: dict,
) -> tuple[np.ndarray, dict]:
    scores = _candidate_arm_scores(frame, meta_outputs, candidate)
    long_reward = frame["LONG_REWARD"].to_numpy(dtype=np.float32)
    short_reward = frame["SHORT_REWARD"].to_numpy(dtype=np.float32)
    rewards = {1: long_reward, 2: short_reward}
    dates = frame["TIME_DT"].dt.date.to_numpy()
    row_count = len(frame)
    indices = np.arange(row_count)
    output = np.zeros((row_count, 3), dtype=np.float32)
    output_score = np.full((row_count, 3), -np.inf, dtype=np.float32)
    champion = {1: None, 2: None}
    pending = {1: None, 2: None}
    pending_count = {1: 0, 2: 0}
    quality = QUALITY_PROFILES[candidate["quality_profile"]]
    trace = {
        "blocks": 0,
        "cash_blocks": {"long": 0, "short": 0},
        "champion_blocks": {arm: 0 for arm in ARM_DIRECTIONS},
        "switches": {"long": 0, "short": 0},
        "emitted_long": 0,
        "emitted_short": 0,
    }

    for block_start in range(0, row_count, ALLOCATOR_CONFIG["block_rows"]):
        block_end = min(row_count, block_start + ALLOCATOR_CONFIG["block_rows"])
        history_end = block_start - ALLOCATOR_CONFIG["maturity_rows"]
        history_start = max(0, history_end - ALLOCATOR_CONFIG["window_rows"])
        trace["blocks"] += 1
        if history_end - history_start < ALLOCATOR_CONFIG["min_rows"]:
            trace["cash_blocks"]["long"] += 1
            trace["cash_blocks"]["short"] += 1
            continue
        split = history_start + int((history_end - history_start) * 0.67)

        for direction, label in ((1, "long"), (2, "short")):
            arm_metrics = {}
            for arm, arm_direction in ARM_DIRECTIONS.items():
                if arm_direction != direction:
                    continue
                arm_score = scores[arm]
                threshold = _top_k_threshold(
                    arm_score,
                    dates,
                    (indices >= history_start) & (indices < split),
                    candidate["top_k_per_day"],
                )
                if threshold is None:
                    continue
                metrics = _realized_metrics(
                    rewards[direction],
                    arm_score,
                    (indices >= split) & (indices < history_end),
                    threshold,
                    candidate["minimum_joint_rank"],
                )
                if (
                    metrics is not None
                    and metrics["trades"] >= ALLOCATOR_CONFIG["champion_min_trades"]
                    and metrics["mean_r"] >= quality["minimum_mean_r"]
                    and metrics["profit_factor"] >= quality["minimum_profit_factor"]
                ):
                    arm_metrics[arm] = metrics["score"]

            best = max(arm_metrics, key=arm_metrics.get) if arm_metrics else None
            current = champion[direction]
            if current is None:
                if best is not None:
                    champion[direction] = best
                    trace["switches"][label] += 1
            elif current not in arm_metrics:
                champion[direction] = None
                pending[direction] = None
                pending_count[direction] = 0
            elif best != current and arm_metrics[best] >= (
                arm_metrics[current] + ALLOCATOR_CONFIG["switch_margin"]
            ):
                if pending[direction] == best:
                    pending_count[direction] += 1
                else:
                    pending[direction] = best
                    pending_count[direction] = 1
                if pending_count[direction] >= ALLOCATOR_CONFIG["confirm_blocks"]:
                    champion[direction] = best
                    pending[direction] = None
                    pending_count[direction] = 0
                    trace["switches"][label] += 1
            else:
                pending[direction] = None
                pending_count[direction] = 0

            selected = champion[direction]
            if selected is None:
                trace["cash_blocks"][label] += 1
                continue
            arm_score = scores[selected]
            threshold = _top_k_threshold(
                arm_score,
                dates,
                (indices >= history_start) & (indices < history_end),
                candidate["top_k_per_day"],
            )
            if threshold is None:
                trace["cash_blocks"][label] += 1
                continue
            block_score = arm_score[block_start:block_end]
            selected_rows = np.isfinite(block_score) & (block_score >= threshold)
            output[block_start:block_end, direction][selected_rows] = 1.0
            output_score[block_start:block_end, direction][selected_rows] = block_score[
                selected_rows
            ]
            emitted = int(selected_rows.sum())
            trace[f"emitted_{label}"] += emitted
            trace["champion_blocks"][selected] += 1

    conflicts = (output[:, 1] > 0.0) & (output[:, 2] > 0.0)
    prefer_long = output_score[:, 1] >= output_score[:, 2]
    output[conflicts & prefer_long, 2] = 0.0
    output[conflicts & ~prefer_long, 1] = 0.0
    output[:, 0] = 1.0 - np.maximum(output[:, 1], output[:, 2])
    return output, trace


def candidate_stats(
    frame: pd.DataFrame,
    meta_outputs: dict[str, dict[str, np.ndarray]],
    candidate: dict,
    cost: float = EXTRA_COST_POINTS,
) -> tuple[dict, dict]:
    probabilities, trace = rolling_cash_allocator_signals(
        frame, meta_outputs, candidate
    )
    # ponytail: keep the proven common exit until this signal gate passes;
    # add per-arm exits only after they can be evaluated without confounding.
    return evaluate_frame(_execution_params(cost), frame, probabilities), trace


def _profit_factor(stats: dict) -> float:
    value = stats["profit_factor"]
    return 3.0 if value is None else float(value)


def _serialize_calibrator(calibrator: IsotonicRegression) -> dict:
    return {
        "x": calibrator.X_thresholds_.tolist(),
        "y": calibrator.y_thresholds_.tolist(),
    }


def save_final_models(
    pool: dict[str, dict], meta_models: dict[int, dict]
) -> dict:
    for arm, expert in pool.items():
        expert["model"].save_model(BASE_MODEL_FILES[arm])
    for direction, label in ((1, "long"), (2, "short")):
        models = meta_models[direction]
        for kind in ("win", "mean_r", "q35_r"):
            models[kind].save_model(META_MODEL_FILES[(label, kind)])
    return {
        "base_models": {
            arm: {
                "file": BASE_MODEL_FILES[arm].name,
                "payoff_r": expert["payoff"].tolist(),
                "isotonic": [
                    _serialize_calibrator(calibrator)
                    for calibrator in expert["calibrators"]
                ],
            }
            for arm, expert in pool.items()
        },
        "meta_models": {
            label: {
                "win": META_MODEL_FILES[(label, "win")].name,
                "mean_r": META_MODEL_FILES[(label, "mean_r")].name,
                "q35_r": META_MODEL_FILES[(label, "q35_r")].name,
                "win_isotonic": _serialize_calibrator(meta_models[direction]["calibrator"]),
                "rank_quantiles": RANK_QUANTILES.tolist(),
                "rank_knots": {
                    arm: {
                        name: values.tolist()
                        for name, values in metrics.items()
                    }
                    for arm, metrics in meta_models[direction]["rank_knots"].items()
                },
            }
            for direction, label in ((1, "long"), (2, "short"))
        },
    }


def markdown_report(report: dict) -> str:
    lines = [
        "# GOLD generation 10 adaptive portfolio",
        "",
        "OOF meta-filtered slow/medium/fast experts with a CASH allocator.",
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
    parameters = report["selected"]["params"]
    lines.extend(
        [
            "",
            f"Qualified selection candidates: `{report['qualified_count']}`",
            f"Selected: `{json.dumps(parameters, ensure_ascii=False)}`",
            f"Current recent: `{json.dumps(report['benchmarks']['current'])}`",
            f"Promotion gate: `{'PASS' if report['promotion_pass'] else 'FAIL'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def self_check() -> None:
    rng = np.random.default_rng(10)
    ranks = _rank_from_past(
        np.array([-1.0, 1.5, 4.0], dtype=np.float32),
        np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
    )
    assert np.allclose(ranks, (0.0, 0.5, 1.0))
    rows = 40_000
    time = pd.date_range("2026-01-01", periods=rows, freq="min")
    frame = pd.DataFrame(
        {
            "TIME_DT": time,
            "LONG_REWARD": rng.normal(0.35, 0.45, rows),
            "SHORT_REWARD": rng.normal(-0.20, 0.45, rows),
            "M1_RSI": np.full(rows, 55.0),
        }
    )
    outputs = {}
    for arm, direction in ARM_DIRECTIONS.items():
        available = np.arange(rows) % 180 == (abs(hash(arm)) % 180)
        win_probability = np.full(rows, np.nan, dtype=np.float32)
        mean_r = np.full(rows, np.nan, dtype=np.float32)
        q35_r = np.full(rows, np.nan, dtype=np.float32)
        win_probability[available] = 0.70 if direction == 1 else 0.55
        mean_r[available] = 0.25 if direction == 1 else -0.10
        q35_r[available] = 0.05 if direction == 1 else -0.20
        outputs[arm] = {
            "win_probability": win_probability,
            "mean_r": mean_r,
            "q35_r": q35_r,
            "win_rank": np.where(available, 0.90, np.nan).astype(np.float32),
            "mean_rank": np.where(
                available, 0.90 if direction == 1 else 0.20, np.nan
            ).astype(np.float32),
            "q35_rank": np.where(
                available, 0.90 if direction == 1 else 0.20, np.nan
            ).astype(np.float32),
        }
    candidate = {
        "minimum_joint_rank": 0.80,
        "minimum_mean_r": 0.0,
        "minimum_q35_r": 0.0,
        "quality_profile": "quality_115",
        "top_k_per_day": 1,
    }
    probabilities, trace = rolling_cash_allocator_signals(frame, outputs, candidate)
    assert trace["cash_blocks"]["short"] > 0
    assert probabilities[:, 2].sum() == 0
    assert probabilities[:, 1].sum() > 0

    toy = pd.DataFrame(rng.normal(size=(300, len(META_FEATURES))), columns=META_FEATURES)
    labels = np.tile(np.array([0, 1], dtype=np.int8), 150)
    classifier = _new_meta_classifier(estimators=5)
    classifier.fit(toy, labels)
    assert np.isfinite(classifier.predict_proba(toy)).all()
    quantile = _new_meta_regressor("reg:quantileerror", estimators=5)
    quantile.fit(toy, rng.normal(size=len(toy)))
    assert np.isfinite(quantile.predict(toy)).all()
    print("generation10_self_check_ok")


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0

    history, features = prepare_barrier_data()
    history = add_generation8_targets(history)
    history["GLOBAL_INDEX"] = np.arange(len(history), dtype=np.int64)
    historical_m5 = _load_mt5_export(HISTORICAL_M5)
    events = build_m5_event_table(historical_m5, history)
    print(
        f"History rows={len(history):,} {history['TIME_DT'].iloc[0]} -> "
        f"{history['TIME_DT'].iloc[-1]} events={event_counts(events)}",
        flush=True,
    )

    oof_rows, model_cache = build_oof_meta_rows(history, events, features)
    print(
        f"OOF meta rows={len(oof_rows):,} "
        f"{oof_rows['ENTRY_TIME'].iloc[0]} -> {oof_rows['ENTRY_TIME'].iloc[-1]}",
        flush=True,
    )
    candidates = [
        {
            "generation": "10_adaptive_portfolio",
            "minimum_joint_rank": joint_rank,
            "minimum_mean_r": mean_r,
            "minimum_q35_r": q35_r,
            "quality_profile": quality,
            "top_k_per_day": 1,
        }
        for joint_rank, mean_r, q35_r, quality in product(
            (0.75, 0.85, 0.90),
            (-0.25, 0.0),
            (-1.10,),
            QUALITY_PROFILES,
        )
    ]
    fold_results = {index: {} for index in range(len(candidates))}
    fold_traces = {index: {} for index in range(len(candidates))}
    anchor_folds = {}
    meta_diagnostics = {}
    for fold_name, fold_start, fold_end in SELECTION_FOLDS:
        cutoff_index = _cutoff_index(history, fold_start)
        validation = history[
            (history["TIME_DT"] >= fold_start) & (history["TIME_DT"] < fold_end)
        ].copy().reset_index(drop=True)
        year = fold_start.year
        pool, anchor = model_cache[year]
        meta_models = train_meta_models(oof_rows, cutoff_index)
        pool_predictions = predict_expert_pool(pool, events, validation)
        anchor_probability = _anchor_probabilities(anchor, validation, features)
        scores = base_arm_scores(validation, pool_predictions, anchor_probability)
        meta_outputs = predict_meta_outputs(meta_models, validation, scores)
        diagnostics = meta_output_diagnostics(meta_outputs)
        meta_diagnostics[fold_name] = diagnostics
        print(f"  meta outputs {fold_name}: {diagnostics}", flush=True)
        anchor_folds[fold_name] = compact_stats(
            anchor_stats(validation, anchor_probability)
        )
        for index, candidate in enumerate(candidates):
            stats, trace = candidate_stats(validation, meta_outputs, candidate)
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
    holdout_pool, holdout_anchor = model_cache[2025]
    holdout_meta = train_meta_models(oof_rows, holdout_cutoff)
    holdout_pool_predictions = predict_expert_pool(holdout_pool, events, holdout)
    holdout_anchor_probability = _anchor_probabilities(
        holdout_anchor, holdout, features
    )
    holdout_scores = base_arm_scores(
        holdout, holdout_pool_predictions, holdout_anchor_probability
    )
    holdout_outputs = predict_meta_outputs(holdout_meta, holdout, holdout_scores)
    meta_diagnostics["2025_2026_05_holdout"] = meta_output_diagnostics(
        holdout_outputs
    )
    holdout_stats, holdout_trace = candidate_stats(
        holdout, holdout_outputs, selected
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
    recent["GLOBAL_INDEX"] = np.arange(len(recent), dtype=np.int64)
    recent_events = build_m5_event_table(recent_m5, recent)

    final_pool = train_expert_pool(events, history, len(history))
    final_meta = train_meta_models(oof_rows, len(history))
    recent_pool_predictions = predict_expert_pool(final_pool, recent_events, recent)
    current_model = load_model(CURRENT_MODEL_FILE)
    current_probability = _anchor_probabilities(current_model, recent, features)
    recent_scores = base_arm_scores(
        recent, recent_pool_predictions, current_probability
    )
    recent_outputs = predict_meta_outputs(final_meta, recent, recent_scores)
    meta_diagnostics["2026_recent"] = meta_output_diagnostics(recent_outputs)
    recent_stats, recent_trace = candidate_stats(recent, recent_outputs, selected)
    recent_cost_stats, _ = candidate_stats(
        recent, recent_outputs, selected, cost=10.0
    )
    current_stats = benchmark_current(recent, features)

    compact_holdout = compact_stats(holdout_stats)
    compact_recent = compact_stats(recent_stats)
    compact_cost = compact_stats(recent_cost_stats)
    compact_current = compact_stats(current_stats)
    promotion_pass = bool(
        qualified
        and fold_pass(holdout_stats, 100)
        and fold_pass(recent_stats, 30)
        and compact_recent["trades"] > compact_current["trades"]
        and compact_recent["win_rate"] > compact_current["win_rate"]
        and compact_recent["pnl"] > compact_current["pnl"]
        and _profit_factor(compact_recent) > _profit_factor(compact_current)
        and compact_cost["pnl"] > 0.0
        and _profit_factor(compact_cost) >= 1.05
    )

    model_files = save_final_models(final_pool, final_meta)
    selected_params = {
        key: value
        for key, value in selected.items()
        if key not in {"candidate_index", "qualified", "score", "folds"}
    }
    config = {
        **selected_params,
        "status": "promotion_pass" if promotion_pass else "research_only",
        "qualified_selection": bool(qualified),
        "windows_days": WINDOWS,
        "meta_features": list(META_FEATURES),
        "meta_profile": META_PROFILE,
        "allocator_config": ALLOCATOR_CONFIG,
        "model_files": model_files,
        "promotion_pass": promotion_pass,
    }
    CONFIG_FILE.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    selected_folds = {
        **selected["folds"],
        "2025_2026_05_holdout": compact_holdout,
        "2026_recent": compact_recent,
        "2026_recent_cost_10": compact_cost,
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "promotion_pass" if promotion_pass else "research_only",
        "data": {
            "history_rows": len(history),
            "history_start": history["TIME_DT"].iloc[0].isoformat(),
            "history_end": history["TIME_DT"].iloc[-1].isoformat(),
            "oof_meta_rows": len(oof_rows),
            "oof_start": oof_rows["ENTRY_TIME"].iloc[0].isoformat(),
            "oof_end": oof_rows["ENTRY_TIME"].iloc[-1].isoformat(),
            "historical_events": event_counts(events),
            "recent_rows": len(recent),
            "recent_events": event_counts(recent_events),
        },
        "qualified_count": len(qualified),
        "selected": {"params": selected_params, "folds": selected_folds},
        "anchor_folds": anchor_folds,
        "meta_output_diagnostics": meta_diagnostics,
        "allocator_trace": {
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
