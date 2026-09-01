from __future__ import annotations

import argparse
import gc
import json
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
from gold_expected_r_walk_forward import (
    EXTRA_COST_POINTS,
    HISTORICAL_HOLDOUT_START,
    HORIZON,
    fold_pass,
    make_params,
    session_mask,
)
from gold_generation11_execution_aligned import _profit_factor, add_targets
from gold_generation12_executable_events import (
    MODEL_PROFILE,
    _new_classifier,
    _new_regressor,
    _sample_weight,
    _serialize_calibrator,
    executable_events_by_expert,
    sequential_event_indices,
)
from gold_recent_walk_forward import DEFAULT_TERMINAL, build_feature_frame
from gold_regime_experts_iterative import EXPERT_NAMES, training_frame
from gold_regime_experts_walk_forward import (
    RECENT_START,
    SELECTION_FOLDS,
    benchmark_current,
    evaluate_frame,
    route_arrays,
)
from gold_short_rule_research import compact_stats


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_JSON = PROJECT_ROOT / "gold_generation14_precision_frequency.json"
REPORT_MD = PROJECT_ROOT / "gold_generation14_precision_frequency.md"
CONFIG_FILE = PROJECT_ROOT / "gold_generation14_candidate.json"
GEN12_REPORT = PROJECT_ROOT / "gold_generation12_executable_events.json"
MODEL_FILES = {
    (name, kind): PROJECT_ROOT / f"gold_generation14_{name}_{kind}_xgb.json"
    for name in EXPERT_NAMES
    for kind in ("win", "mean_r", "meta")
}
EXPERT_CODES = {name: index + 1 for index, name in enumerate(EXPERT_NAMES)}
MODEL_SPLITS = {
    "base_fit_end": 0.62,
    "base_calibration_end": 0.78,
    "meta_fit_end": 0.94,
}
META_PROFILE = {
    "n_estimators": 140,
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_child_weight": 12,
}
ALLOCATOR_CONFIG = {
    "maturity_rows": HORIZON,
    "window_rows": 90_000,
    "min_rows": 30_000,
    "block_rows": 10_080,
    "calibration_min_trades": 10,
    "quality_min_trades": 5,
    "full_history_min_trades": 15,
    "max_drawdown_r": 6.0,
}
THRESHOLD_QUANTILES = (0.0, 0.15, 0.30, 0.45, 0.60, 0.72, 0.82, 0.90, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train generation 14 precision-constrained frequency models."
    )
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


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


def session_codes(frame: pd.DataFrame) -> np.ndarray:
    hours = frame["TIME_DT"].dt.hour.to_numpy(dtype=np.int8)
    output = np.full(len(frame), 3, dtype=np.int8)
    output[hours <= 4] = 0
    output[(hours >= 7) & (hours <= 13)] = 1
    output[(hours >= 14) & (hours <= 23)] = 2
    return output


def volatility_codes(frame: pd.DataFrame) -> np.ndarray:
    ratio = frame["VOLA_RATIO"].to_numpy(dtype=np.float32)
    output = np.ones(len(frame), dtype=np.int8)
    output[ratio < 0.85] = 0
    output[ratio > 1.20] = 2
    return output


def _meta_matrix(
    frame: pd.DataFrame,
    features: list[str],
    p_win: np.ndarray,
    mean_r: np.ndarray,
    expected_r: np.ndarray,
) -> np.ndarray:
    if not (len(frame) == len(p_win) == len(mean_r) == len(expected_r)):
        raise ValueError("Meta feature inputs must have equal lengths")
    return np.column_stack(
        (
            frame[features].to_numpy(dtype=np.float32),
            p_win.astype(np.float32),
            mean_r.astype(np.float32),
            expected_r.astype(np.float32),
            session_codes(frame).astype(np.float32),
            volatility_codes(frame).astype(np.float32),
        )
    ).astype(np.float32, copy=False)


def _base_components(
    model: dict, frame: pd.DataFrame, features: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = frame[features].astype(np.float32)
    p_win = model["calibrator"].predict(
        model["win"].predict_proba(x)[:, 1]
    ).astype(np.float32)
    mean_r = model["mean_r"].predict(x).astype(np.float32)
    probability_r = (
        p_win * model["average_win_r"]
        + (1.0 - p_win) * model["average_loss_r"]
    )
    expected_r = 0.5 * (probability_r + mean_r)
    return p_win, mean_r, expected_r.astype(np.float32)


def train_two_stage_experts(
    frame: pd.DataFrame, features: list[str]
) -> dict[str, dict]:
    event_indices = executable_events_by_expert(frame, features)
    models = {}
    for name, indices in event_indices.items():
        direction = "LONG" if name.startswith("long_") else "SHORT"
        reward_column = f"{direction}_REWARD"
        events = frame.iloc[indices].copy().sort_values("TIME_DT")
        if len(events) < 2_000:
            raise RuntimeError(f"{name} has only {len(events):,} executable events")
        count = len(events)
        base_fit_end = int(count * MODEL_SPLITS["base_fit_end"])
        base_calibration_end = int(
            count * MODEL_SPLITS["base_calibration_end"]
        )
        meta_fit_end = int(count * MODEL_SPLITS["meta_fit_end"])
        base_fit = events.iloc[:base_fit_end]
        base_calibration = events.iloc[base_fit_end:base_calibration_end]
        meta_fit = events.iloc[base_calibration_end:meta_fit_end]
        meta_calibration = events.iloc[meta_fit_end:]

        fit_reward = base_fit[reward_column].to_numpy(dtype=np.float32)
        calibration_reward = base_calibration[reward_column].to_numpy(
            dtype=np.float32
        )
        fit_win = (fit_reward > 0.0).astype(np.int8)
        calibration_win = (calibration_reward > 0.0).astype(np.int8)
        if np.unique(fit_win).size != 2 or np.unique(calibration_win).size != 2:
            raise RuntimeError(f"{name} base split lacks a win/loss class")

        classifier = _new_classifier()
        mean_model = _new_regressor()
        weight = _sample_weight(base_fit["TIME_DT"])
        classifier.fit(
            base_fit[features].astype(np.float32),
            fit_win,
            sample_weight=weight,
        )
        mean_model.fit(
            base_fit[features].astype(np.float32),
            fit_reward,
            sample_weight=weight,
        )
        base_raw = classifier.predict_proba(
            base_calibration[features].astype(np.float32)
        )[:, 1]
        calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        )
        calibrator.fit(base_raw, calibration_win)
        model = {
            "win": classifier,
            "mean_r": mean_model,
            "calibrator": calibrator,
            "average_win_r": float(
                calibration_reward[calibration_reward > 0.0].mean()
            ),
            "average_loss_r": float(
                calibration_reward[calibration_reward <= 0.0].mean()
            ),
        }

        meta_fit_p, meta_fit_mean, meta_fit_expected = _base_components(
            model, meta_fit, features
        )
        meta_cal_p, meta_cal_mean, meta_cal_expected = _base_components(
            model, meta_calibration, features
        )
        meta_fit_win = (
            meta_fit[reward_column].to_numpy(dtype=np.float32) > 0.0
        ).astype(np.int8)
        meta_calibration_win = (
            meta_calibration[reward_column].to_numpy(dtype=np.float32) > 0.0
        ).astype(np.int8)
        if (
            np.unique(meta_fit_win).size != 2
            or np.unique(meta_calibration_win).size != 2
        ):
            raise RuntimeError(f"{name} meta split lacks a win/loss class")
        meta = _new_meta_classifier()
        meta.fit(
            _meta_matrix(
                meta_fit,
                features,
                meta_fit_p,
                meta_fit_mean,
                meta_fit_expected,
            ),
            meta_fit_win,
            sample_weight=_sample_weight(meta_fit["TIME_DT"]),
        )
        meta_raw = meta.predict_proba(
            _meta_matrix(
                meta_calibration,
                features,
                meta_cal_p,
                meta_cal_mean,
                meta_cal_expected,
            )
        )[:, 1]
        meta_calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        )
        meta_calibrator.fit(meta_raw, meta_calibration_win)
        model.update(
            {
                "meta": meta,
                "meta_calibrator": meta_calibrator,
                "events": count,
                "base_fit_events": len(base_fit),
                "base_calibration_events": len(base_calibration),
                "meta_fit_events": len(meta_fit),
                "meta_calibration_events": len(meta_calibration),
            }
        )
        models[name] = model
        print(
            f"  {name}: events={count:,} base={len(base_fit):,}/"
            f"{len(base_calibration):,} meta={len(meta_fit):,}/"
            f"{len(meta_calibration):,}",
            flush=True,
        )
    return models


def predict_two_stage_scores(
    models: dict[str, dict], frame: pd.DataFrame, features: list[str]
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray], dict]:
    _, masks = route_arrays(frame, features)
    base_scores = {
        1: np.full(len(frame), np.nan, dtype=np.float32),
        2: np.full(len(frame), np.nan, dtype=np.float32),
    }
    precision_scores = {
        1: np.full(len(frame), np.nan, dtype=np.float32),
        2: np.full(len(frame), np.nan, dtype=np.float32),
    }
    expert_codes = {
        1: np.zeros(len(frame), dtype=np.int8),
        2: np.zeros(len(frame), dtype=np.int8),
    }
    diagnostics = {}
    for name, model in models.items():
        indices = np.flatnonzero(masks[name])
        if len(indices) == 0:
            continue
        routed = frame.iloc[indices]
        p_win, mean_r, expected_r = _base_components(model, routed, features)
        meta_probability = model["meta_calibrator"].predict(
            model["meta"].predict_proba(
                _meta_matrix(routed, features, p_win, mean_r, expected_r)
            )[:, 1]
        ).astype(np.float32)
        direction = 1 if name.startswith("long_") else 2
        base_scores[direction][indices] = expected_r
        precision_scores[direction][indices] = meta_probability
        expert_codes[direction][indices] = EXPERT_CODES[name]
        diagnostics[name] = {
            "rows": len(indices),
            "base_p_win_q50_q90_q99": np.quantile(
                p_win, (0.50, 0.90, 0.99)
            ).round(4).tolist(),
            "base_expected_r_q50_q90_q99": np.quantile(
                expected_r, (0.50, 0.90, 0.99)
            ).round(4).tolist(),
            "meta_p_win_q50_q90_q99": np.quantile(
                meta_probability, (0.50, 0.90, 0.99)
            ).round(4).tolist(),
        }
    return base_scores, precision_scores, expert_codes, diagnostics


def _event_metrics(
    reward: np.ndarray,
    exits: np.ndarray,
    score: np.ndarray,
    eligible_indices: np.ndarray,
    threshold: float,
) -> dict | None:
    if eligible_indices.size == 0:
        return None
    valid = eligible_indices[
        np.isfinite(reward[eligible_indices])
        & np.isfinite(score[eligible_indices])
        & (exits[eligible_indices] > 0)
        & (score[eligible_indices] > 0.0)
        & (score[eligible_indices] >= threshold)
    ]
    if valid.size == 0:
        return None
    rising = valid[np.r_[True, np.diff(valid) > 1]]
    selected = []
    free_index = 0
    for index in rising:
        if index < free_index:
            continue
        selected.append(index)
        free_index = index + int(exits[index]) + 1
    if not selected:
        return None
    values = reward[np.asarray(selected, dtype=np.int64)]
    gains = float(values[values > 0.0].sum())
    losses = float(-values[values < 0.0].sum())
    profit_factor = float("inf") if losses == 0.0 else gains / losses
    equity = np.cumsum(values)
    drawdown = float(
        np.min(equity - np.maximum.accumulate(np.maximum(equity, 0.0)))
    )
    return {
        "trades": len(selected),
        "win_rate": float(np.mean(values > 0.0)),
        "profit_factor": profit_factor,
        "mean_r": float(values.mean()),
        "drawdown": drawdown,
    }


def _passes_precision_constraints(
    metrics: dict | None, candidate: dict, minimum_trades: int
) -> bool:
    return bool(
        metrics is not None
        and metrics["trades"] >= minimum_trades
        and metrics["win_rate"] >= candidate["target_win_rate"]
        and metrics["profit_factor"] >= candidate["minimum_profit_factor"]
        and metrics["mean_r"] > 0.0
        and metrics["drawdown"] >= -ALLOCATOR_CONFIG["max_drawdown_r"]
    )


def precision_constrained_threshold(
    reward: np.ndarray,
    exits: np.ndarray,
    score: np.ndarray,
    eligible_indices: np.ndarray,
    candidate: dict,
    minimum_trades: int,
) -> tuple[float, dict] | None:
    values = score[eligible_indices]
    values = values[np.isfinite(values) & (values > 0.0)]
    if len(values) < minimum_trades:
        return None
    thresholds = np.unique(np.quantile(values, THRESHOLD_QUANTILES))
    passing = []
    for threshold in thresholds:
        metrics = _event_metrics(
            reward, exits, score, eligible_indices, float(threshold)
        )
        if _passes_precision_constraints(metrics, candidate, minimum_trades):
            passing.append((float(threshold), metrics))
    if not passing:
        return None
    return max(
        passing,
        key=lambda item: (
            item[1]["trades"],
            item[1]["win_rate"],
            item[1]["profit_factor"],
        ),
    )


def _stage1_masks(
    frame: pd.DataFrame,
    base_scores: dict[int, np.ndarray],
    precision_scores: dict[int, np.ndarray],
    expert_codes: dict[int, np.ndarray],
    candidate: dict,
) -> dict[int, np.ndarray]:
    allowed = session_mask(frame, candidate["session_profile"])
    output = {}
    for direction in (1, 2):
        exits = frame[
            "LONG_EXIT_OFFSET" if direction == 1 else "SHORT_EXIT_OFFSET"
        ].to_numpy(dtype=np.int16)
        output[direction] = (
            allowed
            & np.isfinite(base_scores[direction])
            & np.isfinite(precision_scores[direction])
            & (base_scores[direction] >= candidate["minimum_expected_r"])
            & (expert_codes[direction] > 0)
            & (exits > 0)
        )
    return output


def rolling_precision_signals(
    frame: pd.DataFrame,
    base_scores: dict[int, np.ndarray],
    precision_scores: dict[int, np.ndarray],
    expert_codes: dict[int, np.ndarray],
    candidate: dict,
) -> tuple[np.ndarray, dict]:
    row_count = len(frame)
    sessions = session_codes(frame)
    volatility = volatility_codes(frame)
    stage1 = _stage1_masks(
        frame, base_scores, precision_scores, expert_codes, candidate
    )
    rewards = {
        1: frame["LONG_REWARD"].to_numpy(dtype=np.float32),
        2: frame["SHORT_REWARD"].to_numpy(dtype=np.float32),
    }
    exits = {
        1: frame["LONG_EXIT_OFFSET"].to_numpy(dtype=np.int16),
        2: frame["SHORT_EXIT_OFFSET"].to_numpy(dtype=np.int16),
    }
    output = np.zeros((row_count, 3), dtype=np.float32)
    trace = {
        "blocks": 0,
        "warmup_cash_blocks": 0,
        "groups": {"exact": 0, "expert_fallback": 0, "cash": 0},
        "stage1_rows": {"long": int(stage1[1].sum()), "short": int(stage1[2].sum())},
        "emitted_rows": {"long": 0, "short": 0},
        "thresholds": {
            "exact": {"count": 0, "sum": 0.0, "min": None, "max": None},
            "expert_fallback": {
                "count": 0,
                "sum": 0.0,
                "min": None,
                "max": None,
            },
        },
    }

    for block_start in range(0, row_count, ALLOCATOR_CONFIG["block_rows"]):
        block_end = min(row_count, block_start + ALLOCATOR_CONFIG["block_rows"])
        history_end = block_start - ALLOCATOR_CONFIG["maturity_rows"]
        history_start = max(0, history_end - ALLOCATOR_CONFIG["window_rows"])
        trace["blocks"] += 1
        if history_end - history_start < ALLOCATOR_CONFIG["min_rows"]:
            trace["warmup_cash_blocks"] += 1
            continue
        split = history_start + int((history_end - history_start) * 0.67)

        for direction, direction_name in ((1, "long"), (2, "short")):
            current = np.flatnonzero(stage1[direction][block_start:block_end])
            if current.size == 0:
                continue
            current += block_start
            groups = np.unique(
                np.column_stack(
                    (
                        expert_codes[direction][current],
                        sessions[current],
                        volatility[current],
                    )
                ),
                axis=0,
            )
            cache = {}
            for expert, session, vola in groups:
                current_group = current[
                    (expert_codes[direction][current] == expert)
                    & (sessions[current] == session)
                    & (volatility[current] == vola)
                ]
                selected_threshold = None
                selected_level = None
                for level in ("exact", "expert_fallback"):
                    key = (
                        level,
                        int(expert),
                        int(session) if level == "exact" else -1,
                        int(vola) if level == "exact" else -1,
                    )
                    if key not in cache:
                        calibration_mask = (
                            stage1[direction][history_start:split]
                            & (
                                expert_codes[direction][history_start:split]
                                == expert
                            )
                        )
                        quality_mask = (
                            stage1[direction][split:history_end]
                            & (
                                expert_codes[direction][split:history_end]
                                == expert
                            )
                        )
                        full_mask = (
                            stage1[direction][history_start:history_end]
                            & (
                                expert_codes[direction][history_start:history_end]
                                == expert
                            )
                        )
                        if level == "exact":
                            calibration_mask &= (
                                sessions[history_start:split] == session
                            ) & (volatility[history_start:split] == vola)
                            quality_mask &= (
                                sessions[split:history_end] == session
                            ) & (volatility[split:history_end] == vola)
                            full_mask &= (
                                sessions[history_start:history_end] == session
                            ) & (
                                volatility[history_start:history_end] == vola
                            )
                        calibration_indices = (
                            np.flatnonzero(calibration_mask) + history_start
                        )
                        calibrated = precision_constrained_threshold(
                            rewards[direction],
                            exits[direction],
                            precision_scores[direction],
                            calibration_indices,
                            candidate,
                            ALLOCATOR_CONFIG["calibration_min_trades"],
                        )
                        cache[key] = None
                        if calibrated is not None:
                            quality_indices = np.flatnonzero(quality_mask) + split
                            quality_metrics = _event_metrics(
                                rewards[direction],
                                exits[direction],
                                precision_scores[direction],
                                quality_indices,
                                calibrated[0],
                            )
                            if _passes_precision_constraints(
                                quality_metrics,
                                candidate,
                                ALLOCATOR_CONFIG["quality_min_trades"],
                            ):
                                full_indices = (
                                    np.flatnonzero(full_mask) + history_start
                                )
                                full_result = precision_constrained_threshold(
                                    rewards[direction],
                                    exits[direction],
                                    precision_scores[direction],
                                    full_indices,
                                    candidate,
                                    ALLOCATOR_CONFIG["full_history_min_trades"],
                                )
                                if full_result is not None:
                                    cache[key] = full_result[0]
                    if cache[key] is not None:
                        selected_threshold = cache[key]
                        selected_level = level
                        break

                if selected_threshold is None:
                    trace["groups"]["cash"] += 1
                    continue
                selected = current_group[
                    precision_scores[direction][current_group]
                    >= selected_threshold
                ]
                output[selected, direction] = 1.0
                trace["groups"][selected_level] += 1
                trace["emitted_rows"][direction_name] += len(selected)
                threshold_trace = trace["thresholds"][selected_level]
                threshold_trace["count"] += 1
                threshold_trace["sum"] += float(selected_threshold)
                threshold_trace["min"] = (
                    float(selected_threshold)
                    if threshold_trace["min"] is None
                    else min(threshold_trace["min"], float(selected_threshold))
                )
                threshold_trace["max"] = (
                    float(selected_threshold)
                    if threshold_trace["max"] is None
                    else max(threshold_trace["max"], float(selected_threshold))
                )
    output[:, 0] = 1.0 - np.maximum(output[:, 1], output[:, 2])
    for values in trace["thresholds"].values():
        values["average"] = (
            values["sum"] / values["count"] if values["count"] else None
        )
        del values["sum"]
    return output, trace


def filter_diagnostics(
    frame: pd.DataFrame,
    base_scores: dict[int, np.ndarray],
    precision_scores: dict[int, np.ndarray],
    expert_codes: dict[int, np.ndarray],
    candidate: dict,
    signals: np.ndarray,
) -> dict:
    stage1 = _stage1_masks(
        frame, base_scores, precision_scores, expert_codes, candidate
    )
    output = {}
    totals = {
        "stage1_events": 0,
        "stage1_winners": 0,
        "stage1_losers": 0,
        "stage2_events": 0,
        "stage2_winners": 0,
        "stage2_losers": 0,
    }
    for direction, name in ((1, "long"), (2, "short")):
        reward = frame[
            "LONG_REWARD" if direction == 1 else "SHORT_REWARD"
        ].to_numpy(dtype=np.float32)
        exits = frame[
            "LONG_EXIT_OFFSET" if direction == 1 else "SHORT_EXIT_OFFSET"
        ].to_numpy(dtype=np.int16)
        stage1_indices = sequential_event_indices(stage1[direction], exits)
        stage2_indices = sequential_event_indices(signals[:, direction] > 0.0, exits)
        values1 = reward[stage1_indices]
        values2 = reward[stage2_indices]
        stats = {
            "stage1_events": len(stage1_indices),
            "stage1_winners": int(np.sum(values1 > 0.0)),
            "stage1_losers": int(np.sum(values1 <= 0.0)),
            "stage2_events": len(stage2_indices),
            "stage2_winners": int(np.sum(values2 > 0.0)),
            "stage2_losers": int(np.sum(values2 <= 0.0)),
        }
        stats["estimated_winners_removed"] = max(
            0, stats["stage1_winners"] - stats["stage2_winners"]
        )
        stats["estimated_losers_removed"] = max(
            0, stats["stage1_losers"] - stats["stage2_losers"]
        )
        stats["winner_removal_rate"] = (
            stats["estimated_winners_removed"] / stats["stage1_winners"]
            if stats["stage1_winners"]
            else 0.0
        )
        stats["loser_removal_rate"] = (
            stats["estimated_losers_removed"] / stats["stage1_losers"]
            if stats["stage1_losers"]
            else 0.0
        )
        output[name] = stats
        for key in totals:
            totals[key] += stats[key]
    totals["estimated_winners_removed"] = max(
        0, totals["stage1_winners"] - totals["stage2_winners"]
    )
    totals["estimated_losers_removed"] = max(
        0, totals["stage1_losers"] - totals["stage2_losers"]
    )
    totals["winner_removal_rate"] = (
        totals["estimated_winners_removed"] / totals["stage1_winners"]
        if totals["stage1_winners"]
        else 0.0
    )
    totals["loser_removal_rate"] = (
        totals["estimated_losers_removed"] / totals["stage1_losers"]
        if totals["stage1_losers"]
        else 0.0
    )
    output["total"] = totals
    return output


def candidate_stats(
    frame: pd.DataFrame,
    base_scores: dict[int, np.ndarray],
    precision_scores: dict[int, np.ndarray],
    expert_codes: dict[int, np.ndarray],
    candidate: dict,
    cost: float = EXTRA_COST_POINTS,
) -> tuple[dict, dict, dict]:
    signals, trace = rolling_precision_signals(
        frame, base_scores, precision_scores, expert_codes, candidate
    )
    stats = evaluate_frame(make_params(candidate, cost), frame, signals)
    diagnostics = filter_diagnostics(
        frame,
        base_scores,
        precision_scores,
        expert_codes,
        candidate,
        signals,
    )
    return stats, trace, diagnostics


def _candidate_rank(folds: dict[str, dict]) -> tuple:
    values = list(folds.values())
    passing = sum(fold_pass(stats, 100) for stats in values)
    covered = all(stats["trades"] >= 100 for stats in values)
    minimum_win = min(float(stats["win_rate"]) for stats in values)
    minimum_pf = min(_profit_factor(stats) for stats in values)
    total_pnl = sum(float(stats["pnl"]) for stats in values)
    total_trades = sum(int(stats["trades"]) for stats in values)
    return passing, int(covered), minimum_win, minimum_pf, total_pnl, total_trades


def save_models(models: dict[str, dict]) -> dict:
    output = {}
    for name, model in models.items():
        for kind in ("win", "mean_r", "meta"):
            model[kind].save_model(MODEL_FILES[(name, kind)])
        output[name] = {
            "win_file": MODEL_FILES[(name, "win")].name,
            "mean_r_file": MODEL_FILES[(name, "mean_r")].name,
            "meta_file": MODEL_FILES[(name, "meta")].name,
            "base_isotonic": _serialize_calibrator(model["calibrator"]),
            "meta_isotonic": _serialize_calibrator(model["meta_calibrator"]),
            "average_win_r": model["average_win_r"],
            "average_loss_r": model["average_loss_r"],
            "events": model["events"],
        }
    return output


def _gen12_recent_benchmark() -> dict:
    if not GEN12_REPORT.exists():
        return {"trades": 51, "win_rate": 0.607843, "profit_factor": 0.92928, "pnl": -15.337937}
    report = json.loads(GEN12_REPORT.read_text(encoding="utf-8"))
    return report["selected"]["folds"]["2026_recent"]


def markdown_report(report: dict) -> str:
    lines = [
        "# GOLD generation 14 precision-constrained frequency",
        "",
        "Gen12 executable events with a leak-free loser meta-filter and past-only per-regime thresholds.",
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
    filtered = report["filter_diagnostics"]["recent"]["total"]
    lines.extend(
        [
            "",
            f"Stage1 executable opportunities: `{filtered['stage1_events']}`",
            f"Stage2 retained opportunities: `{filtered['stage2_events']}`",
            f"Estimated loser removal: `{filtered['loser_removal_rate']:.2%}`",
            f"Estimated winner removal: `{filtered['winner_removal_rate']:.2%}`",
            f"Qualified selection candidates: `{report['qualified_count']}`",
            f"Selected: `{json.dumps(report['selected']['params'], ensure_ascii=False)}`",
            f"Gen12 recent: `{json.dumps(report['benchmarks']['gen12'])}`",
            f"Current recent: `{json.dumps(report['benchmarks']['current'])}`",
            f"Promotion gate: `{'PASS' if report['promotion_pass'] else 'FAIL'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def self_check() -> None:
    rows = 200
    reward = np.full(rows, np.nan, dtype=np.float32)
    exits = np.ones(rows, dtype=np.int16)
    score = np.zeros(rows, dtype=np.float32)
    indices = np.arange(0, rows, 2, dtype=np.int64)
    score[indices] = np.linspace(1.0, 0.01, len(indices), dtype=np.float32)
    event_reward = np.full(len(indices), -1.0, dtype=np.float32)
    event_reward[:60][np.arange(60) % 10 < 7] = 1.0
    event_reward[60:][np.arange(40) % 5 == 0] = 1.0
    reward[indices] = event_reward
    candidate = {
        "target_win_rate": 0.60,
        "minimum_profit_factor": 1.05,
    }
    result = precision_constrained_threshold(
        reward, exits, score, indices, candidate, minimum_trades=10
    )
    assert result is not None
    assert result[1]["trades"] >= 45
    assert result[1]["win_rate"] >= 0.60
    assert result[1]["profit_factor"] >= 1.05

    frame = pd.DataFrame(
        {
            "TIME_DT": pd.to_datetime(
                ["2026-01-05 02:00", "2026-01-05 10:00", "2026-01-05 18:00"]
            ),
            "VOLA_RATIO": [0.7, 1.0, 1.4],
            "X": [1.0, 2.0, 3.0],
        }
    )
    assert session_codes(frame).tolist() == [0, 1, 2]
    assert volatility_codes(frame).tolist() == [0, 1, 2]
    matrix = _meta_matrix(
        frame,
        ["X"],
        np.full(3, 0.6),
        np.full(3, 0.1),
        np.full(3, 0.05),
    )
    assert matrix.shape == (3, 6)
    print("generation14_self_check_ok")


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0

    history, features = prepare_barrier_data()
    history = add_targets(history)
    print(
        f"History rows={len(history):,} {history['TIME_DT'].iloc[0]} -> "
        f"{history['TIME_DT'].iloc[-1]}",
        flush=True,
    )
    candidates = [
        {
            "generation": "14_precision_frequency",
            "minimum_expected_r": minimum_r,
            "session_profile": session,
            "target_win_rate": 0.60,
            "minimum_profit_factor": 1.05,
        }
        for minimum_r, session in product(
            (-0.05, 0.0), ("may_baseline", "controlled_expanded")
        )
    ]
    fold_results = {index: {} for index in range(len(candidates))}
    fold_traces = {index: {} for index in range(len(candidates))}
    fold_filters = {index: {} for index in range(len(candidates))}
    diagnostics = {}
    for fold_name, fold_start, fold_end in SELECTION_FOLDS:
        train = training_frame(history, fold_start)
        validation = history[
            (history["TIME_DT"] >= fold_start) & (history["TIME_DT"] < fold_end)
        ].copy().reset_index(drop=True)
        models = train_two_stage_experts(train, features)
        base, precision, expert, diagnostics[fold_name] = predict_two_stage_scores(
            models, validation, features
        )
        for index, candidate in enumerate(candidates):
            stats, trace, filtered = candidate_stats(
                validation, base, precision, expert, candidate
            )
            fold_results[index][fold_name] = stats
            fold_traces[index][fold_name] = trace
            fold_filters[index][fold_name] = filtered
        print(
            f"Fold {fold_name}: train={len(train):,} validation={len(validation):,}",
            flush=True,
        )
        del train, validation, models, base, precision, expert
        gc.collect()

    ranked = []
    for index, candidate in enumerate(candidates):
        compact_folds = {
            name: compact_stats(stats)
            for name, stats in fold_results[index].items()
        }
        rank = _candidate_rank(fold_results[index])
        ranked.append(
            {
                **candidate,
                "candidate_index": index,
                "qualified": all(
                    fold_pass(stats, 100)
                    for stats in fold_results[index].values()
                ),
                "rank": list(rank),
                "folds": compact_folds,
            }
        )
    ranked.sort(key=lambda item: tuple(item["rank"]), reverse=True)
    qualified = [item for item in ranked if item["qualified"]]
    if qualified:
        selected = max(
            qualified,
            key=lambda item: sum(
                stats["trades"] for stats in item["folds"].values()
            ),
        )
    else:
        selected = ranked[0]
    selected_index = selected["candidate_index"]

    holdout_train = training_frame(history, HISTORICAL_HOLDOUT_START)
    holdout = history[history["TIME_DT"] >= HISTORICAL_HOLDOUT_START].copy()
    holdout_models = train_two_stage_experts(holdout_train, features)
    holdout_base, holdout_precision, holdout_expert, diagnostics[
        "2025_2026_05_holdout"
    ] = predict_two_stage_scores(holdout_models, holdout, features)
    holdout_stats, holdout_trace, holdout_filter = candidate_stats(
        holdout,
        holdout_base,
        holdout_precision,
        holdout_expert,
        selected,
    )
    del (
        holdout_train,
        holdout,
        holdout_models,
        holdout_base,
        holdout_precision,
        holdout_expert,
    )
    gc.collect()

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

    final_cutoff = history["TIME_DT"].iloc[-1] + timedelta(minutes=1)
    final_train = training_frame(history, final_cutoff)
    final_models = train_two_stage_experts(final_train, features)
    recent_base, recent_precision, recent_expert, diagnostics[
        "2026_recent"
    ] = predict_two_stage_scores(final_models, recent, features)
    recent_stats, recent_trace, recent_filter = candidate_stats(
        recent,
        recent_base,
        recent_precision,
        recent_expert,
        selected,
    )
    recent_cost_stats, _, _ = candidate_stats(
        recent,
        recent_base,
        recent_precision,
        recent_expert,
        selected,
        cost=10.0,
    )
    current_stats = benchmark_current(recent, features)
    gen12_stats = _gen12_recent_benchmark()

    compact_holdout = compact_stats(holdout_stats)
    compact_recent = compact_stats(recent_stats)
    compact_cost = compact_stats(recent_cost_stats)
    compact_current = compact_stats(current_stats)
    promotion_pass = bool(
        qualified
        and fold_pass(holdout_stats, 100)
        and fold_pass(recent_stats, 30)
        and compact_recent["trades"] > compact_current["trades"]
        and compact_recent["trades"] > int(gen12_stats["trades"])
        and compact_recent["win_rate"] >= 0.60
        and compact_recent["pnl"] > compact_current["pnl"]
        and _profit_factor(compact_recent) > _profit_factor(compact_current)
        and compact_cost["pnl"] > 0.0
        and _profit_factor(compact_cost) >= 1.05
    )

    selected_params = {
        key: value
        for key, value in selected.items()
        if key not in {"candidate_index", "qualified", "rank", "folds"}
    }
    config = {
        **selected_params,
        "status": "promotion_pass" if promotion_pass else "research_only",
        "qualified_selection": bool(qualified),
        "model_profile": MODEL_PROFILE,
        "model_splits": MODEL_SPLITS,
        "meta_profile": META_PROFILE,
        "allocator_config": ALLOCATOR_CONFIG,
        "model_files": save_models(final_models),
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
        "objective": "maximize executable trades subject to win_rate>=60%, PF>=1.05, positive expectancy and drawdown guard",
        "data": {
            "history_rows": len(history),
            "history_start": history["TIME_DT"].iloc[0].isoformat(),
            "history_end": history["TIME_DT"].iloc[-1].isoformat(),
            "recent_rows": len(recent),
            "recent_start": recent["TIME_DT"].iloc[0].isoformat(),
            "recent_end": recent["TIME_DT"].iloc[-1].isoformat(),
        },
        "qualified_count": len(qualified),
        "selected": {"params": selected_params, "folds": selected_folds},
        "score_diagnostics": diagnostics,
        "allocator_trace": {
            "selection": fold_traces[selected_index],
            "holdout": holdout_trace,
            "recent": recent_trace,
        },
        "filter_diagnostics": {
            "selection": fold_filters[selected_index],
            "holdout": holdout_filter,
            "recent": recent_filter,
        },
        "benchmarks": {"gen12": gen12_stats, "current": compact_current},
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
