from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from barrier_classifier_strategy import (
    build_first_touch_outcome_and_reward,
    evaluate,
)
from barrier_final_train import prepare_barrier_data
from gold_expected_r_champion import _top_k_threshold
from gold_expected_r_walk_forward import (
    EXTRA_COST_POINTS,
    HISTORICAL_HOLDOUT_START,
    HORIZON,
    MIN_SL_PRICE,
    MIN_TP_PRICE,
    RISK_PER_TRADE,
    aggregate_score,
    fold_pass,
    make_params,
    session_mask,
)
from gold_generation11_execution_aligned import (
    QUALITY_PROFILES,
    _profit_factor,
    execution_realized_metrics,
)
from gold_generation12_executable_events import (
    MODEL_PROFILE,
    _new_classifier,
    _new_regressor,
    _sample_weight,
    _serialize_calibrator,
    sequential_event_indices,
)
from gold_recent_walk_forward import DEFAULT_TERMINAL, build_feature_frame
from gold_regime_experts_iterative import EXPERT_NAMES, training_frame
from gold_regime_experts_walk_forward import (
    RECENT_START,
    SELECTION_FOLDS,
    benchmark_current,
    route_arrays,
)
from gold_short_rule_research import compact_stats
from sklearn.isotonic import IsotonicRegression


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_JSON = PROJECT_ROOT / "gold_generation13_directional_exits.json"
REPORT_MD = PROJECT_ROOT / "gold_generation13_directional_exits.md"
CONFIG_FILE = PROJECT_ROOT / "gold_generation13_candidate.json"

# Two profiles are enough to test the actual Gen12 weakness without an exit-grid
# overfit: the incumbent payoff and a symmetric stop that raises realized R/PF.
EXIT_PROFILES = {
    "baseline_13_16": {"tp_atr": 1.3, "sl_atr": 1.6},
    "symmetric_13_13": {"tp_atr": 1.3, "sl_atr": 1.3},
}
PROFILE_IDS = {name: index + 1 for index, name in enumerate(EXIT_PROFILES)}
MODEL_FILES = {
    (profile, name, kind): PROJECT_ROOT
    / f"gold_generation13_{profile}_{name}_{kind}_xgb.json"
    for profile in EXIT_PROFILES
    for name in EXPERT_NAMES
    for kind in ("win", "mean_r")
}
ALLOCATOR_CONFIG = {
    "maturity_rows": HORIZON,
    "window_rows": 90_000,
    "min_rows": 30_000,
    "block_rows": 10_080,
    "minimum_quality_trades": 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train generation 13 with direction-specific rolling exits."
    )
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def first_touch_exit_offsets(
    frame: pd.DataFrame, direction: int, profile: dict
) -> np.ndarray:
    if direction not in (1, 2):
        raise ValueError("direction must be 1 or 2")
    close = frame["CLOSE"].to_numpy(dtype=np.float64)
    high = frame["HIGH"].to_numpy(dtype=np.float64)
    low = frame["LOW"].to_numpy(dtype=np.float64)
    atr = frame["ATR"].to_numpy(dtype=np.float64)
    take_profit = np.maximum(atr * profile["tp_atr"], MIN_TP_PRICE)
    stop_loss = np.maximum(atr * profile["sl_atr"], MIN_SL_PRICE)
    valid = (
        np.isfinite(close)
        & np.isfinite(high)
        & np.isfinite(low)
        & np.isfinite(take_profit)
        & np.isfinite(stop_loss)
    )
    unresolved = valid.copy()
    offsets = np.full(len(frame), -1, dtype=np.int16)
    for offset in range(1, HORIZON + 1):
        limit = len(frame) - offset
        if limit <= 0:
            break
        active = unresolved[:limit]
        if not active.any():
            break
        if direction == 1:
            hit = (high[offset:] >= close[:limit] + take_profit[:limit]) | (
                low[offset:] <= close[:limit] - stop_loss[:limit]
            )
        else:
            hit = (low[offset:] <= close[:limit] - take_profit[:limit]) | (
                high[offset:] >= close[:limit] + stop_loss[:limit]
            )
        resolved = active & hit
        offsets[:limit][resolved] = offset
        unresolved[:limit][resolved] = False
    timeout_limit = len(frame) - HORIZON
    if timeout_limit > 0:
        offsets[:timeout_limit][unresolved[:timeout_limit]] = HORIZON
    return offsets


def build_exit_targets(frame: pd.DataFrame) -> dict:
    targets = {}
    for profile_name, profile in EXIT_PROFILES.items():
        targets[profile_name] = {}
        for direction in (1, 2):
            outcome, reward = build_first_touch_outcome_and_reward(
                frame,
                HORIZON,
                profile["tp_atr"],
                profile["sl_atr"],
                MIN_TP_PRICE,
                MIN_SL_PRICE,
                direction,
                extra_cost_points=EXTRA_COST_POINTS,
            )
            targets[profile_name][direction] = {
                "outcome": outcome.astype(np.int8, copy=False),
                "reward": reward.astype(np.float32),
                "exit": first_touch_exit_offsets(frame, direction, profile),
            }
        print(
            f"  targets {profile_name}: "
            f"long_R={np.nanmean(targets[profile_name][1]['reward']):.4f} "
            f"short_R={np.nanmean(targets[profile_name][2]['reward']):.4f}",
            flush=True,
        )
    return targets


def slice_targets(targets: dict, positions: np.ndarray) -> dict:
    return {
        profile: {
            direction: {
                key: values[key][positions]
                for key in ("outcome", "reward", "exit")
            }
            for direction, values in directions.items()
        }
        for profile, directions in targets.items()
    }


def _train_one_expert(
    frame: pd.DataFrame,
    features: list[str],
    reward: np.ndarray,
    indices: np.ndarray,
    label: str,
) -> dict:
    if len(indices) < 1_000:
        raise RuntimeError(f"{label} has only {len(indices):,} executable events")
    split = int(len(indices) * (1.0 - MODEL_PROFILE["calibration_ratio"]))
    fit_indices = indices[:split]
    calibration_indices = indices[split:]
    fit_reward = reward[fit_indices]
    calibration_reward = reward[calibration_indices]
    fit_win = (fit_reward > 0.0).astype(np.int8)
    calibration_win = (calibration_reward > 0.0).astype(np.int8)
    if np.unique(fit_win).size != 2 or np.unique(calibration_win).size != 2:
        raise RuntimeError(f"{label} executable events lack a win/loss class")

    fit = frame.iloc[fit_indices]
    calibration = frame.iloc[calibration_indices]
    classifier = _new_classifier()
    mean_model = _new_regressor()
    weight = _sample_weight(fit["TIME_DT"])
    classifier.fit(
        fit[features].astype(np.float32), fit_win, sample_weight=weight
    )
    mean_model.fit(
        fit[features].astype(np.float32), fit_reward, sample_weight=weight
    )
    raw = classifier.predict_proba(
        calibration[features].astype(np.float32)
    )[:, 1]
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(raw, calibration_win)
    return {
        "win": classifier,
        "mean_r": mean_model,
        "calibrator": calibrator,
        "average_win_r": float(calibration_reward[calibration_reward > 0.0].mean()),
        "average_loss_r": float(
            calibration_reward[calibration_reward <= 0.0].mean()
        ),
        "events": len(indices),
        "fit_events": len(fit_indices),
        "calibration_events": len(calibration_indices),
    }


def train_profile_experts(
    frame: pd.DataFrame,
    targets: dict,
    positions: np.ndarray,
    features: list[str],
) -> dict:
    if len(frame) != len(positions):
        raise ValueError("frame and target positions must have equal length")
    _, masks = route_arrays(frame, features)
    allowed = session_mask(frame, "controlled_expanded")
    models = {profile: {} for profile in EXIT_PROFILES}
    for profile_name in EXIT_PROFILES:
        for direction, prefix in ((1, "long"), (2, "short")):
            reward = targets[profile_name][direction]["reward"][positions]
            exits = targets[profile_name][direction]["exit"][positions]
            direction_mask = masks[f"{prefix}_trend"] | masks[f"{prefix}_pullback"]
            sequential = sequential_event_indices(
                allowed & direction_mask & np.isfinite(reward), exits
            )
            for style in ("trend", "pullback"):
                name = f"{prefix}_{style}"
                indices = sequential[masks[name][sequential]]
                label = f"{profile_name}/{name}"
                models[profile_name][name] = _train_one_expert(
                    frame, features, reward, indices, label
                )
                model = models[profile_name][name]
                print(
                    f"  {label}: events={model['events']:,} "
                    f"fit={model['fit_events']:,} "
                    f"calibrate={model['calibration_events']:,}",
                    flush=True,
                )
            del reward, exits
    return models


def predict_profile_scores(
    models: dict, frame: pd.DataFrame, features: list[str]
) -> tuple[dict, dict]:
    _, masks = route_arrays(frame, features)
    scores = {
        profile: {
            1: np.full(len(frame), np.nan, dtype=np.float32),
            2: np.full(len(frame), np.nan, dtype=np.float32),
        }
        for profile in EXIT_PROFILES
    }
    diagnostics = {}
    for profile_name, profile_models in models.items():
        diagnostics[profile_name] = {}
        for name, model in profile_models.items():
            indices = np.flatnonzero(masks[name])
            if len(indices) == 0:
                continue
            x = frame.iloc[indices][features].astype(np.float32)
            probability = model["calibrator"].predict(
                model["win"].predict_proba(x)[:, 1]
            ).astype(np.float32)
            probability_r = (
                probability * model["average_win_r"]
                + (1.0 - probability) * model["average_loss_r"]
            )
            mean_r = model["mean_r"].predict(x).astype(np.float32)
            expected_r = 0.5 * (probability_r + mean_r)
            direction = 1 if name.startswith("long_") else 2
            scores[profile_name][direction][indices] = expected_r
            diagnostics[profile_name][name] = {
                "rows": len(indices),
                "p_win_q50_q90_q99": np.quantile(
                    probability, (0.50, 0.90, 0.99)
                ).round(4).tolist(),
                "expected_r_q50_q90_q99": np.quantile(
                    expected_r, (0.50, 0.90, 0.99)
                ).round(4).tolist(),
            }
    return scores, diagnostics


def rolling_directional_exit_signals(
    frame: pd.DataFrame, scores: dict, targets: dict, candidate: dict
) -> tuple[np.ndarray, np.ndarray, dict]:
    if set(scores) != set(EXIT_PROFILES) or set(targets) != set(EXIT_PROFILES):
        raise ValueError("scores and targets must contain every exit profile")
    row_count = len(frame)
    dates = frame["TIME_DT"].dt.date.to_numpy()
    indices = np.arange(row_count)
    allowed = session_mask(frame, candidate["session_profile"])
    quality = QUALITY_PROFILES[candidate["quality_profile"]]
    output = np.zeros((row_count, 3), dtype=np.float32)
    profile_choice = np.zeros((row_count, 3), dtype=np.int8)
    trace = {
        "blocks": 0,
        "cash_blocks": {"long": 0, "short": 0},
        "champion_blocks": {
            "long": {name: 0 for name in EXIT_PROFILES},
            "short": {name: 0 for name in EXIT_PROFILES},
        },
        "emitted": {"long": 0, "short": 0},
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
        for direction, direction_name in ((1, "long"), (2, "short")):
            arms = {}
            masks = {}
            for profile_name in EXIT_PROFILES:
                score = scores[profile_name][direction]
                exits = targets[profile_name][direction]["exit"]
                candidate_mask = (
                    allowed
                    & np.isfinite(score)
                    & (score > 0.0)
                    & (score >= candidate["minimum_expected_r"])
                    & (exits > 0)
                )
                masks[profile_name] = candidate_mask
                threshold = _top_k_threshold(
                    score,
                    dates,
                    candidate_mask
                    & (indices >= history_start)
                    & (indices < split),
                    candidate["top_k_per_day"],
                )
                if threshold is None:
                    continue
                metrics = execution_realized_metrics(
                    targets[profile_name][direction]["reward"],
                    exits,
                    score,
                    candidate_mask
                    & (indices >= split)
                    & (indices < history_end),
                    threshold,
                )
                if (
                    metrics is not None
                    and metrics["trades"]
                    >= ALLOCATOR_CONFIG["minimum_quality_trades"]
                    and metrics["mean_r"] >= quality["minimum_mean_r"]
                    and metrics["profit_factor"]
                    >= quality["minimum_profit_factor"]
                ):
                    arms[profile_name] = metrics["score"]
            if not arms:
                trace["cash_blocks"][direction_name] += 1
                continue
            champion = max(arms, key=arms.get)
            score = scores[champion][direction]
            candidate_mask = masks[champion]
            threshold = _top_k_threshold(
                score,
                dates,
                candidate_mask
                & (indices >= history_start)
                & (indices < history_end),
                candidate["top_k_per_day"],
            )
            if threshold is None:
                trace["cash_blocks"][direction_name] += 1
                continue
            selected = candidate_mask[block_start:block_end] & (
                score[block_start:block_end] >= threshold
            )
            output[block_start:block_end, direction][selected] = 1.0
            profile_choice[block_start:block_end, direction][selected] = (
                PROFILE_IDS[champion]
            )
            trace["champion_blocks"][direction_name][champion] += 1
            trace["emitted"][direction_name] += int(selected.sum())
    output[:, 0] = 1.0 - np.maximum(output[:, 1], output[:, 2])
    return output, profile_choice, trace


def entry_distances(
    frame: pd.DataFrame, signals: np.ndarray, profile_choice: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    atr = frame["ATR"].to_numpy(dtype=np.float64)
    default = EXIT_PROFILES["baseline_13_16"]
    take_profit = np.maximum(atr * default["tp_atr"], MIN_TP_PRICE)
    stop_loss = np.maximum(atr * default["sl_atr"], MIN_SL_PRICE)
    chosen_direction = np.where(signals[:, 1] >= signals[:, 2], 1, 2)
    active = np.maximum(signals[:, 1], signals[:, 2]) > 0.0
    for profile_name, profile in EXIT_PROFILES.items():
        profile_id = PROFILE_IDS[profile_name]
        chosen = active & (
            profile_choice[np.arange(len(frame)), chosen_direction] == profile_id
        )
        take_profit[chosen] = np.maximum(
            atr[chosen] * profile["tp_atr"], MIN_TP_PRICE
        )
        stop_loss[chosen] = np.maximum(
            atr[chosen] * profile["sl_atr"], MIN_SL_PRICE
        )
    return take_profit, stop_loss


def evaluate_directional_frame(
    candidate: dict,
    frame: pd.DataFrame,
    signals: np.ndarray,
    profile_choice: np.ndarray,
    cost: float,
) -> dict:
    take_profit, stop_loss = entry_distances(frame, signals, profile_choice)
    params = make_params(candidate, cost)
    return evaluate(
        params,
        frame["CLOSE"].to_numpy(dtype=np.float64),
        frame["ATR"].to_numpy(dtype=np.float64),
        signals,
        hours=frame["TIME_DT"].dt.hour.to_numpy(dtype=np.int16),
        weekdays=frame["TIME_DT"].dt.dayofweek.to_numpy(dtype=np.int8),
        dates=frame["TIME_DT"].dt.date.to_numpy(),
        rsi_values=frame["M1_RSI"].to_numpy(dtype=np.float64),
        highs=frame["HIGH"].to_numpy(dtype=np.float64),
        lows=frame["LOW"].to_numpy(dtype=np.float64),
        entry_tp_values=take_profit,
        entry_sl_values=stop_loss,
    )


def candidate_stats(
    frame: pd.DataFrame,
    scores: dict,
    targets: dict,
    candidate: dict,
    cost: float = EXTRA_COST_POINTS,
) -> tuple[dict, dict]:
    signals, profile_choice, trace = rolling_directional_exit_signals(
        frame, scores, targets, candidate
    )
    return (
        evaluate_directional_frame(
            candidate, frame, signals, profile_choice, cost
        ),
        trace,
    )


def save_models(models: dict) -> dict:
    output = {}
    for profile_name, profile_models in models.items():
        output[profile_name] = {}
        for name, model in profile_models.items():
            win_path = MODEL_FILES[(profile_name, name, "win")]
            mean_path = MODEL_FILES[(profile_name, name, "mean_r")]
            model["win"].save_model(win_path)
            model["mean_r"].save_model(mean_path)
            output[profile_name][name] = {
                "win_file": win_path.name,
                "mean_r_file": mean_path.name,
                "isotonic": _serialize_calibrator(model["calibrator"]),
                "average_win_r": model["average_win_r"],
                "average_loss_r": model["average_loss_r"],
                "events": model["events"],
            }
    return output


def markdown_report(report: dict) -> str:
    lines = [
        "# GOLD generation 13 directional exit champions",
        "",
        "Two payoff-specific executable-event model sets with past-only rolling direction champions.",
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
            f"Qualified selection candidates: `{report['qualified_count']}`",
            f"Selected: `{json.dumps(report['selected']['params'], ensure_ascii=False)}`",
            f"Current recent: `{json.dumps(report['benchmarks']['current'])}`",
            f"Promotion gate: `{'PASS' if report['promotion_pass'] else 'FAIL'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def self_check() -> None:
    rows = HORIZON + 3
    frame = pd.DataFrame(
        {
            "TIME_DT": pd.date_range("2026-01-05", periods=rows, freq="min"),
            "CLOSE": np.full(rows, 100.0),
            "HIGH": np.full(rows, 100.2),
            "LOW": np.full(rows, 99.8),
            "ATR": np.full(rows, 1.0),
            "M1_RSI": np.full(rows, 55.0),
        }
    )
    frame.loc[1, ["HIGH", "LOW"]] = [100.2, 98.6]
    frame.loc[2, ["HIGH", "LOW"]] = [101.6, 99.0]
    targets = build_exit_targets(frame)
    assert targets["baseline_13_16"][1]["outcome"][0] == 1
    assert targets["baseline_13_16"][1]["exit"][0] == 2
    assert targets["symmetric_13_13"][1]["outcome"][0] == 2
    assert targets["symmetric_13_13"][1]["exit"][0] == 1

    signals = np.zeros((rows, 3), dtype=np.float32)
    signals[:, 0] = 1.0
    signals[0] = (0.0, 1.0, 0.0)
    choice = np.zeros((rows, 3), dtype=np.int8)
    candidate = {
        "session_profile": "controlled_expanded",
        "top_k_per_day": 3,
        "minimum_expected_r": 0.0,
        "quality_profile": "quality_105",
    }
    choice[0, 1] = PROFILE_IDS["baseline_13_16"]
    baseline = evaluate_directional_frame(
        candidate, frame, signals, choice, EXTRA_COST_POINTS
    )
    choice[0, 1] = PROFILE_IDS["symmetric_13_13"]
    symmetric = evaluate_directional_frame(
        candidate, frame, signals, choice, EXTRA_COST_POINTS
    )
    assert baseline["take_profit_exits"] == 1 and baseline["pnl"] > 0.0
    assert symmetric["stop_loss_exits"] == 1 and symmetric["pnl"] < 0.0
    print("generation13_self_check_ok")


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0

    history, features = prepare_barrier_data()
    if not history.index.equals(pd.RangeIndex(len(history))):
        history = history.reset_index(drop=True)
    print(
        f"History rows={len(history):,} {history['TIME_DT'].iloc[0]} -> "
        f"{history['TIME_DT'].iloc[-1]}",
        flush=True,
    )
    history_targets = build_exit_targets(history)
    candidates = [
        {
            "generation": "13_directional_exits",
            "top_k_per_day": top_k,
            "minimum_expected_r": minimum_r,
            "session_profile": session,
            "quality_profile": quality,
        }
        for top_k, minimum_r, session, quality in product(
            (2, 3, 4),
            (0.0, 0.03),
            ("may_baseline", "controlled_expanded"),
            QUALITY_PROFILES,
        )
    ]
    fold_results = {index: {} for index in range(len(candidates))}
    fold_traces = {index: {} for index in range(len(candidates))}
    diagnostics = {}
    for fold_name, fold_start, fold_end in SELECTION_FOLDS:
        train = training_frame(history, fold_start)
        train_positions = train.index.to_numpy(dtype=np.int64)
        validation_mask = (
            (history["TIME_DT"] >= fold_start) & (history["TIME_DT"] < fold_end)
        ).to_numpy()
        validation_positions = np.flatnonzero(validation_mask)
        validation = history.iloc[validation_positions].copy().reset_index(drop=True)
        validation_targets = slice_targets(history_targets, validation_positions)
        models = train_profile_experts(
            train, history_targets, train_positions, features
        )
        scores, diagnostics[fold_name] = predict_profile_scores(
            models, validation, features
        )
        for index, candidate in enumerate(candidates):
            stats, trace = candidate_stats(
                validation, scores, validation_targets, candidate
            )
            fold_results[index][fold_name] = stats
            fold_traces[index][fold_name] = trace
        print(
            f"Fold {fold_name}: train={len(train):,} "
            f"validation={len(validation):,}",
            flush=True,
        )
        del models, scores, validation_targets, validation, train
        gc.collect()

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

    holdout_train = training_frame(history, HISTORICAL_HOLDOUT_START)
    holdout_train_positions = holdout_train.index.to_numpy(dtype=np.int64)
    holdout_positions = np.flatnonzero(
        (history["TIME_DT"] >= HISTORICAL_HOLDOUT_START).to_numpy()
    )
    holdout = history.iloc[holdout_positions].copy().reset_index(drop=True)
    holdout_targets = slice_targets(history_targets, holdout_positions)
    holdout_models = train_profile_experts(
        holdout_train, history_targets, holdout_train_positions, features
    )
    holdout_scores, diagnostics["2025_2026_05_holdout"] = (
        predict_profile_scores(holdout_models, holdout, features)
    )
    holdout_stats, holdout_trace = candidate_stats(
        holdout, holdout_scores, holdout_targets, selected
    )
    del holdout_models, holdout_scores, holdout_targets, holdout, holdout_train
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
    recent = recent.reset_index(drop=True)
    recent_targets = build_exit_targets(recent)

    final_cutoff = history["TIME_DT"].iloc[-1] + timedelta(minutes=1)
    final_train = training_frame(history, final_cutoff)
    final_positions = final_train.index.to_numpy(dtype=np.int64)
    final_models = train_profile_experts(
        final_train, history_targets, final_positions, features
    )
    recent_scores, diagnostics["2026_recent"] = predict_profile_scores(
        final_models, recent, features
    )
    recent_stats, recent_trace = candidate_stats(
        recent, recent_scores, recent_targets, selected
    )
    recent_cost_stats, _ = candidate_stats(
        recent, recent_scores, recent_targets, selected, cost=10.0
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

    selected_params = {
        key: value
        for key, value in selected.items()
        if key not in {"candidate_index", "qualified", "score", "folds"}
    }
    config = {
        **selected_params,
        "status": "promotion_pass" if promotion_pass else "research_only",
        "qualified_selection": bool(qualified),
        "exit_profiles": EXIT_PROFILES,
        "model_profile": MODEL_PROFILE,
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
        "exit_profiles": EXIT_PROFILES,
        "score_diagnostics": diagnostics,
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
