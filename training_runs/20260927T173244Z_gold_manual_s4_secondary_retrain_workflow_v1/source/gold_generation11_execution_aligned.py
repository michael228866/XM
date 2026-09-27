from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from barrier_final_train import prepare_barrier_data
from gold_expected_r_champion import _top_k_threshold
from gold_expected_r_walk_forward import (
    EXTRA_COST_POINTS,
    HISTORICAL_HOLDOUT_START,
    HORIZON,
    MIN_SL_PRICE,
    MIN_TP_PRICE,
    SL_ATR,
    TP_ATR,
    aggregate_score,
    fold_pass,
    make_params,
    session_mask,
)
from gold_generation8_residual_walk_forward import add_generation8_targets
from gold_recent_walk_forward import DEFAULT_TERMINAL, build_feature_frame
from gold_regime_experts_iterative import (
    EXPERT_NAMES,
    MODEL_PROFILES,
    predict_experts,
    train_experts,
    training_frame,
)
from gold_regime_experts_walk_forward import (
    CURRENT_MODEL_FILE,
    RECENT_START,
    SELECTION_FOLDS,
    benchmark_current,
    evaluate_frame,
)
from gold_short_rule_research import compact_stats


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_JSON = PROJECT_ROOT / "gold_generation11_execution_aligned.json"
REPORT_MD = PROJECT_ROOT / "gold_generation11_execution_aligned.md"
CONFIG_FILE = PROJECT_ROOT / "gold_generation11_candidate.json"
MODEL_FILES = {
    generation: {
        name: PROJECT_ROOT / f"gold_generation11_{generation}_{name}_xgb.json"
        for name in EXPERT_NAMES
    }
    for generation in MODEL_PROFILES
}

ALLOCATOR_CONFIG = {
    "maturity_rows": HORIZON,
    "window_rows": 90_000,
    "min_rows": 30_000,
    "block_rows": 10_080,
    "champion_min_trades": 8,
    "switch_margin": 0.03,
    "confirm_blocks": 2,
}
QUALITY_PROFILES = {
    "quality_105": {"minimum_mean_r": 0.00, "minimum_profit_factor": 1.05},
    "quality_115": {"minimum_mean_r": 0.03, "minimum_profit_factor": 1.15},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train generation 11 execution-aligned GOLD experts."
    )
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def first_touch_exit_offsets(frame: pd.DataFrame, direction: int) -> np.ndarray:
    if direction not in (1, 2):
        raise ValueError("direction must be 1 or 2")
    close = frame["CLOSE"].to_numpy(dtype=np.float64)
    high = frame["HIGH"].to_numpy(dtype=np.float64)
    low = frame["LOW"].to_numpy(dtype=np.float64)
    atr = frame["ATR"].to_numpy(dtype=np.float64)
    take_profit = np.maximum(atr * TP_ATR, MIN_TP_PRICE)
    stop_loss = np.maximum(atr * SL_ATR, MIN_SL_PRICE)
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


def add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    result = add_generation8_targets(frame)
    result["LONG_TARGET"] = (result["LONG_OUTCOME"] == 1).astype(np.int8)
    result["SHORT_TARGET"] = (result["SHORT_OUTCOME"] == 1).astype(np.int8)
    result["LONG_EXIT_OFFSET"] = first_touch_exit_offsets(result, 1)
    result["SHORT_EXIT_OFFSET"] = first_touch_exit_offsets(result, 2)
    return result


def execution_realized_metrics(
    reward: np.ndarray,
    exit_offset: np.ndarray,
    score: np.ndarray,
    mask: np.ndarray,
    threshold: float,
) -> dict | None:
    active = (
        mask
        & np.isfinite(reward)
        & np.isfinite(score)
        & (exit_offset > 0)
        & (score > 0.0)
        & (score >= threshold)
    )
    rising = active & ~np.r_[False, active[:-1]]
    starts = np.flatnonzero(rising)
    selected = []
    free_index = 0
    for index in starts:
        if index < free_index:
            continue
        selected.append(index)
        free_index = index + int(exit_offset[index]) + 1
    if not selected:
        return None
    values = reward[np.asarray(selected, dtype=np.int64)]
    gains = float(values[values > 0.0].sum())
    losses = float(-values[values < 0.0].sum())
    profit_factor = float("inf") if losses == 0.0 else gains / losses
    mean_r = float(values.mean())
    equity = np.cumsum(values)
    drawdown = float(
        np.min(equity - np.maximum.accumulate(np.maximum(equity, 0.0)))
    )
    return {
        "trades": len(selected),
        "mean_r": mean_r,
        "win_rate": float(np.mean(values > 0.0)),
        "profit_factor": profit_factor,
        "score": mean_r + 0.05 * min(profit_factor, 3.0) + 0.01 * drawdown,
    }


def rolling_execution_aligned_signals(
    frame: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    trend_strength: np.ndarray,
    candidate: dict,
) -> tuple[np.ndarray, dict]:
    names = tuple(MODEL_PROFILES)
    if set(probabilities) != set(names):
        raise ValueError(f"Expected probabilities for {names}")
    row_count = len(frame)
    if any(values.shape != (row_count, 3) for values in probabilities.values()):
        raise ValueError("All probability arrays must have shape (rows, 3)")

    dates = frame["TIME_DT"].dt.date.to_numpy()
    indices = np.arange(row_count)
    allowed = session_mask(frame, candidate["session_profile"]) & (
        trend_strength >= candidate["minimum_trend_strength"]
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
    output_score = np.full((row_count, 3), -np.inf, dtype=np.float32)
    champion = {1: None, 2: None}
    pending = {1: None, 2: None}
    pending_count = {1: 0, 2: 0}
    quality = QUALITY_PROFILES[candidate["quality_profile"]]
    trace = {
        "blocks": 0,
        "cash_blocks": {"long": 0, "short": 0},
        "champion_blocks": {
            "long": {name: 0 for name in names},
            "short": {name: 0 for name in names},
        },
        "switches": {"long": 0, "short": 0},
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

        for direction, label in ((1, "long"), (2, "short")):
            arm_metrics = {}
            for name in names:
                score = probabilities[name][:, direction]
                calibration_mask = (
                    (indices >= history_start)
                    & (indices < split)
                    & allowed
                    & (score > 0.0)
                )
                threshold = _top_k_threshold(
                    score, dates, calibration_mask, candidate["top_k_per_day"]
                )
                if threshold is None:
                    continue
                metrics = execution_realized_metrics(
                    rewards[direction],
                    exits[direction],
                    score,
                    (indices >= split) & (indices < history_end) & allowed,
                    threshold,
                )
                if (
                    metrics is not None
                    and metrics["trades"] >= ALLOCATOR_CONFIG["champion_min_trades"]
                    and metrics["mean_r"] >= quality["minimum_mean_r"]
                    and metrics["profit_factor"] >= quality["minimum_profit_factor"]
                ):
                    arm_metrics[name] = metrics["score"]

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
            score = probabilities[selected][:, direction]
            threshold = _top_k_threshold(
                score,
                dates,
                (indices >= history_start)
                & (indices < history_end)
                & allowed
                & (score > 0.0),
                candidate["top_k_per_day"],
            )
            if threshold is None:
                trace["cash_blocks"][label] += 1
                continue
            block_score = score[block_start:block_end]
            selected_rows = (
                allowed[block_start:block_end]
                & (block_score > 0.0)
                & (block_score >= threshold)
            )
            output[block_start:block_end, direction][selected_rows] = 1.0
            output_score[block_start:block_end, direction][selected_rows] = block_score[
                selected_rows
            ]
            trace["emitted"][label] += int(selected_rows.sum())
            trace["champion_blocks"][label][selected] += 1

    conflicts = (output[:, 1] > 0.0) & (output[:, 2] > 0.0)
    prefer_long = output_score[:, 1] >= output_score[:, 2]
    output[conflicts & prefer_long, 2] = 0.0
    output[conflicts & ~prefer_long, 1] = 0.0
    output[:, 0] = 1.0 - np.maximum(output[:, 1], output[:, 2])
    return output, trace


def candidate_stats(
    frame: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    trend_strength: np.ndarray,
    candidate: dict,
    cost: float = EXTRA_COST_POINTS,
) -> tuple[dict, dict]:
    signals, trace = rolling_execution_aligned_signals(
        frame, probabilities, trend_strength, candidate
    )
    return evaluate_frame(make_params(candidate, cost), frame, signals), trace


def train_generation_pair(
    train: pd.DataFrame, validation: pd.DataFrame, features: list[str]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    predictions = {}
    trend_strength = None
    for generation in MODEL_PROFILES:
        models = train_experts(train, features, generation)
        prediction, strength = predict_experts(models, validation, features)
        predictions[generation] = prediction
        if trend_strength is None:
            trend_strength = strength
        elif not np.allclose(trend_strength, strength):
            raise RuntimeError("Trend routing differs between generations")
    return predictions, trend_strength


def _profit_factor(stats: dict) -> float:
    value = stats["profit_factor"]
    return 3.0 if value is None else float(value)


def markdown_report(report: dict) -> str:
    lines = [
        "# GOLD generation 11 execution-aligned selective ensemble",
        "",
        "Balanced/time-decay direction experts selected by executable realized-R; CASH is the default.",
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
    reward = np.array([np.nan] * 20, dtype=np.float32)
    reward[[2, 8, 14]] = (0.5, -1.0, 0.6)
    score = np.zeros(20, dtype=np.float32)
    score[[2, 3, 8, 14]] = 0.9
    exits = np.full(20, -1, dtype=np.int16)
    exits[[2, 8, 14]] = 4
    metrics = execution_realized_metrics(
        reward, exits, score, np.ones(20, dtype=bool), 0.8
    )
    assert metrics is not None and metrics["trades"] == 3

    rng = np.random.default_rng(11)
    rows = 120_000
    frame = pd.DataFrame(
        {
            "TIME_DT": pd.date_range("2026-01-05", periods=rows, freq="min"),
            "M1_RSI": np.full(rows, 55.0),
            "LONG_REWARD": np.full(rows, -1.0, dtype=np.float32),
            "SHORT_REWARD": np.full(rows, -1.0, dtype=np.float32),
            "LONG_EXIT_OFFSET": np.full(rows, 30, dtype=np.int16),
            "SHORT_EXIT_OFFSET": np.full(rows, 30, dtype=np.int16),
        }
    )
    good = rng.random(rows).astype(np.float32)
    weak = rng.random(rows).astype(np.float32)
    frame.loc[good >= 0.995, "LONG_REWARD"] = 0.8
    probabilities = {}
    for name, long_score in (("balanced", good), ("time_decay", weak)):
        values = np.zeros((rows, 3), dtype=np.float32)
        values[:, 1] = long_score
        values[:, 2] = 0.1
        values[:, 0] = 1.0 - np.maximum(values[:, 1], values[:, 2])
        probabilities[name] = values
    candidate = {
        "generation": "11_execution_aligned",
        "top_k_per_day": 3,
        "minimum_trend_strength": 0.0,
        "session_profile": "controlled_expanded",
        "quality_profile": "quality_105",
    }
    signals, trace = rolling_execution_aligned_signals(
        frame, probabilities, np.ones(rows, dtype=np.float32), candidate
    )
    assert signals[:, 1].sum() > 0
    assert signals[:, 2].sum() == 0
    assert trace["cash_blocks"]["short"] > 0
    print("generation11_self_check_ok")


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
            "generation": "11_execution_aligned",
            "top_k_per_day": top_k,
            "minimum_trend_strength": strength,
            "session_profile": session,
            "quality_profile": quality,
        }
        for top_k, strength, session, quality in product(
            (1, 2, 3),
            (0.0, 0.2),
            ("may_baseline", "controlled_expanded"),
            QUALITY_PROFILES,
        )
    ]
    fold_results = {index: {} for index in range(len(candidates))}
    fold_traces = {index: {} for index in range(len(candidates))}
    for fold_name, fold_start, fold_end in SELECTION_FOLDS:
        train = training_frame(history, fold_start)
        validation = history[
            (history["TIME_DT"] >= fold_start) & (history["TIME_DT"] < fold_end)
        ].copy().reset_index(drop=True)
        predictions, strength = train_generation_pair(train, validation, features)
        for index, candidate in enumerate(candidates):
            stats, trace = candidate_stats(
                validation, predictions, strength, candidate
            )
            fold_results[index][fold_name] = stats
            fold_traces[index][fold_name] = trace
        print(
            f"Fold {fold_name}: train={len(train):,} validation={len(validation):,}",
            flush=True,
        )

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
    holdout = history[history["TIME_DT"] >= HISTORICAL_HOLDOUT_START].copy()
    holdout_predictions, holdout_strength = train_generation_pair(
        holdout_train, holdout, features
    )
    holdout_stats, holdout_trace = candidate_stats(
        holdout, holdout_predictions, holdout_strength, selected
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

    final_cutoff = history["TIME_DT"].iloc[-1] + timedelta(minutes=1)
    final_train = training_frame(history, final_cutoff)
    recent_predictions = {}
    recent_strength = None
    for generation in MODEL_PROFILES:
        models = train_experts(final_train, features, generation)
        for name, model in models.items():
            model.save_model(MODEL_FILES[generation][name])
        prediction, strength = predict_experts(models, recent, features)
        recent_predictions[generation] = prediction
        if recent_strength is None:
            recent_strength = strength
    recent_stats, recent_trace = candidate_stats(
        recent, recent_predictions, recent_strength, selected
    )
    recent_cost_stats, _ = candidate_stats(
        recent, recent_predictions, recent_strength, selected, cost=10.0
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
    model_files = {
        generation: {name: path.name for name, path in files.items()}
        for generation, files in MODEL_FILES.items()
    }
    config = {
        **selected_params,
        "status": "promotion_pass" if promotion_pass else "research_only",
        "qualified_selection": bool(qualified),
        "tp_atr": TP_ATR,
        "sl_atr": SL_ATR,
        "max_hold": HORIZON,
        "model_profiles": MODEL_PROFILES,
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
            "recent_rows": len(recent),
            "recent_start": recent["TIME_DT"].iloc[0].isoformat(),
            "recent_end": recent["TIME_DT"].iloc[-1].isoformat(),
        },
        "qualified_count": len(qualified),
        "selected": {"params": selected_params, "folds": selected_folds},
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
