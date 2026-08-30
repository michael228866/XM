from __future__ import annotations

from collections.abc import Mapping

import numpy as np


EXPERT_DIRECTIONS = {
    "long_trend": 1,
    "long_pullback": 1,
    "short_trend": 2,
    "short_pullback": 2,
}


def _top_k_threshold(prediction, dates, mask, top_k_per_day):
    valid = mask & np.isfinite(prediction)
    values = prediction[valid]
    if len(values) == 0:
        return None
    day_count = len(np.unique(dates[valid]))
    keep = min(len(values), max(1, int(top_k_per_day) * day_count))
    return float(np.partition(values, len(values) - keep)[len(values) - keep])


def _realized_metrics(reward, prediction, mask, threshold, minimum_expected_r):
    selected = (
        mask
        & np.isfinite(prediction)
        & np.isfinite(reward)
        & (prediction >= threshold)
        & (prediction >= minimum_expected_r)
    )
    values = reward[selected]
    if len(values) == 0:
        return None
    gains = values[values > 0.0].sum()
    losses = -values[values < 0.0].sum()
    profit_factor = float("inf") if losses == 0.0 else float(gains / losses)
    downside = float(np.std(np.minimum(values, 0.0)))
    equity = np.cumsum(values)
    drawdown = float(np.min(equity - np.maximum.accumulate(np.maximum(equity, 0.0))))
    mean_reward = float(np.mean(values))
    win_rate = float(np.mean(values > 0.0))
    score = (
        mean_reward
        + 0.05 * min(profit_factor, 3.0)
        + 0.05 * win_rate
        - 0.20 * downside
        + 0.01 * drawdown
    )
    return {
        "trades": int(len(values)),
        "mean_r": mean_reward,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "downside": downside,
        "drawdown_r": drawdown,
        "score": score,
    }


def rolling_top_k_champion_signals(
    expert_predictions: Mapping[str, np.ndarray],
    long_reward: np.ndarray,
    short_reward: np.ndarray,
    dates: np.ndarray,
    session_mask: np.ndarray,
    *,
    top_k_per_day: int,
    minimum_expected_r: float,
    maturity_rows: int,
    window_rows: int = 60_000,
    min_rows: int = 20_000,
    block_rows: int = 10_080,
    champion_min_trades: int = 20,
    switch_margin: float = 0.05,
    confirm_blocks: int = 2,
    minimum_champion_mean_r: float | None = None,
    minimum_champion_profit_factor: float | None = None,
):
    """Return no-lookahead top-k signals from realized-R direction champions."""
    names = tuple(EXPERT_DIRECTIONS)
    if set(expert_predictions) != set(names):
        raise ValueError(f"Expected expert predictions for {names}")
    predictions = {
        name: np.asarray(expert_predictions[name], dtype=np.float32)
        for name in names
    }
    row_count = len(long_reward)
    one_dimensional = (
        np.asarray(short_reward),
        np.asarray(dates),
        np.asarray(session_mask),
    )
    if row_count == 0 or any(len(value) != row_count for value in one_dimensional):
        raise ValueError("Reward, date, and session arrays must have equal length")
    if any(value.shape != (row_count,) for value in predictions.values()):
        raise ValueError("Each expert prediction must be a one-dimensional array")
    if min(
        top_k_per_day,
        maturity_rows,
        window_rows,
        min_rows,
        block_rows,
        champion_min_trades,
        confirm_blocks,
    ) < 1:
        raise ValueError("Rolling top-k settings must be positive")
    if switch_margin < 0.0:
        raise ValueError("switch_margin must be non-negative")
    if (
        minimum_champion_profit_factor is not None
        and minimum_champion_profit_factor < 0.0
    ):
        raise ValueError("minimum_champion_profit_factor must be non-negative")

    rewards = {
        1: np.asarray(long_reward, dtype=np.float32),
        2: np.asarray(short_reward, dtype=np.float32),
    }
    dates = np.asarray(dates)
    session_mask = np.asarray(session_mask, dtype=bool)
    indices = np.arange(row_count)
    output = np.zeros((row_count, 3), dtype=np.float32)
    champion = {1: None, 2: None}
    pending = {1: None, 2: None}
    pending_count = {1: 0, 2: 0}
    trace = {
        "blocks": 0,
        "ready_blocks": 0,
        "first_ready_index": None,
        "directions": {
            "long": {"switches": 0, "champion_blocks": {name: 0 for name in names if EXPERT_DIRECTIONS[name] == 1}},
            "short": {"switches": 0, "champion_blocks": {name: 0 for name in names if EXPERT_DIRECTIONS[name] == 2}},
        },
        "emitted_long": 0,
        "emitted_short": 0,
    }

    for block_start in range(0, row_count, block_rows):
        block_end = min(row_count, block_start + block_rows)
        history_end = block_start - maturity_rows
        history_start = max(0, history_end - window_rows)
        trace["blocks"] += 1
        if history_end - history_start < min_rows:
            continue

        split = history_start + int((history_end - history_start) * 0.67)
        block_ready = False
        for direction, label in ((1, "long"), (2, "short")):
            scores = {}
            for name in names:
                if EXPERT_DIRECTIONS[name] != direction:
                    continue
                prediction = predictions[name]
                threshold = _top_k_threshold(
                    prediction,
                    dates,
                    session_mask & (indices >= history_start) & (indices < split),
                    top_k_per_day,
                )
                if threshold is None:
                    continue
                score_mask = (
                    session_mask
                    & (indices >= split)
                    & (indices < history_end)
                )
                metrics = _realized_metrics(
                    rewards[direction],
                    prediction,
                    score_mask,
                    threshold,
                    minimum_expected_r,
                )
                if (
                    metrics is not None
                    and metrics["trades"] >= champion_min_trades
                    and (
                        minimum_champion_mean_r is None
                        or metrics["mean_r"] >= minimum_champion_mean_r
                    )
                    and (
                        minimum_champion_profit_factor is None
                        or metrics["profit_factor"]
                        >= minimum_champion_profit_factor
                    )
                ):
                    scores[name] = metrics["score"]

            best = max(scores, key=scores.get) if scores else None
            current = champion[direction]
            if current is None:
                if best is not None:
                    champion[direction] = best
            elif current not in scores:
                champion[direction] = None
                pending[direction] = None
                pending_count[direction] = 0
            elif best != current and scores[best] >= scores[current] + switch_margin:
                if pending[direction] == best:
                    pending_count[direction] += 1
                else:
                    pending[direction] = best
                    pending_count[direction] = 1
                if pending_count[direction] >= confirm_blocks:
                    champion[direction] = best
                    pending[direction] = None
                    pending_count[direction] = 0
                    trace["directions"][label]["switches"] += 1
            else:
                pending[direction] = None
                pending_count[direction] = 0

            selected = champion[direction]
            if selected is None:
                continue
            prediction = predictions[selected]
            history_mask = (
                session_mask
                & (indices >= history_start)
                & (indices < history_end)
            )
            threshold = _top_k_threshold(
                prediction, dates, history_mask, top_k_per_day
            )
            if threshold is None:
                continue
            block_mask = (
                session_mask[block_start:block_end]
                & np.isfinite(prediction[block_start:block_end])
                & (prediction[block_start:block_end] >= threshold)
                & (prediction[block_start:block_end] >= minimum_expected_r)
            )
            output[block_start:block_end, direction][block_mask] = 1.0
            emitted = int(block_mask.sum())
            trace[f"emitted_{label}"] += emitted
            trace["directions"][label]["champion_blocks"][selected] += 1
            block_ready = True

        if block_ready:
            trace["ready_blocks"] += 1
            if trace["first_ready_index"] is None:
                trace["first_ready_index"] = block_start

    output[:, 0] = 1.0 - np.maximum(output[:, 1], output[:, 2])
    return output, trace


def self_check():
    rng = np.random.default_rng(7)
    rows = 12_000
    dates = np.arange(rows) // 100
    session = np.ones(rows, dtype=bool)
    long_reward = rng.normal(0.05, 0.8, rows).astype(np.float32)
    short_reward = rng.normal(0.03, 0.8, rows).astype(np.float32)
    predictions = {}
    for index, name in enumerate(EXPERT_DIRECTIONS):
        reward = long_reward if EXPERT_DIRECTIONS[name] == 1 else short_reward
        route = (np.arange(rows) + index) % 2 == 0
        values = np.full(rows, np.nan, dtype=np.float32)
        values[route] = reward[route] + rng.normal(0.0, 0.15, route.sum())
        predictions[name] = values
    settings = {
        "top_k_per_day": 2,
        "minimum_expected_r": 0.0,
        "maturity_rows": 10,
        "window_rows": 4_000,
        "min_rows": 2_000,
        "block_rows": 500,
        "champion_min_trades": 10,
        "switch_margin": 0.01,
        "confirm_blocks": 2,
    }
    output, trace = rolling_top_k_champion_signals(
        predictions, long_reward, short_reward, dates, session, **settings
    )
    mutated = long_reward.copy()
    mutated[8_000:] *= -1.0
    output_mutated, _ = rolling_top_k_champion_signals(
        predictions, mutated, short_reward, dates, session, **settings
    )
    assert np.array_equal(output[:8_000], output_mutated[:8_000])
    assert trace["ready_blocks"] > 0
    assert not output[: trace["first_ready_index"], 1:].any()
    assert set(np.unique(output)).issubset({0.0, 1.0})
    print("expected_r_champion_self_check_ok")


if __name__ == "__main__":
    self_check()
