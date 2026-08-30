from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from sklearn.isotonic import IsotonicRegression


def _fit_calibrator(probability, target):
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.int8)
    valid = np.isfinite(probability) & (probability > 0.0)
    probability = probability[valid]
    target = target[valid]
    if len(target) < 1_000 or np.unique(target).size < 2:
        return None
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(probability, target)
    return model


def rolling_champion_probabilities(
    model_probabilities: Mapping[str, np.ndarray],
    long_target: np.ndarray,
    short_target: np.ndarray,
    *,
    maturity_rows: int,
    window_rows: int = 60_000,
    min_rows: int = 20_000,
    block_rows: int = 10_080,
    switch_margin: float = 0.002,
    confirm_blocks: int = 2,
):
    """Calibrate and select a direction-specific model using past data only."""
    names = tuple(model_probabilities)
    if len(names) < 2:
        raise ValueError("At least two model probability arrays are required")
    arrays = {
        name: np.asarray(model_probabilities[name], dtype=np.float32)
        for name in names
    }
    row_count = len(long_target)
    if row_count == 0 or any(value.shape != (row_count, 3) for value in arrays.values()):
        raise ValueError("All probability arrays must have shape (rows, 3)")
    if len(short_target) != row_count:
        raise ValueError("Long and short targets must have equal length")
    if min(maturity_rows, window_rows, min_rows, block_rows, confirm_blocks) < 1:
        raise ValueError("Rolling calibration settings must be positive")
    if not 0.0 <= switch_margin < 1.0:
        raise ValueError("switch_margin must be in [0, 1)")

    targets = {
        1: np.asarray(long_target, dtype=np.int8),
        2: np.asarray(short_target, dtype=np.int8),
    }
    output = np.zeros((row_count, 3), dtype=np.float32)
    champion = {1: None, 2: None}
    pending = {1: None, 2: None}
    pending_count = {1: 0, 2: 0}
    trace = {
        "blocks": 0,
        "ready_blocks": 0,
        "first_ready_index": None,
        "directions": {
            "long": {"switches": 0, "champion_blocks": {name: 0 for name in names}},
            "short": {"switches": 0, "champion_blocks": {name: 0 for name in names}},
        },
    }

    for block_start in range(0, row_count, block_rows):
        block_end = min(row_count, block_start + block_rows)
        history_end = block_start - maturity_rows
        history_start = max(0, history_end - window_rows)
        trace["blocks"] += 1
        if history_end - history_start < min_rows:
            continue

        block_ready = False
        for direction, label in ((1, "long"), (2, "short")):
            history_target = targets[direction][history_start:history_end]
            split = history_start + int((history_end - history_start) * 0.67)
            scores = {}
            calibrators = {}
            for name in names:
                raw = arrays[name][:, direction]
                scoring_model = _fit_calibrator(
                    raw[history_start:split],
                    targets[direction][history_start:split],
                )
                full_model = _fit_calibrator(
                    raw[history_start:history_end],
                    history_target,
                )
                score_raw = raw[split:history_end]
                score_target = targets[direction][split:history_end]
                valid = np.isfinite(score_raw) & (score_raw > 0.0)
                if scoring_model is None or full_model is None or valid.sum() < 500:
                    continue
                calibrated = scoring_model.predict(score_raw[valid])
                scores[name] = float(
                    np.mean((calibrated - score_target[valid]) ** 2)
                )
                calibrators[name] = full_model
            if len(scores) != len(names):
                continue

            best = min(scores, key=scores.get)
            current = champion[direction]
            if current is None:
                champion[direction] = best
            elif best != current and scores[best] + switch_margin < scores[current]:
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
            raw_block = arrays[selected][block_start:block_end, direction]
            active = np.isfinite(raw_block) & (raw_block > 0.0)
            if active.any():
                calibrated_block = np.zeros(len(raw_block), dtype=np.float32)
                calibrated_block[active] = calibrators[selected].predict(
                    raw_block[active]
                ).astype(np.float32)
                output[block_start:block_end, direction] = calibrated_block
            trace["directions"][label]["champion_blocks"][selected] += 1
            block_ready = True

        if block_ready:
            trace["ready_blocks"] += 1
            if trace["first_ready_index"] is None:
                trace["first_ready_index"] = block_start

    output[:, 0] = 1.0 - np.maximum(output[:, 1], output[:, 2])
    return np.clip(output, 0.0, 1.0), trace


def self_check():
    rng = np.random.default_rng(42)
    rows = 10_000
    target = rng.binomial(1, 0.35, rows).astype(np.int8)
    noise = rng.normal(0.0, 0.08, rows)
    good = np.clip(0.15 + 0.65 * target + noise, 0.01, 0.99)
    weak = np.clip(0.30 + 0.20 * target + noise, 0.01, 0.99)
    probabilities = {}
    for name, long_probability in (("good", good), ("weak", weak)):
        values = np.zeros((rows, 3), dtype=np.float32)
        values[:, 1] = long_probability
        values[:, 2] = long_probability[::-1]
        values[:, 0] = 1.0 - np.maximum(values[:, 1], values[:, 2])
        probabilities[name] = values
    settings = {
        "maturity_rows": 10,
        "window_rows": 4_000,
        "min_rows": 2_000,
        "block_rows": 500,
        "switch_margin": 0.001,
        "confirm_blocks": 2,
    }
    output, trace = rolling_champion_probabilities(
        probabilities, target, target[::-1], **settings
    )
    mutated = target.copy()
    mutated[7_000:] = 1 - mutated[7_000:]
    output_mutated, _ = rolling_champion_probabilities(
        probabilities, mutated, target[::-1], **settings
    )
    assert np.array_equal(output[:7_000], output_mutated[:7_000])
    assert not output[: trace["first_ready_index"], 1:].any()
    assert trace["ready_blocks"] > 0
    assert np.isfinite(output).all()
    print("rolling_champion_self_check_ok")


if __name__ == "__main__":
    self_check()
