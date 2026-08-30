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
from gold_expected_r_champion import _top_k_threshold
from gold_expected_r_walk_forward import (
    EXTRA_COST_POINTS,
    HISTORICAL_HOLDOUT_START,
    HORIZON,
    aggregate_score,
    fold_pass,
    make_params,
    session_mask,
)
from gold_generation11_execution_aligned import (
    QUALITY_PROFILES,
    _profit_factor,
    add_targets,
    execution_realized_metrics,
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
REPORT_JSON = PROJECT_ROOT / "gold_generation12_executable_events.json"
REPORT_MD = PROJECT_ROOT / "gold_generation12_executable_events.md"
CONFIG_FILE = PROJECT_ROOT / "gold_generation12_candidate.json"
MODEL_FILES = {
    (name, kind): PROJECT_ROOT / f"gold_generation12_{name}_{kind}_xgb.json"
    for name in EXPERT_NAMES
    for kind in ("win", "mean_r")
}
MODEL_PROFILE = {
    "n_estimators": 180,
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_child_weight": 20,
    "recency_half_life_days": 1_095.0,
    "calibration_ratio": 0.20,
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
        description="Train generation 12 on non-overlapping executable events."
    )
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def sequential_event_indices(
    eligible: np.ndarray, exit_offset: np.ndarray
) -> np.ndarray:
    if eligible.ndim != 1 or exit_offset.shape != eligible.shape:
        raise ValueError("eligible and exit_offset must be equal one-dimensional arrays")
    selected = []
    free_index = 0
    for index in np.flatnonzero(eligible & (exit_offset > 0)):
        if index < free_index:
            continue
        selected.append(index)
        free_index = index + int(exit_offset[index]) + 1
    return np.asarray(selected, dtype=np.int64)


def executable_events_by_expert(
    frame: pd.DataFrame, features: list[str]
) -> dict[str, np.ndarray]:
    _, masks = route_arrays(frame, features)
    allowed = session_mask(frame, "controlled_expanded")
    output = {}
    for direction, prefix in ((1, "long"), (2, "short")):
        direction_mask = masks[f"{prefix}_trend"] | masks[f"{prefix}_pullback"]
        reward = frame[f"{prefix.upper()}_REWARD"].to_numpy(dtype=np.float32)
        exits = frame[f"{prefix.upper()}_EXIT_OFFSET"].to_numpy(dtype=np.int16)
        indices = sequential_event_indices(
            allowed & direction_mask & np.isfinite(reward), exits
        )
        for style in ("trend", "pullback"):
            name = f"{prefix}_{style}"
            output[name] = indices[masks[name][indices]]
    return output


def _sample_weight(times: pd.Series) -> np.ndarray:
    latest = times.iloc[-1]
    age_days = (
        (latest - times).dt.total_seconds().to_numpy(dtype=np.float64) / 86_400.0
    )
    recency = 0.25 + 0.75 * np.exp(
        -math.log(2.0) * age_days / MODEL_PROFILE["recency_half_life_days"]
    )
    if not np.isfinite(recency).all() or np.any(recency <= 0.0):
        raise RuntimeError("Event weights must be finite and positive")
    return recency.astype(np.float32)


def _new_classifier(estimators: int | None = None) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
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


def _new_regressor(estimators: int | None = None) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:pseudohubererror",
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


def train_executable_experts(
    frame: pd.DataFrame, features: list[str]
) -> dict[str, dict]:
    event_indices = executable_events_by_expert(frame, features)
    models = {}
    for name, indices in event_indices.items():
        direction = "LONG" if name.startswith("long_") else "SHORT"
        reward_column = f"{direction}_REWARD"
        events = frame.iloc[indices].copy().sort_values("TIME_DT")
        if len(events) < 1_000:
            raise RuntimeError(f"{name} has only {len(events):,} executable events")
        split = int(len(events) * (1.0 - MODEL_PROFILE["calibration_ratio"]))
        fit = events.iloc[:split].copy()
        calibration = events.iloc[split:].copy()
        reward = fit[reward_column].to_numpy(dtype=np.float32)
        calibration_reward = calibration[reward_column].to_numpy(dtype=np.float32)
        win = (reward > 0.0).astype(np.int8)
        calibration_win = (calibration_reward > 0.0).astype(np.int8)
        if np.unique(win).size != 2 or np.unique(calibration_win).size != 2:
            raise RuntimeError(f"{name} executable events lack a win/loss class")
        weight = _sample_weight(fit["TIME_DT"])
        classifier = _new_classifier()
        mean_model = _new_regressor()
        x_fit = fit[features].astype(np.float32)
        classifier.fit(x_fit, win, sample_weight=weight)
        mean_model.fit(x_fit, reward, sample_weight=weight)
        raw = classifier.predict_proba(calibration[features].astype(np.float32))[:, 1]
        calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        )
        calibrator.fit(raw, calibration_win)
        average_win = float(calibration_reward[calibration_reward > 0.0].mean())
        average_loss = float(calibration_reward[calibration_reward <= 0.0].mean())
        models[name] = {
            "win": classifier,
            "mean_r": mean_model,
            "calibrator": calibrator,
            "average_win_r": average_win,
            "average_loss_r": average_loss,
            "events": len(events),
            "fit_events": len(fit),
            "calibration_events": len(calibration),
        }
        print(
            f"  {name}: events={len(events):,} fit={len(fit):,} "
            f"calibrate={len(calibration):,} win={win.mean():.2%}",
            flush=True,
        )
    return models


def predict_executable_scores(
    models: dict[str, dict], frame: pd.DataFrame, features: list[str]
) -> tuple[dict[int, np.ndarray], dict[str, dict]]:
    _, masks = route_arrays(frame, features)
    scores = {
        1: np.full(len(frame), np.nan, dtype=np.float32),
        2: np.full(len(frame), np.nan, dtype=np.float32),
    }
    diagnostics = {}
    for name, model in models.items():
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
        scores[direction][indices] = expected_r
        diagnostics[name] = {
            "rows": len(indices),
            "p_win_q50_q90_q99": np.quantile(
                probability, (0.50, 0.90, 0.99)
            ).round(4).tolist(),
            "expected_r_q50_q90_q99": np.quantile(
                expected_r, (0.50, 0.90, 0.99)
            ).round(4).tolist(),
        }
    return scores, diagnostics


def rolling_score_cash_signals(
    frame: pd.DataFrame, scores: dict[int, np.ndarray], candidate: dict
) -> tuple[np.ndarray, dict]:
    if set(scores) != {1, 2}:
        raise ValueError("scores must contain long=1 and short=2")
    row_count = len(frame)
    dates = frame["TIME_DT"].dt.date.to_numpy()
    indices = np.arange(row_count)
    allowed = session_mask(frame, candidate["session_profile"])
    rewards = {
        1: frame["LONG_REWARD"].to_numpy(dtype=np.float32),
        2: frame["SHORT_REWARD"].to_numpy(dtype=np.float32),
    }
    exits = {
        1: frame["LONG_EXIT_OFFSET"].to_numpy(dtype=np.int16),
        2: frame["SHORT_EXIT_OFFSET"].to_numpy(dtype=np.int16),
    }
    quality = QUALITY_PROFILES[candidate["quality_profile"]]
    output = np.zeros((row_count, 3), dtype=np.float32)
    trace = {
        "blocks": 0,
        "cash_blocks": {"long": 0, "short": 0},
        "active_blocks": {"long": 0, "short": 0},
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
            score = scores[direction]
            candidate_mask = (
                allowed
                & np.isfinite(score)
                & (score >= candidate["minimum_expected_r"])
            )
            threshold = _top_k_threshold(
                score,
                dates,
                candidate_mask & (indices >= history_start) & (indices < split),
                candidate["top_k_per_day"],
            )
            metrics = None
            if threshold is not None:
                metrics = execution_realized_metrics(
                    rewards[direction],
                    exits[direction],
                    score,
                    candidate_mask & (indices >= split) & (indices < history_end),
                    threshold,
                )
            if (
                metrics is None
                or metrics["trades"] < ALLOCATOR_CONFIG["minimum_quality_trades"]
                or metrics["mean_r"] < quality["minimum_mean_r"]
                or metrics["profit_factor"] < quality["minimum_profit_factor"]
            ):
                trace["cash_blocks"][label] += 1
                continue
            threshold = _top_k_threshold(
                score,
                dates,
                candidate_mask
                & (indices >= history_start)
                & (indices < history_end),
                candidate["top_k_per_day"],
            )
            if threshold is None:
                trace["cash_blocks"][label] += 1
                continue
            block_score = score[block_start:block_end]
            selected = (
                candidate_mask[block_start:block_end] & (block_score >= threshold)
            )
            output[block_start:block_end, direction][selected] = 1.0
            trace["active_blocks"][label] += 1
            trace["emitted"][label] += int(selected.sum())
    output[:, 0] = 1.0 - np.maximum(output[:, 1], output[:, 2])
    return output, trace


def candidate_stats(
    frame: pd.DataFrame,
    scores: dict[int, np.ndarray],
    candidate: dict,
    cost: float = EXTRA_COST_POINTS,
) -> tuple[dict, dict]:
    signals, trace = rolling_score_cash_signals(frame, scores, candidate)
    return evaluate_frame(make_params(candidate, cost), frame, signals), trace


def _serialize_calibrator(calibrator: IsotonicRegression) -> dict:
    return {
        "x": calibrator.X_thresholds_.tolist(),
        "y": calibrator.y_thresholds_.tolist(),
    }


def save_models(models: dict[str, dict]) -> dict:
    output = {}
    for name, model in models.items():
        model["win"].save_model(MODEL_FILES[(name, "win")])
        model["mean_r"].save_model(MODEL_FILES[(name, "mean_r")])
        output[name] = {
            "win_file": MODEL_FILES[(name, "win")].name,
            "mean_r_file": MODEL_FILES[(name, "mean_r")].name,
            "isotonic": _serialize_calibrator(model["calibrator"]),
            "average_win_r": model["average_win_r"],
            "average_loss_r": model["average_loss_r"],
            "events": model["events"],
        }
    return output


def markdown_report(report: dict) -> str:
    lines = [
        "# GOLD generation 12 executable-event Expected-R",
        "",
        "Natural P(win) and Mean-R models trained only on non-overlapping executable events.",
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
    eligible = np.ones(20, dtype=bool)
    exits = np.full(20, 3, dtype=np.int16)
    selected = sequential_event_indices(eligible, exits)
    assert selected.tolist() == [0, 4, 8, 12, 16]

    rng = np.random.default_rng(12)
    rows = 120_000
    score = rng.normal(0.0, 1.0, rows).astype(np.float32)
    frame = pd.DataFrame(
        {
            "TIME_DT": pd.date_range("2026-01-05", periods=rows, freq="min"),
            "M1_RSI": np.full(rows, 55.0),
            "LONG_REWARD": np.where(score > 2.2, 0.8, -1.0).astype(np.float32),
            "SHORT_REWARD": np.full(rows, -1.0, dtype=np.float32),
            "LONG_EXIT_OFFSET": np.full(rows, 30, dtype=np.int16),
            "SHORT_EXIT_OFFSET": np.full(rows, 30, dtype=np.int16),
        }
    )
    candidate = {
        "generation": "12_executable_events",
        "top_k_per_day": 3,
        "minimum_expected_r": -10.0,
        "session_profile": "controlled_expanded",
        "quality_profile": "quality_105",
    }
    signals, trace = rolling_score_cash_signals(
        frame, {1: score, 2: np.full(rows, -1.0, dtype=np.float32)}, candidate
    )
    assert signals[:, 1].sum() > 0
    assert signals[:, 2].sum() == 0
    assert trace["cash_blocks"]["short"] > 0

    toy = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("ABCD"))
    labels = np.tile(np.array([0, 1], dtype=np.int8), 150)
    classifier = _new_classifier(estimators=5)
    classifier.fit(toy, labels)
    regressor = _new_regressor(estimators=5)
    regressor.fit(toy, rng.normal(size=len(toy)))
    assert np.isfinite(classifier.predict_proba(toy)).all()
    assert np.isfinite(regressor.predict(toy)).all()
    print("generation12_self_check_ok")


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
            "generation": "12_executable_events",
            "top_k_per_day": top_k,
            "minimum_expected_r": minimum_r,
            "session_profile": session,
            "quality_profile": quality,
        }
        for top_k, minimum_r, session, quality in product(
            (1, 2, 3),
            (-0.05, 0.0),
            ("may_baseline", "controlled_expanded"),
            QUALITY_PROFILES,
        )
    ]
    fold_results = {index: {} for index in range(len(candidates))}
    fold_traces = {index: {} for index in range(len(candidates))}
    diagnostics = {}
    for fold_name, fold_start, fold_end in SELECTION_FOLDS:
        train = training_frame(history, fold_start)
        validation = history[
            (history["TIME_DT"] >= fold_start) & (history["TIME_DT"] < fold_end)
        ].copy().reset_index(drop=True)
        models = train_executable_experts(train, features)
        scores, diagnostics[fold_name] = predict_executable_scores(
            models, validation, features
        )
        for index, candidate in enumerate(candidates):
            stats, trace = candidate_stats(validation, scores, candidate)
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
    holdout_models = train_executable_experts(holdout_train, features)
    holdout_scores, diagnostics["2025_2026_05_holdout"] = (
        predict_executable_scores(holdout_models, holdout, features)
    )
    holdout_stats, holdout_trace = candidate_stats(
        holdout, holdout_scores, selected
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
    final_models = train_executable_experts(final_train, features)
    recent_scores, diagnostics["2026_recent"] = predict_executable_scores(
        final_models, recent, features
    )
    recent_stats, recent_trace = candidate_stats(recent, recent_scores, selected)
    recent_cost_stats, _ = candidate_stats(
        recent, recent_scores, selected, cost=10.0
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
