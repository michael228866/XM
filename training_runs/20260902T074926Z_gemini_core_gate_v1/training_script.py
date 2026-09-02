from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

import drl_trading_v2
from barrier_classifier_strategy import (
    HORIZON as LABEL_HORIZON,
    LABEL_SL_ATR,
    LABEL_TP_ATR,
    MIN_SL_PRICE as LABEL_MIN_SL_PRICE,
    MIN_TP_PRICE as LABEL_MIN_TP_PRICE,
)
from barrier_final_train import prepare_barrier_data
from barrier_research_suite import predict_positive, train_binary_model
from gold_generation11_execution_aligned import add_targets


ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "gold_long_recent_candidate_xgb.json"
GEMINI_FILE = ROOT / "gemini.py"

EXPERIMENT = "GEMINI CORE GATE OPTIMIZATION V1 — QUALITY-PRESERVING FREQUENCY"
PREVIOUS_FORWARD_CUTOFF = "2026-09-01T02:00:00Z"
PREVIOUS_FORWARD_STATUS = "contaminated_for_future_gate_selection"
RESEARCH_END = pd.Timestamp("2025-01-01 00:00:00")
LABEL_BUFFER_END = pd.Timestamp("2025-01-15 00:00:00")

FOLDS = (
    ("2018_2020", pd.Timestamp("2018-01-01"), pd.Timestamp("2021-01-01")),
    ("2021_2022", pd.Timestamp("2021-01-01"), pd.Timestamp("2023-01-01")),
    ("2023_2024", pd.Timestamp("2023-01-01"), pd.Timestamp("2025-01-01")),
)
THRESHOLDS = {
    "T0": 0.75,
    "T1": 0.72,
    "T2": 0.70,
    "T3": 0.67,
    "T4": 0.65,
}
RSI_POLICIES: dict[str, tuple[float, float] | None] = {
    "R0": (35.0, 45.0),
    "R1": None,
    "R2": (38.0, 42.0),
    "R3": (40.0, 45.0),
}
RSI_DESCRIPTIONS = {
    "R0": "exclude 35–45",
    "R1": "no 35–45 exclusion",
    "R2": "exclude 38–42",
    "R3": "exclude 40–45",
}
MARGINAL_BANDS = (
    ("B1_072_075", 0.72, 0.75),
    ("B2_070_072", 0.70, 0.72),
    ("B3_067_070", 0.67, 0.70),
    ("B4_065_067", 0.65, 0.67),
)

N_ESTIMATORS = 220
RANDOM_STATE = 42
TP_ATR = 1.3
SL_ATR = 1.6
MIN_TP_PRICE = 1.5
MIN_SL_PRICE = 0.6
MAX_HOLD_MINUTES = 90
MIN_ENTRY_RSI = 22.0
ALLOWED_HOURS = frozenset((0, 1, 2, 3, 4, 8, 9, 11, 12, 17, 18, 19, 20, 22, 23))
ALLOWED_WEEKDAYS = frozenset((0, 1, 2, 3, 4))
FALLBACK_SPREAD_POINTS = 30.0
BASE_EXTRA_COST_POINTS = 5.0
STRESS_EXTRA_COST_POINTS = 10.0
POINT = 0.01
RISK_PER_TRADE = 0.014
LOSS_COOLDOWN_MINUTES = 15
MAX_DAILY_LOSS_PCT = 0.05
ROLLING_GUARD_WINDOW = 30
ROLLING_GUARD_MIN_TRADES = 18
ROLLING_GUARD_MIN_PF = 1.15
ROLLING_GUARD_RISK_MULT = 0.50

QUALITY_FLOOR = {
    "pooled_realized_wr_min": 0.60,
    "pooled_pf_strict_min": 1.05,
    "pooled_mean_r_strict_min": 0.0,
    "pooled_pnl_strict_min": 0.0,
    "break_even_edge_strict_min": 0.0,
    "cost_stress_pf_strict_min": 1.00,
}
CATASTROPHIC_FOLD = {
    "minimum_trades": 10,
    "realized_wr_min": 0.50,
    "pf_min": 0.80,
    "mean_r_min": -0.10,
    "max_dd_r_min": -20.0,
}

CANDIDATE_FIELDS = [
    "candidate_id",
    "parameters",
    "fold",
    "threshold",
    "rsi_policy",
    "raw_qualifying_rows",
    "independent_signal_episodes",
    "eligible_signal_episodes",
    "executable_trades",
    "trades_per_day",
    "robust_trades_per_day",
    "frequency_uplift",
    "wins",
    "losses",
    "tp_first_wr",
    "realized_wr",
    "average_winner_r",
    "average_loser_r",
    "payoff_ratio",
    "break_even_wr",
    "break_even_adjusted_edge",
    "pf",
    "mean_r",
    "pnl",
    "max_dd",
    "tp_count",
    "sl_count",
    "timeout_count",
    "observed_spread_trades",
    "fallback_spread_trades",
    "nominal_cost_result",
    "cost_stress_pf",
    "cost_stress_mean_r",
    "cost_stress_pnl",
    "cost_stress_result",
    "qualification_verdict",
]
TRADE_FIELDS = [
    "analysis_type",
    "portfolio_id",
    "candidate_id",
    "fold",
    "entry_time",
    "exit_time",
    "score",
    "rsi",
    "outcome",
    "exit_type",
    "spread_points",
    "spread_observed",
    "gross_pnl_price",
    "denominator",
    "reward",
    "stress_reward",
    "risk_multiplier",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=EXPERIMENT)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def json_number(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return int(value) if isinstance(value, (int, np.integer)) else number


def effective_spread(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    raw = pd.to_numeric(frame.get("SPREAD"), errors="coerce").to_numpy(dtype=np.float64)
    observed = np.isfinite(raw) & (raw > 0.0)
    return np.where(observed, raw, FALLBACK_SPREAD_POINTS), observed


def spread_limit(atr: np.ndarray) -> np.ndarray:
    take = np.maximum(atr * TP_ATR, MIN_TP_PRICE)
    return np.minimum(100.0, np.maximum(45.0, take * 0.25 / POINT))


def preregister(run_dir: Path) -> dict[str, str]:
    manifest_path = run_dir / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "in_progress":
        raise RuntimeError("Training run is not in progress")
    if file_sha256(Path(__file__)) != manifest.get("training_script_sha256"):
        raise RuntimeError("Executed script differs from immutable run snapshot")

    operational_hashes = {
        "gemini.py": file_sha256(GEMINI_FILE),
        MODEL_FILE.name: file_sha256(MODEL_FILE),
    }
    manifest["evidence_status"] = {
        "claim": "development_chronological_oof_only",
        "previous_forward_cutoff": PREVIOUS_FORWARD_CUTOFF,
        "previous_forward_status": PREVIOUS_FORWARD_STATUS,
        "new_forward_cutoff": None,
        "post_cutoff_data_inspected": False,
    }
    manifest["search"].update(
        {
            "performed": True,
            "predefined_search_space": {
                "thresholds": THRESHOLDS,
                "rsi_policies": {
                    key: RSI_DESCRIPTIONS[key] for key in RSI_POLICIES
                },
                "minimum_entry_rsi": MIN_ENTRY_RSI,
                "total_configurations": 20,
                "marginal_probability_bands": [
                    {"id": name, "low_inclusive": low, "high_exclusive": high}
                    for name, low, high in MARGINAL_BANDS
                ],
            },
            "candidate_results_file": "candidates.csv",
            "selection_rule": (
                "Reject economic/quality failures; maximize minimum-fold executable "
                "trades/day, then pooled executable trades/day; use quality tie-breakers."
            ),
            "preregistered_at_utc": now_utc(),
        }
    )
    manifest["model"].update(
        {
            "trained": False,
            "not_applicable_reason": (
                "No operational candidate model is trained; fold replicas exist only "
                "in memory to create genuine chronological OOF probabilities."
            ),
            "oof_models_planned": True,
            "oof_models_trained": False,
            "new_operational_candidate_model_trained": False,
            "model_type": "XGBoost binary logistic fold replicas",
            "parameters": {
                "n_estimators": N_ESTIMATORS,
                "learning_rate": 0.05,
                "max_depth": 4,
                "min_child_weight": 80,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "random_state": RANDOM_STATE,
                "tree_method": "hist",
            },
            "boosted_rounds_or_estimators": N_ESTIMATORS,
            "label_definition": "BARRIER_TARGET == 1 versus all other classes",
            "horizon": LABEL_HORIZON,
            "label_tp_sl_semantics": (
                f"clean long barrier: future {LABEL_HORIZON} M1 bars; "
                f"TP=max({LABEL_TP_ATR} ATR,{LABEL_MIN_TP_PRICE}); "
                f"SL=max({LABEL_SL_ATR} ATR,{LABEL_MIN_SL_PRICE}); "
                "positive only if TP reachable and SL never reachable in horizon"
            ),
            "execution_tp_sl_semantics": (
                "long first-touch HIGH/LOW from the next bar; TP=1.3 ATR, "
                "SL=1.6 ATR, stop-first if both touch, timeout at 90 M1 bars"
            ),
            "calibration_method": "none",
        }
    )
    manifest["search"]["quality_floor"] = QUALITY_FLOOR
    manifest["search"]["catastrophic_fold_definition"] = CATASTROPHIC_FOLD
    manifest["promotion"].update(
        {
            "requested": False,
            "gate_result": "not_requested_research_only",
            "replacement_authorized": False,
            "operational_artifact_changed": False,
        }
    )
    manifest["operational_hashes_before"] = operational_hashes
    write_json(manifest_path, manifest)
    print("PREREGISTRATION_FROZEN 20 configurations", flush=True)
    return operational_hashes


def make_oof_frame(history: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    current_model = xgb.XGBClassifier()
    current_model.load_model(MODEL_FILE)
    current_features = current_model.get_booster().feature_names
    if current_features != features or len(features) != 31:
        raise RuntimeError("Historical feature order does not match the 31-feature incumbent")
    del current_model

    times = history["TIME_DT"].to_numpy(dtype="datetime64[ns]")
    close = history["CLOSE"].to_numpy(dtype=np.float64)
    atr = history["ATR"].to_numpy(dtype=np.float64)
    spread, spread_observed = effective_spread(history)
    outcomes = history["LONG_OUTCOME"].to_numpy(dtype=np.int8)
    offsets = history["LONG_EXIT_OFFSET"].to_numpy(dtype=np.int16)
    parts: list[pd.DataFrame] = []
    provenance: list[dict[str, Any]] = []

    for fold_name, fold_start, fold_end in FOLDS:
        pre_fold = history.index[history["TIME_DT"] < fold_start].to_numpy(dtype=np.int64)
        if len(pre_fold) <= LABEL_HORIZON:
            raise RuntimeError(f"Insufficient training history for {fold_name}")
        train_indices = pre_fold[:-LABEL_HORIZON]
        score_indices = history.index[
            (history["TIME_DT"] >= fold_start) & (history["TIME_DT"] < fold_end)
        ].to_numpy(dtype=np.int64)
        if not len(score_indices):
            raise RuntimeError(f"Empty scoring fold: {fold_name}")
        maturity_index = int(train_indices[-1]) + LABEL_HORIZON
        if maturity_index >= int(score_indices[0]):
            raise RuntimeError(f"Label interval overlaps scoring fold: {fold_name}")
        if history["TIME_DT"].iat[int(train_indices[-1])] >= fold_start:
            raise RuntimeError(f"Training chronology failed: {fold_name}")

        train = history.loc[train_indices]
        score_frame = history.loc[score_indices]
        print(
            f"Training {fold_name}: train={len(train):,} "
            f"{train['TIME_DT'].iat[0]}..{train['TIME_DT'].iat[-1]} "
            f"score={len(score_frame):,} {fold_start}..{fold_end}",
            flush=True,
        )
        model = train_binary_model(train, features, 1, N_ESTIMATORS)
        score = predict_positive(model, score_frame, features)
        del model, train, score_frame
        gc.collect()

        offset = offsets[score_indices]
        exit_indices = score_indices + offset.astype(np.int64)
        mature = (
            (outcomes[score_indices] >= 0)
            & (offset > 0)
            & (exit_indices < len(history))
        )
        if not mature.all():
            raise RuntimeError(f"Immature outcomes entered {fold_name}")
        take = np.maximum(atr[score_indices] * TP_ATR, MIN_TP_PRICE)
        stop = np.maximum(atr[score_indices] * SL_ATR, MIN_SL_PRICE)
        outcome = outcomes[score_indices]
        gross = np.where(
            outcome == 1,
            take,
            np.where(outcome == 2, -stop, close[exit_indices] - close[score_indices]),
        )
        denominator = stop + spread[score_indices] * POINT
        nominal_reward = (
            gross - (spread[score_indices] + BASE_EXTRA_COST_POINTS) * POINT
        ) / denominator
        stress_reward = (
            gross - (spread[score_indices] + STRESS_EXTRA_COST_POINTS) * POINT
        ) / denominator
        timestamp = pd.Series(times[score_indices])
        rsi = history["M1_RSI"].to_numpy(dtype=np.float64)[score_indices]
        part = pd.DataFrame(
            {
                "global_index": score_indices,
                "fold": fold_name,
                "time": timestamp,
                "score": score,
                "rsi": rsi,
                "session_ok": (
                    timestamp.dt.hour.isin(ALLOWED_HOURS).to_numpy()
                    & timestamp.dt.dayofweek.isin(ALLOWED_WEEKDAYS).to_numpy()
                ),
                "spread_points": spread[score_indices],
                "spread_observed": spread_observed[score_indices],
                "spread_ok": spread[score_indices] <= spread_limit(atr[score_indices]),
                "outcome": outcome,
                "exit_offset": offset,
                "exit_time": pd.Series(times[exit_indices]),
                "gross_pnl_price": gross,
                "denominator": denominator,
                "reward": nominal_reward,
                "stress_reward": stress_reward,
            }
        )
        parts.append(part)
        provenance.append(
            {
                "fold": fold_name,
                "train_start": train_indices.size and history["TIME_DT"].iat[int(train_indices[0])].isoformat(),
                "train_end": history["TIME_DT"].iat[int(train_indices[-1])].isoformat(),
                "train_rows": int(len(train_indices)),
                "label_horizon_rows": LABEL_HORIZON,
                "latest_training_label_bar": history["TIME_DT"].iat[maturity_index].isoformat(),
                "score_start": history["TIME_DT"].iat[int(score_indices[0])].isoformat(),
                "score_end": history["TIME_DT"].iat[int(score_indices[-1])].isoformat(),
                "score_rows": int(len(score_indices)),
                "chronology_assertion": bool(maturity_index < int(score_indices[0])),
            }
        )
        print(
            f"OOF {fold_name}: p75={np.quantile(score, 0.75):.4f} "
            f"p95={np.quantile(score, 0.95):.4f} max={score.max():.4f}",
            flush=True,
        )

    oof = pd.concat(parts, ignore_index=True)
    if not oof["time"].is_monotonic_increasing:
        raise RuntimeError("OOF rows are not chronological")
    return oof, provenance


def episode_mask(oof: pd.DataFrame, threshold: float, high: float | None = None) -> np.ndarray:
    score = oof["score"].to_numpy(dtype=np.float64)
    active = np.isfinite(score) & (score >= threshold)
    times = oof["time"].to_numpy(dtype="datetime64[s]").astype(np.int64)
    folds = oof["fold"].to_numpy()
    reset = np.r_[
        True,
        (times[1:] - times[:-1] > 120) | (folds[1:] != folds[:-1]),
    ]
    previous_active = np.r_[False, active[:-1]] & ~reset
    episodes = active & ~previous_active
    if high is not None:
        episodes &= score < high
    return episodes


def rolling_risk_multiplier(trades: list[dict[str, Any]]) -> float:
    if len(trades) < ROLLING_GUARD_MIN_TRADES:
        return 1.0
    values = np.asarray(
        [trade["reward"] for trade in trades[-ROLLING_GUARD_WINDOW:]],
        dtype=np.float64,
    )
    loss = -float(values[values < 0.0].sum())
    pf = math.inf if loss <= 0.0 else float(values[values > 0.0].sum() / loss)
    return ROLLING_GUARD_RISK_MULT if pf < ROLLING_GUARD_MIN_PF else 1.0


def execute(
    oof: pd.DataFrame,
    threshold: float,
    rsi_policy: str,
    high: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    episodes = episode_mask(oof, threshold, high)
    score = oof["score"].to_numpy(dtype=np.float64)
    raw_mask = np.isfinite(score) & (score >= threshold)
    if high is not None:
        raw_mask &= score < high
    rsi = oof["rsi"].to_numpy(dtype=np.float64)
    excluded = RSI_POLICIES[rsi_policy]
    session_ok = oof["session_ok"].to_numpy(dtype=bool)
    spread_ok = oof["spread_ok"].to_numpy(dtype=bool)
    static_audit = {
        name: {
            "raw_qualifying_rows": int(raw_mask[oof["fold"].to_numpy() == name].sum()),
            "independent_signal_episodes": int(episodes[oof["fold"].to_numpy() == name].sum()),
            "session_blocked": 0,
            "rsi_floor_blocked": 0,
            "rsi_policy_blocked": 0,
            "spread_blocked": 0,
            "eligible_signal_episodes": 0,
            "position_blocked": 0,
            "loss_cooldown_blocked": 0,
            "daily_loss_blocked": 0,
        }
        for name, _, _ in FOLDS
    }
    eligible: list[int] = []
    for index in np.flatnonzero(episodes):
        fold = str(oof["fold"].iat[index])
        audit = static_audit[fold]
        if not session_ok[index]:
            audit["session_blocked"] += 1
        elif rsi[index] < MIN_ENTRY_RSI:
            audit["rsi_floor_blocked"] += 1
        elif excluded is not None and excluded[0] <= rsi[index] <= excluded[1]:
            audit["rsi_policy_blocked"] += 1
        elif not spread_ok[index]:
            audit["spread_blocked"] += 1
        else:
            audit["eligible_signal_episodes"] += 1
            eligible.append(int(index))

    trades: list[dict[str, Any]] = []
    free_after: pd.Timestamp | None = None
    last_loss_exit: pd.Timestamp | None = None
    daily_return: dict[object, float] = {}
    for index in eligible:
        fold = str(oof["fold"].iat[index])
        entry_time = pd.Timestamp(oof["time"].iat[index])
        audit = static_audit[fold]
        if free_after is not None and entry_time <= free_after:
            audit["position_blocked"] += 1
            continue
        if (
            last_loss_exit is not None
            and entry_time < last_loss_exit + pd.Timedelta(minutes=LOSS_COOLDOWN_MINUTES)
        ):
            audit["loss_cooldown_blocked"] += 1
            continue
        if daily_return.get(entry_time.date(), 0.0) <= -MAX_DAILY_LOSS_PCT:
            audit["daily_loss_blocked"] += 1
            continue
        risk_multiplier = rolling_risk_multiplier(trades)
        reward = float(oof["reward"].iat[index])
        exit_time = pd.Timestamp(oof["exit_time"].iat[index])
        outcome = int(oof["outcome"].iat[index])
        trade = {
            "fold": fold,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "score": float(oof["score"].iat[index]),
            "rsi": float(oof["rsi"].iat[index]),
            "outcome": outcome,
            "exit_type": {0: "timeout", 1: "tp", 2: "sl"}[outcome],
            "spread_points": float(oof["spread_points"].iat[index]),
            "spread_observed": bool(oof["spread_observed"].iat[index]),
            "gross_pnl_price": float(oof["gross_pnl_price"].iat[index]),
            "denominator": float(oof["denominator"].iat[index]),
            "reward": reward,
            "stress_reward": float(oof["stress_reward"].iat[index]),
            "risk_multiplier": risk_multiplier,
        }
        trades.append(trade)
        free_after = exit_time
        daily_return[exit_time.date()] = daily_return.get(exit_time.date(), 0.0) + (
            RISK_PER_TRADE * risk_multiplier * reward
        )
        if reward <= 0.0:
            last_loss_exit = exit_time

    return trades, static_audit


def reward_metrics(values: np.ndarray) -> dict[str, Any]:
    if not len(values):
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "realized_wr": 0.0,
            "average_winner_r": None,
            "average_loser_r": None,
            "payoff_ratio": None,
            "break_even_wr": None,
            "break_even_adjusted_edge": None,
            "pf": 0.0,
            "mean_r": 0.0,
            "pnl": 0.0,
            "max_dd": 0.0,
        }
    winners = values[values > 0.0]
    losers = values[values <= 0.0]
    gains = float(winners.sum())
    loss = -float(losers.sum())
    average_winner = float(winners.mean()) if len(winners) else None
    average_loser = float(losers.mean()) if len(losers) else None
    payoff = (
        average_winner / abs(average_loser)
        if average_winner is not None and average_loser not in (None, 0.0)
        else None
    )
    break_even = 1.0 / (1.0 + payoff) if payoff is not None else None
    realized_wr = float(len(winners) / len(values))
    equity = np.cumsum(values)
    peaks = np.maximum.accumulate(np.maximum(equity, 0.0))
    return {
        "trades": int(len(values)),
        "wins": int(len(winners)),
        "losses": int(len(losers)),
        "realized_wr": realized_wr,
        "average_winner_r": average_winner,
        "average_loser_r": average_loser,
        "payoff_ratio": payoff,
        "break_even_wr": break_even,
        "break_even_adjusted_edge": (
            realized_wr - break_even if break_even is not None else None
        ),
        "pf": math.inf if loss <= 0.0 and gains > 0.0 else (gains / loss if loss > 0.0 else 0.0),
        "mean_r": float(values.mean()),
        "pnl": float(values.sum()),
        "max_dd": float(np.min(equity - peaks)),
    }


def trade_metrics(trades: list[dict[str, Any]], days: int) -> dict[str, Any]:
    nominal = reward_metrics(np.asarray([trade["reward"] for trade in trades], dtype=np.float64))
    stress = reward_metrics(np.asarray([trade["stress_reward"] for trade in trades], dtype=np.float64))
    nominal.update(
        {
            "trades_per_day": nominal["trades"] / max(days, 1),
            "tp_first_wr": (
                sum(trade["outcome"] == 1 for trade in trades) / len(trades)
                if trades
                else 0.0
            ),
            "tp_count": int(sum(trade["outcome"] == 1 for trade in trades)),
            "sl_count": int(sum(trade["outcome"] == 2 for trade in trades)),
            "timeout_count": int(sum(trade["outcome"] == 0 for trade in trades)),
            "observed_spread_trades": int(sum(trade["spread_observed"] for trade in trades)),
            "fallback_spread_trades": int(sum(not trade["spread_observed"] for trade in trades)),
            "cost_stress_pf": stress["pf"],
            "cost_stress_mean_r": stress["mean_r"],
            "cost_stress_pnl": stress["pnl"],
        }
    )
    return nominal


def fold_days(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return int((end - start).days)


def candidate_result(oof: pd.DataFrame, threshold: float, rsi_policy: str) -> dict[str, Any]:
    trades, audit = execute(oof, threshold, rsi_policy)
    folds: dict[str, dict[str, Any]] = {}
    for name, start, end in FOLDS:
        fold_trades = [trade for trade in trades if trade["fold"] == name]
        metrics = trade_metrics(fold_trades, fold_days(start, end))
        metrics.update(audit[name])
        folds[name] = metrics
    pooled = trade_metrics(trades, sum(fold_days(start, end) for _, start, end in FOLDS))
    for key in audit[next(iter(audit))]:
        pooled[key] = int(sum(value[key] for value in audit.values()))
    return {"folds": folds, "pooled": pooled, "trades": trades, "audit": audit}


def catastrophic(fold: dict[str, Any]) -> bool:
    pf = float(fold["pf"])
    return bool(
        fold["trades"] < CATASTROPHIC_FOLD["minimum_trades"]
        or fold["realized_wr"] < CATASTROPHIC_FOLD["realized_wr_min"]
        or pf < CATASTROPHIC_FOLD["pf_min"]
        or fold["mean_r"] < CATASTROPHIC_FOLD["mean_r_min"]
        or fold["max_dd"] < CATASTROPHIC_FOLD["max_dd_r_min"]
    )


def qualify(result: dict[str, Any]) -> tuple[bool, list[str]]:
    pooled = result["pooled"]
    reasons: list[str] = []
    if pooled["pf"] <= 1.0:
        reasons.append("PF<=1")
    if pooled["mean_r"] <= 0.0:
        reasons.append("Mean-R<=0")
    if pooled["pnl"] <= 0.0:
        reasons.append("PnL<=0")
    if (pooled["break_even_adjusted_edge"] or -1.0) <= 0.0:
        reasons.append("BE-edge<=0")
    if pooled["realized_wr"] < QUALITY_FLOOR["pooled_realized_wr_min"]:
        reasons.append("WR<60%")
    if pooled["pf"] <= QUALITY_FLOOR["pooled_pf_strict_min"]:
        reasons.append("PF<=1.05")
    if pooled["cost_stress_pf"] <= QUALITY_FLOOR["cost_stress_pf_strict_min"]:
        reasons.append("stress-PF<=1")
    catastrophic_folds = [name for name, fold in result["folds"].items() if catastrophic(fold)]
    if catastrophic_folds:
        reasons.append("catastrophic-fold=" + ",".join(catastrophic_folds))
    return not reasons, reasons


def pareto_ids(results: dict[str, dict[str, Any]]) -> list[str]:
    frontier = []
    for candidate_id, result in results.items():
        candidate = result["pooled"]
        dominated = False
        for other_id, other_result in results.items():
            if other_id == candidate_id:
                continue
            other = other_result["pooled"]
            if (
                other["realized_wr"] >= candidate["realized_wr"]
                and other["trades_per_day"] >= candidate["trades_per_day"]
                and (
                    other["realized_wr"] > candidate["realized_wr"]
                    or other["trades_per_day"] > candidate["trades_per_day"]
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate_id)
    return sorted(frontier)


def metric_subset(trades: list[dict[str, Any]], fold: str | None = None) -> dict[str, Any]:
    values = trades if fold is None else [trade for trade in trades if trade["fold"] == fold]
    days = (
        sum(fold_days(start, end) for _, start, end in FOLDS)
        if fold is None
        else next(fold_days(start, end) for name, start, end in FOLDS if name == fold)
    )
    return trade_metrics(values, days)


def write_candidates(
    run_dir: Path,
    results: dict[str, dict[str, Any]],
    control_id: str,
) -> None:
    control = results[control_id]
    with (run_dir / "candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        for candidate_id, result in results.items():
            threshold_id, rsi_policy = candidate_id.split("_")
            threshold = THRESHOLDS[threshold_id]
            robust_tpd = min(fold["trades_per_day"] for fold in result["folds"].values())
            for fold_name in (*[name for name, _, _ in FOLDS], "pooled"):
                metrics = result[fold_name] if fold_name == "pooled" else result["folds"][fold_name]
                control_metrics = control[fold_name] if fold_name == "pooled" else control["folds"][fold_name]
                uplift = (
                    metrics["trades_per_day"] / control_metrics["trades_per_day"] - 1.0
                    if control_metrics["trades_per_day"] > 0.0
                    else None
                )
                verdict = (
                    "QUALITY_PASS"
                    if fold_name == "pooled" and result["quality_pass"]
                    else (
                        "QUALITY_FAIL:" + ";".join(result["failure_reasons"])
                        if fold_name == "pooled"
                        else ("CATASTROPHIC" if catastrophic(metrics) else "DIAGNOSTIC")
                    )
                )
                row = {
                    "candidate_id": candidate_id,
                    "parameters": json.dumps(
                        {"threshold": threshold, "rsi_policy": RSI_DESCRIPTIONS[rsi_policy]},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "fold": fold_name,
                    "threshold": threshold,
                    "rsi_policy": rsi_policy,
                    "raw_qualifying_rows": metrics["raw_qualifying_rows"],
                    "independent_signal_episodes": metrics["independent_signal_episodes"],
                    "eligible_signal_episodes": metrics["eligible_signal_episodes"],
                    "executable_trades": metrics["trades"],
                    "trades_per_day": metrics["trades_per_day"],
                    "robust_trades_per_day": robust_tpd,
                    "frequency_uplift": uplift,
                    "wins": metrics["wins"],
                    "losses": metrics["losses"],
                    "tp_first_wr": metrics["tp_first_wr"],
                    "realized_wr": metrics["realized_wr"],
                    "average_winner_r": metrics["average_winner_r"],
                    "average_loser_r": metrics["average_loser_r"],
                    "payoff_ratio": metrics["payoff_ratio"],
                    "break_even_wr": metrics["break_even_wr"],
                    "break_even_adjusted_edge": metrics["break_even_adjusted_edge"],
                    "pf": metrics["pf"],
                    "mean_r": metrics["mean_r"],
                    "pnl": metrics["pnl"],
                    "max_dd": metrics["max_dd"],
                    "tp_count": metrics["tp_count"],
                    "sl_count": metrics["sl_count"],
                    "timeout_count": metrics["timeout_count"],
                    "observed_spread_trades": metrics["observed_spread_trades"],
                    "fallback_spread_trades": metrics["fallback_spread_trades"],
                    "nominal_cost_result": "PASS" if metrics["pf"] > 1 and metrics["mean_r"] > 0 else "FAIL",
                    "cost_stress_pf": metrics["cost_stress_pf"],
                    "cost_stress_mean_r": metrics["cost_stress_mean_r"],
                    "cost_stress_pnl": metrics["cost_stress_pnl"],
                    "cost_stress_result": "PASS" if metrics["cost_stress_pf"] > 1 else "FAIL",
                    "qualification_verdict": verdict,
                }
                writer.writerow({key: json_number(value) if isinstance(value, (float, np.floating)) else value for key, value in row.items()})


def write_trade_ledger(
    run_dir: Path,
    candidate_results: dict[str, dict[str, Any]],
    marginal_results: dict[str, dict[str, Any]],
) -> None:
    rows = []
    for candidate_id, result in candidate_results.items():
        for trade in result["trades"]:
            rows.append(
                {
                    "analysis_type": "candidate",
                    "portfolio_id": candidate_id,
                    "candidate_id": candidate_id,
                    **trade,
                }
            )
    for band_id, result in marginal_results.items():
        for trade in result["trades"]:
            rows.append(
                {
                    "analysis_type": "threshold_marginal",
                    "portfolio_id": band_id,
                    "candidate_id": "",
                    **trade,
                }
            )
    frame = pd.DataFrame(rows, columns=TRADE_FIELDS)
    frame.to_csv(run_dir / "trade_ledger.csv.gz", index=False, compression="gzip")


def serialize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            result[key] = serialize_metrics(value)
        elif isinstance(value, list):
            result[key] = [serialize_metrics(item) if isinstance(item, dict) else item for item in value]
        elif isinstance(value, (bool, np.bool_)):
            result[key] = bool(value)
        elif isinstance(value, (float, np.floating)):
            result[key] = json_number(value)
        elif isinstance(value, (int, np.integer)):
            result[key] = int(value)
        else:
            result[key] = value
    return result


def markdown_report(metrics: dict[str, Any]) -> str:
    summary = metrics["summary"]
    results = metrics["candidates"]
    control = results["T0_R0"]["pooled"]
    selected_id = summary["selected_candidate_id"]
    selected = results[selected_id]["pooled"] if selected_id else None
    lines = [
        f"# {EXPERIMENT}",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Development-only genuine chronological OOF experiment. The old forward cutoff is contaminated; no untouched final claim is made.",
        "",
        "## Pooled 20-configuration comparison",
        "",
        "| Candidate | Threshold | RSI | Trades | Trades/day | Uplift | WR | PF | Mean-R | PnL-R | Max DD-R | Stress PF | Gate |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for candidate_id in sorted(results, key=lambda key: (results[key]["threshold"], key), reverse=True):
        result = results[candidate_id]
        pooled = result["pooled"]
        lines.append(
            f"| {candidate_id} | {result['threshold']:.2f} | {result['rsi_policy']} | "
            f"{pooled['trades']} | {pooled['trades_per_day']:.4f} | "
            f"{result['frequency_uplift']:.1%} | {pooled['realized_wr']:.2%} | "
            f"{pooled['pf'] if pooled['pf'] is not None else 'inf'} | {pooled['mean_r']:.4f} | "
            f"{pooled['pnl']:.2f} | {pooled['max_dd']:.2f} | "
            f"{pooled['cost_stress_pf'] if pooled['cost_stress_pf'] is not None else 'inf'} | "
            f"{'PASS' if result['quality_pass'] else 'FAIL'} |"
        )

    reference_id = selected_id or summary["diagnostic_reference_id"]
    lines.extend(
        [
            "",
            f"Quality-pass count: **{summary['quality_pass_count']} / 20**.",
            f"Selected candidate: **{selected_id or 'none'}**.",
            "",
            f"## Fold stability — {reference_id}",
            "",
            "| Fold | Trades | Trades/day | TP-first WR | Realized WR | PF | Mean-R | PnL-R | Max DD-R | Stress PF |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for fold_name, fold in results[reference_id]["folds"].items():
        lines.append(
            f"| {fold_name} | {fold['trades']} | {fold['trades_per_day']:.4f} | "
            f"{fold['tp_first_wr']:.2%} | {fold['realized_wr']:.2%} | "
            f"{fold['pf'] if fold['pf'] is not None else 'inf'} | {fold['mean_r']:.4f} | "
            f"{fold['pnl']:.2f} | {fold['max_dd']:.2f} | "
            f"{fold['cost_stress_pf'] if fold['cost_stress_pf'] is not None else 'inf'} |"
        )

    lines.extend(
        [
            "",
            "## Threshold marginal cohorts (R0)",
            "",
            "| Band | Episodes | Trades | WR | PF | Mean-R | PnL-R | Avg win R | Avg loss R | Verdict |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for band_id, value in metrics["threshold_marginal_cohorts"].items():
        pooled = value["pooled"]
        lines.append(
            f"| {band_id} | {pooled['independent_signal_episodes']} | {pooled['trades']} | "
            f"{pooled['realized_wr']:.2%} | {pooled['pf'] if pooled['pf'] is not None else 'inf'} | "
            f"{pooled['mean_r']:.4f} | {pooled['pnl']:.2f} | "
            f"{pooled['average_winner_r'] if pooled['average_winner_r'] is not None else 'n/a'} | "
            f"{pooled['average_loser_r'] if pooled['average_loser_r'] is not None else 'n/a'} | {value['verdict']} |"
        )

    lines.extend(
        [
            "",
            "## RSI marginal value",
            "",
            "| Threshold | Alternative | Recovered episodes | Recovered trades | Winners | Losers | WR | PF | Mean-R | PnL-R |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in metrics["rsi_marginal_analysis"]:
        recovered = item["recovered_metrics"]
        lines.append(
            f"| {item['threshold']:.2f} | {item['alternative_policy']} | "
            f"{item['independent_episodes_recovered']} | {item['unique_trades_recovered']} | "
            f"{recovered['wins']} | {recovered['losses']} | {recovered['realized_wr']:.2%} | "
            f"{recovered['pf'] if recovered['pf'] is not None else 'inf'} | "
            f"{recovered['mean_r']:.4f} | {recovered['pnl']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Pareto frontier",
            "",
            ", ".join(metrics["pareto_frontier"]) or "none",
            "",
            "## Required answers",
            "",
        ]
    )
    for index, answer in enumerate(metrics["answers"], start=1):
        lines.append(f"{index}. {answer}")
    lines.extend(
        [
            "",
            "## Operational safety",
            "",
            "`gemini.py` and `gold_long_recent_candidate_xgb.json` were hash-checked and unchanged by the research execution.",
            "",
            f"Control: {control['trades']} trades, {control['trades_per_day']:.4f}/day, "
            f"WR {control['realized_wr']:.2%}, PF {control['pf']}, Mean-R {control['mean_r']:.4f}.",
        ]
    )
    if selected is not None:
        lines.append(
            f"Frozen shadow candidate: {selected_id}, {selected['trades_per_day']:.4f}/day, "
            f"WR {selected['realized_wr']:.2%}, PF {selected['pf']}, Mean-R {selected['mean_r']:.4f}."
        )
    return "\n".join(lines) + "\n"


def self_check() -> None:
    sample = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:01", "2026-01-01 00:05"]),
            "fold": ["x", "x", "x"],
            "score": [0.76, 0.77, 0.78],
        }
    )
    assert episode_mask(sample, 0.75).tolist() == [True, False, True]
    reward = reward_metrics(np.asarray([1.0, -1.0]))
    assert reward["trades"] == 2 and reward["pf"] == 1.0
    assert len(THRESHOLDS) * len(RSI_POLICIES) == 20
    print("SELF_CHECK_OK", flush=True)


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0
    if args.run_dir is None:
        raise ValueError("--run-dir is required")
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if not MODEL_FILE.is_file() or not GEMINI_FILE.is_file():
        raise FileNotFoundError("Operational model or gemini.py is missing")

    operational_hashes = preregister(run_dir)
    self_check()
    drl_trading_v2.DATA_DIR = str(ROOT)
    history, features = prepare_barrier_data()
    history = history.loc[history["TIME_DT"] < LABEL_BUFFER_END].copy().reset_index(drop=True)
    history = add_targets(history)
    print(
        f"History loaded: {len(history):,} rows "
        f"{history['TIME_DT'].iat[0]}..{history['TIME_DT'].iat[-1]}",
        flush=True,
    )
    oof, provenance = make_oof_frame(history, features)
    del history
    gc.collect()

    qualifying = oof["score"] >= min(THRESHOLDS.values())
    qualifying_frame = oof.loc[
        qualifying,
        [
            "global_index",
            "fold",
            "time",
            "score",
            "rsi",
            "session_ok",
            "spread_points",
            "spread_observed",
            "spread_ok",
            "outcome",
            "exit_offset",
            "exit_time",
            "reward",
            "stress_reward",
        ],
    ].copy()
    previous_score = oof["score"].shift(1)
    qualifying_frame["previous_score"] = previous_score.loc[qualifying].to_numpy()
    qualifying_frame.to_csv(
        run_dir / "oof_qualifying_rows.csv.gz", index=False, compression="gzip"
    )
    fold_codes = pd.Categorical(
        oof["fold"], categories=[name for name, _, _ in FOLDS]
    ).codes.astype(np.int8)
    np.savez_compressed(
        run_dir / "oof_predictions.npz",
        global_index=oof["global_index"].to_numpy(dtype=np.int64),
        time_ns=oof["time"].to_numpy(dtype="datetime64[ns]").astype(np.int64),
        fold_code=fold_codes,
        score=oof["score"].to_numpy(dtype=np.float32),
    )
    write_json(run_dir / "oof_model_provenance.json", {"folds": provenance, "features": features})

    candidate_results: dict[str, dict[str, Any]] = {}
    for threshold_id, threshold in THRESHOLDS.items():
        for rsi_policy in RSI_POLICIES:
            candidate_id = f"{threshold_id}_{rsi_policy}"
            result = candidate_result(oof, threshold, rsi_policy)
            passed, reasons = qualify(result)
            result.update(
                {
                    "candidate_id": candidate_id,
                    "threshold": threshold,
                    "rsi_policy": rsi_policy,
                    "quality_pass": passed,
                    "failure_reasons": reasons,
                    "robust_trades_per_day": min(
                        fold["trades_per_day"] for fold in result["folds"].values()
                    ),
                }
            )
            candidate_results[candidate_id] = result
            print(
                f"{candidate_id}: trades={result['pooled']['trades']} "
                f"tpd={result['pooled']['trades_per_day']:.4f} "
                f"WR={result['pooled']['realized_wr']:.2%} "
                f"PF={result['pooled']['pf']:.3f} pass={passed}",
                flush=True,
            )

    control_id = "T0_R0"
    control_tpd = candidate_results[control_id]["pooled"]["trades_per_day"]
    for result in candidate_results.values():
        result["frequency_uplift"] = (
            result["pooled"]["trades_per_day"] / control_tpd - 1.0
            if control_tpd > 0.0
            else 0.0
        )
    qualified_results = [result for result in candidate_results.values() if result["quality_pass"]]
    qualified_results.sort(
        key=lambda result: (
            result["robust_trades_per_day"],
            result["pooled"]["trades_per_day"],
            result["pooled"]["realized_wr"],
            result["pooled"]["pf"],
            result["pooled"]["mean_r"],
            result["pooled"]["break_even_adjusted_edge"] or -math.inf,
            result["pooled"]["max_dd"],
            result["pooled"]["cost_stress_pf"],
        ),
        reverse=True,
    )
    selected = qualified_results[0] if qualified_results else None
    diagnostic = max(
        candidate_results.values(),
        key=lambda result: (
            result["pooled"]["pf"] > 1.0,
            result["pooled"]["realized_wr"],
            result["pooled"]["pf"],
            result["pooled"]["trades_per_day"],
        ),
    )

    marginal_results: dict[str, dict[str, Any]] = {}
    marginal_metrics: dict[str, dict[str, Any]] = {}
    for band_id, low, high in MARGINAL_BANDS:
        trades, audit = execute(oof, low, "R0", high=high)
        folds = {}
        for fold_name, start, end in FOLDS:
            values = [trade for trade in trades if trade["fold"] == fold_name]
            fold_metric = trade_metrics(values, fold_days(start, end))
            fold_metric.update(audit[fold_name])
            folds[fold_name] = fold_metric
        pooled = trade_metrics(trades, sum(fold_days(start, end) for _, start, end in FOLDS))
        for key in audit[next(iter(audit))]:
            pooled[key] = int(sum(value[key] for value in audit.values()))
        verdict = "POSITIVE_EXPECTANCY" if pooled["pf"] > 1 and pooled["mean_r"] > 0 and pooled["pnl"] > 0 else "NEGATIVE_EXPECTANCY"
        marginal_results[band_id] = {"trades": trades}
        marginal_metrics[band_id] = {
            "low_inclusive": low,
            "high_exclusive": high,
            "folds": folds,
            "pooled": pooled,
            "verdict": verdict,
        }

    rsi_analysis = []
    for threshold_id, threshold in THRESHOLDS.items():
        baseline = candidate_results[f"{threshold_id}_R0"]
        baseline_entries = {trade["entry_time"] for trade in baseline["trades"]}
        baseline_episodes = {
            oof["time"].iat[index]
            for index in np.flatnonzero(episode_mask(oof, threshold))
            if oof["session_ok"].iat[index]
            and oof["rsi"].iat[index] >= MIN_ENTRY_RSI
            and not (35.0 <= oof["rsi"].iat[index] <= 45.0)
            and oof["spread_ok"].iat[index]
        }
        for alternative in ("R1", "R2", "R3"):
            compared = candidate_results[f"{threshold_id}_{alternative}"]
            recovered = [
                trade for trade in compared["trades"] if trade["entry_time"] not in baseline_entries
            ]
            alternative_excluded = RSI_POLICIES[alternative]
            alternative_episodes = {
                oof["time"].iat[index]
                for index in np.flatnonzero(episode_mask(oof, threshold))
                if oof["session_ok"].iat[index]
                and oof["rsi"].iat[index] >= MIN_ENTRY_RSI
                and (
                    alternative_excluded is None
                    or not (
                        alternative_excluded[0]
                        <= oof["rsi"].iat[index]
                        <= alternative_excluded[1]
                    )
                )
                and oof["spread_ok"].iat[index]
            }
            recovered_metrics = metric_subset(recovered)
            rsi_analysis.append(
                {
                    "threshold_id": threshold_id,
                    "threshold": threshold,
                    "baseline_policy": "R0",
                    "alternative_policy": alternative,
                    "independent_episodes_recovered": len(alternative_episodes - baseline_episodes),
                    "unique_trades_recovered": len(recovered),
                    "winners_removed_by_R0": recovered_metrics["wins"],
                    "losers_removed_by_R0": recovered_metrics["losses"],
                    "net_expectancy_effect_of_R0": -recovered_metrics["pnl"],
                    "recovered_metrics": recovered_metrics,
                }
            )

    frontier = pareto_ids(candidate_results)
    for result in candidate_results.values():
        result["pareto_member"] = result["candidate_id"] in frontier

    def any_pass(threshold_id: str) -> bool:
        return any(
            result["quality_pass"]
            for candidate_id, result in candidate_results.items()
            if candidate_id.startswith(threshold_id + "_")
        )

    current_removed = next(
        item for item in rsi_analysis if item["threshold_id"] == "T0" and item["alternative_policy"] == "R1"
    )
    recovered = current_removed["recovered_metrics"]
    viable_uplifts = [result["frequency_uplift"] for result in candidate_results.values() if result["quality_pass"]]
    pf105_uplifts = [
        result["frequency_uplift"]
        for result in candidate_results.values()
        if result["pooled"]["pf"] > 1.05 and result["pooled"]["mean_r"] > 0
    ]
    pf115_uplifts = [
        result["frequency_uplift"]
        for result in candidate_results.values()
        if result["pooled"]["pf"] >= 1.15 and result["pooled"]["mean_r"] > 0
    ]
    positive_marginals = [
        band for band, result in marginal_metrics.items() if result["verdict"] == "POSITIVE_EXPECTANCY"
    ]
    selected_id = selected["candidate_id"] if selected else None
    answers = [
        f"T0_R0 Pareto efficient: {'yes' if control_id in frontier else 'no'}.",
        f"Primary quality gate pass count: {len(qualified_results)} of 20.",
        f"Threshold 0.72 viable: {'yes' if any_pass('T1') else 'no'}.",
        f"Threshold 0.70 viable: {'yes' if any_pass('T2') else 'no'}.",
        f"Threshold 0.67 viable: {'yes' if any_pass('T3') else 'no'}.",
        f"Threshold 0.65 viable: {'yes' if any_pass('T4') else 'no'}.",
        (
            "At T0, RSI 35–45 removes more losers than winners."
            if current_removed["losers_removed_by_R0"] > current_removed["winners_removed_by_R0"]
            else "At T0, RSI 35–45 does not remove more losers than winners."
        ),
        f"T0 trades removed by RSI 35–45: PF={recovered['pf']}, Mean-R={recovered['mean_r']:.4f}.",
        "Narrowing/removal is evaluated in the RSI marginal table; no alternative is accepted unless its full candidate passes the primary gate.",
        f"Highest robust trades/day with WR >=60% and all gates: {selected_id or 'none'}.",
        f"Maximum frequency uplift with PF >1.05 and positive Mean-R: {max(pf105_uplifts, default=0.0):.1%}.",
        f"Maximum frequency uplift with PF >=1.15 and positive Mean-R: {max(pf115_uplifts, default=0.0):.1%}.",
        f"Positive-expectancy marginal lower-threshold bands: {', '.join(positive_marginals) if positive_marginals else 'none'}.",
        f"At least +25% quality-preserving uplift: {'yes' if max(viable_uplifts, default=0.0) >= 0.25 else 'no'}.",
        f"At least +50% quality-preserving uplift: {'yes' if max(viable_uplifts, default=0.0) >= 0.50 else 'no'}.",
        f"Approximately +100% quality-preserving uplift: {'yes' if max(viable_uplifts, default=0.0) >= 1.00 else 'no'}.",
        (
            "Under this frozen gate-only information set, the model is signal-scarce."
            if not selected and not positive_marginals
            else "The gate-only results do not by themselves prove fundamental signal scarcity."
        ),
        f"Frozen shadow candidate worthy of forward testing before validation: {selected_id or 'none'}.",
    ]

    public_candidates: dict[str, Any] = {}
    for candidate_id, result in candidate_results.items():
        public_candidates[candidate_id] = serialize_metrics(
            {
                key: value
                for key, value in result.items()
                if key not in {"trades", "audit"}
            }
        )
    metrics = {
        "summary": {
            "run_id": run_dir.name,
            "status": "research_only" if selected else "fail",
            "configurations_evaluated": len(candidate_results),
            "quality_pass_count": len(qualified_results),
            "control_candidate_id": control_id,
            "selected_candidate_id": selected_id,
            "diagnostic_reference_id": diagnostic["candidate_id"],
            "candidate_frozen": bool(selected),
            "new_forward_cutoff": None,
            "validator_result": "pending",
            "operational_artifact_changed": False,
        },
        "folds": provenance,
        "candidates": public_candidates,
        "pareto_frontier": frontier,
        "threshold_marginal_cohorts": serialize_metrics(marginal_metrics),
        "rsi_marginal_analysis": serialize_metrics(rsi_analysis),
        "answers": answers,
        "execution_semantics": {
            "decision": "completed-bar features only",
            "episodes": "threshold rising edge; reset after >2-minute data gap or fold boundary",
            "direction": "long only",
            "tp_atr": TP_ATR,
            "sl_atr": SL_ATR,
            "max_hold_m1_bars": MAX_HOLD_MINUTES,
            "same_bar_rule": "stop-first",
            "max_positions": 1,
            "loss_cooldown_minutes": LOSS_COOLDOWN_MINUTES,
            "daily_loss_guard": "5% realized-risk approximation; exact historical account/floating state unavailable",
            "spread": "observed entry spread when >0, otherwise 30-point fallback",
            "base_extra_cost_points": BASE_EXTRA_COST_POINTS,
            "stress_extra_cost_points": STRESS_EXTRA_COST_POINTS,
            "pnl_unit": "net stop-risk R",
        },
    }

    candidate_spec_path = None
    if selected:
        freeze_time = now_utc()
        candidate_spec = {
            "run_id": run_dir.name,
            "candidate_id": selected_id,
            "status": "research_only_shadow_candidate",
            "frozen_at_utc": freeze_time,
            "new_forward_cutoff": freeze_time,
            "threshold": selected["threshold"],
            "rsi_policy": RSI_DESCRIPTIONS[selected["rsi_policy"]],
            "model_artifact": MODEL_FILE.name,
            "model_artifact_sha256": operational_hashes[MODEL_FILE.name],
            "tp_atr": TP_ATR,
            "sl_atr": SL_ATR,
            "minimum_tp_price": MIN_TP_PRICE,
            "minimum_sl_price": MIN_SL_PRICE,
            "max_hold_minutes": MAX_HOLD_MINUTES,
            "allowed_hours": sorted(ALLOWED_HOURS),
            "allowed_weekdays": sorted(ALLOWED_WEEKDAYS),
            "max_positions": 1,
            "cost_assumptions": {
                "observed_or_fallback_spread_points": FALLBACK_SPREAD_POINTS,
                "extra_cost_points": BASE_EXTRA_COST_POINTS,
                "stress_extra_cost_points": STRESS_EXTRA_COST_POINTS,
            },
            "execution_semantics": metrics["execution_semantics"],
            "research_script_sha256": file_sha256(Path(__file__)),
            "replacement_authorized": False,
        }
        candidate_spec_path = run_dir / "candidate_spec.json"
        write_json(candidate_spec_path, candidate_spec)
        (run_dir / "candidate_spec.sha256").write_text(
            f"{file_sha256(candidate_spec_path)}  candidate_spec.json\n", encoding="utf-8"
        )
        metrics["summary"]["new_forward_cutoff"] = freeze_time

    write_candidates(run_dir, candidate_results, control_id)
    write_trade_ledger(run_dir, candidate_results, marginal_results)
    write_json(run_dir / "metrics.json", serialize_metrics(metrics))
    (run_dir / "report.md").write_text(markdown_report(serialize_metrics(metrics)), encoding="utf-8")

    source_files = []
    for path in sorted(ROOT.glob("GOLD#_*.csv")):
        print(f"Hashing source {path.name}", flush=True)
        source_files.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "retention_status": "retained_local_repository_workspace",
            }
        )
    manifest = read_json(run_dir / "manifest.json")
    manifest["data"].update(
        {
            "symbols": ["GOLD#"],
            "data_sources": ["XM MT5 historical CSV exports retained locally"],
            "source_files": source_files,
            "timezone": "XM export/server timestamps; exact UTC mapping unproven",
            "data_start_utc": oof["time"].iat[0].isoformat(),
            "data_end_utc": oof["time"].iat[-1].isoformat(),
            "train_start_utc": provenance[0]["train_start"],
            "train_end_utc": provenance[-1]["train_end"],
            "train_rows": sum(item["train_rows"] for item in provenance),
            "validation_start_utc": oof["time"].iat[0].isoformat(),
            "validation_end_utc": oof["time"].iat[-1].isoformat(),
            "validation_rows": int(len(oof)),
            "test_start_utc": "not_applicable_development_only",
            "test_end_utc": "not_applicable_development_only",
            "test_rows": 0,
            "purge_details": f"{LABEL_HORIZON} source rows removed before every scoring fold",
            "embargo_details": "fold scoring starts strictly after the latest purged training label bar",
            "raw_snapshot_retained": True,
            "reproducibility_claim": "code_and_local_raw_exports_retained_timestamp_semantics_unproven",
            "folds": provenance,
        }
    )
    manifest["data"]["mt5_fetch"].update(
        {"used": False, "not_applicable_reason": "No dynamic MT5 fetch; local historical exports only"}
    )
    manifest["model"].update(
        {
            "features": features,
            "feature_count": len(features),
            "oof_models_trained": True,
            "oof_model_count": len(provenance),
            "oof_model_retention": "not_saved; reproducible from immutable script, source hashes, and seed",
            "artifact_path": None,
            "artifact_sha256": None,
            "retention_status": "no_new_operational_candidate_model",
        }
    )
    control = metrics["candidates"][control_id]["pooled"]
    registry_result = metrics["candidates"][selected_id]["pooled"] if selected_id else control
    manifest["registry"].update(
        {
            "parent_or_incumbent": f"{MODEL_FILE.name}@{operational_hashes[MODEL_FILE.name]}",
            "selected_configuration": selected_id or "none_quality_pass_control_retained",
            "trades_per_day": registry_result["trades_per_day"],
            "realized_win_rate": registry_result["realized_wr"],
            "pf": registry_result["pf"],
            "mean_r": registry_result["mean_r"],
            "pnl": registry_result["pnl"],
            "max_dd": registry_result["max_dd"],
            "validator_result": "pending",
        }
    )
    manifest["result"] = {
        "configurations_evaluated": len(candidate_results),
        "quality_pass_count": len(qualified_results),
        "selected_candidate_id": selected_id,
        "candidate_frozen": bool(selected),
    }
    if selected:
        manifest["evidence_status"]["new_forward_cutoff"] = metrics["summary"]["new_forward_cutoff"]
    for path, kind in (
        (run_dir / "oof_model_provenance.json", "oof_model_provenance"),
        (run_dir / "oof_predictions.npz", "complete_oof_predictions"),
        (run_dir / "oof_qualifying_rows.csv.gz", "oof_predictions"),
        (run_dir / "trade_ledger.csv.gz", "executable_trade_ledger"),
        (candidate_spec_path, "frozen_candidate_spec"),
    ):
        if path is not None and path.is_file():
            manifest["artifacts"].append(
                {
                    "kind": kind,
                    "path": path.relative_to(run_dir).as_posix(),
                    "sha256": file_sha256(path),
                    "retention_status": "stored_in_run_directory",
                }
            )
    current_hashes = {
        "gemini.py": file_sha256(GEMINI_FILE),
        MODEL_FILE.name: file_sha256(MODEL_FILE),
    }
    if current_hashes != operational_hashes:
        raise RuntimeError("Operational artifact changed during research")
    manifest["operational_hashes_after"] = current_hashes
    manifest["promotion"]["operational_artifact_changed"] = False
    write_json(run_dir / "manifest.json", manifest)
    print(markdown_report(serialize_metrics(metrics)), flush=True)
    print(f"RUN_COMPUTATION_COMPLETE {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
