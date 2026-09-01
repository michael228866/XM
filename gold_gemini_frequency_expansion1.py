from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import drl_trading_v2
from barrier_final_train import prepare_barrier_data
from barrier_research_suite import predict_positive
from gold_generation11_execution_aligned import add_targets


ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "gold_long_recent_candidate_xgb.json"
GEMINI_FILE = ROOT / "gemini.py"
REPORT_JSON = ROOT / "gold_gemini_frequency_expansion1.json"
REPORT_MD = ROOT / "gold_gemini_frequency_expansion1.md"

RESEARCH_START = pd.Timestamp("2018-01-01 00:00:00")
RESEARCH_END = pd.Timestamp("2025-01-01 00:00:00")
FORWARD_CUTOFF_UTC = "2026-09-01T02:00:00Z"
FOLDS = (
    ("2018_2020", pd.Timestamp("2018-01-01"), pd.Timestamp("2021-01-01")),
    ("2021_2022", pd.Timestamp("2021-01-01"), pd.Timestamp("2023-01-01")),
    ("2023_2024", pd.Timestamp("2023-01-01"), pd.Timestamp("2025-01-01")),
)

PRODUCTION_THRESHOLD = 0.75
NEAR_MISS_LOW = 0.65
ADJACENT_LOW = 0.60
TP_ATR = 1.3
SL_ATR = 1.6
MIN_TP_PRICE = 1.5
MIN_SL_PRICE = 0.6
MAX_HOLD_MINUTES = 90
FALLBACK_SPREAD_POINTS = 30.0
BASE_EXTRA_COST_POINTS = 5.0
STRESS_EXTRA_COST_POINTS = 10.0
POINT = 0.01
RISK_PER_TRADE = 0.014
LOSS_COOLDOWN_MINUTES = 15
MAX_DAILY_LOSS_PCT = 0.05
ALLOWED_HOURS = frozenset((0, 1, 2, 3, 4, 8, 9, 11, 12, 17, 18, 19, 20, 22, 23))
ALLOWED_WEEKDAYS = frozenset((0, 1, 2, 3, 4))

POLICIES = {
    "near_miss_persistence": ("near_miss_persistence",),
    "near_miss_pullback": ("near_miss_pullback",),
    "near_miss_union": ("near_miss_persistence", "near_miss_pullback"),
    "reentry_only": ("reentry_controlled_pullback",),
    "near_miss_plus_reentry": (
        "near_miss_persistence",
        "near_miss_pullback",
        "reentry_controlled_pullback",
    ),
    "all_predeclared": (
        "near_miss_persistence",
        "near_miss_pullback",
        "reentry_controlled_pullback",
        "adjacent_strong_continuation",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GEMINI Frequency Expansion 1 development replay"
    )
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def effective_spread(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if "SPREAD" not in frame:
        return (
            np.full(len(frame), FALLBACK_SPREAD_POINTS, dtype=np.float64),
            np.zeros(len(frame), dtype=bool),
        )
    raw = pd.to_numeric(frame["SPREAD"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    observed = np.isfinite(raw) & (raw > 0.0)
    return np.where(observed, raw, FALLBACK_SPREAD_POINTS), observed


def rising_edge(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    return mask & ~np.r_[False, mask[:-1]]


def load_research_frame() -> tuple[pd.DataFrame, list[str]]:
    drl_trading_v2.DATA_DIR = str(ROOT)
    frame, features = prepare_barrier_data()
    frame = add_targets(frame)
    frame = frame.loc[
        (frame["TIME_DT"] >= RESEARCH_START)
        & (frame["TIME_DT"] < RESEARCH_END)
    ].copy()
    if frame.empty:
        raise RuntimeError("The 2018-2024 research frame is empty")
    if frame["TIME_DT"].max() >= pd.Timestamp(FORWARD_CUTOFF_UTC).tz_localize(None):
        raise RuntimeError("Untouched-forward rows entered the research frame")

    model = xgb.XGBClassifier()
    model.load_model(MODEL_FILE)
    model.set_params(device="cpu")
    model_features = model.get_booster().feature_names
    if model_features != features:
        raise RuntimeError("Production model features differ from historical features")
    frame["PRODUCTION_SCORE"] = predict_positive(model, frame, features)
    return frame.reset_index(drop=True), features


def add_entry_state(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    result = frame.copy()
    spread_points, spread_observed = effective_spread(result)
    result["EFFECTIVE_SPREAD_POINTS"] = spread_points
    result["SPREAD_OBSERVED"] = spread_observed

    trend_columns = [name for name in features if name.endswith("_TREND")]
    if not trend_columns:
        raise RuntimeError("No completed-bar MTF trend columns were found")
    trend = result[trend_columns].to_numpy(dtype=np.float64)
    result["LONG_TREND_ALIGNMENT"] = np.nanmean(trend > 0.0, axis=1)

    score = result["PRODUCTION_SCORE"].to_numpy(dtype=np.float64)
    previous_score = np.r_[np.nan, score[:-1]]
    second_previous_score = np.r_[np.nan, np.nan, score[:-2]]
    score_rising = (score > previous_score) & (previous_score > second_previous_score)
    score_one_step_rising = score > previous_score

    timestamp = result["TIME_DT"]
    rsi = result["M1_RSI"].to_numpy(dtype=np.float64)
    atr = result["ATR"].to_numpy(dtype=np.float64)
    take = np.maximum(atr * TP_ATR, MIN_TP_PRICE)
    spread_limit = np.minimum(
        100.0,
        np.maximum(45.0, take * 0.25 / POINT),
    )
    session_ok = (
        timestamp.dt.hour.isin(ALLOWED_HOURS).to_numpy()
        & timestamp.dt.dayofweek.isin(ALLOWED_WEEKDAYS).to_numpy()
    )
    rsi_ok = (rsi >= 22.0) & ~((rsi >= 35.0) & (rsi <= 45.0))
    spread_ok = spread_points <= spread_limit
    mature = (
        result["LONG_OUTCOME"].to_numpy(dtype=np.int8) >= 0
    ) & (result["LONG_EXIT_OFFSET"].to_numpy(dtype=np.int16) > 0)
    common = session_ok & rsi_ok & spread_ok & mature & np.isfinite(score)

    result["SESSION_OK"] = session_ok
    result["RSI_OK"] = rsi_ok
    result["SPREAD_OK"] = spread_ok
    result["MATURE"] = mature
    result["PRODUCTION_ELIGIBLE"] = common & (score >= PRODUCTION_THRESHOLD)

    near_band = common & (score >= NEAR_MISS_LOW) & (score < PRODUCTION_THRESHOLD)
    bias = result["BIAS_20"].to_numpy(dtype=np.float64)
    roc = result["ROC_5"].to_numpy(dtype=np.float64)
    body = result["BODY_PCT"].to_numpy(dtype=np.float64)
    alignment = result["LONG_TREND_ALIGNMENT"].to_numpy(dtype=np.float64)

    persistence = (
        near_band & score_rising & (alignment >= 0.70) & (roc > 0.0)
    )
    pullback = (
        near_band
        & score_one_step_rising
        & (alignment >= 0.70)
        & (bias >= -0.002)
        & (bias < 0.0)
        & (roc > 0.0)
        & (rsi > 45.0)
    )
    adjacent = (
        common
        & (score >= ADJACENT_LOW)
        & (score < NEAR_MISS_LOW)
        & score_rising
        & (alignment >= 0.85)
        & (bias >= 0.0)
        & (roc > 0.0)
        & (body >= 0.50)
    )
    result["EVENT_NEAR_MISS_PERSISTENCE"] = rising_edge(persistence)
    result["EVENT_NEAR_MISS_PULLBACK"] = rising_edge(pullback)
    result["EVENT_ADJACENT_STRONG_CONTINUATION"] = rising_edge(adjacent)
    return result


def natural_trade(frame: pd.DataFrame, index: int, source: str) -> dict:
    offset = int(frame["LONG_EXIT_OFFSET"].iat[index])
    if offset <= 0 or index + offset >= len(frame):
        raise ValueError(f"Immature entry at index {index}")
    spread = float(frame["EFFECTIVE_SPREAD_POINTS"].iat[index])
    stop = max(float(frame["ATR"].iat[index]) * SL_ATR, MIN_SL_PRICE)
    old_reward = float(frame["LONG_REWARD"].iat[index])
    gross = old_reward * (stop + FALLBACK_SPREAD_POINTS * POINT) + (
        FALLBACK_SPREAD_POINTS + BASE_EXTRA_COST_POINTS
    ) * POINT
    denominator = stop + spread * POINT
    outcome = int(frame["LONG_OUTCOME"].iat[index])
    return {
        "index": index,
        "exit_index": index + offset,
        "entry_time": frame["TIME_DT"].iat[index],
        "exit_time": frame["TIME_DT"].iat[index + offset],
        "source": source,
        "score": float(frame["PRODUCTION_SCORE"].iat[index]),
        "outcome": outcome,
        "exit_type": {0: "timeout", 1: "tp", 2: "sl"}[outcome],
        "gross_pnl_price": gross,
        "spread_points": spread,
        "spread_observed": bool(frame["SPREAD_OBSERVED"].iat[index]),
        "stop_distance": stop,
        "denominator": denominator,
        "reward": (gross - (spread + BASE_EXTRA_COST_POINTS) * POINT)
        / denominator,
    }


def preempt_trade(
    frame: pd.DataFrame, index: int, exit_index: int, source: str
) -> dict:
    spread = float(frame["EFFECTIVE_SPREAD_POINTS"].iat[index])
    stop = max(float(frame["ATR"].iat[index]) * SL_ATR, MIN_SL_PRICE)
    gross = float(frame["CLOSE"].iat[exit_index] - frame["CLOSE"].iat[index])
    denominator = stop + spread * POINT
    return {
        "index": index,
        "exit_index": exit_index,
        "entry_time": frame["TIME_DT"].iat[index],
        "exit_time": frame["TIME_DT"].iat[exit_index],
        "source": source,
        "score": float(frame["PRODUCTION_SCORE"].iat[index]),
        "outcome": -1,
        "exit_type": "production_preempt",
        "gross_pnl_price": gross,
        "spread_points": spread,
        "spread_observed": bool(frame["SPREAD_OBSERVED"].iat[index]),
        "stop_distance": stop,
        "denominator": denominator,
        "reward": (gross - (spread + BASE_EXTRA_COST_POINTS) * POINT)
        / denominator,
    }


def reward_at_cost(record: dict, extra_cost_points: float) -> float:
    return (
        float(record["gross_pnl_price"])
        - (float(record["spread_points"]) + extra_cost_points) * POINT
    ) / float(record["denominator"])


def rolling_risk_multiplier(records: list[dict]) -> float:
    if len(records) < 18:
        return 1.0
    rewards = np.asarray(
        [record["reward"] for record in records[-30:]], dtype=np.float64
    )
    loss = -float(rewards[rewards < 0.0].sum())
    profit_factor = np.inf if loss <= 0.0 else float(rewards[rewards > 0.0].sum() / loss)
    return 0.5 if profit_factor < 1.15 else 1.0


def execute_production(frame: pd.DataFrame) -> tuple[list[dict], dict]:
    candidates = np.flatnonzero(
        frame["PRODUCTION_ELIGIBLE"].to_numpy(dtype=bool)
    )
    records: list[dict] = []
    free_index = 0
    last_loss_exit: pd.Timestamp | None = None
    daily_return: dict[object, float] = {}
    skipped = {"occupancy": 0, "loss_cooldown": 0, "daily_loss": 0}

    for index in candidates:
        if index < free_index:
            skipped["occupancy"] += 1
            continue
        entry_time = frame["TIME_DT"].iat[index]
        if (
            last_loss_exit is not None
            and entry_time < last_loss_exit + pd.Timedelta(minutes=LOSS_COOLDOWN_MINUTES)
        ):
            skipped["loss_cooldown"] += 1
            continue
        if daily_return.get(entry_time.date(), 0.0) <= -MAX_DAILY_LOSS_PCT:
            skipped["daily_loss"] += 1
            continue

        record = natural_trade(frame, int(index), "production")
        record["risk_multiplier"] = rolling_risk_multiplier(records)
        records.append(record)
        free_index = int(record["exit_index"]) + 1
        risk_return = RISK_PER_TRADE * record["risk_multiplier"] * record["reward"]
        exit_date = record["exit_time"].date()
        daily_return[exit_date] = daily_return.get(exit_date, 0.0) + risk_return
        if record["reward"] <= 0.0:
            last_loss_exit = record["exit_time"]

    return records, {"raw_eligible_rows": int(len(candidates)), **skipped}


def reentry_events(frame: pd.DataFrame, production: list[dict]) -> np.ndarray:
    base = frame["EVENT_NEAR_MISS_PULLBACK"].to_numpy(dtype=bool)
    times = frame["TIME_DT"].to_numpy(dtype="datetime64[ns]")
    output = np.zeros(len(frame), dtype=bool)
    base_indices = np.flatnonzero(base)
    for record in production:
        start_time = np.datetime64(record["exit_time"] + pd.Timedelta(minutes=5))
        end_time = np.datetime64(record["exit_time"] + pd.Timedelta(minutes=30))
        left = int(np.searchsorted(times, start_time, side="left"))
        right = int(np.searchsorted(times, end_time, side="right"))
        position = int(np.searchsorted(base_indices, left, side="left"))
        if position < len(base_indices) and base_indices[position] < right:
            output[base_indices[position]] = True
    return output


def event_sources(
    frame: pd.DataFrame, production: list[dict]
) -> tuple[dict[str, np.ndarray], dict]:
    sources = {
        "near_miss_persistence": np.flatnonzero(
            frame["EVENT_NEAR_MISS_PERSISTENCE"].to_numpy(dtype=bool)
        ),
        "near_miss_pullback": np.flatnonzero(
            frame["EVENT_NEAR_MISS_PULLBACK"].to_numpy(dtype=bool)
        ),
        "reentry_controlled_pullback": np.flatnonzero(
            reentry_events(frame, production)
        ),
        "adjacent_strong_continuation": np.flatnonzero(
            frame["EVENT_ADJACENT_STRONG_CONTINUATION"].to_numpy(dtype=bool)
        ),
    }
    sets = {name: set(map(int, values)) for name, values in sources.items()}
    overlap = {}
    names = tuple(sources)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap[f"{left}__{right}"] = len(sets[left] & sets[right])
    return sources, {
        "raw_events": {name: len(values) for name, values in sources.items()},
        "pairwise_timestamp_overlap": overlap,
    }


def merge_events(
    sources: dict[str, np.ndarray], selected_sources: tuple[str, ...]
) -> list[tuple[int, str]]:
    by_index: dict[int, list[str]] = {}
    for source in selected_sources:
        for index in sources[source]:
            by_index.setdefault(int(index), []).append(source)
    return [
        (index, "+".join(sorted(names)))
        for index, names in sorted(by_index.items())
    ]


def execute_expansion(
    frame: pd.DataFrame,
    production: list[dict],
    events: list[tuple[int, str]],
) -> tuple[list[dict], dict]:
    production_entries = [int(record["index"]) for record in production]
    occupied = np.zeros(len(frame), dtype=bool)
    for record in production:
        occupied[int(record["index"]) : int(record["exit_index"]) + 1] = True

    records: list[dict] = []
    free_index = 0
    last_loss_exit: pd.Timestamp | None = None
    daily_return: dict[object, float] = {}
    skipped = {
        "production_occupancy": 0,
        "expansion_occupancy": 0,
        "loss_cooldown": 0,
        "daily_loss": 0,
        "production_preemptions": 0,
    }
    for index, source in events:
        if occupied[index]:
            skipped["production_occupancy"] += 1
            continue
        if index < free_index:
            skipped["expansion_occupancy"] += 1
            continue
        entry_time = frame["TIME_DT"].iat[index]
        if (
            last_loss_exit is not None
            and entry_time < last_loss_exit + pd.Timedelta(minutes=LOSS_COOLDOWN_MINUTES)
        ):
            skipped["loss_cooldown"] += 1
            continue
        if daily_return.get(entry_time.date(), 0.0) <= -MAX_DAILY_LOSS_PCT:
            skipped["daily_loss"] += 1
            continue

        natural = natural_trade(frame, index, source)
        next_position = bisect.bisect_right(production_entries, index)
        next_production = (
            production_entries[next_position]
            if next_position < len(production_entries)
            else None
        )
        if (
            next_production is not None
            and next_production <= int(natural["exit_index"])
        ):
            record = preempt_trade(frame, index, next_production, source)
            skipped["production_preemptions"] += 1
        else:
            record = natural
        records.append(record)
        free_index = int(record["exit_index"]) + 1
        risk_return = RISK_PER_TRADE * record["reward"]
        exit_date = record["exit_time"].date()
        daily_return[exit_date] = daily_return.get(exit_date, 0.0) + risk_return
        if record["reward"] <= 0.0:
            last_loss_exit = record["exit_time"]
    return records, {"raw_unique_events": len(events), **skipped}


def subset_records(
    records: list[dict], start: pd.Timestamp, end: pd.Timestamp
) -> list[dict]:
    return [
        record
        for record in records
        if start <= record["entry_time"] < end
    ]


def small_metrics(records: list[dict]) -> dict:
    rewards = np.asarray([record["reward"] for record in records], dtype=np.float64)
    if len(rewards) == 0:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": None,
            "mean_r": 0.0,
            "sum_r": 0.0,
        }
    loss = -float(rewards[rewards < 0.0].sum())
    return {
        "trades": len(records),
        "win_rate": float(np.mean(rewards > 0.0)),
        "profit_factor": None
        if loss <= 0.0
        else float(rewards[rewards > 0.0].sum() / loss),
        "mean_r": float(rewards.mean()),
        "sum_r": float(rewards.sum()),
    }


def metrics(
    records: list[dict], days: int, extra_cost_points: float = BASE_EXTRA_COST_POINTS
) -> dict:
    rewards = np.asarray(
        [reward_at_cost(record, extra_cost_points) for record in records],
        dtype=np.float64,
    )
    outcomes = np.asarray([record["outcome"] for record in records], dtype=np.int8)
    if len(rewards) == 0:
        return {
            "trades": 0,
            "trades_per_day": 0.0,
            "wins": 0,
            "losses": 0,
            "timeouts": 0,
            "tp_exits": 0,
            "sl_exits": 0,
            "preempt_exits": 0,
            "tp_first_win_rate": 0.0,
            "realized_win_rate": 0.0,
            "profit_factor": None,
            "mean_r": 0.0,
            "pnl_r": 0.0,
            "max_drawdown_r": 0.0,
            "max_drawdown_pct_at_1_4pct_risk": 0.0,
            "average_winner_r": None,
            "average_loser_r": None,
            "payoff_ratio": None,
            "break_even_win_rate": None,
            "break_even_adjusted_edge": None,
            "observed_spread_trades": 0,
            "fallback_spread_trades": 0,
            "long_contribution": small_metrics([]),
            "short_contribution": small_metrics([]),
            "source_contribution": {},
        }

    gains = rewards[rewards > 0.0]
    losses = rewards[rewards <= 0.0]
    gross_loss = -float(losses.sum())
    profit_factor = None if gross_loss <= 0.0 else float(gains.sum() / gross_loss)
    equity_r = np.cumsum(rewards)
    peak_r = np.maximum.accumulate(np.maximum(equity_r, 0.0))
    max_drawdown_r = float(np.min(equity_r - peak_r))
    equity = np.cumprod(1.0 + RISK_PER_TRADE * rewards)
    peak = np.maximum.accumulate(np.maximum(equity, 1.0))
    max_drawdown_pct = float(np.min(equity / peak - 1.0))
    average_winner = None if len(gains) == 0 else float(gains.mean())
    average_loser = None if len(losses) == 0 else float(losses.mean())
    payoff = (
        None
        if average_winner is None or average_loser is None or average_loser == 0.0
        else average_winner / abs(average_loser)
    )
    break_even = None if payoff is None else 1.0 / (1.0 + payoff)
    win_rate = float(np.mean(rewards > 0.0))
    sources = sorted({str(record["source"]) for record in records})
    return {
        "trades": len(records),
        "trades_per_day": len(records) / max(days, 1),
        "wins": int(np.sum(rewards > 0.0)),
        "losses": int(np.sum(rewards <= 0.0)),
        "timeouts": int(np.sum(outcomes == 0)),
        "tp_exits": int(np.sum(outcomes == 1)),
        "sl_exits": int(np.sum(outcomes == 2)),
        "preempt_exits": int(sum(record["exit_type"] == "production_preempt" for record in records)),
        "tp_first_win_rate": float(np.mean(outcomes == 1)),
        "realized_win_rate": win_rate,
        "profit_factor": profit_factor,
        "mean_r": float(rewards.mean()),
        "pnl_r": float(rewards.sum()),
        "max_drawdown_r": max_drawdown_r,
        "max_drawdown_pct_at_1_4pct_risk": max_drawdown_pct,
        "average_winner_r": average_winner,
        "average_loser_r": average_loser,
        "payoff_ratio": payoff,
        "break_even_win_rate": break_even,
        "break_even_adjusted_edge": None
        if break_even is None
        else win_rate - break_even,
        "observed_spread_trades": int(sum(record["spread_observed"] for record in records)),
        "fallback_spread_trades": int(sum(not record["spread_observed"] for record in records)),
        "long_contribution": small_metrics(records),
        "short_contribution": small_metrics([]),
        "source_contribution": {
            source: small_metrics(
                [record for record in records if record["source"] == source]
            )
            for source in sources
        },
    }


def compare_portfolio(
    frame: pd.DataFrame,
    production: list[dict],
    expansion: list[dict],
) -> dict:
    folds = {}
    for name, start, end in FOLDS:
        days = int((end - start).days)
        base_records = subset_records(production, start, end)
        expansion_records = subset_records(expansion, start, end)
        combined_records = sorted(
            base_records + expansion_records, key=lambda record: record["index"]
        )
        baseline = metrics(base_records, days)
        added = metrics(expansion_records, days)
        combined = metrics(combined_records, days)
        combined_stress = metrics(
            combined_records, days, STRESS_EXTRA_COST_POINTS
        )
        added_stress = metrics(
            expansion_records, days, STRESS_EXTRA_COST_POINTS
        )
        folds[name] = {
            "production": baseline,
            "expansion": added,
            "combined": combined,
            "expansion_cost_stress": added_stress,
            "combined_cost_stress": combined_stress,
            "unique_added_trades": added["trades"],
            "frequency_uplift": (
                0.0
                if baseline["trades"] == 0
                else (combined["trades"] - baseline["trades"])
                / baseline["trades"]
            ),
        }

    days = int((RESEARCH_END - RESEARCH_START).days)
    production_all = subset_records(production, RESEARCH_START, RESEARCH_END)
    expansion_all = subset_records(expansion, RESEARCH_START, RESEARCH_END)
    combined_all = sorted(
        production_all + expansion_all, key=lambda record: record["index"]
    )
    baseline = metrics(production_all, days)
    added = metrics(expansion_all, days)
    combined = metrics(combined_all, days)
    added_stress = metrics(expansion_all, days, STRESS_EXTRA_COST_POINTS)
    combined_stress = metrics(combined_all, days, STRESS_EXTRA_COST_POINTS)
    positive_expansion_folds = sum(
        value["expansion"]["mean_r"] > 0.0 for value in folds.values()
    )
    improved_folds = sum(
        value["unique_added_trades"] > 0 for value in folds.values()
    )
    drawdown_ok = abs(combined["max_drawdown_r"]) <= max(
        abs(baseline["max_drawdown_r"]) * 1.50, 3.0
    )
    discovery_pass = bool(
        added["trades"] > 0
        and added["profit_factor"] is not None
        and added["profit_factor"] > 1.0
        and added["mean_r"] > 0.0
        and added_stress["profit_factor"] is not None
        and added_stress["profit_factor"] > 1.0
        and added_stress["mean_r"] > 0.0
        and combined["profit_factor"] is not None
        and combined["profit_factor"] > 1.0
        and combined["mean_r"] > 0.0
        and combined_stress["profit_factor"] is not None
        and combined_stress["profit_factor"] > 1.0
        and combined_stress["mean_r"] > 0.0
        and positive_expansion_folds >= 2
        and improved_folds >= 2
        and drawdown_ok
    )
    shadow_gate = bool(
        discovery_pass
        and combined["realized_win_rate"] >= 0.60
        and combined["trades"] > baseline["trades"]
    )
    return {
        "folds": folds,
        "pooled": {
            "production": baseline,
            "expansion": added,
            "combined": combined,
            "expansion_cost_stress": added_stress,
            "combined_cost_stress": combined_stress,
            "unique_added_trades": added["trades"],
            "frequency_uplift": 0.0
            if baseline["trades"] == 0
            else (combined["trades"] - baseline["trades"])
            / baseline["trades"],
        },
        "gate": {
            "positive_expansion_folds": positive_expansion_folds,
            "folds_with_frequency_uplift": improved_folds,
            "drawdown_ok": drawdown_ok,
            "internal_discovery_pass": discovery_pass,
            "paper_shadow_gate": shadow_gate,
        },
    }


def data_bottleneck(frame: pd.DataFrame, production_audit: dict) -> dict:
    score = frame["PRODUCTION_SCORE"].to_numpy(dtype=np.float64)
    mature = frame["MATURE"].to_numpy(dtype=bool)
    session = frame["SESSION_OK"].to_numpy(dtype=bool)
    rsi = frame["RSI_OK"].to_numpy(dtype=bool)
    spread = frame["SPREAD_OK"].to_numpy(dtype=bool)
    high_score = mature & (score >= PRODUCTION_THRESHOLD)
    near = mature & (score >= NEAR_MISS_LOW) & (score < PRODUCTION_THRESHOLD)
    return {
        "mature_m1_rows": int(mature.sum()),
        "score_at_or_above_0_75": int(high_score.sum()),
        "high_score_rejected_by_session": int((high_score & ~session).sum()),
        "high_score_rejected_by_rsi": int((high_score & session & ~rsi).sum()),
        "high_score_rejected_by_spread": int(
            (high_score & session & rsi & ~spread).sum()
        ),
        "production_raw_eligible": production_audit["raw_eligible_rows"],
        "production_occupancy_rejections": production_audit["occupancy"],
        "near_miss_0_65_to_0_75_all_mature": int(near.sum()),
        "near_miss_with_other_production_guards": int(
            (near & session & rsi & spread).sum()
        ),
    }


def fmt_metric(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def markdown_report(report: dict) -> str:
    lines = [
        "# GEMINI FREQUENCY EXPANSION 1",
        "",
        "Status: `research_only`; `gemini.py` and the production model were not modified.",
        "",
        "> This is a development-period counterfactual replay, not a new OOS claim. "
        "The current model was trained/selected on 2025-2026 data and is applied "
        "backward to 2018-2024 here.",
        "",
        "## Frozen production reconstruction",
        "",
        "Long-only, score >= 0.75, frozen production sessions and RSI guards, "
        "TP/SL 1.3/1.6 ATR, 90-minute maximum hold, one position, 15-minute "
        "post-loss cooldown, spread gate, and 30-point fallback where historical "
        "spread is unavailable.",
        "",
        "Historical broker-account state, exact fills/slippage/commission, and "
        "the server-time-to-UTC mapping are unavailable; the ledger is the closest "
        "causal reconstruction, not a byte-for-byte broker ledger.",
        "",
        "## Pooled portfolio comparison (2018-2024; calendar-day denominator)",
        "",
        "| Portfolio | Prod trades | Added | Uplift | Combined/day | Combined WR | Combined PF | Combined Mean-R | Combined PnL-R | Max DD-R | Expansion PF | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for candidate in report["pareto_table"]:
        pooled = candidate["pooled"]
        baseline = pooled["production"]
        expansion = pooled["expansion"]
        combined = pooled["combined"]
        lines.append(
            f"| {candidate['policy']} | {baseline['trades']} | "
            f"{pooled['unique_added_trades']} | {pooled['frequency_uplift']:.1%} | "
            f"{combined['trades_per_day']:.4f} | "
            f"{combined['realized_win_rate']:.2%} | "
            f"{fmt_metric(combined['profit_factor'], 2)} | "
            f"{combined['mean_r']:.4f} | {combined['pnl_r']:.2f} | "
            f"{combined['max_drawdown_r']:.2f} | "
            f"{fmt_metric(expansion['profit_factor'], 2)} | "
            f"{'PASS' if candidate['gate']['internal_discovery_pass'] else 'FAIL'} |"
        )

    best_name = report["decision"]["best_internal_policy"]
    reference_name = report["decision"]["diagnostic_reference_policy"]
    best = report["portfolios"][reference_name]
    lines.extend(
        [
            "",
            "## Diagnostic least-degraded portfolio by fold",
            "",
            (
                f"Accepted internal portfolio: `{best_name}`; diagnostic reference: "
                f"`{reference_name}`."
            ),
            "",
            "| Fold | Layer | Trades | Trades/day | WR | PF | Mean-R | PnL-R | Max DD-R | TP | SL | Timeout | Preempt |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for fold_name, values in best["folds"].items():
        for layer in ("production", "expansion", "combined"):
            item = values[layer]
            lines.append(
                f"| {fold_name} | {layer} | {item['trades']} | "
                f"{item['trades_per_day']:.4f} | {item['realized_win_rate']:.2%} | "
                f"{fmt_metric(item['profit_factor'], 2)} | {item['mean_r']:.4f} | "
                f"{item['pnl_r']:.2f} | {item['max_drawdown_r']:.2f} | "
                f"{item['tp_exits']} | {item['sl_exits']} | {item['timeouts']} | "
                f"{item['preempt_exits']} |"
            )

    answers = report["answers"]
    lines.extend(["", "## Required answers", ""])
    for index, answer in enumerate(answers, start=1):
        lines.append(f"{index}. {answer}")
    lines.extend(
        [
            "",
            "## Method constraints",
            "",
            "- Production always has priority. An expansion position is closed at market "
            "when a frozen production entry arrives, so the production ledger is unchanged.",
            "- No TP/SL, production threshold, production model, or production session was changed.",
            "- The candidate list was frozen before outcome calculation; no broad threshold/model sweep was run.",
            "- Historical spread uses the Gen19 observed-value method with a 30-point fallback. "
            "This is not proof of exact historical fill cost.",
            f"- Untouched-forward cutoff `{FORWARD_CUTOFF_UTC}` was not inspected.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_answers(report: dict) -> list[str]:
    bottleneck = report["frequency_bottleneck"]
    best_name = report["decision"]["best_internal_policy"]
    reference_name = report["decision"]["diagnostic_reference_policy"]
    best = report["portfolios"][reference_name]["pooled"]
    positive_near = [
        name
        for name in ("near_miss_persistence", "near_miss_pullback", "near_miss_union")
        if report["portfolios"][name]["pooled"]["expansion"]["mean_r"] > 0.0
        and (report["portfolios"][name]["pooled"]["expansion"]["profit_factor"] or 0.0) > 1.0
    ]
    reentry = report["portfolios"]["reentry_only"]["pooled"]["expansion"]
    source_counts = report["event_audit"]["raw_events"]
    largest_source = max(source_counts, key=source_counts.get)
    pf_candidates = [
        item
        for item in report["pareto_table"]
        if (item["pooled"]["combined"]["profit_factor"] or 0.0) > 1.0
        and item["pooled"]["combined"]["mean_r"] > 0.0
    ]
    max_uplift = max(
        (item["pooled"]["frequency_uplift"] for item in pf_candidates),
        default=0.0,
    )
    near_sixty = [
        item["policy"]
        for item in pf_candidates
        if item["pooled"]["combined"]["realized_win_rate"] >= 0.60
        and item["pooled"]["frequency_uplift"] > 0.0
    ]
    return [
        "主要限制是高門檻後的訊號稀疏與持倉占用："
        f"{bottleneck['production_raw_eligible']} 個合格 M1 rows 中有 "
        f"{bottleneck['production_occupancy_rejections']} 個被既有持倉占用。",
        f"固定 0.65-0.75 near-miss 帶共有 {bottleneck['near_miss_0_65_to_0_75_all_mature']} rows；"
        f"套用其他 production guards 後為 {bottleneck['near_miss_with_other_production_guards']} rows。",
        "具有正 pooled expectancy 的 near-miss selector："
        + (", ".join(positive_near) if positive_near else "沒有"),
        f"Re-entry 可執行 {reentry['trades']} 筆，WR {reentry['realized_win_rate']:.2%}、"
        f"PF {fmt_metric(reentry['profit_factor'], 2)}、Mean-R {reentry['mean_r']:.4f}。",
        f"原始候選最多的新增來源是 {largest_source}（{source_counts[largest_source]} events）；"
        "真正可用增量仍以 non-overlapping executable trades 為準。",
        f"在 combined PF > 1 且 Mean-R > 0 的候選中，最大 frequency uplift 為 {max_uplift:.1%}。",
        "維持 combined WR >= 60% 且增加頻率的組合："
        + (", ".join(near_sixty) if near_sixty else "沒有"),
        (
            "沒有通過凍結 gate 的最佳組合；最少劣化的 diagnostic reference 是 "
            f"{reference_name}：combined {best['combined']['trades']} trades、"
            f"{best['combined']['trades_per_day']:.4f}/day、WR "
            f"{best['combined']['realized_win_rate']:.2%}、PF "
            f"{fmt_metric(best['combined']['profit_factor'], 2)}。"
            if best_name is None
            else f"通過 gate 的最佳開發期組合是 {best_name}。"
        ),
        "僅當 paper_shadow_gate 與獨立 validator 都通過才值得建立 sidecar；"
        f"本輪 paper_shadow_gate={'PASS' if report['decision']['paper_shadow_gate'] else 'FAIL'}。"
        "即使內部結果漂亮，也必須先收集 cutoff 後完全未看的 forward data。",
    ]


def self_check() -> None:
    assert rising_edge(np.asarray([False, True, True, False, True])).tolist() == [
        False,
        True,
        False,
        False,
        True,
    ]
    record = {
        "gross_pnl_price": 1.0,
        "spread_points": 30.0,
        "denominator": 2.0,
    }
    assert reward_at_cost(record, 10.0) < reward_at_cost(record, 5.0)
    values = metrics([], 10)
    assert values["trades"] == 0 and values["trades_per_day"] == 0.0
    print("SELF_CHECK_OK")


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0
    if not MODEL_FILE.exists() or not GEMINI_FILE.exists():
        raise FileNotFoundError("Production model or gemini.py is missing")

    immutable_hashes = {
        "gemini_py_sha256": file_hash(GEMINI_FILE),
        "production_model_sha256": file_hash(MODEL_FILE),
    }
    frame, features = load_research_frame()
    frame = add_entry_state(frame, features)
    production, production_audit = execute_production(frame)
    sources, event_audit = event_sources(frame, production)

    portfolios = {}
    execution_audits = {}
    for policy, selected_sources in POLICIES.items():
        events = merge_events(sources, selected_sources)
        expansion, audit = execute_expansion(frame, production, events)
        portfolios[policy] = compare_portfolio(frame, production, expansion)
        execution_audits[policy] = audit

    pareto = []
    for policy, value in portfolios.items():
        pareto.append({"policy": policy, **value})
    pareto.sort(
        key=lambda item: (
            item["gate"]["internal_discovery_pass"],
            item["pooled"]["combined"]["realized_win_rate"] >= 0.60,
            item["pooled"]["combined"]["profit_factor"] or 0.0,
            item["pooled"]["combined"]["mean_r"],
            item["pooled"]["frequency_uplift"],
        ),
        reverse=True,
    )
    qualified = [
        item for item in pareto if item["gate"]["internal_discovery_pass"]
    ]
    best_name = qualified[0]["policy"] if qualified else None
    reference_name = pareto[0]["policy"]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "study": "GEMINI FREQUENCY EXPANSION 1",
        "status": "research_only",
        "production_immutable": True,
        "production_or_gemini_modified": False,
        "untouched_forward_cutoff": FORWARD_CUTOFF_UTC,
        "untouched_forward_inspected": False,
        "data": {
            "start": frame["TIME_DT"].iat[0],
            "end": frame["TIME_DT"].iat[-1],
            "rows": len(frame),
            "folds": [
                {"name": name, "start": start, "end": end}
                for name, start, end in FOLDS
            ],
            "trades_per_day_denominator": "calendar days",
            "historical_timestamp_semantics": "XM export/server timestamps; UTC mapping unproven",
        },
        "immutable_hashes": immutable_hashes,
        "production_spec": {
            "direction": "long_only",
            "threshold": PRODUCTION_THRESHOLD,
            "tp_atr": TP_ATR,
            "sl_atr": SL_ATR,
            "min_tp_price": MIN_TP_PRICE,
            "min_sl_price": MIN_SL_PRICE,
            "max_hold_minutes": MAX_HOLD_MINUTES,
            "risk_per_trade": RISK_PER_TRADE,
            "max_open_positions": 1,
            "loss_cooldown_minutes": LOSS_COOLDOWN_MINUTES,
            "max_daily_loss_pct": MAX_DAILY_LOSS_PCT,
            "allowed_hours": sorted(ALLOWED_HOURS),
            "allowed_weekdays": sorted(ALLOWED_WEEKDAYS),
            "rsi_minimum": 22.0,
            "excluded_rsi_range_inclusive": [35.0, 45.0],
            "spread_fallback_points": FALLBACK_SPREAD_POINTS,
            "extra_cost_points": BASE_EXTRA_COST_POINTS,
            "stress_extra_cost_points": STRESS_EXTRA_COST_POINTS,
        },
        "predeclared_expansion_spec": {
            "near_miss_score_band": [NEAR_MISS_LOW, PRODUCTION_THRESHOLD],
            "adjacent_score_band": [ADJACENT_LOW, NEAR_MISS_LOW],
            "near_miss_persistence": "score rises for two completed bars, long MTF alignment >=70%, ROC_5 > 0",
            "near_miss_pullback": "score rises, MTF alignment >=70%, -0.002 <= BIAS_20 < 0, ROC_5 > 0, RSI >45",
            "reentry": "first near_miss_pullback event 5-30 minutes after a production exit",
            "adjacent_strong_continuation": "score rises for two bars, MTF alignment >=85%, BIAS_20 >=0, ROC_5 >0, BODY_PCT >=0.50",
            "production_priority": "expansion exits at market when a frozen production entry arrives",
        },
        "methodology_limitations": [
            "The production model was trained/selected in 2025-2026 and its 2018-2024 backward replay is not OOF.",
            "All evaluated periods are development history; no untouched test is claimed.",
            "Historical broker account state, fills, slippage, commissions, and exact server-time UTC mapping are unavailable.",
            "Missing/zero spread uses the validated Gen19 30-point fallback.",
            "The live rolling-PF guard changes size, not signal identity; historical metrics use fixed 1.4% risk for DD comparability.",
        ],
        "frequency_bottleneck": data_bottleneck(frame, production_audit),
        "production_execution_audit": production_audit,
        "event_audit": event_audit,
        "expansion_execution_audits": execution_audits,
        "portfolios": portfolios,
        "pareto_table": pareto,
        "decision": {
            "best_internal_policy": best_name,
            "diagnostic_reference_policy": reference_name,
            "internal_discovery_pass": best_name is not None,
            "paper_shadow_gate": bool(
                best_name is not None
                and portfolios[best_name]["gate"]["paper_shadow_gate"]
            ),
            "production_promotion": False,
            "sidecar_created": False,
            "reason": "A frozen future interval and independent validator are still required.",
        },
    }
    report["answers"] = build_answers(report)
    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    REPORT_MD.write_text(markdown_report(report), encoding="utf-8")
    print(markdown_report(report), flush=True)
    print(f"Saved {REPORT_JSON.name} and {REPORT_MD.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
