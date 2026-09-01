from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

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
from gold_generation12_executable_events import (
    MODEL_PROFILE,
    _serialize_calibrator,
    executable_events_by_expert,
    predict_executable_scores,
    rolling_score_cash_signals,
    train_executable_experts,
)
from gold_recent_walk_forward import DEFAULT_TERMINAL, build_feature_frame
from gold_regime_experts_iterative import EXPERT_NAMES, training_frame
from gold_regime_experts_walk_forward import (
    RECENT_START,
    SELECTION_FOLDS,
    benchmark_current,
    route_arrays,
)


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_JSON = PROJECT_ROOT / "gold_generation15_signal_mining.json"
REPORT_MD = PROJECT_ROOT / "gold_generation15_signal_mining.md"
CONFIG_FILE = PROJECT_ROOT / "gold_generation15_candidate.json"
MODEL_FILES = {
    (name, kind): PROJECT_ROOT / f"gold_generation15_{name}_{kind}_xgb.json"
    for name in EXPERT_NAMES
    for kind in ("win", "mean_r")
}

TARGET_WIN_RATE = 0.60
MIN_PROFIT_FACTOR = 1.0
MIN_SELECTION_TRADES = 20
MAX_DRAWDOWN_PCT = -0.20
INITIAL_DISCOVERY_START = datetime(2016, 1, 1)
INITIAL_DISCOVERY_END = datetime(2018, 1, 1)

PARENT = {
    "generation": "12_executable_events",
    "top_k_per_day": 3,
    "minimum_expected_r": -0.05,
    "session_profile": "may_baseline",
    "quality_profile": "quality_105",
}

EXPERT_CODES = {name: index + 1 for index, name in enumerate(EXPERT_NAMES)}
EXPERT_LABELS = {value: key for key, value in EXPERT_CODES.items()}
FAMILY_LABELS = {
    1: "trend_continuation",
    2: "pullback",
    3: "breakout",
    4: "mean_reversion",
    5: "volatility_expansion",
}
GROUP_MODES = (
    "expert_session",
    "expert_session_vola",
    "expert_session_rsi",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run generation 15 chronological signal-mining research."
    )
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def filter_specs() -> list[dict]:
    return [
        {
            "mode": mode,
            "minimum_events": minimum_events,
            "maximum_win_rate": maximum_win_rate,
        }
        for mode in GROUP_MODES
        for minimum_events in (12, 24)
        for maximum_win_rate in (0.45, 0.50)
    ]


def add_specs() -> list[dict]:
    return [
        {
            "mode": mode,
            "minimum_events": minimum_events,
            "target_win_rate": target_win_rate,
            "top_k_per_day": top_k,
            "minimum_expected_r": 0.0,
        }
        for mode in GROUP_MODES
        for minimum_events in (40, 80)
        for target_win_rate in (0.60, 0.62)
        for top_k in (1, 2)
    ]


def candidate_grid() -> list[dict]:
    filters = filter_specs()
    additions = add_specs()
    candidates = []
    for filter_index, filter_spec in enumerate(filters):
        candidates.append(
            {
                "candidate_id": f"filter_{filter_index:02d}",
                "filter": filter_spec,
                "addition": None,
            }
        )
    for add_index, add_spec in enumerate(additions):
        candidates.append(
            {
                "candidate_id": f"add_{add_index:02d}",
                "filter": None,
                "addition": add_spec,
            }
        )
    for filter_index, filter_spec in enumerate(filters):
        for add_index, add_spec in enumerate(additions):
            candidates.append(
                {
                    "candidate_id": f"combined_{filter_index:02d}_{add_index:02d}",
                    "filter": filter_spec,
                    "addition": add_spec,
                }
            )
    return candidates


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


def rsi_codes(frame: pd.DataFrame) -> np.ndarray:
    rsi = frame["M1_RSI"].to_numpy(dtype=np.float32)
    output = np.full(len(frame), 2, dtype=np.int8)
    output[rsi < 35.0] = 0
    output[(rsi >= 35.0) & (rsi < 45.0)] = 1
    output[(rsi >= 55.0) & (rsi < 65.0)] = 3
    output[rsi >= 65.0] = 4
    return output


def build_metadata(frame: pd.DataFrame, features: list[str]) -> dict:
    _, masks = route_arrays(frame, features)
    expert = {
        1: np.zeros(len(frame), dtype=np.int8),
        2: np.zeros(len(frame), dtype=np.int8),
    }
    for name, code in EXPERT_CODES.items():
        direction = 1 if name.startswith("long_") else 2
        expert[direction][masks[name]] = code

    session = session_codes(frame)
    volatility = volatility_codes(frame)
    rsi_group = rsi_codes(frame)
    family = {
        1: np.zeros(len(frame), dtype=np.int8),
        2: np.zeros(len(frame), dtype=np.int8),
    }
    rsi = frame["M1_RSI"].to_numpy(dtype=np.float32)
    vola = frame["VOLA_RATIO"].to_numpy(dtype=np.float32)
    body = frame["BODY_PCT"].to_numpy(dtype=np.float32)
    roc = frame["ROC_5"].to_numpy(dtype=np.float32)
    macd = frame["MACD_HIST"].to_numpy(dtype=np.float32)
    close = frame["CLOSE"].to_numpy(dtype=np.float64)
    atr = frame["ATR"].to_numpy(dtype=np.float64)
    roc_atr = np.abs(roc) * close / np.maximum(atr, 1e-9)

    for direction in (1, 2):
        direction_expert = expert[direction]
        trend_codes = (
            (EXPERT_CODES["long_trend"],)
            if direction == 1
            else (EXPERT_CODES["short_trend"],)
        )
        pullback_codes = (
            (EXPERT_CODES["long_pullback"],)
            if direction == 1
            else (EXPERT_CODES["short_pullback"],)
        )
        trend = np.isin(direction_expert, trend_codes)
        pullback = np.isin(direction_expert, pullback_codes)
        aligned_roc = roc > 0.0 if direction == 1 else roc < 0.0
        aligned_macd = macd > 0.0 if direction == 1 else macd < 0.0
        extreme_rsi = rsi <= 32.0 if direction == 1 else rsi >= 68.0

        output = family[direction]
        output[trend] = 1
        output[pullback] = 2
        breakout = (
            trend
            & aligned_roc
            & aligned_macd
            & (body >= 0.55)
            & (roc_atr >= 0.20)
        )
        mean_reversion = pullback & extreme_rsi & aligned_roc
        volatility_expansion = (
            (direction_expert > 0)
            & (vola >= 1.20)
            & aligned_roc
            & (body >= 0.50)
            & (roc_atr >= 0.15)
        )
        output[breakout] = 3
        output[mean_reversion] = 4
        output[volatility_expansion] = 5

    groups = {}
    for mode in GROUP_MODES:
        groups[mode] = {}
        for direction in (1, 2):
            code = expert[direction].astype(np.int32)
            if mode == "expert_session":
                code = code * 10 + session
            elif mode == "expert_session_vola":
                code = code * 100 + session * 10 + volatility
            elif mode == "expert_session_rsi":
                code = code * 100 + session * 10 + rsi_group
            groups[mode][direction] = code

    return {
        "expert": expert,
        "family": family,
        "groups": groups,
    }


def empty_entries() -> pd.DataFrame:
    columns = [
        "index",
        "direction",
        "priority",
        "source",
        "expert",
        "family",
        *(f"group_{mode}" for mode in GROUP_MODES),
    ]
    return pd.DataFrame({column: pd.Series(dtype="float64") for column in columns})


def make_entries(
    indices: np.ndarray,
    direction: int,
    priorities: np.ndarray,
    source: int,
    metadata: dict,
) -> pd.DataFrame:
    if len(indices) == 0:
        return empty_entries()
    data = {
        "index": indices.astype(np.int64),
        "direction": np.full(len(indices), direction, dtype=np.int8),
        "priority": priorities.astype(np.float32),
        "source": np.full(len(indices), source, dtype=np.int8),
        "expert": metadata["expert"][direction][indices],
        "family": metadata["family"][direction][indices],
    }
    for mode in GROUP_MODES:
        data[f"group_{mode}"] = metadata["groups"][mode][direction][indices]
    return pd.DataFrame(data)


def parent_entries(
    signals: np.ndarray, scores: dict[int, np.ndarray], metadata: dict
) -> pd.DataFrame:
    pieces = []
    for direction in (1, 2):
        active = signals[:, direction] >= 0.5
        rising = active & ~np.r_[False, active[:-1]]
        indices = np.flatnonzero(rising)
        priorities = scores[direction][indices]
        finite = np.isfinite(priorities)
        pieces.append(
            make_entries(
                indices[finite],
                direction,
                priorities[finite],
                0,
                metadata,
            )
        )
    return pd.concat(pieces, ignore_index=True).sort_values("index")


def missed_opportunity_entries(
    frame: pd.DataFrame,
    features: list[str],
    scores: dict[int, np.ndarray],
    metadata: dict,
    parent: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    parent_ids = set(
        zip(
            parent["index"].astype(int),
            parent["direction"].astype(int),
        )
    )
    pieces = []
    raw_total = 0
    raw_overlap = 0
    for name, indices in executable_events_by_expert(frame, features).items():
        direction = 1 if name.startswith("long_") else 2
        priorities = scores[direction][indices]
        finite = np.isfinite(priorities)
        indices = indices[finite]
        priorities = priorities[finite]
        raw_total += len(indices)
        keep = np.fromiter(
            ((int(index), direction) not in parent_ids for index in indices),
            dtype=bool,
            count=len(indices),
        )
        raw_overlap += int((~keep).sum())
        pieces.append(
            make_entries(
                indices[keep],
                direction,
                priorities[keep],
                1,
                metadata,
            )
        )
    if not pieces:
        return empty_entries(), {"raw_events": 0, "raw_overlap": 0}
    entries = pd.concat(pieces, ignore_index=True)
    entries = entries.drop_duplicates(["index", "direction"]).sort_values("index")
    return entries, {"raw_events": raw_total, "raw_overlap": raw_overlap}


def adjusted_reward(
    frame: pd.DataFrame, index: int, direction: int, cost_points: float
) -> float:
    column = "LONG_REWARD" if direction == 1 else "SHORT_REWARD"
    reward = float(frame[column].iat[index])
    if cost_points == EXTRA_COST_POINTS:
        return reward
    stop_loss = max(float(frame["ATR"].iat[index]) * SL_ATR, MIN_SL_PRICE)
    extra_price = (cost_points - EXTRA_COST_POINTS) * 0.01
    return reward - extra_price / (stop_loss + SPREAD_POINTS * 0.01)


def execute_entries(
    frame: pd.DataFrame,
    entries: pd.DataFrame,
    period: str,
    cost_points: float = EXTRA_COST_POINTS,
) -> list[dict]:
    if entries.empty:
        return []
    ordered = entries.copy()
    ordered["parent_rank"] = (ordered["source"] != 0).astype(np.int8)
    ordered = ordered.sort_values(
        ["index", "parent_rank", "priority"],
        ascending=[True, True, False],
    )
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
        family_code = int(entry["family"])
        source = (
            "parent"
            if int(entry["source"]) == 0
            else FAMILY_LABELS.get(family_code, "unknown")
        )
        record = {
            "trade_id": f"{timestamp.isoformat()}|{direction}",
            "period": period,
            "index": index,
            "exit_index": index + exit_offset,
            "time": timestamp.isoformat(),
            "direction": direction,
            "expert": int(entry["expert"]),
            "family": family_code,
            "source": source,
            "outcome": int(frame[outcome_column].iat[index]),
            "reward": reward,
        }
        for mode in GROUP_MODES:
            record[f"group_{mode}"] = int(entry[f"group_{mode}"])
        records.append(record)
        free_index = index + exit_offset + 1
    return records


def profit_factor(rewards: np.ndarray) -> float | None:
    gains = float(rewards[rewards > 0.0].sum())
    losses = float(-rewards[rewards < 0.0].sum())
    return None if losses <= 0.0 else gains / losses


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
    rewards = np.asarray([record["reward"] for record in records], dtype=np.float64)
    wins = sum(record["outcome"] == 1 for record in records)
    losses = sum(record["outcome"] == 2 for record in records)
    timeouts = sum(record["outcome"] == 0 for record in records)
    return {
        "trades": len(records),
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "win_rate": wins / len(records),
        "profit_factor": profit_factor(rewards),
        "sum_r": float(rewards.sum()),
        "mean_r": float(rewards.mean()),
    }


def metrics(records: list[dict], frame: pd.DataFrame) -> dict:
    base = contribution(records)
    evaluated_days = int(frame["TIME_DT"].dt.date.nunique())
    balance = 1000.0
    peak = balance
    max_drawdown = 0.0
    pnl_values = []
    for record in records:
        pnl = balance * RISK_PER_TRADE * record["reward"]
        balance += pnl
        pnl_values.append(pnl)
        peak = max(peak, balance)
        max_drawdown = min(max_drawdown, balance / peak - 1.0)
    pnl_array = np.asarray(pnl_values, dtype=np.float64)
    by_direction = {}
    for direction, name in ((1, "long"), (2, "short")):
        by_direction[name] = contribution(
            [record for record in records if record["direction"] == direction]
        )
    by_expert = {}
    for code, name in EXPERT_LABELS.items():
        by_expert[name] = contribution(
            [record for record in records if record["expert"] == code]
        )
    by_source = {}
    for name in ("parent", *FAMILY_LABELS.values()):
        by_source[name] = contribution(
            [record for record in records if record["source"] == name]
        )
    return {
        **base,
        "evaluated_days": evaluated_days,
        "trades_per_day": base["trades"] / max(evaluated_days, 1),
        "pnl": float(pnl_array.sum()) if len(pnl_array) else 0.0,
        "ending_balance": balance,
        "max_drawdown_pct": max_drawdown,
        "take_profit_exits": sum(record["outcome"] == 1 for record in records),
        "stop_loss_exits": sum(record["outcome"] == 2 for record in records),
        "direction_contribution": by_direction,
        "expert_contribution": by_expert,
        "source_contribution": by_source,
    }


def compact_metrics(value: dict) -> dict:
    keys = (
        "trades",
        "evaluated_days",
        "trades_per_day",
        "wins",
        "losses",
        "timeouts",
        "take_profit_exits",
        "stop_loss_exits",
        "win_rate",
        "profit_factor",
        "pnl",
        "sum_r",
        "mean_r",
        "max_drawdown_pct",
    )
    return {key: value[key] for key in keys}


def group_stats(records: list[dict], field: str) -> dict[int, dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[int(record[field])].append(record)
    return {code: contribution(values) for code, values in grouped.items()}


def periods(records: list[dict]) -> set[str]:
    return {record["period"] for record in records}


def bad_groups(records: list[dict], spec: dict) -> tuple[set[int], dict]:
    field = f"group_{spec['mode']}"
    aggregate = group_stats(records, field)
    required_periods = min(2, len(periods(records)))
    minimum_period_events = max(4, spec["minimum_events"] // 4)
    output = set()
    for code, stats in aggregate.items():
        stable_periods = 0
        for period in periods(records):
            period_stats = contribution(
                [
                    record
                    for record in records
                    if record["period"] == period and record[field] == code
                ]
            )
            if (
                period_stats["trades"] >= minimum_period_events
                and period_stats["win_rate"] <= spec["maximum_win_rate"] + 0.05
                and period_stats["mean_r"] < 0.0
            ):
                stable_periods += 1
        pf = stats["profit_factor"]
        if (
            stats["trades"] >= spec["minimum_events"]
            and stats["win_rate"] <= spec["maximum_win_rate"]
            and pf is not None
            and pf < 1.0
            and stats["mean_r"] < 0.0
            and stable_periods >= required_periods
        ):
            output.add(code)
    return output, {str(code): aggregate[code] for code in sorted(output)}


def good_groups(records: list[dict], spec: dict) -> tuple[set[int], dict]:
    field = f"group_{spec['mode']}"
    aggregate = group_stats(records, field)
    required_periods = min(2, len(periods(records)))
    minimum_period_events = max(8, spec["minimum_events"] // 4)
    output = set()
    for code, stats in aggregate.items():
        stable_periods = 0
        for period in periods(records):
            period_stats = contribution(
                [
                    record
                    for record in records
                    if record["period"] == period and record[field] == code
                ]
            )
            period_pf = period_stats["profit_factor"]
            if (
                period_stats["trades"] >= minimum_period_events
                and period_stats["win_rate"] >= spec["target_win_rate"] - 0.05
                and (period_pf is None or period_pf >= 0.90)
                and period_stats["mean_r"] > 0.0
            ):
                stable_periods += 1
        pf = stats["profit_factor"]
        if (
            stats["trades"] >= spec["minimum_events"]
            and stats["win_rate"] >= spec["target_win_rate"]
            and (pf is None or pf > MIN_PROFIT_FACTOR)
            and stats["mean_r"] > 0.0
            and stable_periods >= required_periods
        ):
            output.add(code)
    return output, {str(code): aggregate[code] for code in sorted(output)}


def accepted_families(
    records: list[dict], spec: dict, groups: set[int]
) -> tuple[set[int], dict]:
    field = f"group_{spec['mode']}"
    selected = [record for record in records if record[field] in groups]
    output = set()
    diagnostics = {}
    for family_code, label in FAMILY_LABELS.items():
        family_records = [
            record for record in selected if record["family"] == family_code
        ]
        stats = contribution(family_records)
        diagnostics[label] = stats
        pf = stats["profit_factor"]
        if (
            stats["trades"] >= max(20, spec["minimum_events"] // 2)
            and stats["win_rate"] >= spec["target_win_rate"]
            and (pf is None or pf > MIN_PROFIT_FACTOR)
            and stats["mean_r"] > 0.0
        ):
            output.add(family_code)
    return output, diagnostics


def filter_entries_by_groups(
    entries: pd.DataFrame, spec: dict, groups: set[int]
) -> pd.DataFrame:
    if entries.empty or not groups:
        return entries.copy()
    field = f"group_{spec['mode']}"
    return entries.loc[~entries[field].isin(groups)].copy()


def top_k_by_day(
    entries: pd.DataFrame,
    frame: pd.DataFrame,
    spec: dict,
    profile: dict,
    groups: set[int],
    families: set[int],
) -> pd.DataFrame:
    if entries.empty or not groups or not families:
        return empty_entries()
    field = f"group_{spec['mode']}"
    selected = entries.loc[
        entries[field].isin(groups)
        & entries["family"].isin(families)
        & (entries["priority"] >= spec["minimum_expected_r"])
    ].copy()
    if selected.empty:
        return empty_entries()
    quality = {int(code): stats["win_rate"] for code, stats in profile.items()}
    selected["group_quality"] = selected[field].map(quality).fillna(0.0)
    selected["rank_score"] = selected["group_quality"] + 0.02 * np.tanh(
        selected["priority"].astype(float)
    )
    selected["date"] = frame["TIME_DT"].iloc[
        selected["index"].astype(int).to_numpy()
    ].dt.date.to_numpy()
    selected = (
        selected.sort_values(["date", "rank_score"], ascending=[True, False])
        .groupby("date", sort=False)
        .head(spec["top_k_per_day"])
    )
    return selected.drop(columns=["group_quality", "rank_score", "date"]).sort_values(
        "index"
    )


def direct_filter_diagnostics(
    records: list[dict], spec: dict | None, groups: set[int]
) -> dict:
    if spec is None:
        return {
            "baseline_winners": sum(record["reward"] > 0.0 for record in records),
            "baseline_losers": sum(record["reward"] <= 0.0 for record in records),
            "winners_accidentally_removed": 0,
            "losers_removed": 0,
        }
    field = f"group_{spec['mode']}"
    removed = [record for record in records if record[field] in groups]
    return {
        "baseline_winners": sum(record["reward"] > 0.0 for record in records),
        "baseline_losers": sum(record["reward"] <= 0.0 for record in records),
        "winners_accidentally_removed": sum(
            record["reward"] > 0.0 for record in removed
        ),
        "losers_removed": sum(record["reward"] <= 0.0 for record in removed),
    }


def compare_records(candidate: list[dict], baseline: list[dict]) -> dict:
    candidate_by_id = {record["trade_id"]: record for record in candidate}
    baseline_by_id = {record["trade_id"]: record for record in baseline}
    added_ids = set(candidate_by_id) - set(baseline_by_id)
    removed_ids = set(baseline_by_id) - set(candidate_by_id)
    return {
        "unique_executable_trades_added": len(added_ids),
        "unique_added_winners": sum(
            candidate_by_id[key]["reward"] > 0.0 for key in added_ids
        ),
        "unique_added_losers": sum(
            candidate_by_id[key]["reward"] <= 0.0 for key in added_ids
        ),
        "baseline_trades_removed_by_filter_or_occupancy": len(removed_ids),
        "losers_removed": sum(
            baseline_by_id[key]["reward"] <= 0.0 for key in removed_ids
        ),
        "winners_accidentally_removed": sum(
            baseline_by_id[key]["reward"] > 0.0 for key in removed_ids
        ),
        "execution_overlap": len(set(candidate_by_id) & set(baseline_by_id)),
    }


def compact_ledger(records: list[dict]) -> list[dict]:
    keys = (
        "trade_id",
        "period",
        "index",
        "exit_index",
        "time",
        "direction",
        "expert",
        "family",
        "source",
        "outcome",
        "reward",
        *(f"group_{mode}" for mode in GROUP_MODES),
    )
    return [{key: record[key] for key in keys} for record in records]


def compact_result(
    value: dict,
    comparison: dict,
    filter_diagnostic: dict,
    records: list[dict],
) -> dict:
    return {
        "metrics": compact_metrics(value),
        "comparison": comparison,
        "filter_diagnostics": filter_diagnostic,
        "direction_contribution": value["direction_contribution"],
        "expert_contribution": value["expert_contribution"],
        "source_contribution": value["source_contribution"],
        "trade_ledger": compact_ledger(records),
    }


def build_period_inputs(
    frame: pd.DataFrame,
    features: list[str],
    scores: dict[int, np.ndarray],
    period: str,
) -> dict:
    metadata = build_metadata(frame, features)
    signals, allocator_trace = rolling_score_cash_signals(frame, scores, PARENT)
    parent = parent_entries(signals, scores, metadata)
    parent_records = execute_entries(frame, parent, period)
    missed, overlap = missed_opportunity_entries(
        frame, features, scores, metadata, parent
    )
    missed_records = execute_entries(frame, missed, period)
    return {
        "metadata": metadata,
        "parent_entries": parent,
        "parent_records": parent_records,
        "missed_entries": missed,
        "missed_records": missed_records,
        "allocator_trace": allocator_trace,
        "overlap": {
            **overlap,
            "execution_overlap": len(
                {record["trade_id"] for record in parent_records}
                & {record["trade_id"] for record in missed_records}
            ),
        },
    }


def evaluate_candidates(
    frame: pd.DataFrame,
    period: str,
    period_inputs: dict,
    discovery_parent: list[dict],
    discovery_missed: list[dict],
    candidates: list[dict],
    include_cost_stress: bool = False,
) -> tuple[dict, dict, dict]:
    baseline_records = period_inputs["parent_records"]
    baseline_metrics = metrics(baseline_records, frame)
    filter_cache = {}
    for index, spec in enumerate(filter_specs()):
        groups, profile = bad_groups(discovery_parent, spec)
        entries = filter_entries_by_groups(
            period_inputs["parent_entries"], spec, groups
        )
        filter_cache[index] = {
            "entries": entries,
            "groups": groups,
            "profile": profile,
            "diagnostics": direct_filter_diagnostics(
                baseline_records, spec, groups
            ),
        }

    add_cache = {}
    for index, spec in enumerate(add_specs()):
        groups, profile = good_groups(discovery_missed, spec)
        families, family_diagnostics = accepted_families(
            discovery_missed, spec, groups
        )
        entries = top_k_by_day(
            period_inputs["missed_entries"],
            frame,
            spec,
            profile,
            groups,
            families,
        )
        add_cache[index] = {
            "entries": entries,
            "groups": groups,
            "profile": profile,
            "families": families,
            "family_diagnostics": family_diagnostics,
        }

    results = {}
    frozen_details = {}
    filters = filter_specs()
    additions = add_specs()
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        filter_spec = candidate["filter"]
        add_spec = candidate["addition"]
        if filter_spec is None:
            pieces = [period_inputs["parent_entries"]]
            filter_diagnostic = direct_filter_diagnostics(
                baseline_records, None, set()
            )
            filter_detail = None
        else:
            filter_index = filters.index(filter_spec)
            filter_item = filter_cache[filter_index]
            pieces = [filter_item["entries"]]
            filter_diagnostic = filter_item["diagnostics"]
            filter_detail = {
                "groups": sorted(filter_item["groups"]),
                "profile": filter_item["profile"],
            }
        add_detail = None
        if add_spec is not None:
            add_index = additions.index(add_spec)
            add_item = add_cache[add_index]
            pieces.append(add_item["entries"])
            add_detail = {
                "groups": sorted(add_item["groups"]),
                "profile": add_item["profile"],
                "families": [
                    FAMILY_LABELS[code] for code in sorted(add_item["families"])
                ],
                "family_diagnostics": add_item["family_diagnostics"],
            }
        nonempty_pieces = [piece for piece in pieces if not piece.empty]
        entries = (
            pd.concat(nonempty_pieces, ignore_index=True)
            if nonempty_pieces
            else empty_entries()
        )
        records = execute_entries(frame, entries, period)
        value = metrics(records, frame)
        comparison = compare_records(records, baseline_records)
        results[candidate_id] = compact_result(
            value, comparison, filter_diagnostic, records
        )
        frozen_details[candidate_id] = {
            "records": records,
            "entries": entries,
            "filter": filter_detail,
            "addition": add_detail,
        }
        if include_cost_stress:
            cost_records = execute_entries(frame, entries, period, cost_points=10.0)
            results[candidate_id]["cost_stress"] = compact_metrics(
                metrics(cost_records, frame)
            )
    return compact_metrics(baseline_metrics), results, frozen_details


def pf_value(value: float | None) -> float:
    return float("inf") if value is None else float(value)


def fold_pass(value: dict) -> bool:
    return bool(
        value["trades"] >= MIN_SELECTION_TRADES
        and value["win_rate"] >= TARGET_WIN_RATE
        and pf_value(value["profit_factor"]) > MIN_PROFIT_FACTOR
        and value["sum_r"] > 0.0
        and value["max_drawdown_pct"] >= MAX_DRAWDOWN_PCT
    )


def aggregate_candidate(candidate: dict, fold_results: dict) -> dict:
    values = [fold_results[name][candidate["candidate_id"]]["metrics"] for name, *_ in SELECTION_FOLDS]
    total_trades = sum(value["trades"] for value in values)
    total_wins = sum(value["wins"] for value in values)
    return {
        **candidate,
        "folds_passed": sum(fold_pass(value) for value in values),
        "qualified": all(fold_pass(value) for value in values),
        "minimum_fold_trades": min(value["trades"] for value in values),
        "total_trades": total_trades,
        "weighted_win_rate": total_wins / max(total_trades, 1),
        "minimum_win_rate": min(value["win_rate"] for value in values),
        "minimum_profit_factor": min(
            pf_value(value["profit_factor"]) for value in values
        ),
        "total_sum_r": sum(value["sum_r"] for value in values),
        "worst_drawdown_pct": min(value["max_drawdown_pct"] for value in values),
    }


def pareto_frontier(ranked: list[dict]) -> list[str]:
    feasible = [item for item in ranked if item["qualified"]]
    output = []
    for item in feasible:
        dominated = any(
            other["weighted_win_rate"] >= item["weighted_win_rate"]
            and other["total_trades"] >= item["total_trades"]
            and (
                other["weighted_win_rate"] > item["weighted_win_rate"]
                or other["total_trades"] > item["total_trades"]
            )
            for other in feasible
            if other["candidate_id"] != item["candidate_id"]
        )
        if not dominated:
            output.append(item["candidate_id"])
    return output


def select_candidate(candidates: list[dict], fold_results: dict) -> tuple[dict, list[dict]]:
    ranked = [aggregate_candidate(candidate, fold_results) for candidate in candidates]
    ranked.sort(
        key=lambda item: (
            item["qualified"],
            item["folds_passed"],
            item["minimum_fold_trades"] if item["qualified"] else item["minimum_win_rate"],
            item["total_trades"],
            item["weighted_win_rate"],
            item["minimum_profit_factor"],
            item["total_sum_r"],
        ),
        reverse=True,
    )
    return ranked[0], ranked


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


def legacy_baselines() -> dict:
    files = {
        "generation8": "gold_generation8_residual_walk_forward.json",
        "generation12": "gold_generation12_executable_events.json",
        "generation13": "gold_generation13_directional_exits.json",
        "generation14": "gold_generation14_precision_frequency.json",
    }
    output = {}
    for name, filename in files.items():
        path = PROJECT_ROOT / filename
        report = json.loads(path.read_text(encoding="utf-8"))
        output[name] = {
            "status": report.get("status"),
            "promotion_pass": bool(report.get("promotion_pass")),
            "folds": report["selected"]["folds"],
        }
    return output


def markdown_report(report: dict) -> str:
    lines = [
        "# GOLD generation 15 chronological signal mining",
        "",
        "Gen12 parent plus OOF loser-cluster filtering and missed-winner families.",
        "All results use non-overlapping first-touch events and fixed 1.4% fractional R execution.",
        "",
        (
            "## Parent baseline versus frozen candidate"
            if report["selection"]["qualified_count"]
            else "## Parent baseline versus diagnostic fallback"
        ),
        "",
        "| Period | Strategy | Trades | Trades/day | Wins | Losses | Timeouts | Win | PF | PnL | Mean-R | DD |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for period, item in report["comparison"].items():
        for strategy in ("parent", "candidate"):
            stats = item[strategy]
            pf = "inf" if stats["profit_factor"] is None else f"{stats['profit_factor']:.2f}"
            lines.append(
                f"| {period} | {strategy} | {stats['trades']} | "
                f"{stats['trades_per_day']:.3f} | {stats['wins']} | "
                f"{stats['losses']} | {stats['timeouts']} | "
                f"{stats['win_rate']:.2%} | {pf} | {stats['pnl']:.2f} | "
                f"{stats['mean_r']:.4f} | {stats['max_drawdown_pct']:.2%} |"
            )
    lines.extend(
        [
            "",
            f"Selection-qualified candidates: `{report['selection']['qualified_count']}`",
            f"Pareto candidates: `{json.dumps(report['selection']['pareto_frontier'])}`",
            (
                f"Frozen candidate: `{json.dumps(report['selected']['params'])}`"
                if report["selection"]["qualified_count"]
                else "Qualified candidate: `none`; the displayed fallback is not deployable"
            ),
            f"Simultaneous win/frequency improvement: `{report['selected']['simultaneous_improvement']}`",
            f"Research status: `{report['status']}`",
            "",
            "## False-positive and missed-winner diagnostics",
            "",
            "| Period | Unique added | Added winners | Added losers | Losers removed | Winners accidentally removed |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for period, diagnostics in report["selected"]["diagnostics"].items():
        comparison = diagnostics["comparison"]
        lines.append(
            f"| {period} | {comparison['unique_executable_trades_added']} | "
            f"{comparison['unique_added_winners']} | {comparison['unique_added_losers']} | "
            f"{comparison['losers_removed']} | "
            f"{comparison['winners_accidentally_removed']} |"
        )
    lines.extend(
        [
            "",
            "The candidate remains research_only and does not modify gemini.py.",
        ]
    )
    return "\n".join(lines) + "\n"


def self_check() -> None:
    rows = HORIZON + 20
    frame = pd.DataFrame(
        {
            "TIME_DT": pd.date_range("2026-01-05", periods=rows, freq="min"),
            "ATR": np.ones(rows),
            "LONG_REWARD": np.full(rows, 0.75),
            "SHORT_REWARD": np.full(rows, -1.0),
            "LONG_EXIT_OFFSET": np.full(rows, 3),
            "SHORT_EXIT_OFFSET": np.full(rows, 3),
            "LONG_OUTCOME": np.ones(rows),
            "SHORT_OUTCOME": np.full(rows, 2),
        }
    )
    metadata = {
        "expert": {1: np.ones(rows, dtype=np.int8), 2: np.full(rows, 3, dtype=np.int8)},
        "family": {1: np.ones(rows, dtype=np.int8), 2: np.ones(rows, dtype=np.int8)},
        "groups": {
            mode: {1: np.full(rows, 10), 2: np.full(rows, 20)} for mode in GROUP_MODES
        },
    }
    entries = pd.concat(
        [
            make_entries(np.array([0, 2, 5]), 1, np.array([1.0, 1.0, 1.0]), 0, metadata),
            make_entries(np.array([0]), 2, np.array([2.0]), 1, metadata),
        ],
        ignore_index=True,
    )
    records = execute_entries(frame, entries, "self_check")
    assert [record["index"] for record in records] == [0, 5]
    assert all(record["direction"] == 1 for record in records)
    value = metrics(records, frame)
    assert value["trades"] == 2 and value["wins"] == 2
    assert len(candidate_grid()) == 324
    print("generation15_self_check_ok")


def train_and_score(
    history: pd.DataFrame,
    features: list[str],
    train_cutoff: datetime,
    evaluation: pd.DataFrame,
) -> tuple[dict, dict[int, np.ndarray], dict]:
    train = training_frame(history, train_cutoff)
    models = train_executable_experts(train, features)
    scores, diagnostics = predict_executable_scores(models, evaluation, features)
    return models, scores, diagnostics


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0

    candidates = candidate_grid()
    history, features = prepare_barrier_data()
    history = add_targets(history)
    print(
        f"History rows={len(history):,} {history['TIME_DT'].iloc[0]} -> "
        f"{history['TIME_DT'].iloc[-1]}",
        flush=True,
    )

    initial = history[
        (history["TIME_DT"] >= INITIAL_DISCOVERY_START)
        & (history["TIME_DT"] < INITIAL_DISCOVERY_END)
    ].copy().reset_index(drop=True)
    initial_models, initial_scores, initial_diagnostics = train_and_score(
        history, features, INITIAL_DISCOVERY_START, initial
    )
    initial_inputs = build_period_inputs(
        initial, features, initial_scores, "2016_2017_discovery"
    )
    discovery_parent = list(initial_inputs["parent_records"])
    discovery_missed = list(initial_inputs["missed_records"])
    del initial_models, initial_scores, initial, initial_inputs
    gc.collect()

    fold_results = {}
    fold_baselines = {}
    fold_diagnostics = {}
    score_diagnostics = {"2016_2017_discovery": initial_diagnostics}
    for fold_name, fold_start, fold_end in SELECTION_FOLDS:
        validation = history[
            (history["TIME_DT"] >= fold_start)
            & (history["TIME_DT"] < fold_end)
        ].copy().reset_index(drop=True)
        models, scores, diagnostics = train_and_score(
            history, features, fold_start, validation
        )
        period_inputs = build_period_inputs(validation, features, scores, fold_name)
        baseline, results, _ = evaluate_candidates(
            validation,
            fold_name,
            period_inputs,
            discovery_parent,
            discovery_missed,
            candidates,
        )
        fold_results[fold_name] = results
        fold_baselines[fold_name] = baseline
        fold_diagnostics[fold_name] = {
            "allocator_trace": period_inputs["allocator_trace"],
            "overlap": period_inputs["overlap"],
            "discovery_parent_trades": len(discovery_parent),
            "discovery_missed_trades": len(discovery_missed),
            "parent_trade_ledger": compact_ledger(
                period_inputs["parent_records"]
            ),
        }
        score_diagnostics[fold_name] = diagnostics
        discovery_parent.extend(period_inputs["parent_records"])
        discovery_missed.extend(period_inputs["missed_records"])
        print(
            f"Fold {fold_name}: parent={baseline['trades']} "
            f"discovery_parent={len(discovery_parent)} "
            f"discovery_missed={len(discovery_missed)}",
            flush=True,
        )
        del models, scores, validation, period_inputs, results
        gc.collect()

    selected, ranked = select_candidate(candidates, fold_results)
    frontier = pareto_frontier(ranked)
    qualified_count = sum(item["qualified"] for item in ranked)
    selected_id = selected["candidate_id"]
    frozen = next(item for item in candidates if item["candidate_id"] == selected_id)

    holdout = history[history["TIME_DT"] >= HISTORICAL_HOLDOUT_START].copy().reset_index(drop=True)
    holdout_models, holdout_scores, holdout_diagnostics = train_and_score(
        history, features, HISTORICAL_HOLDOUT_START, holdout
    )
    holdout_inputs = build_period_inputs(
        holdout, features, holdout_scores, "2025_2026_05_holdout"
    )
    holdout_parent, holdout_results, holdout_details = evaluate_candidates(
        holdout,
        "2025_2026_05_holdout",
        holdout_inputs,
        discovery_parent,
        discovery_missed,
        [frozen],
    )
    holdout_parent_ledger = compact_ledger(holdout_inputs["parent_records"])
    discovery_parent.extend(holdout_inputs["parent_records"])
    discovery_missed.extend(holdout_inputs["missed_records"])
    score_diagnostics["2025_2026_05_holdout"] = holdout_diagnostics
    del holdout_models, holdout_scores, holdout_inputs
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
    final_models, recent_scores, recent_diagnostics = train_and_score(
        history, features, final_cutoff, recent
    )
    recent_inputs = build_period_inputs(recent, features, recent_scores, "2026_recent")
    recent_parent, recent_results, recent_details = evaluate_candidates(
        recent,
        "2026_recent",
        recent_inputs,
        discovery_parent,
        discovery_missed,
        [frozen],
        include_cost_stress=True,
    )
    recent_parent_ledger = compact_ledger(recent_inputs["parent_records"])
    recent_parent_cost_records = execute_entries(
        recent,
        recent_inputs["parent_entries"],
        "2026_recent_cost_10",
        cost_points=10.0,
    )
    recent_parent_cost = compact_metrics(metrics(recent_parent_cost_records, recent))
    current_stats = benchmark_current(recent, features)
    score_diagnostics["2026_recent"] = recent_diagnostics

    selected_fold_diagnostics = {
        fold_name: fold_results[fold_name][selected_id]
        for fold_name, *_ in SELECTION_FOLDS
    }
    selected_fold_diagnostics["2025_2026_05_holdout"] = holdout_results[selected_id]
    selected_fold_diagnostics["2026_recent"] = recent_results[selected_id]
    candidate_selection_metrics = {
        name: selected_fold_diagnostics[name]["metrics"]
        for name, *_ in SELECTION_FOLDS
    }
    parent_total_trades = sum(value["trades"] for value in fold_baselines.values())
    parent_total_wins = sum(value["wins"] for value in fold_baselines.values())
    parent_weighted_win = parent_total_wins / max(parent_total_trades, 1)
    simultaneous_improvement = bool(
        selected["total_trades"] > parent_total_trades
        and selected["weighted_win_rate"] > parent_weighted_win
    )
    selection_pass = bool(selected["qualified"] and simultaneous_improvement)

    comparison = {
        name: {
            "parent": fold_baselines[name],
            "candidate": candidate_selection_metrics[name],
        }
        for name, *_ in SELECTION_FOLDS
    }
    comparison["2025_2026_05_holdout"] = {
        "parent": holdout_parent,
        "candidate": holdout_results[selected_id]["metrics"],
    }
    comparison["2026_recent"] = {
        "parent": recent_parent,
        "candidate": recent_results[selected_id]["metrics"],
    }
    comparison["2026_recent_cost_10"] = {
        "parent": recent_parent_cost,
        "candidate": recent_results[selected_id]["cost_stress"],
    }

    model_files = save_models(final_models)
    recent_frozen_details = recent_details[selected_id]
    config = {
        "generation": "15_signal_mining",
        "status": "research_only",
        "parent": PARENT,
        "selected": frozen,
        "candidate_status": (
            "frozen_candidate" if qualified_count else "diagnostic_fallback"
        ),
        "selection_pass": selection_pass,
        "target_win_rate": TARGET_WIN_RATE,
        "execution": {
            "horizon": HORIZON,
            "tp_atr": 1.3,
            "sl_atr": 1.6,
            "base_extra_cost_points": EXTRA_COST_POINTS,
            "stress_cost_points": 10.0,
            "risk_per_trade": RISK_PER_TRADE,
            "single_position": True,
        },
        "recent_profiles": {
            "filter": recent_frozen_details["filter"],
            "addition": recent_frozen_details["addition"],
        },
        "model_profile": MODEL_PROFILE,
        "model_files": model_files,
        "promotion_pass": False,
    }
    CONFIG_FILE.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_only",
        "objective": "maximize executable frequency subject to OOS win_rate>=60%, PF>1 and positive expectancy",
        "data": {
            "history_rows": len(history),
            "history_start": history["TIME_DT"].iloc[0].isoformat(),
            "history_end": history["TIME_DT"].iloc[-1].isoformat(),
            "recent_rows": len(recent),
            "recent_start": recent["TIME_DT"].iloc[0].isoformat(),
            "recent_end": recent["TIME_DT"].iloc[-1].isoformat(),
        },
        "legacy_baselines": legacy_baselines(),
        "parent": {
            "params": PARENT,
            "selection_folds": fold_baselines,
            "weighted_win_rate": parent_weighted_win,
            "total_trades": parent_total_trades,
        },
        "selection": {
            "candidate_count": len(candidates),
            "qualified_count": qualified_count,
            "pareto_frontier": frontier,
            "ranked": ranked,
            "candidate_fold_results": fold_results,
        },
        "selected": {
            "status": (
                "frozen_candidate" if qualified_count else "diagnostic_fallback"
            ),
            "params": frozen,
            "selection_summary": selected,
            "selection_pass": selection_pass,
            "simultaneous_improvement": simultaneous_improvement,
            "diagnostics": selected_fold_diagnostics,
            "holdout_profiles": {
                "filter": holdout_details[selected_id]["filter"],
                "addition": holdout_details[selected_id]["addition"],
            },
            "recent_profiles": {
                "filter": recent_frozen_details["filter"],
                "addition": recent_frozen_details["addition"],
            },
        },
        "comparison": comparison,
        "baseline_trade_ledgers": {
            **{
                name: fold_diagnostics[name]["parent_trade_ledger"]
                for name, *_ in SELECTION_FOLDS
            },
            "2025_2026_05_holdout": holdout_parent_ledger,
            "2026_recent": recent_parent_ledger,
        },
        "fold_diagnostics": fold_diagnostics,
        "score_diagnostics": score_diagnostics,
        "benchmarks": {"production_recent": current_stats},
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
