from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import gold_generation16_independent_families as gen16
from barrier_final_train import prepare_barrier_data
from gold_generation11_execution_aligned import add_targets
from gold_recent_walk_forward import DEFAULT_TERMINAL, build_feature_frame
from gold_regime_experts_iterative import training_frame
from gold_regime_experts_walk_forward import RECENT_START, SELECTION_FOLDS


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_JSON = PROJECT_ROOT / "gold_generation17_cross_regime.json"
REPORT_MD = PROJECT_ROOT / "gold_generation17_cross_regime.md"
TRANSFER_JSON = PROJECT_ROOT / "gold_generation17_regime_transfer.json"
TRANSFER_MD = PROJECT_ROOT / "gold_generation17_regime_transfer.md"
CONFIG_FILE = PROJECT_ROOT / "gold_generation17_candidate.json"
GEN15_REPORT = PROJECT_ROOT / "gold_generation15_signal_mining.json"
GEN16_REPORT = PROJECT_ROOT / "gold_generation16_independent_families.json"

TARGET_EXPERTS = (
    "long_breakout",
    "short_breakout",
    "short_trend_continuation",
)
WINNING_REGIMES = {
    "long_breakout": "2021_2022",
    "short_breakout": "2021_2022",
    "short_trend_continuation": "2023_2024",
}
RESEARCH_WIN_RATE = 0.58
FINAL_TARGET_WIN_RATE = 0.60
WORST_FOLD_WIN_RATE = 0.50
MIN_FOLD_TRADES = 5
MAX_CATASTROPHIC_DRAWDOWN = -0.25
MIN_CATASTROPHIC_PF = 0.65
FIXED_TOP_K = 2
FIXED_MIN_PWIN = 0.60
FIXED_MIN_EXPECTED_R = 0.0
HISTORICAL_HOLDOUT_START = datetime(2025, 1, 1)
RECENT_WARMUP_START = datetime(2026, 4, 1, tzinfo=timezone.utc)
DEVELOPMENT_END = datetime(2026, 8, 31, tzinfo=timezone.utc)

MODEL_FILES = {
    (name, kind): PROJECT_ROOT / f"gold_generation17_{name}_{kind}_xgb.json"
    for name in TARGET_EXPERTS
    for kind in ("win", "mean_r")
}

REGIME_FEATURES = (
    "REG_ATR_Z_20D",
    "REG_ATR_PCTL_20D",
    "REG_RV_Z_20D",
    "REG_RV_PCTL_20D",
    "REG_VOL_TERM_5_1",
    "REG_VOL_TERM_15_5",
    "REG_VOL_TERM_60_15",
    "REG_TREND_EFF_30",
    "REG_TREND_EFF_120",
    "REG_RANGE_EXPANSION",
    "REG_DIST_HIGH_60_ATR",
    "REG_DIST_LOW_60_ATR",
    "REG_BODY_SIGNED",
    "REG_DIR_PERSIST_30",
    "REG_CONSEC_DIRECTION",
    "REG_MTF_ALIGNMENT",
    "REG_MTF_TRANSITION",
    "REG_SESSION_REL_ATR",
    "REG_PAST_LONG_BREAKOUT_WIN50",
    "REG_PAST_LONG_BREAKOUT_MEAN_R50",
    "REG_PAST_SHORT_BREAKOUT_WIN50",
    "REG_PAST_SHORT_BREAKOUT_MEAN_R50",
    "REG_PAST_SHORT_TREND_CONTINUATION_WIN50",
    "REG_PAST_SHORT_TREND_CONTINUATION_MEAN_R50",
)
ABSOLUTE_SCALE_FEATURES = ("ATR",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generation 17 cross-regime generalization research."
    )
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def rolling_zscore(
    values: pd.Series, window: int, minimum: int
) -> pd.Series:
    history = values.shift(1)
    mean = history.rolling(window, min_periods=minimum).mean()
    std = history.rolling(window, min_periods=minimum).std()
    return (values - mean) / std.replace(0.0, np.nan)


def rolling_percentile(
    values: pd.Series, window: int, minimum: int
) -> pd.Series:
    return values.rolling(window, min_periods=minimum).rank(pct=True)


def signed_run_length(values: pd.Series) -> np.ndarray:
    signs = np.sign(values.to_numpy(dtype=np.float64))
    output = np.zeros(len(signs), dtype=np.float32)
    run = 0
    previous = 0.0
    for index, sign in enumerate(signs):
        if not np.isfinite(sign) or sign == 0.0:
            run = 0
            previous = 0.0
            continue
        run = run + 1 if sign == previous else 1
        output[index] = float(run) * float(sign)
        previous = sign
    return output


def session_relative_atr(frame: pd.DataFrame, atr: pd.Series) -> np.ndarray:
    hours = frame["TIME_DT"].dt.hour.to_numpy(dtype=np.int8)
    values = atr.to_numpy(dtype=np.float64)
    reference = np.full(len(frame), np.nan, dtype=np.float64)
    for hour in range(24):
        indices = np.flatnonzero(hours == hour)
        if len(indices) == 0:
            continue
        history = pd.Series(values[indices]).shift(1)
        reference[indices] = history.ewm(
            span=1_440, min_periods=240, adjust=False
        ).mean().to_numpy(dtype=np.float64)
    return (values / np.maximum(reference, 1e-12)).astype(np.float32)


def add_past_family_state(
    frame: pd.DataFrame,
    base_features: list[str],
) -> list[str]:
    event_indices = gen16.expert_event_indices(frame, base_features)
    created = []
    for expert in TARGET_EXPERTS:
        direction = gen16.EXPERT_DIRECTION[expert]
        outcome_column = "LONG_OUTCOME" if direction == 1 else "SHORT_OUTCOME"
        reward_column = "LONG_REWARD" if direction == 1 else "SHORT_REWARD"
        exit_column = "LONG_EXIT_OFFSET" if direction == 1 else "SHORT_EXIT_OFFSET"
        indices = event_indices[expert]
        exits = frame[exit_column].to_numpy(dtype=np.int16)[indices]
        maturity = indices + exits + 1
        valid = maturity < len(frame)
        indices = indices[valid]
        maturity = maturity[valid]
        outcomes = frame[outcome_column].to_numpy(dtype=np.int8)[indices]
        rewards = frame[reward_column].to_numpy(dtype=np.float32)[indices]
        maturity_order = np.argsort(maturity, kind="stable")
        maturity = maturity[maturity_order]
        outcomes = outcomes[maturity_order]
        rewards = rewards[maturity_order]
        win_state = pd.Series((outcomes == 1).astype(np.float32)).rolling(
            50, min_periods=10
        ).mean().to_numpy(dtype=np.float32)
        reward_state = pd.Series(rewards).rolling(
            50, min_periods=10
        ).mean().to_numpy(dtype=np.float32)
        win_rows = np.full(len(frame), np.nan, dtype=np.float32)
        reward_rows = np.full(len(frame), np.nan, dtype=np.float32)
        maturity_last = np.r_[maturity[1:] != maturity[:-1], True]
        win_rows[maturity[maturity_last]] = win_state[maturity_last]
        reward_rows[maturity[maturity_last]] = reward_state[maturity_last]
        win_rows = pd.Series(win_rows).ffill().fillna(0.50).to_numpy(dtype=np.float32)
        reward_rows = pd.Series(reward_rows).ffill().fillna(0.0).to_numpy(dtype=np.float32)
        prefix = f"REG_PAST_{expert.upper()}"
        win_name = f"{prefix}_WIN50"
        reward_name = f"{prefix}_MEAN_R50"
        frame[win_name] = win_rows
        frame[reward_name] = reward_rows
        created.extend((win_name, reward_name))
    return created


def add_regime_features(
    frame: pd.DataFrame,
    base_features: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    result = frame.copy()
    atr = result["ATR"].astype(np.float64)
    known_close = result["CLOSE"].shift(1).astype(np.float64)
    known_open = result["OPEN"].shift(1).astype(np.float64)
    known_high = result["HIGH"].shift(1).astype(np.float64)
    known_low = result["LOW"].shift(1).astype(np.float64)
    returns_1 = np.log(known_close).diff()
    returns_5 = np.log(known_close).diff(5) / math.sqrt(5.0)
    returns_15 = np.log(known_close).diff(15) / math.sqrt(15.0)
    returns_60 = np.log(known_close).diff(60) / math.sqrt(60.0)
    rv_1 = returns_1.rolling(60, min_periods=30).std()
    rv_5 = returns_5.rolling(60, min_periods=30).std()
    rv_15 = returns_15.rolling(60, min_periods=30).std()
    rv_60 = returns_60.rolling(60, min_periods=30).std()
    window = 10_080
    minimum = 1_440

    result["REG_ATR_Z_20D"] = rolling_zscore(atr, window, minimum).astype(np.float32)
    result["REG_ATR_PCTL_20D"] = rolling_percentile(
        atr, window, minimum
    ).astype(np.float32)
    result["REG_RV_Z_20D"] = rolling_zscore(rv_1, window, minimum).astype(np.float32)
    result["REG_RV_PCTL_20D"] = rolling_percentile(
        rv_1, window, minimum
    ).astype(np.float32)
    result["REG_VOL_TERM_5_1"] = (rv_5 / rv_1.replace(0.0, np.nan)).astype(np.float32)
    result["REG_VOL_TERM_15_5"] = (rv_15 / rv_5.replace(0.0, np.nan)).astype(np.float32)
    result["REG_VOL_TERM_60_15"] = (rv_60 / rv_15.replace(0.0, np.nan)).astype(np.float32)

    absolute_move = known_close.diff().abs()
    for length in (30, 120):
        efficiency = known_close.diff(length).abs() / absolute_move.rolling(
            length, min_periods=length
        ).sum().replace(0.0, np.nan)
        result[f"REG_TREND_EFF_{length}"] = efficiency.astype(np.float32)
    known_range = known_high - known_low
    range_reference = known_range.shift(1).rolling(240, min_periods=60).median()
    result["REG_RANGE_EXPANSION"] = (
        known_range / range_reference.replace(0.0, np.nan)
    ).astype(np.float32)
    result["REG_DIST_HIGH_60_ATR"] = (
        (known_high.rolling(60, min_periods=30).max() - known_close)
        / atr.replace(0.0, np.nan)
    ).astype(np.float32)
    result["REG_DIST_LOW_60_ATR"] = (
        (known_close - known_low.rolling(60, min_periods=30).min())
        / atr.replace(0.0, np.nan)
    ).astype(np.float32)
    result["REG_BODY_SIGNED"] = (
        (known_close - known_open) / known_range.replace(0.0, np.nan)
    ).astype(np.float32)
    direction = np.sign(known_close.diff())
    result["REG_DIR_PERSIST_30"] = direction.rolling(
        30, min_periods=15
    ).mean().astype(np.float32)
    result["REG_CONSEC_DIRECTION"] = signed_run_length(direction)

    trend_columns = [
        name
        for name in base_features
        if name.endswith("_TREND")
        and name.startswith(
            (
                "H1_",
                "H2_",
                "H3_",
                "H4_",
                "H6_",
                "H8_",
                "H12_",
                "Daily_",
                "Weekly_",
                "Monthly_",
            )
        )
    ]
    alignment = result[trend_columns].mean(axis=1)
    result["REG_MTF_ALIGNMENT"] = alignment.astype(np.float32)
    result["REG_MTF_TRANSITION"] = alignment.diff().astype(np.float32)
    result["REG_SESSION_REL_ATR"] = session_relative_atr(result, atr)
    created = [name for name in REGIME_FEATURES if not name.startswith("REG_PAST_")]
    created.extend(add_past_family_state(result, base_features))
    if tuple(created) != REGIME_FEATURES:
        raise RuntimeError("Regime feature inventory changed")
    result[list(REGIME_FEATURES)] = result[list(REGIME_FEATURES)].replace(
        [np.inf, -np.inf], np.nan
    )
    return result, list(REGIME_FEATURES)


def train_target_experts(
    frame: pd.DataFrame,
    base_features: list[str],
    model_features: list[str],
) -> tuple[dict[str, dict], dict]:
    event_indices = gen16.expert_event_indices(frame, base_features)
    models = {}
    diagnostics = {}
    for name in TARGET_EXPERTS:
        indices = event_indices[name]
        direction = gen16.EXPERT_DIRECTION[name]
        outcome_column = "LONG_OUTCOME" if direction == 1 else "SHORT_OUTCOME"
        reward_column = "LONG_REWARD" if direction == 1 else "SHORT_REWARD"
        exit_column = "LONG_EXIT_OFFSET" if direction == 1 else "SHORT_EXIT_OFFSET"
        fit_end = int(len(indices) * gen16.MODEL_PROFILE["fit_ratio"])
        calibration_end = int(
            len(indices)
            * (
                gen16.MODEL_PROFILE["fit_ratio"]
                + gen16.MODEL_PROFILE["calibration_ratio"]
            )
        )
        if fit_end <= 0 or calibration_end >= len(indices):
            raise RuntimeError(f"{name} has insufficient chronological events")
        calibration_start = int(indices[fit_end])
        policy_start = int(indices[calibration_end])
        exits = frame[exit_column].to_numpy(dtype=np.int16)
        fit_indices = indices[
            (indices < calibration_start)
            & (indices + exits[indices] < calibration_start)
        ]
        calibration_indices = indices[
            (indices >= calibration_start)
            & (indices < policy_start)
            & (indices + exits[indices] < policy_start)
        ]
        policy_indices = indices[indices >= policy_start]
        if min(len(fit_indices), len(calibration_indices), len(policy_indices)) < 100:
            raise RuntimeError(
                f"{name} split too small: {len(fit_indices)}/"
                f"{len(calibration_indices)}/{len(policy_indices)}"
            )
        outcomes = frame[outcome_column].to_numpy(dtype=np.int8)
        rewards = frame[reward_column].to_numpy(dtype=np.float32)
        fit_target = (outcomes[fit_indices] == 1).astype(np.int8)
        calibration_target = (
            outcomes[calibration_indices] == 1
        ).astype(np.int8)
        if np.unique(fit_target).size != 2 or np.unique(calibration_target).size != 2:
            raise RuntimeError(f"{name} lacks a TP/non-TP class")
        classifier = gen16.new_classifier()
        mean_r_model = gen16.new_regressor()
        x_fit = frame.iloc[fit_indices][model_features].astype(np.float32)
        weights = gen16.sample_weights(
            frame["TIME_DT"].iloc[fit_indices], fit_target
        )
        classifier.fit(x_fit, fit_target, sample_weight=weights)
        mean_r_model.fit(x_fit, rewards[fit_indices], sample_weight=weights)
        raw_calibration = classifier.predict_proba(
            frame.iloc[calibration_indices][model_features].astype(np.float32)
        )[:, 1]
        calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        )
        calibrator.fit(raw_calibration, calibration_target)
        calibration_rewards = rewards[calibration_indices]
        average_win_r = float(
            calibration_rewards[calibration_target == 1].mean()
        )
        average_nonwin_r = float(
            calibration_rewards[calibration_target == 0].mean()
        )
        x_policy = frame.iloc[policy_indices][model_features].astype(np.float32)
        policy_probability = calibrator.predict(
            classifier.predict_proba(x_policy)[:, 1]
        ).astype(np.float32)
        probability_r = (
            policy_probability * average_win_r
            + (1.0 - policy_probability) * average_nonwin_r
        )
        policy_expected_r = 0.5 * (
            probability_r + mean_r_model.predict(x_policy).astype(np.float32)
        )
        selected = (
            (policy_probability >= FIXED_MIN_PWIN)
            & (policy_expected_r >= FIXED_MIN_EXPECTED_R)
        )
        policy_stats = gen16.array_stats(
            outcomes[policy_indices][selected],
            rewards[policy_indices][selected],
        )
        models[name] = {
            "win": classifier,
            "mean_r": mean_r_model,
            "calibrator": calibrator,
            "average_win_r": average_win_r,
            "average_nonwin_r": average_nonwin_r,
            "accepted_contexts": set(),
            "context_profile": {},
        }
        diagnostics[name] = {
            "events": len(indices),
            "fit": len(fit_indices),
            "calibration": len(calibration_indices),
            "policy": len(policy_indices),
            "fit_max_label_end_index": int(
                np.max(fit_indices + exits[fit_indices])
            ),
            "calibration_start_index": calibration_start,
            "calibration_max_label_end_index": int(
                np.max(calibration_indices + exits[calibration_indices])
            ),
            "policy_start_index": policy_start,
            "policy_stats": policy_stats,
            "top_gain_features": sorted(
                classifier.get_booster().get_score(
                    importance_type="gain"
                ).items(),
                key=lambda item: item[1],
                reverse=True,
            )[:15],
        }
        print(
            f"  {name}: events={len(indices):,} fit={len(fit_indices):,} "
            f"cal={len(calibration_indices):,} policy={len(policy_indices):,} "
            f"policy_selected={policy_stats['trades']}",
            flush=True,
        )
    return models, diagnostics


def score_target_experts(
    models: dict[str, dict],
    frame: pd.DataFrame,
    base_features: list[str],
    model_features: list[str],
) -> tuple[dict[str, pd.DataFrame], dict]:
    masks = gen16.family_masks(frame, base_features)
    contexts = gen16.context_codes(frame)
    scored = {}
    diagnostics = {}
    for name in TARGET_EXPERTS:
        indices = np.flatnonzero(gen16.rising_edges(masks[name]))
        x = frame.iloc[indices][model_features].astype(np.float32)
        model = models[name]
        probability = model["calibrator"].predict(
            model["win"].predict_proba(x)[:, 1]
        ).astype(np.float32)
        probability_r = (
            probability * model["average_win_r"]
            + (1.0 - probability) * model["average_nonwin_r"]
        )
        mean_r = model["mean_r"].predict(x).astype(np.float32)
        expected_r = 0.5 * (probability_r + mean_r)
        selected = (
            (probability >= FIXED_MIN_PWIN)
            & (expected_r >= FIXED_MIN_EXPECTED_R)
        )
        chosen = indices[selected]
        direction = gen16.EXPERT_DIRECTION[name]
        family = name.split("_", 1)[1]
        scored[name] = pd.DataFrame(
            {
                "index": chosen.astype(np.int64),
                "direction": np.full(len(chosen), direction, dtype=np.int8),
                "priority": (
                    probability[selected]
                    + 0.05 * np.tanh(expected_r[selected])
                ).astype(np.float32),
                "expert": np.full(len(chosen), name, dtype=object),
                "family": np.full(len(chosen), family, dtype=object),
                "context": contexts[chosen],
                "p_win": probability[selected],
                "expected_r": expected_r[selected],
            }
        )
        diagnostics[name] = {
            "stage1_rows": len(indices),
            "stage2_rows": len(chosen),
            "p_win_quantiles": np.quantile(
                probability, (0.50, 0.90, 0.99)
            ).round(5).tolist(),
        }
    return scored, diagnostics


def candidate_grid() -> list[dict]:
    subsets = {
        "long_breakout": ("long_breakout",),
        "short_breakout": ("short_breakout",),
        "short_trend_continuation": ("short_trend_continuation",),
        "breakout_pair": ("long_breakout", "short_breakout"),
        "target_portfolio": TARGET_EXPERTS,
    }
    return [
        {
            "candidate_id": f"gen17_{label}",
            "label": label,
            "experts": experts,
            "context_mode": "none",
            "top_k_per_expert_day": FIXED_TOP_K,
            "minimum_p_win": FIXED_MIN_PWIN,
            "minimum_expected_r": FIXED_MIN_EXPECTED_R,
        }
        for label, experts in subsets.items()
    ]


def pooled_metrics(values: list[dict], key: str) -> dict:
    ledgers = [record for value in values for record in value[key]]
    rewards = np.asarray(
        [record["reward"] for record in ledgers], dtype=np.float64
    )
    outcomes = np.asarray(
        [record["outcome"] for record in ledgers], dtype=np.int8
    )
    trades = len(ledgers)
    wins = int((outcomes == 1).sum())
    losses = int((outcomes == 2).sum())
    timeouts = int((outcomes == 0).sum())
    days = sum(value["metrics"]["evaluated_days"] for value in values)
    balance = 1000.0
    peak = balance
    maximum_drawdown = 0.0
    pnl = 0.0
    for reward in rewards:
        trade_pnl = balance * gen16.RISK_PER_TRADE * reward
        balance += trade_pnl
        pnl += trade_pnl
        peak = max(peak, balance)
        maximum_drawdown = min(maximum_drawdown, balance / peak - 1.0)
    return {
        "trades": trades,
        "evaluated_days": days,
        "trades_per_day": trades / max(days, 1),
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "win_rate": wins / max(trades, 1),
        "profit_factor": gen16.profit_factor(rewards),
        "pnl": pnl,
        "sum_r": float(rewards.sum()) if trades else 0.0,
        "mean_r": float(rewards.mean()) if trades else 0.0,
        "max_drawdown_pct": maximum_drawdown,
        "direction_contribution": {
            label: gen16.contribution(
                [
                    record
                    for record in ledgers
                    if record["direction"] == direction
                ]
            )
            for direction, label in ((1, "long"), (2, "short"))
        },
        "expert_contribution": {
            expert: gen16.contribution(
                [record for record in ledgers if record["expert"] == expert]
            )
            for expert in TARGET_EXPERTS
        },
    }


def aggregate_candidates(
    candidates: list[dict], fold_results: dict
) -> list[dict]:
    ranked = []
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        values = [
            fold_results[name][candidate_id]
            for name, *_ in SELECTION_FOLDS
        ]
        pooled = pooled_metrics(values, "trade_ledger")
        pooled_stress = pooled_metrics(values, "cost_stress_trade_ledger")
        fold_win_rates = [value["metrics"]["win_rate"] for value in values]
        catastrophic = any(
            value["metrics"]["trades"] < MIN_FOLD_TRADES
            or value["metrics"]["win_rate"] < WORST_FOLD_WIN_RATE
            or value["metrics"]["max_drawdown_pct"] < MAX_CATASTROPHIC_DRAWDOWN
            or (
                value["metrics"]["profit_factor"] is not None
                and value["metrics"]["profit_factor"] < MIN_CATASTROPHIC_PF
            )
            for value in values
        )
        unique_added = sum(
            value["comparison"]["unique_executable_trades_added"]
            for value in values
        )
        losers_removed = sum(
            value["comparison"]["losers_removed"] for value in values
        )
        winners_removed = sum(
            value["comparison"]["winners_accidentally_removed"]
            for value in values
        )
        unique_added_winners = sum(
            value["comparison"]["unique_added_winners"] for value in values
        )
        unique_added_losers = sum(
            value["comparison"]["unique_added_losers"] for value in values
        )
        discovery_pass = bool(
            pooled["win_rate"] >= RESEARCH_WIN_RATE
            and min(fold_win_rates) >= WORST_FOLD_WIN_RATE
            and pooled["profit_factor"] is not None
            and pooled["profit_factor"] > 1.0
            and pooled["mean_r"] > 0.0
            and pooled_stress["profit_factor"] is not None
            and pooled_stress["profit_factor"] > 1.0
            and pooled_stress["mean_r"] > 0.0
            and not catastrophic
            and (unique_added > 0 or losers_removed > 0)
        )
        ranked.append(
            {
                **candidate,
                "discovery_pass": discovery_pass,
                "pooled": pooled,
                "pooled_cost_stress": pooled_stress,
                "worst_fold_win_rate": min(fold_win_rates),
                "fold_win_rate_std": float(np.std(fold_win_rates)),
                "catastrophic_fold": catastrophic,
                "unique_executable_trades_added": unique_added,
                "unique_added_winners": unique_added_winners,
                "unique_added_losers": unique_added_losers,
                "losers_removed": losers_removed,
                "winners_accidentally_removed": winners_removed,
                "production_win_target_met": pooled["win_rate"]
                >= FINAL_TARGET_WIN_RATE,
            }
        )
    ranked.sort(
        key=lambda item: (
            item["discovery_pass"],
            item["pooled"]["win_rate"],
            item["pooled"]["trades_per_day"],
            item["pooled"]["trades"],
        ),
        reverse=True,
    )
    return ranked


def pareto_frontier(ranked: list[dict]) -> list[str]:
    feasible = [item for item in ranked if item["discovery_pass"]]
    output = []
    for item in feasible:
        dominated = any(
            other["pooled"]["win_rate"] >= item["pooled"]["win_rate"]
            and other["pooled"]["trades_per_day"]
            >= item["pooled"]["trades_per_day"]
            and (
                other["pooled"]["win_rate"] > item["pooled"]["win_rate"]
                or other["pooled"]["trades_per_day"]
                > item["pooled"]["trades_per_day"]
            )
            for other in feasible
            if other["candidate_id"] != item["candidate_id"]
        )
        if not dominated:
            output.append(item["candidate_id"])
    return output


def describe_values(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {"count": 0, "q25": None, "median": None, "q75": None}
    q25, median, q75 = np.quantile(finite, (0.25, 0.50, 0.75))
    return {
        "count": len(finite),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
    }


def selected_ledger_feature_rows(
    report16: dict,
    history: pd.DataFrame,
) -> pd.DataFrame:
    pieces = []
    for fold_name, fold_start, fold_end in SELECTION_FOLDS:
        evaluation = history[
            (history["TIME_DT"] >= fold_start)
            & (history["TIME_DT"] < fold_end)
        ].reset_index(drop=True)
        for expert in TARGET_EXPERTS:
            candidate_id = f"{expert}__none__top2"
            ledger = report16["selection"]["candidate_fold_results"][
                fold_name
            ][candidate_id]["trade_ledger"]
            if not ledger:
                continue
            indices = np.asarray(
                [int(record["index"]) for record in ledger], dtype=np.int64
            )
            values = evaluation.iloc[indices][list(REGIME_FEATURES)].copy()
            values["fold"] = fold_name
            values["expert"] = expert
            values["outcome"] = [int(record["outcome"]) for record in ledger]
            values["reward"] = [float(record["reward"]) for record in ledger]
            values["trade_id"] = [record["trade_id"] for record in ledger]
            pieces.append(values)
    if not pieces:
        raise RuntimeError("No Generation 16 target-family ledgers found")
    return pd.concat(pieces, ignore_index=True)


def stage1_feature_rows(
    history: pd.DataFrame,
    base_features: list[str],
) -> pd.DataFrame:
    pieces = []
    for fold_name, fold_start, fold_end in SELECTION_FOLDS:
        evaluation = history[
            (history["TIME_DT"] >= fold_start)
            & (history["TIME_DT"] < fold_end)
        ].reset_index(drop=True)
        masks = gen16.family_masks(evaluation, base_features)
        contexts = gen16.context_codes(evaluation)
        for expert in TARGET_EXPERTS:
            indices = np.flatnonzero(gen16.rising_edges(masks[expert]))
            direction = gen16.EXPERT_DIRECTION[expert]
            entries = pd.DataFrame(
                {
                    "index": indices,
                    "direction": np.full(len(indices), direction),
                    "priority": np.zeros(len(indices)),
                    "expert": np.full(len(indices), expert, dtype=object),
                    "family": np.full(
                        len(indices), expert.split("_", 1)[1], dtype=object
                    ),
                    "context": contexts[indices],
                    "p_win": np.zeros(len(indices)),
                    "expected_r": np.zeros(len(indices)),
                }
            )
            ledger = gen16.execute_entries(
                evaluation, entries, f"{fold_name}_{expert}_stage1"
            )
            ledger_indices = np.asarray(
                [int(record["index"]) for record in ledger], dtype=np.int64
            )
            values = evaluation.iloc[ledger_indices][list(REGIME_FEATURES)].copy()
            values["fold"] = fold_name
            values["expert"] = expert
            values["outcome"] = [int(record["outcome"]) for record in ledger]
            values["reward"] = [float(record["reward"]) for record in ledger]
            values["trade_id"] = [record["trade_id"] for record in ledger]
            pieces.append(values)
    return pd.concat(pieces, ignore_index=True)


def regime_transfer_report(
    selected_rows: pd.DataFrame,
    stage1_rows: pd.DataFrame,
) -> dict:
    report = {
        "feature_availability": {
            name: "entry_time_past_only" for name in REGIME_FEATURES
        },
        "families": {},
    }
    for expert in TARGET_EXPERTS:
        family_rows = selected_rows[selected_rows["expert"] == expert]
        stage1_family_rows = stage1_rows[stage1_rows["expert"] == expert]
        winning_fold = WINNING_REGIMES[expert]
        winning_rows = family_rows[family_rows["fold"] == winning_fold]
        other_losers = stage1_family_rows[
            (stage1_family_rows["fold"] != winning_fold)
            & (stage1_family_rows["outcome"] != 1)
        ]
        fold_report = {}
        stage1_fold_report = {}
        for fold_name, *_ in SELECTION_FOLDS:
            fold_rows = family_rows[family_rows["fold"] == fold_name]
            rewards = fold_rows["reward"].to_numpy(dtype=np.float64)
            outcomes = fold_rows["outcome"].to_numpy(dtype=np.int8)
            fold_report[fold_name] = {
                **gen16.array_stats(outcomes, rewards),
                "feature_distribution": {
                    name: describe_values(
                        fold_rows[name].to_numpy(dtype=np.float64)
                    )
                    for name in REGIME_FEATURES
                },
            }
            stage1_fold_rows = stage1_family_rows[
                stage1_family_rows["fold"] == fold_name
            ]
            stage1_rewards = stage1_fold_rows["reward"].to_numpy(
                dtype=np.float64
            )
            stage1_outcomes = stage1_fold_rows["outcome"].to_numpy(
                dtype=np.int8
            )
            stage1_fold_report[fold_name] = {
                **gen16.array_stats(stage1_outcomes, stage1_rewards),
                "feature_distribution": {
                    name: describe_values(
                        stage1_fold_rows[name].to_numpy(dtype=np.float64)
                    )
                    for name in REGIME_FEATURES
                },
            }

        feature_comparison = {}
        reproducible = []
        for name in REGIME_FEATURES:
            pooled = stage1_family_rows[name].to_numpy(dtype=np.float64)
            finite = pooled[np.isfinite(pooled)]
            scale = (
                float(np.subtract(*np.quantile(finite, (0.75, 0.25))))
                if len(finite)
                else 0.0
            )
            winning_median = describe_values(
                winning_rows[name].to_numpy(dtype=np.float64)
            )["median"]
            other_loser_median = describe_values(
                other_losers[name].to_numpy(dtype=np.float64)
            )["median"]
            transfer_effects = {}
            stable_signs = []
            for fold_name, *_ in SELECTION_FOLDS:
                fold_rows = stage1_family_rows[
                    stage1_family_rows["fold"] == fold_name
                ]
                wins = fold_rows[fold_rows["outcome"] == 1][name].to_numpy(
                    dtype=np.float64
                )
                nonwins = fold_rows[fold_rows["outcome"] != 1][name].to_numpy(
                    dtype=np.float64
                )
                win_stat = describe_values(wins)
                nonwin_stat = describe_values(nonwins)
                effect = None
                if (
                    win_stat["median"] is not None
                    and nonwin_stat["median"] is not None
                    and scale > 1e-12
                ):
                    effect = (
                        win_stat["median"] - nonwin_stat["median"]
                    ) / scale
                    if len(wins) >= 3 and len(nonwins) >= 3 and abs(effect) >= 0.15:
                        stable_signs.append(1 if effect > 0.0 else -1)
                transfer_effects[fold_name] = {
                    "winner_median": win_stat["median"],
                    "nonwinner_median": nonwin_stat["median"],
                    "iqr_scaled_effect": effect,
                }
            winning_vs_other_losers = (
                None
                if winning_median is None
                or other_loser_median is None
                or scale <= 1e-12
                else (winning_median - other_loser_median) / scale
            )
            reproduced = bool(
                len(stable_signs) >= 2
                and abs(sum(stable_signs)) == len(stable_signs)
                and winning_vs_other_losers is not None
                and abs(winning_vs_other_losers) >= 0.25
                and (
                    1 if winning_vs_other_losers > 0.0 else -1
                ) == stable_signs[0]
            )
            if reproduced:
                reproducible.append(name)
            feature_comparison[name] = {
                "winning_regime_median": winning_median,
                "other_fold_loser_median": other_loser_median,
                "winning_selected_vs_other_loser_iqr_effect": (
                    winning_vs_other_losers
                ),
                "winner_vs_nonwinner_by_fold": transfer_effects,
                "reproduced_in_at_least_two_folds": reproduced,
            }
        report["families"][expert] = {
            "winning_regime": winning_fold,
            "selected_signal_folds": fold_report,
            "stage1_executable_family_folds": stage1_fold_report,
            "feature_comparison": feature_comparison,
            "reproducible_entry_features": reproducible,
            "generalizable_entry_condition_found": bool(reproducible),
        }
    return report


def reference_gen16_summary(report16: dict) -> dict:
    output = {}
    for expert in TARGET_EXPERTS:
        values = [
            report16["selection"]["candidate_fold_results"][fold_name][
                f"{expert}__none__top2"
            ]
            for fold_name, *_ in SELECTION_FOLDS
        ]
        output[expert] = {
            "pooled": pooled_metrics(values, "trade_ledger"),
            "folds": {
                fold_name: report16["selection"]["candidate_fold_results"][
                    fold_name
                ][f"{expert}__none__top2"]["metrics"]
                for fold_name, *_ in SELECTION_FOLDS
            },
        }
    return output


def save_models(models: dict[str, dict]) -> dict:
    output = {}
    for name, model in models.items():
        model["win"].save_model(MODEL_FILES[(name, "win")])
        model["mean_r"].save_model(MODEL_FILES[(name, "mean_r")])
        output[name] = {
            "win_file": MODEL_FILES[(name, "win")].name,
            "mean_r_file": MODEL_FILES[(name, "mean_r")].name,
            "isotonic": {
                "x": model["calibrator"].X_thresholds_.tolist(),
                "y": model["calibrator"].y_thresholds_.tolist(),
            },
        }
    return output


def dataset_manifest() -> list[dict]:
    manifest = []
    for path in sorted(PROJECT_ROOT.glob("GOLD#_*.csv")):
        stat = path.stat()
        manifest.append(
            {
                "file": path.name,
                "size_bytes": stat.st_size,
                "modified_utc": datetime.fromtimestamp(
                    stat.st_mtime, timezone.utc
                ).isoformat(),
            }
        )
    return manifest


def markdown_report(report: dict) -> str:
    lines = [
        "# GOLD Generation 17 - Cross-Regime Generalization",
        "",
        "Only long breakout, short breakout, and short trend-continuation are studied.",
        "No confidence-threshold sweep and no new signal family were used.",
        "",
        "## Discovery Pareto",
        "",
        "| Candidate | Trades | Trades/day | Pooled win | Worst fold | PF | Mean-R | Max DD | Stress PF | Discovery |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["selection"]["ranked"]:
        pooled = item["pooled"]
        stress = item["pooled_cost_stress"]
        lines.append(
            f"| {item['candidate_id']} | {pooled['trades']} | "
            f"{pooled['trades_per_day']:.3f} | {pooled['win_rate']:.2%} | "
            f"{item['worst_fold_win_rate']:.2%} | "
            f"{pooled['profit_factor'] or 0.0:.2f} | {pooled['mean_r']:.4f} | "
            f"{pooled['max_drawdown_pct']:.2%} | "
            f"{stress['profit_factor'] or 0.0:.2f} | "
            f"{'PASS' if item['discovery_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"Discovery-qualified: `{report['selection']['qualified_count']}`",
            f"Pareto frontier: `{json.dumps(report['selection']['pareto_frontier'])}`",
            f"Frozen status: `{report['selected']['status']}`",
            "Production promotion: `False`",
            "",
            "The 2025 holdout and 2026 recent data are development/diagnostic only.",
        ]
    )
    return "\n".join(lines) + "\n"


def transfer_markdown(report: dict) -> str:
    lines = [
        "# Generation 17 regime-transfer report",
        "",
        "| Expert | Fold | Trades | Win | PF | Mean-R |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for expert, family in report["families"].items():
        for fold, value in family["selected_signal_folds"].items():
            pf = value["profit_factor"] or 0.0
            lines.append(
                f"| {expert} | {fold} | {value['trades']} | "
                f"{value['win_rate']:.2%} | {pf:.2f} | {value['mean_r']:.4f} |"
            )
        reproduced = family["reproducible_entry_features"]
        lines.append(
            f"| {expert} | reproducible features | {len(reproduced)} | - | - | "
            f"{', '.join(reproduced) if reproduced else 'none'} |"
        )
        for fold, value in family["stage1_executable_family_folds"].items():
            pf = value["profit_factor"] or 0.0
            lines.append(
                f"| {expert} stage1 | {fold} | {value['trades']} | "
                f"{value['win_rate']:.2%} | {pf:.2f} | "
                f"{value['mean_r']:.4f} |"
            )
    lines.extend(
        [
            "",
            "Full feature distributions, drift, and winner/non-winner effects are in the JSON report.",
        ]
    )
    return "\n".join(lines) + "\n"


def self_check() -> None:
    assert len(candidate_grid()) == 5
    assert set(WINNING_REGIMES) == set(TARGET_EXPERTS)
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    zscore = rolling_zscore(values, 3, 2)
    assert math.isclose(float(zscore.iloc[-1]), 2.0)
    run = signed_run_length(pd.Series([1.0, 1.0, -1.0, -1.0, 0.0]))
    assert run.tolist() == [1.0, 2.0, -1.0, -2.0, 0.0]
    print("generation17_self_check_ok")


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0
    report15 = json.loads(GEN15_REPORT.read_text(encoding="utf-8"))
    report16 = json.loads(GEN16_REPORT.read_text(encoding="utf-8"))
    history, base_features = prepare_barrier_data()
    history = add_targets(history)
    history, regime_features = add_regime_features(history, base_features)
    robust_base_features = [
        name for name in base_features if name not in ABSOLUTE_SCALE_FEATURES
    ]
    model_features = [*robust_base_features, *regime_features]
    print(
        f"History={len(history):,} features={len(model_features)} "
        f"regime_features={len(regime_features)}",
        flush=True,
    )

    transfer_rows = selected_ledger_feature_rows(report16, history)
    stage1_rows = stage1_feature_rows(history, base_features)
    transfer = regime_transfer_report(transfer_rows, stage1_rows)
    transfer.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_generation": "16_independent_families",
            "development_only": True,
            "target_experts": list(TARGET_EXPERTS),
            "regime_features": list(REGIME_FEATURES),
        }
    )
    TRANSFER_JSON.write_text(
        json.dumps(transfer, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    TRANSFER_MD.write_text(transfer_markdown(transfer), encoding="utf-8")

    candidates = candidate_grid()
    fold_results = {}
    model_diagnostics = {}
    score_diagnostics = {}
    for fold_name, fold_start, fold_end in SELECTION_FOLDS:
        train = training_frame(history, fold_start)
        evaluation = history[
            (history["TIME_DT"] >= fold_start)
            & (history["TIME_DT"] < fold_end)
        ].copy().reset_index(drop=True)
        models, diagnostics = train_target_experts(
            train, base_features, model_features
        )
        scored, score_detail = score_target_experts(
            models, evaluation, base_features, model_features
        )
        baseline = gen16.baseline_records(report15, fold_name)
        fold_results[fold_name] = gen16.evaluate_candidates(
            evaluation,
            fold_name,
            models,
            scored,
            candidates,
            baseline,
        )
        model_diagnostics[fold_name] = diagnostics
        score_diagnostics[fold_name] = score_detail
        print(f"Fold {fold_name} complete", flush=True)

    ranked = aggregate_candidates(candidates, fold_results)
    qualified = [item for item in ranked if item["discovery_pass"]]
    selected_summary = qualified[0] if qualified else ranked[0]
    selected_id = selected_summary["candidate_id"]
    selected = next(
        candidate
        for candidate in candidates
        if candidate["candidate_id"] == selected_id
    )
    selected_results = {
        fold_name: fold_results[fold_name][selected_id]
        for fold_name, *_ in SELECTION_FOLDS
    }

    holdout = history[
        history["TIME_DT"] >= HISTORICAL_HOLDOUT_START
    ].copy().reset_index(drop=True)
    holdout_train = training_frame(history, HISTORICAL_HOLDOUT_START)
    holdout_models, holdout_model_detail = train_target_experts(
        holdout_train, base_features, model_features
    )
    holdout_scored, holdout_score_detail = score_target_experts(
        holdout_models, holdout, base_features, model_features
    )
    holdout_result = gen16.evaluate_candidates(
        holdout,
        "2025_2026_05_development",
        holdout_models,
        holdout_scored,
        [selected],
        gen16.baseline_records(report15, "2025_2026_05_holdout"),
    )[selected_id]

    if not mt5.initialize(path=str(DEFAULT_TERMINAL), timeout=10_000):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        recent_all, recent_features = build_feature_frame(
            RECENT_WARMUP_START, DEVELOPMENT_END
        )
    finally:
        mt5.shutdown()
    if set(base_features) != set(recent_features):
        raise RuntimeError("Historical and MT5 base feature sets differ")
    recent_all = add_targets(recent_all)
    recent_all, _ = add_regime_features(recent_all, base_features)
    comparable_recent_end = pd.Timestamp(
        report15["data"]["recent_end"]
    ) + pd.Timedelta(minutes=1)
    recent = recent_all[
        (recent_all["TIME_DT"] >= RECENT_START.replace(tzinfo=None))
        & (recent_all["TIME_DT"] < comparable_recent_end)
    ].copy().reset_index(drop=True)
    final_cutoff = history["TIME_DT"].iloc[-1] + pd.Timedelta(minutes=1)
    final_train = training_frame(history, final_cutoff)
    final_models, final_model_detail = train_target_experts(
        final_train, base_features, model_features
    )
    recent_scored, recent_score_detail = score_target_experts(
        final_models, recent, base_features, model_features
    )
    recent_result = gen16.evaluate_candidates(
        recent,
        "2026_recent_development",
        final_models,
        recent_scored,
        [selected],
        gen16.baseline_records(report15, "2026_recent"),
    )[selected_id]
    selected_results["2025_2026_05_development"] = holdout_result
    selected_results["2026_recent_development"] = recent_result
    model_diagnostics["2025_2026_05_development"] = holdout_model_detail
    model_diagnostics["2026_recent_development"] = final_model_detail
    score_diagnostics["2025_2026_05_development"] = holdout_score_detail
    score_diagnostics["2026_recent_development"] = recent_score_detail

    model_files = save_models(final_models)
    config = {
        "generation": "17_cross_regime_generalization",
        "status": "research_only",
        "candidate_status": (
            "research_candidate" if selected_summary["discovery_pass"]
            else "diagnostic_fallback"
        ),
        "selected": selected if selected_summary["discovery_pass"] else None,
        "diagnostic_fallback": (
            None if selected_summary["discovery_pass"] else selected
        ),
        "research_discovery_gate": {
            "pooled_win_rate": RESEARCH_WIN_RATE,
            "worst_fold_win_rate": WORST_FOLD_WIN_RATE,
            "pooled_pf": 1.0,
            "pooled_mean_r": 0.0,
        },
        "production_target_win_rate": FINAL_TARGET_WIN_RATE,
        "model_features": model_features,
        "model_files": model_files,
        "promotion_pass": False,
    }
    CONFIG_FILE.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_only",
        "generation": "17_cross_regime_generalization",
        "development_history_policy": {
            "all_data_through": "2026-08-30",
            "selection_folds": "internal chronological development validation",
            "2025_holdout": "contaminated development diagnostic",
            "recent": "monitoring/development diagnostic only",
            "final_untouched_outer_interval": None,
        },
        "architecture": {
            "target_experts": list(TARGET_EXPERTS),
            "new_signal_families": 0,
            "threshold_sweep": False,
            "fixed_min_p_win": FIXED_MIN_PWIN,
            "fixed_min_expected_r": FIXED_MIN_EXPECTED_R,
            "fixed_top_k_per_expert_day": FIXED_TOP_K,
            "base_features": base_features,
            "absolute_scale_features_excluded_from_model": list(
                ABSOLUTE_SCALE_FEATURES
            ),
            "pre_freeze_methodology_correction": (
                "The first implementation check exposed raw ATR as the "
                "dominant gain feature. It was rejected before validation "
                "because Generation 17 requires cross-regime normalization, "
                "not because of candidate performance. No threshold or "
                "family setting changed."
            ),
            "regime_features": regime_features,
        },
        "data": {
            "history_rows": len(history),
            "history_start": history["TIME_DT"].iloc[0].isoformat(),
            "history_end": history["TIME_DT"].iloc[-1].isoformat(),
            "recent_development_rows": len(recent),
            "recent_development_start": recent["TIME_DT"].iloc[0].isoformat(),
            "recent_development_end": recent["TIME_DT"].iloc[-1].isoformat(),
            "mt5_loaded_development_end": DEVELOPMENT_END.isoformat(),
            "recent_comparison_limited_to_parent_ledger_end": True,
            "historical_dataset_manifest": dataset_manifest(),
        },
        "research_discovery_gate": {
            "pooled_win_rate": RESEARCH_WIN_RATE,
            "worst_fold_win_rate": WORST_FOLD_WIN_RATE,
            "pooled_pf_above": 1.0,
            "pooled_mean_r_above": 0.0,
            "no_catastrophic_fold": True,
            "incremental_value_required": True,
        },
        "production_promotion_gate": {
            "target_win_rate": FINAL_TARGET_WIN_RATE,
            "requires_untouched_outer_test": True,
            "available": False,
            "promotion_pass": False,
        },
        "selection": {
            "candidate_count": len(candidates),
            "qualified_count": len(qualified),
            "pareto_frontier": pareto_frontier(ranked),
            "ranked": ranked,
            "candidate_fold_results": fold_results,
        },
        "selected": {
            "status": (
                "research_candidate" if selected_summary["discovery_pass"]
                else "diagnostic_fallback"
            ),
            "params": selected,
            "selection_summary": selected_summary,
            "results": selected_results,
        },
        "regime_transfer_report": TRANSFER_JSON.name,
        "gen16_reference": reference_gen16_summary(report16),
        "model_diagnostics": model_diagnostics,
        "score_diagnostics": score_diagnostics,
        "validation_pending": True,
        "promotion_pass": False,
    }
    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REPORT_MD.write_text(markdown_report(report), encoding="utf-8")
    print(markdown_report(report), flush=True)
    print(transfer_markdown(transfer), flush=True)
    print(
        f"Saved {REPORT_JSON.name}, {REPORT_MD.name}, "
        f"{TRANSFER_JSON.name}, {TRANSFER_MD.name}, {CONFIG_FILE.name}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
