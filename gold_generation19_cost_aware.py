from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

import gold_generation16_independent_families as gen16
import gold_generation17_cross_regime as gen17
import gold_generation18_payoff_alignment as gen18
import gold_generation18_payoff_audit as payoff
from barrier_final_train import prepare_barrier_data
from gold_generation11_execution_aligned import add_targets
from gold_recent_walk_forward import DEFAULT_TERMINAL, build_feature_frame
from gold_regime_experts_iterative import training_frame
from gold_regime_experts_walk_forward import RECENT_START, SELECTION_FOLDS


ROOT = Path(__file__).resolve().parent
GEN15_REPORT = ROOT / "gold_generation15_signal_mining.json"
GEN17_REPORT = ROOT / "gold_generation17_cross_regime.json"
COST_AUDIT = ROOT / "gold_generation19_transaction_cost_audit.json"
REPORT_JSON = ROOT / "gold_generation19_cost_aware.json"
REPORT_MD = ROOT / "gold_generation19_cost_aware.md"
CONFIG_JSON = ROOT / "gold_generation19_candidate.json"

POINT = 0.01
FALLBACK_SPREAD_POINTS = 30.0
BASE_EXTRA_COST_POINTS = 5.0
STRESS_EXTRA_COST_POINTS = 10.0
SAFETY_MARGINS = (0.00, 0.02, 0.04, 0.06, 0.08)
RESEARCH_WIN_RATE = 0.58
WORST_FOLD_WIN_RATE = 0.50
MIN_INNER_TRADES = 10
PORTFOLIOS = {
    "gen19_dynamic_short_trend": ("short_trend_continuation",),
    "gen19_dynamic_existing_portfolio": gen17.TARGET_EXPERTS,
}
EXIT_PROFILES = {
    "current_13_16": (1.3, 1.6),
    "tp15_sl16": (1.5, 1.6),
    "tp16_sl16": (1.6, 1.6),
    "tp13_sl15": (1.3, 1.5),
}
RECENT_WARMUP_START = datetime(2026, 4, 1, tzinfo=timezone.utc)
DEVELOPMENT_END = datetime(2026, 8, 31, tzinfo=timezone.utc)
HISTORICAL_HOLDOUT_START = datetime(2025, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generation 19 cost-aware research")
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def effective_spread_points(frame: pd.DataFrame, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if "SPREAD" not in frame:
        return np.full(len(indices), FALLBACK_SPREAD_POINTS), np.zeros(len(indices), dtype=bool)
    raw = pd.to_numeric(frame["SPREAD"], errors="coerce").to_numpy(dtype=np.float64)[indices]
    observed = np.isfinite(raw) & (raw > 0.0)
    return np.where(observed, raw, FALLBACK_SPREAD_POINTS), observed


def add_economic_state(scores: pd.DataFrame, frame: pd.DataFrame, extra_cost_points: float = BASE_EXTRA_COST_POINTS) -> pd.DataFrame:
    if scores.empty:
        return scores.copy()
    result = scores.copy()
    indices = result["index"].to_numpy(dtype=np.int64)
    spread_points, observed = effective_spread_points(frame, indices)
    atr = frame["ATR"].to_numpy(dtype=np.float64)[indices]
    stop = np.maximum(atr * gen16.SL_ATR, gen16.MIN_SL_PRICE)
    take = np.maximum(atr * 1.3, 1.5)
    spread_price = spread_points * POINT
    extra_price = extra_cost_points * POINT
    denominator = stop + spread_price
    net_win_r = (take - spread_price - extra_price) / denominator
    net_loss_r = (-stop - spread_price - extra_price) / denominator
    payoff_ratio = np.divide(
        net_win_r,
        np.abs(net_loss_r),
        out=np.full(len(result), np.nan),
        where=net_loss_r != 0.0,
    )
    break_even = 1.0 / (1.0 + payoff_ratio)
    p_win = result["p_win"].to_numpy(dtype=np.float64)
    result["atr"] = atr
    result["spread_points"] = spread_points
    result["spread_observed"] = observed
    result["spread_atr"] = spread_price / np.maximum(atr, 1e-9)
    result["spread_sl"] = spread_price / stop
    result["total_cost_sl"] = (spread_price + extra_price) / stop
    result["estimated_net_tp_r"] = net_win_r
    result["estimated_net_sl_r"] = net_loss_r
    result["estimated_payoff_ratio"] = payoff_ratio
    result["estimated_break_even_win_rate"] = break_even
    result["economic_edge"] = p_win - break_even
    result["estimated_net_r"] = p_win * net_win_r + (1.0 - p_win) * net_loss_r
    directions = result["direction"].to_numpy(dtype=np.int8)
    fixed_reward = np.where(
        directions == 1,
        frame["LONG_REWARD"].to_numpy(dtype=np.float64)[indices],
        frame["SHORT_REWARD"].to_numpy(dtype=np.float64)[indices],
    )
    gross_pnl = fixed_reward * (stop + FALLBACK_SPREAD_POINTS * POINT) + (
        FALLBACK_SPREAD_POINTS + BASE_EXTRA_COST_POINTS
    ) * POINT
    result["realized_reward_observed_cost"] = (
        gross_pnl - spread_price - extra_price
    ) / denominator
    if "REG_RV_PCTL_20D" in frame:
        result["volatility_percentile"] = frame["REG_RV_PCTL_20D"].to_numpy(dtype=np.float64)[indices]
    else:
        result["volatility_percentile"] = np.nan
    return result


def entry_frame(scores: pd.DataFrame, margin: float) -> pd.DataFrame:
    if scores.empty:
        return gen16.empty_entries()
    keep = (
        np.isfinite(scores["economic_edge"])
        & np.isfinite(scores["estimated_net_r"])
        & (scores["economic_edge"] >= margin)
        & (scores["estimated_net_tp_r"] > 0.0)
    )
    selected = scores.loc[keep].copy()
    if selected.empty:
        return gen16.empty_entries()
    output = gen18.entry_frame(
        selected,
        selected["economic_edge"].to_numpy(dtype=np.float64),
    )
    for name in (
        "atr",
        "spread_points",
        "spread_observed",
        "spread_atr",
        "spread_sl",
        "total_cost_sl",
        "estimated_net_tp_r",
        "estimated_net_sl_r",
        "estimated_payoff_ratio",
        "estimated_break_even_win_rate",
        "economic_edge",
        "estimated_net_r",
        "volatility_percentile",
    ):
        output[name] = selected[name].to_numpy()
    return output


def reprice_record(record: dict, frame: pd.DataFrame, extra_cost_points: float, entry: pd.Series | None = None) -> dict:
    result = dict(record)
    index = int(result["index"])
    atr = float(frame["ATR"].iat[index])
    stop = max(atr * gen16.SL_ATR, gen16.MIN_SL_PRICE)
    spread_points, observed = effective_spread_points(frame, np.asarray([index]))
    spread_points = float(spread_points[0])
    old_reward = float(result["reward"])
    gross_pnl = old_reward * (stop + FALLBACK_SPREAD_POINTS * POINT) + (
        FALLBACK_SPREAD_POINTS + BASE_EXTRA_COST_POINTS
    ) * POINT
    denominator = stop + spread_points * POINT
    spread_r = spread_points * POINT / denominator
    extra_r = extra_cost_points * POINT / denominator
    result.update(
        {
            "reward": (gross_pnl - (spread_points + extra_cost_points) * POINT) / denominator,
            "gross_pnl_price": gross_pnl,
            "gross_reward_before_cost": gross_pnl / denominator,
            "spread_r": spread_r,
            "extra_cost_r": extra_r,
            "total_cost_r": spread_r + extra_r,
            "atr": atr,
            "spread_points": spread_points,
            "spread_observed": bool(observed[0]),
            "spread_atr": spread_points * POINT / max(atr, 1e-9),
            "spread_sl": spread_points * POINT / stop,
            "total_cost_sl": (spread_points + extra_cost_points) * POINT / stop,
        }
    )
    if entry is not None:
        for name in (
            "estimated_net_tp_r",
            "estimated_net_sl_r",
            "estimated_payoff_ratio",
            "estimated_break_even_win_rate",
            "economic_edge",
            "estimated_net_r",
            "volatility_percentile",
        ):
            result[name] = float(entry[name])
    else:
        take = max(atr * 1.3, 1.5)
        net_win_r = (take - (spread_points + extra_cost_points) * POINT) / denominator
        net_loss_r = (-stop - (spread_points + extra_cost_points) * POINT) / denominator
        payoff_ratio = net_win_r / abs(net_loss_r)
        break_even = 1.0 / (1.0 + payoff_ratio)
        p_win = float(result.get("p_win", np.nan))
        result.update(
            {
                "estimated_net_tp_r": net_win_r,
                "estimated_net_sl_r": net_loss_r,
                "estimated_payoff_ratio": payoff_ratio,
                "estimated_break_even_win_rate": break_even,
                "economic_edge": p_win - break_even,
                "estimated_net_r": p_win * net_win_r + (1.0 - p_win) * net_loss_r,
                "volatility_percentile": float(frame["REG_RV_PCTL_20D"].iat[index]) if "REG_RV_PCTL_20D" in frame else np.nan,
            }
        )
    return result


def execute_entries(frame: pd.DataFrame, entries: pd.DataFrame, period: str, extra_cost_points: float = BASE_EXTRA_COST_POINTS) -> list[dict]:
    fixed = gen16.execute_entries(frame, entries, period)
    if not fixed:
        return []
    lookup = entries.sort_values(["index", "priority"], ascending=[True, False]).drop_duplicates(["index", "direction"])
    lookup = lookup.set_index(["index", "direction"])
    return [
        reprice_record(
            record,
            frame,
            extra_cost_points,
            lookup.loc[(int(record["index"]), int(record["direction"]))],
        )
        for record in fixed
    ]


def reprice_ledger(records: list[dict], frame: pd.DataFrame, extra_cost_points: float = BASE_EXTRA_COST_POINTS) -> list[dict]:
    return [reprice_record(record, frame, extra_cost_points) for record in records]


def small_metrics(records: list[dict]) -> dict:
    rewards = np.asarray([record["reward"] for record in records], dtype=np.float64)
    if len(rewards) == 0:
        return {"trades": 0, "realized_win_rate": 0.0, "profit_factor": None, "mean_r": 0.0, "sum_r": 0.0}
    loss = float(-rewards[rewards < 0.0].sum())
    return {
        "trades": len(records),
        "realized_win_rate": float((rewards > 0.0).mean()),
        "profit_factor": None if loss <= 0.0 else float(rewards[rewards > 0.0].sum() / loss),
        "mean_r": float(rewards.mean()),
        "sum_r": float(rewards.sum()),
    }


def metrics(records: list[dict], evaluated_days: int) -> dict:
    enriched = [
        {
            "outcome": record["outcome"],
            "reward": record["reward"],
            "gross_reward_before_cost": record["gross_reward_before_cost"],
            "spread_r": record["spread_r"],
            "extra_cost_r": record["extra_cost_r"],
            "total_cost_r": record["total_cost_r"],
        }
        for record in records
    ]
    value = payoff.payoff_metrics(enriched, evaluated_days)
    for name in (
        "spread_atr",
        "spread_sl",
        "total_cost_sl",
        "estimated_break_even_win_rate",
        "economic_edge",
        "estimated_net_r",
    ):
        values = np.asarray([record.get(name, np.nan) for record in records], dtype=np.float64)
        value[f"average_{name}"] = None if not np.isfinite(values).any() else float(np.nanmean(values))
    value["wins"] = value["realized_positive_trades"]
    value["losses"] = value["realized_nonpositive_trades"]
    value["timeouts"] = value["timeout_exits"]
    value["direction_contribution"] = {
        label: small_metrics([record for record in records if record["direction"] == direction])
        for direction, label in ((1, "long"), (2, "short"))
    }
    value["expert_contribution"] = {
        expert: small_metrics([record for record in records if record["expert"] == expert])
        for expert in gen17.TARGET_EXPERTS
    }
    value["family_contribution"] = {
        family: small_metrics([record for record in records if record["family"] == family])
        for family in ("breakout", "trend_continuation")
    }
    return value


def policy_scores(train: pd.DataFrame, models: dict, diagnostics: dict, base_features: list[str], model_features: list[str]) -> dict[str, pd.DataFrame]:
    masks = gen16.family_masks(train, base_features)
    output = {}
    for expert in gen17.TARGET_EXPERTS:
        direction = gen16.EXPERT_DIRECTION[expert]
        outcome_column = "LONG_OUTCOME" if direction == 1 else "SHORT_OUTCOME"
        reward_column = "LONG_REWARD" if direction == 1 else "SHORT_REWARD"
        indices = np.flatnonzero(gen16.rising_edges(masks[expert]))
        keep = (
            (indices >= diagnostics[expert]["policy_start_index"])
            & (train[outcome_column].to_numpy(dtype=np.int8)[indices] >= 0)
            & np.isfinite(train[reward_column].to_numpy(dtype=np.float64)[indices])
        )
        scores = gen18.score_indices(models[expert], train, indices[keep], expert, model_features)
        output[expert] = add_economic_state(scores, train)
    return output


def combine_entries(scores: dict[str, pd.DataFrame], experts: tuple[str, ...], margin: float) -> pd.DataFrame:
    pieces = [entry_frame(scores[expert], margin) for expert in experts]
    pieces = [piece for piece in pieces if not piece.empty]
    return gen16.empty_entries() if not pieces else pd.concat(pieces, ignore_index=True)


def policy_pass(base: dict, stress: dict) -> bool:
    return bool(
        base["trades"] >= MIN_INNER_TRADES
        and base["realized_positive_trade_win_rate"] >= RESEARCH_WIN_RATE
        and base["profit_factor"] is not None
        and base["profit_factor"] > 1.0
        and base["mean_r"] > 0.0
        and base["break_even_adjusted_win_rate_edge"] is not None
        and base["break_even_adjusted_win_rate_edge"] > 0.0
        and stress["profit_factor"] is not None
        and stress["profit_factor"] > 1.0
        and stress["mean_r"] > 0.0
    )


def choose_policies(train: pd.DataFrame, scores: dict[str, pd.DataFrame]) -> dict:
    output = {}
    for candidate_id, experts in PORTFOLIOS.items():
        candidates = []
        index_parts = [
            scores[expert]["index"].to_numpy(dtype=np.int64)
            for expert in experts
            if not scores[expert].empty
        ]
        scored_indices = np.concatenate(index_parts) if index_parts else np.asarray([], dtype=np.int64)
        days = (
            int(train["TIME_DT"].iloc[int(scored_indices.min()) : int(scored_indices.max()) + 1].dt.date.nunique())
            if len(scored_indices)
            else 0
        )
        for margin in SAFETY_MARGINS:
            entries = combine_entries(scores, experts, margin)
            records = execute_entries(train, entries, f"inner_{candidate_id}_{margin:.2f}")
            stress_records = execute_entries(train, entries, f"inner_{candidate_id}_{margin:.2f}_stress", STRESS_EXTRA_COST_POINTS)
            base = metrics(records, days)
            stress = metrics(stress_records, days)
            candidates.append({"safety_margin": margin, "qualified": policy_pass(base, stress), "metrics": base, "cost_stress": stress})
        candidates.sort(
            key=lambda item: (
                item["qualified"],
                item["metrics"]["trades"] if item["qualified"] else 0,
                item["metrics"]["mean_r"],
                item["metrics"]["realized_positive_trade_win_rate"],
            ),
            reverse=True,
        )
        output[candidate_id] = {"selected": candidates[0], "all_candidates": candidates}
    return output


def score_evaluation(models: dict, frame: pd.DataFrame, base_features: list[str], model_features: list[str]) -> dict[str, pd.DataFrame]:
    raw = gen18.score_all_experts(models, frame, base_features, model_features)
    return {expert: add_economic_state(scores, frame) for expert, scores in raw.items()}


def selection_diagnostics(scores: dict[str, pd.DataFrame]) -> dict:
    output = {}
    for expert, values in scores.items():
        mature = values[np.isfinite(values["realized_reward_observed_cost"])].copy()
        p_corr = mature["p_win"].corr(mature["realized_reward_observed_cost"], method="spearman")
        e_corr = mature["economic_edge"].corr(mature["realized_reward_observed_cost"], method="spearman")
        output[expert] = {
            "events": len(mature),
            "p_win_vs_realized_r_spearman": None if pd.isna(p_corr) else float(p_corr),
            "economic_edge_vs_realized_r_spearman": None if pd.isna(e_corr) else float(e_corr),
            "positive_economic_edge_rate": float((mature["economic_edge"] > 0.0).mean()) if len(mature) else 0.0,
            "average_spread_atr": float(mature["spread_atr"].mean()) if len(mature) else None,
        }
    return output


def compare_records(candidate: list[dict], baseline: list[dict]) -> dict:
    value = gen16.compare_records(candidate, baseline)
    baseline_losers = sum(record["reward"] <= 0.0 for record in baseline)
    baseline_winners = sum(record["reward"] > 0.0 for record in baseline)
    value["loser_removal_rate"] = value["losers_removed"] / max(baseline_losers, 1)
    value["winner_accidental_removal_rate"] = value["winners_accidentally_removed"] / max(baseline_winners, 1)
    return value


def process_period(train: pd.DataFrame, evaluation: pd.DataFrame, period: str, base_features: list[str], model_features: list[str], report17: dict) -> tuple[dict, dict, dict, dict]:
    models, diagnostics = gen17.train_target_experts(train, base_features, model_features)
    inner_scores = policy_scores(train, models, diagnostics, base_features, model_features)
    policies = choose_policies(train, inner_scores)
    scores = score_evaluation(models, evaluation, base_features, model_features)

    absolute = gen18.absolute_entries(scores["short_trend_continuation"], evaluation)
    reproduced = gen16.execute_entries(evaluation, absolute, f"{period}_absolute")
    expected = report17["selected"]["results"][period]["trade_ledger"]
    if [record["trade_id"] for record in reproduced] != [record["trade_id"] for record in expected]:
        raise RuntimeError(f"Gen17 ledger mismatch in {period}")
    baseline = reprice_ledger(expected, evaluation)
    baseline_stress = reprice_ledger(expected, evaluation, STRESS_EXTRA_COST_POINTS)
    days = int(evaluation["TIME_DT"].dt.date.nunique())
    results = {}
    for candidate_id, experts in PORTFOLIOS.items():
        choice = policies[candidate_id]["selected"]
        entries = combine_entries(scores, experts, float(choice["safety_margin"]))
        records = execute_entries(evaluation, entries, f"{period}_{candidate_id}")
        stress_records = execute_entries(evaluation, entries, f"{period}_{candidate_id}_stress", STRESS_EXTRA_COST_POINTS)
        results[candidate_id] = {
            "safety_margin": choice["safety_margin"],
            "inner_policy_qualified": choice["qualified"],
            "metrics": metrics(records, days),
            "cost_stress": metrics(stress_records, days),
            "comparison_to_gen17": compare_records(records, baseline),
            "trade_ledger": records,
            "cost_stress_trade_ledger": stress_records,
        }
    baseline_result = {
        "metrics": metrics(baseline, days),
        "cost_stress": metrics(baseline_stress, days),
        "trade_ledger": baseline,
        "cost_stress_trade_ledger": baseline_stress,
    }
    return results, baseline_result, policies, selection_diagnostics(scores)


def pooled_result(results: list[dict], evaluated_days: int) -> dict:
    records = [record for result in results for record in result["trade_ledger"]]
    stress = [record for result in results for record in result["cost_stress_trade_ledger"]]
    return {"metrics": metrics(records, evaluated_days), "cost_stress": metrics(stress, evaluated_days), "trade_ledger": records, "cost_stress_trade_ledger": stress}


def candidate_summary(candidate_id: str, fold_results: dict, pooled: dict, baseline_pooled: dict) -> dict:
    folds = [fold_results[name][candidate_id] for name, *_ in SELECTION_FOLDS]
    fold_metrics = [value["metrics"] for value in folds]
    base = pooled["metrics"]
    stress = pooled["cost_stress"]
    no_catastrophe = all(
        value["trades"] >= 5
        and value["realized_positive_trade_win_rate"] >= WORST_FOLD_WIN_RATE
        and value["max_drawdown_pct"] >= -0.25
        and (value["profit_factor"] is None or value["profit_factor"] >= 0.65)
        for value in fold_metrics
    )
    discovery_pass = bool(
        all(value["inner_policy_qualified"] for value in folds)
        and base["realized_positive_trade_win_rate"] >= RESEARCH_WIN_RATE
        and base["profit_factor"] is not None
        and base["profit_factor"] > 1.0
        and base["mean_r"] > 0.0
        and base["pnl"] > 0.0
        and base["break_even_adjusted_win_rate_edge"] is not None
        and base["break_even_adjusted_win_rate_edge"] > 0.0
        and stress["profit_factor"] is not None
        and stress["profit_factor"] > 1.0
        and stress["mean_r"] > 0.0
        and no_catastrophe
    )
    parent = baseline_pooled["metrics"]
    records = pooled["trade_ledger"]
    baseline_records = baseline_pooled["trade_ledger"]
    return {
        "candidate_id": candidate_id,
        "experts": list(PORTFOLIOS[candidate_id]),
        "discovery_pass": discovery_pass,
        "no_catastrophic_fold": no_catastrophe,
        "worst_fold_realized_win_rate": min(value["realized_positive_trade_win_rate"] for value in fold_metrics),
        "simultaneous_win_and_frequency_improvement": bool(
            base["trades"] > parent["trades"]
            and base["realized_positive_trade_win_rate"] > parent["realized_positive_trade_win_rate"]
            and base["profit_factor"] is not None
            and base["profit_factor"] > 1.0
            and base["mean_r"] > 0.0
        ),
        "metrics": base,
        "cost_stress": stress,
        "comparison_to_gen17": compare_records(records, baseline_records),
    }


def pareto_frontier(summaries: list[dict]) -> list[str]:
    feasible = [value for value in summaries if value["discovery_pass"]]
    return [
        value["candidate_id"]
        for value in feasible
        if not any(
            other["metrics"]["realized_positive_trade_win_rate"] >= value["metrics"]["realized_positive_trade_win_rate"]
            and other["metrics"]["trades_per_day"] >= value["metrics"]["trades_per_day"]
            and (
                other["metrics"]["realized_positive_trade_win_rate"] > value["metrics"]["realized_positive_trade_win_rate"]
                or other["metrics"]["trades_per_day"] > value["metrics"]["trades_per_day"]
            )
            for other in feasible
            if other["candidate_id"] != value["candidate_id"]
        )
    ]


def diagnostic_group(records: list[dict], keys: np.ndarray) -> dict:
    output = {}
    keys = np.asarray(keys, dtype=object)
    for key in pd.unique(keys):
        if pd.isna(key):
            continue
        group = [record for record, current in zip(records, keys) if current == key]
        output[str(key)] = small_metrics(group)
    return output


def cost_regime_report(records: list[dict], frames: dict[str, pd.DataFrame]) -> dict:
    if not records:
        return {}
    rows = []
    for record in records:
        fold = next(name for name in frames if record["period"].startswith(name))
        frame = frames[fold]
        index = int(record["index"])
        timestamp = frame["TIME_DT"].iat[index]
        rows.append(
            {
                **record,
                "hour": int(timestamp.hour),
                "session": ("utc_00_05" if timestamp.hour <= 5 else "utc_06_12" if timestamp.hour <= 12 else "utc_13_18" if timestamp.hour <= 18 else "utc_19_23"),
                "volatility_percentile": float(frame["REG_RV_PCTL_20D"].iat[index]),
            }
        )
    table = pd.DataFrame(rows)
    table["atr_percentile"] = pd.qcut(table["atr"].rank(method="first"), 4, labels=("q1", "q2", "q3", "q4"))
    table["spread_percentile"] = pd.qcut(table["spread_points"].rank(method="first"), 4, labels=("q1", "q2", "q3", "q4"))
    table["volatility_bucket"] = pd.cut(table["volatility_percentile"], (-np.inf, 0.25, 0.50, 0.75, np.inf), labels=("q1", "q2", "q3", "q4"))
    table["spread_atr_bucket"] = pd.cut(table["spread_atr"], (-np.inf, 0.05, 0.10, 0.20, np.inf), labels=("<=0.05", "0.05-0.10", "0.10-0.20", ">0.20"))
    output = {
        "note": "ATR/spread quartiles are post-evaluation diagnostics and were not candidate-selection inputs.",
        "by_session": diagnostic_group(rows, table["session"].to_numpy()),
        "by_hour": diagnostic_group(rows, table["hour"].to_numpy()),
        "by_atr_percentile": diagnostic_group(rows, table["atr_percentile"].astype(object).to_numpy()),
        "by_volatility_percentile": diagnostic_group(rows, table["volatility_bucket"].astype(object).to_numpy()),
        "by_spread_percentile": diagnostic_group(rows, table["spread_percentile"].astype(object).to_numpy()),
        "by_spread_atr": diagnostic_group(rows, table["spread_atr_bucket"].astype(object).to_numpy()),
        "by_direction": diagnostic_group(rows, np.where(table["direction"] == 1, "long", "short")),
        "by_expert": diagnostic_group(rows, table["expert"].to_numpy()),
        "by_family": diagnostic_group(rows, table["family"].to_numpy()),
    }
    high_cutoff = float(table["spread_atr"].quantile(0.75))
    high = table["spread_atr"] >= high_cutoff
    total_cost = float(table["total_cost_r"].sum())
    negative_loss = float(-table.loc[table["reward"] < 0.0, "reward"].sum())
    high_negative = float(-table.loc[high & (table["reward"] < 0.0), "reward"].sum())
    gross_positive_net_nonpositive = (table["gross_reward_before_cost"] > 0.0) & (table["reward"] <= 0.0)
    output["high_relative_cost_impact"] = {
        "spread_atr_p75": high_cutoff,
        "trades": int(high.sum()),
        "share_of_total_cost_r": float(table.loc[high, "total_cost_r"].sum() / total_cost) if total_cost else None,
        "share_of_total_negative_r": high_negative / negative_loss if negative_loss else None,
        "gross_positive_but_net_nonpositive_trades": int(gross_positive_net_nonpositive.sum()),
        "gross_positive_but_net_nonpositive_high_cost_trades": int((gross_positive_net_nonpositive & high).sum()),
    }
    return output


def first_touch_profile(frame: pd.DataFrame, seed_records: list[dict], period: str, tp_atr: float, sl_atr: float) -> list[dict]:
    close = frame["CLOSE"].to_numpy(dtype=np.float64)
    high = frame["HIGH"].to_numpy(dtype=np.float64)
    low = frame["LOW"].to_numpy(dtype=np.float64)
    atr = frame["ATR"].to_numpy(dtype=np.float64)
    records = []
    free_index = 0
    for seed in sorted(seed_records, key=lambda value: value["index"]):
        index = int(seed["index"])
        if index < free_index or index + gen16.HORIZON >= len(frame):
            continue
        direction = int(seed["direction"])
        take = max(atr[index] * tp_atr, 1.5)
        stop = max(atr[index] * sl_atr, gen16.MIN_SL_PRICE)
        outcome, gross, exit_offset = 0, None, gen16.HORIZON
        for offset in range(1, gen16.HORIZON + 1):
            if direction == 1:
                take_hit = high[index + offset] >= close[index] + take
                stop_hit = low[index + offset] <= close[index] - stop
            else:
                take_hit = low[index + offset] <= close[index] - take
                stop_hit = high[index + offset] >= close[index] + stop
            if stop_hit:
                outcome, gross, exit_offset = 2, -stop, offset
                break
            if take_hit:
                outcome, gross, exit_offset = 1, take, offset
                break
        if gross is None:
            gross = close[index + gen16.HORIZON] - close[index]
            if direction == 2:
                gross = -gross
        spread_points, observed = effective_spread_points(frame, np.asarray([index]))
        spread_points = float(spread_points[0])
        denominator = stop + spread_points * POINT
        spread_r = spread_points * POINT / denominator
        extra_r = BASE_EXTRA_COST_POINTS * POINT / denominator
        net_win_r = (take - (spread_points + BASE_EXTRA_COST_POINTS) * POINT) / denominator
        net_loss_r = (-stop - (spread_points + BASE_EXTRA_COST_POINTS) * POINT) / denominator
        payoff_ratio = net_win_r / abs(net_loss_r)
        break_even = 1.0 / (1.0 + payoff_ratio)
        p_win = float(seed["p_win"])
        record = dict(seed)
        record.update(
            {
                "period": period,
                "exit_index": index + exit_offset,
                "outcome": outcome,
                "reward": (gross - (spread_points + BASE_EXTRA_COST_POINTS) * POINT) / denominator,
                "gross_pnl_price": gross,
                "gross_reward_before_cost": gross / denominator,
                "spread_r": spread_r,
                "extra_cost_r": extra_r,
                "total_cost_r": spread_r + extra_r,
                "atr": atr[index],
                "spread_points": spread_points,
                "spread_observed": bool(observed[0]),
                "spread_atr": spread_points * POINT / max(atr[index], 1e-9),
                "spread_sl": spread_points * POINT / stop,
                "total_cost_sl": (spread_points + BASE_EXTRA_COST_POINTS) * POINT / stop,
                "estimated_net_tp_r": net_win_r,
                "estimated_net_sl_r": net_loss_r,
                "estimated_payoff_ratio": payoff_ratio,
                "estimated_break_even_win_rate": break_even,
                "economic_edge": p_win - break_even,
                "estimated_net_r": p_win * net_win_r + (1.0 - p_win) * net_loss_r,
            }
        )
        records.append(record)
        free_index = index + exit_offset + 1
    return records


def exit_economics(baselines: dict, frames: dict) -> dict:
    output = {}
    total_days = sum(int(frame["TIME_DT"].dt.date.nunique()) for frame in frames.values())
    for profile, (tp_atr, sl_atr) in EXIT_PROFILES.items():
        folds = {}
        pooled = []
        for fold_name, *_ in SELECTION_FOLDS:
            records = first_touch_profile(
                frames[fold_name],
                baselines[fold_name]["trade_ledger"],
                f"{fold_name}_{profile}",
                tp_atr,
                sl_atr,
            )
            pooled.extend(records)
            folds[fold_name] = metrics(records, int(frames[fold_name]["TIME_DT"].dt.date.nunique()))
        output[profile] = {
            "tp_atr": tp_atr,
            "sl_atr": sl_atr,
            "selection_eligible": False,
            "reason": "paired post-evaluation exit diagnostic on the fixed Gen17 executable cohort",
            "folds": folds,
            "pooled": metrics(pooled, total_days),
        }
    return output


def markdown(report: dict) -> str:
    parent = report["baseline"]["selection_pooled"]["metrics"]
    lines = [
        "# Generation 19 - Cost-Aware Dynamic Break-Even",
        "",
        "Status: `research_only`. No new signal family or ML architecture was introduced.",
        "",
        "## Selection-period comparison",
        "",
        "| Candidate | Trades | Trades/day | Realized win | TP-first | PF | Mean-R | PnL | Max DD | Break-even | Edge | Avg spread/ATR | Avg cost R |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| Gen17 repriced parent | {parent['trades']} | {parent['trades_per_day']:.3f} | {parent['realized_positive_trade_win_rate']:.2%} | {parent['tp_first_rate']:.2%} | {parent['profit_factor'] or 0.0:.3f} | {parent['mean_r']:.4f} | {parent['pnl']:.2f} | {parent['max_drawdown_pct']:.2%} | {parent['realized_break_even_win_rate'] or 0.0:.2%} | {parent['break_even_adjusted_win_rate_edge'] or 0.0:.2%} | {parent['average_spread_atr'] or 0.0:.4f} | {parent['average_total_cost_r_per_trade']:.4f} |",
    ]
    for value in report["candidate_summaries"]:
        metric = value["metrics"]
        lines.append(
            f"| {value['candidate_id']} | {metric['trades']} | {metric['trades_per_day']:.3f} | "
            f"{metric['realized_positive_trade_win_rate']:.2%} | {metric['tp_first_rate']:.2%} | "
            f"{metric['profit_factor'] or 0.0:.3f} | {metric['mean_r']:.4f} | {metric['pnl']:.2f} | "
            f"{metric['max_drawdown_pct']:.2%} | {metric['realized_break_even_win_rate'] or 0.0:.2%} | "
            f"{metric['break_even_adjusted_win_rate_edge'] or 0.0:.2%} | {metric['average_spread_atr'] or 0.0:.4f} | "
            f"{metric['average_total_cost_r_per_trade']:.4f} |"
        )
    lines.extend(["", "## Chronological fold results", "", "| Candidate | Fold | Margin | Trades | Trades/day | Win | PF | Mean-R | PnL | DD | Edge | Stress PF |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for fold_name, *_ in SELECTION_FOLDS:
        base = report["baseline"]["folds"][fold_name]["metrics"]
        lines.append(f"| Gen17 repriced | {fold_name} | n/a | {base['trades']} | {base['trades_per_day']:.3f} | {base['realized_positive_trade_win_rate']:.2%} | {base['profit_factor'] or 0.0:.3f} | {base['mean_r']:.4f} | {base['pnl']:.2f} | {base['max_drawdown_pct']:.2%} | {base['break_even_adjusted_win_rate_edge'] or 0.0:.2%} | {report['baseline']['folds'][fold_name]['cost_stress']['profit_factor'] or 0.0:.3f} |")
        for candidate_id in PORTFOLIOS:
            value = report["fold_results"][fold_name][candidate_id]
            metric = value["metrics"]
            lines.append(f"| {candidate_id} | {fold_name} | {value['safety_margin']:.2%} | {metric['trades']} | {metric['trades_per_day']:.3f} | {metric['realized_positive_trade_win_rate']:.2%} | {metric['profit_factor'] or 0.0:.3f} | {metric['mean_r']:.4f} | {metric['pnl']:.2f} | {metric['max_drawdown_pct']:.2%} | {metric['break_even_adjusted_win_rate_edge'] or 0.0:.2%} | {value['cost_stress']['profit_factor'] or 0.0:.3f} |")
    lines.extend(["", "## Separate paired exit-economics diagnostic", "", "| Profile | TP/SL ATR | Trades | Win | PF | Mean-R | Break-even | Edge |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for name, value in report["exit_economics"].items():
        metric = value["pooled"]
        lines.append(f"| {name} | {value['tp_atr']}/{value['sl_atr']} | {metric['trades']} | {metric['realized_positive_trade_win_rate']:.2%} | {metric['profit_factor'] or 0.0:.3f} | {metric['mean_r']:.4f} | {metric['realized_break_even_win_rate'] or 0.0:.2%} | {metric['break_even_adjusted_win_rate_edge'] or 0.0:.2%} |")
    lines.extend(["", "Exit profiles are diagnostic only: they were viewed on development folds and cannot support an OOS selection claim."])
    return "\n".join(lines) + "\n"


def self_check() -> None:
    frame = pd.DataFrame(
        {
            "ATR": [1.0],
            "SPREAD": [20.0],
            "LONG_REWARD": [0.5],
            "SHORT_REWARD": [-1.0],
            "REG_RV_PCTL_20D": [0.5],
        }
    )
    scores = pd.DataFrame(
        {
            "index": [0],
            "direction": [1],
            "p_win": [0.7],
            "reward": [0.5],
        }
    )
    enriched = add_economic_state(scores, frame)
    assert enriched["spread_points"].iat[0] == 20.0
    assert math.isclose(enriched["economic_edge"].iat[0], 0.7 - enriched["estimated_break_even_win_rate"].iat[0])
    assert len(SAFETY_MARGINS) == 5 and len(EXIT_PROFILES) == 4
    print("generation19_cost_aware_self_check_ok")


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0
    report15 = json.loads(GEN15_REPORT.read_text(encoding="utf-8"))
    del report15
    report17 = json.loads(GEN17_REPORT.read_text(encoding="utf-8"))
    cost_audit = json.loads(COST_AUDIT.read_text(encoding="utf-8"))
    if not cost_audit["verdict"]["methodology_correction_required"]:
        raise RuntimeError("Phase 1 cost audit must be completed before Gen19 selection")

    history, base_features = prepare_barrier_data()
    history = add_targets(history)
    history, regime_features = gen17.add_regime_features(history, base_features)
    robust_base = [feature for feature in base_features if feature not in gen17.ABSOLUTE_SCALE_FEATURES]
    model_features = [*robust_base, *regime_features]
    fold_results, baselines, policies, score_diagnostics, frames = {}, {}, {}, {}, {}
    for fold_name, fold_start, fold_end in SELECTION_FOLDS:
        train = training_frame(history, fold_start)
        evaluation = history[(history["TIME_DT"] >= fold_start) & (history["TIME_DT"] < fold_end)].reset_index(drop=True)
        result, baseline, fold_policies, diagnostics = process_period(train, evaluation, fold_name, base_features, model_features, report17)
        fold_results[fold_name], baselines[fold_name] = result, baseline
        policies[fold_name], score_diagnostics[fold_name], frames[fold_name] = fold_policies, diagnostics, evaluation
        print(f"Fold {fold_name} complete", flush=True)

    selection_days = sum(baselines[name]["metrics"]["evaluated_days"] for name, *_ in SELECTION_FOLDS)
    baseline_pooled = pooled_result([baselines[name] for name, *_ in SELECTION_FOLDS], selection_days)
    pooled = {
        candidate_id: pooled_result([fold_results[name][candidate_id] for name, *_ in SELECTION_FOLDS], selection_days)
        for candidate_id in PORTFOLIOS
    }
    summaries = [candidate_summary(candidate_id, fold_results, pooled[candidate_id], baseline_pooled) for candidate_id in PORTFOLIOS]
    summaries.sort(
        key=lambda value: (
            value["discovery_pass"],
            value["simultaneous_win_and_frequency_improvement"],
            value["metrics"]["trades"],
            value["metrics"]["realized_positive_trade_win_rate"],
        ),
        reverse=True,
    )
    qualified = [value for value in summaries if value["discovery_pass"]]
    frozen = qualified[0] if qualified else summaries[0]
    frozen_id = frozen["candidate_id"]

    holdout = history[history["TIME_DT"] >= HISTORICAL_HOLDOUT_START].reset_index(drop=True)
    holdout_result, holdout_baseline, holdout_policy, holdout_diag = process_period(
        training_frame(history, HISTORICAL_HOLDOUT_START), holdout, "2025_2026_05_development", base_features, model_features, report17
    )

    if not mt5.initialize(path=str(DEFAULT_TERMINAL), timeout=10_000):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        recent_all, recent_features = build_feature_frame(RECENT_WARMUP_START, DEVELOPMENT_END)
    finally:
        mt5.shutdown()
    if set(recent_features) != set(base_features):
        raise RuntimeError("Historical and recent features differ")
    recent_all = add_targets(recent_all)
    recent_all, _ = gen17.add_regime_features(recent_all, base_features)
    recent_end = pd.Timestamp(report17["data"]["recent_development_end"]) + pd.Timedelta(minutes=1)
    recent = recent_all[(recent_all["TIME_DT"] >= RECENT_START.replace(tzinfo=None)) & (recent_all["TIME_DT"] < recent_end)].reset_index(drop=True)
    final_cutoff = history["TIME_DT"].iloc[-1] + pd.Timedelta(minutes=1)
    recent_result, recent_baseline, recent_policy, recent_diag = process_period(
        training_frame(history, final_cutoff), recent, "2026_recent_development", base_features, model_features, report17
    )

    exit_report = exit_economics(baselines, frames)
    selected_records = pooled[frozen_id]["trade_ledger"]
    regime_report = {
        "selected_candidate": cost_regime_report(selected_records, frames),
        "gen17_parent": cost_regime_report(baseline_pooled["trade_ledger"], frames),
        "selected_candidate_by_fold": {
            fold_name: cost_regime_report(
                fold_results[fold_name][frozen_id]["trade_ledger"],
                {fold_name: frames[fold_name]},
            )
            for fold_name, *_ in SELECTION_FOLDS
        },
        "gen17_parent_by_fold": {
            fold_name: cost_regime_report(
                baselines[fold_name]["trade_ledger"],
                {fold_name: frames[fold_name]},
            )
            for fold_name, *_ in SELECTION_FOLDS
        },
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generation": "19_cost_aware_dynamic_break_even",
        "status": "research_only",
        "development_history_policy": report17["development_history_policy"],
        "phase1_cost_audit": COST_AUDIT.name,
        "architecture": {
            "new_signal_families": 0,
            "new_ml_architecture": False,
            "absolute_probability_threshold_sweep": False,
            "frozen_signal_recipe": "Generation 17 target experts and past-only isotonic calibration",
            "selection_rule": "calibrated P(win) - entry-known realized-payoff break-even >= past-only safety margin",
            "safety_margins": list(SAFETY_MARGINS),
            "cost_rule": "observed entry SPREAD when >0, otherwise 30 points; plus 5 points extra",
            "cost_stress": "same trades repriced with 10 extra points",
            "tp_atr": 1.3,
            "sl_atr": 1.6,
            "horizon": gen16.HORIZON,
            "same_bar_tp_sl": "stop_first",
            "position_limit": 1,
        },
        "baseline": {"folds": baselines, "selection_pooled": baseline_pooled},
        "fold_results": fold_results,
        "inner_policy_selection": policies,
        "score_stability": score_diagnostics,
        "candidate_summaries": summaries,
        "pareto_frontier": pareto_frontier(summaries),
        "frozen_candidate_id": frozen_id if frozen["discovery_pass"] else None,
        "diagnostic_fallback_id": None if frozen["discovery_pass"] else frozen_id,
        "frozen_candidate": frozen,
        "development_diagnostics": {
            "2025_2026_05": holdout_result[frozen_id],
            "2025_2026_05_baseline": holdout_baseline,
            "2025_2026_05_policy": holdout_policy[frozen_id],
            "2025_2026_05_score_stability": holdout_diag,
            "2026_recent": recent_result[frozen_id],
            "2026_recent_baseline": recent_baseline,
            "2026_recent_policy": recent_policy[frozen_id],
            "2026_recent_score_stability": recent_diag,
        },
        "cost_regime_report": regime_report,
        "exit_economics": exit_report,
        "selection_inventory": {
            "portfolios": {key: list(value) for key, value in PORTFOLIOS.items()},
            "safety_margins_tested": list(SAFETY_MARGINS),
            "exit_profiles_diagnostic_only": {key: list(value) for key, value in EXIT_PROFILES.items()},
            "total_dynamic_policy_combinations_per_fold": len(PORTFOLIOS) * len(SAFETY_MARGINS),
        },
        "promotion_pass": False,
        "gemini_modified": False,
        "final_untouched_test_validity": "FAIL_no_untouched_future_interval",
    }
    CONFIG_JSON.write_text(
        json.dumps(
            {
                "generation": report["generation"],
                "status": "research_only",
                "frozen_candidate_id": report["frozen_candidate_id"],
                "diagnostic_fallback_id": report["diagnostic_fallback_id"],
                "promotion_pass": False,
                "selection_rule": report["architecture"]["selection_rule"],
                "fold_safety_margins": {
                    fold: policies[fold][frozen_id]["selected"]["safety_margin"]
                    for fold, *_ in SELECTION_FOLDS
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_MD.write_text(markdown(report), encoding="utf-8")
    print(f"Wrote {REPORT_JSON.name}, {REPORT_MD.name}, and {CONFIG_JSON.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
