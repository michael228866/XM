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
import gold_generation18_payoff_audit as payoff
from barrier_final_train import prepare_barrier_data
from gold_generation11_execution_aligned import add_targets
from gold_recent_walk_forward import DEFAULT_TERMINAL, build_feature_frame
from gold_regime_experts_iterative import training_frame
from gold_regime_experts_walk_forward import RECENT_START, SELECTION_FOLDS


PROJECT_ROOT = Path(__file__).resolve().parent
GEN15_REPORT = PROJECT_ROOT / "gold_generation15_signal_mining.json"
GEN17_REPORT = PROJECT_ROOT / "gold_generation17_cross_regime.json"
PAYOFF_REPORT = PROJECT_ROOT / "gold_generation18_payoff_audit.json"
REPORT_JSON = PROJECT_ROOT / "gold_generation18_payoff_alignment.json"
REPORT_MD = PROJECT_ROOT / "gold_generation18_payoff_alignment.md"
CALIBRATION_JSON = PROJECT_ROOT / "gold_generation18_calibration_drift.json"
CALIBRATION_MD = PROJECT_ROOT / "gold_generation18_calibration_drift.md"
CONFIG_JSON = PROJECT_ROOT / "gold_generation18_candidate.json"

TARGET_EXPERTS = gen17.TARGET_EXPERTS
SCORE_MODES = ("p_win_rank", "expected_r_rank", "joint_rank")
TOP_FRACTIONS = (0.01, 0.02, 0.05, 0.10)
BLOCK_ROWS = 10_080
MAX_REFERENCE_EVENTS = 60_000
INNER_WARMUP_FRACTION = 0.40
MIN_INNER_WARMUP = 500
MIN_INNER_TRADES = 10
RESEARCH_WIN_RATE = 0.58
WORST_FOLD_WIN_RATE = 0.50
MAX_CATASTROPHIC_DRAWDOWN = -0.25
MIN_CATASTROPHIC_PF = 0.65
RECENT_WARMUP_START = datetime(2026, 4, 1, tzinfo=timezone.utc)
DEVELOPMENT_END = datetime(2026, 8, 31, tzinfo=timezone.utc)
HISTORICAL_HOLDOUT_START = datetime(2025, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generation 18 payoff-aligned ranking research"
    )
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def score_indices(
    model: dict,
    frame: pd.DataFrame,
    indices: np.ndarray,
    expert: str,
    model_features: list[str],
) -> pd.DataFrame:
    if len(indices) == 0:
        return pd.DataFrame()
    x = frame.iloc[indices][model_features].astype(np.float32)
    raw_probability = model["win"].predict_proba(x)[:, 1]
    probability = model["calibrator"].predict(raw_probability).astype(np.float32)
    probability_r = (
        probability * model["average_win_r"]
        + (1.0 - probability) * model["average_nonwin_r"]
    )
    mean_r_prediction = model["mean_r"].predict(x).astype(np.float32)
    expected_r = 0.5 * (probability_r + mean_r_prediction)
    direction = gen16.EXPERT_DIRECTION[expert]
    outcome_column = "LONG_OUTCOME" if direction == 1 else "SHORT_OUTCOME"
    reward_column = "LONG_REWARD" if direction == 1 else "SHORT_REWARD"
    exit_column = "LONG_EXIT_OFFSET" if direction == 1 else "SHORT_EXIT_OFFSET"
    contexts = gen16.context_codes(frame)
    return pd.DataFrame(
        {
            "index": indices.astype(np.int64),
            "direction": np.full(len(indices), direction, dtype=np.int8),
            "expert": np.full(len(indices), expert, dtype=object),
            "family": np.full(
                len(indices), expert.split("_", 1)[1], dtype=object
            ),
            "context": contexts[indices],
            "raw_p_win": raw_probability.astype(np.float32),
            "p_win": probability,
            "mean_r_prediction": mean_r_prediction,
            "expected_r": expected_r.astype(np.float32),
            "outcome": frame[outcome_column].to_numpy(dtype=np.int8)[indices],
            "reward": frame[reward_column].to_numpy(dtype=np.float32)[indices],
            "exit_offset": frame[exit_column].to_numpy(dtype=np.int16)[indices],
        }
    ).sort_values("index").reset_index(drop=True)


def score_all_experts(
    models: dict[str, dict],
    frame: pd.DataFrame,
    base_features: list[str],
    model_features: list[str],
) -> dict[str, pd.DataFrame]:
    masks = gen16.family_masks(frame, base_features)
    return {
        expert: score_indices(
            models[expert],
            frame,
            np.flatnonzero(gen16.rising_edges(masks[expert])),
            expert,
            model_features,
        )
        for expert in TARGET_EXPERTS
    }


def entry_frame(scores: pd.DataFrame, priority: np.ndarray) -> pd.DataFrame:
    if scores.empty:
        return gen16.empty_entries()
    return pd.DataFrame(
        {
            "index": scores["index"].to_numpy(dtype=np.int64),
            "direction": scores["direction"].to_numpy(dtype=np.int8),
            "priority": np.asarray(priority, dtype=np.float32),
            "expert": scores["expert"].to_numpy(dtype=object),
            "family": scores["family"].to_numpy(dtype=object),
            "context": scores["context"].to_numpy(dtype=np.int64),
            "p_win": scores["p_win"].to_numpy(dtype=np.float32),
            "expected_r": scores["expected_r"].to_numpy(dtype=np.float32),
        }
    )


def daily_top_k(
    entries: pd.DataFrame, frame: pd.DataFrame, top_k: int
) -> pd.DataFrame:
    if entries.empty:
        return entries
    output = entries.copy()
    output["date"] = frame["TIME_DT"].iloc[
        output["index"].to_numpy(dtype=np.int64)
    ].dt.date.to_numpy()
    return (
        output.sort_values(["date", "priority"], ascending=[True, False])
        .groupby("date", sort=False)
        .head(top_k)
        .drop(columns="date")
        .reset_index(drop=True)
    )


def absolute_entries(scores: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    keep = (scores["p_win"] >= gen17.FIXED_MIN_PWIN) & (
        scores["expected_r"] >= gen17.FIXED_MIN_EXPECTED_R
    )
    selected = scores.loc[keep].copy()
    priority = selected["p_win"].to_numpy(dtype=np.float64) + 0.05 * np.tanh(
        selected["expected_r"].to_numpy(dtype=np.float64)
    )
    return daily_top_k(entry_frame(selected, priority), frame, gen17.FIXED_TOP_K)


def percentile_against_history(
    current: np.ndarray, history: np.ndarray
) -> np.ndarray:
    finite_history = np.sort(history[np.isfinite(history)])
    output = np.full(len(current), np.nan, dtype=np.float64)
    finite = np.isfinite(current)
    if len(finite_history) == 0:
        return output
    output[finite] = np.searchsorted(
        finite_history, current[finite], side="right"
    ) / len(finite_history)
    return output


def rank_values(
    current: pd.DataFrame,
    history: pd.DataFrame,
    mode: str,
) -> np.ndarray:
    p_rank = percentile_against_history(
        current["p_win"].to_numpy(dtype=np.float64),
        history["p_win"].to_numpy(dtype=np.float64),
    )
    r_rank = percentile_against_history(
        current["expected_r"].to_numpy(dtype=np.float64),
        history["expected_r"].to_numpy(dtype=np.float64),
    )
    if mode == "p_win_rank":
        return p_rank
    if mode == "expected_r_rank":
        return r_rank
    if mode == "joint_rank":
        return 0.5 * (p_rank + r_rank)
    raise ValueError(mode)


def rolling_rank_entries(
    scores: pd.DataFrame,
    reference: pd.DataFrame,
    mode: str,
    top_fraction: float,
) -> tuple[pd.DataFrame, list[dict]]:
    if scores.empty or reference.empty:
        return gen16.empty_entries(), []
    history = reference[["p_win", "expected_r"]].copy().tail(
        MAX_REFERENCE_EVENTS
    )
    selected = []
    trace = []
    block_ids = scores["index"].to_numpy(dtype=np.int64) // BLOCK_ROWS
    for block_id in np.unique(block_ids):
        block = scores.loc[block_ids == block_id].copy()
        ranks = rank_values(block, history, mode)
        keep = np.isfinite(ranks) & (ranks >= 1.0 - top_fraction)
        if keep.any():
            selected.append(entry_frame(block.loc[keep], ranks[keep]))
        trace.append(
            {
                "block_id": int(block_id),
                "reference_events": len(history),
                "eligible_events": len(block),
                "selected_rows": int(keep.sum()),
                "rank_floor": 1.0 - top_fraction,
            }
        )
        history = pd.concat(
            [history, block[["p_win", "expected_r"]]], ignore_index=True
        ).tail(MAX_REFERENCE_EVENTS)
    if not selected:
        return gen16.empty_entries(), trace
    return pd.concat(selected, ignore_index=True), trace


def compact_economic_metrics(
    records: list[dict], frame: pd.DataFrame
) -> dict:
    days = int(frame["TIME_DT"].dt.date.nunique())
    enriched = payoff.enrich_ledger(records, frame)
    return payoff.payoff_metrics(enriched, days)


def inner_policy_metrics(
    frame: pd.DataFrame,
    entries: pd.DataFrame,
    evaluation_scores: pd.DataFrame,
) -> dict:
    records = gen16.execute_entries(frame, entries, "inner_rank_policy")
    start = int(evaluation_scores["index"].min())
    end = int(evaluation_scores["index"].max())
    days = int(frame["TIME_DT"].iloc[start : end + 1].dt.date.nunique())
    enriched = payoff.enrich_ledger(records, frame)
    return payoff.payoff_metrics(enriched, days)


def inner_policy_pass(value: dict) -> bool:
    return bool(
        value["trades"] >= MIN_INNER_TRADES
        and value["realized_positive_trade_win_rate"] >= RESEARCH_WIN_RATE
        and value["profit_factor"] is not None
        and value["profit_factor"] > 1.0
        and value["mean_r"] > 0.0
        and value["break_even_adjusted_win_rate_edge"] is not None
        and value["break_even_adjusted_win_rate_edge"] > 0.0
    )


def select_inner_policy(
    frame: pd.DataFrame,
    policy_scores: pd.DataFrame,
) -> dict:
    warmup_count = max(
        MIN_INNER_WARMUP,
        int(len(policy_scores) * INNER_WARMUP_FRACTION),
    )
    if warmup_count >= len(policy_scores) - MIN_INNER_TRADES:
        raise RuntimeError("Insufficient policy events for rank selection")
    reference = policy_scores.iloc[:warmup_count].copy()
    evaluation = policy_scores.iloc[warmup_count:].copy()
    candidates = []
    for mode in SCORE_MODES:
        for top_fraction in TOP_FRACTIONS:
            entries, _ = rolling_rank_entries(
                evaluation, reference, mode, top_fraction
            )
            value = inner_policy_metrics(frame, entries, evaluation)
            candidates.append(
                {
                    "score_mode": mode,
                    "top_fraction": top_fraction,
                    "qualified": inner_policy_pass(value),
                    "metrics": value,
                }
            )
    candidates.sort(
        key=lambda item: (
            item["qualified"],
            item["metrics"]["trades"] if item["qualified"] else 0,
            item["metrics"]["realized_positive_trade_win_rate"],
            item["metrics"]["profit_factor"] or 0.0,
            item["metrics"]["mean_r"],
        ),
        reverse=True,
    )
    selected = candidates[0]
    return {
        "warmup_events": len(reference),
        "evaluation_events": len(evaluation),
        "candidate_count": len(candidates),
        "selected": selected,
        "all_candidates": candidates,
        "reference_scores": reference,
        "full_policy_scores": policy_scores,
    }


def train_and_choose_policies(
    train: pd.DataFrame,
    base_features: list[str],
    model_features: list[str],
) -> tuple[dict, dict, dict]:
    models, model_diagnostics = gen17.train_target_experts(
        train, base_features, model_features
    )
    masks = gen16.family_masks(train, base_features)
    policies = {}
    for expert in TARGET_EXPERTS:
        start = model_diagnostics[expert]["policy_start_index"]
        direction = gen16.EXPERT_DIRECTION[expert]
        outcome_column = "LONG_OUTCOME" if direction == 1 else "SHORT_OUTCOME"
        reward_column = "LONG_REWARD" if direction == 1 else "SHORT_REWARD"
        indices = np.flatnonzero(gen16.rising_edges(masks[expert]))
        eligible = (
            (indices >= start)
            & (train[outcome_column].to_numpy(dtype=np.int8)[indices] >= 0)
            & np.isfinite(
                train[reward_column].to_numpy(dtype=np.float32)[indices]
            )
        )
        indices = indices[eligible]
        policy_scores = score_indices(
            models[expert], train, indices, expert, model_features
        )
        policies[expert] = select_inner_policy(train, policy_scores)
    return models, model_diagnostics, policies


def fixed_probability_bins(scores: pd.DataFrame) -> tuple[list[dict], float]:
    probability = scores["p_win"].to_numpy(dtype=np.float64)
    target = (scores["outcome"].to_numpy(dtype=np.int8) == 1).astype(np.float64)
    output = []
    weighted_error = 0.0
    for lower in np.arange(0.0, 1.0, 0.1):
        upper = lower + 0.1
        mask = (probability >= lower) & (
            probability <= upper if upper >= 1.0 else probability < upper
        )
        count = int(mask.sum())
        predicted = None if count == 0 else float(probability[mask].mean())
        observed = None if count == 0 else float(target[mask].mean())
        if count:
            weighted_error += count * abs(predicted - observed)
        output.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_predicted_probability": predicted,
                "observed_tp_first_rate": observed,
            }
        )
    return output, weighted_error / max(len(scores), 1)


def probability_deciles(scores: pd.DataFrame) -> list[dict]:
    if scores.empty:
        return []
    ranked = scores.copy()
    ranked["decile"] = pd.qcut(
        ranked["p_win"].rank(method="first"), 10, labels=False
    )
    output = []
    for decile, group in ranked.groupby("decile", sort=True):
        output.append(
            {
                "decile": int(decile) + 1,
                "count": len(group),
                "p_win_min": float(group["p_win"].min()),
                "p_win_max": float(group["p_win"].max()),
                "p_win_mean": float(group["p_win"].mean()),
                "observed_tp_first_rate": float((group["outcome"] == 1).mean()),
                "realized_positive_rate": float((group["reward"] > 0.0).mean()),
                "mean_r": float(group["reward"].mean()),
            }
        )
    return output


def top_rank_diagnostic(
    scores: pd.DataFrame,
    frame: pd.DataFrame,
    column: str,
    fraction: float,
) -> dict:
    cutoff = float(scores[column].quantile(1.0 - fraction))
    selected = scores[scores[column] >= cutoff].copy()
    entries = entry_frame(selected, selected[column].to_numpy(dtype=np.float64))
    records = gen16.execute_entries(
        frame, entries, f"diagnostic_{column}_top_{fraction:.2f}"
    )
    return {
        "evaluated_interval_used_for_cutoff": True,
        "not_candidate_eligible": True,
        "cutoff": cutoff,
        "metrics": compact_economic_metrics(records, frame),
    }


def calibration_diagnostics(
    scores: pd.DataFrame,
    frame: pd.DataFrame,
) -> dict:
    mature = scores[
        (scores["outcome"] >= 0) & np.isfinite(scores["reward"])
    ].copy()
    curve, ece = fixed_probability_bins(mature)
    target = (mature["outcome"].to_numpy(dtype=np.int8) == 1).astype(np.float64)
    p_win = mature["p_win"].to_numpy(dtype=np.float64)
    absolute = absolute_entries(scores, frame)
    absolute_records = gen16.execute_entries(
        frame, absolute, "absolute_probability_diagnostic"
    )
    def correlation(left: pd.Series, right: pd.Series) -> float | None:
        value = left.corr(right, method="spearman")
        return None if pd.isna(value) else float(value)

    return {
        "events": len(scores),
        "mature_events": len(mature),
        "p_win_quantiles": np.quantile(
            p_win, (0.01, 0.10, 0.50, 0.90, 0.99)
        ).tolist(),
        "expected_r_quantiles": np.quantile(
            mature["expected_r"].to_numpy(dtype=np.float64),
            (0.01, 0.10, 0.50, 0.90, 0.99),
        ).tolist(),
        "brier_score": float(np.mean((p_win - target) ** 2)),
        "expected_calibration_error": ece,
        "reliability_curve": curve,
        "probability_deciles": probability_deciles(mature),
        "rank_correlation": {
            "p_win_vs_tp_first": correlation(
                mature["p_win"], mature["outcome"] == 1
            ),
            "p_win_vs_realized_r": correlation(
                mature["p_win"], mature["reward"]
            ),
            "expected_r_vs_realized_r": correlation(
                mature["expected_r"], mature["reward"]
            ),
        },
        "absolute_selection": compact_economic_metrics(absolute_records, frame),
        "top_percentile_diagnostic": {
            column: {
                str(fraction): top_rank_diagnostic(
                    mature, frame, column, fraction
                )
                for fraction in TOP_FRACTIONS
            }
            for column in ("p_win", "expected_r")
        },
    }


def apply_rank_policies(
    frame: pd.DataFrame,
    all_scores: dict[str, pd.DataFrame],
    policies: dict,
) -> tuple[dict[str, pd.DataFrame], dict]:
    entries = {}
    traces = {}
    for expert in TARGET_EXPERTS:
        choice = policies[expert]["selected"]
        selected, trace = rolling_rank_entries(
            all_scores[expert],
            policies[expert]["full_policy_scores"],
            choice["score_mode"],
            choice["top_fraction"],
        )
        entries[expert] = selected
        traces[expert] = trace
    return entries, traces


def candidate_grid() -> list[dict]:
    return [
        {
            "candidate_id": f"gen18_rank_{label}",
            "experts": experts,
        }
        for label, experts in (
            ("long_breakout", ("long_breakout",)),
            ("short_breakout", ("short_breakout",)),
            ("short_trend_continuation", ("short_trend_continuation",)),
            ("target_portfolio", TARGET_EXPERTS),
        )
    ]


def evaluate_candidates(
    frame: pd.DataFrame,
    period: str,
    rank_entries: dict[str, pd.DataFrame],
    candidates: list[dict],
    gen17_baseline: list[dict],
    parent_baseline: list[dict],
) -> dict:
    output = {}
    for candidate in candidates:
        pieces = [
            rank_entries[expert]
            for expert in candidate["experts"]
            if not rank_entries[expert].empty
        ]
        entries = (
            pd.concat(pieces, ignore_index=True)
            if pieces
            else gen16.empty_entries()
        )
        records = gen16.execute_entries(frame, entries, period)
        stress_records = gen16.execute_entries(
            frame, entries, f"{period}_cost_10", gen16.STRESS_COST_POINTS
        )
        value = gen16.metrics(records, frame)
        stress = gen16.metrics(stress_records, frame)
        result = gen16.result_record(
            value, stress, records, stress_records, gen17_baseline
        )
        result["payoff"] = compact_economic_metrics(records, frame)
        result["comparison_to_parent"] = gen16.compare_records(
            records, parent_baseline
        )
        output[candidate["candidate_id"]] = result
    return output


def pooled_payoff(
    values: list[dict],
    frames: dict[str, pd.DataFrame],
) -> dict:
    enriched = []
    days = 0
    for (fold_name, *_), value in zip(SELECTION_FOLDS, values):
        enriched.extend(payoff.enrich_ledger(value["trade_ledger"], frames[fold_name]))
        days += value["metrics"]["evaluated_days"]
    return payoff.payoff_metrics(enriched, days)


def aggregate_candidates(
    candidates: list[dict],
    fold_results: dict,
    fold_frames: dict[str, pd.DataFrame],
    policy_diagnostics: dict,
    report17: dict,
) -> list[dict]:
    ranked = []
    parent = report17["selected"]["selection_summary"]["pooled"]
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        values = [
            fold_results[fold_name][candidate_id]
            for fold_name, *_ in SELECTION_FOLDS
        ]
        pooled = pooled_payoff(values, fold_frames)
        stress_records = [
            record
            for value in values
            for record in value["cost_stress_trade_ledger"]
        ]
        stress_rewards = np.asarray(
            [record["reward"] for record in stress_records], dtype=np.float64
        )
        fold_payoffs = [value["payoff"] for value in values]
        policies_qualified = all(
            policy_diagnostics[fold_name][expert]["selected"]["qualified"]
            for fold_name, *_ in SELECTION_FOLDS
            for expert in candidate["experts"]
        )
        catastrophic = any(
            value["trades"] < 5
            or value["realized_positive_trade_win_rate"]
            < WORST_FOLD_WIN_RATE
            or value["max_drawdown_pct"] < MAX_CATASTROPHIC_DRAWDOWN
            or (
                value["profit_factor"] is not None
                and value["profit_factor"] < MIN_CATASTROPHIC_PF
            )
            for value in fold_payoffs
        )
        stress_pf = gen16.profit_factor(stress_rewards)
        stress_mean_r = (
            float(stress_rewards.mean()) if len(stress_rewards) else 0.0
        )
        discovery_pass = bool(
            policies_qualified
            and pooled["realized_positive_trade_win_rate"] >= RESEARCH_WIN_RATE
            and min(
                value["realized_positive_trade_win_rate"]
                for value in fold_payoffs
            )
            >= WORST_FOLD_WIN_RATE
            and pooled["profit_factor"] is not None
            and pooled["profit_factor"] > 1.0
            and pooled["mean_r"] > 0.0
            and pooled["pnl"] > 0.0
            and pooled["break_even_adjusted_win_rate_edge"] is not None
            and pooled["break_even_adjusted_win_rate_edge"] > 0.0
            and stress_pf is not None
            and stress_pf > 1.0
            and stress_mean_r > 0.0
            and not catastrophic
        )
        simultaneous = bool(
            pooled["trades"] > parent["trades"]
            and pooled["realized_positive_trade_win_rate"]
            > parent["win_rate"]
            and pooled["profit_factor"] is not None
            and pooled["profit_factor"] > 1.0
            and pooled["mean_r"] > 0.0
        )
        ranked.append(
            {
                **candidate,
                "inner_policies_qualified": policies_qualified,
                "discovery_pass": discovery_pass,
                "simultaneously_improves_gen17_win_and_frequency": simultaneous,
                "pooled": pooled,
                "cost_stress": {
                    "trades": len(stress_records),
                    "profit_factor": stress_pf,
                    "mean_r": stress_mean_r,
                },
                "worst_fold_realized_win_rate": min(
                    value["realized_positive_trade_win_rate"]
                    for value in fold_payoffs
                ),
                "catastrophic_fold": catastrophic,
            }
        )
    ranked.sort(
        key=lambda item: (
            item["discovery_pass"],
            item["simultaneously_improves_gen17_win_and_frequency"],
            item["pooled"]["break_even_adjusted_win_rate_edge"] or -1.0,
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
            other["pooled"]["realized_positive_trade_win_rate"]
            >= item["pooled"]["realized_positive_trade_win_rate"]
            and other["pooled"]["trades_per_day"]
            >= item["pooled"]["trades_per_day"]
            and (
                other["pooled"]["realized_positive_trade_win_rate"]
                > item["pooled"]["realized_positive_trade_win_rate"]
                or other["pooled"]["trades_per_day"]
                > item["pooled"]["trades_per_day"]
            )
            for other in feasible
            if other["candidate_id"] != item["candidate_id"]
        )
        if not dominated:
            output.append(item["candidate_id"])
    return output


def public_policy_diagnostics(policies: dict) -> dict:
    return {
        fold: {
            expert: {
                "warmup_events": value["warmup_events"],
                "evaluation_events": value["evaluation_events"],
                "candidate_count": value["candidate_count"],
                "selected": value["selected"],
                "all_candidates": value["all_candidates"],
            }
            for expert, value in experts.items()
        }
        for fold, experts in policies.items()
    }


def stability_summary(calibration: dict) -> dict:
    output = {}
    for expert in TARGET_EXPERTS:
        folds = [calibration[fold][expert] for fold, *_ in SELECTION_FOLDS]
        ece = [value["expected_calibration_error"] for value in folds]
        medians = [value["p_win_quantiles"][2] for value in folds]
        top_results = {}
        for column in ("p_win", "expected_r"):
            for fraction in TOP_FRACTIONS:
                key = f"{column}_top_{fraction:.2f}"
                win_rates = [
                    value["top_percentile_diagnostic"][column][str(fraction)][
                        "metrics"
                    ]["realized_positive_trade_win_rate"]
                    for value in folds
                ]
                top_results[key] = {
                    "fold_realized_win_rates": win_rates,
                    "minimum": min(win_rates),
                    "maximum": max(win_rates),
                    "standard_deviation": float(np.std(win_rates)),
                }
        output[expert] = {
            "ece_by_fold": ece,
            "ece_range": max(ece) - min(ece),
            "p_win_median_by_fold": medians,
            "p_win_median_range": max(medians) - min(medians),
            "top_rank_stability": top_results,
        }
    return output


def markdown_report(report: dict) -> str:
    lines = [
        "# Generation 18 - Payoff Alignment and Calibration-Robust Ranking",
        "",
        "No signal-family expansion and no absolute-threshold sweep were used.",
        "",
        "| Candidate | Trades | Trades/day | TP-first | Realized win | Avg win R | Avg loss R | Payoff | Break-even | Edge | PF | Mean-R | PnL | Max DD | Discovery |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["selection"]["ranked"]:
        value = item["pooled"]
        lines.append(
            f"| {item['candidate_id']} | {value['trades']} | "
            f"{value['trades_per_day']:.3f} | {value['tp_first_rate']:.2%} | "
            f"{value['realized_positive_trade_win_rate']:.2%} | "
            f"{value['average_winning_r'] or 0.0:.4f} | "
            f"{value['average_losing_r'] or 0.0:.4f} | "
            f"{value['payoff_ratio'] or 0.0:.4f} | "
            f"{value['realized_break_even_win_rate'] or 0.0:.2%} | "
            f"{value['break_even_adjusted_win_rate_edge'] or 0.0:.2%} | "
            f"{value['profit_factor'] or 0.0:.2f} | {value['mean_r']:.4f} | "
            f"{value['pnl']:.2f} | {value['max_drawdown_pct']:.2%} | "
            f"{'PASS' if item['discovery_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"Qualified candidates: `{report['selection']['qualified_count']}`",
            f"Pareto frontier: `{json.dumps(report['selection']['pareto_frontier'])}`",
            f"Frozen status: `{report['selected']['status']}`",
            "Production promotion: `False`",
        ]
    )
    return "\n".join(lines) + "\n"


def calibration_markdown(report: dict) -> str:
    lines = [
        "# Generation 18 calibration drift",
        "",
        "| Expert | Fold | Events | P50 P(win) | ECE | Absolute trades | Absolute realized win | Absolute PF |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fold, experts in report["folds"].items():
        for expert, value in experts.items():
            absolute = value["absolute_selection"]
            lines.append(
                f"| {expert} | {fold} | {value['events']} | "
                f"{value['p_win_quantiles'][2]:.4f} | "
                f"{value['expected_calibration_error']:.4f} | "
                f"{absolute['trades']} | "
                f"{absolute['realized_positive_trade_win_rate']:.2%} | "
                f"{absolute['profit_factor'] or 0.0:.2f} |"
            )
    return "\n".join(lines) + "\n"


def self_check() -> None:
    history = pd.DataFrame({"p_win": [0.1, 0.2, 0.3], "expected_r": [-1, 0, 1]})
    current = pd.DataFrame({"p_win": [0.15, 0.35], "expected_r": [-0.5, 2]})
    ranks = rank_values(current, history, "joint_rank")
    assert np.allclose(ranks, (1 / 3, 1.0))
    assert len(candidate_grid()) == 4
    assert len(SCORE_MODES) * len(TOP_FRACTIONS) == 12
    print("generation18_payoff_alignment_self_check_ok")


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0
    report15 = json.loads(GEN15_REPORT.read_text(encoding="utf-8"))
    report17 = json.loads(GEN17_REPORT.read_text(encoding="utf-8"))
    payoff_report = json.loads(PAYOFF_REPORT.read_text(encoding="utf-8"))
    history, base_features = prepare_barrier_data()
    history = add_targets(history)
    history, regime_features = gen17.add_regime_features(history, base_features)
    robust_base_features = [
        feature
        for feature in base_features
        if feature not in gen17.ABSOLUTE_SCALE_FEATURES
    ]
    model_features = [*robust_base_features, *regime_features]
    candidates = candidate_grid()
    fold_results = {}
    fold_frames = {}
    calibration = {}
    absolute_reproduction = {}
    model_diagnostics = {}
    policies = {}
    rank_traces = {}
    for fold_name, fold_start, fold_end in SELECTION_FOLDS:
        train = training_frame(history, fold_start)
        evaluation = history[
            (history["TIME_DT"] >= fold_start)
            & (history["TIME_DT"] < fold_end)
        ].reset_index(drop=True)
        models, diagnostics, fold_policies = train_and_choose_policies(
            train, base_features, model_features
        )
        all_scores = score_all_experts(
            models, evaluation, base_features, model_features
        )
        calibration[fold_name] = {
            expert: calibration_diagnostics(scores, evaluation)
            for expert, scores in all_scores.items()
        }
        absolute_short = gen16.execute_entries(
            evaluation,
            absolute_entries(all_scores["short_trend_continuation"], evaluation),
            f"{fold_name}_absolute_reproduction",
        )
        expected_ids = [
            record["trade_id"]
            for record in report17["selected"]["results"][fold_name][
                "trade_ledger"
            ]
        ]
        actual_ids = [record["trade_id"] for record in absolute_short]
        if actual_ids != expected_ids:
            raise RuntimeError(f"Gen17 absolute ledger mismatch in {fold_name}")
        absolute_reproduction[fold_name] = True
        ranked_entries, traces = apply_rank_policies(
            evaluation, all_scores, fold_policies
        )
        gen17_baseline = report17["selected"]["results"][fold_name][
            "trade_ledger"
        ]
        parent_baseline = gen16.baseline_records(report15, fold_name)
        fold_results[fold_name] = evaluate_candidates(
            evaluation,
            fold_name,
            ranked_entries,
            candidates,
            gen17_baseline,
            parent_baseline,
        )
        fold_frames[fold_name] = evaluation
        model_diagnostics[fold_name] = diagnostics
        policies[fold_name] = fold_policies
        rank_traces[fold_name] = traces
        print(f"Fold {fold_name} complete", flush=True)

    public_policies = public_policy_diagnostics(policies)
    ranked = aggregate_candidates(
        candidates, fold_results, fold_frames, public_policies, report17
    )
    qualified = [item for item in ranked if item["discovery_pass"]]
    selected_summary = qualified[0] if qualified else ranked[0]
    selected_candidate = next(
        candidate
        for candidate in candidates
        if candidate["candidate_id"] == selected_summary["candidate_id"]
    )
    selected_id = selected_candidate["candidate_id"]
    selected_results = {
        fold_name: fold_results[fold_name][selected_id]
        for fold_name, *_ in SELECTION_FOLDS
    }

    holdout = history[
        history["TIME_DT"] >= HISTORICAL_HOLDOUT_START
    ].reset_index(drop=True)
    holdout_train = training_frame(history, HISTORICAL_HOLDOUT_START)
    holdout_models, holdout_diagnostics, holdout_policies = (
        train_and_choose_policies(holdout_train, base_features, model_features)
    )
    holdout_scores = score_all_experts(
        holdout_models, holdout, base_features, model_features
    )
    holdout_entries, holdout_traces = apply_rank_policies(
        holdout, holdout_scores, holdout_policies
    )
    holdout_result = evaluate_candidates(
        holdout,
        "2025_2026_05_development",
        holdout_entries,
        [selected_candidate],
        report17["selected"]["results"]["2025_2026_05_development"][
            "trade_ledger"
        ],
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
    if set(recent_features) != set(base_features):
        raise RuntimeError("Historical and recent base features differ")
    recent_all = add_targets(recent_all)
    recent_all, _ = gen17.add_regime_features(recent_all, base_features)
    recent_end = pd.Timestamp(report17["data"]["recent_development_end"]) + pd.Timedelta(
        minutes=1
    )
    recent = recent_all[
        (recent_all["TIME_DT"] >= RECENT_START.replace(tzinfo=None))
        & (recent_all["TIME_DT"] < recent_end)
    ].reset_index(drop=True)
    final_cutoff = history["TIME_DT"].iloc[-1] + pd.Timedelta(minutes=1)
    final_train = training_frame(history, final_cutoff)
    final_models, final_diagnostics, final_policies = train_and_choose_policies(
        final_train, base_features, model_features
    )
    recent_scores = score_all_experts(
        final_models, recent, base_features, model_features
    )
    recent_entries, recent_traces = apply_rank_policies(
        recent, recent_scores, final_policies
    )
    recent_result = evaluate_candidates(
        recent,
        "2026_recent_development",
        recent_entries,
        [selected_candidate],
        report17["selected"]["results"]["2026_recent_development"][
            "trade_ledger"
        ],
        gen16.baseline_records(report15, "2026_recent"),
    )[selected_id]
    selected_results["2025_2026_05_development"] = holdout_result
    selected_results["2026_recent_development"] = recent_result

    calibration_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_only",
        "folds": calibration,
        "absolute_gen17_ledger_reproduced": absolute_reproduction,
        "stability": stability_summary(calibration),
        "diagnostic_cutoffs_use_evaluated_interval": True,
        "diagnostic_top_percentiles_not_candidate_eligible": True,
    }
    CALIBRATION_JSON.write_text(
        json.dumps(calibration_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    CALIBRATION_MD.write_text(
        calibration_markdown(calibration_report), encoding="utf-8"
    )

    config = {
        "generation": "18_payoff_alignment_calibration_robust_ranking",
        "status": "research_only",
        "candidate_status": (
            "research_candidate" if selected_summary["discovery_pass"]
            else "diagnostic_fallback"
        ),
        "selected": selected_candidate if selected_summary["discovery_pass"] else None,
        "diagnostic_fallback": (
            None if selected_summary["discovery_pass"] else selected_candidate
        ),
        "score_modes": list(SCORE_MODES),
        "top_fractions": list(TOP_FRACTIONS),
        "block_rows": BLOCK_ROWS,
        "selection_folds": public_policies,
        "promotion_pass": False,
    }
    CONFIG_JSON.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generation": "18_payoff_alignment_calibration_robust_ranking",
        "status": "research_only",
        "development_history_policy": report17["development_history_policy"],
        "phase1_payoff_audit": PAYOFF_REPORT.name,
        "phase1_key_result": payoff_report["strategies"][
            "gen17_short_trend_diagnostic"
        ]["selection_pooled"],
        "data": report17["data"],
        "architecture": {
            "signal_family_expansion": False,
            "absolute_threshold_sweep": False,
            "target_experts": list(TARGET_EXPERTS),
            "score_modes": list(SCORE_MODES),
            "top_fractions": list(TOP_FRACTIONS),
            "inner_policy_candidate_count_per_expert": len(SCORE_MODES)
            * len(TOP_FRACTIONS),
            "ranking": "past-only empirical CDF, updated after each block",
            "block_rows": BLOCK_ROWS,
            "maximum_reference_events": MAX_REFERENCE_EVENTS,
            "model_features": model_features,
            "execution_profile_unchanged": True,
        },
        "calibration_drift_report": CALIBRATION_JSON.name,
        "inner_policy_selection": public_policies,
        "model_diagnostics": model_diagnostics,
        "diagnostic_model_diagnostics": {
            "2025_2026_05_development": holdout_diagnostics,
            "2026_recent_development": final_diagnostics,
        },
        "rank_trace": rank_traces,
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
            "params": selected_candidate,
            "selection_summary": selected_summary,
            "results": selected_results,
            "diagnostic_policy_selection": {
                "2025_2026_05_development": public_policy_diagnostics(
                    {"period": holdout_policies}
                )["period"],
                "2026_recent_development": public_policy_diagnostics(
                    {"period": final_policies}
                )["period"],
            },
            "diagnostic_rank_traces": {
                "2025_2026_05_development": holdout_traces,
                "2026_recent_development": recent_traces,
            },
        },
        "production_promotion": False,
        "validation_pending": True,
    }
    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REPORT_MD.write_text(markdown_report(report), encoding="utf-8")
    print(markdown_report(report), flush=True)
    print(calibration_markdown(calibration_report), flush=True)
    print(
        f"Saved {REPORT_JSON.name}, {REPORT_MD.name}, "
        f"{CALIBRATION_JSON.name}, {CALIBRATION_MD.name}, {CONFIG_JSON.name}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
