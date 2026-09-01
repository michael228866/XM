from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import gold_generation16_independent_families as gen16
import gold_generation17_cross_regime as gen17
import gold_generation19_cost_aware as gen19
from barrier_final_train import prepare_barrier_data
from gold_generation11_execution_aligned import add_targets
from gold_recent_walk_forward import DEFAULT_TERMINAL, build_feature_frame
from gold_regime_experts_iterative import training_frame
from gold_regime_experts_walk_forward import RECENT_START, SELECTION_FOLDS


ROOT = Path(__file__).resolve().parent
GEN19_REPORT = ROOT / "gold_generation19_cost_aware.json"
REPORT_JSON = ROOT / "gold_generation20_direct_net_edge.json"
REPORT_MD = ROOT / "gold_generation20_direct_net_edge.md"
CONFIG_JSON = ROOT / "gold_generation20_candidate.json"

EXPERT = "short_trend_continuation"
DIRECTION = 2
BLOCK_ROWS = 10_080
REFERENCE_ROWS = 28_800
MIN_REFERENCE_ROWS = 1_000
POLICIES = (
    "p_net_ge_050",
    "e_net_positive",
    "joint_positive",
    "e_net_top75_past",
)
COST_FEATURES = (
    "COST_SPREAD_POINTS",
    "COST_SPREAD_ATR",
    "COST_SPREAD_SL",
    "COST_TOTAL_SL",
    "COST_SPREAD_PCTL",
    "COST_SPREAD_ATR_PCTL",
    "COST_SPREAD_ATR_Z",
    "COST_ATR_PCTL",
    "COST_VOLATILITY_PCTL",
    "COST_SESSION",
)
RECENT_WARMUP_START = datetime(2026, 4, 1, tzinfo=timezone.utc)
DEVELOPMENT_END = datetime(2026, 9, 1, tzinfo=timezone.utc)
HISTORICAL_HOLDOUT_START = datetime(2025, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generation 20 direct net-edge learning")
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def past_block_relative(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    percentile = np.full(len(values), 0.5, dtype=np.float32)
    zscore = np.zeros(len(values), dtype=np.float32)
    for start in range(0, len(values), BLOCK_ROWS):
        end = min(start + BLOCK_ROWS, len(values))
        history = values[max(0, start - REFERENCE_ROWS) : start]
        history = history[np.isfinite(history)]
        current = values[start:end]
        finite = np.isfinite(current)
        if len(history) < MIN_REFERENCE_ROWS or not finite.any():
            continue
        ordered = np.sort(history)
        percentile[start:end][finite] = (
            np.searchsorted(ordered, current[finite], side="right") / len(ordered)
        ).astype(np.float32)
        scale = float(history.std())
        if scale > 1e-12:
            zscore[start:end][finite] = ((current[finite] - history.mean()) / scale).astype(np.float32)
    return percentile, zscore


def add_cost_targets_and_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    indices = np.arange(len(result), dtype=np.int64)
    spread_points, observed = gen19.effective_spread_points(result, indices)
    atr = result["ATR"].to_numpy(dtype=np.float64)
    stop = np.maximum(atr * gen16.SL_ATR, gen16.MIN_SL_PRICE)
    spread_price = spread_points * gen19.POINT
    fixed_reward = result["SHORT_REWARD"].to_numpy(dtype=np.float64)
    gross_pnl = fixed_reward * (stop + gen19.FALLBACK_SPREAD_POINTS * gen19.POINT) + (
        gen19.FALLBACK_SPREAD_POINTS + gen19.BASE_EXTRA_COST_POINTS
    ) * gen19.POINT
    net_reward = (
        gross_pnl
        - (spread_points + gen19.BASE_EXTRA_COST_POINTS) * gen19.POINT
    ) / (stop + spread_price)
    spread_atr = spread_price / np.maximum(atr, 1e-9)
    spread_pctl, _ = past_block_relative(spread_points)
    spread_atr_pctl, spread_atr_z = past_block_relative(spread_atr)
    atr_pctl, _ = past_block_relative(atr)
    hour = result["TIME_DT"].dt.hour.to_numpy(dtype=np.int8)
    session = np.select(
        (hour <= 5, hour <= 12, hour <= 18),
        (0, 1, 2),
        default=3,
    )
    result["NET_REWARD"] = net_reward
    result["NET_POSITIVE"] = (net_reward > 0.0).astype(np.int8)
    result["SPREAD_QUALITY"] = np.where(observed, "observed", "fallback")
    result["COST_SPREAD_POINTS"] = spread_points.astype(np.float32)
    result["COST_SPREAD_ATR"] = spread_atr.astype(np.float32)
    result["COST_SPREAD_SL"] = (spread_price / stop).astype(np.float32)
    result["COST_TOTAL_SL"] = (
        (spread_price + gen19.BASE_EXTRA_COST_POINTS * gen19.POINT) / stop
    ).astype(np.float32)
    result["COST_SPREAD_PCTL"] = spread_pctl
    result["COST_SPREAD_ATR_PCTL"] = spread_atr_pctl
    result["COST_SPREAD_ATR_Z"] = spread_atr_z
    result["COST_ATR_PCTL"] = atr_pctl
    result["COST_VOLATILITY_PCTL"] = result["REG_RV_PCTL_20D"].astype(np.float32)
    result["COST_SESSION"] = session.astype(np.float32)
    return result


def event_indices(frame: pd.DataFrame, base_features: list[str]) -> np.ndarray:
    masks = gen16.family_masks(frame, base_features)
    indices = np.flatnonzero(gen16.rising_edges(masks[EXPERT]))
    mature = (
        (frame["SHORT_OUTCOME"].to_numpy(dtype=np.int8)[indices] >= 0)
        & np.isfinite(frame["NET_REWARD"].to_numpy(dtype=np.float64)[indices])
        & (frame["SHORT_EXIT_OFFSET"].to_numpy(dtype=np.int16)[indices] > 0)
    )
    return indices[mature]


def quality_stats(frame: pd.DataFrame, indices: np.ndarray) -> dict:
    output = {}
    reward = frame["NET_REWARD"].to_numpy(dtype=np.float64)
    quality = frame["SPREAD_QUALITY"].to_numpy(dtype=object)
    for label in ("observed", "fallback", "combined"):
        selected = indices if label == "combined" else indices[quality[indices] == label]
        values = reward[selected]
        output[label] = {
            "events": len(selected),
            "positive_rate": float((values > 0.0).mean()) if len(values) else None,
            "mean_r": float(values.mean()) if len(values) else None,
        }
    return output


def train_direct_models(
    frame: pd.DataFrame,
    base_features: list[str],
    model_features: list[str],
) -> tuple[dict, dict]:
    indices = event_indices(frame, base_features)
    fit_end = int(len(indices) * gen16.MODEL_PROFILE["fit_ratio"])
    calibration_end = int(
        len(indices)
        * (gen16.MODEL_PROFILE["fit_ratio"] + gen16.MODEL_PROFILE["calibration_ratio"])
    )
    if fit_end <= 0 or calibration_end >= len(indices):
        raise RuntimeError("Insufficient direct-model events")
    calibration_start = int(indices[fit_end])
    policy_start = int(indices[calibration_end])
    exits = frame["SHORT_EXIT_OFFSET"].to_numpy(dtype=np.int16)
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
            f"Direct split too small: {len(fit_indices)}/{len(calibration_indices)}/{len(policy_indices)}"
        )
    target = frame["NET_POSITIVE"].to_numpy(dtype=np.int8)
    reward = frame["NET_REWARD"].to_numpy(dtype=np.float32)
    if np.unique(target[fit_indices]).size != 2 or np.unique(target[calibration_indices]).size != 2:
        raise RuntimeError("Direct classifier split lacks both classes")
    classifier = gen16.new_classifier()
    regressor = gen16.new_regressor()
    classifier.fit(frame.iloc[fit_indices][model_features].astype(np.float32), target[fit_indices])
    regressor.fit(frame.iloc[fit_indices][model_features].astype(np.float32), reward[fit_indices])
    raw_calibration = classifier.predict_proba(
        frame.iloc[calibration_indices][model_features].astype(np.float32)
    )[:, 1]
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(raw_calibration, target[calibration_indices])
    policy_x = frame.iloc[policy_indices][model_features].astype(np.float32)
    policy_p = calibrator.predict(classifier.predict_proba(policy_x)[:, 1]).astype(np.float32)
    policy_e = regressor.predict(policy_x).astype(np.float32)
    thresholds = {
        "p_net_ge_050": {"p_min": 0.5, "e_min": None},
        "e_net_positive": {"p_min": None, "e_min": 0.0},
        "joint_positive": {"p_min": 0.5, "e_min": 0.0},
        "e_net_top75_past": {
            "p_min": None,
            "e_min": float(np.quantile(policy_e[np.isfinite(policy_e)], 0.25)),
        },
    }
    models = {
        "classifier": classifier,
        "regressor": regressor,
        "calibrator": calibrator,
        "thresholds": thresholds,
    }
    diagnostics = {
        "events": len(indices),
        "fit": len(fit_indices),
        "calibration": len(calibration_indices),
        "policy": len(policy_indices),
        "fit_max_label_end_index": int(np.max(fit_indices + exits[fit_indices])),
        "calibration_start_index": calibration_start,
        "calibration_max_label_end_index": int(
            np.max(calibration_indices + exits[calibration_indices])
        ),
        "policy_start_index": policy_start,
        "spread_quality": {
            "fit": quality_stats(frame, fit_indices),
            "calibration": quality_stats(frame, calibration_indices),
            "policy": quality_stats(frame, policy_indices),
        },
        "sample_weighting": "none; spread quality is stratification metadata only",
        "thresholds": thresholds,
        "policy_score_distribution": {
            "p_net": np.quantile(policy_p, (0.1, 0.25, 0.5, 0.75, 0.9)).tolist(),
            "e_net": np.quantile(policy_e, (0.1, 0.25, 0.5, 0.75, 0.9)).tolist(),
        },
        "feature_importance": {
            "p_net": classifier.get_booster().get_score(importance_type="gain"),
            "e_net": regressor.get_booster().get_score(importance_type="gain"),
        },
    }
    return models, diagnostics


def predict_direct(
    models: dict,
    frame: pd.DataFrame,
    indices: np.ndarray,
    model_features: list[str],
) -> pd.DataFrame:
    if len(indices) == 0:
        return pd.DataFrame(columns=("index", "p_net", "e_net"))
    x = frame.iloc[indices][model_features].astype(np.float32)
    p_net = models["calibrator"].predict(
        models["classifier"].predict_proba(x)[:, 1]
    ).astype(np.float32)
    e_net = models["regressor"].predict(x).astype(np.float32)
    return pd.DataFrame(
        {
            "index": indices.astype(np.int64),
            "p_net": p_net,
            "e_net": e_net,
            "net_reward": frame["NET_REWARD"].to_numpy(dtype=np.float64)[indices],
            "net_positive": frame["NET_POSITIVE"].to_numpy(dtype=np.int8)[indices],
            "spread_quality": frame["SPREAD_QUALITY"].to_numpy(dtype=object)[indices],
        }
    )


def calibration_stats(probability: np.ndarray, target: np.ndarray) -> dict:
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    finite = np.isfinite(probability) & np.isfinite(target)
    probability, target = probability[finite], target[finite]
    if len(probability) == 0:
        return {"events": 0, "brier": None, "ece": None, "bins": []}
    bins, weighted_error = [], 0.0
    for lower in np.arange(0.0, 1.0, 0.2):
        upper = lower + 0.2
        keep = (probability >= lower) & (
            probability <= upper if upper >= 1.0 else probability < upper
        )
        count = int(keep.sum())
        predicted = float(probability[keep].mean()) if count else None
        observed = float(target[keep].mean()) if count else None
        if count:
            weighted_error += count * abs(predicted - observed)
        bins.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_predicted": predicted,
                "observed_positive_rate": observed,
            }
        )
    return {
        "events": len(probability),
        "brier": float(np.mean((probability - target) ** 2)),
        "ece": weighted_error / len(probability),
        "bins": bins,
    }


def attach_predictions(
    records: list[dict],
    predictions: pd.DataFrame,
    frame: pd.DataFrame,
) -> list[dict]:
    lookup = predictions.set_index("index")
    output = []
    for record in records:
        index = int(record["index"])
        if index not in lookup.index:
            raise RuntimeError(f"Fixed-cohort index {index} is not an eligible direct-model event")
        score = lookup.loc[index]
        expected_reward = float(frame["NET_REWARD"].iat[index])
        if not math.isclose(expected_reward, float(record["reward"]), rel_tol=1e-6, abs_tol=1e-6):
            raise RuntimeError(f"Gen19 cost-label mismatch for {record['trade_id']}")
        enriched = dict(record)
        enriched.update(
            {
                "p_tp_first": float(record["p_win"]),
                "p_net": float(score["p_net"]),
                "e_net": float(score["e_net"]),
                "spread_quality": str(score["spread_quality"]),
            }
        )
        output.append(enriched)
    return output


def policy_keep(record: dict, policy: str, threshold: dict) -> bool:
    if policy == "p_net_ge_050":
        return record["p_net"] >= 0.5
    if policy in ("e_net_positive", "e_net_top75_past"):
        return record["e_net"] >= float(threshold["e_min"])
    if policy == "joint_positive":
        return record["p_net"] >= 0.5 and record["e_net"] >= 0.0
    raise ValueError(policy)


def spread_quality_contribution(records: list[dict]) -> dict:
    output = {}
    for quality in ("observed", "fallback"):
        selected = [record for record in records if record["spread_quality"] == quality]
        output[quality] = gen19.small_metrics(selected)
    return output


def filter_comparison(candidate: list[dict], baseline: list[dict]) -> dict:
    return gen19.compare_records(candidate, baseline)


def ranking_diagnostic(
    records: list[dict],
    score_name: str,
    evaluated_days: int,
) -> dict:
    score = np.asarray([record[score_name] for record in records], dtype=np.float64)
    reward = np.asarray([record["reward"] for record in records], dtype=np.float64)
    correlation = pd.Series(score).corr(pd.Series(reward), method="spearman")
    cutoff = float(np.quantile(score[np.isfinite(score)], 0.5))
    selected = [record for record in records if record[score_name] >= cutoff]
    return {
        "spearman_vs_net_r": None if pd.isna(correlation) else float(correlation),
        "evaluated_interval_median_cutoff": cutoff,
        "selection_eligible": False,
        "top_half_metrics": gen19.metrics(selected, evaluated_days),
    }


def process_fixed_period(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    baseline: dict,
    period: str,
    base_features: list[str],
    model_features: list[str],
) -> tuple[dict, dict, dict, pd.DataFrame]:
    models, model_diagnostics = train_direct_models(
        train, base_features, model_features
    )
    eligible = event_indices(evaluation, base_features)
    predictions = predict_direct(models, evaluation, eligible, model_features)
    baseline_records = attach_predictions(
        baseline["trade_ledger"], predictions, evaluation
    )
    stress_by_id = {
        record["trade_id"]: record
        for record in baseline["cost_stress_trade_ledger"]
    }
    days = int(evaluation["TIME_DT"].dt.date.nunique())
    target = np.asarray(
        [record["reward"] > 0.0 for record in baseline_records], dtype=np.int8
    )
    calibration = {
        "p_tp_first": calibration_stats(
            np.asarray([record["p_tp_first"] for record in baseline_records]),
            target,
        ),
        "p_net": calibration_stats(
            np.asarray([record["p_net"] for record in baseline_records]),
            target,
        ),
    }
    ranking = {
        score: ranking_diagnostic(baseline_records, score, days)
        for score in ("p_tp_first", "p_net", "e_net")
    }
    candidates = {}
    for policy in POLICIES:
        threshold = models["thresholds"][policy]
        selected = [
            record
            for record in baseline_records
            if policy_keep(record, policy, threshold)
        ]
        selected_ids = {record["trade_id"] for record in selected}
        stress = [
            {**stress_by_id[record["trade_id"]], **{
                "p_tp_first": record["p_tp_first"],
                "p_net": record["p_net"],
                "e_net": record["e_net"],
                "spread_quality": record["spread_quality"],
            }}
            for record in selected
            if record["trade_id"] in stress_by_id
        ]
        if len(stress) != len(selected_ids):
            raise RuntimeError(f"Stress ledger identity mismatch in {period}/{policy}")
        metrics = gen19.metrics(selected, days)
        metrics["p_net_calibration"] = calibration_stats(
            np.asarray([record["p_net"] for record in selected]),
            np.asarray([record["reward"] > 0.0 for record in selected]),
        )
        candidates[policy] = {
            "threshold": threshold,
            "metrics": metrics,
            "cost_stress": gen19.metrics(stress, days),
            "comparison_to_gen17": filter_comparison(selected, baseline_records),
            "spread_quality_contribution": spread_quality_contribution(selected),
            "trade_ledger": selected,
            "cost_stress_trade_ledger": stress,
        }
    baseline_with_predictions = {
        "metrics": gen19.metrics(baseline_records, days),
        "cost_stress": None,
        "spread_quality_contribution": spread_quality_contribution(
            baseline_records
        ),
        "trade_ledger": baseline_records,
    }
    baseline_stress = [
        {
            **stress_by_id[source["trade_id"]],
            **{
                "p_tp_first": source["p_tp_first"],
                "p_net": source["p_net"],
                "e_net": source["e_net"],
                "spread_quality": source["spread_quality"],
            },
        }
        for source in baseline_records
    ]
    baseline_with_predictions["cost_stress"] = gen19.metrics(
        baseline_stress, days
    )
    baseline_with_predictions["cost_stress_trade_ledger"] = baseline_stress
    diagnostics = {
        "model": model_diagnostics,
        "fixed_cohort_calibration": calibration,
        "fixed_cohort_ranking": ranking,
        "fixed_cohort_spread_quality": spread_quality_contribution(
            baseline_records
        ),
    }
    return candidates, baseline_with_predictions, diagnostics, predictions


def pooled(records_by_fold: list[dict], evaluated_days: int) -> dict:
    base = [record for value in records_by_fold for record in value["trade_ledger"]]
    stress = [
        record
        for value in records_by_fold
        for record in value["cost_stress_trade_ledger"]
    ]
    return {
        "metrics": gen19.metrics(base, evaluated_days),
        "cost_stress": gen19.metrics(stress, evaluated_days),
        "trade_ledger": base,
        "cost_stress_trade_ledger": stress,
    }


def phase5_summary(
    policy: str,
    fold_results: dict,
    pooled_value: dict,
    baseline_pooled: dict,
) -> dict:
    folds = [fold_results[name][policy] for name, *_ in SELECTION_FOLDS]
    metric = pooled_value["metrics"]
    stress = pooled_value["cost_stress"]
    retention = metric["trades"] / max(
        baseline_pooled["metrics"]["trades"], 1
    )
    stable_folds = all(
        value["metrics"]["trades"] >= 5
        and value["metrics"]["profit_factor"] is not None
        and value["metrics"]["profit_factor"] > 1.0
        and value["metrics"]["mean_r"] > 0.0
        and value["metrics"]["break_even_adjusted_win_rate_edge"] is not None
        and value["metrics"]["break_even_adjusted_win_rate_edge"] > 0.0
        for value in folds
    )
    success = bool(
        metric["profit_factor"] is not None
        and metric["profit_factor"] >= 1.05
        and metric["mean_r"] > 0.0
        and metric["pnl"] > 0.0
        and metric["break_even_adjusted_win_rate_edge"] is not None
        and metric["break_even_adjusted_win_rate_edge"] > 0.0
        and metric["realized_positive_trade_win_rate"] >= 0.58
        and retention >= 0.50
        and stress["profit_factor"] is not None
        and stress["profit_factor"] > 1.0
        and stress["mean_r"] > 0.0
        and stable_folds
    )
    return {
        "policy": policy,
        "phase5_success": success,
        "stable_positive_all_folds": stable_folds,
        "frequency_retention": retention,
        "metrics": metric,
        "cost_stress": stress,
        "comparison_to_gen17": filter_comparison(
            pooled_value["trade_ledger"], baseline_pooled["trade_ledger"]
        ),
        "spread_quality_contribution": spread_quality_contribution(
            pooled_value["trade_ledger"]
        ),
    }


def aggregate_feature_importance(diagnostics: dict) -> dict:
    output = {}
    for target in ("p_net", "e_net"):
        total: dict[str, float] = {}
        appearances: dict[str, int] = {}
        for fold_name, *_ in SELECTION_FOLDS:
            gains = diagnostics[fold_name]["model"]["feature_importance"][target]
            denominator = sum(float(value) for value in gains.values()) or 1.0
            for feature, gain in gains.items():
                total[feature] = total.get(feature, 0.0) + float(gain) / denominator
                appearances[feature] = appearances.get(feature, 0) + 1
        ranked = sorted(total.items(), key=lambda item: item[1], reverse=True)
        output[target] = [
            {
                "feature": feature,
                "mean_normalized_gain": gain / len(SELECTION_FOLDS),
                "fold_appearances": appearances[feature],
                "cost_relative_feature": feature in COST_FEATURES,
            }
            for feature, gain in ranked[:20]
        ]
    return output


def stability_summary(diagnostics: dict) -> dict:
    output = {}
    for score in ("p_tp_first", "p_net", "e_net"):
        correlations = [
            diagnostics[fold]["fixed_cohort_ranking"][score][
                "spearman_vs_net_r"
            ]
            for fold, *_ in SELECTION_FOLDS
        ]
        valid = np.asarray(
            [value for value in correlations if value is not None],
            dtype=np.float64,
        )
        output[score] = {
            "fold_spearman_vs_net_r": correlations,
            "mean_spearman": float(valid.mean()) if len(valid) else None,
            "std_spearman": float(valid.std()) if len(valid) else None,
        }
        if score in ("p_tp_first", "p_net"):
            brier = [
                diagnostics[fold]["fixed_cohort_calibration"][score]["brier"]
                for fold, *_ in SELECTION_FOLDS
            ]
            ece = [
                diagnostics[fold]["fixed_cohort_calibration"][score]["ece"]
                for fold, *_ in SELECTION_FOLDS
            ]
            output[score]["fold_brier"] = brier
            output[score]["mean_brier"] = float(np.mean(brier))
            output[score]["fold_ece"] = ece
            output[score]["mean_ece"] = float(np.mean(ece))
    return output


def expansion_entries(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    policy: str,
    threshold: dict,
) -> pd.DataFrame:
    if predictions.empty:
        return gen16.empty_entries()
    selected = predictions[
        predictions.apply(
            lambda row: policy_keep(
                {"p_net": float(row["p_net"]), "e_net": float(row["e_net"])},
                policy,
                threshold,
            ),
            axis=1,
        )
    ].copy()
    if selected.empty:
        return gen16.empty_entries()
    indices = selected["index"].to_numpy(dtype=np.int64)
    scores = pd.DataFrame(
        {
            "index": indices,
            "direction": np.full(len(indices), DIRECTION, dtype=np.int8),
            "expert": np.full(len(indices), EXPERT, dtype=object),
            "family": np.full(len(indices), "trend_continuation", dtype=object),
            "context": gen16.context_codes(frame)[indices],
            "p_win": selected["p_net"].to_numpy(dtype=np.float32),
            "expected_r": selected["e_net"].to_numpy(dtype=np.float32),
        }
    )
    economic = gen19.add_economic_state(scores, frame)
    priority = (
        economic["e_net"].to_numpy(dtype=np.float64)
        if "e_net" in economic
        else selected["e_net"].to_numpy(dtype=np.float64)
    )
    entries = gen19.entry_frame(economic, priority)
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
        entries[name] = economic[name].to_numpy()
    return entries


def run_phase6(
    policy: str,
    runtime: dict,
    fold_results: dict,
) -> dict:
    folds = {}
    pooled_records, pooled_stress = [], []
    days = 0
    for fold_name, *_ in SELECTION_FOLDS:
        frame = runtime[fold_name]["frame"]
        models = runtime[fold_name]["models"]
        predictions = runtime[fold_name]["predictions"]
        entries = expansion_entries(
            frame, predictions, policy, models["thresholds"][policy]
        )
        records = gen19.execute_entries(
            frame, entries, f"{fold_name}_gen20_expansion"
        )
        stress = gen19.execute_entries(
            frame,
            entries,
            f"{fold_name}_gen20_expansion_stress",
            gen19.STRESS_EXTRA_COST_POINTS,
        )
        evaluated_days = int(frame["TIME_DT"].dt.date.nunique())
        days += evaluated_days
        pooled_records.extend(records)
        pooled_stress.extend(stress)
        folds[fold_name] = {
            "metrics": gen19.metrics(records, evaluated_days),
            "cost_stress": gen19.metrics(stress, evaluated_days),
            "comparison_to_fixed_selector": gen19.compare_records(
                records, fold_results[fold_name][policy]["trade_ledger"]
            ),
            "trade_ledger": records,
            "cost_stress_trade_ledger": stress,
        }
    pooled_metrics = gen19.metrics(pooled_records, days)
    pooled_stress_metrics = gen19.metrics(pooled_stress, days)
    success = bool(
        pooled_metrics["profit_factor"] is not None
        and pooled_metrics["profit_factor"] >= 1.05
        and pooled_metrics["mean_r"] > 0.0
        and pooled_metrics["break_even_adjusted_win_rate_edge"] is not None
        and pooled_metrics["break_even_adjusted_win_rate_edge"] > 0.0
        and pooled_stress_metrics["profit_factor"] is not None
        and pooled_stress_metrics["profit_factor"] > 1.0
        and pooled_stress_metrics["mean_r"] > 0.0
    )
    return {
        "executed": True,
        "policy": policy,
        "success": success,
        "universe": "all eligible short_trend_continuation rising-edge events",
        "folds": folds,
        "pooled": pooled_metrics,
        "pooled_cost_stress": pooled_stress_metrics,
    }


def compact_metrics(value: dict) -> dict:
    keys = (
        "trades",
        "evaluated_days",
        "trades_per_day",
        "wins",
        "losses",
        "timeouts",
        "tp_first_rate",
        "realized_positive_trade_win_rate",
        "average_winning_r",
        "average_losing_r",
        "payoff_ratio",
        "realized_break_even_win_rate",
        "break_even_adjusted_win_rate_edge",
        "profit_factor",
        "mean_r",
        "pnl",
        "max_drawdown_pct",
    )
    return {key: value.get(key) for key in keys}


def markdown(report: dict) -> str:
    parent = report["baseline"]["selection_pooled"]["metrics"]
    lines = [
        "# Generation 20 - Direct Net-Edge Learning",
        "",
        "Status: `research_only`. Phase 1-5 use only the fixed Gen17 executable cohort.",
        "",
        "## Fixed-cohort selection summary",
        "",
        "| Policy | Trades | Retention | Trades/day | Realized WR | TP-first | PF | Mean-R | PnL | Max DD | Break-even edge | Stress PF | Phase 5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        f"| Gen17 observed-cost parent | {parent['trades']} | 100.00% | {parent['trades_per_day']:.3f} | {parent['realized_positive_trade_win_rate']:.2%} | {parent['tp_first_rate']:.2%} | {parent['profit_factor'] or 0.0:.3f} | {parent['mean_r']:.4f} | {parent['pnl']:.2f} | {parent['max_drawdown_pct']:.2%} | {parent['break_even_adjusted_win_rate_edge'] or 0.0:.2%} | {report['baseline']['selection_pooled']['cost_stress']['profit_factor'] or 0.0:.3f} | parent |",
    ]
    for value in report["phase5_summaries"]:
        metric = value["metrics"]
        lines.append(
            f"| {value['policy']} | {metric['trades']} | {value['frequency_retention']:.2%} | "
            f"{metric['trades_per_day']:.3f} | {metric['realized_positive_trade_win_rate']:.2%} | "
            f"{metric['tp_first_rate']:.2%} | {metric['profit_factor'] or 0.0:.3f} | "
            f"{metric['mean_r']:.4f} | {metric['pnl']:.2f} | {metric['max_drawdown_pct']:.2%} | "
            f"{metric['break_even_adjusted_win_rate_edge'] or 0.0:.2%} | "
            f"{value['cost_stress']['profit_factor'] or 0.0:.3f} | {'PASS' if value['phase5_success'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Chronological folds",
            "",
            "| Policy | Fold | Trades | Trades/day | WR | PF | Mean-R | Edge | Losers removed | Winners removed |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for fold_name, *_ in SELECTION_FOLDS:
        for policy in POLICIES:
            value = report["fold_results"][fold_name][policy]
            metric = value["metrics"]
            comparison = value["comparison_to_gen17"]
            lines.append(
                f"| {policy} | {fold_name} | {metric['trades']} | {metric['trades_per_day']:.3f} | "
                f"{metric['realized_positive_trade_win_rate']:.2%} | {metric['profit_factor'] or 0.0:.3f} | "
                f"{metric['mean_r']:.4f} | {metric['break_even_adjusted_win_rate_edge'] or 0.0:.2%} | "
                f"{comparison['losers_removed']} | {comparison['winners_accidentally_removed']} |"
            )
    lines.extend(
        [
            "",
            f"Phase 6 executed: {report['phase6']['executed']}.",
            "",
            "All 2025 and recent results are development diagnostics, not untouched OOS evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def self_check() -> None:
    values = np.arange(2_000, dtype=np.float64)
    percentile, zscore = past_block_relative(values)
    assert np.all(percentile[:BLOCK_ROWS] == 0.5)
    assert np.all(zscore[:BLOCK_ROWS] == 0.0)
    assert set(POLICIES) == {
        "p_net_ge_050",
        "e_net_positive",
        "joint_positive",
        "e_net_top75_past",
    }
    assert "SPREAD_QUALITY" not in COST_FEATURES
    print("generation20_direct_net_edge_self_check_ok")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0
    report19 = json.loads(GEN19_REPORT.read_text(encoding="utf-8"))
    report17 = json.loads(
        (ROOT / "gold_generation17_cross_regime.json").read_text(encoding="utf-8")
    )
    gemini_hash = file_hash(ROOT / "gemini.py")

    history, base_features = prepare_barrier_data()
    history = add_targets(history)
    history, regime_features = gen17.add_regime_features(history, base_features)
    history = add_cost_targets_and_features(history)
    robust_base = [
        feature
        for feature in base_features
        if feature not in gen17.ABSOLUTE_SCALE_FEATURES
    ]
    model_features = [*robust_base, *regime_features, *COST_FEATURES]

    fold_results: dict[str, dict] = {}
    baselines: dict[str, dict] = {}
    diagnostics: dict[str, dict] = {}
    runtime: dict[str, dict] = {}
    for fold_name, fold_start, fold_end in SELECTION_FOLDS:
        train = training_frame(history, fold_start)
        evaluation = history[
            (history["TIME_DT"] >= fold_start)
            & (history["TIME_DT"] < fold_end)
        ].reset_index(drop=True)
        results, baseline, fold_diagnostics, predictions = process_fixed_period(
            train,
            evaluation,
            report19["baseline"]["folds"][fold_name],
            fold_name,
            base_features,
            model_features,
        )
        fold_results[fold_name] = results
        baselines[fold_name] = baseline
        diagnostics[fold_name] = fold_diagnostics
        runtime[fold_name] = {
            "frame": evaluation,
            "predictions": predictions,
            "models": {"thresholds": fold_diagnostics["model"]["thresholds"]},
        }
        print(f"Fold {fold_name} complete", flush=True)

    selection_days = sum(
        baselines[name]["metrics"]["evaluated_days"]
        for name, *_ in SELECTION_FOLDS
    )
    baseline_pooled = pooled(
        [baselines[name] for name, *_ in SELECTION_FOLDS], selection_days
    )
    pooled_candidates = {
        policy: pooled(
            [fold_results[name][policy] for name, *_ in SELECTION_FOLDS],
            selection_days,
        )
        for policy in POLICIES
    }
    summaries = [
        phase5_summary(
            policy,
            fold_results,
            pooled_candidates[policy],
            baseline_pooled,
        )
        for policy in POLICIES
    ]
    successful = [value for value in summaries if value["phase5_success"]]
    successful.sort(
        key=lambda value: (
            value["metrics"]["realized_positive_trade_win_rate"],
            value["metrics"]["trades"],
            value["metrics"]["profit_factor"] or 0.0,
        ),
        reverse=True,
    )
    diagnostic = sorted(
        summaries,
        key=lambda value: (
            value["metrics"]["profit_factor"] or 0.0,
            value["metrics"]["mean_r"],
            value["metrics"]["trades"],
        ),
        reverse=True,
    )[0]
    selected = successful[0] if successful else diagnostic
    selected_policy = selected["policy"]
    phase6 = (
        run_phase6(selected_policy, runtime, fold_results)
        if successful
        else {
            "executed": False,
            "reason": "Phase 5 found no stable positive-expectancy fixed-cohort selector",
        }
    )

    holdout = history[
        history["TIME_DT"] >= HISTORICAL_HOLDOUT_START
    ].reset_index(drop=True)
    holdout_results, holdout_baseline, holdout_diagnostics, _ = (
        process_fixed_period(
            training_frame(history, HISTORICAL_HOLDOUT_START),
            holdout,
            report19["development_diagnostics"][
                "2025_2026_05_baseline"
            ],
            "2025_2026_05_development",
            base_features,
            model_features,
        )
    )

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
    recent_all = add_cost_targets_and_features(recent_all)
    recent_end = pd.Timestamp(
        report17["data"]["recent_development_end"]
    ) + pd.Timedelta(minutes=1)
    recent = recent_all[
        (recent_all["TIME_DT"] >= RECENT_START.replace(tzinfo=None))
        & (recent_all["TIME_DT"] < recent_end)
    ].reset_index(drop=True)
    final_cutoff = history["TIME_DT"].iat[-1] + pd.Timedelta(minutes=1)
    recent_results, recent_baseline, recent_diagnostics, _ = process_fixed_period(
        training_frame(history, final_cutoff),
        recent,
        report19["development_diagnostics"]["2026_recent_baseline"],
        "2026_recent_development",
        base_features,
        model_features,
    )

    stability = stability_summary(diagnostics)
    importance = aggregate_feature_importance(diagnostics)
    pnet_more_stable = bool(
        stability["p_net"]["mean_brier"]
        < stability["p_tp_first"]["mean_brier"]
        and stability["p_net"]["mean_ece"]
        < stability["p_tp_first"]["mean_ece"]
    )
    enet_more_predictive = bool(
        stability["e_net"]["mean_spearman"]
        > stability["p_tp_first"]["mean_spearman"]
        and stability["e_net"]["mean_spearman"]
        > stability["p_net"]["mean_spearman"]
    )
    summaries.sort(
        key=lambda value: (
            value["phase5_success"],
            value["metrics"]["profit_factor"] or 0.0,
            value["metrics"]["mean_r"],
        ),
        reverse=True,
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generation": "20_direct_net_edge_learning",
        "status": "research_only",
        "development_history_policy": report19["development_history_policy"],
        "architecture": {
            "fixed_cohort": "Gen17 observed-cost executable trades",
            "selection_fold_fixed_cohort_trades": baseline_pooled["metrics"]["trades"],
            "new_signals_phase1_to_5": 0,
            "new_family": False,
            "tp_sl_change": False,
            "execution_change": False,
            "targets": ["NET_POSITIVE", "NET_REWARD"],
            "models": ["XGBClassifier plus past-only isotonic calibration", "XGBRegressor pseudohuber"],
            "policies": list(POLICIES),
            "model_features": model_features,
            "cost_features": list(COST_FEATURES),
            "spread_quality_is_model_feature": False,
            "spread_quality_sample_weighting_tested": False,
            "relative_feature_method": {
                "block_rows": BLOCK_ROWS,
                "reference_rows": REFERENCE_ROWS,
                "minimum_reference_rows": MIN_REFERENCE_ROWS,
                "current_block_uses": "strictly previous rows only",
            },
            "cost_method": report19["architecture"]["cost_rule"],
            "extra_cost_points": gen19.BASE_EXTRA_COST_POINTS,
            "stress_extra_cost_points": gen19.STRESS_EXTRA_COST_POINTS,
            "horizon": gen16.HORIZON,
            "tp_atr": 1.3,
            "sl_atr": 1.6,
            "same_bar_tp_sl": "stop_first",
            "position_limit": 1,
        },
        "phase5_gate": {
            "pooled_pf_min": 1.05,
            "pooled_mean_r_positive": True,
            "pooled_break_even_edge_positive": True,
            "pooled_realized_win_rate_min": 0.58,
            "frequency_retention_min": 0.50,
            "cost_stress_pf_min_exclusive": 1.0,
            "each_fold_pf_min_exclusive": 1.0,
            "each_fold_mean_r_positive": True,
        },
        "baseline": {
            "folds": baselines,
            "selection_pooled": baseline_pooled,
        },
        "fold_results": fold_results,
        "phase5_summaries": summaries,
        "phase5_success": bool(successful),
        "frozen_candidate_id": selected_policy if successful else None,
        "diagnostic_fallback_id": None if successful else selected_policy,
        "phase6": phase6,
        "model_diagnostics": diagnostics,
        "score_stability": stability,
        "feature_importance": importance,
        "development_diagnostics": {
            "2025_2026_05": {
                "baseline": holdout_baseline,
                "candidates": holdout_results,
                "model_diagnostics": holdout_diagnostics,
            },
            "2026_recent": {
                "baseline": recent_baseline,
                "candidates": recent_results,
                "model_diagnostics": recent_diagnostics,
            },
        },
        "answers": {
            "p_net_more_stable_than_p_tp_first": pnet_more_stable,
            "e_net_more_predictive_than_probability_scores": enet_more_predictive,
            "gen17_pf_raised_above_1_05_with_acceptable_retention": bool(successful),
            "phase6_frequency_expansion_executed": phase6["executed"],
        },
        "selection_inventory": {
            "direct_targets": 2,
            "direct_models": 2,
            "fixed_policies": list(POLICIES),
            "sample_weight_variants": 1,
            "sample_weight_variant": "uniform",
            "frequency_expansion_gate": "Phase 5 success only",
        },
        "promotion_pass": False,
        "gemini_modified": False,
        "gemini_sha256_before_and_after": gemini_hash,
        "final_untouched_test_validity": "FAIL_no_untouched_future_interval",
    }
    config = {
        "generation": report["generation"],
        "status": "research_only",
        "phase5_success": report["phase5_success"],
        "frozen_candidate_id": report["frozen_candidate_id"],
        "diagnostic_fallback_id": report["diagnostic_fallback_id"],
        "phase6_executed": phase6["executed"],
        "promotion_pass": False,
    }
    CONFIG_JSON.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REPORT_MD.write_text(markdown(report), encoding="utf-8")
    print(
        f"Wrote {REPORT_JSON.name}, {REPORT_MD.name}, and {CONFIG_JSON.name}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
