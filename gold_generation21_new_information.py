from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import gold_generation17_cross_regime as gen17
import gold_generation19_cost_aware as gen19
import gold_generation20_direct_net_edge as gen20
from barrier_final_train import prepare_barrier_data
from gold_generation11_execution_aligned import add_targets
from gold_regime_experts_iterative import training_frame
from gold_regime_experts_walk_forward import SELECTION_FOLDS


ROOT = Path(__file__).resolve().parent
GEN20_REPORT = ROOT / "gold_generation20_direct_net_edge.json"
GEN17_REPORT = ROOT / "gold_generation17_cross_regime.json"
GOLD_M1 = ROOT / "GOLD#_M1_201401020000_202605082357.csv"
SILVER_M1 = (
    ROOT / "數據集" / "SILVER#_M1_201401020000_202605281623.csv"
)
REPORT_JSON = ROOT / "gold_generation21_new_information.json"
REPORT_MD = ROOT / "gold_generation21_new_information.md"
CONFIG_JSON = ROOT / "gold_generation21_candidate.json"
FORWARD_JSON = ROOT / "gold_generation21_forward_protocol.json"
FORWARD_MD = ROOT / "gold_generation21_forward_protocol.md"

VERSIONS = (
    "A_technical_control",
    "B_microstructure",
    "C_cross_market_xag",
    "E_microstructure_plus_xag",
)
MICRO_FEATURES = (
    "MICRO_TICKVOL_LOG",
    "MICRO_TICKVOL_CHANGE",
    "MICRO_TICKVOL_ACCEL",
    "MICRO_TICKVOL_PCTL",
    "MICRO_TICKVOL_Z",
    "MICRO_REALVOL_LOG",
    "MICRO_REALVOL_PCTL",
    "MICRO_REALVOL_Z",
    "MICRO_SPREAD_CHANGE",
    "MICRO_SPREAD_ACCEL",
    "MICRO_SPREAD_CHANGE_ATR",
    "MICRO_RV5",
    "MICRO_RV5_ACCEL",
    "MICRO_RV5_PCTL",
    "MICRO_RV5_Z",
)
CROSS_FEATURES = (
    "XAG_RET_5",
    "XAG_RET_15",
    "XAG_RET_60",
    "XAG_CLOSE_Z_1D",
    "XAG_RV15",
    "XAG_ACTIVITY_PCTL",
    "GOLD_XAG_DIV_5",
    "GOLD_XAG_DIV_15",
    "GOLD_XAG_DIV_60",
    "GOLD_XAG_CORR_120",
    "XAG_LEAD_GOLD_CORR_120",
)

INFORMATION_GATE = {
    "primary_score": "e_net_spearman_vs_net_r",
    "mean_spearman_improvement_min": 0.05,
    "positive_folds_min": 2,
    "improved_folds_min": 2,
    "worst_fold_min": -0.05,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generation 21 new entry-time information ablation"
    )
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3:
        return None
    value = pd.Series(left[finite]).corr(
        pd.Series(right[finite]), method="spearman"
    )
    return None if pd.isna(value) else float(value)


def add_microstructure_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    result = frame.copy()
    tick_volume = pd.to_numeric(result["TICKVOL"], errors="coerce").clip(lower=0)
    completed_tick_log = np.log1p(tick_volume).shift(1)
    tick_change = completed_tick_log.diff()
    tick_accel = tick_change.diff()
    tick_pctl, tick_z = gen20.past_block_relative(
        completed_tick_log.to_numpy(dtype=np.float64)
    )

    raw_spread = pd.to_numeric(result["SPREAD"], errors="coerce")
    observed = raw_spread.gt(0.0) & raw_spread.notna()
    spread_change = raw_spread.diff().where(observed & observed.shift(1, fill_value=False))
    spread_accel = spread_change.diff().where(
        observed
        & observed.shift(1, fill_value=False)
        & observed.shift(2, fill_value=False)
    )

    log_close = np.log(result["CLOSE"].clip(lower=1e-9))
    rv5 = log_close.diff().rolling(5, min_periods=5).std().shift(1)
    rv5_accel = rv5.diff(5)
    rv_pctl, rv_z = gen20.past_block_relative(
        rv5.to_numpy(dtype=np.float64)
    )

    result["MICRO_TICKVOL_LOG"] = completed_tick_log.astype(np.float32)
    result["MICRO_TICKVOL_CHANGE"] = tick_change.astype(np.float32)
    result["MICRO_TICKVOL_ACCEL"] = tick_accel.astype(np.float32)
    result["MICRO_TICKVOL_PCTL"] = tick_pctl
    result["MICRO_TICKVOL_Z"] = tick_z

    real_volume = pd.to_numeric(result.get("VOL", 0.0), errors="coerce")
    real_volume_log = np.log1p(real_volume.where(real_volume > 0.0)).shift(1)
    real_volume_pctl, real_volume_z = gen20.past_block_relative(
        real_volume_log.to_numpy(dtype=np.float64)
    )
    real_volume_available = real_volume_log.notna().to_numpy()
    real_volume_pctl[~real_volume_available] = np.nan
    real_volume_z[~real_volume_available] = np.nan
    result["MICRO_REALVOL_LOG"] = real_volume_log.astype(np.float32)
    result["MICRO_REALVOL_PCTL"] = real_volume_pctl
    result["MICRO_REALVOL_Z"] = real_volume_z
    result["MICRO_SPREAD_CHANGE"] = spread_change.astype(np.float32)
    result["MICRO_SPREAD_ACCEL"] = spread_accel.astype(np.float32)
    result["MICRO_SPREAD_CHANGE_ATR"] = (
        spread_change * gen19.POINT / result["ATR"].clip(lower=1e-9)
    ).astype(np.float32)
    result["MICRO_RV5"] = rv5.astype(np.float32)
    result["MICRO_RV5_ACCEL"] = rv5_accel.astype(np.float32)
    result["MICRO_RV5_PCTL"] = rv_pctl
    result["MICRO_RV5_Z"] = rv_z

    real_volume_positive = int(pd.Series(real_volume).gt(0.0).sum())
    inventory = {
        "tick_volume": {
            "field_present": "TICKVOL" in result,
            "positive_rows": int(tick_volume.gt(0.0).sum()),
            "features_added": [
                feature for feature in MICRO_FEATURES if "TICKVOL" in feature
            ],
            "semantics": "M1 broker tick-count/activity proxy, shifted one completed bar",
        },
        "real_volume": {
            "field_present": "VOL" in result,
            "positive_rows": real_volume_positive,
            "features_added": [
                "MICRO_REALVOL_LOG",
                "MICRO_REALVOL_PCTL",
                "MICRO_REALVOL_Z",
            ],
            "missing_handling": (
                "Only positive observed values are used; zero/unavailable rows remain NaN."
            ),
        },
        "spread": {
            "field_present": "SPREAD" in result,
            "observed_rows": int(observed.sum()),
            "unavailable_rows": int((~observed).sum()),
            "features_added": [
                feature for feature in MICRO_FEATURES if "SPREAD" in feature
            ],
            "missing_handling": "NaN for spread dynamics; no free-spread or synthetic dynamics.",
            "already_in_control": [
                "COST_SPREAD_POINTS",
                "COST_SPREAD_ATR",
                "COST_SPREAD_SL",
                "COST_SPREAD_PCTL",
                "COST_SPREAD_ATR_PCTL",
                "COST_SPREAD_ATR_Z",
            ],
        },
        "short_term_realized_volatility": {
            "features_added": [
                "MICRO_RV5",
                "MICRO_RV5_ACCEL",
                "MICRO_RV5_PCTL",
                "MICRO_RV5_Z",
            ],
            "availability": "derived only from completed GOLD M1 closes",
        },
        "tick_history": {
            "available": False,
            "features_added": [],
            "reason": "No reproducible historical tick file in the repository.",
        },
        "intrabar_tick_path": {
            "available": False,
            "features_added": [],
            "reason": "M1 OHLC cannot reconstruct tick order or directional pressure.",
        },
    }
    return result, inventory


def load_silver_features() -> tuple[pd.DataFrame, dict]:
    if not SILVER_M1.exists():
        raise FileNotFoundError(SILVER_M1)
    silver = pd.read_csv(
        SILVER_M1,
        sep="\t",
        usecols=[
            "<DATE>",
            "<TIME>",
            "<CLOSE>",
            "<TICKVOL>",
            "<VOL>",
            "<SPREAD>",
        ],
    ).rename(
        columns={
            "<DATE>": "DATE",
            "<TIME>": "TIME",
            "<CLOSE>": "CLOSE",
            "<TICKVOL>": "TICKVOL",
            "<VOL>": "VOL",
            "<SPREAD>": "SPREAD",
        }
    )
    silver["TIME_DT"] = pd.to_datetime(
        silver["DATE"] + " " + silver["TIME"],
        format="%Y.%m.%d %H:%M:%S",
        errors="raise",
    )
    silver = silver.sort_values("TIME_DT").drop_duplicates("TIME_DT")
    close = pd.to_numeric(silver["CLOSE"], errors="coerce").clip(lower=1e-9)
    log_close = np.log(close)
    log_tick = np.log1p(
        pd.to_numeric(silver["TICKVOL"], errors="coerce").clip(lower=0)
    )
    mean_1d = log_close.rolling(1_440, min_periods=240).mean()
    std_1d = log_close.rolling(1_440, min_periods=240).std()
    activity_pctl, _ = gen20.past_block_relative(
        log_tick.shift(1).to_numpy(dtype=np.float64)
    )
    features = pd.DataFrame(
        {
            "TIME_DT": silver["TIME_DT"],
            "XAG_RET_1_INTERNAL": log_close.diff().shift(1),
            "XAG_RET_5": log_close.diff(5).shift(1),
            "XAG_RET_15": log_close.diff(15).shift(1),
            "XAG_RET_60": log_close.diff(60).shift(1),
            "XAG_CLOSE_Z_1D": ((log_close - mean_1d) / std_1d).shift(1),
            "XAG_RV15": log_close.diff().rolling(15, min_periods=15).std().shift(1),
            "XAG_ACTIVITY_PCTL": activity_pctl,
        }
    )
    numeric = [column for column in features if column != "TIME_DT"]
    features[numeric] = features[numeric].astype(np.float32)
    inventory = {
        "symbol": "SILVER#",
        "economic_equivalent": "broker XAGUSD silver CFD",
        "path": SILVER_M1.relative_to(ROOT).as_posix(),
        "rows": len(silver),
        "start": silver["TIME_DT"].iloc[0].isoformat(),
        "end": silver["TIME_DT"].iloc[-1].isoformat(),
        "sha256": file_hash(SILVER_M1),
        "real_volume_positive_rows": int(
            pd.to_numeric(silver["VOL"], errors="coerce").gt(0.0).sum()
        ),
        "spread_observed_rows": int(
            pd.to_numeric(silver["SPREAD"], errors="coerce").gt(0.0).sum()
        ),
    }
    return features, inventory


def add_cross_market_features(
    frame: pd.DataFrame, silver: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    aligned = pd.merge_asof(
        frame[["TIME_DT"]].sort_values("TIME_DT"),
        silver.sort_values("TIME_DT"),
        on="TIME_DT",
        direction="backward",
        tolerance=pd.Timedelta(minutes=5),
    )
    result = frame.copy()
    for feature in (
        "XAG_RET_5",
        "XAG_RET_15",
        "XAG_RET_60",
        "XAG_CLOSE_Z_1D",
        "XAG_RV15",
        "XAG_ACTIVITY_PCTL",
    ):
        result[feature] = aligned[feature].to_numpy(dtype=np.float32)
    gold_log = np.log(result["CLOSE"].clip(lower=1e-9))
    gold_ret_1 = gold_log.diff().shift(1)
    xag_ret_1 = aligned["XAG_RET_1_INTERNAL"].astype(np.float64)
    for minutes in (5, 15, 60):
        result[f"GOLD_XAG_DIV_{minutes}"] = (
            gold_log.diff(minutes).shift(1)
            - aligned[f"XAG_RET_{minutes}"].astype(np.float64)
        ).astype(np.float32)
    result["GOLD_XAG_CORR_120"] = (
        gold_ret_1.rolling(120, min_periods=60).corr(xag_ret_1).astype(np.float32)
    )
    result["XAG_LEAD_GOLD_CORR_120"] = (
        gold_ret_1.rolling(120, min_periods=60)
        .corr(xag_ret_1.shift(1))
        .astype(np.float32)
    )
    complete = result[list(CROSS_FEATURES)].notna().all(axis=1)
    return result, {
        "aligned_rows": int(complete.sum()),
        "unavailable_rows": int((~complete).sum()),
        "alignment": "backward as-of, maximum staleness 5 minutes",
        "feature_values": "computed from the last completed SILVER bar",
    }


def feature_sets(control: list[str]) -> dict[str, list[str]]:
    return {
        "A_technical_control": list(control),
        "B_microstructure": [*control, *MICRO_FEATURES],
        "C_cross_market_xag": [*control, *CROSS_FEATURES],
        "E_microstructure_plus_xag": [
            *control,
            *MICRO_FEATURES,
            *CROSS_FEATURES,
        ],
    }


def complete_feature_counts(
    frame: pd.DataFrame,
    indices: np.ndarray,
    added_features: list[str],
) -> dict:
    if not added_features:
        return {
            "events": len(indices),
            "any_new_information": len(indices),
            "complete": len(indices),
            "coverage": 1.0,
            "per_feature_available": {},
        }
    availability = frame.iloc[indices][added_features].notna()
    complete = availability.all(axis=1)
    return {
        "events": len(indices),
        "any_new_information": int(availability.any(axis=1).sum()),
        "complete": int(complete.sum()),
        "coverage": float(complete.mean()) if len(complete) else 0.0,
        "per_feature_available": {
            feature: int(availability[feature].sum()) for feature in added_features
        },
    }


def score_fixed_cohort(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    baseline: dict,
    base_features: list[str],
    model_features: list[str],
    added_features: list[str],
) -> tuple[dict, dict, list[dict]]:
    models, diagnostics = gen20.train_direct_models(
        train, base_features, model_features
    )
    eligible = gen20.event_indices(evaluation, base_features)
    predictions = gen20.predict_direct(
        models, evaluation, eligible, model_features
    )
    records = gen20.attach_predictions(
        baseline["trade_ledger"], predictions, evaluation
    )
    reward = np.asarray([record["reward"] for record in records], dtype=np.float64)
    net_positive = (reward > 0.0).astype(np.int8)
    tp_first = np.asarray(
        [int(record["outcome"]) == 1 for record in records], dtype=np.int8
    )
    p_net = np.asarray([record["p_net"] for record in records], dtype=np.float64)
    e_net = np.asarray([record["e_net"] for record in records], dtype=np.float64)
    p_tp = np.asarray(
        [record["p_tp_first"] for record in records], dtype=np.float64
    )
    fixed_indices = np.asarray([record["index"] for record in records], dtype=np.int64)
    train_events = gen20.event_indices(train, base_features)
    result = {
        "usable_observations": {
            "train_events": complete_feature_counts(
                train, train_events, added_features
            ),
            "eligible_evaluation_events": complete_feature_counts(
                evaluation, eligible, added_features
            ),
            "fixed_executable_cohort": complete_feature_counts(
                evaluation, fixed_indices, added_features
            ),
        },
        "ranking": {
            "existing_p_tp_first_vs_tp_first": spearman(p_tp, tp_first),
            "existing_p_tp_first_vs_net_r": spearman(p_tp, reward),
            "p_net_vs_positive_net_r": spearman(p_net, net_positive),
            "p_net_vs_net_r": spearman(p_net, reward),
            "e_net_vs_net_r": spearman(e_net, reward),
        },
        "calibration": {
            "p_net": gen20.calibration_stats(p_net, net_positive),
        },
        "executable_metrics": baseline["metrics"],
        "cost_stress": baseline["cost_stress"],
        "fixed_cohort_prediction_ledger": records,
    }
    return result, diagnostics, records


def summarize_versions(fold_results: dict) -> dict:
    output = {}
    for version in VERSIONS:
        folds = [fold_results[fold][version] for fold, *_ in SELECTION_FOLDS]
        p_net_correlations = [
            value["ranking"]["p_net_vs_positive_net_r"] for value in folds
        ]
        valid_p_net_correlations = [
            value for value in p_net_correlations if value is not None
        ]
        output[version] = {
            "fold_e_net_spearman_vs_net_r": [
                value["ranking"]["e_net_vs_net_r"] for value in folds
            ],
            "mean_e_net_spearman_vs_net_r": float(
                np.mean(
                    [value["ranking"]["e_net_vs_net_r"] for value in folds]
                )
            ),
            "fold_p_net_spearman_vs_positive_net_r": p_net_correlations,
            "mean_p_net_spearman_vs_positive_net_r": (
                float(np.mean(valid_p_net_correlations))
                if valid_p_net_correlations
                else None
            ),
            "mean_p_net_brier": float(
                np.mean([value["calibration"]["p_net"]["brier"] for value in folds])
            ),
            "mean_p_net_ece": float(
                np.mean([value["calibration"]["p_net"]["ece"] for value in folds])
            ),
        }
    control = output["A_technical_control"]
    control_folds = control["fold_e_net_spearman_vs_net_r"]
    for version, value in output.items():
        correlations = value["fold_e_net_spearman_vs_net_r"]
        improvements = [
            current - baseline
            for current, baseline in zip(correlations, control_folds)
        ]
        value["mean_e_net_information_gain"] = (
            value["mean_e_net_spearman_vs_net_r"]
            - control["mean_e_net_spearman_vs_net_r"]
        )
        value["fold_e_net_information_gain"] = improvements
        value["positive_folds"] = sum(current > 0.0 for current in correlations)
        value["improved_folds"] = sum(delta > 0.0 for delta in improvements)
        value["worst_fold"] = min(correlations)
        value["information_gate_pass"] = bool(
            version != "A_technical_control"
            and value["mean_e_net_information_gain"]
            >= INFORMATION_GATE["mean_spearman_improvement_min"]
            and value["positive_folds"]
            >= INFORMATION_GATE["positive_folds_min"]
            and value["improved_folds"]
            >= INFORMATION_GATE["improved_folds_min"]
            and value["worst_fold"] >= INFORMATION_GATE["worst_fold_min"]
        )
    return output


def univariate_separation(
    history: pd.DataFrame, report20: dict, features: tuple[str, ...]
) -> list[dict]:
    rows = []
    for feature in features:
        fold_values = []
        for fold, start, end in SELECTION_FOLDS:
            evaluation = history[
                (history["TIME_DT"] >= start) & (history["TIME_DT"] < end)
            ].reset_index(drop=True)
            ledger = report20["baseline"]["folds"][fold]["trade_ledger"]
            indices = np.asarray([record["index"] for record in ledger], dtype=np.int64)
            values = evaluation.iloc[indices][feature].to_numpy(dtype=np.float64)
            rewards = np.asarray([record["reward"] for record in ledger], dtype=np.float64)
            finite = np.isfinite(values)
            winners = values[finite & (rewards > 0.0)]
            losers = values[finite & (rewards <= 0.0)]
            fold_values.append(
                {
                    "fold": fold,
                    "usable": int(finite.sum()),
                    "spearman_vs_net_r": spearman(values, rewards),
                    "winner_median": float(np.median(winners)) if len(winners) else None,
                    "loser_median": float(np.median(losers)) if len(losers) else None,
                }
            )
        correlations = [
            value["spearman_vs_net_r"]
            for value in fold_values
            if value["spearman_vs_net_r"] is not None
        ]
        rows.append(
            {
                "feature": feature,
                "folds": fold_values,
                "mean_spearman": float(np.mean(correlations)) if correlations else None,
                "consistent_sign_folds": (
                    max(
                        sum(value > 0.0 for value in correlations),
                        sum(value < 0.0 for value in correlations),
                    )
                    if correlations
                    else 0
                ),
            }
        )
    rows.sort(
        key=lambda value: abs(value["mean_spearman"] or 0.0), reverse=True
    )
    return rows


def candidate_from_information(
    version: str,
    records_by_fold: dict,
    report20: dict,
) -> dict:
    folds = {}
    pooled, pooled_stress = [], []
    evaluated_days = 0
    for fold, *_ in SELECTION_FOLDS:
        baseline = report20["baseline"]["folds"][fold]
        records = [
            record
            for record in records_by_fold[fold][version]
            if float(record["e_net"]) > 0.0
        ]
        ids = {record["trade_id"] for record in records}
        stress = [
            record
            for record in baseline["cost_stress_trade_ledger"]
            if record["trade_id"] in ids
        ]
        days = int(baseline["metrics"]["evaluated_days"])
        evaluated_days += days
        pooled.extend(records)
        pooled_stress.extend(stress)
        folds[fold] = {
            "metrics": gen19.metrics(records, days),
            "cost_stress": gen19.metrics(stress, days),
            "comparison_to_control": gen19.compare_records(
                records, baseline["trade_ledger"]
            ),
            "trade_ledger": records,
            "cost_stress_trade_ledger": stress,
        }
    metrics = gen19.metrics(pooled, evaluated_days)
    stress = gen19.metrics(pooled_stress, evaluated_days)
    stable_folds = sum(
        value["metrics"]["profit_factor"] is not None
        and value["metrics"]["profit_factor"] > 1.0
        and value["metrics"]["mean_r"] > 0.0
        for value in folds.values()
    )
    passed = bool(
        metrics["profit_factor"] is not None
        and metrics["profit_factor"] > 1.0
        and metrics["mean_r"] > 0.0
        and metrics["break_even_adjusted_win_rate_edge"] is not None
        and metrics["break_even_adjusted_win_rate_edge"] > 0.0
        and metrics["max_drawdown_pct"] >= -0.25
        and stress["profit_factor"] is not None
        and stress["profit_factor"] > 1.0
        and stress["mean_r"] > 0.0
        and stable_folds >= 2
    )
    return {
        "executed": True,
        "source_version": version,
        "selector": "single predeclared economic rule: E(net_R) > 0",
        "folds": folds,
        "pooled_metrics": metrics,
        "pooled_cost_stress": stress,
        "positive_expectancy_folds": stable_folds,
        "research_gate_pass": passed,
        "trade_ledger": pooled,
        "cost_stress_trade_ledger": pooled_stress,
    }


def forward_protocol(last_inspected: str, generated_at: datetime) -> dict:
    if FORWARD_JSON.exists():
        return json.loads(FORWARD_JSON.read_text(encoding="utf-8"))
    cutoff = (generated_at + timedelta(hours=1)).replace(
        minute=0, second=0, microsecond=0
    )
    protocol = {
        "generation": "21",
        "status": "untouched_pending",
        "last_known_inspected_market_timestamp": last_inspected,
        "untouched_forward_cutoff_utc": cutoff.isoformat(),
        "eligible_data_rule": "market timestamp >= untouched_forward_cutoff_utc",
        "storage_path": "untouched_forward/generation21/data/",
        "prohibited_before_candidate_freeze": [
            "feature selection",
            "model selection",
            "threshold selection",
            "family selection",
            "calibration",
            "strategy-logic debugging from outcomes",
        ],
        "contamination_rule": (
            "If strategy outcomes are inspected for any model decision, reclassify "
            "the entire inspected interval as development and set a new later cutoff."
        ),
    }
    FORWARD_JSON.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    FORWARD_MD.write_text(
        "# Generation 21 untouched-forward protocol\n\n"
        f"Cutoff (UTC): `{protocol['untouched_forward_cutoff_utc']}`\n\n"
        f"Storage: `{protocol['storage_path']}`\n\n"
        "No outcome inspection, feature/model/threshold/family selection, calibration, "
        "or strategy debugging is allowed before a candidate is frozen. Any such "
        "inspection contaminates the interval and requires a new later cutoff.\n",
        encoding="utf-8",
    )
    return protocol


def markdown(report: dict) -> str:
    lines = [
        "# Generation 21 - New Information Source Study",
        "",
        "Status: `research_only`. No TP/SL, cost, signal-universe, execution, or model-hyperparameter change.",
        "",
        "## Information-source availability",
        "",
        "| Family | Status | Evidence / treatment |",
        "|---|---|---|",
        "| Microstructure | tested | GOLD M1 tick volume, sparse positive real volume, observed spread dynamics, and completed-bar short RV; tick path unavailable |",
        "| Cross-market | partial; XAG tested | Complete local SILVER# M1; no aligned historical DXY/yields/VIX/WTI files |",
        "| Economic events | unavailable | No timestamped historical calendar/release file; no dates or surprises fabricated |",
        "",
        "## Ablation summary",
        "",
        "| Version | Mean E(net-R) Spearman | Fold Spearman | Gain vs control | Positive/improved folds | Mean P(net+) Spearman | Brier | ECE | Info gate |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for version in VERSIONS:
        value = report["version_summary"][version]
        folds = ", ".join(
            f"{score:.3f}" for score in value["fold_e_net_spearman_vs_net_r"]
        )
        lines.append(
            f"| {version} | {value['mean_e_net_spearman_vs_net_r']:.3f} | {folds} | "
            f"{value['mean_e_net_information_gain']:+.3f} | "
            f"{value['positive_folds']}/{value['improved_folds']} | "
            f"{value['mean_p_net_spearman_vs_positive_net_r']:.3f} | "
            f"{value['mean_p_net_brier']:.3f} | {value['mean_p_net_ece']:.3f} | "
            f"{'PASS' if value['information_gate_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "| D_event_context | n/a | n/a | n/a | n/a | n/a | n/a | n/a | NOT TESTABLE |",
            "| F_all_justified | same as E | same as E | same as E | same as E | same as E | same as E | same as E | same as E |",
            "",
            "## Usable fixed-cohort observations",
            "",
            "Models retain NaN as missing; `complete` means every added family feature was genuinely observed/derivable.",
            "",
            "| Fold | Version | Model-scored | Any new information | Complete all added features |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for fold, *_ in SELECTION_FOLDS:
        for version in VERSIONS:
            usable = report["fold_results"][fold][version][
                "usable_observations"
            ]["fixed_executable_cohort"]
            lines.append(
                f"| {fold} | {version} | {usable['events']} | "
                f"{usable['any_new_information']} | {usable['complete']} |"
            )
    lines.extend(
        [
            "",
            "## Frozen executable control by fold",
            "",
            "All ablations rank the same fixed executable cohort. Trade metrics therefore remain the control metrics until Phase 6 is allowed.",
            "",
            "| Fold | Trades | Trades/day | Realized WR | PF | Mean-R | PnL | Max DD |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for fold, *_ in SELECTION_FOLDS:
        metric = report["control_baseline"]["folds"][fold]["metrics"]
        lines.append(
            f"| {fold} | {metric['trades']} | {metric['trades_per_day']:.3f} | "
            f"{metric['realized_positive_trade_win_rate']:.2%} | "
            f"{metric['profit_factor'] or 0.0:.3f} | {metric['mean_r']:.4f} | "
            f"{metric['pnl']:.2f} | {metric['max_drawdown_pct']:.2%} |"
        )
    candidate = report["candidate_construction"]
    lines.extend(["", "## Candidate construction", ""])
    if not candidate["executed"]:
        lines.append(
            "Not executed: no new information family passed the predeclared cross-regime discrimination gate."
        )
    else:
        metric = candidate["pooled_metrics"]
        lines.append(
            f"Information gate allowed one `{candidate['selector']}` diagnostic candidate: "
            f"{metric['trades']} trades, WR {metric['realized_positive_trade_win_rate']:.2%}, "
            f"PF {metric['profit_factor'] or 0.0:.3f}, Mean-R {metric['mean_r']:.4f}. "
            f"Research gate: {'PASS' if candidate['research_gate_pass'] else 'FAIL'}."
        )
    lines.extend(
        [
            "",
            "## Untouched forward protocol",
            "",
            f"Cutoff: `{report['untouched_forward_protocol']['untouched_forward_cutoff_utc']}`.",
            "",
            "All prior history remains development data. No production change was made.",
        ]
    )
    return "\n".join(lines) + "\n"


def self_check() -> None:
    values = np.arange(2_000, dtype=np.float64)
    frame = pd.DataFrame(
        {
            "TICKVOL": values + 1.0,
            "SPREAD": np.where(values > 1_000, 30.0, 0.0),
            "VOL": np.zeros(len(values)),
            "CLOSE": values + 1_000.0,
            "ATR": np.full(len(values), 1.0),
        }
    )
    enriched, inventory = add_microstructure_features(frame)
    assert all(feature in enriched for feature in MICRO_FEATURES)
    assert inventory["real_volume"]["positive_rows"] == 0
    assert enriched["MICRO_TICKVOL_LOG"].isna().iat[0]
    assert len(set(MICRO_FEATURES) & set(CROSS_FEATURES)) == 0
    print("generation21_new_information_self_check_ok")


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0
    generated_at = datetime.now(timezone.utc)
    report20 = json.loads(GEN20_REPORT.read_text(encoding="utf-8"))
    report17 = json.loads(GEN17_REPORT.read_text(encoding="utf-8"))
    gemini_hash = file_hash(ROOT / "gemini.py")

    history, base_features = prepare_barrier_data()
    history = add_targets(history)
    history, regime_features = gen17.add_regime_features(history, base_features)
    history = gen20.add_cost_targets_and_features(history)
    robust_base = [
        feature
        for feature in base_features
        if feature not in gen17.ABSOLUTE_SCALE_FEATURES
    ]
    control_features = [
        *robust_base,
        *regime_features,
        *gen20.COST_FEATURES,
    ]
    if control_features != report20["architecture"]["model_features"]:
        raise RuntimeError("Gen20 control feature set changed")

    history, micro_inventory = add_microstructure_features(history)
    silver, silver_inventory = load_silver_features()
    history, cross_alignment = add_cross_market_features(history, silver)
    sets = feature_sets(control_features)

    fold_results: dict[str, dict] = {}
    model_diagnostics: dict[str, dict] = {}
    records_by_fold: dict[str, dict] = {}
    for fold, start, end in SELECTION_FOLDS:
        train = training_frame(history, start)
        evaluation = history[
            (history["TIME_DT"] >= start) & (history["TIME_DT"] < end)
        ].reset_index(drop=True)
        fold_results[fold] = {}
        model_diagnostics[fold] = {}
        records_by_fold[fold] = {}
        for version in VERSIONS:
            if version == "A_technical_control":
                added: list[str] = []
            elif version == "B_microstructure":
                added = list(MICRO_FEATURES)
            elif version == "C_cross_market_xag":
                added = list(CROSS_FEATURES)
            else:
                added = [*MICRO_FEATURES, *CROSS_FEATURES]
            result, diagnostics, records = score_fixed_cohort(
                train,
                evaluation,
                report20["baseline"]["folds"][fold],
                base_features,
                sets[version],
                added,
            )
            fold_results[fold][version] = result
            model_diagnostics[fold][version] = diagnostics
            records_by_fold[fold][version] = records
            print(f"{fold} {version} complete", flush=True)

    summaries = summarize_versions(fold_results)
    control_expected = report20["score_stability"]["e_net"][
        "fold_spearman_vs_net_r"
    ]
    control_actual = summaries["A_technical_control"][
        "fold_e_net_spearman_vs_net_r"
    ]
    if not np.allclose(control_expected, control_actual, atol=1e-8, rtol=1e-8):
        raise RuntimeError("Frozen Gen20 control could not be reproduced")

    passing = [
        version
        for version in VERSIONS
        if summaries[version]["information_gate_pass"]
    ]
    passing.sort(
        key=lambda version: summaries[version]["mean_e_net_information_gain"],
        reverse=True,
    )
    candidate = (
        candidate_from_information(passing[0], records_by_fold, report20)
        if passing
        else {
            "executed": False,
            "reason": "No feature family passed the frozen information gate",
            "research_gate_pass": False,
        }
    )
    frozen_candidate = (
        f"{passing[0]}_e_net_positive"
        if candidate["executed"] and candidate["research_gate_pass"]
        else None
    )

    last_inspected = report17["data"]["recent_development_end"]
    protocol = forward_protocol(last_inspected, generated_at)
    source_inventory = {
        "microstructure": micro_inventory,
        "cross_market": {
            "SILVER_XAG": {**silver_inventory, **cross_alignment},
            "DXY_USD_index": {
                "available": False,
                "reason": "No local aligned historical dataset; MT5 exposes only a current dated futures contract.",
            },
            "US_2Y_yield": {
                "available": False,
                "reason": "No local or broker historical series.",
            },
            "US_10Y_yield": {
                "available": False,
                "reason": "No local or broker historical series.",
            },
            "real_yield_proxy": {
                "available": False,
                "reason": "No timestamp-aligned intraday source.",
            },
            "VIX": {
                "available": False,
                "reason": "No local aligned history; MT5 exposes only a current dated futures contract.",
            },
            "WTI": {
                "available": False,
                "reason": "No reproducible local aligned historical dataset; server download was not used.",
            },
        },
        "economic_event_context": {
            "available": False,
            "tested": False,
            "reason": (
                "No repository-local, timestamped historical calendar with documented "
                "release availability; event dates and surprises were not fabricated."
            ),
            "required_future_schema": [
                "scheduled_release_utc",
                "actual_release_utc",
                "event_category",
                "impact",
                "forecast_available_utc",
                "actual_available_utc",
                "forecast",
                "actual",
            ],
        },
    }

    pooled_control = report20["baseline"]["selection_pooled"]
    report = {
        "generated_at": generated_at.isoformat(),
        "generation": "21_new_information_source_study",
        "status": "research_only",
        "research_question": (
            "Can genuinely new entry-time information improve cross-regime "
            "OOS discrimination with the Gen20 architecture frozen?"
        ),
        "development_history_policy": report20["development_history_policy"],
        "frozen_comparability": {
            "parent": "Generation 20 technical/cost direct-model control",
            "fixed_cohort": "Gen17 observed-cost 206 executable trades",
            "signal_definition_changed": False,
            "tp_sl_changed": False,
            "execution_changed": False,
            "cost_changed": False,
            "model_hyperparameters_changed": False,
            "threshold_sweep": False,
            "meta_model_sweep": False,
            "xgboost_hyperparameter_sweep": False,
            "tp_atr": 1.3,
            "sl_atr": 1.6,
            "horizon": 90,
            "same_bar_tp_sl": "stop_first",
            "position_limit": 1,
            "base_extra_cost_points": gen19.BASE_EXTRA_COST_POINTS,
            "stress_extra_cost_points": gen19.STRESS_EXTRA_COST_POINTS,
        },
        "information_gate": INFORMATION_GATE,
        "feature_sets": {
            "A_technical_control": control_features,
            "B_microstructure": list(MICRO_FEATURES),
            "C_cross_market_xag": list(CROSS_FEATURES),
            "D_event_context": [],
            "E_strongest_two_predeclared": [*MICRO_FEATURES, *CROSS_FEATURES],
            "F_all_justified": [*MICRO_FEATURES, *CROSS_FEATURES],
            "combination_note": (
                "E was predeclared as B+C because these were the only two testable "
                "new families; F is identical because event context was unavailable."
            ),
        },
        "source_inventory": source_inventory,
        "control_baseline": {
            "folds": report20["baseline"]["folds"],
            "selection_pooled": pooled_control,
        },
        "fold_results": fold_results,
        "version_summary": summaries,
        "model_diagnostics": model_diagnostics,
        "univariate_new_feature_separation": {
            "microstructure": univariate_separation(
                history, report20, MICRO_FEATURES
            ),
            "cross_market_xag": univariate_separation(
                history, report20, CROSS_FEATURES
            ),
        },
        "candidate_construction": candidate,
        "frozen_candidate_id": frozen_candidate,
        "candidate_evidence_sufficient": bool(frozen_candidate),
        "untouched_forward_protocol": protocol,
        "answers": {
            "technical_feature_set_apparent_ceiling": not bool(passing),
            "largest_information_gain_family": max(
                (version for version in VERSIONS if version != "A_technical_control"),
                key=lambda version: summaries[version][
                    "mean_e_net_information_gain"
                ],
            ),
            "microstructure_information_gate_pass": summaries[
                "B_microstructure"
            ]["information_gate_pass"],
            "xag_cross_market_information_gate_pass": summaries[
                "C_cross_market_xag"
            ]["information_gate_pass"],
            "dxy_yield_testable": False,
            "event_context_testable": False,
            "feature_family_stable_across_two_regimes": passing,
            "enough_evidence_for_candidate": bool(frozen_candidate),
            "additional_data_required_if_no_candidate": not bool(
                frozen_candidate
            ),
        },
        "selection_inventory": {
            "model_architectures": 1,
            "feature_ablations": [
                "control",
                "control+microstructure",
                "control+XAG",
                "control+microstructure+XAG",
            ],
            "candidate_selectors": 1 if passing else 0,
            "candidate_selector": "E(net_R)>0" if passing else None,
            "hyperparameter_variants": 0,
            "threshold_sweep_variants": 0,
        },
        "promotion_pass": False,
        "gemini_modified": False,
        "gemini_sha256_before_and_after": gemini_hash,
        "final_untouched_test_validity": "FAIL_pending_future_data",
    }
    config = {
        "generation": report["generation"],
        "status": "research_only",
        "information_gate_passing_versions": passing,
        "candidate_construction_executed": candidate["executed"],
        "frozen_candidate_id": frozen_candidate,
        "promotion_pass": False,
        "untouched_forward_cutoff_utc": protocol[
            "untouched_forward_cutoff_utc"
        ],
    }
    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REPORT_MD.write_text(markdown(report), encoding="utf-8")
    CONFIG_JSON.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Wrote {REPORT_JSON.name}, {REPORT_MD.name}, and {CONFIG_JSON.name}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
