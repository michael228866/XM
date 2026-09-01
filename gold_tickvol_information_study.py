from __future__ import annotations

import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import gold_generation16_independent_families as gen16
import gold_generation17_cross_regime as gen17
import gold_generation19_cost_aware as gen19
import gold_generation20_direct_net_edge as gen20
import gold_generation21_new_information as gen21
import drl_trading_v2
from barrier_final_train import prepare_barrier_data
from gold_generation11_execution_aligned import add_targets
from gold_regime_experts_iterative import training_frame
from gold_regime_experts_walk_forward import SELECTION_FOLDS


ROOT = Path(__file__).resolve().parent
GOLD_M1 = ROOT / "GOLD#_M1_201401020000_202605082357.csv"
GEN20_REPORT = ROOT / "gold_generation20_direct_net_edge.json"
GEN21_REPORT = ROOT / "gold_generation21_new_information.json"
REPORT_JSON = ROOT / "gold_tickvol_information_study.json"
REPORT_MD = ROOT / "gold_tickvol_information_study.md"

FORWARD_CUTOFF_UTC = "2026-09-01T02:00:00Z"
MIXED_GRANULARITY_TRANSITION = pd.Timestamp("2014-06-13 01:34:00")
TICKVOL_STUDY_START = pd.Timestamp("2015-01-01 00:00:00")
SHORT_WINDOW = 60
MEDIUM_WINDOW = 1_440

H1_FEATURE = "TV_ACCEL"
MINIMAL_FEATURES = (
    "TV_LOG_LEVEL",
    "TV_VELOCITY",
    "TV_ACCEL",
    "TV_PCTL_60",
    "TV_PCTL_1440",
    "TV_Z_60",
    "TV_Z_1440",
    "TV_BURST_LOG_RATIO_60_1440",
)
INTERACTION_FEATURES = (
    "TV_ACCEL_X_ATR_PCTL",
    "TV_ACCEL_X_TREND_EFF",
    "TV_ACCEL_X_RV_PCTL",
)
ALL_TICKVOL_FEATURES = (*MINIMAL_FEATURES, *INTERACTION_FEATURES)
VERSIONS = (
    "A_technical_control",
    "B_control_plus_frozen_H1",
    "C_control_plus_minimal_tickvol",
    "D_control_plus_predeclared_interactions",
)
INFORMATION_GATE = {
    "mean_spearman_improvement_min": 0.05,
    "positive_folds_min": 2,
    "improved_folds_min": 2,
    "worst_fold_spearman_min": -0.05,
    "model_free_support_required": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isolated historical GOLD M1 TICKVOL information study"
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


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3:
        return None
    value = pd.Series(left[finite]).corr(
        pd.Series(right[finite]), method="spearman"
    )
    return None if pd.isna(value) else float(value)


def audit_tickvol() -> dict:
    raw = pd.read_csv(
        GOLD_M1,
        sep="\t",
        usecols=["<DATE>", "<TIME>", "<TICKVOL>"],
    ).rename(
        columns={
            "<DATE>": "DATE",
            "<TIME>": "TIME",
            "<TICKVOL>": "TICKVOL",
        }
    )
    raw["TIME_DT"] = pd.to_datetime(
        raw["DATE"].astype(str) + " " + raw["TIME"].astype(str),
        format="%Y.%m.%d %H:%M:%S",
        errors="raise",
    )
    tick = pd.to_numeric(raw["TICKVOL"], errors="coerce")
    raw["TICKVOL"] = tick
    raw["YEAR"] = raw["TIME_DT"].dt.year
    raw["DATE_ONLY"] = raw["TIME_DT"].dt.date
    raw["GAP_MINUTES"] = raw["TIME_DT"].diff().dt.total_seconds() / 60.0

    daily_counts_2014 = raw.loc[raw["YEAR"] == 2014].groupby("DATE_ONLY").size()
    transition_day = MIXED_GRANULARITY_TRANSITION.date()
    if int(daily_counts_2014.get(transition_day, 0)) < 1_000:
        raise RuntimeError("Expected 2014 H1-to-M1 transition was not reproduced")
    before_transition_day = daily_counts_2014.loc[
        daily_counts_2014.index < transition_day
    ]
    if int(before_transition_day.tail(1).iloc[0]) > 24:
        raise RuntimeError("2014 pre-transition bars no longer look hourly")

    years = []
    previous_median: float | None = None
    for year, part in raw.groupby("YEAR", sort=True):
        values = part["TICKVOL"]
        quantiles = values.quantile([0.25, 0.50, 0.75, 0.90, 0.95])
        median = float(quantiles.loc[0.50])
        ratio = None if previous_median is None else median / previous_median
        valid_m1 = (
            part["TIME_DT"] >= MIXED_GRANULARITY_TRANSITION
            if int(year) == 2014
            else pd.Series(True, index=part.index)
        )
        structural_notes = []
        if int(year) == 2014:
            structural_notes.append(
                "Mixed H1 and M1 granularity; exclude the full year from modeling"
            )
        if ratio is not None and (ratio < 0.67 or ratio > 1.50):
            structural_notes.append(
                f"Annual median shifted materially versus prior year ({ratio:.3f}x)"
            )
        if int(year) == 2026:
            structural_notes.append("Partial year ending 2026-05-08")
        gap = part["GAP_MINUTES"]
        years.append(
            {
                "year": int(year),
                "m1_bars": int(len(part)),
                "valid_m1_bars": int(valid_m1.sum()),
                "missing_tickvol": int(values.isna().sum()),
                "zero_tickvol": int(values.eq(0.0).sum()),
                "p25": float(quantiles.loc[0.25]),
                "median": median,
                "p75": float(quantiles.loc[0.75]),
                "p90": float(quantiles.loc[0.90]),
                "p95": float(quantiles.loc[0.95]),
                "median_ratio_vs_prior_year": ratio,
                "one_minute_gap_share": float(gap.eq(1.0).mean()),
                "structural_notes": structural_notes,
                "modeling_eligible": int(year) >= 2015,
            }
        )
        previous_median = median

    result = {
        "source": GOLD_M1.name,
        "sha256": file_hash(GOLD_M1),
        "rows": int(len(raw)),
        "first_timestamp_server": raw["TIME_DT"].iat[0].isoformat(),
        "last_timestamp_server": raw["TIME_DT"].iat[-1].isoformat(),
        "duplicates": int(raw["TIME_DT"].duplicated().sum()),
        "non_monotonic_timestamps": int(
            raw["TIME_DT"].diff().dropna().le(pd.Timedelta(0)).sum()
        ),
        "first_valid_m1_timestamp_server": (
            MIXED_GRANULARITY_TRANSITION.isoformat()
        ),
        "conservative_modeling_start_server": TICKVOL_STUDY_START.isoformat(),
        "excluded_years": {
            "2014": (
                "The file begins with H1 bars and switches to M1 on "
                "2014-06-13; the entire mixed-granularity year is excluded."
            )
        },
        "retained_years": "2015-2024 for the three OOS folds; 2025-2026 remain development only",
        "discontinuity_interpretation": (
            "TICKVOL is observed and nonzero after the M1 transition, but large "
            "year-to-year level shifts show that absolute volume is not stationary."
        ),
        "by_year": years,
    }
    del raw
    gc.collect()
    return result


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    historical = series.shift(1)
    mean = historical.rolling(window, min_periods=window).mean()
    std = historical.rolling(window, min_periods=window).std()
    return (series - mean) / std.replace(0.0, np.nan)


def add_tickvol_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    tick = pd.to_numeric(result["TICKVOL"], errors="coerce")
    valid = tick.where(
        (result["TIME_DT"] >= TICKVOL_STUDY_START) & tick.gt(0.0)
    )
    completed_log = np.log1p(valid).shift(1)
    velocity = completed_log.diff()
    acceleration = velocity.diff()
    short_mean = completed_log.rolling(
        SHORT_WINDOW, min_periods=SHORT_WINDOW
    ).mean()
    prior_medium_mean = completed_log.shift(SHORT_WINDOW).rolling(
        MEDIUM_WINDOW, min_periods=MEDIUM_WINDOW
    ).mean()

    result["TV_LOG_LEVEL"] = completed_log.astype(np.float32)
    result["TV_VELOCITY"] = velocity.astype(np.float32)
    result["TV_ACCEL"] = acceleration.astype(np.float32)
    result["TV_PCTL_60"] = completed_log.rolling(
        SHORT_WINDOW, min_periods=SHORT_WINDOW
    ).rank(pct=True).astype(np.float32)
    result["TV_PCTL_1440"] = completed_log.rolling(
        MEDIUM_WINDOW, min_periods=MEDIUM_WINDOW
    ).rank(pct=True).astype(np.float32)
    result["TV_Z_60"] = rolling_zscore(completed_log, SHORT_WINDOW).astype(
        np.float32
    )
    result["TV_Z_1440"] = rolling_zscore(completed_log, MEDIUM_WINDOW).astype(
        np.float32
    )
    result["TV_BURST_LOG_RATIO_60_1440"] = (
        short_mean - prior_medium_mean
    ).astype(np.float32)
    result["TV_ACCEL_X_ATR_PCTL"] = (
        acceleration * (result["COST_ATR_PCTL"] - 0.5)
    ).astype(np.float32)
    result["TV_ACCEL_X_TREND_EFF"] = (
        acceleration * result["REG_TREND_EFF_30"]
    ).astype(np.float32)
    result["TV_ACCEL_X_RV_PCTL"] = (
        acceleration * (result["REG_RV_PCTL_20D"] - 0.5)
    ).astype(np.float32)
    return result


def feature_sets(control: list[str]) -> dict[str, list[str]]:
    return {
        "A_technical_control": list(control),
        "B_control_plus_frozen_H1": [*control, H1_FEATURE],
        "C_control_plus_minimal_tickvol": [*control, *MINIMAL_FEATURES],
        "D_control_plus_predeclared_interactions": [
            *control,
            *MINIMAL_FEATURES,
            *INTERACTION_FEATURES,
        ],
    }


def added_features(version: str) -> list[str]:
    if version == "A_technical_control":
        return []
    if version == "B_control_plus_frozen_H1":
        return [H1_FEATURE]
    if version == "C_control_plus_minimal_tickvol":
        return list(MINIMAL_FEATURES)
    return list(ALL_TICKVOL_FEATURES)


def compact_fold_result(result: dict) -> dict:
    return {
        "usable_observations": result["usable_observations"],
        "ranking": result["ranking"],
        "calibration": result["calibration"],
        "fixed_cohort_metrics": result["executable_metrics"],
    }


def summarize_versions(fold_results: dict) -> dict:
    output = {}
    for version in VERSIONS:
        e_folds = [
            fold_results[fold][version]["ranking"]["e_net_vs_net_r"]
            for fold, *_ in SELECTION_FOLDS
        ]
        p_folds = [
            fold_results[fold][version]["ranking"][
                "p_net_vs_positive_net_r"
            ]
            for fold, *_ in SELECTION_FOLDS
        ]
        valid_p_folds = [value for value in p_folds if value is not None]
        output[version] = {
            "fold_e_net_spearman_vs_net_r": e_folds,
            "mean_e_net_spearman_vs_net_r": float(np.mean(e_folds)),
            "fold_p_net_spearman_vs_positive_net_r": p_folds,
            "mean_p_net_spearman_vs_positive_net_r": (
                float(np.mean(valid_p_folds)) if valid_p_folds else None
            ),
            "mean_p_net_brier": float(
                np.mean(
                    [
                        fold_results[fold][version]["calibration"]["p_net"][
                            "brier"
                        ]
                        for fold, *_ in SELECTION_FOLDS
                    ]
                )
            ),
            "mean_p_net_ece": float(
                np.mean(
                    [
                        fold_results[fold][version]["calibration"]["p_net"][
                            "ece"
                        ]
                        for fold, *_ in SELECTION_FOLDS
                    ]
                )
            ),
        }
    control = output["A_technical_control"]
    for version, value in output.items():
        deltas = [
            current - baseline
            for current, baseline in zip(
                value["fold_e_net_spearman_vs_net_r"],
                control["fold_e_net_spearman_vs_net_r"],
            )
        ]
        value["fold_delta_vs_control"] = deltas
        value["mean_delta_vs_control"] = (
            value["mean_e_net_spearman_vs_net_r"]
            - control["mean_e_net_spearman_vs_net_r"]
        )
        value["positive_folds"] = sum(
            score > 0.0 for score in value["fold_e_net_spearman_vs_net_r"]
        )
        value["improved_folds"] = sum(delta > 0.0 for delta in deltas)
        value["model_gate_without_model_free"] = bool(
            version != "A_technical_control"
            and value["mean_delta_vs_control"]
            >= INFORMATION_GATE["mean_spearman_improvement_min"]
            and value["positive_folds"]
            >= INFORMATION_GATE["positive_folds_min"]
            and value["improved_folds"]
            >= INFORMATION_GATE["improved_folds_min"]
            and min(value["fold_e_net_spearman_vs_net_r"])
            >= INFORMATION_GATE["worst_fold_spearman_min"]
        )
    return output


def decile_table(rows: pd.DataFrame, feature: str) -> dict:
    usable = rows[["fold", feature, "net_r", "net_positive", "tp_first"]].dropna()
    if usable.empty:
        return {
            "rows": [],
            "mean_r_trend_spearman": None,
            "positive_rate_trend_spearman": None,
            "mean_r_adjacent_increase_fraction": None,
            "strict_mean_r_monotonic": False,
        }
    usable = usable.copy()
    usable["decile"] = usable.groupby("fold")[feature].transform(
        lambda values: np.ceil(
            values.rank(method="first", pct=True) * 10.0
        ).clip(1, 10)
    )
    table = (
        usable.groupby("decile", sort=True)
        .agg(
            trades=("net_r", "size"),
            mean_r=("net_r", "mean"),
            positive_return_rate=("net_positive", "mean"),
            tp_first_rate=("tp_first", "mean"),
        )
        .reset_index()
    )
    mean_r = table["mean_r"].to_numpy(dtype=np.float64)
    adjacent = np.diff(mean_r)
    return {
        "assignment": "fold-relative deciles; no cross-fold absolute cutoff",
        "rows": table.to_dict(orient="records"),
        "mean_r_trend_spearman": safe_spearman(table["decile"], mean_r),
        "positive_rate_trend_spearman": safe_spearman(
            table["decile"], table["positive_return_rate"]
        ),
        "mean_r_adjacent_increase_fraction": (
            float(np.mean(adjacent >= 0.0)) if len(adjacent) else None
        ),
        "strict_mean_r_monotonic": bool(
            len(adjacent) > 0 and np.all(adjacent >= 0.0)
        ),
    }


def model_free_feature(rows: pd.DataFrame, feature: str) -> dict:
    folds = []
    for fold, *_ in SELECTION_FOLDS:
        part = rows.loc[rows["fold"] == fold]
        finite = np.isfinite(part[feature]) & np.isfinite(part["net_r"])
        part = part.loc[finite]
        folds.append(
            {
                "fold": fold,
                "usable": int(len(part)),
                "spearman_vs_net_r": safe_spearman(
                    part[feature], part["net_r"]
                ),
                "spearman_vs_positive_net_r": safe_spearman(
                    part[feature], part["net_positive"]
                ),
                "spearman_vs_tp_first": safe_spearman(
                    part[feature], part["tp_first"]
                ),
                "winner_median": (
                    float(part.loc[part["net_positive"] == 1, feature].median())
                    if (part["net_positive"] == 1).any()
                    else None
                ),
                "loser_median": (
                    float(part.loc[part["net_positive"] == 0, feature].median())
                    if (part["net_positive"] == 0).any()
                    else None
                ),
                "deciles": decile_table(part, feature),
            }
        )
    fold_correlations = [value["spearman_vs_net_r"] for value in folds]
    pooled = rows.loc[
        np.isfinite(rows[feature]) & np.isfinite(rows["net_r"])
    ]
    return {
        "feature": feature,
        "folds": folds,
        "mean_fold_spearman_vs_net_r": float(np.mean(fold_correlations)),
        "positive_spearman_folds": sum(value > 0.0 for value in fold_correlations),
        "pooled_raw_spearman_vs_net_r": safe_spearman(
            pooled[feature], pooled["net_r"]
        ),
        "pooled_raw_spearman_vs_positive_net_r": safe_spearman(
            pooled[feature], pooled["net_positive"]
        ),
        "pooled_raw_spearman_vs_tp_first": safe_spearman(
            pooled[feature], pooled["tp_first"]
        ),
        "fold_relative_deciles": decile_table(pooled, feature),
    }


def blocked_permutation_p_value(
    rows: pd.DataFrame, feature: str, permutations: int = 5_000
) -> dict:
    rng = np.random.default_rng(20260901)
    pairs = []
    observed = []
    for fold, *_ in SELECTION_FOLDS:
        part = rows.loc[rows["fold"] == fold, [feature, "net_r"]].dropna()
        x = part[feature].rank(method="average").to_numpy(dtype=np.float64)
        y = part["net_r"].rank(method="average").to_numpy(dtype=np.float64)
        pairs.append((x, y))
        observed.append(float(np.corrcoef(x, y)[0, 1]))
    observed_mean = float(np.mean(observed))
    null = np.empty(permutations, dtype=np.float64)
    for position in range(permutations):
        null[position] = np.mean(
            [float(np.corrcoef(x, rng.permutation(y))[0, 1]) for x, y in pairs]
        )
    return {
        "statistic": "mean of the three fold Spearman correlations",
        "observed": observed_mean,
        "permutations": permutations,
        "two_sided_p_value": float(
            (1 + np.sum(np.abs(null) >= abs(observed_mean)))
            / (permutations + 1)
        ),
        "note": "Development-data diagnostic only; no multiplicity correction can restore untouched confirmation.",
    }


def fixed_cohort_rows(
    history: pd.DataFrame, records_by_fold: dict[str, dict[str, list[dict]]]
) -> pd.DataFrame:
    rows = []
    for fold, start, end in SELECTION_FOLDS:
        evaluation = history.loc[
            (history["TIME_DT"] >= start) & (history["TIME_DT"] < end)
        ].reset_index(drop=True)
        records = records_by_fold[fold]["A_technical_control"]
        for record in records:
            index = int(record["index"])
            values = evaluation.iloc[index]
            row = {
                "fold": fold,
                "index": index,
                "time": record["time"],
                "net_r": float(record["reward"]),
                "net_positive": int(float(record["reward"]) > 0.0),
                "tp_first": int(record["outcome"] == 1),
                "direction": "short",
                "expert": "short_trend_continuation",
                "session": int(values["COST_SESSION"]),
                "atr_percentile": float(values["COST_ATR_PCTL"]),
                "volatility_percentile": float(values["REG_RV_PCTL_20D"]),
            }
            row.update(
                {
                    feature: float(values[feature])
                    for feature in ALL_TICKVOL_FEATURES
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def group_diagnostic(rows: pd.DataFrame, key: str) -> list[dict]:
    output = []
    for value, part in rows.groupby(key, sort=True):
        output.append(
            {
                "group": str(value),
                "trades": int(len(part)),
                "spearman": safe_spearman(part[H1_FEATURE], part["net_r"]),
                "mean_r": float(part["net_r"].mean()),
                "positive_rate": float(part["net_positive"].mean()),
            }
        )
    return output


def fixed_cohort_stratification(rows: pd.DataFrame) -> dict:
    work = rows.copy()
    work["session_name"] = work["session"].map(
        {0: "00-05", 1: "06-12", 2: "13-18", 3: "19-23"}
    )
    work["atr_regime"] = pd.cut(
        work["atr_percentile"],
        bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
        labels=["low", "mid", "high"],
    )
    work["volatility_regime"] = pd.cut(
        work["volatility_percentile"],
        bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
        labels=["low", "mid", "high"],
    )
    return {
        "scope": (
            "Frozen Gen17 cohort contains only short trend-continuation trades; "
            "long/other-expert evidence is reported separately as non-selection diagnostics."
        ),
        "by_session": group_diagnostic(work, "session_name"),
        "by_atr_percentile": group_diagnostic(work, "atr_regime"),
        "by_volatility_regime": group_diagnostic(work, "volatility_regime"),
    }


def observed_cost_net_reward(frame: pd.DataFrame, direction: int) -> np.ndarray:
    reward_column = "LONG_REWARD" if direction == 1 else "SHORT_REWARD"
    fixed_reward = frame[reward_column].to_numpy(dtype=np.float64)
    spread_points = frame["COST_SPREAD_POINTS"].to_numpy(dtype=np.float64)
    atr = frame["ATR"].to_numpy(dtype=np.float64)
    stop = np.maximum(atr * gen16.SL_ATR, gen16.MIN_SL_PRICE)
    spread_price = spread_points * gen19.POINT
    gross_pnl = fixed_reward * (
        stop + gen19.FALLBACK_SPREAD_POINTS * gen19.POINT
    ) + (
        gen19.FALLBACK_SPREAD_POINTS + gen19.BASE_EXTRA_COST_POINTS
    ) * gen19.POINT
    return (
        gross_pnl
        - (spread_points + gen19.BASE_EXTRA_COST_POINTS) * gen19.POINT
    ) / (stop + spread_price)


def all_expert_diagnostics(
    history: pd.DataFrame, base_features: list[str]
) -> dict:
    output = {}
    for fold, start, end in SELECTION_FOLDS:
        evaluation = history.loc[
            (history["TIME_DT"] >= start) & (history["TIME_DT"] < end)
        ].reset_index(drop=True)
        net_by_direction = {
            1: observed_cost_net_reward(evaluation, 1),
            2: observed_cost_net_reward(evaluation, 2),
        }
        events = gen16.expert_event_indices(evaluation, base_features)
        fold_output = {}
        for expert, indices in events.items():
            direction = gen16.EXPERT_DIRECTION[expert]
            exit_column = (
                "LONG_EXIT_OFFSET" if direction == 1 else "SHORT_EXIT_OFFSET"
            )
            exits = evaluation[exit_column].to_numpy(dtype=np.int64)
            indices = indices[indices + exits[indices] < len(evaluation)]
            values = evaluation[H1_FEATURE].to_numpy(dtype=np.float64)[indices]
            rewards = net_by_direction[direction][indices]
            finite = np.isfinite(values) & np.isfinite(rewards)
            fold_output[expert] = {
                "events": int(finite.sum()),
                "spearman": safe_spearman(values[finite], rewards[finite]),
                "mean_r": (
                    float(np.mean(rewards[finite])) if finite.any() else None
                ),
                "diagnostic_only": True,
            }
        output[fold] = fold_output
    aggregate = {}
    for expert in gen16.EXPERTS:
        correlations = [
            output[fold][expert]["spearman"] for fold, *_ in SELECTION_FOLDS
        ]
        correlations = [value for value in correlations if value is not None]
        aggregate[expert] = {
            "fold_spearman": [
                output[fold][expert]["spearman"] for fold, *_ in SELECTION_FOLDS
            ],
            "positive_folds": sum(value > 0.0 for value in correlations),
            "mean_fold_spearman": (
                float(np.mean(correlations)) if correlations else None
            ),
            "total_events": sum(
                output[fold][expert]["events"] for fold, *_ in SELECTION_FOLDS
            ),
        }
    return {
        "folds": output,
        "aggregate": aggregate,
        "warning": (
            "These static-family events are an explanatory diagnostic, not a new "
            "portfolio, subgroup selection, or OOS candidate."
        ),
    }


def feature_drift(rows: pd.DataFrame) -> dict:
    output = {}
    for feature in ALL_TICKVOL_FEATURES:
        distributions = {}
        for fold, *_ in SELECTION_FOLDS:
            values = rows.loc[rows["fold"] == fold, feature].dropna()
            distributions[fold] = {
                "usable": int(len(values)),
                "p25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "p75": float(values.quantile(0.75)),
                "p90": float(values.quantile(0.90)),
            }
        medians = [value["median"] for value in distributions.values()]
        output[feature] = {
            "folds": distributions,
            "median_range": float(max(medians) - min(medians)),
        }
    return output


def feature_gain(diagnostics: dict, features: tuple[str, ...]) -> dict:
    output = {}
    for version in VERSIONS[1:]:
        by_feature = {}
        for feature in features:
            gains = []
            for fold, *_ in SELECTION_FOLDS:
                gain = diagnostics[fold][version]["feature_importance"][
                    "e_net"
                ].get(feature, 0.0)
                gains.append(float(gain))
            by_feature[feature] = {
                "fold_gain": gains,
                "mean_gain": float(np.mean(gains)),
            }
        output[version] = by_feature
    return {
        "values": output,
        "interpretation": (
            "Gain is secondary only. No feature is accepted unless chronological "
            "ranking and model-free diagnostics agree."
        ),
    }


def expected_gen21_h1(report21: dict) -> dict[str, float]:
    for row in report21["univariate_new_feature_separation"]["microstructure"]:
        if row["feature"] == "MICRO_TICKVOL_ACCEL":
            return {
                fold["fold"]: float(fold["spearman_vs_net_r"])
                for fold in row["folds"]
            }
    raise RuntimeError("Gen21 H1 diagnostic was not found")


def markdown(report: dict) -> str:
    summaries = report["incremental_test"]["version_summary"]
    labels = {
        "A_technical_control": "Technical control",
        "B_control_plus_frozen_H1": "Control + frozen H1 TV_ACCEL",
        "C_control_plus_minimal_tickvol": "Control + minimal TICKVOL family",
        "D_control_plus_predeclared_interactions": (
            "Control + predeclared TICKVOL interactions"
        ),
    }
    lines = [
        "# TICKVOL Information Study",
        "",
        f"Status: **{report['status']}**",
        "",
        "No Generation 22, candidate, champion, production, TP/SL, execution, or gemini.py change was made.",
        "",
        "## Incremental chronological OOS ranking",
        "",
        "| Information set | Fold 1 Spearman | Fold 2 Spearman | Fold 3 Spearman | Mean Spearman | Delta vs Control | Verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for version in VERSIONS:
        value = summaries[version]
        fold_scores = value["fold_e_net_spearman_vs_net_r"]
        lines.append(
            f"| {labels[version]} | {fold_scores[0]:.4f} | {fold_scores[1]:.4f} | "
            f"{fold_scores[2]:.4f} | {value['mean_e_net_spearman_vs_net_r']:.4f} | "
            f"{value['mean_delta_vs_control']:+.4f} | {value['verdict']} |"
        )

    lines.extend(
        [
            "",
            "### Positive-net-R ranking and calibration",
            "",
            "| Information set | Mean P(net-R>0) Spearman | Mean Brier | Mean ECE |",
            "|---|---:|---:|---:|",
        ]
    )
    for version in VERSIONS:
        value = summaries[version]
        p_spearman = value["mean_p_net_spearman_vs_positive_net_r"]
        p_text = "undefined" if p_spearman is None else f"{p_spearman:.4f}"
        lines.append(
            f"| {labels[version]} | {p_text} | {value['mean_p_net_brier']:.4f} | "
            f"{value['mean_p_net_ece']:.4f} |"
        )

    h1 = report["frozen_h1"]
    lines.extend(
        [
            "",
            "## Frozen H1 result",
            "",
            "| Fold | Trades | Spearman net-R | Spearman positive net-R | Spearman TP-first |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for fold in h1["model_free"]["folds"]:
        lines.append(
            f"| {fold['fold']} | {fold['usable']} | {fold['spearman_vs_net_r']:.4f} | "
            f"{fold['spearman_vs_positive_net_r']:.4f} | {fold['spearman_vs_tp_first']:.4f} |"
        )
    deciles = h1["model_free"]["fold_relative_deciles"]
    lines.extend(
        [
            "",
            f"Mean fold Spearman: **{h1['model_free']['mean_fold_spearman_vs_net_r']:.4f}**. "
            f"Blocked permutation p-value: **{h1['blocked_permutation']['two_sided_p_value']:.4f}**.",
            "",
            f"Strict decile monotonicity: **{deciles['strict_mean_r_monotonic']}**; "
            f"adjacent increase fraction: **{deciles['mean_r_adjacent_increase_fraction']:.1%}**.",
            "",
            "## Model-free minimal ablation",
            "",
            "| Feature | Mean fold Spearman | Positive folds | Pooled decile trend | Adjacent increases | Strict monotonic |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for feature, value in report["minimal_ablation_model_free"].items():
        decile = value["fold_relative_deciles"]
        lines.append(
            f"| {feature} | {value['mean_fold_spearman_vs_net_r']:.4f} | "
            f"{value['positive_spearman_folds']}/3 | {decile['mean_r_trend_spearman']:.4f} | "
            f"{decile['mean_r_adjacent_increase_fraction']:.1%} | "
            f"{decile['strict_mean_r_monotonic']} |"
        )
    lines.extend(
        [
            "",
            "## Data audit",
            "",
            "The file contains a mixed H1/M1 2014 segment. TICKVOL modeling starts conservatively at 2015-01-01; all three requested folds have observed, nonzero M1 TICKVOL. Large annual median shifts remain, so absolute levels are nonstationary.",
            "",
            "| Year | File rows | Valid M1 rows | Missing | Zero | p25 | Median | p75 | p90 | p95 | Eligible |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["data_audit"]["by_year"]:
        lines.append(
            f"| {row['year']} | {row['m1_bars']} | {row['valid_m1_bars']} | "
            f"{row['missing_tickvol']} | {row['zero_tickvol']} | {row['p25']:.0f} | {row['median']:.0f} | "
            f"{row['p75']:.0f} | {row['p90']:.0f} | {row['p95']:.0f} | "
            f"{'yes' if row['modeling_eligible'] else 'no'} |"
        )

    lines.extend(["", "### Structural changes", ""])
    for row in report["data_audit"]["by_year"]:
        for note in row["structural_notes"]:
            lines.append(f"- {row['year']}: {note}")
    lines.extend(
        [
            "",
            "The export contains no source identifier, so activity changes cannot be conclusively separated from broker-feed changes. The discontinuities are treated as data drift, not predictive signals.",
        ]
    )

    lines.extend(["", "## Answers", ""])
    for number, answer in enumerate(report["answers"], start=1):
        lines.append(f"{number}. {answer}")
    lines.extend(
        [
            "",
            "## Forward-data protection",
            "",
            f"Cutoff: `{FORWARD_CUTOFF_UTC}`. No post-cutoff data or outcomes were opened or scored by this study.",
            "",
            f"Generation 22 justified: **{report['generation22_justified']}**.",
        ]
    )
    return "\n".join(lines) + "\n"


def self_check() -> None:
    rows = 2_000
    times = pd.date_range("2018-01-01", periods=rows, freq="min")
    frame = pd.DataFrame(
        {
            "TIME_DT": times,
            "TICKVOL": np.arange(rows, dtype=np.float64) + 10.0,
            "COST_ATR_PCTL": np.full(rows, 0.5),
            "REG_TREND_EFF_30": np.full(rows, 0.2),
            "REG_RV_PCTL_20D": np.full(rows, 0.5),
        }
    )
    base = add_tickvol_features(frame)
    changed = frame.copy()
    changed.loc[1_700, "TICKVOL"] = 999_999.0
    changed = add_tickvol_features(changed)
    assert np.isclose(base.loc[1_700, H1_FEATURE], changed.loc[1_700, H1_FEATURE])
    assert not np.isclose(
        base.loc[1_701, H1_FEATURE], changed.loc[1_701, H1_FEATURE]
    )
    assert base.loc[0, H1_FEATURE] != base.loc[0, H1_FEATURE]
    assert set(ALL_TICKVOL_FEATURES).issubset(base.columns)
    print("gold_tickvol_information_study_self_check_ok")


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0

    generated_at = datetime.now(timezone.utc)
    gemini_hash_before = file_hash(ROOT / "gemini.py")
    report20 = json.loads(GEN20_REPORT.read_text(encoding="utf-8"))
    report21 = json.loads(GEN21_REPORT.read_text(encoding="utf-8"))
    print("Auditing raw TICKVOL", flush=True)
    data_audit = audit_tickvol()

    print("Building frozen Gen21 technical control", flush=True)
    # The historical loader contains a legacy locale-corrupted absolute path.
    # Point the already-imported module at this repository without editing it.
    drl_trading_v2.DATA_DIR = str(ROOT)
    history, base_features = prepare_barrier_data()
    history = add_targets(history)
    history, regime_features = gen17.add_regime_features(history, base_features)
    history = gen20.add_cost_targets_and_features(history)
    robust_base = [
        feature
        for feature in base_features
        if feature not in gen17.ABSOLUTE_SCALE_FEATURES
    ]
    control_features = [*robust_base, *regime_features, *gen20.COST_FEATURES]
    if control_features != report20["architecture"]["model_features"]:
        raise RuntimeError("Gen20 technical control feature set changed")
    history = add_tickvol_features(history)
    sets = feature_sets(control_features)

    fold_results: dict[str, dict] = {}
    model_diagnostics: dict[str, dict] = {}
    records_by_fold: dict[str, dict] = {}
    fold_boundaries: dict[str, dict] = {}
    for fold, start, end in SELECTION_FOLDS:
        train = training_frame(history, start)
        evaluation = history.loc[
            (history["TIME_DT"] >= start) & (history["TIME_DT"] < end)
        ].reset_index(drop=True)
        fold_boundaries[fold] = {
            "train_last_timestamp": train["TIME_DT"].iat[-1].isoformat(),
            "evaluation_start": evaluation["TIME_DT"].iat[0].isoformat(),
            "evaluation_end": evaluation["TIME_DT"].iat[-1].isoformat(),
            "declared_fold_start": start.isoformat(),
            "declared_fold_end_exclusive": end.isoformat(),
        }
        fold_results[fold] = {}
        model_diagnostics[fold] = {}
        records_by_fold[fold] = {}
        for version in VERSIONS:
            result, diagnostics, records = gen21.score_fixed_cohort(
                train,
                evaluation,
                report20["baseline"]["folds"][fold],
                base_features,
                sets[version],
                added_features(version),
            )
            fold_results[fold][version] = compact_fold_result(result)
            model_diagnostics[fold][version] = diagnostics
            records_by_fold[fold][version] = records
            print(f"{fold} {version} complete", flush=True)

    summaries = summarize_versions(fold_results)
    expected_control = report20["score_stability"]["e_net"][
        "fold_spearman_vs_net_r"
    ]
    actual_control = summaries["A_technical_control"][
        "fold_e_net_spearman_vs_net_r"
    ]
    if not np.allclose(expected_control, actual_control, atol=1e-8, rtol=1e-8):
        raise RuntimeError("Frozen Gen20/21 technical control was not reproduced")

    cohort = fixed_cohort_rows(history, records_by_fold)
    model_free = {
        feature: model_free_feature(cohort, feature)
        for feature in ALL_TICKVOL_FEATURES
    }
    h1_expected = expected_gen21_h1(report21)
    h1_actual = {
        value["fold"]: value["spearman_vs_net_r"]
        for value in model_free[H1_FEATURE]["folds"]
    }
    if not all(
        np.isclose(h1_expected[fold], h1_actual[fold], atol=1e-8, rtol=1e-8)
        for fold, *_ in SELECTION_FOLDS
    ):
        raise RuntimeError("Frozen H1 did not reproduce the Gen21 diagnostic")

    h1_deciles = model_free[H1_FEATURE]["fold_relative_deciles"]
    h1_model_free_support = bool(
        model_free[H1_FEATURE]["positive_spearman_folds"] >= 2
        and h1_deciles["mean_r_trend_spearman"] is not None
        and h1_deciles["mean_r_trend_spearman"] > 0.0
        and h1_deciles["mean_r_adjacent_increase_fraction"] >= 0.6
    )
    for version, value in summaries.items():
        if version == "A_technical_control":
            value["model_free_support"] = None
            value["information_gate_pass"] = False
            value["verdict"] = "CONTROL"
            continue
        features = added_features(version)
        stable_features = [
            feature
            for feature in features
            if model_free[feature]["positive_spearman_folds"] >= 2
            and model_free[feature]["fold_relative_deciles"][
                "mean_r_trend_spearman"
            ]
            is not None
            and model_free[feature]["fold_relative_deciles"][
                "mean_r_trend_spearman"
            ]
            > 0.0
            and model_free[feature]["fold_relative_deciles"][
                "mean_r_adjacent_increase_fraction"
            ]
            >= 0.6
        ]
        value["model_free_support"] = stable_features
        value["information_gate_pass"] = bool(
            value["model_gate_without_model_free"] and stable_features
        )
        value["verdict"] = (
            "PROMISING" if value["information_gate_pass"] else "FAIL"
        )

    passing = [
        version for version in VERSIONS if summaries[version]["information_gate_pass"]
    ]
    generation22_justified = bool(passing)
    strongest = max(
        ALL_TICKVOL_FEATURES,
        key=lambda feature: model_free[feature]["mean_fold_spearman_vs_net_r"],
    )
    strongest_version = max(
        VERSIONS[1:],
        key=lambda version: summaries[version]["mean_delta_vs_control"],
    )
    stratification = fixed_cohort_stratification(cohort)
    expert_diagnostic = all_expert_diagnostics(history, base_features)
    drift = feature_drift(cohort)
    strongest_expert = max(
        expert_diagnostic["aggregate"],
        key=lambda expert: expert_diagnostic["aggregate"][expert][
            "mean_fold_spearman"
        ]
        or -np.inf,
    )
    split_evidence = {}
    for fold, *_ in SELECTION_FOLDS:
        split_evidence[fold] = {}
        for version in VERSIONS:
            diagnostic = model_diagnostics[fold][version]
            split_evidence[fold][version] = {
                "events": diagnostic["events"],
                "fit": diagnostic["fit"],
                "calibration": diagnostic["calibration"],
                "policy": diagnostic["policy"],
                "fit_max_label_end_index": diagnostic[
                    "fit_max_label_end_index"
                ],
                "calibration_start_index": diagnostic[
                    "calibration_start_index"
                ],
                "calibration_max_label_end_index": diagnostic[
                    "calibration_max_label_end_index"
                ],
                "policy_start_index": diagnostic["policy_start_index"],
                "fit_labels_end_before_calibration": bool(
                    diagnostic["fit_max_label_end_index"]
                    < diagnostic["calibration_start_index"]
                ),
                "calibration_labels_end_before_policy": bool(
                    diagnostic["calibration_max_label_end_index"]
                    < diagnostic["policy_start_index"]
                ),
            }

    answers = [
        (
            "Frozen H1 reproduced the three positive raw development-fold correlations "
            f"exactly: {list(h1_actual.values())}."
        ),
        (
            "The fold-relative realized Mean-R deciles were "
            f"{'strictly monotonic' if h1_deciles['strict_mean_r_monotonic'] else 'not strictly monotonic'} "
            f"({h1_deciles['mean_r_adjacent_increase_fraction']:.1%} adjacent increases)."
        ),
        (
            "Control + H1 "
            f"changed mean OOS E(net-R) Spearman by {summaries['B_control_plus_frozen_H1']['mean_delta_vs_control']:+.4f}; "
            f"it improved {summaries['B_control_plus_frozen_H1']['improved_folds']}/3 folds."
        ),
        (
            f"The strongest raw model-free TICKVOL feature was {strongest} by mean fold Spearman; "
            "this ranking is diagnostic, not a selected trading feature."
        ),
        (
            f"The best predeclared information set was {strongest_version}, improving "
            f"{summaries[strongest_version]['improved_folds']}/3 chronological regimes."
        ),
        (
            "The frozen cohort itself is only short trend-continuation. In the broader static-family "
            f"diagnostic, the largest mean fold correlation was only {expert_diagnostic['aggregate'][strongest_expert]['mean_fold_spearman']:.4f} "
            f"for {strongest_expert}; the apparent fixed-cohort relationship therefore does not generalize broadly across direction/expert."
        ),
        (
            f"Model-free support for frozen H1 was {'present' if h1_model_free_support else 'insufficient'}; "
            "XGBoost gain was not used as acceptance evidence."
        ),
        (
            f"Generation 22 is {'justified for a later explicit task' if generation22_justified else 'not justified'}; "
            "no Generation 22 artifact or strategy candidate was created."
        ),
    ]

    report = {
        "generated_at": generated_at.isoformat(),
        "study": "TICKVOL_INFORMATION_STUDY_isolated_volume_signal_validation",
        "status": "research_only_no_candidate",
        "preregistration": {
            "frozen_h1": "log1p(TICKVOL).shift(1).diff().diff()",
            "short_window_completed_m1_bars": SHORT_WINDOW,
            "medium_window_completed_m1_bars": MEDIUM_WINDOW,
            "windows_selected_before_evaluation": True,
            "lag_transform_smoothing_sign_threshold_tuned": False,
            "model_hyperparameters_tuned": False,
            "tp_sl_execution_cost_signal_definition_changed": False,
            "posthoc_origin_warning": (
                "H1 was discovered after inspecting multiple Gen21 features; this "
                "study is development evidence and cannot confirm it on an untouched test."
            ),
        },
        "forward_data_protection": {
            "untouched_forward_cutoff_utc": FORWARD_CUTOFF_UTC,
            "post_cutoff_paths_opened": False,
            "post_cutoff_features_or_outcomes_scored": False,
            "last_local_historical_source_timestamp_server": data_audit[
                "last_timestamp_server"
            ],
        },
        "data_audit": data_audit,
        "feature_specification": {
            "invalid_source_period_handling": (
                "TICKVOL is NaN before 2015-01-01 for every volume feature; "
                "technical control is unchanged."
            ),
            "minimal_features": list(MINIMAL_FEATURES),
            "interactions": list(INTERACTION_FEATURES),
            "rolling_semantics": (
                "All raw TICKVOL values are shifted one bar before differences or "
                "rolling transforms; z baselines exclude the completed bar being scored."
            ),
            "feature_sets": sets,
        },
        "fixed_control": {
            "architecture": report20["architecture"],
            "selection_pooled_executable_metrics": report20["baseline"][
                "selection_pooled"
            ]["metrics"],
            "fold_control_reproduced": True,
            "expected_fold_spearman": expected_control,
            "actual_fold_spearman": actual_control,
        },
        "frozen_h1": {
            "gen21_expected_fold_spearman": h1_expected,
            "reproduced_exactly": True,
            "model_free": model_free[H1_FEATURE],
            "blocked_permutation": blocked_permutation_p_value(
                cohort, H1_FEATURE
            ),
            "model_free_support": h1_model_free_support,
        },
        "minimal_ablation_model_free": model_free,
        "incremental_test": {
            "method": (
                "Same Gen21 chronological OOF direct P(net-R>0)/E(net-R) models, "
                "fixed Gen17 206-trade executable cohort, no selection threshold."
            ),
            "information_gate": INFORMATION_GATE,
            "fold_results": fold_results,
            "version_summary": summaries,
            "passing_information_sets": passing,
        },
        "methodology_evidence": {
            "fold_boundaries": fold_boundaries,
            "inner_fit_calibration_policy_purge": split_evidence,
            "fixed_cohort_trades": int(len(cohort)),
            "fixed_cohort_unique_trade_ids": int(
                cohort[["fold", "time", "direction"]].drop_duplicates().shape[0]
            ),
            "fixed_cohort_complete_tickvol_coverage": bool(
                cohort[list(ALL_TICKVOL_FEATURES)].notna().all(axis=1).all()
            ),
            "execution_scope": (
                "The unchanged Gen17 206-trade non-overlapping executable ledger; "
                "raw qualifying rows are not strategy performance observations."
            ),
            "cost_scope": (
                "Unchanged Gen19 observed entry spread, 30-point fallback, and "
                "existing extra-cost assumption embedded in NET_REWARD."
            ),
        },
        "robustness": {
            "fixed_cohort_stratification": stratification,
            "all_expert_model_free_diagnostic": expert_diagnostic,
            "feature_distribution_drift": drift,
        },
        "xgboost_gain_secondary_only": feature_gain(
            model_diagnostics, ALL_TICKVOL_FEATURES
        ),
        "selection_inventory": {
            "information_sets": list(VERSIONS),
            "rolling_windows": [SHORT_WINDOW, MEDIUM_WINDOW],
            "rolling_window_grid_search": False,
            "hyperparameter_variants": 0,
            "threshold_variants": 0,
            "strategy_candidates": 0,
        },
        "answers": answers,
        "generation22_justified": generation22_justified,
        "generation22_created": False,
        "promotion_pass": False,
        "gemini_modified": False,
        "gemini_sha256_before_and_after": gemini_hash_before,
        "final_untouched_test_validity": "FAIL_pending_post_cutoff_forward_test",
    }
    if file_hash(ROOT / "gemini.py") != gemini_hash_before:
        raise RuntimeError("gemini.py changed during the study")
    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    REPORT_MD.write_text(markdown(report), encoding="utf-8")
    print(
        f"Wrote {REPORT_JSON.name} and {REPORT_MD.name}; Gen22 justified={generation22_justified}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
