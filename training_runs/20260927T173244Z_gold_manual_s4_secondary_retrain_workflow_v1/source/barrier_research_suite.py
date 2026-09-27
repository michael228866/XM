import argparse
import os
from dataclasses import dataclass

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from barrier_classifier_strategy import HORIZON, build_barrier_target, evaluate
from barrier_final_train import FINAL_PARAMS
from drl_train_candidate import format_stats
from drl_trading_v2 import load_and_prepare_data


RANDOM_STATE = 42
DEFAULT_N_ESTIMATORS = 180
QUICK_N_ESTIMATORS = 80


@dataclass
class ModelBundle:
    main: xgb.XGBClassifier
    long_model: xgb.XGBClassifier
    short_model: xgb.XGBClassifier
    regime_model: xgb.XGBClassifier
    features: list[str]
    regime_features: list[str]


def add_high_frequency_features(df):
    df = df.copy()
    close = df["CLOSE"]
    high = df["HIGH"]
    low = df["LOW"]
    open_ = df["OPEN"]
    candle_range = (high - low).replace(0, np.nan)

    df["HF_RET_1"] = close.pct_change(1)
    df["HF_RET_3"] = close.pct_change(3)
    df["HF_RET_5"] = close.pct_change(5)
    df["HF_RET_15"] = close.pct_change(15)
    df["HF_MOM_30"] = close.pct_change(30)
    df["HF_MOM_60"] = close.pct_change(60)
    df["HF_RANGE_ATR"] = (high - low) / (df["ATR"] + 1e-6)
    df["HF_BODY_SIGNED"] = (close - open_) / (candle_range + 1e-6)
    df["HF_UPPER_WICK"] = (high - np.maximum(open_, close)) / (candle_range + 1e-6)
    df["HF_LOWER_WICK"] = (np.minimum(open_, close) - low) / (candle_range + 1e-6)
    df["HF_RET_STD_15"] = df["HF_RET_1"].rolling(15).std()
    df["HF_RET_STD_60"] = df["HF_RET_1"].rolling(60).std()
    df["HF_RANGE_MEAN_30"] = (high - low).rolling(30).mean() / (df["ATR"] + 1e-6)
    df["HF_VOLA_BURST"] = df["HF_RET_STD_15"] / (df["HF_RET_STD_60"] + 1e-9)
    df["HF_RSI_SLOPE_5"] = df["M1_RSI"].diff(5)

    trend_cols = [col for col in df.columns if col.endswith("_TREND")]
    df["TREND_SCORE"] = (df[trend_cols] > 0).sum(axis=1).astype(np.float32) if trend_cols else 0.0

    hf_features = [
        "HF_RET_1",
        "HF_RET_3",
        "HF_RET_5",
        "HF_RET_15",
        "HF_MOM_30",
        "HF_MOM_60",
        "HF_RANGE_ATR",
        "HF_BODY_SIGNED",
        "HF_UPPER_WICK",
        "HF_LOWER_WICK",
        "HF_RET_STD_15",
        "HF_RET_STD_60",
        "HF_RANGE_MEAN_30",
        "HF_VOLA_BURST",
        "HF_RSI_SLOPE_5",
        "TREND_SCORE",
    ]
    df[hf_features] = df[hf_features].shift(1)
    return df, hf_features


def prepare_research_data(use_high_frequency_features=True):
    df, base_features = load_and_prepare_data()
    df = df.copy()
    df["BARRIER_TARGET"] = build_barrier_target(df)
    hf_features = []
    if use_high_frequency_features:
        df, hf_features = add_high_frequency_features(df)
    else:
        trend_cols = [col for col in df.columns if col.endswith("_TREND")]
        df["TREND_SCORE"] = (df[trend_cols] > 0).sum(axis=1).astype(np.float32) if trend_cols else 0.0
        base_features = base_features + ["TREND_SCORE"]

    features = base_features + hf_features
    df = df.iloc[:-HORIZON].dropna(subset=features + ["BARRIER_TARGET", "ATR"])
    return df.reset_index(drop=True), features, base_features


def class_weight(target, positive_class=None):
    if positive_class is None:
        counts = np.bincount(target.astype(np.int64), minlength=3)
        weights = len(target) / (3.0 * np.maximum(counts, 1))
        return weights[target.astype(np.int64)]

    binary = (target == positive_class).astype(np.int8)
    positives = max(int(binary.sum()), 1)
    negatives = max(len(binary) - positives, 1)
    weights = np.where(binary == 1, len(binary) / (2.0 * positives), len(binary) / (2.0 * negatives))
    return weights


def make_classifier(objective, num_class=None, n_estimators=DEFAULT_N_ESTIMATORS):
    params = {
        "objective": objective,
        "tree_method": "hist",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "n_estimators": n_estimators,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 80,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": RANDOM_STATE,
        "verbosity": 0,
    }
    if num_class is not None:
        params["num_class"] = num_class
    return xgb.XGBClassifier(**params)


def train_main_model(train_df, features, n_estimators):
    target = train_df["BARRIER_TARGET"].to_numpy(dtype=np.int8)
    model = make_classifier("multi:softprob", num_class=3, n_estimators=n_estimators)
    model.fit(train_df[features], target, sample_weight=class_weight(target))
    model.set_params(device="cpu")
    return model


def train_binary_model(train_df, features, positive_class, n_estimators):
    target = train_df["BARRIER_TARGET"].to_numpy(dtype=np.int8)
    binary = (target == positive_class).astype(np.int8)
    model = make_classifier("binary:logistic", n_estimators=n_estimators)
    model.fit(train_df[features], binary, sample_weight=class_weight(target, positive_class))
    model.set_params(device="cpu")
    return model


def train_regime_model(train_df, regime_features, n_estimators):
    target = (train_df["BARRIER_TARGET"].to_numpy(dtype=np.int8) == 1).astype(np.int8)
    model = make_classifier("binary:logistic", n_estimators=n_estimators)
    positives = max(int(target.sum()), 1)
    negatives = max(len(target) - positives, 1)
    weights = np.where(target == 1, len(target) / (2.0 * positives), len(target) / (2.0 * negatives))
    model.fit(train_df[regime_features], target, sample_weight=weights)
    model.set_params(device="cpu")
    return model


def predict_positive(model, df, features):
    if hasattr(model, "classes_") and len(model.classes_) == 1:
        return np.full(len(df), float(model.classes_[0]), dtype=np.float32)
    probs = model.predict_proba(df[features])
    if probs.ndim == 1:
        return probs.astype(np.float32)
    class_to_index = {int(cls): idx for idx, cls in enumerate(model.classes_)}
    return probs[:, class_to_index.get(1, probs.shape[1] - 1)].astype(np.float32)


def make_direction_probs(direction_prob, direction):
    probs = np.zeros((len(direction_prob), 3), dtype=np.float32)
    if direction == "long":
        probs[:, 1] = direction_prob
    elif direction == "short":
        probs[:, 2] = direction_prob
    else:
        raise ValueError(direction)
    probs[:, 0] = 1.0 - probs[:, 1] - probs[:, 2]
    return np.clip(probs, 0.0, 1.0)


def make_ensemble_probs(main_probs, long_prob, short_prob, regime_prob):
    probs = np.zeros_like(main_probs, dtype=np.float32)
    probs[:, 1] = main_probs[:, 1] * long_prob * regime_prob
    probs[:, 2] = main_probs[:, 2] * short_prob * (1.0 - (0.25 * regime_prob))
    probs[:, 0] = np.maximum(0.0, 1.0 - probs[:, 1] - probs[:, 2])
    row_sum = probs.sum(axis=1, keepdims=True)
    return probs / np.maximum(row_sum, 1e-9)


def evaluate_df(params, df, probs, entry_quality=None):
    return evaluate(
        params,
        df["CLOSE"].to_numpy(dtype=np.float64),
        df["ATR"].to_numpy(dtype=np.float64),
        probs,
        hours=df["TIME_DT"].dt.hour.to_numpy(dtype=np.int16),
        weekdays=df["TIME_DT"].dt.dayofweek.to_numpy(dtype=np.int8),
        dates=df["TIME_DT"].dt.date.to_numpy(),
        months=df["TIME_DT"].dt.month.to_numpy(dtype=np.int8),
        rsi_values=df["M1_RSI"].to_numpy(dtype=np.float64),
        vola_ratio_values=df["VOLA_RATIO"].to_numpy(dtype=np.float64),
        trend_score_values=df["TREND_SCORE"].to_numpy(dtype=np.float64),
        entry_quality=entry_quality,
    )


def trades_per_year(stats, df):
    elapsed_years = max(
        (df["TIME_DT"].iloc[-1] - df["TIME_DT"].iloc[0]).total_seconds() / (365.25 * 86400.0),
        1e-9,
    )
    return stats["trades"] / elapsed_years


def research_params(**overrides):
    params = dict(FINAL_PARAMS)
    params.update(overrides)
    return params


def select_train_rows(df, max_train_rows):
    if max_train_rows is None or len(df) <= max_train_rows:
        return df
    return df.iloc[-max_train_rows:].copy()


def train_bundle(train_df, features, n_estimators, max_train_rows=None):
    train_df = select_train_rows(train_df, max_train_rows)
    regime_features = [
        feature
        for feature in [
            "ATR",
            "VOLA_RATIO",
            "M1_RSI",
            "HOUR_SIN",
            "HOUR_COS",
            "DAY_OF_WEEK",
            "TREND_SCORE",
            "HF_RET_5",
            "HF_MOM_30",
            "HF_RANGE_ATR",
            "HF_VOLA_BURST",
            "HF_RSI_SLOPE_5",
        ]
        if feature in features
    ]
    return ModelBundle(
        main=train_main_model(train_df, features, n_estimators),
        long_model=train_binary_model(train_df, features, 1, n_estimators),
        short_model=train_binary_model(train_df, features, 2, n_estimators),
        regime_model=train_regime_model(train_df, regime_features, n_estimators),
        features=features,
        regime_features=regime_features,
    )


def evaluate_bundle(bundle, test_df):
    main_probs = bundle.main.predict_proba(test_df[bundle.features]).astype(np.float32)
    long_prob = predict_positive(bundle.long_model, test_df, bundle.features)
    short_prob = predict_positive(bundle.short_model, test_df, bundle.features)
    regime_prob = predict_positive(bundle.regime_model, test_df, bundle.regime_features)

    long_params = research_params(direction_mode="long", threshold=0.68)
    short_params = research_params(
        direction_mode="short",
        threshold=0.70,
        allowed_entry_weekdays=[0, 1, 2, 3, 4],
        excluded_rsi_ranges=[],
        risk_per_trade=0.018,
    )
    ensemble_params = research_params(
        threshold=0.12,
        edge_threshold=0.0,
        min_entry_quality=0.20,
        risk_per_trade=0.030,
    )

    long_stats = evaluate_df(long_params, test_df, make_direction_probs(long_prob, "long"))
    short_stats = evaluate_df(short_params, test_df, make_direction_probs(short_prob, "short"))
    ensemble_probs = make_ensemble_probs(main_probs, long_prob, short_prob, regime_prob)
    entry_quality = (0.45 * long_prob) + (0.35 * regime_prob) + (0.20 * main_probs[:, 1])
    ensemble_stats = evaluate_df(ensemble_params, test_df, ensemble_probs, entry_quality=entry_quality)
    main_stats = evaluate_df(research_params(), test_df, main_probs)
    return {
        "main": main_stats,
        "long_binary": long_stats,
        "short_binary": short_stats,
        "ensemble": ensemble_stats,
    }


def print_eval_block(label, stats_by_name, test_df):
    print(label)
    for name, stats in stats_by_name.items():
        print(f"   {name} trades/year={trades_per_year(stats, test_df):.1f} | " + format_stats(name, stats))


def run_final_split(df, features, n_estimators, max_train_rows):
    train_end = int(len(df) * 0.85)
    train_df = df.iloc[: max(0, train_end - HORIZON)].copy()
    test_df = df.iloc[train_end:].copy().reset_index(drop=True)
    print(f"Final split rows | train={len(train_df):,} test={len(test_df):,}")
    bundle = train_bundle(train_df, features, n_estimators, max_train_rows=max_train_rows)
    stats = evaluate_bundle(bundle, test_df)
    print_eval_block("Final split model comparison:", stats, test_df)


def run_walk_forward(df, features, n_estimators, max_train_rows, quick):
    years = sorted(df["TIME_DT"].dt.year.unique())
    test_years = [year for year in years if year >= 2020]
    if quick:
        test_years = test_years[-2:]
    print("Walk-forward:")
    aggregate = {name: [] for name in ["main", "long_binary", "short_binary", "ensemble"]}
    for test_year in test_years:
        train_df = df[df["TIME_DT"].dt.year < test_year].copy()
        test_df = df[df["TIME_DT"].dt.year == test_year].copy().reset_index(drop=True)
        if len(train_df) < 100_000 or len(test_df) < 10_000:
            continue
        train_df = train_df.iloc[: max(0, len(train_df) - HORIZON)].copy()
        print(f"Fold {test_year} | train={len(train_df):,} test={len(test_df):,}")
        bundle = train_bundle(train_df, features, n_estimators, max_train_rows=max_train_rows)
        stats = evaluate_bundle(bundle, test_df)
        print_eval_block(f"Fold {test_year} results:", stats, test_df)
        for name, item in stats.items():
            aggregate[name].append(item)

    print("Walk-forward summary:")
    for name, items in aggregate.items():
        if not items:
            continue
        profitable = sum(1 for item in items if item["pnl"] > 0)
        avg_roi = float(np.mean([item["roi"] for item in items]))
        worst_dd = float(np.min([item["max_drawdown_pct"] for item in items]))
        avg_pf = float(np.mean([item["profit_factor"] for item in items if np.isfinite(item["profit_factor"])]))
        print(
            f"   {name}: profitable_folds={profitable}/{len(items)}, "
            f"avg_roi={avg_roi:.2%}, worst_dd={worst_dd:.2%}, avg_pf={avg_pf:.2f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use fewer trees and last rows for faster experiments.")
    parser.add_argument("--skip-walk-forward", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=None)
    args = parser.parse_args()

    n_estimators = QUICK_N_ESTIMATORS if args.quick else DEFAULT_N_ESTIMATORS
    max_train_rows = args.max_train_rows
    if args.quick and max_train_rows is None:
        max_train_rows = 600_000

    df, features, _ = prepare_research_data(use_high_frequency_features=True)
    print(f"Prepared research data rows={len(df):,} features={len(features)} n_estimators={n_estimators}")
    run_final_split(df, features, n_estimators, max_train_rows)
    if not args.skip_walk_forward:
        run_walk_forward(df, features, n_estimators, max_train_rows, args.quick)


if __name__ == "__main__":
    main()
