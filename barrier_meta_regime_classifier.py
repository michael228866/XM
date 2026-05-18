import os
from itertools import product

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import torch
import xgboost as xgb

from barrier_classifier_strategy import HORIZON, build_barrier_target, build_profit_sample_weight, evaluate
from barrier_final_train import FINAL_PARAMS
from drl_train_candidate import format_stats
from drl_trading_v2 import load_and_prepare_data


BASE_TRAIN_END_RATIO = 0.70
META_TRAIN_END_RATIO = 0.85
BASE_THRESHOLD = 0.525


def add_regime_features(df):
    df = df.copy()
    close = df["CLOSE"]
    high = df["HIGH"]
    low = df["LOW"]
    open_ = df["OPEN"]
    candle_range = (high - low).replace(0, np.nan)
    trend_cols = [col for col in df.columns if col.endswith("_TREND")]

    df["REG_TREND_SCORE"] = (df[trend_cols] > 0).sum(axis=1).astype(np.float32) if trend_cols else 0.0
    df["REG_RET_5"] = close.pct_change(5)
    df["REG_RET_15"] = close.pct_change(15)
    df["REG_RET_60"] = close.pct_change(60)
    df["REG_RANGE_ATR"] = (high - low) / (df["ATR"] + 1e-6)
    df["REG_BODY_SIGNED"] = (close - open_) / (candle_range + 1e-6)
    df["REG_RET_STD_30"] = close.pct_change().rolling(30).std()
    df["REG_RET_STD_120"] = close.pct_change().rolling(120).std()
    df["REG_VOLA_BURST"] = df["REG_RET_STD_30"] / (df["REG_RET_STD_120"] + 1e-9)
    df["REG_ATR_SLOPE_60"] = df["ATR"].pct_change(60)
    df["REG_RSI_SLOPE_10"] = df["M1_RSI"].diff(10)
    df["REG_HOUR"] = df["TIME_DT"].dt.hour.astype(np.float32) / 23.0
    df["REG_MONTH"] = df["TIME_DT"].dt.month.astype(np.float32) / 12.0

    regime_features = [
        "REG_TREND_SCORE",
        "REG_RET_5",
        "REG_RET_15",
        "REG_RET_60",
        "REG_RANGE_ATR",
        "REG_BODY_SIGNED",
        "REG_RET_STD_30",
        "REG_RET_STD_120",
        "REG_VOLA_BURST",
        "REG_ATR_SLOPE_60",
        "REG_RSI_SLOPE_10",
        "REG_HOUR",
        "REG_MONTH",
        "VOLA_RATIO",
        "M1_RSI",
        "ATR",
        "HOUR_SIN",
        "HOUR_COS",
        "DAY_OF_WEEK",
    ]
    df[regime_features] = df[regime_features].shift(1)
    return df, regime_features


def prepare_data():
    df, features = load_and_prepare_data()
    df = df.copy()
    df["BARRIER_TARGET"] = build_barrier_target(df)
    df, regime_features = add_regime_features(df)
    df = df.iloc[:-HORIZON].dropna(subset=features + regime_features + ["BARRIER_TARGET", "ATR"])
    return df.reset_index(drop=True), features, regime_features


def train_base_model(train_df, features):
    target = train_df["BARRIER_TARGET"].to_numpy(dtype=np.int8)
    sample_weight = build_profit_sample_weight(train_df, target)
    sample_weight = np.nan_to_num(sample_weight, nan=1.0, posinf=1.0, neginf=1.0)
    sample_weight = np.maximum(sample_weight, 1e-6)
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        device="cuda" if torch.cuda.is_available() else "cpu",
        n_estimators=320,
        learning_rate=0.045,
        max_depth=5,
        min_child_weight=80,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbosity=0,
    )
    model.fit(train_df[features], train_df["BARRIER_TARGET"], sample_weight=sample_weight)
    model.set_params(device="cpu")
    return model


def make_meta_frame(df, base_probs, regime_features):
    meta = df[regime_features].copy()
    buy_prob = base_probs[:, 1]
    sell_prob = base_probs[:, 2]
    meta["BASE_BUY_PROB"] = buy_prob
    meta["BASE_SELL_PROB"] = sell_prob
    meta["BASE_EDGE"] = buy_prob - sell_prob
    meta["BASE_CONF"] = np.maximum(buy_prob, sell_prob)
    meta["BASE_NO_TRADE_PROB"] = base_probs[:, 0]
    return meta


def train_meta_regime(meta_df, target, candidate_mask):
    binary = (target == 1).astype(np.int8)
    positives = max(int(binary.sum()), 1)
    negatives = max(len(binary) - positives, 1)
    sample_weight = np.where(binary == 1, len(binary) / (2.0 * positives), len(binary) / (2.0 * negatives))
    sample_weight = sample_weight.astype(np.float64)
    sample_weight[candidate_mask] *= 2.0
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        device="cuda" if torch.cuda.is_available() else "cpu",
        n_estimators=260,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=60,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbosity=0,
    )
    model.fit(meta_df, binary, sample_weight=sample_weight)
    model.set_params(device="cpu")
    return model


def predict_meta(model, meta_df):
    probs = model.predict_proba(meta_df)
    class_to_index = {int(cls): idx for idx, cls in enumerate(model.classes_)}
    return probs[:, class_to_index.get(1, probs.shape[1] - 1)].astype(np.float32)


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
        entry_quality=entry_quality,
    )


def trades_per_year(stats, df):
    elapsed_years = max(
        (df["TIME_DT"].iloc[-1] - df["TIME_DT"].iloc[0]).total_seconds()
        / (365.25 * 86400.0),
        1e-9,
    )
    return stats["trades"] / elapsed_years


def score(stats, df):
    tpy = trades_per_year(stats, df)
    if stats["stopped_out"] or stats["pnl"] <= 0 or tpy < 80:
        return -1_000_000.0 + stats["pnl"]
    dd = abs(min(stats["max_drawdown_pct"], 0.0))
    dd_penalty = max(0.0, dd - 0.30) * 20_000.0
    pf_penalty = max(0.0, 1.55 - stats["profit_factor"]) * 8_000.0
    return stats["pnl"] - dd_penalty - pf_penalty + min(tpy, 160.0) * 10.0


def make_params():
    base = dict(FINAL_PARAMS)
    for threshold, min_quality, risk, max_daily_trades in product(
        [0.50, 0.515, 0.525],
        [0.48, 0.52, 0.56, 0.60],
        [0.028, 0.034, 0.040],
        [None, 3],
    ):
        params = dict(base)
        params.update(
            {
                "threshold": threshold,
                "risk_per_trade": risk,
                "min_entry_quality": min_quality,
                "max_daily_trades": max_daily_trades,
            }
        )
        yield params


def main():
    print("Training stronger meta-regime classifier...")
    df, features, regime_features = prepare_data()
    base_end = int(len(df) * BASE_TRAIN_END_RATIO)
    meta_end = int(len(df) * META_TRAIN_END_RATIO)

    base_train = df.iloc[: max(0, base_end - HORIZON)].copy()
    meta_train = df.iloc[base_end:meta_end].copy().reset_index(drop=True)
    test_df = df.iloc[meta_end:].copy().reset_index(drop=True)
    print(
        f"Rows | base_train={len(base_train):,} meta_train={len(meta_train):,} "
        f"test={len(test_df):,}"
    )

    base_model = train_base_model(base_train, features)
    meta_train_probs = base_model.predict_proba(meta_train[features]).astype(np.float32)
    test_probs = base_model.predict_proba(test_df[features]).astype(np.float32)

    meta_train_frame = make_meta_frame(meta_train, meta_train_probs, regime_features)
    test_meta_frame = make_meta_frame(test_df, test_probs, regime_features)
    candidate_mask = (
        (meta_train_probs[:, 1] >= BASE_THRESHOLD)
        & (meta_train_probs[:, 1] >= meta_train_probs[:, 2])
    )
    meta_model = train_meta_regime(
        meta_train_frame,
        meta_train["BARRIER_TARGET"].to_numpy(dtype=np.int8),
        candidate_mask,
    )
    meta_train_quality = predict_meta(meta_model, meta_train_frame)
    test_quality = predict_meta(meta_model, test_meta_frame)

    baseline_stats = evaluate_df(FINAL_PARAMS, test_df, test_probs)
    print("Baseline base model:")
    print(f"   trades/year={trades_per_year(baseline_stats, test_df):.1f} | " + format_stats("test", baseline_stats))

    results = []
    for params in make_params():
        meta_stats = evaluate_df(params, meta_train, meta_train_probs, meta_train_quality)
        meta_score = score(meta_stats, meta_train)
        if meta_score <= -100_000.0:
            continue
        test_stats = evaluate_df(params, test_df, test_probs, test_quality)
        results.append((meta_score, params, meta_stats, test_stats))

    results.sort(key=lambda item: item[0], reverse=True)
    print("Top meta-regime candidates:")
    for rank, (meta_score, params, meta_stats, test_stats) in enumerate(results[:15], start=1):
        summary_params = {
            key: params[key]
            for key in ["threshold", "min_entry_quality", "risk_per_trade", "max_daily_trades"]
        }
        print(f"#{rank} meta_score={meta_score:.2f}, params={summary_params}")
        print(f"   meta_train trades/year={trades_per_year(meta_stats, meta_train):.1f} | " + format_stats("meta", meta_stats))
        print(f"   test trades/year={trades_per_year(test_stats, test_df):.1f} | " + format_stats("test", test_stats))


if __name__ == "__main__":
    main()
