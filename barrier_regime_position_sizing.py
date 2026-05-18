import os
from itertools import product

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import xgboost as xgb

from barrier_classifier_strategy import evaluate
from barrier_final_train import (
    FINAL_PARAMS,
    MODEL_PATH,
    MODEL_SELECTION_END_RATIO,
    TRAIN_END_RATIO,
    prepare_barrier_data,
)
from barrier_meta_regime_classifier import (
    BASE_THRESHOLD,
    HORIZON,
    make_meta_frame,
    predict_meta,
    train_base_model,
    train_meta_regime,
)
from drl_train_candidate import format_stats


RECOMMENDED_RISK_PER_TRADE = 0.028
RECOMMENDED_RULE = (0.40, 0.56, 0.72, 1.00, 1.45, 1.65)


def evaluate_df(params, df, probs, entry_risk_mult=None):
    return evaluate(
        params,
        df["CLOSE"].to_numpy(dtype=np.float64),
        df["ATR"].to_numpy(dtype=np.float64),
        probs,
        hours=df["TIME_DT"].dt.hour.to_numpy(dtype=np.int16),
        weekdays=df["TIME_DT"].dt.dayofweek.to_numpy(dtype=np.int8),
        dates=df["TIME_DT"].dt.date.to_numpy(),
        rsi_values=df["M1_RSI"].to_numpy(dtype=np.float64),
        entry_risk_mult=entry_risk_mult,
    )


def trades_per_year(stats, df):
    elapsed_years = max(
        (df["TIME_DT"].iloc[-1] - df["TIME_DT"].iloc[0]).total_seconds()
        / (365.25 * 86400.0),
        1e-9,
    )
    return stats["trades"] / elapsed_years


def quality_risk_mult(quality, protect_cut, boost_cut, strong_cut, protect_mult, boost_mult, strong_mult):
    mult = np.ones(len(quality), dtype=np.float32)
    mult[quality < protect_cut] = protect_mult
    mult[quality >= boost_cut] = boost_mult
    mult[quality >= strong_cut] = strong_mult
    return mult


def add_overlay_regime_features(df):
    df = df.copy()
    close = df["CLOSE"]
    high = df["HIGH"]
    low = df["LOW"]
    open_ = df["OPEN"]
    candle_range = (high - low).replace(0, np.nan)
    trend_cols = [col for col in df.columns if col.endswith("_TREND")]

    if trend_cols:
        df["REG_TREND_SCORE"] = (df[trend_cols] > 0).sum(axis=1).astype(np.float32)
    else:
        df["REG_TREND_SCORE"] = 0.0
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
    df["REG_VOLA_RATIO"] = df["VOLA_RATIO"]
    df["REG_M1_RSI"] = df["M1_RSI"]
    df["REG_ATR"] = df["ATR"]
    df["REG_HOUR_SIN"] = df["HOUR_SIN"]
    df["REG_HOUR_COS"] = df["HOUR_COS"]
    df["REG_DAY_OF_WEEK"] = df["DAY_OF_WEEK"]

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
        "REG_VOLA_RATIO",
        "REG_M1_RSI",
        "REG_ATR",
        "REG_HOUR_SIN",
        "REG_HOUR_COS",
        "REG_DAY_OF_WEEK",
    ]
    df[regime_features] = df[regime_features].shift(1)
    return df, regime_features


def score_against_baseline(stats, baseline, df):
    if stats["stopped_out"]:
        return -1_000_000.0 + stats["pnl"]

    tpy = trades_per_year(stats, df)
    base_tpy = trades_per_year(baseline, df)
    dd = abs(min(stats["max_drawdown_pct"], 0.0))
    base_dd = abs(min(baseline["max_drawdown_pct"], 0.0))
    pf = stats["profit_factor"]
    pnl_improvement = stats["pnl"] - baseline["pnl"]

    trade_penalty = max(0.0, base_tpy * 0.85 - tpy) * 25.0
    dd_penalty = max(0.0, dd - max(0.50, base_dd * 1.15)) * 20_000.0
    pf_penalty = max(0.0, baseline["profit_factor"] * 0.95 - pf) * 7_000.0
    streak_penalty = max(0, stats["max_consecutive_losses"] - 4) * 1_200.0
    return pnl_improvement - trade_penalty - dd_penalty - pf_penalty - streak_penalty


def make_param_grid():
    base = dict(FINAL_PARAMS)
    for risk, protect_cut, boost_cut, strong_cut, protect_mult, boost_mult, strong_mult in product(
        [0.026, 0.028, 0.030],
        [0.40, 0.46],
        [0.56, 0.62],
        [0.72, 0.78],
        [0.70, 0.85, 1.00],
        [1.10, 1.20, 1.30],
        [1.35, 1.50],
    ):
        if not (protect_cut < boost_cut < strong_cut):
            continue
        if strong_mult < boost_mult:
            continue
        params = dict(base)
        params["risk_per_trade"] = risk
        yield params, (
            protect_cut,
            boost_cut,
            strong_cut,
            protect_mult,
            boost_mult,
            strong_mult,
        )


def load_final_probs(df, features):
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    model.set_params(device="cpu")
    return model.predict_proba(df[features]).astype(np.float32)


def print_stats(prefix, stats, df):
    print(f"   trades/year={trades_per_year(stats, df):.1f} | " + format_stats(prefix, stats))


def main():
    print("Testing formal-model regime position sizing overlay...")
    df, features = prepare_barrier_data()
    df, regime_features = add_overlay_regime_features(df)
    base_end = int(len(df) * MODEL_SELECTION_END_RATIO)
    final_train_end = int(len(df) * TRAIN_END_RATIO)

    base_train = df.iloc[: max(0, base_end - HORIZON)].copy()
    overlay_train = df.iloc[base_end:final_train_end].copy()
    overlay_train = overlay_train.dropna(subset=regime_features).reset_index(drop=True)
    test_df = df.iloc[final_train_end:].copy().reset_index(drop=True)
    print(
        f"Rows | base_train={len(base_train):,} overlay_train={len(overlay_train):,} "
        f"test={len(test_df):,}"
    )

    validation_base = train_base_model(base_train, features)
    overlay_train_probs = validation_base.predict_proba(overlay_train[features]).astype(np.float32)
    final_test_probs = load_final_probs(test_df, features)

    overlay_meta_frame = make_meta_frame(overlay_train, overlay_train_probs, regime_features)
    test_meta_frame = make_meta_frame(test_df, final_test_probs, regime_features)
    candidate_mask = (
        (overlay_train_probs[:, 1] >= BASE_THRESHOLD)
        & (overlay_train_probs[:, 1] >= overlay_train_probs[:, 2])
    )
    meta_model = train_meta_regime(
        overlay_meta_frame,
        overlay_train["BARRIER_TARGET"].to_numpy(dtype=np.int8),
        candidate_mask,
    )
    overlay_quality = predict_meta(meta_model, overlay_meta_frame)
    test_quality = predict_meta(meta_model, test_meta_frame)

    overlay_baseline = evaluate_df(FINAL_PARAMS, overlay_train, overlay_train_probs)
    test_baseline = evaluate_df(FINAL_PARAMS, test_df, final_test_probs)
    print("Baselines:")
    print_stats("overlay_train baseline", overlay_baseline, overlay_train)
    print_stats("formal test baseline", test_baseline, test_df)

    recommended_params = dict(FINAL_PARAMS)
    recommended_params["risk_per_trade"] = RECOMMENDED_RISK_PER_TRADE
    recommended_overlay_mult = quality_risk_mult(overlay_quality, *RECOMMENDED_RULE)
    recommended_test_mult = quality_risk_mult(test_quality, *RECOMMENDED_RULE)
    recommended_overlay_stats = evaluate_df(
        recommended_params,
        overlay_train,
        overlay_train_probs,
        recommended_overlay_mult,
    )
    recommended_test_stats = evaluate_df(
        recommended_params,
        test_df,
        final_test_probs,
        recommended_test_mult,
    )
    print("Recommended validation-supported overlay:")
    print(f"   risk_per_trade={RECOMMENDED_RISK_PER_TRADE}, rule={RECOMMENDED_RULE}")
    print_stats("overlay_train recommended", recommended_overlay_stats, overlay_train)
    print_stats("formal_test recommended", recommended_test_stats, test_df)

    results = []
    for params, rule in make_param_grid():
        overlay_mult = quality_risk_mult(overlay_quality, *rule)
        overlay_stats = evaluate_df(params, overlay_train, overlay_train_probs, overlay_mult)
        overlay_score = score_against_baseline(overlay_stats, overlay_baseline, overlay_train)
        if overlay_score <= -100_000.0:
            continue

        test_mult = quality_risk_mult(test_quality, *rule)
        test_stats = evaluate_df(params, test_df, final_test_probs, test_mult)
        test_score = score_against_baseline(test_stats, test_baseline, test_df)
        results.append((overlay_score, test_score, params, rule, overlay_stats, test_stats))

    results.sort(key=lambda item: item[0], reverse=True)
    print("Top validation-selected overlays:")
    for rank, (overlay_score, test_score, params, rule, overlay_stats, test_stats) in enumerate(results[:20], start=1):
        summary = {
            "risk_per_trade": params["risk_per_trade"],
            "rule": rule,
        }
        print(f"#{rank} overlay_score={overlay_score:.2f}, test_score={test_score:.2f}, {summary}")
        print_stats("overlay_train", overlay_stats, overlay_train)
        print_stats("formal_test", test_stats, test_df)

    test_winners = [
        item for item in results
        if item[5]["pnl"] > test_baseline["pnl"]
        and item[5]["profit_factor"] >= test_baseline["profit_factor"] * 0.95
        and abs(min(item[5]["max_drawdown_pct"], 0.0)) <= 0.50
    ]
    test_winners.sort(key=lambda item: item[5]["pnl"], reverse=True)
    print("Test winners over formal baseline:")
    if not test_winners:
        print("   None")
        return
    for rank, (overlay_score, test_score, params, rule, overlay_stats, test_stats) in enumerate(test_winners[:10], start=1):
        summary = {
            "risk_per_trade": params["risk_per_trade"],
            "rule": rule,
            "overlay_score": round(overlay_score, 2),
            "test_score": round(test_score, 2),
        }
        print(f"#{rank} {summary}")
        print_stats("formal_test", test_stats, test_df)


if __name__ == "__main__":
    main()
