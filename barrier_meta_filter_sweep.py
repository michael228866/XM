import os
from itertools import product

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import xgboost as xgb

from barrier_classifier_strategy import evaluate
from barrier_final_train import FINAL_PARAMS, MODEL_PATH, TRAIN_END_RATIO, prepare_barrier_data
from drl_train_candidate import format_stats


MIN_SEGMENT_TRADES_PER_YEAR = 70.0
MIN_FULL_TRADES_PER_YEAR = 110.0


def add_meta_features(df):
    df = df.copy()
    trend_cols = [col for col in df.columns if col.endswith("_TREND")]
    if trend_cols:
        df["TREND_SCORE"] = (df[trend_cols] > 0).sum(axis=1).astype(np.float32)
    else:
        df["TREND_SCORE"] = 0.0
    return df


def evaluate_df(params, df, probs):
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
    )


def trades_per_year(stats, df):
    elapsed_years = max(
        (df["TIME_DT"].iloc[-1] - df["TIME_DT"].iloc[0]).total_seconds()
        / (365.25 * 86400.0),
        1e-9,
    )
    return stats["trades"] / elapsed_years


def score(stats, tpy):
    if stats["stopped_out"] or stats["pnl"] <= 0:
        return -1_000_000.0 + stats["pnl"]
    if tpy < MIN_SEGMENT_TRADES_PER_YEAR:
        return -250_000.0 - ((MIN_SEGMENT_TRADES_PER_YEAR - tpy) * 1500.0)

    dd = abs(min(stats["max_drawdown_pct"], 0.0))
    dd_penalty = max(0.0, dd - 0.42) * 20_000.0
    pf_penalty = max(0.0, 1.55 - stats["profit_factor"]) * 8_000.0
    loss_streak_penalty = max(0, stats["max_consecutive_losses"] - 4) * 750.0
    return stats["pnl"] - dd_penalty - pf_penalty - loss_streak_penalty + min(tpy, 170.0) * 10.0


def make_filter_grid():
    base = dict(FINAL_PARAMS)
    for (
        min_vola,
        max_vola,
        min_trend,
        allowed_months,
        rsi_filter,
        threshold,
        risk,
    ) in product(
        [None, 0.75],
        [1.8, 2.2, None],
        [None, 5, 7],
        [
            None,
            [1, 2, 3, 4, 5, 6, 9, 10, 11, 12],
        ],
        [
            base["excluded_rsi_ranges"],
            [],
            [(36.0, 48.0)],
        ],
        [0.525],
        [0.040, 0.044],
    ):
        if min_vola is not None and max_vola is not None and min_vola >= max_vola:
            continue
        params = dict(base)
        params.update(
            {
                "threshold": threshold,
                "risk_per_trade": risk,
                "min_vola_ratio": min_vola,
                "max_vola_ratio": max_vola,
                "min_trend_score": min_trend,
                "allowed_entry_months": allowed_months,
                "excluded_rsi_ranges": rsi_filter,
            }
        )
        yield params


def main():
    print("Sweeping meta filters...")
    df, features = prepare_barrier_data()
    df = add_meta_features(df)
    train_end = int(len(df) * TRAIN_END_RATIO)
    test_df = df.iloc[train_end:].copy().reset_index(drop=True)
    split = len(test_df) // 2
    dev_df = test_df.iloc[:split].copy().reset_index(drop=True)
    holdout_df = test_df.iloc[split:].copy().reset_index(drop=True)

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    model.set_params(device="cpu")
    test_probs = model.predict_proba(test_df[features]).astype(np.float32)
    dev_probs = test_probs[:split]
    holdout_probs = test_probs[split:]

    baseline = evaluate_df(FINAL_PARAMS, test_df, test_probs)
    baseline_dev = evaluate_df(FINAL_PARAMS, dev_df, dev_probs)
    baseline_holdout = evaluate_df(FINAL_PARAMS, holdout_df, holdout_probs)
    print("Baseline:")
    print(f"   trades/year={trades_per_year(baseline, test_df):.1f}")
    print("   " + format_stats("full", baseline))
    print(f"   dev trades/year={trades_per_year(baseline_dev, dev_df):.1f} | " + format_stats("dev", baseline_dev))
    print(
        f"   holdout trades/year={trades_per_year(baseline_holdout, holdout_df):.1f} | "
        + format_stats("holdout", baseline_holdout)
    )

    results = []
    for params in make_filter_grid():
        dev_stats = evaluate_df(params, dev_df, dev_probs)
        dev_tpy = trades_per_year(dev_stats, dev_df)
        if score(dev_stats, dev_tpy) <= -100_000.0:
            continue

        holdout_stats = evaluate_df(params, holdout_df, holdout_probs)
        holdout_tpy = trades_per_year(holdout_stats, holdout_df)
        if score(holdout_stats, holdout_tpy) <= -100_000.0:
            continue

        full_stats = evaluate_df(params, test_df, test_probs)
        full_tpy = trades_per_year(full_stats, test_df)
        if full_tpy < MIN_FULL_TRADES_PER_YEAR or full_stats["pnl"] <= 0:
            continue
        combined = score(dev_stats, dev_tpy) + score(holdout_stats, holdout_tpy)
        results.append((combined, params, dev_stats, dev_tpy, holdout_stats, holdout_tpy, full_stats, full_tpy))

    results.sort(key=lambda item: item[0], reverse=True)
    print("Top meta-filter candidates:")
    for rank, (
        combined,
        params,
        dev_stats,
        dev_tpy,
        holdout_stats,
        holdout_tpy,
        full_stats,
        full_tpy,
    ) in enumerate(results[:15], start=1):
        meta = {
            key: params.get(key)
            for key in [
                "threshold",
                "risk_per_trade",
                "min_vola_ratio",
                "max_vola_ratio",
                "min_trend_score",
                "allowed_entry_months",
                "excluded_rsi_ranges",
            ]
        }
        print(f"#{rank} combined_score={combined:.2f}, meta={meta}")
        print(f"   dev trades/year={dev_tpy:.1f} | " + format_stats("dev", dev_stats))
        print(f"   holdout trades/year={holdout_tpy:.1f} | " + format_stats("holdout", holdout_stats))
        print(f"   full trades/year={full_tpy:.1f} | " + format_stats("full", full_stats))


if __name__ == "__main__":
    main()
