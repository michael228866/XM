import os
from itertools import product

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import xgboost as xgb

from barrier_classifier_strategy import evaluate
from barrier_final_train import FINAL_PARAMS, MODEL_PATH, TRAIN_END_RATIO, prepare_barrier_data
from drl_train_candidate import format_stats


MIN_DEV_TRADES = 70
MIN_HOLDOUT_TRADES = 70


def score_candidate(stats):
    if stats["stopped_out"] or stats["pnl"] <= 0:
        return -1_000_000.0 + stats["pnl"]
    trade_penalty = max(0, MIN_DEV_TRADES - stats["trades"]) * 1500.0
    drawdown_penalty = abs(min(stats["max_drawdown"], 0.0)) * 0.65
    win_penalty = max(0.0, 0.58 - stats["win_rate"]) * 8000.0
    trade_bonus = min(stats["trades"], 220) * 12.0
    return stats["pnl"] - drawdown_penalty - win_penalty - trade_penalty + trade_bonus


def evaluate_df(params, df, features, probs):
    return evaluate(
        params,
        df["CLOSE"].to_numpy(dtype=np.float64),
        df["ATR"].to_numpy(dtype=np.float64),
        probs,
        hours=df["TIME_DT"].dt.hour.to_numpy(dtype=np.int16),
        weekdays=df["TIME_DT"].dt.dayofweek.to_numpy(dtype=np.int8),
        dates=df["TIME_DT"].dt.date.to_numpy(),
        rsi_values=df["M1_RSI"].to_numpy(dtype=np.float64),
    )


def make_param_grid():
    base = dict(FINAL_PARAMS)
    hour_sets = [
        base["allowed_entry_hours"],
        [0, 1, 3, 8, 9, 11, 12, 13, 17, 19, 20, 21, 22, 23],
    ]
    weekday_sets = [
        base["allowed_entry_weekdays"],
        [0, 1, 2, 3, 4],
    ]
    rsi_filters = [
        base["excluded_rsi_ranges"],
        [],
    ]
    for threshold, tp_atr, sl_atr, risk, hours, weekdays, rsi_filter in product(
        [0.515, 0.525, 0.535, 0.545],
        [1.0, 1.1, 1.2],
        [1.7, 2.0, 2.3],
        [0.030, 0.034, 0.038],
        hour_sets,
        weekday_sets,
        rsi_filters,
    ):
        params = dict(base)
        params.update(
            {
                "threshold": threshold,
                "tp_atr": tp_atr,
                "sl_atr": sl_atr,
                "risk_per_trade": risk,
                "allowed_entry_hours": hours,
                "allowed_entry_weekdays": weekdays,
                "excluded_rsi_ranges": rsi_filter,
            }
        )
        yield params


def main():
    print("Loading final barrier model for parameter sweep...")
    df, features = prepare_barrier_data()
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

    baseline_dev = evaluate_df(FINAL_PARAMS, dev_df, features, dev_probs)
    baseline_holdout = evaluate_df(FINAL_PARAMS, holdout_df, features, holdout_probs)
    baseline_full = evaluate_df(FINAL_PARAMS, test_df, features, test_probs)
    print("Baseline:")
    print("   " + format_stats("dev", baseline_dev))
    print("   " + format_stats("holdout", baseline_holdout))
    print("   " + format_stats("full", baseline_full))

    results = []
    for params in make_param_grid():
        dev_stats = evaluate_df(params, dev_df, features, dev_probs)
        if dev_stats["trades"] < MIN_DEV_TRADES or dev_stats["pnl"] <= 0:
            continue
        results.append((score_candidate(dev_stats), params, dev_stats))

    results.sort(key=lambda item: item[0], reverse=True)
    print("Top candidates:")
    accepted = 0
    for rank, (score, params, dev_stats) in enumerate(results[:40], start=1):
        holdout_stats = evaluate_df(params, holdout_df, features, holdout_probs)
        full_stats = evaluate_df(params, test_df, features, test_probs)
        if holdout_stats["trades"] < MIN_HOLDOUT_TRADES or holdout_stats["pnl"] <= 0:
            continue
        accepted += 1
        print(f"#{accepted} dev_score={score:.2f} params={params}")
        print("   " + format_stats("dev", dev_stats))
        print("   " + format_stats("holdout", holdout_stats))
        print("   " + format_stats("full", full_stats))
        if accepted >= 12:
            break


if __name__ == "__main__":
    main()
