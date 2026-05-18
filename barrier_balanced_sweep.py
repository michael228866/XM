import os
from itertools import product

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import xgboost as xgb

from barrier_classifier_strategy import evaluate
from barrier_final_train import FINAL_PARAMS, MODEL_PATH, TRAIN_END_RATIO, prepare_barrier_data
from drl_train_candidate import format_stats


MIN_FULL_TRADES = 260
MAX_DRAWDOWN_LIMIT = -320.0


def evaluate_df(params, df, probs):
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


def balanced_score(full_stats, dev_stats, holdout_stats):
    if any(stats["stopped_out"] or stats["pnl"] <= 0 for stats in [full_stats, dev_stats, holdout_stats]):
        return -1_000_000.0
    if full_stats["trades"] < MIN_FULL_TRADES or full_stats["max_drawdown"] < MAX_DRAWDOWN_LIMIT:
        return -500_000.0
    balance_penalty = abs(dev_stats["roi"] - holdout_stats["roi"]) * 900.0
    drawdown_penalty = abs(min(full_stats["max_drawdown"], 0.0)) * 2.4
    win_penalty = max(0.0, 0.70 - full_stats["win_rate"]) * 10_000.0
    return full_stats["pnl"] - drawdown_penalty - win_penalty - balance_penalty


def make_param_grid():
    base = dict(FINAL_PARAMS)
    hour_sets = [
        base["allowed_entry_hours"],
    ]
    weekday_sets = [
        base["allowed_entry_weekdays"],
    ]
    rsi_filters = [
        base["excluded_rsi_ranges"],
        [(34.0, 46.0)],
    ]
    for threshold, tp_atr, sl_atr, risk, daily_loss, hours, weekdays, rsi_filter in product(
        [0.525, 0.535],
        [1.1, 1.2],
        [2.0, 2.2],
        [0.024, 0.026, 0.028],
        [0.05, 0.07],
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
                "max_daily_loss_pct": daily_loss,
                "allowed_entry_hours": hours,
                "allowed_entry_weekdays": weekdays,
                "excluded_rsi_ranges": rsi_filter,
            }
        )
        yield params


def main():
    print("Loading final barrier model for balanced sweep...")
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

    baseline_full = evaluate_df(FINAL_PARAMS, test_df, test_probs)
    print("Current params:")
    print("   " + format_stats("full", baseline_full))

    results = []
    for params in make_param_grid():
        dev_stats = evaluate_df(params, dev_df, dev_probs)
        if dev_stats["trades"] < 70 or dev_stats["pnl"] <= 0:
            continue
        holdout_stats = evaluate_df(params, holdout_df, holdout_probs)
        if holdout_stats["trades"] < 170 or holdout_stats["pnl"] <= 0:
            continue
        full_stats = evaluate_df(params, test_df, test_probs)
        score = balanced_score(full_stats, dev_stats, holdout_stats)
        if score <= -100_000.0:
            continue
        results.append((score, params, dev_stats, holdout_stats, full_stats))

    results.sort(key=lambda item: item[0], reverse=True)
    print("Top balanced candidates:")
    for rank, (score, params, dev_stats, holdout_stats, full_stats) in enumerate(
        results[:12], start=1
    ):
        print(f"#{rank} balanced_score={score:.2f} params={params}")
        print("   " + format_stats("dev", dev_stats))
        print("   " + format_stats("holdout", holdout_stats))
        print("   " + format_stats("full", full_stats))


if __name__ == "__main__":
    main()
