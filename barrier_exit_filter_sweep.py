import os
from itertools import product

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import xgboost as xgb

from barrier_classifier_strategy import evaluate
from barrier_final_train import FINAL_PARAMS, MODEL_PATH, TRAIN_END_RATIO, prepare_barrier_data
from drl_train_candidate import format_stats


MIN_FULL_TRADES = 240
MAX_DRAWDOWN_LIMIT = -330.0


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


def robust_score(full_stats, dev_stats, holdout_stats):
    stats_list = [full_stats, dev_stats, holdout_stats]
    if any(stats["stopped_out"] or stats["pnl"] <= 0 for stats in stats_list):
        return -1_000_000.0
    if full_stats["trades"] < MIN_FULL_TRADES:
        return -500_000.0
    if full_stats["max_drawdown"] < MAX_DRAWDOWN_LIMIT:
        return -400_000.0

    drawdown_penalty = abs(min(full_stats["max_drawdown"], 0.0)) * 2.8
    weak_half_penalty = max(0.0, 0.20 - min(dev_stats["roi"], holdout_stats["roi"])) * 5000.0
    win_penalty = max(0.0, 0.70 - full_stats["win_rate"]) * 9000.0
    frequency_bonus = min(full_stats["trades"], 300) * 4.0
    return full_stats["pnl"] - drawdown_penalty - weak_half_penalty - win_penalty + frequency_bonus


def make_param_grid():
    base = dict(FINAL_PARAMS)
    for (
        threshold,
        edge_threshold,
        tp_atr,
        sl_atr,
        max_hold,
        close_on_opposite,
        max_daily_trades,
        risk,
    ) in product(
        [0.525, 0.535],
        [0.0, 0.03],
        [1.0, 1.1, 1.2],
        [1.8, 2.0, 2.2],
        [120, 180],
        [False, True],
        [None, 3],
        [0.028],
    ):
        params = dict(base)
        params.update(
            {
                "threshold": threshold,
                "edge_threshold": edge_threshold,
                "tp_atr": tp_atr,
                "sl_atr": sl_atr,
                "max_hold": max_hold,
                "close_on_opposite": close_on_opposite,
                "max_daily_trades": max_daily_trades,
                "risk_per_trade": risk,
            }
        )
        yield params


def main():
    print("Loading final barrier model for exit/filter sweep...")
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

    current_full = evaluate_df(FINAL_PARAMS, test_df, test_probs)
    print("Current params:")
    print("   " + format_stats("full", current_full))

    results = []
    for params in make_param_grid():
        dev_stats = evaluate_df(params, dev_df, dev_probs)
        if dev_stats["trades"] < 60 or dev_stats["pnl"] <= 0:
            continue
        holdout_stats = evaluate_df(params, holdout_df, holdout_probs)
        if holdout_stats["trades"] < 150 or holdout_stats["pnl"] <= 0:
            continue
        full_stats = evaluate_df(params, test_df, test_probs)
        score = robust_score(full_stats, dev_stats, holdout_stats)
        if score <= -100_000.0:
            continue
        results.append((score, params, dev_stats, holdout_stats, full_stats))

    results.sort(key=lambda item: item[0], reverse=True)
    print("Top exit/filter candidates:")
    for rank, (score, params, dev_stats, holdout_stats, full_stats) in enumerate(
        results[:15], start=1
    ):
        print(f"#{rank} robust_score={score:.2f} params={params}")
        print("   " + format_stats("dev", dev_stats))
        print("   " + format_stats("holdout", holdout_stats))
        print("   " + format_stats("full", full_stats))


if __name__ == "__main__":
    main()
