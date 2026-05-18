import os
from itertools import product

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import xgboost as xgb

from barrier_classifier_strategy import evaluate
from barrier_final_train import FINAL_PARAMS, MODEL_PATH, TRAIN_END_RATIO, prepare_barrier_data
from drl_train_candidate import format_stats


MIN_TRADES = 220
MAX_DRAWDOWN_LIMIT = -420.0


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


def risk_score(stats):
    if stats["stopped_out"] or stats["pnl"] <= 0:
        return -1_000_000.0 + stats["pnl"]
    if stats["trades"] < MIN_TRADES:
        return -250_000.0 - ((MIN_TRADES - stats["trades"]) * 1000.0)
    drawdown = abs(min(stats["max_drawdown"], 0.0))
    dd_penalty = drawdown * 2.2
    win_penalty = max(0.0, 0.68 - stats["win_rate"]) * 12000.0
    return stats["pnl"] - dd_penalty - win_penalty + min(stats["trades"], 320) * 6.0


def make_param_grid():
    base = dict(FINAL_PARAMS)
    for threshold, tp_atr, sl_atr, risk, daily_loss, max_daily_trades in product(
        [0.525, 0.535, 0.545, 0.555],
        [1.1, 1.2],
        [2.0, 2.3, 2.6],
        [0.018, 0.022, 0.026, 0.030],
        [0.05, 0.07, 0.10],
        [1, 2, None],
    ):
        params = dict(base)
        params.update(
            {
                "threshold": threshold,
                "tp_atr": tp_atr,
                "sl_atr": sl_atr,
                "risk_per_trade": risk,
                "max_daily_loss_pct": daily_loss,
                "max_daily_trades": max_daily_trades,
            }
        )
        yield params


def main():
    print("Loading final barrier model for low-risk parameter sweep...")
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
    print("Baseline:")
    print("   " + format_stats("full", baseline_full))

    results = []
    for params in make_param_grid():
        dev_stats = evaluate_df(params, dev_df, dev_probs)
        if dev_stats["pnl"] <= 0 or dev_stats["trades"] < 70:
            continue
        holdout_stats = evaluate_df(params, holdout_df, holdout_probs)
        if holdout_stats["pnl"] <= 0 or holdout_stats["trades"] < 70:
            continue
        full_stats = evaluate_df(params, test_df, test_probs)
        if full_stats["max_drawdown"] < MAX_DRAWDOWN_LIMIT:
            continue
        results.append((risk_score(full_stats), params, dev_stats, holdout_stats, full_stats))

    results.sort(key=lambda item: item[0], reverse=True)
    print("Top low-risk candidates:")
    for rank, (score, params, dev_stats, holdout_stats, full_stats) in enumerate(
        results[:12], start=1
    ):
        print(f"#{rank} risk_score={score:.2f} params={params}")
        print("   " + format_stats("dev", dev_stats))
        print("   " + format_stats("holdout", holdout_stats))
        print("   " + format_stats("full", full_stats))


if __name__ == "__main__":
    main()
