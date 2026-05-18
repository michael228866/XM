import os
from itertools import product

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import xgboost as xgb

from barrier_classifier_strategy import evaluate
from barrier_final_train import FINAL_PARAMS, MODEL_PATH, TRAIN_END_RATIO, prepare_barrier_data
from drl_train_candidate import format_stats


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
    )


def trades_per_year(stats, df):
    elapsed_years = max(
        (df["TIME_DT"].iloc[-1] - df["TIME_DT"].iloc[0]).total_seconds()
        / (365.25 * 86400.0),
        1e-9,
    )
    return stats["trades"] / elapsed_years


def trader_score(stats, tpy, min_tpy=70.0):
    if stats["stopped_out"] or stats["pnl"] <= 0 or tpy < min_tpy:
        return -1_000_000.0 + stats["pnl"]
    dd = abs(min(stats["max_drawdown_pct"], 0.0))
    dd_penalty = max(0.0, dd - 0.38) * 25_000.0
    pf_penalty = max(0.0, 1.55 - stats["profit_factor"]) * 8_000.0
    loss_streak_penalty = max(0, stats["max_consecutive_losses"] - 4) * 1_000.0
    return stats["pnl"] - dd_penalty - pf_penalty - loss_streak_penalty + min(tpy, 160.0) * 12.0


def make_overlay_grid():
    base = dict(FINAL_PARAMS)
    for (
        loss_pause_threshold,
        loss_pause_ticks,
        rolling_window,
        min_pf,
        min_win,
        risk_mult,
        pause_ticks,
        risk,
    ) in product(
        [3, 4],
        [120, 240],
        [20, 30],
        [1.05, 1.15],
        [None, 0.55],
        [0.50, 0.70],
        [0, 120],
        [0.040, 0.044],
    ):
        if loss_pause_threshold is None and loss_pause_ticks:
            continue
        if rolling_window is None and (min_pf is not None or min_win is not None or pause_ticks):
            continue
        params = dict(base)
        params.update(
            {
                "risk_per_trade": risk,
                "loss_streak_pause_threshold": loss_pause_threshold,
                "loss_streak_pause_ticks": loss_pause_ticks,
                "rolling_guard_window": rolling_window,
                "rolling_guard_min_trades": 18,
                "rolling_guard_min_profit_factor": min_pf,
                "rolling_guard_min_win_rate": min_win,
                "rolling_guard_risk_mult": risk_mult,
                "rolling_guard_pause_ticks": pause_ticks,
            }
        )
        yield params


def main():
    print("Sweeping trader-like overlay...")
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

    baseline = evaluate_df(FINAL_PARAMS, test_df, test_probs)
    print("Baseline:")
    print(f"   trades/year={trades_per_year(baseline, test_df):.1f}")
    print("   " + format_stats("full", baseline))

    results = []
    for params in make_overlay_grid():
        dev_stats = evaluate_df(params, dev_df, dev_probs)
        dev_tpy = trades_per_year(dev_stats, dev_df)
        if trader_score(dev_stats, dev_tpy, min_tpy=60.0) <= -100_000.0:
            continue
        holdout_stats = evaluate_df(params, holdout_df, holdout_probs)
        holdout_tpy = trades_per_year(holdout_stats, holdout_df)
        if trader_score(holdout_stats, holdout_tpy, min_tpy=100.0) <= -100_000.0:
            continue
        full_stats = evaluate_df(params, test_df, test_probs)
        full_tpy = trades_per_year(full_stats, test_df)
        if full_tpy < 110.0:
            continue
        combined = trader_score(dev_stats, dev_tpy, min_tpy=60.0) + trader_score(
            holdout_stats,
            holdout_tpy,
            min_tpy=100.0,
        )
        results.append((combined, params, dev_stats, dev_tpy, holdout_stats, holdout_tpy, full_stats, full_tpy))

    results.sort(key=lambda item: item[0], reverse=True)
    print("Top trader overlay candidates:")
    for rank, (combined, params, dev_stats, dev_tpy, holdout_stats, holdout_tpy, full_stats, full_tpy) in enumerate(
        results[:15],
        start=1,
    ):
        overlay = {
            key: params.get(key)
            for key in [
                "risk_per_trade",
                "loss_streak_pause_threshold",
                "loss_streak_pause_ticks",
                "rolling_guard_window",
                "rolling_guard_min_profit_factor",
                "rolling_guard_min_win_rate",
                "rolling_guard_risk_mult",
                "rolling_guard_pause_ticks",
            ]
        }
        print(f"#{rank} combined_score={combined:.2f}, overlay={overlay}")
        print(f"   dev trades/year={dev_tpy:.1f} | " + format_stats("dev", dev_stats))
        print(f"   holdout trades/year={holdout_tpy:.1f} | " + format_stats("holdout", holdout_stats))
        print(f"   full trades/year={full_tpy:.1f} | " + format_stats("full", full_stats))


if __name__ == "__main__":
    main()
