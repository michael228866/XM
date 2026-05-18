import os
from itertools import product

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import xgboost as xgb

from barrier_classifier_strategy import evaluate
from barrier_final_train import FINAL_PARAMS, MODEL_PATH, TRAIN_END_RATIO, prepare_barrier_data
from drl_train_candidate import format_stats


MIN_TRADES_PER_YEAR = 120.0
MAX_ACCEPTABLE_DRAWDOWN_PCT = -0.30
EXTRA_COST_POINTS = 5.0


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


def trades_per_year(stats, df):
    elapsed_years = max(
        (df["TIME_DT"].iloc[-1] - df["TIME_DT"].iloc[0]).total_seconds()
        / (365.25 * 86400.0),
        1e-9,
    )
    return stats["trades"] / elapsed_years


def growth_score(stats, tpy):
    if stats["stopped_out"] or stats["pnl"] <= 0:
        return -1_000_000.0
    if tpy < MIN_TRADES_PER_YEAR:
        return -500_000.0 - ((MIN_TRADES_PER_YEAR - tpy) * 2000.0)

    dd_pct = abs(min(stats["max_drawdown_pct"], 0.0))
    dd_penalty = dd_pct * 9000.0
    if stats["max_drawdown_pct"] < MAX_ACCEPTABLE_DRAWDOWN_PCT:
        dd_penalty += (abs(stats["max_drawdown_pct"]) - abs(MAX_ACCEPTABLE_DRAWDOWN_PCT)) * 20000.0

    pf_penalty = max(0.0, 1.45 - stats["profit_factor"]) * 7000.0
    loss_streak_penalty = max(0, stats["max_consecutive_losses"] - 5) * 700.0
    frequency_bonus = min(tpy, 220.0) * 8.0
    return stats["pnl"] - dd_penalty - pf_penalty - loss_streak_penalty + frequency_bonus


def make_param_grid():
    base = dict(FINAL_PARAMS)
    for (
        threshold,
        edge_threshold,
        tp_atr,
        sl_atr,
        risk,
        max_daily_loss,
        max_daily_trades,
    ) in product(
        [0.515, 0.525],
        [0.0],
        [1.0, 1.1],
        [1.8, 2.0],
        [0.028, 0.030, 0.031, 0.032, 0.033, 0.034, 0.036, 0.040, 0.044, 0.050],
        [0.05, 0.06],
        [3, None],
    ):
        params = dict(base)
        params.update(
            {
                "threshold": threshold,
                "edge_threshold": edge_threshold,
                "tp_atr": tp_atr,
                "sl_atr": sl_atr,
                "risk_per_trade": risk,
                "max_daily_loss_pct": max_daily_loss,
                "max_daily_trades": max_daily_trades,
                "extra_cost_points": EXTRA_COST_POINTS,
            }
        )
        yield params


def main():
    print("Searching growth frontier candidates...")
    df, features = prepare_barrier_data()
    train_end = int(len(df) * TRAIN_END_RATIO)
    test_df = df.iloc[train_end:].copy().reset_index(drop=True)

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    model.set_params(device="cpu")
    probs = model.predict_proba(test_df[features]).astype(np.float32)

    results = []
    for params in make_param_grid():
        stats = evaluate_df(params, test_df, probs)
        tpy = trades_per_year(stats, test_df)
        score = growth_score(stats, tpy)
        if score <= -100_000.0:
            continue
        results.append((score, params, stats, tpy))

    results.sort(key=lambda item: item[0], reverse=True)
    print("Top growth frontier candidates:")
    for rank, (score, params, stats, tpy) in enumerate(results[:20], start=1):
        print(
            f"#{rank} growth_score={score:.2f}, trades/year={tpy:.1f}, "
            f"balance={stats['balance']:.2f}, params={params}"
        )
        print("   " + format_stats("test", stats))

    print("Top balance candidates:")
    balance_ranked = sorted(results, key=lambda item: item[2]["balance"], reverse=True)
    for rank, (score, params, stats, tpy) in enumerate(balance_ranked[:12], start=1):
        print(
            f"#{rank} growth_score={score:.2f}, trades/year={tpy:.1f}, "
            f"balance={stats['balance']:.2f}, params={params}"
        )
        print("   " + format_stats("test", stats))


if __name__ == "__main__":
    main()
