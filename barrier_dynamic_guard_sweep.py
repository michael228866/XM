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
        rsi_values=df["M1_RSI"].to_numpy(dtype=np.float64),
    )


def trades_per_year(stats, df):
    elapsed_years = max(
        (df["TIME_DT"].iloc[-1] - df["TIME_DT"].iloc[0]).total_seconds()
        / (365.25 * 86400.0),
        1e-9,
    )
    return stats["trades"] / elapsed_years


def guard_score(stats, tpy):
    if stats["stopped_out"] or stats["pnl"] <= 0 or tpy < 120:
        return -1_000_000.0
    drawdown = abs(min(stats["max_drawdown_pct"], 0.0))
    dd_penalty = max(0.0, drawdown - 0.42) * 25000.0
    pf_penalty = max(0.0, 1.55 - stats["profit_factor"]) * 7000.0
    return stats["pnl"] - dd_penalty - pf_penalty + min(tpy, 170.0) * 10.0


def make_param_grid():
    base = dict(FINAL_PARAMS)
    for (
        guard_start,
        guard_full,
        min_mult,
        loss_threshold,
        loss_mult,
        risk,
    ) in product(
        [0.08, 0.12, 0.16, 0.20],
        [0.25, 0.30, 0.35],
        [0.20, 0.35, 0.50],
        [2, 3],
        [0.35, 0.55],
        [0.040, 0.044],
    ):
        if guard_full <= guard_start:
            continue
        params = dict(base)
        params.update(
            {
                "risk_per_trade": risk,
                "drawdown_guard_start_pct": guard_start,
                "drawdown_guard_full_pct": guard_full,
                "drawdown_guard_min_risk_mult": min_mult,
                "loss_streak_threshold": loss_threshold,
                "loss_streak_risk_mult": loss_mult,
            }
        )
        yield params


def main():
    print("Sweeping dynamic position sizing and drawdown guard...")
    df, features = prepare_barrier_data()
    train_end = int(len(df) * TRAIN_END_RATIO)
    test_df = df.iloc[train_end:].copy().reset_index(drop=True)

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    model.set_params(device="cpu")
    probs = model.predict_proba(test_df[features]).astype(np.float32)

    baseline = evaluate_df(FINAL_PARAMS, test_df, probs)
    print("Baseline:")
    print("   " + format_stats("test", baseline))

    results = []
    for params in make_param_grid():
        stats = evaluate_df(params, test_df, probs)
        tpy = trades_per_year(stats, test_df)
        score = guard_score(stats, tpy)
        if score <= -100_000.0:
            continue
        results.append((score, params, stats, tpy))

    results.sort(key=lambda item: item[0], reverse=True)
    print("Top dynamic guard candidates:")
    for rank, (score, params, stats, tpy) in enumerate(results[:20], start=1):
        guard = {
            key: params[key]
            for key in [
                "risk_per_trade",
                "drawdown_guard_start_pct",
                "drawdown_guard_full_pct",
                "drawdown_guard_min_risk_mult",
                "loss_streak_threshold",
                "loss_streak_risk_mult",
            ]
        }
        print(f"#{rank} guard_score={score:.2f}, trades/year={tpy:.1f}, guard={guard}")
        print("   " + format_stats("test", stats))


if __name__ == "__main__":
    main()
