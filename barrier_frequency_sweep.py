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


def frequency_score(stats, tpy):
    if stats["stopped_out"] or stats["pnl"] <= 0:
        return -1_000_000.0
    if tpy < 180:
        return -500_000.0 - ((180 - tpy) * 1500.0)
    dd = abs(min(stats["max_drawdown_pct"], 0.0))
    dd_penalty = max(0.0, dd - 0.55) * 20_000.0
    pf_penalty = max(0.0, 1.35 - stats["profit_factor"]) * 9_000.0
    return stats["pnl"] + min(tpy, 450.0) * 12.0 - dd_penalty - pf_penalty


def make_param_grid():
    base = dict(FINAL_PARAMS)
    for (
        threshold,
        edge_threshold,
        tp_atr,
        sl_atr,
        max_hold,
        cooldown,
        close_on_opposite,
        hours,
        weekdays,
        rsi_filter,
        risk,
    ) in product(
        [0.47, 0.49, 0.50, 0.515],
        [0.0],
        [0.8, 0.9],
        [1.8],
        [60, 90],
        [0],
        [False, True],
        [
            list(range(24)),
        ],
        [
            [0, 1, 2, 3, 4],
        ],
        [
            [],
        ],
        [0.028, 0.032],
    ):
        params = dict(base)
        params.update(
            {
                "threshold": threshold,
                "edge_threshold": edge_threshold,
                "tp_atr": tp_atr,
                "sl_atr": sl_atr,
                "max_hold": max_hold,
                "cooldown_ticks": cooldown,
                "close_on_opposite": close_on_opposite,
                "allowed_entry_hours": hours,
                "allowed_entry_weekdays": weekdays,
                "excluded_rsi_ranges": rsi_filter,
                "risk_per_trade": risk,
            }
        )
        yield params


def main():
    print("Sweeping higher-frequency barrier variants...")
    df, features = prepare_barrier_data()
    train_end = int(len(df) * TRAIN_END_RATIO)
    test_df = df.iloc[train_end:].copy().reset_index(drop=True)

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    model.set_params(device="cpu")
    probs = model.predict_proba(test_df[features]).astype(np.float32)

    baseline = evaluate_df(FINAL_PARAMS, test_df, probs)
    print("Baseline:")
    print(f"   trades/year={trades_per_year(baseline, test_df):.1f}")
    print("   " + format_stats("test", baseline))

    results = []
    highest_frequency = []
    for params in make_param_grid():
        stats = evaluate_df(params, test_df, probs)
        tpy = trades_per_year(stats, test_df)
        highest_frequency.append((tpy, params, stats))
        score = frequency_score(stats, tpy)
        if score <= -100_000.0:
            continue
        results.append((score, tpy, params, stats))

    results.sort(key=lambda item: item[0], reverse=True)
    highest_frequency.sort(key=lambda item: item[0], reverse=True)

    print("Top profitable high-frequency candidates:")
    for rank, (score, tpy, params, stats) in enumerate(results[:15], start=1):
        print(f"#{rank} frequency_score={score:.2f}, trades/year={tpy:.1f}, params={params}")
        print("   " + format_stats("test", stats))

    print("Highest-frequency candidates regardless of quality:")
    for rank, (tpy, params, stats) in enumerate(highest_frequency[:12], start=1):
        print(f"#{rank} trades/year={tpy:.1f}, params={params}")
        print("   " + format_stats("test", stats))


if __name__ == "__main__":
    main()
