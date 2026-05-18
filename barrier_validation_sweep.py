import os
from itertools import product

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np

from barrier_classifier_strategy import evaluate
from barrier_final_train import (
    FINAL_PARAMS,
    HORIZON,
    MODEL_SELECTION_END_RATIO,
    TRAIN_END_RATIO,
    prepare_barrier_data,
    train_final_classifier,
)
from drl_train_candidate import format_stats


MIN_VALIDATION_TRADES = 120
MIN_TEST_TRADES = 120
MAX_DRAWDOWN_PCT = -0.45


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


def robust_score(stats):
    if stats["stopped_out"] or stats["pnl"] <= 0:
        return -1_000_000.0 + stats["pnl"]
    if stats["trades"] < MIN_VALIDATION_TRADES:
        return -500_000.0 - ((MIN_VALIDATION_TRADES - stats["trades"]) * 1000.0)
    if stats["max_drawdown_pct"] < MAX_DRAWDOWN_PCT:
        return -250_000.0 + stats["pnl"]

    drawdown_penalty = abs(min(stats["max_drawdown_pct"], 0.0)) * 20_000.0
    loss_streak_penalty = stats["max_consecutive_losses"] * 350.0
    win_penalty = max(0.0, 0.60 - stats["win_rate"]) * 12_000.0
    pf_bonus = min(stats["profit_factor"], 3.0) * 1200.0
    trade_bonus = min(stats["trades"], 320) * 8.0
    return (
        stats["pnl"]
        - drawdown_penalty
        - loss_streak_penalty
        - win_penalty
        + pf_bonus
        + trade_bonus
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
        [(34.0, 46.0)],
        [],
    ]

    for threshold, tp_atr, sl_atr, risk, daily_loss, max_daily_trades, hours, weekdays, rsi_filter in product(
        [0.525, 0.535, 0.545, 0.555],
        [1.0, 1.1, 1.2],
        [1.8, 2.0, 2.2, 2.4],
        [0.018, 0.022, 0.026, 0.030],
        [0.04, 0.05, 0.07],
        [1, 2, None],
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
                "max_daily_trades": max_daily_trades,
                "allowed_entry_hours": hours,
                "allowed_entry_weekdays": weekdays,
                "excluded_rsi_ranges": rsi_filter,
            }
        )
        yield params


def main():
    print("Running clean validation sweep...")
    df, features = prepare_barrier_data()
    selection_end = int(len(df) * MODEL_SELECTION_END_RATIO)
    train_end = int(len(df) * TRAIN_END_RATIO)

    train_df = df.iloc[: max(0, selection_end - HORIZON)].copy()
    validation_df = df.iloc[selection_end:train_end].copy().reset_index(drop=True)
    test_df = df.iloc[train_end:].copy().reset_index(drop=True)
    print(
        f"Rows | train={len(train_df):,} validation={len(validation_df):,} "
        f"frozen_test={len(test_df):,}"
    )

    model = train_final_classifier(train_df, features)
    model.set_params(device="cpu")
    validation_probs = model.predict_proba(validation_df[features]).astype(np.float32)
    test_probs = model.predict_proba(test_df[features]).astype(np.float32)

    baseline_validation = evaluate_df(FINAL_PARAMS, validation_df, validation_probs)
    baseline_test = evaluate_df(FINAL_PARAMS, test_df, test_probs)
    print("Baseline:")
    print("   " + format_stats("validation", baseline_validation))
    print("   " + format_stats("frozen_test", baseline_test))

    candidates = []
    for params in make_param_grid():
        validation_stats = evaluate_df(params, validation_df, validation_probs)
        score = robust_score(validation_stats)
        if score <= -100_000.0:
            continue
        candidates.append((score, params, validation_stats))

    candidates.sort(key=lambda item: item[0], reverse=True)
    print("Top validation-selected candidates with frozen test check:")
    accepted = 0
    for score, params, validation_stats in candidates[:40]:
        test_stats = evaluate_df(params, test_df, test_probs)
        if (
            test_stats["stopped_out"]
            or test_stats["pnl"] <= 0
            or test_stats["trades"] < MIN_TEST_TRADES
        ):
            continue
        accepted += 1
        print(f"#{accepted} validation_score={score:.2f} params={params}")
        print("   " + format_stats("validation", validation_stats))
        print("   " + format_stats("frozen_test", test_stats))
        if accepted >= 12:
            break


if __name__ == "__main__":
    main()
