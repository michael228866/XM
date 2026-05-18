import os
from itertools import product

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from barrier_classifier_strategy import forward_rolling, simulate_barrier
from drl_train_candidate import format_stats
from drl_trading_v2 import INITIAL_BALANCE, load_and_prepare_data


MODEL_SELECTION_END_RATIO = 0.70
TRAIN_END_RATIO = 0.85
MIN_TP_PRICE = 0.8
MIN_SL_PRICE = 0.5
EXTRA_COST_POINTS = 5.0


def build_target(df, horizon, label_tp_atr, label_sl_atr):
    close = df["CLOSE"].to_numpy(dtype=np.float64)
    atr = df["ATR"].to_numpy(dtype=np.float64)
    tp = np.maximum(atr * label_tp_atr, MIN_TP_PRICE)
    sl = np.maximum(atr * label_sl_atr, MIN_SL_PRICE)
    future_high = forward_rolling(df["HIGH"], horizon, "max")
    future_low = forward_rolling(df["LOW"], horizon, "min")

    up = future_high - close
    down = close - future_low
    long_clean = (up >= tp) & (down < sl)
    short_clean = (down >= tp) & (up < sl)

    target = np.zeros(len(df), dtype=np.int8)
    target[long_clean & ~short_clean] = 1
    target[short_clean & ~long_clean] = 2
    return target


def build_sample_weight(target):
    class_counts = np.bincount(target.astype(np.int64), minlength=3)
    class_weight = len(target) / (3.0 * np.maximum(class_counts, 1))
    weights = class_weight[target.astype(np.int64)]
    weights[target == 0] *= 0.75
    return weights


def prepare_data(horizon, label_tp_atr, label_sl_atr):
    df, features = load_and_prepare_data()
    df = df.copy()
    df["SHORT_TARGET"] = build_target(df, horizon, label_tp_atr, label_sl_atr)
    df = df.iloc[:-horizon].dropna(subset=features + ["SHORT_TARGET", "ATR"])
    return df.reset_index(drop=True), features


def train_model(train_df, features):
    target = train_df["SHORT_TARGET"].to_numpy(dtype=np.int8)
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        device="cuda" if torch.cuda.is_available() else "cpu",
        n_estimators=140,
        learning_rate=0.07,
        max_depth=4,
        min_child_weight=70,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbosity=0,
    )
    model.fit(train_df[features], target, sample_weight=build_sample_weight(target))
    model.set_params(device="cpu")
    return model


def evaluate_df(params, df, probs):
    return simulate_barrier(
        prices=df["CLOSE"].to_numpy(dtype=np.float64),
        atr=df["ATR"].to_numpy(dtype=np.float64),
        probs=probs,
        hours=df["TIME_DT"].dt.hour.to_numpy(dtype=np.int16),
        weekdays=df["TIME_DT"].dt.dayofweek.to_numpy(dtype=np.int8),
        dates=df["TIME_DT"].dt.date.to_numpy(),
        rsi_values=df["M1_RSI"].to_numpy(dtype=np.float64),
        **params,
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
    if tpy < 180:
        return -250_000.0 - ((180 - tpy) * 1000.0)
    dd = abs(min(stats["max_drawdown_pct"], 0.0))
    pf_penalty = max(0.0, 1.30 - stats["profit_factor"]) * 10000.0
    dd_penalty = max(0.0, dd - 0.55) * 18000.0
    return stats["pnl"] + min(tpy, 400.0) * 10.0 - pf_penalty - dd_penalty


def make_param_grid(horizon):
    for threshold, tp_atr, sl_atr, max_hold, risk, hours in product(
        [0.48, 0.50, 0.52],
        [0.8, 0.9],
        [1.6, 1.8],
        [horizon],
        [0.018, 0.024],
        [
            [0, 1, 3, 8, 9, 11, 12, 17, 19, 20, 22, 23],
            list(range(24)),
        ],
    ):
        yield {
            "threshold": threshold,
            "edge_threshold": 0.0,
            "tp_atr": tp_atr,
            "sl_atr": sl_atr,
            "min_tp_price": MIN_TP_PRICE,
            "min_sl_price": MIN_SL_PRICE,
            "max_hold": max_hold,
            "cooldown_ticks": 0,
            "close_on_opposite": False,
            "direction_mode": "long",
            "initial_balance": INITIAL_BALANCE,
            "stop_out_balance": 0,
            "risk_per_trade": risk,
            "allowed_entry_hours": hours,
            "allowed_entry_weekdays": [0, 1, 2, 3, 4],
            "excluded_rsi_ranges": [],
            "max_daily_loss_pct": 0.06,
            "max_daily_trades": None,
            "extra_cost_points": EXTRA_COST_POINTS,
            "drawdown_guard_start_pct": 0.08,
            "drawdown_guard_full_pct": 0.35,
            "drawdown_guard_min_risk_mult": 0.5,
            "loss_streak_threshold": 3,
            "loss_streak_risk_mult": 0.55,
        }


def run_one(horizon, label_tp_atr, label_sl_atr):
    print(
        f"\n=== Short horizon experiment | horizon={horizon}, "
        f"label_tp_atr={label_tp_atr}, label_sl_atr={label_sl_atr} ==="
    )
    df, features = prepare_data(horizon, label_tp_atr, label_sl_atr)
    selection_end = int(len(df) * MODEL_SELECTION_END_RATIO)
    train_end = int(len(df) * TRAIN_END_RATIO)
    train_df = df.iloc[: max(0, selection_end - horizon)].copy()
    validation_df = df.iloc[selection_end:train_end].copy().reset_index(drop=True)
    test_df = df.iloc[train_end:].copy().reset_index(drop=True)

    print(
        f"Rows | train={len(train_df):,} validation={len(validation_df):,} "
        f"test={len(test_df):,}"
    )
    for name, part in [("train", train_df), ("validation", validation_df), ("test", test_df)]:
        ratios = part["SHORT_TARGET"].value_counts(normalize=True).sort_index()
        print(f"{name} target ratio: {ratios.to_dict()}")

    model = train_model(train_df, features)
    validation_probs = model.predict_proba(validation_df[features]).astype(np.float32)
    test_probs = model.predict_proba(test_df[features]).astype(np.float32)

    candidates = []
    for params in make_param_grid(horizon):
        validation_stats = evaluate_df(params, validation_df, validation_probs)
        validation_tpy = trades_per_year(validation_stats, validation_df)
        validation_score = score(validation_stats, validation_tpy)
        if validation_score <= -100_000.0:
            continue
        test_stats = evaluate_df(params, test_df, test_probs)
        test_tpy = trades_per_year(test_stats, test_df)
        candidates.append(
            (
                validation_score,
                params,
                validation_stats,
                validation_tpy,
                test_stats,
                test_tpy,
            )
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    if not candidates:
        print("No profitable validation candidates above 180 trades/year.")
        return []

    print("Top validation-selected candidates:")
    for rank, (
        validation_score,
        params,
        validation_stats,
        validation_tpy,
        test_stats,
        test_tpy,
    ) in enumerate(candidates[:8], start=1):
        print(f"#{rank} validation_score={validation_score:.2f}, params={params}")
        print(f"   validation trades/year={validation_tpy:.1f} | " + format_stats("validation", validation_stats))
        print(f"   test trades/year={test_tpy:.1f} | " + format_stats("test", test_stats))
    return candidates[:8]


def main():
    all_results = []
    for horizon, label_tp_atr, label_sl_atr in [
        (60, 0.9, 1.1),
        (90, 1.0, 1.2),
    ]:
        all_results.extend(run_one(horizon, label_tp_atr, label_sl_atr))

    print("\n=== Cross-experiment top tests ===")
    all_results.sort(key=lambda item: item[4]["pnl"], reverse=True)
    for rank, (_, params, _, _, test_stats, test_tpy) in enumerate(all_results[:10], start=1):
        print(f"#{rank} test trades/year={test_tpy:.1f}, params={params}")
        print("   " + format_stats("test", test_stats))


if __name__ == "__main__":
    main()
