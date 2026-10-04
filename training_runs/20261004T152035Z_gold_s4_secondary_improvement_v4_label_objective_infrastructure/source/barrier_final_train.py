import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import torch
import xgboost as xgb

from barrier_classifier_strategy import (
    HORIZON,
    build_barrier_target,
    build_profit_sample_weight,
    evaluate,
)
from drl_train_candidate import format_stats
from drl_trading_v2 import load_and_prepare_data


MODEL_SELECTION_END_RATIO = 0.70
TRAIN_END_RATIO = 0.85
MODEL_PATH = "gold_barrier_final_xgb.json"
FINAL_PARAMS = {
    "threshold": 0.525,
    "edge_threshold": 0.0,
    "tp_atr": 1.1,
    "sl_atr": 2.0,
    "min_tp_price": 1.5,
    "min_sl_price": 0.6,
    "max_hold": 180,
    "cooldown_ticks": 0,
    "close_on_opposite": False,
    "direction_mode": "long",
    "initial_balance": 1000,
    "stop_out_balance": 0,
    "risk_per_trade": 0.028,
    "allowed_entry_hours": [0, 1, 3, 8, 9, 11, 12, 17, 19, 20, 22, 23],
    "allowed_entry_weekdays": [0, 1, 2, 4],
    "excluded_rsi_ranges": [(35.0, 45.0)],
    "max_daily_loss_pct": 0.05,
    "max_daily_trades": None,
    "extra_cost_points": 5.0,
    "drawdown_guard_start_pct": 0.08,
    "drawdown_guard_full_pct": 0.35,
    "drawdown_guard_min_risk_mult": 0.50,
    "loss_streak_threshold": 3,
    "loss_streak_risk_mult": 0.55,
    "loss_streak_pause_threshold": 3,
    "loss_streak_pause_ticks": 120,
    "rolling_guard_window": 30,
    "rolling_guard_min_trades": 18,
    "rolling_guard_min_profit_factor": 1.15,
    "rolling_guard_min_win_rate": None,
    "rolling_guard_risk_mult": 0.50,
    "rolling_guard_pause_ticks": 0,
}

def prepare_barrier_data():
    df, features = load_and_prepare_data()
    df = df.copy()
    df["BARRIER_TARGET"] = build_barrier_target(df)
    df = df.iloc[:-HORIZON].dropna(subset=features + ["BARRIER_TARGET", "ATR"])
    return df.reset_index(drop=True), features


def train_final_classifier(train_df, features, model_path=None):
    sample_weight = build_profit_sample_weight(
        train_df,
        train_df["BARRIER_TARGET"].to_numpy(dtype=np.int8),
    )
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        device="cuda" if torch.cuda.is_available() else "cpu",
        n_estimators=450,
        learning_rate=0.04,
        max_depth=5,
        min_child_weight=80,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbosity=1,
    )
    model.fit(train_df[features], train_df["BARRIER_TARGET"], sample_weight=sample_weight)
    if model_path:
        model.save_model(model_path)
    return model


def evaluate_model(model, test_df, features):
    model.set_params(device="cpu")
    probs = model.predict_proba(test_df[features]).astype(np.float32)
    return evaluate(
        FINAL_PARAMS,
        test_df["CLOSE"].to_numpy(dtype=np.float64),
        test_df["ATR"].to_numpy(dtype=np.float64),
        probs,
        hours=test_df["TIME_DT"].dt.hour.to_numpy(dtype=np.int16),
        weekdays=test_df["TIME_DT"].dt.dayofweek.to_numpy(dtype=np.int8),
        dates=test_df["TIME_DT"].dt.date.to_numpy(),
        rsi_values=test_df["M1_RSI"].to_numpy(dtype=np.float64),
    )


def print_target_ratios(parts):
    for name, part in parts:
        ratios = part["BARRIER_TARGET"].value_counts(normalize=True).sort_index()
        print(f"{name} target ratio: {ratios.to_dict()}")


def main():
    print("Training final barrier classifier...")
    df, features = prepare_barrier_data()
    selection_end = int(len(df) * MODEL_SELECTION_END_RATIO)
    train_end = int(len(df) * TRAIN_END_RATIO)
    selection_train_df = df.iloc[: max(0, selection_end - HORIZON)].copy()
    validation_df = df.iloc[selection_end:train_end].copy().reset_index(drop=True)
    train_df = df.iloc[: max(0, train_end - HORIZON)].copy()
    test_df = df.iloc[train_end:].copy().reset_index(drop=True)

    print(
        f"Rows | selection_train={len(selection_train_df):,} "
        f"validation={len(validation_df):,} final_train={len(train_df):,} "
        f"test={len(test_df):,}"
    )
    print_target_ratios(
        [
            ("selection_train", selection_train_df),
            ("validation", validation_df),
            ("final_train", train_df),
            ("test", test_df),
        ]
    )

    validation_model = train_final_classifier(selection_train_df, features)
    validation_stats = evaluate_model(validation_model, validation_df, features)
    print(format_stats("Frozen validation", validation_stats))

    model = train_final_classifier(train_df, features, MODEL_PATH)
    test_stats = evaluate_model(model, test_df, features)
    print(format_stats("Final barrier test", test_stats))
    print(f"Saved final barrier model as {MODEL_PATH}")


if __name__ == "__main__":
    main()
