import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import xgboost as xgb

from barrier_final_train import FINAL_PARAMS, MODEL_PATH, TRAIN_END_RATIO, prepare_barrier_data
from barrier_classifier_strategy import evaluate
from drl_train_candidate import format_stats


def main():
    print("Starting final barrier model backtest...")
    df, features = prepare_barrier_data()
    train_end = int(len(df) * TRAIN_END_RATIO)
    test_df = df.iloc[train_end:].copy().reset_index(drop=True)

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    model.set_params(device="cpu")
    probs = model.predict_proba(test_df[features]).astype(np.float32)
    stats = evaluate(
        FINAL_PARAMS,
        test_df["CLOSE"].to_numpy(dtype=np.float64),
        test_df["ATR"].to_numpy(dtype=np.float64),
        probs,
        hours=test_df["TIME_DT"].dt.hour.to_numpy(dtype=np.int16),
        weekdays=test_df["TIME_DT"].dt.dayofweek.to_numpy(dtype=np.int8),
        dates=test_df["TIME_DT"].dt.date.to_numpy(),
        rsi_values=test_df["M1_RSI"].to_numpy(dtype=np.float64),
    )
    start_time = test_df["TIME_DT"].iloc[0]
    end_time = test_df["TIME_DT"].iloc[-1]
    elapsed_days = max((end_time - start_time).total_seconds() / 86400.0, 1e-9)
    trades_per_day = stats["trades"] / elapsed_days
    trades_per_year = stats["trades"] / (elapsed_days / 365.25)
    print(
        f"Test period: {start_time} -> {end_time} "
        f"({elapsed_days:.1f} days, {trades_per_day:.2f} trades/day, "
        f"{trades_per_year:.1f} trades/year)"
    )
    print(
        f"Capital: {stats['initial_balance']:.2f} -> {stats['balance']:.2f} "
        f"| stopped_out={stats['stopped_out']}"
    )
    print(format_stats("Final barrier test", stats))


if __name__ == "__main__":
    main()
