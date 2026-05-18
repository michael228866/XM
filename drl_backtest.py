import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from stable_baselines3 import PPO

DATA_DIR = r"D:\XM\數據"
INITIAL_BALANCE = 10000
SPREAD_POINTS = 30
MAX_HOLD_TICKS = 240
STOP_LOSS_PRICE = -0.2
TAKE_PROFIT_PRICE = None

BASE_FEATURES = [
    "M1_RSI",
    "ATR",
    "MACD_HIST",
    "BB_WIDTH",
    "BIAS_20",
    "BODY_PCT",
    "ROC_5",
    "VOLA_RATIO",
    "HOUR_SIN",
    "HOUR_COS",
    "DAY_OF_WEEK",
]


def add_indicators(df):
    df = df.copy()
    high_low = df["HIGH"] - df["LOW"]
    tr = pd.concat(
        [
            high_low,
            np.abs(df["HIGH"] - df["CLOSE"].shift()),
            np.abs(df["LOW"] - df["CLOSE"].shift()),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["HOUR_SIN"] = np.sin(2 * np.pi * df["TIME_DT"].dt.hour / 24)
    df["HOUR_COS"] = np.cos(2 * np.pi * df["TIME_DT"].dt.hour / 24)
    df["DAY_OF_WEEK"] = df["TIME_DT"].dt.dayofweek / 7.0

    ema12 = df["CLOSE"].ewm(span=12, adjust=False).mean()
    ema26 = df["CLOSE"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df["MACD_HIST"] = macd - macd.ewm(span=9, adjust=False).mean()

    ma20 = df["CLOSE"].rolling(20).mean()
    df["BB_WIDTH"] = (df["CLOSE"].rolling(20).std() * 4) / (ma20 + 1e-6)
    df["BIAS_20"] = (df["CLOSE"] - ma20) / (ma20 + 1e-6)
    df["ROC_5"] = df["CLOSE"].pct_change(5)
    candle_range = df["HIGH"] - df["LOW"] + 1e-6
    df["BODY_PCT"] = np.abs(df["CLOSE"] - df["OPEN"]) / candle_range
    return df


class Experts:
    def __init__(self, features):
        self.features = features
        self.xgb = xgb.Booster()
        self.lgb = None

    def load(self, prefix):
        self.xgb.load_model(f"{prefix}_xgb.json")
        self.lgb = lgb.Booster(model_file=f"{prefix}_lgb.txt")
        print("Experts Loaded.")

    def predict_probs(self, df):
        dtrain = xgb.DMatrix(df[self.features])
        p_xgb = np.asarray(self.xgb.predict(dtrain), dtype=np.float32)
        p_lgb = np.asarray(self.lgb.predict(df[self.features]), dtype=np.float32)
        return np.hstack([p_xgb, p_lgb])


def load_test_data():
    all_files = sorted(
        f for f in os.listdir(DATA_DIR) if f.startswith("GOLD#_") and f.endswith(".csv")
    )
    data_dict = {}

    for filename in all_files:
        tf_name = filename.split("_")[1]
        path = os.path.join(DATA_DIR, filename)
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception as exc:
            print(f"Skipped {filename}: {exc}")
            continue

        df.columns = [c.replace("<", "").replace(">", "").upper() for c in df.columns]
        if "DATE" in df.columns and "TIME" in df.columns:
            df["TIME_DT"] = pd.to_datetime(df["DATE"] + " " + df["TIME"])
        elif "DATE" in df.columns:
            df["TIME_DT"] = pd.to_datetime(df["DATE"])
        else:
            continue

        data_dict[tf_name] = df.sort_values("TIME_DT")

    if "M1" not in data_dict:
        raise FileNotFoundError("M1 data file was not found.")

    m1 = add_indicators(data_dict["M1"])
    diff = m1["CLOSE"].diff()
    gain = diff.where(diff > 0, 0).rolling(14).mean()
    loss = (-diff.where(diff < 0, 0)).rolling(14).mean()
    m1["M1_RSI"] = 100 - (100 / (1 + (gain / (loss + 1e-6))))
    m1["VOLA_MA"] = m1["ATR"].rolling(240).mean()
    m1["VOLA_RATIO"] = m1["ATR"] / (m1["VOLA_MA"] + 1e-6)

    df = m1.sort_values("TIME_DT")
    mtf_features = []
    for tf_name, tdf in data_dict.items():
        if tf_name == "M1":
            continue
        tdf = add_indicators(tdf)
        trend_col = f"{tf_name}_TREND"
        tdf[trend_col] = np.where(tdf["CLOSE"] > tdf["CLOSE"].rolling(20).mean(), 1, -1)
        tdf[trend_col] = tdf[trend_col].shift(1)
        df = pd.merge_asof(df, tdf[["TIME_DT", trend_col]], on="TIME_DT")
        mtf_features.append(trend_col)

    full_features = BASE_FEATURES + mtf_features
    df[full_features] = df[full_features].shift(1)
    return df.dropna(subset=full_features), full_features


def close_position(balance, position, entry_price, curr_price):
    pnl = curr_price - entry_price if position == 1 else entry_price - curr_price
    trade_reward = (pnl * 100) - (SPREAD_POINTS * 0.01 * 100)
    return balance + trade_reward, trade_reward


if __name__ == "__main__":
    print("Starting Ensemble-DRL Backtest...")
    df, features = load_test_data()

    split = int(len(df) * 0.85)
    test_df = df.iloc[split:].copy().reset_index(drop=True)

    exp = Experts(features)
    exp.load("gold_mtf")
    expert_signals = exp.predict_probs(test_df)

    device = "cpu"
    model = PPO.load("gold_ppo_commander", device=device)

    print(f"Running Simulation on {len(test_df)} minutes of data ({device.upper()} Mode)...")
    balance = INITIAL_BALANCE
    balance_history = [balance]
    position = 0
    entry_price = 0
    hold_ticks = 0
    trades = 0
    wins = 0
    forced_exits = 0
    take_profit_exits = 0
    stop_loss_exits = 0
    timeout_exits = 0
    trade_rewards = []
    action_counts = {0: 0, 1: 0, 2: 0}

    for i in range(len(test_df)):
        curr_price = test_df.loc[i, "CLOSE"]
        probs = expert_signals[i].astype(np.float32)
        state = np.array(
            [
                position / 2.0,
                (curr_price - entry_price) / (entry_price + 1e-6)
                if position != 0
                else 0,
                min(balance / INITIAL_BALANCE, 2.0),
            ],
            dtype=np.float32,
        )
        obs = np.concatenate([probs, state])

        action, _ = model.predict(obs, deterministic=True)
        action = int(np.asarray(action).item())
        action_counts[action] += 1

        if position == 0:
            if action == 1:
                position = 1
                entry_price = curr_price
                hold_ticks = 0
            elif action == 2:
                position = 2
                entry_price = curr_price
                hold_ticks = 0
        elif action == 0:
            balance, trade_reward = close_position(balance, position, entry_price, curr_price)
            trade_rewards.append(trade_reward)
            trades += 1
            wins += int(trade_reward > 0)
            position = 0
            entry_price = 0
            hold_ticks = 0

        if position != 0:
            hold_ticks += 1
            float_pnl = curr_price - entry_price if position == 1 else entry_price - curr_price
            exit_reason = None
            if TAKE_PROFIT_PRICE is not None and float_pnl > TAKE_PROFIT_PRICE:
                exit_reason = "take_profit"
            elif float_pnl < STOP_LOSS_PRICE:
                exit_reason = "stop_loss"
            elif hold_ticks > MAX_HOLD_TICKS:
                exit_reason = "timeout"

            if exit_reason is not None:
                balance, trade_reward = close_position(balance, position, entry_price, curr_price)
                trade_rewards.append(trade_reward)
                trades += 1
                wins += int(trade_reward > 0)
                forced_exits += 1
                take_profit_exits += int(exit_reason == "take_profit")
                stop_loss_exits += int(exit_reason == "stop_loss")
                timeout_exits += int(exit_reason == "timeout")
                position = 0
                entry_price = 0
                hold_ticks = 0

        balance_history.append(balance)

    if position != 0:
        curr_price = test_df.loc[len(test_df) - 1, "CLOSE"]
        balance, trade_reward = close_position(balance, position, entry_price, curr_price)
        trade_rewards.append(trade_reward)
        trades += 1
        wins += int(trade_reward > 0)
        forced_exits += 1
        balance_history[-1] = balance

    roi = (balance - INITIAL_BALANCE) / INITIAL_BALANCE
    win_rate = wins / trades if trades > 0 else 0
    trade_rewards_arr = np.array(trade_rewards, dtype=np.float64)
    avg_trade = trade_rewards_arr.mean() if len(trade_rewards_arr) else 0
    median_trade = np.median(trade_rewards_arr) if len(trade_rewards_arr) else 0
    best_trade = trade_rewards_arr.max() if len(trade_rewards_arr) else 0
    worst_trade = trade_rewards_arr.min() if len(trade_rewards_arr) else 0
    equity = np.array(balance_history, dtype=np.float64)
    running_peak = np.maximum.accumulate(equity)
    max_drawdown = (equity - running_peak).min()
    print("\n" + "$" * 20)
    print("BACKTEST RESULT (Test Set)")
    print(f"Final Balance: ${balance:.2f}")
    print(f"Total ROI: {roi:.2%}")
    print(f"Total Trades: {trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Forced Exits: {forced_exits}")
    print(f"Exit Reasons: TP={take_profit_exits}, SL={stop_loss_exits}, Timeout={timeout_exits}")
    print(f"Avg/Median Trade: ${avg_trade:.2f} / ${median_trade:.2f}")
    print(f"Best/Worst Trade: ${best_trade:.2f} / ${worst_trade:.2f}")
    print(f"Max Drawdown: ${max_drawdown:.2f}")
    print(
        "Actions: "
        f"Wait/Close={action_counts[0]}, "
        f"Buy={action_counts[1]}, "
        f"Sell={action_counts[2]}"
    )
    print("$" * 20)

    plt.figure(figsize=(12, 6))
    plt.plot(balance_history, label="Ensemble-DRL Equity")
    plt.axhline(y=INITIAL_BALANCE, color="r", linestyle="--", label="Baseline")
    plt.title("Gold Trading Ensemble-DRL Backtest (Test Set)")
    plt.xlabel("Time (Minutes)")
    plt.ylabel("Balance ($)")
    plt.legend()
    plt.grid(True)
    plt.savefig("drl_backtest_result.png")
    plt.close()
    print("Equity curve saved as 'drl_backtest_result.png'")
