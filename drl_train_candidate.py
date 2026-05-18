import os
import shutil
from datetime import datetime

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from drl_trading_v2 import (
    EXPERT_TRAIN_END_RATIO,
    INITIAL_BALANCE,
    LABEL_LOOKAHEAD,
    MAX_HOLD_TICKS,
    PPO_MODEL_PATH,
    STOP_LOSS_PRICE,
    Experts,
    GoldDRLEnv,
    check_gpu,
    load_and_prepare_data,
)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
CANDIDATE_PREFIX = f"gold_mtf_candidate_{RUN_ID}"
CANDIDATE_MODEL = f"gold_ppo_candidate_{RUN_ID}"
BEST_CANDIDATE_MODEL = f"gold_ppo_candidate_best_{RUN_ID}"
CURRENT_EXPERT_PREFIX = "gold_mtf"
TOTAL_TIMESTEPS = 500_000
EVAL_FREQ = 100_000
VAL_START_RATIO = 0.80
VAL_END_RATIO = 0.85
MIN_VAL_TRADES = 3
AUTO_PROMOTE = False


def close_position(balance, position, entry_price, curr_price, spread_points=30):
    pnl = curr_price - entry_price if position == 1 else entry_price - curr_price
    trade_reward = (pnl * 100) - (spread_points * 0.01 * 100)
    return balance + trade_reward, trade_reward


class SavedExperts:
    def __init__(self, features):
        self.features = features
        self.xgb = xgb.Booster()
        self.lgb = None

    def load(self, prefix):
        self.xgb.load_model(f"{prefix}_xgb.json")
        self.xgb.set_param({"device": "cpu"})
        self.lgb = lgb.Booster(model_file=f"{prefix}_lgb.txt")

    def predict_probs(self, df):
        dmatrix = xgb.DMatrix(df[self.features])
        p_xgb = np.asarray(self.xgb.predict(dmatrix), dtype=np.float32)
        p_lgb = np.asarray(self.lgb.predict(df[self.features]), dtype=np.float32)
        return np.hstack([p_xgb, p_lgb])


def simulate_policy(
    model,
    df,
    expert_probs,
    stop_loss=STOP_LOSS_PRICE,
    max_hold_ticks=MAX_HOLD_TICKS,
):
    balance = INITIAL_BALANCE
    balance_history = [balance]
    position = 0
    entry_price = 0.0
    hold_ticks = 0
    trades = 0
    wins = 0
    rewards = []
    actions = {0: 0, 1: 0, 2: 0}
    forced_exits = 0
    stop_loss_exits = 0
    timeout_exits = 0

    prices = df["CLOSE"].to_numpy()
    for i, curr_price in enumerate(prices):
        probs = expert_probs[i].astype(np.float32)
        state = np.array(
            [
                position / 2.0,
                (curr_price - entry_price) / (entry_price + 1e-6) if position else 0,
                min(balance / INITIAL_BALANCE, 2.0),
            ],
            dtype=np.float32,
        )
        action, _ = model.predict(np.concatenate([probs, state]), deterministic=True)
        action = int(np.asarray(action).item())
        actions[action] += 1

        if position == 0:
            if action in (1, 2):
                position = action
                entry_price = curr_price
                hold_ticks = 0
        elif action == 0:
            balance, reward = close_position(balance, position, entry_price, curr_price)
            rewards.append(reward)
            trades += 1
            wins += int(reward > 0)
            position = 0
            entry_price = 0.0
            hold_ticks = 0

        if position:
            hold_ticks += 1
            float_pnl = curr_price - entry_price if position == 1 else entry_price - curr_price
            exit_reason = None
            if float_pnl < stop_loss:
                exit_reason = "stop_loss"
            elif hold_ticks > max_hold_ticks:
                exit_reason = "timeout"

            if exit_reason is not None:
                balance, reward = close_position(balance, position, entry_price, curr_price)
                rewards.append(reward)
                trades += 1
                wins += int(reward > 0)
                forced_exits += 1
                stop_loss_exits += int(exit_reason == "stop_loss")
                timeout_exits += int(exit_reason == "timeout")
                position = 0
                entry_price = 0.0
                hold_ticks = 0

        balance_history.append(balance)

    if position:
        balance, reward = close_position(balance, position, entry_price, prices[-1])
        rewards.append(reward)
        trades += 1
        wins += int(reward > 0)
        forced_exits += 1
        balance_history[-1] = balance

    equity = np.asarray(balance_history, dtype=np.float64)
    drawdown = equity - np.maximum.accumulate(equity)
    max_drawdown = float(drawdown.min())
    pnl = float(balance - INITIAL_BALANCE)
    roi = pnl / INITIAL_BALANCE
    win_rate = wins / trades if trades else 0.0
    avg_trade = float(np.mean(rewards)) if rewards else 0.0
    median_trade = float(np.median(rewards)) if rewards else 0.0

    # Penalize no-trade policies and heavy drawdown. This is validation scoring,
    # not a claim of live profitability.
    trade_penalty = 3000.0 if trades < MIN_VAL_TRADES else 0.0
    score = pnl + (0.5 * max_drawdown) - trade_penalty
    return {
        "balance": balance,
        "roi": roi,
        "pnl": pnl,
        "trades": trades,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "avg_trade": avg_trade,
        "median_trade": median_trade,
        "forced_exits": forced_exits,
        "stop_loss_exits": stop_loss_exits,
        "timeout_exits": timeout_exits,
        "actions": actions,
        "score": score,
    }


def format_stats(name, stats):
    profit_factor = stats.get("profit_factor")
    if profit_factor is None:
        pf_text = "n/a"
    elif np.isinf(profit_factor):
        pf_text = "inf"
    else:
        pf_text = f"{profit_factor:.2f}"
    dd_pct = stats.get("max_drawdown_pct")
    dd_pct_text = f", dd_pct={dd_pct:.2%}" if dd_pct is not None else ""
    risk_text = (
        f"pf={pf_text}, sharpe_like={stats.get('sharpe_like', 0.0):.2f}, "
        f"max_loss_streak={stats.get('max_consecutive_losses', 0)}"
        if "profit_factor" in stats
        else ""
    )
    risk_text = f"{risk_text}, " if risk_text else ""
    return (
        f"{name}: score={stats['score']:.2f}, roi={stats['roi']:.2%}, "
        f"pnl={stats['pnl']:.2f}, trades={stats['trades']}, "
        f"win={stats['win_rate']:.2%}, dd={stats['max_drawdown']:.2f}"
        f"{dd_pct_text}, "
        f"avg/med={stats['avg_trade']:.2f}/{stats['median_trade']:.2f}, "
        f"{risk_text}"
        f"forced={stats['forced_exits']} "
        f"(SL={stats['stop_loss_exits']}, timeout={stats['timeout_exits']}), "
        f"actions={stats['actions']}"
    )


class ValidationCallback(BaseCallback):
    def __init__(self, val_df, val_probs, save_path, eval_freq=EVAL_FREQ):
        super().__init__()
        self.val_df = val_df
        self.val_probs = val_probs
        self.save_path = save_path
        self.eval_freq = eval_freq
        self.best_score = -np.inf
        self.best_stats = None

    def _on_step(self):
        if self.num_timesteps % self.eval_freq != 0:
            return True
        stats = simulate_policy(self.model, self.val_df, self.val_probs)
        print(
            "[validation] "
            f"steps={self.num_timesteps:,} score={stats['score']:.2f} "
            f"roi={stats['roi']:.2%} trades={stats['trades']} "
            f"dd={stats['max_drawdown']:.2f} actions={stats['actions']}"
        )
        if stats["score"] > self.best_score:
            self.best_score = stats["score"]
            self.best_stats = stats
            self.model.save(self.save_path)
            print(f"[validation] saved new best candidate: {self.save_path}.zip")
        return True


def main():
    train_device = check_gpu()
    df, features = load_and_prepare_data()

    expert_split = int(len(df) * EXPERT_TRAIN_END_RATIO)
    val_start = int(len(df) * VAL_START_RATIO)
    val_end = int(len(df) * VAL_END_RATIO)

    train_experts_df = df.iloc[: max(0, expert_split - LABEL_LOOKAHEAD)].copy()
    rl_train_df = df.iloc[expert_split:val_start].copy().reset_index(drop=True)
    val_df = df.iloc[val_start:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)

    print(
        f"Rows | expert={len(train_experts_df):,} "
        f"rl={len(rl_train_df):,} val={len(val_df):,} test={len(test_df):,}"
    )

    exp = Experts(features, device=train_device)
    exp.train(train_experts_df)
    exp.save(CANDIDATE_PREFIX)

    print("Generating candidate expert signals...")
    rl_probs = exp.predict_probs(rl_train_df)
    val_probs = exp.predict_probs(val_df)
    test_probs = exp.predict_probs(test_df)

    env = DummyVecEnv([lambda: GoldDRLEnv(rl_train_df, rl_probs)])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        device=train_device,
        learning_rate=1e-4,
        n_steps=4096,
        batch_size=128,
        ent_coef=0.02,
    )

    callback = ValidationCallback(val_df, val_probs, BEST_CANDIDATE_MODEL)
    print(f"Training candidate for {TOTAL_TIMESTEPS:,} steps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    model.save(CANDIDATE_MODEL)

    model_path = (
        BEST_CANDIDATE_MODEL
        if os.path.exists(f"{BEST_CANDIDATE_MODEL}.zip")
        else CANDIDATE_MODEL
    )
    best_model = PPO.load(model_path, device=train_device)
    val_stats = simulate_policy(best_model, val_df, val_probs)
    test_stats = simulate_policy(best_model, test_df, test_probs)
    print(f"Best candidate model: {model_path}.zip")
    print(format_stats("Candidate validation", val_stats))
    print(format_stats("Candidate test", test_stats))

    current_test_stats = None
    if (
        os.path.exists(f"{PPO_MODEL_PATH}.zip")
        and os.path.exists(f"{CURRENT_EXPERT_PREFIX}_xgb.json")
        and os.path.exists(f"{CURRENT_EXPERT_PREFIX}_lgb.txt")
    ):
        current_model = PPO.load(PPO_MODEL_PATH, device=train_device)
        current_exp = SavedExperts(features)
        current_exp.load(CURRENT_EXPERT_PREFIX)
        current_test_probs = current_exp.predict_probs(test_df)
        current_test_stats = simulate_policy(current_model, test_df, current_test_probs)
        print(format_stats("Current main test", current_test_stats))

    should_promote = (
        AUTO_PROMOTE
        and current_test_stats is not None
        and val_stats["trades"] >= MIN_VAL_TRADES
        and test_stats["trades"] >= MIN_VAL_TRADES
        and val_stats["pnl"] > 0
        and test_stats["score"] > current_test_stats["score"]
    )

    if should_promote:
        print("Promoting candidate to main model files.")
        shutil.copyfile(f"{model_path}.zip", f"{PPO_MODEL_PATH}.zip")
        shutil.copyfile(f"{CANDIDATE_PREFIX}_xgb.json", "gold_mtf_xgb.json")
        shutil.copyfile(f"{CANDIDATE_PREFIX}_lgb.txt", "gold_mtf_lgb.txt")
    else:
        print("Candidate kept separate; main model files were not changed.")


if __name__ == "__main__":
    main()
