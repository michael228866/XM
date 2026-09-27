import os
from itertools import product

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import torch

from drl_train_candidate import SavedExperts, format_stats
from drl_trading_v2 import (
    EXPERT_TRAIN_END_RATIO,
    INITIAL_BALANCE,
    LABEL_LOOKAHEAD,
    SPREAD_POINTS,
    load_and_prepare_data,
)


VAL_START_RATIO = 0.80
VAL_END_RATIO = 0.85
MIN_VAL_TRADES = 30
TARGET_WIN_RATE = 0.35


def close_reward(position, entry_price, curr_price):
    pnl = curr_price - entry_price if position == 1 else entry_price - curr_price
    return (pnl * 100.0) - (SPREAD_POINTS * 0.01 * 100.0)


def score_stats(stats):
    low_trade_penalty = max(0, MIN_VAL_TRADES - stats["trades"]) * 1000.0
    win_rate_penalty = max(0.0, TARGET_WIN_RATE - stats["win_rate"]) * 15000.0
    trade_bonus = min(stats["trades"], 120) * 20.0
    return (
        stats["pnl"]
        + (0.5 * stats["max_drawdown"])
        + trade_bonus
        - low_trade_penalty
        - win_rate_penalty
    )


def combined_score(segment_stats):
    score = sum(stats["score"] for stats in segment_stats)
    for stats in segment_stats:
        if stats["pnl"] <= 0:
            score -= 5000.0 + abs(stats["pnl"])
        if stats["trades"] < MIN_VAL_TRADES:
            score -= (MIN_VAL_TRADES - stats["trades"]) * 1000.0
        if stats["win_rate"] < TARGET_WIN_RATE:
            score -= (TARGET_WIN_RATE - stats["win_rate"]) * 15000.0
    return score


def summarize(
    balance_history,
    rewards,
    trades,
    wins,
    actions,
    forced,
    sl_count,
    tp_count,
    timeout_count,
    initial_balance=INITIAL_BALANCE,
    stopped_out=False,
):
    equity = np.asarray(balance_history, dtype=np.float64)
    drawdown = equity - np.maximum.accumulate(equity)
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
    balance = float(balance_history[-1])
    pnl = balance - initial_balance
    rewards_array = np.asarray(rewards, dtype=np.float64)
    gross_profit = float(rewards_array[rewards_array > 0].sum()) if rewards else 0.0
    gross_loss = float(-rewards_array[rewards_array < 0].sum()) if rewards else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    downside = rewards_array[rewards_array < 0]
    downside_std = float(np.std(downside)) if len(downside) > 1 else 0.0
    reward_std = float(np.std(rewards_array)) if len(rewards_array) > 1 else 0.0
    sharpe_like = (
        float(np.mean(rewards_array) / reward_std * np.sqrt(len(rewards_array)))
        if reward_std > 0
        else 0.0
    )
    sortino_like = (
        float(np.mean(rewards_array) / downside_std * np.sqrt(len(rewards_array)))
        if downside_std > 0
        else 0.0
    )
    max_consecutive_losses = 0
    current_loss_streak = 0
    for reward in rewards_array:
        if reward < 0:
            current_loss_streak += 1
            max_consecutive_losses = max(max_consecutive_losses, current_loss_streak)
        else:
            current_loss_streak = 0
    stats = {
        "balance": balance,
        "roi": pnl / initial_balance if initial_balance else 0.0,
        "pnl": pnl,
        "trades": trades,
        "win_rate": wins / trades if trades else 0.0,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown / initial_balance if initial_balance else 0.0,
        "avg_trade": float(np.mean(rewards_array)) if rewards else 0.0,
        "median_trade": float(np.median(rewards_array)) if rewards else 0.0,
        "profit_factor": profit_factor,
        "sharpe_like": sharpe_like,
        "sortino_like": sortino_like,
        "max_consecutive_losses": max_consecutive_losses,
        "forced_exits": forced,
        "stop_loss_exits": sl_count,
        "take_profit_exits": tp_count,
        "timeout_exits": timeout_count,
        "actions": actions,
        "initial_balance": initial_balance,
        "stopped_out": stopped_out,
    }
    stats["score"] = score_stats(stats)
    return stats


def simulate_rule(
    prices,
    buy_prob,
    sell_prob,
    min_conf,
    min_edge,
    stop_loss,
    take_profit,
    max_hold,
    close_on_opposite,
    direction_mode,
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
    forced = 0
    sl_count = 0
    tp_count = 0
    timeout_count = 0

    for i, curr_price in enumerate(prices):
        bp = buy_prob[i]
        sp = sell_prob[i]
        signal = 1 if bp >= sp else 2
        conf = bp if bp >= sp else sp
        edge = abs(bp - sp)
        has_signal = conf >= min_conf and edge >= min_edge

        if direction_mode == "long" and signal == 2:
            has_signal = False
        elif direction_mode == "short" and signal == 1:
            has_signal = False

        if position == 0:
            if has_signal:
                position = signal
                entry_price = curr_price
                hold_ticks = 0
                actions[signal] += 1
            else:
                actions[0] += 1
        else:
            hold_ticks += 1
            float_pnl = curr_price - entry_price if position == 1 else entry_price - curr_price
            exit_reason = None
            if close_on_opposite and has_signal and signal != position:
                exit_reason = "model"
            elif float_pnl < -stop_loss:
                exit_reason = "stop_loss"
            elif take_profit is not None and float_pnl > take_profit:
                exit_reason = "take_profit"
            elif hold_ticks > max_hold:
                exit_reason = "timeout"

            if exit_reason is None:
                actions[position] += 1
            else:
                reward = close_reward(position, entry_price, curr_price)
                balance += reward
                rewards.append(reward)
                trades += 1
                wins += int(reward > 0)
                forced += int(exit_reason != "model")
                sl_count += int(exit_reason == "stop_loss")
                tp_count += int(exit_reason == "take_profit")
                timeout_count += int(exit_reason == "timeout")
                position = 0
                entry_price = 0.0
                hold_ticks = 0
                actions[0] += 1

        balance_history.append(balance)

    if position:
        reward = close_reward(position, entry_price, prices[-1])
        balance += reward
        rewards.append(reward)
        trades += 1
        wins += int(reward > 0)
        forced += 1
        balance_history[-1] = balance

    return summarize(balance_history, rewards, trades, wins, actions, forced, sl_count, tp_count, timeout_count)


def make_probs(experts, df):
    probs = experts.predict_probs(df)
    buy_prob = ((probs[:, 1] + probs[:, 4]) / 2.0).astype(np.float32)
    sell_prob = ((probs[:, 2] + probs[:, 5]) / 2.0).astype(np.float32)
    return buy_prob, sell_prob


def evaluate(params, prices, buy_prob, sell_prob):
    return simulate_rule(prices=prices, buy_prob=buy_prob, sell_prob=sell_prob, **params)


def evaluate_segments(params, segments):
    stats = [
        evaluate(params, prices, buy_prob, sell_prob)
        for _, prices, buy_prob, sell_prob in segments
    ]
    return combined_score(stats), stats


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Rule search using expert probabilities; torch device visible: {device}")
    df, features = load_and_prepare_data()
    expert_split = int(len(df) * EXPERT_TRAIN_END_RATIO)
    val_start = int(len(df) * VAL_START_RATIO)
    val_end = int(len(df) * VAL_END_RATIO)

    train_tail = df.iloc[expert_split:val_start].copy().reset_index(drop=True)
    val_df = df.iloc[val_start:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)
    print(
        f"Rows | train_tail={len(train_tail):,} val={len(val_df):,} "
        f"test={len(test_df):,} label_lookahead={LABEL_LOOKAHEAD}"
    )

    experts = SavedExperts(features)
    experts.load("gold_mtf")
    train_tail_buy, train_tail_sell = make_probs(experts, train_tail)
    val_buy, val_sell = make_probs(experts, val_df)
    test_buy, test_sell = make_probs(experts, test_df)
    train_tail_prices = train_tail["CLOSE"].to_numpy(dtype=np.float64)
    val_prices = val_df["CLOSE"].to_numpy(dtype=np.float64)
    test_prices = test_df["CLOSE"].to_numpy(dtype=np.float64)
    validation_segments = [
        ("train_tail", train_tail_prices, train_tail_buy, train_tail_sell),
        ("validation", val_prices, val_buy, val_sell),
    ]

    rough_results = []
    rough_grid = product(
        [0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        [0.00, 0.03, 0.06, 0.10],
        [False, True],
        ["both", "long", "short"],
    )
    for min_conf, min_edge, close_on_opposite, direction_mode in rough_grid:
        params = {
            "min_conf": min_conf,
            "min_edge": min_edge,
            "stop_loss": 0.2,
            "take_profit": None,
            "max_hold": 240,
            "close_on_opposite": close_on_opposite,
            "direction_mode": direction_mode,
        }
        score, stats = evaluate_segments(params, validation_segments)
        rough_results.append((score, params, stats))

    rough_results.sort(key=lambda item: item[0], reverse=True)
    print("Top rough validation candidates:")
    for rank, (score, params, stats_list) in enumerate(rough_results[:8], start=1):
        print(f"#{rank} combined_score={score:.2f} {params}")
        for (segment_name, *_), stats in zip(validation_segments, stats_list):
            print("   " + format_stats(segment_name, stats))

    refined_results = []
    seen = set()
    for _, base_params, _ in rough_results[:8]:
        for stop_loss, take_profit, max_hold in product(
            [0.2, 0.5, 1.0, 2.0],
            [None, 0.8, 1.5, 3.0],
            [60, 120, 240, 480],
        ):
            params = dict(base_params)
            params.update(
                {
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "max_hold": max_hold,
                }
            )
            key = tuple(sorted(params.items(), key=lambda item: item[0]))
            if key in seen:
                continue
            seen.add(key)
            score, stats = evaluate_segments(params, validation_segments)
            refined_results.append((score, params, stats))

    refined_results.sort(key=lambda item: item[0], reverse=True)
    print("Top refined validation candidates with test check:")
    for rank, (score, params, val_stats_list) in enumerate(refined_results[:12], start=1):
        test_stats = evaluate(params, test_prices, test_buy, test_sell)
        print(f"#{rank} combined_score={score:.2f} params={params}")
        for (segment_name, *_), stats in zip(validation_segments, val_stats_list):
            print("   " + format_stats(segment_name, stats))
        print("   " + format_stats("test", test_stats))


if __name__ == "__main__":
    main()
