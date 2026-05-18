import os
from itertools import product

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import torch
import xgboost as xgb

from drl_train_candidate import format_stats
from drl_trading_v2 import (
    EXPERT_TRAIN_END_RATIO,
    INITIAL_BALANCE,
    SPREAD_POINTS,
    load_and_prepare_data,
)
from strategy_grid_search import combined_score, summarize


HORIZON = 240
VAL_START_RATIO = 0.80
VAL_END_RATIO = 0.85
LABEL_TP_ATR = 1.8
LABEL_SL_ATR = 1.2
MIN_TP_PRICE = 1.0
MIN_SL_PRICE = 0.8
MODEL_PATH = "gold_barrier_xgb.json"
NO_TRADE_SAMPLE_WEIGHT = 1.0
MAX_PROFIT_SAMPLE_WEIGHT = 2.0


def forward_rolling(series, window, op):
    future = series.shift(-1)
    rev = future.iloc[::-1]
    if op == "max":
        rolled = rev.rolling(window, min_periods=1).max()
    elif op == "min":
        rolled = rev.rolling(window, min_periods=1).min()
    else:
        raise ValueError(op)
    return rolled.iloc[::-1].to_numpy()


def build_barrier_target(df):
    close = df["CLOSE"].to_numpy(dtype=np.float64)
    atr = df["ATR"].to_numpy(dtype=np.float64)
    tp = np.maximum(atr * LABEL_TP_ATR, MIN_TP_PRICE)
    sl = np.maximum(atr * LABEL_SL_ATR, MIN_SL_PRICE)
    future_high = forward_rolling(df["HIGH"], HORIZON, "max")
    future_low = forward_rolling(df["LOW"], HORIZON, "min")

    up = future_high - close
    down = close - future_low
    long_clean = (up >= tp) & (down < sl)
    short_clean = (down >= tp) & (up < sl)

    target = np.zeros(len(df), dtype=np.int8)
    target[long_clean & ~short_clean] = 1
    target[short_clean & ~long_clean] = 2
    return target


def build_profit_sample_weight(df, target):
    close = df["CLOSE"].to_numpy(dtype=np.float64)
    atr = df["ATR"].to_numpy(dtype=np.float64)
    tp = np.maximum(atr * LABEL_TP_ATR, MIN_TP_PRICE)
    sl = np.maximum(atr * LABEL_SL_ATR, MIN_SL_PRICE)
    future_high = forward_rolling(df["HIGH"], HORIZON, "max")
    future_low = forward_rolling(df["LOW"], HORIZON, "min")

    long_favorable = future_high - close
    long_adverse = close - future_low
    short_favorable = close - future_low
    short_adverse = future_high - close

    quality = np.zeros(len(df), dtype=np.float64)
    long_mask = target == 1
    short_mask = target == 2
    quality[long_mask] = (
        long_favorable[long_mask] / (tp[long_mask] + 1e-9)
        - long_adverse[long_mask] / (sl[long_mask] + 1e-9)
    )
    quality[short_mask] = (
        short_favorable[short_mask] / (tp[short_mask] + 1e-9)
        - short_adverse[short_mask] / (sl[short_mask] + 1e-9)
    )

    class_counts = np.bincount(target.astype(np.int64), minlength=3)
    class_weight = len(target) / (3.0 * np.maximum(class_counts, 1))
    weights = class_weight[target.astype(np.int64)]
    weights[target == 0] *= NO_TRADE_SAMPLE_WEIGHT
    trade_boost = 1.0 + np.clip(quality, 0.0, MAX_PROFIT_SAMPLE_WEIGHT - 1.0)
    weights[target != 0] *= trade_boost[target != 0]
    return weights


def close_reward(position, entry_price, curr_price, extra_cost_points=0.0):
    pnl = curr_price - entry_price if position == 1 else entry_price - curr_price
    trading_cost = (SPREAD_POINTS + extra_cost_points) * 0.01 * 100.0
    return (pnl * 100.0) - trading_cost


def simulate_barrier(
    prices,
    atr,
    probs,
    threshold,
    edge_threshold,
    tp_atr,
    sl_atr,
    min_tp_price,
    min_sl_price,
    max_hold,
    cooldown_ticks,
    close_on_opposite,
    direction_mode,
    initial_balance=INITIAL_BALANCE,
    stop_out_balance=None,
    risk_per_trade=None,
    hours=None,
    weekdays=None,
    dates=None,
    months=None,
    allowed_entry_hours=None,
    allowed_entry_weekdays=None,
    allowed_entry_months=None,
    rsi_values=None,
    excluded_rsi_ranges=None,
    vola_ratio_values=None,
    min_vola_ratio=None,
    max_vola_ratio=None,
    trend_score_values=None,
    min_trend_score=None,
    max_trend_score=None,
    max_daily_loss_pct=None,
    max_daily_trades=None,
    entry_quality=None,
    min_entry_quality=None,
    entry_risk_mult=None,
    extra_cost_points=0.0,
    drawdown_guard_start_pct=None,
    drawdown_guard_full_pct=None,
    drawdown_guard_min_risk_mult=0.5,
    loss_streak_risk_mult=1.0,
    loss_streak_threshold=None,
    loss_streak_pause_threshold=None,
    loss_streak_pause_ticks=0,
    rolling_guard_window=None,
    rolling_guard_min_trades=20,
    rolling_guard_min_profit_factor=None,
    rolling_guard_min_win_rate=None,
    rolling_guard_risk_mult=1.0,
    rolling_guard_pause_ticks=0,
):
    balance = initial_balance
    balance_history = [balance]
    position = 0
    position_scale = 1.0
    entry_price = 0.0
    entry_tp = 0.0
    entry_sl = 0.0
    hold_ticks = 0
    trades = 0
    wins = 0
    rewards = []
    actions = {0: 0, 1: 0, 2: 0}
    forced = 0
    sl_count = 0
    tp_count = 0
    timeout_count = 0
    cooldown = 0
    prev_signal = 0
    stopped_out = False
    allowed_entry_hours = (
        None if allowed_entry_hours is None else set(allowed_entry_hours)
    )
    allowed_entry_weekdays = (
        None if allowed_entry_weekdays is None else set(allowed_entry_weekdays)
    )
    allowed_entry_months = (
        None if allowed_entry_months is None else set(allowed_entry_months)
    )
    excluded_rsi_ranges = excluded_rsi_ranges or []
    current_date = None
    day_start_balance = balance
    daily_trades = 0
    daily_locked = False
    peak_balance = balance
    loss_streak = 0
    trade_pause = 0

    for i, curr_price in enumerate(prices):
        if dates is not None:
            bar_date = dates[i]
            if current_date is None or bar_date != current_date:
                current_date = bar_date
                day_start_balance = balance
                daily_trades = 0
                daily_locked = False

        p = probs[i]
        buy_prob = float(p[1])
        sell_prob = float(p[2])
        signal = 1 if buy_prob >= sell_prob else 2
        conf = buy_prob if signal == 1 else sell_prob
        edge = abs(buy_prob - sell_prob)
        has_signal = conf >= threshold and edge >= edge_threshold
        if direction_mode == "long" and signal == 2:
            has_signal = False
        elif direction_mode == "short" and signal == 1:
            has_signal = False
        if (
            has_signal
            and allowed_entry_hours is not None
            and hours is not None
            and int(hours[i]) not in allowed_entry_hours
        ):
            has_signal = False
        if (
            has_signal
            and allowed_entry_weekdays is not None
            and weekdays is not None
            and int(weekdays[i]) not in allowed_entry_weekdays
        ):
            has_signal = False
        if (
            has_signal
            and allowed_entry_months is not None
            and months is not None
            and int(months[i]) not in allowed_entry_months
        ):
            has_signal = False
        if has_signal and rsi_values is not None:
            curr_rsi = float(rsi_values[i])
            for low, high in excluded_rsi_ranges:
                if low <= curr_rsi <= high:
                    has_signal = False
                    break
        if has_signal and vola_ratio_values is not None:
            curr_vola_ratio = float(vola_ratio_values[i])
            if min_vola_ratio is not None and curr_vola_ratio < min_vola_ratio:
                has_signal = False
            if max_vola_ratio is not None and curr_vola_ratio > max_vola_ratio:
                has_signal = False
        if has_signal and trend_score_values is not None:
            curr_trend_score = float(trend_score_values[i])
            if min_trend_score is not None and curr_trend_score < min_trend_score:
                has_signal = False
            if max_trend_score is not None and curr_trend_score > max_trend_score:
                has_signal = False
        if has_signal and daily_locked:
            has_signal = False
        if has_signal and trade_pause > 0:
            has_signal = False
        if (
            has_signal
            and max_daily_trades is not None
            and daily_trades >= max_daily_trades
        ):
            has_signal = False
        if (
            has_signal
            and entry_quality is not None
            and min_entry_quality is not None
            and float(entry_quality[i]) < min_entry_quality
        ):
            has_signal = False

        if position == 0:
            can_enter = has_signal and cooldown <= 0 and signal != prev_signal
            if can_enter:
                position = signal
                entry_price = curr_price
                entry_tp = max(atr[i] * tp_atr, min_tp_price)
                entry_sl = max(atr[i] * sl_atr, min_sl_price)
                if risk_per_trade is None:
                    position_scale = 1.0
                else:
                    peak_balance = max(peak_balance, balance)
                    risk_mult = 1.0
                    if (
                        drawdown_guard_start_pct is not None
                        and drawdown_guard_full_pct is not None
                        and drawdown_guard_full_pct > drawdown_guard_start_pct
                        and peak_balance > 0
                    ):
                        drawdown_pct = max(0.0, 1.0 - (balance / peak_balance))
                        if drawdown_pct >= drawdown_guard_start_pct:
                            guard_span = drawdown_guard_full_pct - drawdown_guard_start_pct
                            guard_progress = min(
                                (drawdown_pct - drawdown_guard_start_pct) / guard_span,
                                1.0,
                            )
                            min_mult = max(0.0, min(drawdown_guard_min_risk_mult, 1.0))
                            risk_mult *= 1.0 - ((1.0 - min_mult) * guard_progress)
                    if (
                        loss_streak_threshold is not None
                        and loss_streak >= loss_streak_threshold
                    ):
                        risk_mult *= max(0.0, min(loss_streak_risk_mult, 1.0))
                    if rolling_guard_window is not None and len(rewards) >= rolling_guard_min_trades:
                        recent_rewards = np.asarray(
                            rewards[-int(rolling_guard_window) :],
                            dtype=np.float64,
                        )
                        gross_profit = float(recent_rewards[recent_rewards > 0].sum())
                        gross_loss = float(-recent_rewards[recent_rewards < 0].sum())
                        recent_pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
                        recent_win_rate = float((recent_rewards > 0).mean()) if len(recent_rewards) else 0.0
                        guard_triggered = False
                        if (
                            rolling_guard_min_profit_factor is not None
                            and recent_pf < rolling_guard_min_profit_factor
                        ):
                            guard_triggered = True
                        if (
                            rolling_guard_min_win_rate is not None
                            and recent_win_rate < rolling_guard_min_win_rate
                        ):
                            guard_triggered = True
                        if guard_triggered:
                            risk_mult *= max(0.0, min(rolling_guard_risk_mult, 1.0))
                    stop_cost = (entry_sl * 100.0) + (SPREAD_POINTS * 0.01 * 100.0)
                    if entry_risk_mult is not None:
                        risk_mult *= max(0.0, float(entry_risk_mult[i]))
                    risk_budget = max(balance, 0.0) * risk_per_trade * risk_mult
                    position_scale = risk_budget / max(stop_cost, 1e-9)
                hold_ticks = 0
                actions[signal] += 1
                daily_trades += 1
            else:
                actions[0] += 1
                if cooldown > 0:
                    cooldown -= 1
        else:
            hold_ticks += 1
            float_pnl = curr_price - entry_price if position == 1 else entry_price - curr_price
            exit_reason = None
            if close_on_opposite and has_signal and signal != position:
                exit_reason = "model"
            elif float_pnl >= entry_tp:
                exit_reason = "take_profit"
            elif float_pnl <= -entry_sl:
                exit_reason = "stop_loss"
            elif hold_ticks >= max_hold:
                exit_reason = "timeout"

            if exit_reason is None:
                actions[position] += 1
            else:
                reward = (
                    close_reward(
                        position,
                        entry_price,
                        curr_price,
                        extra_cost_points=extra_cost_points,
                    )
                    * position_scale
                )
                balance += reward
                rewards.append(reward)
                trades += 1
                is_win = reward > 0
                wins += int(is_win)
                loss_streak = 0 if is_win else loss_streak + 1
                peak_balance = max(peak_balance, balance)
                if (
                    not is_win
                    and loss_streak_pause_threshold is not None
                    and loss_streak >= loss_streak_pause_threshold
                ):
                    trade_pause = max(trade_pause, int(loss_streak_pause_ticks))
                if rolling_guard_window is not None and len(rewards) >= rolling_guard_min_trades:
                    recent_rewards = np.asarray(
                        rewards[-int(rolling_guard_window) :],
                        dtype=np.float64,
                    )
                    gross_profit = float(recent_rewards[recent_rewards > 0].sum())
                    gross_loss = float(-recent_rewards[recent_rewards < 0].sum())
                    recent_pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
                    recent_win_rate = float((recent_rewards > 0).mean()) if len(recent_rewards) else 0.0
                    guard_triggered = False
                    if (
                        rolling_guard_min_profit_factor is not None
                        and recent_pf < rolling_guard_min_profit_factor
                    ):
                        guard_triggered = True
                    if (
                        rolling_guard_min_win_rate is not None
                        and recent_win_rate < rolling_guard_min_win_rate
                    ):
                        guard_triggered = True
                    if guard_triggered:
                        trade_pause = max(trade_pause, int(rolling_guard_pause_ticks))
                forced += int(exit_reason != "model")
                sl_count += int(exit_reason == "stop_loss")
                tp_count += int(exit_reason == "take_profit")
                timeout_count += int(exit_reason == "timeout")
                position = 0
                position_scale = 1.0
                entry_price = 0.0
                entry_tp = 0.0
                entry_sl = 0.0
                hold_ticks = 0
                cooldown = cooldown_ticks
                actions[0] += 1
                if stop_out_balance is not None and balance <= stop_out_balance:
                    balance = stop_out_balance
                    balance_history.append(balance)
                    stopped_out = True
                    break
                if (
                    max_daily_loss_pct is not None
                    and balance <= day_start_balance * (1.0 - max_daily_loss_pct)
                ):
                    daily_locked = True
        if position == 0 and trade_pause > 0:
            trade_pause -= 1

        balance_history.append(balance)
        prev_signal = signal if has_signal else 0

    if position and not stopped_out:
        reward = (
            close_reward(
                position,
                entry_price,
                prices[-1],
                extra_cost_points=extra_cost_points,
            )
            * position_scale
        )
        balance += reward
        if stop_out_balance is not None and balance <= stop_out_balance:
            balance = stop_out_balance
            stopped_out = True
        rewards.append(reward)
        trades += 1
        is_win = reward > 0
        wins += int(is_win)
        loss_streak = 0 if is_win else loss_streak + 1
        peak_balance = max(peak_balance, balance)
        forced += 1
        balance_history[-1] = balance

    return summarize(
        balance_history,
        rewards,
        trades,
        wins,
        actions,
        forced,
        sl_count,
        tp_count,
        timeout_count,
        initial_balance=initial_balance,
        stopped_out=stopped_out,
    )


def evaluate(
    params,
    prices,
    atr,
    probs,
    hours=None,
    weekdays=None,
    dates=None,
    months=None,
    rsi_values=None,
    vola_ratio_values=None,
    trend_score_values=None,
    entry_quality=None,
    entry_risk_mult=None,
):
    return simulate_barrier(
        prices=prices,
        atr=atr,
        probs=probs,
        hours=hours,
        weekdays=weekdays,
        dates=dates,
        months=months,
        rsi_values=rsi_values,
        vola_ratio_values=vola_ratio_values,
        trend_score_values=trend_score_values,
        entry_quality=entry_quality,
        entry_risk_mult=entry_risk_mult,
        **params,
    )


def evaluate_segments(params, segments):
    stats = [evaluate(params, prices, atr, probs) for _, prices, atr, probs in segments]
    return combined_score(stats), stats


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training barrier classifier; torch device visible: {device}")
    df, features = load_and_prepare_data()
    df = df.copy()
    df["BARRIER_TARGET"] = build_barrier_target(df)
    df = df.iloc[:-HORIZON].dropna(subset=features + ["BARRIER_TARGET", "ATR"]).reset_index(drop=True)

    expert_split = int(len(df) * EXPERT_TRAIN_END_RATIO)
    val_start = int(len(df) * VAL_START_RATIO)
    val_end = int(len(df) * VAL_END_RATIO)
    train_df = df.iloc[: max(0, expert_split - HORIZON)].copy()
    train_tail = df.iloc[expert_split:val_start].copy().reset_index(drop=True)
    val_df = df.iloc[val_start:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)

    print(
        f"Rows | train={len(train_df):,} train_tail={len(train_tail):,} "
        f"val={len(val_df):,} test={len(test_df):,}"
    )
    for name, part in [
        ("train", train_df),
        ("train_tail", train_tail),
        ("validation", val_df),
        ("test", test_df),
    ]:
        counts = part["BARRIER_TARGET"].value_counts(normalize=True).sort_index()
        print(f"{name} target ratio: {counts.to_dict()}")

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
    model.save_model(MODEL_PATH)
    print(f"Saved classifier as {MODEL_PATH}")

    train_tail_probs = model.predict_proba(train_tail[features]).astype(np.float32)
    val_probs = model.predict_proba(val_df[features]).astype(np.float32)
    test_probs = model.predict_proba(test_df[features]).astype(np.float32)

    train_tail_prices = train_tail["CLOSE"].to_numpy(dtype=np.float64)
    val_prices = val_df["CLOSE"].to_numpy(dtype=np.float64)
    test_prices = test_df["CLOSE"].to_numpy(dtype=np.float64)
    train_tail_atr = train_tail["ATR"].to_numpy(dtype=np.float64)
    val_atr = val_df["ATR"].to_numpy(dtype=np.float64)
    test_atr = test_df["ATR"].to_numpy(dtype=np.float64)
    validation_segments = [
        ("train_tail", train_tail_prices, train_tail_atr, train_tail_probs),
        ("validation", val_prices, val_atr, val_probs),
    ]

    rough_results = []
    for threshold, edge_threshold, direction_mode in product(
        [0.45, 0.55, 0.65, 0.75],
        [0.00, 0.05, 0.10],
        ["both", "long", "short"],
    ):
        params = {
            "threshold": threshold,
            "edge_threshold": edge_threshold,
            "tp_atr": LABEL_TP_ATR,
            "sl_atr": LABEL_SL_ATR,
            "min_tp_price": MIN_TP_PRICE,
            "min_sl_price": MIN_SL_PRICE,
            "max_hold": HORIZON,
            "cooldown_ticks": 60,
            "close_on_opposite": False,
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
        for tp_atr, sl_atr, max_hold, cooldown_ticks, close_on_opposite in product(
            [0.9, 1.2, 1.6],
            [1.6, 2.4, 3.2],
            [90, 180, 360],
            [0, 60, 240],
            [False, True],
        ):
            params = dict(base_params)
            params.update(
                {
                    "tp_atr": tp_atr,
                    "sl_atr": sl_atr,
                    "max_hold": max_hold,
                    "cooldown_ticks": cooldown_ticks,
                    "close_on_opposite": close_on_opposite,
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
        test_stats = evaluate(params, test_prices, test_atr, test_probs)
        print(f"#{rank} combined_score={score:.2f} params={params}")
        for (segment_name, *_), stats in zip(validation_segments, val_stats_list):
            print("   " + format_stats(segment_name, stats))
        print("   " + format_stats("test", test_stats))


if __name__ == "__main__":
    main()
