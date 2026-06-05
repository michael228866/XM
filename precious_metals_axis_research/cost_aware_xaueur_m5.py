from __future__ import annotations

import csv
import json
import os
import sys
from itertools import product
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

RESEARCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RESEARCH_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import MetaTrader5 as mt5  # noqa: E402
import numpy as np  # noqa: E402

from precious_metals_axis_research.axis_timeframe_smoke import (  # noqa: E402
    TRAIN_END_RATIO,
    VALIDATION_END_RATIO,
    load_case,
    train_model,
)


SYMBOL = "XAUEUR#"
BASE_TIMEFRAME = "M5"
OUTPUT_CSV = RESEARCH_DIR / "xaueur_m5_cost_aware_results.csv"
OUTPUT_JSON = RESEARCH_DIR / "xaueur_m5_cost_aware_results.json"
OUTPUT_MD = RESEARCH_DIR / "xaueur_m5_cost_aware_report.md"
OUTPUT_BEST = RESEARCH_DIR / "xaueur_m5_cost_aware_best.json"

DEFAULT_SYMBOL_POINTS = {
    "GOLD#": 0.01,
    "XAUEUR#": 0.01,
    "XPTUSD#": 0.01,
    "XPDUSD#": 0.01,
    "GAUCNH#": 0.01,
    "SILVER#": 0.001,
}


def get_symbol_point(symbol: str) -> float:
    fallback = DEFAULT_SYMBOL_POINTS.get(symbol, 0.01)
    if os.environ.get("PM_USE_MT5_POINT") != "1":
        return fallback
    if mt5.initialize():
        try:
            info = mt5.symbol_info(symbol)
            if info is not None and info.point > 0:
                return float(info.point)
        finally:
            mt5.shutdown()
    return fallback


def make_grid():
    for threshold, edge, tp, sl, hold, direction in product(
        [0.60, 0.62, 0.63, 0.64, 0.66],
        [0.05, 0.08, 0.10, 0.12, 0.15],
        [1.3, 1.4, 1.5, 1.6],
        [2.2, 2.5, 2.8, 3.0],
        [180, 240, 300],
        ["both", "short"],
    ):
        yield {
            "threshold": threshold,
            "edge_threshold": edge,
            "tp_atr": tp,
            "sl_atr": sl,
            "max_hold": hold,
            "direction_mode": direction,
        }


def simulate_cost_aware(frame, probs, params, point: float) -> dict:
    close = frame["CLOSE"].to_numpy(dtype=np.float64)
    atr = frame["ATR"].to_numpy(dtype=np.float64)
    spread_points = (
        frame["SPREAD"].fillna(0).to_numpy(dtype=np.float64)
        if "SPREAD" in frame.columns
        else np.zeros(len(frame), dtype=np.float64)
    )

    position = 0
    entry_price = 0.0
    entry_tp = 0.0
    entry_sl = 0.0
    entry_cost = 0.0
    hold = 0
    trades = 0
    wins = 0
    rewards = []
    equity = [0.0]

    for i, price in enumerate(close):
        buy_prob = float(probs[i, 1])
        sell_prob = float(probs[i, 2])
        signal = 1 if buy_prob >= sell_prob else 2
        conf = buy_prob if signal == 1 else sell_prob
        edge = abs(buy_prob - sell_prob)
        has_signal = conf >= params["threshold"] and edge >= params["edge_threshold"]
        if params["direction_mode"] == "long" and signal == 2:
            has_signal = False
        if params["direction_mode"] == "short" and signal == 1:
            has_signal = False

        if position == 0:
            if has_signal:
                position = signal
                entry_price = price
                entry_tp = max(float(atr[i]) * params["tp_atr"], 1e-9)
                entry_sl = max(float(atr[i]) * params["sl_atr"], 1e-9)
                entry_cost = max(float(spread_points[i]) * point, 0.0)
                hold = 0
            equity.append(equity[-1])
            continue

        hold += 1
        gross = price - entry_price if position == 1 else entry_price - price
        exit_now = (
            gross >= entry_tp
            or gross <= -entry_sl
            or hold >= params["max_hold"]
        )
        if not exit_now:
            equity.append(equity[-1])
            continue

        net_price_reward = gross - entry_cost
        reward_r = net_price_reward / entry_sl
        rewards.append(reward_r)
        trades += 1
        wins += int(reward_r > 0)
        equity.append(equity[-1] + reward_r)
        position = 0
        entry_price = 0.0
        entry_tp = 0.0
        entry_sl = 0.0
        entry_cost = 0.0
        hold = 0

    if position:
        gross = close[-1] - entry_price if position == 1 else entry_price - close[-1]
        reward_r = (gross - entry_cost) / entry_sl
        rewards.append(reward_r)
        trades += 1
        wins += int(reward_r > 0)
        equity[-1] += reward_r

    rewards_array = np.asarray(rewards, dtype=np.float64)
    equity_array = np.asarray(equity, dtype=np.float64)
    drawdown = equity_array - np.maximum.accumulate(equity_array)
    gross_profit = float(rewards_array[rewards_array > 0].sum()) if trades else 0.0
    gross_loss = float(-rewards_array[rewards_array < 0].sum()) if trades else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    max_loss_streak = 0
    loss_streak = 0
    for reward in rewards_array:
        if reward < 0:
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)
        else:
            loss_streak = 0

    return {
        "pnl_r": round(float(rewards_array.sum()) if trades else 0.0, 4),
        "trades": trades,
        "win_rate": round(wins / trades, 4) if trades else 0.0,
        "profit_factor": round(profit_factor, 4) if np.isfinite(profit_factor) else 999.0,
        "max_drawdown_r": round(float(drawdown.min()) if len(drawdown) else 0.0, 4),
        "avg_r": round(float(rewards_array.mean()) if trades else 0.0, 4),
        "max_loss_streak": max_loss_streak,
    }


def passes_gate(validation, test) -> bool:
    return (
        validation["pnl_r"] > 0
        and validation["profit_factor"] >= 1.05
        and test["pnl_r"] > 0
        and test["profit_factor"] >= 1.25
        and test["win_rate"] >= 0.62
        and test["trades"] >= 30
        and abs(test["max_drawdown_r"]) <= 18.0
    )


def score_row(row: dict) -> float:
    if row["test_trades"] < 30:
        return -100_000.0 + row["test_pnl_r"]
    return (
        row["test_pnl_r"] * 120.0
        + row["validation_pnl_r"] * 55.0
        + row["test_win_rate"] * 300.0
        + min(row["test_profit_factor"], 3.0) * 130.0
        - abs(row["test_max_drawdown_r"]) * 18.0
    )


def write_outputs(rows: list[dict]) -> None:
    rows = sorted(
        rows,
        key=lambda row: (row["passes_gate"], row["score"], row["test_pnl_r"]),
        reverse=True,
    )
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    OUTPUT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    OUTPUT_BEST.write_text(json.dumps(rows[0], indent=2), encoding="utf-8")

    lines = [
        "# XAUEUR M5 Cost-Aware Optimization",
        "",
        "Uses CSV `<SPREAD>` converted by MT5 symbol point and reports R-multiple results.",
        "",
        "| Rank | Pass | Score | Test R | Test Win | Test PF | Test Trades | Val R | Val PF | Params |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(rows[:20], start=1):
        lines.append(
            "| {rank} | {passes_gate} | {score:.1f} | {test_pnl_r:.2f} | "
            "{test_win_rate:.2%} | {test_profit_factor:.2f} | {test_trades} | "
            "{validation_pnl_r:.2f} | {validation_profit_factor:.2f} | "
            "conf={threshold}, edge={edge_threshold}, tp/sl={tp_atr}/{sl_atr}, "
            "hold={max_hold}, dir={direction_mode} |".format(rank=rank, **row)
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    point = get_symbol_point(SYMBOL)
    print(f"Loading {SYMBOL} {BASE_TIMEFRAME}; point={point}...")
    frame, features = load_case(SYMBOL, BASE_TIMEFRAME)
    train_end = int(len(frame) * TRAIN_END_RATIO)
    validation_end = int(len(frame) * VALIDATION_END_RATIO)
    train_df = frame.iloc[:train_end].copy()
    validation_df = frame.iloc[train_end:validation_end].copy()
    test_df = frame.iloc[validation_end:].copy()
    print(
        f"Rows train={len(train_df):,} validation={len(validation_df):,} "
        f"test={len(test_df):,}"
    )

    model = train_model(train_df, features)
    validation_probs = model.predict_proba(validation_df[features]).astype(np.float32)
    test_probs = model.predict_proba(test_df[features]).astype(np.float32)

    rows = []
    total = 0
    for params in make_grid():
        total += 1
        validation = simulate_cost_aware(validation_df, validation_probs, params, point)
        test = simulate_cost_aware(test_df, test_probs, params, point)
        row = {
            "symbol": SYMBOL,
            "base_timeframe": BASE_TIMEFRAME,
            **params,
            "validation_pnl_r": validation["pnl_r"],
            "validation_trades": validation["trades"],
            "validation_win_rate": validation["win_rate"],
            "validation_profit_factor": validation["profit_factor"],
            "validation_max_drawdown_r": validation["max_drawdown_r"],
            "test_pnl_r": test["pnl_r"],
            "test_trades": test["trades"],
            "test_win_rate": test["win_rate"],
            "test_profit_factor": test["profit_factor"],
            "test_max_drawdown_r": test["max_drawdown_r"],
            "test_avg_r": test["avg_r"],
            "test_max_loss_streak": test["max_loss_streak"],
            "passes_gate": passes_gate(validation, test),
        }
        row["score"] = round(score_row(row), 4)
        rows.append(row)

    write_outputs(rows)
    passed = sum(1 for row in rows if row["passes_gate"])
    print(f"Swept {total} candidates, passed {passed}.")
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_BEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
