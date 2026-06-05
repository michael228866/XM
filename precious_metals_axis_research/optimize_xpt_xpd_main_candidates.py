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

import numpy as np  # noqa: E402

from precious_metals_axis_research.axis_timeframe_smoke import load_case  # noqa: E402
from precious_metals_axis_research.cost_aware_xaueur_m5 import get_symbol_point  # noqa: E402
from precious_metals_axis_research.optimize_training_profiles_silver_xaueur import (  # noqa: E402
    MODEL_PROFILES,
    train_model,
)
from precious_metals_axis_research.readiness_silver_xaueur import (  # noqa: E402
    fold_pass,
    scale_spread,
)
from precious_metals_axis_research.walk_forward_long_tf_cost import (  # noqa: E402
    FOLDS,
    compact_period,
    slice_by_ratio,
)


OUTPUT_CSV = RESEARCH_DIR / "xpt_xpd_main_candidate_results.csv"
OUTPUT_JSON = RESEARCH_DIR / "xpt_xpd_main_candidate_results.json"
OUTPUT_MD = RESEARCH_DIR / "xpt_xpd_main_candidate_report.md"
OUTPUT_BEST = RESEARCH_DIR / "xpt_xpd_main_candidate_best.json"

SYMBOLS = ["XPTUSD#", "XPDUSD#"]
COST_MULTIPLIERS = [1.0, 2.0, 3.0, 4.0, 5.0]
TREND_COLUMNS = [
    "H4_TREND",
    "H8_TREND",
    "H12_TREND",
    "Daily_TREND",
    "Weekly_TREND",
    "Monthly_TREND",
]

FILTERS = [
    {
        "filter_name": "none",
        "trend_min": -1.0,
        "rsi_min": 0.0,
        "rsi_max": 100.0,
        "vola_max": 99.0,
        "spread_atr_max": 99.0,
        "macd_min": -999.0,
    },
    {
        "filter_name": "low_vola",
        "trend_min": -1.0,
        "rsi_min": 0.0,
        "rsi_max": 100.0,
        "vola_max": 1.2,
        "spread_atr_max": 99.0,
        "macd_min": -999.0,
    },
    {
        "filter_name": "trend_nonnegative",
        "trend_min": 0.0,
        "rsi_min": 0.0,
        "rsi_max": 100.0,
        "vola_max": 99.0,
        "spread_atr_max": 99.0,
        "macd_min": -999.0,
    },
    {
        "filter_name": "not_overbought",
        "trend_min": -1.0,
        "rsi_min": 0.0,
        "rsi_max": 85.0,
        "vola_max": 99.0,
        "spread_atr_max": 99.0,
        "macd_min": -999.0,
    },
    {
        "filter_name": "stable_combo",
        "trend_min": 0.0,
        "rsi_min": 0.0,
        "rsi_max": 85.0,
        "vola_max": 1.6,
        "spread_atr_max": 0.75,
        "macd_min": -999.0,
    },
    {
        "filter_name": "momentum_combo",
        "trend_min": 0.0,
        "rsi_min": 45.0,
        "rsi_max": 85.0,
        "vola_max": 1.8,
        "spread_atr_max": 0.75,
        "macd_min": 0.0,
    },
]


def param_grid():
    for threshold, edge, tp, sl, hold, direction in product(
        [0.54, 0.60, 0.66],
        [0.0, 0.05],
        [2.0, 3.2],
        [3.4, 5.2],
        [72, 168, 288],
        ["long", "short", "both"],
    ):
        if tp > sl:
            continue
        yield {
            "threshold": threshold,
            "edge_threshold": edge,
            "tp_atr": tp,
            "sl_atr": sl,
            "max_hold": hold,
            "direction_mode": direction,
        }


def prepare_symbol(symbol: str, profile_name: str, profile: dict) -> list[dict]:
    frame, features = load_case(symbol, "H1")
    point = get_symbol_point(symbol)
    prepared = []
    for fold in FOLDS:
        train_df = slice_by_ratio(frame, *fold["train"])
        test_df = slice_by_ratio(frame, *fold["test"])
        print(
            f"{profile_name} {symbol} {fold['name']}: "
            f"train={len(train_df):,} test={len(test_df):,}"
        )
        model = train_model(train_df, features, profile)
        probs = model.predict_proba(test_df[features]).astype(np.float32)
        prepared.append(
            {
                "fold": fold["name"],
                "test_period": compact_period(test_df),
                "test_df": test_df,
                "probs": probs,
                "point": point,
            }
        )
    return prepared


def simulate_filtered(frame, probs, params: dict, point: float) -> dict:
    close = frame["CLOSE"].to_numpy(dtype=np.float64)
    atr = frame["ATR"].to_numpy(dtype=np.float64)
    spread_points = (
        frame["SPREAD"].fillna(0).to_numpy(dtype=np.float64)
        if "SPREAD" in frame.columns
        else np.zeros(len(frame), dtype=np.float64)
    )
    trend_score = frame[TREND_COLUMNS].mean(axis=1).to_numpy(dtype=np.float64)
    rsi = frame["BASE_RSI"].to_numpy(dtype=np.float64)
    vola = frame["VOLA_RATIO"].to_numpy(dtype=np.float64)
    macd = frame["MACD_ATR"].to_numpy(dtype=np.float64)
    spread_atr = np.divide(
        spread_points * point,
        np.maximum(atr, 1e-9),
        out=np.zeros(len(frame), dtype=np.float64),
        where=atr > 0,
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
        has_signal = (
            conf >= params["threshold"]
            and edge >= params["edge_threshold"]
            and trend_score[i] >= params["trend_min"]
            and params["rsi_min"] <= rsi[i] <= params["rsi_max"]
            and vola[i] <= params["vola_max"]
            and spread_atr[i] <= params["spread_atr_max"]
            and macd[i] >= params["macd_min"]
        )
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

        reward_r = (gross - entry_cost) / entry_sl
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


def aggregate(symbol: str, fold_rows: list[dict]) -> dict:
    total_trades = sum(row["trades"] for row in fold_rows)
    total_r = sum(row["pnl_r"] for row in fold_rows)
    weighted_win = (
        sum(row["win_rate"] * row["trades"] for row in fold_rows) / total_trades
        if total_trades
        else 0.0
    )
    positive = sum(row["pnl_r"] > 0 for row in fold_rows)
    passed = sum(row["fold_pass"] for row in fold_rows)
    mean_pf = sum(row["profit_factor"] for row in fold_rows) / len(fold_rows)
    worst = min(row["pnl_r"] for row in fold_rows)
    max_dd = min(row["max_drawdown_r"] for row in fold_rows)
    recent = fold_rows[-1]
    gate = (
        positive == 4
        and passed >= 3
        and total_r >= 8.0
        and total_trades >= 32
        and recent["pnl_r"] > 0
        and recent["profit_factor"] >= 1.10
        and max_dd >= -12.0
    )
    score = (
        total_r * 170.0
        + positive * 800.0
        + passed * 500.0
        + weighted_win * 650.0
        + min(mean_pf, 4.0) * 220.0
        - abs(max_dd) * 50.0
        + min(worst, 0.0) * 240.0
    )
    return {
        "symbol": symbol,
        "total_r": round(total_r, 4),
        "trades": total_trades,
        "positive_folds": positive,
        "passed_folds": passed,
        "weighted_win_rate": round(weighted_win, 4),
        "mean_profit_factor": round(mean_pf, 4),
        "worst_fold_r": round(worst, 4),
        "max_drawdown_r": round(max_dd, 4),
        "recent_paper_r": recent["pnl_r"],
        "gate": gate,
        "score": round(score, 4),
        "folds": fold_rows,
    }


def evaluate(symbol: str, params: dict, prepared: list[dict], cost_multiplier: float) -> dict:
    fold_rows = []
    for fold in prepared:
        stats = simulate_filtered(
            scale_spread(fold["test_df"], cost_multiplier),
            fold["probs"],
            params,
            fold["point"],
        )
        fold_rows.append(
            {
                "fold": fold["fold"],
                "test_start": fold["test_period"]["start"],
                "test_end": fold["test_period"]["end"],
                "pnl_r": stats["pnl_r"],
                "trades": stats["trades"],
                "win_rate": stats["win_rate"],
                "profit_factor": stats["profit_factor"],
                "max_drawdown_r": stats["max_drawdown_r"],
                "avg_r": stats["avg_r"],
                "max_loss_streak": stats["max_loss_streak"],
                "fold_pass": fold_pass(stats),
            }
        )
    return {**params, "cost_multiplier": cost_multiplier, **aggregate(symbol, fold_rows)}


def choose_best(grid_results: list[dict]) -> dict:
    return sorted(
        grid_results,
        key=lambda row: (row["gate"], row["score"], row["total_r"], row["trades"]),
        reverse=True,
    )[0]


def flat_row(row: dict, profile_name: str, group: str) -> dict:
    return {key: value for key, value in {"profile": profile_name, "group": group, **row}.items() if key != "folds"}


def write_outputs(rows: list[dict], selected: dict) -> None:
    flat_rows = [flat_row(row["result"], row["profile"], row["group"]) for row in rows]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    OUTPUT_JSON.write_text(json.dumps({"rows": rows, "selected": selected}, indent=2), encoding="utf-8")
    OUTPUT_BEST.write_text(json.dumps(selected, indent=2), encoding="utf-8")

    lines = [
        "# XPT / XPD Main Candidate Optimization",
        "",
        "H1 symbol models with profile search, regime filters, and 1x-5x spread stress.",
        "",
        "## Selected",
        "",
        "| Symbol | Profile | Cost Gates | 3x Gate | 3x R | Trades | Win | PF | Worst R | DD | Params |",
        "|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for symbol in SYMBOLS:
        item = selected[symbol]
        row = item["best_3x"]
        lines.append(
            "| {symbol} | {profile} | {cost_gate_count}/5 | {gate} | {total_r:.2f} | "
            "{trades} | {weighted_win_rate:.2%} | {mean_profit_factor:.2f} | "
            "{worst_fold_r:.2f} | {max_drawdown_r:.2f} | "
            "{filter_name}: conf={threshold}, edge={edge_threshold}, "
            "tp/sl={tp_atr}/{sl_atr}, hold={max_hold}, dir={direction_mode} |".format(
                profile=item["profile"],
                cost_gate_count=item["cost_gate_count"],
                **row,
            )
        )

    lines.extend(
        [
            "",
            "## Cost Stress",
            "",
            "| Symbol | Profile | Cost | Gate | R | Positive | Passed | Trades | Win | Worst R | DD |",
            "|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for symbol in SYMBOLS:
        item = selected[symbol]
        for row in item["cost_stress"]:
            lines.append(
                "| {symbol} | {profile} | {cost_multiplier:.1f}x | {gate} | "
                "{total_r:.2f} | {positive_folds}/4 | {passed_folds}/4 | "
                "{trades} | {weighted_win_rate:.2%} | {worst_fold_r:.2f} | "
                "{max_drawdown_r:.2f} |".format(
                    profile=item["profile"],
                    **row,
                )
            )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = []
    candidates = {symbol: [] for symbol in SYMBOLS}
    for symbol in SYMBOLS:
        for profile_name, profile in MODEL_PROFILES.items():
            print(f"=== {symbol} profile={profile_name} ===")
            prepared = prepare_symbol(symbol, profile_name, profile)
            grid_results = []
            for base_params in param_grid():
                for filter_params in FILTERS:
                    params = {**base_params, **filter_params}
                    grid_results.append(evaluate(symbol, params, prepared, 3.0))
            best_3x = choose_best(grid_results)
            cost_stress = [
                evaluate(symbol, best_3x, prepared, cost)
                for cost in COST_MULTIPLIERS
            ]
            candidates[symbol].append(
                {
                    "profile": profile_name,
                    "best_3x": best_3x,
                    "cost_stress": cost_stress,
                    "cost_gate_count": sum(row["gate"] for row in cost_stress),
                    "cost_total_r": round(sum(row["total_r"] for row in cost_stress), 4),
                }
            )
            rows.extend(
                {"profile": profile_name, "group": f"{symbol}_grid_3x", "result": row}
                for row in grid_results
            )
            rows.extend(
                {"profile": profile_name, "group": f"{symbol}_selected_cost", "result": row}
                for row in cost_stress
            )

    selected = {}
    for symbol, items in candidates.items():
        selected[symbol] = sorted(
            items,
            key=lambda item: (
                item["cost_gate_count"],
                item["best_3x"]["gate"],
                item["best_3x"]["total_r"],
                item["cost_total_r"],
            ),
            reverse=True,
        )[0]
    write_outputs(rows, selected)
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_BEST}")
    for symbol, item in selected.items():
        row = item["best_3x"]
        print(
            f"{symbol}: profile={item['profile']} gates={item['cost_gate_count']}/5 "
            f"r3x={row['total_r']:.2f} gate3x={row['gate']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
