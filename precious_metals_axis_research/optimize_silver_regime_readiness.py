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
from precious_metals_axis_research.readiness_silver_xaueur import (  # noqa: E402
    SILVER_FOLDS,
    fold_pass,
    scale_spread,
)
from precious_metals_axis_research.walk_forward_all_metals_shared import slice_by_ratio  # noqa: E402
from precious_metals_axis_research.walk_forward_long_tf_cost import (  # noqa: E402
    compact_period,
    train_fold_model,
)


OUTPUT_CSV = RESEARCH_DIR / "silver_regime_readiness_results.csv"
OUTPUT_JSON = RESEARCH_DIR / "silver_regime_readiness_results.json"
OUTPUT_MD = RESEARCH_DIR / "silver_regime_readiness_report.md"
OUTPUT_BEST = RESEARCH_DIR / "silver_regime_readiness_best.json"

TREND_COLUMNS = [
    "H4_TREND",
    "H8_TREND",
    "H12_TREND",
    "Daily_TREND",
    "Weekly_TREND",
    "Monthly_TREND",
]

BASE_PARAMS = [
    {
        "threshold": 0.58,
        "edge_threshold": 0.0,
        "tp_atr": 2.0,
        "sl_atr": 6.0,
        "max_hold": 216,
        "direction_mode": "long",
    },
    {
        "threshold": 0.56,
        "edge_threshold": 0.0,
        "tp_atr": 3.2,
        "sl_atr": 4.4,
        "max_hold": 120,
        "direction_mode": "long",
    },
    {
        "threshold": 0.62,
        "edge_threshold": 0.0,
        "tp_atr": 3.6,
        "sl_atr": 6.0,
        "max_hold": 216,
        "direction_mode": "long",
    },
]


def filter_grid():
    for trend_min, rsi_min, rsi_max, vola_max, spread_atr_max, macd_min in product(
        [0.0, 0.34, 0.67, 1.0],
        [0.0, 45.0, 50.0],
        [75.0, 85.0, 100.0],
        [1.2, 1.6, 2.5],
        [0.45, 0.75, 1.25, 99.0],
        [-999.0, 0.0],
    ):
        if rsi_min >= rsi_max:
            continue
        yield {
            "trend_min": trend_min,
            "rsi_min": rsi_min,
            "rsi_max": rsi_max,
            "vola_max": vola_max,
            "spread_atr_max": spread_atr_max,
            "macd_min": macd_min,
        }


def param_grid():
    for threshold, edge, tp, sl, hold in product(
        [0.54, 0.58, 0.62, 0.66, 0.70],
        [0.0, 0.08],
        [2.0, 2.6, 3.2, 3.8],
        [4.4, 5.2, 6.0, 7.0],
        [120, 216, 336],
    ):
        if tp > sl:
            continue
        yield {
            "threshold": threshold,
            "edge_threshold": edge,
            "tp_atr": tp,
            "sl_atr": sl,
            "max_hold": hold,
            "direction_mode": "long",
        }


def prepare_folds():
    frame, features = load_case("SILVER#", "H1")
    point = get_symbol_point("SILVER#")
    prepared = []
    for fold in SILVER_FOLDS:
        train_df = slice_by_ratio(frame, *fold["train"])
        test_df = slice_by_ratio(frame, *fold["test"])
        print(f"SILVER {fold['name']}: train={len(train_df):,} test={len(test_df):,}")
        model = train_fold_model(train_df, features)
        probs = model.predict_proba(test_df[features]).astype(np.float32)
        prepared.append(
            {
                "fold": fold["name"],
                "test_period": compact_period(test_df),
                "stress_df": scale_spread(test_df, 3.0),
                "probs": probs,
                "point": point,
            }
        )
    return prepared


def simulate_filtered(frame, probs, params, point: float) -> dict:
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
            signal == 1
            and conf >= params["threshold"]
            and edge >= params["edge_threshold"]
            and trend_score[i] >= params["trend_min"]
            and params["rsi_min"] <= rsi[i] <= params["rsi_max"]
            and vola[i] <= params["vola_max"]
            and spread_atr[i] <= params["spread_atr_max"]
            and macd[i] >= params["macd_min"]
        )

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
        gross = price - entry_price
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
        gross = close[-1] - entry_price
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


def evaluate(params: dict, prepared: list[dict]) -> dict:
    fold_rows = []
    for fold in prepared:
        stats = simulate_filtered(fold["stress_df"], fold["probs"], params, fold["point"])
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
    battle_gate = (
        positive == 5
        and passed >= 4
        and total_r >= 12.0
        and total_trades >= 60
        and recent["pnl_r"] > 0
        and recent["profit_factor"] >= 1.15
        and max_dd >= -10.0
    )
    score = (
        total_r * 180.0
        + positive * 800.0
        + passed * 550.0
        + weighted_win * 700.0
        + min(mean_pf, 4.0) * 250.0
        - abs(max_dd) * 45.0
        + min(worst, 0.0) * 250.0
        + min(recent["pnl_r"], 0.0) * 350.0
    )
    return {
        "symbol": "SILVER#",
        **params,
        "stress_total_r": round(total_r, 4),
        "stress_total_trades": total_trades,
        "stress_positive_folds": positive,
        "stress_passed_folds": passed,
        "stress_weighted_win_rate": round(weighted_win, 4),
        "stress_mean_profit_factor": round(mean_pf, 4),
        "stress_worst_fold_r": round(worst, 4),
        "stress_max_drawdown_r": round(max_dd, 4),
        "recent_paper_r": recent["pnl_r"],
        "battle_gate": battle_gate,
        "score": round(score, 4),
        "folds": fold_rows,
    }


def flatten(row: dict, rank: int) -> dict:
    flat = {key: value for key, value in row.items() if key != "folds"}
    flat["rank"] = rank
    return flat


def write_outputs(results: list[dict]) -> None:
    results = sorted(
        results,
        key=lambda row: (row["battle_gate"], row["score"], row["stress_total_r"]),
        reverse=True,
    )
    flat_rows = [flatten(row, rank) for rank, row in enumerate(results, start=1)]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    OUTPUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    OUTPUT_BEST.write_text(json.dumps(results[0], indent=2), encoding="utf-8")

    best = results[0]
    lines = [
        "# SILVER Regime Readiness Optimization",
        "",
        "SILVER# H1, 5 rolling folds, 3x spread, entry regime filters.",
        "",
        "| Gate | R | Positive | Passed | Trades | Win | PF | Worst R | Recent R | Params |",
        "|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        (
            "| {battle_gate} | {stress_total_r:.2f} | {stress_positive_folds}/5 | "
            "{stress_passed_folds}/5 | {stress_total_trades} | "
            "{stress_weighted_win_rate:.2%} | {stress_mean_profit_factor:.2f} | "
            "{stress_worst_fold_r:.2f} | {recent_paper_r:.2f} | "
            "conf={threshold}, edge={edge_threshold}, tp/sl={tp_atr}/{sl_atr}, "
            "hold={max_hold}, trend_min={trend_min}, rsi={rsi_min}-{rsi_max}, "
            "vola_max={vola_max}, spread_atr_max={spread_atr_max}, macd_min={macd_min} |"
        ).format(**best),
        "",
        "## Fold Detail",
        "",
        "| Fold | R | Trades | Win | PF | DD | Pass |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in best["folds"]:
        lines.append(
            "| {fold} | {pnl_r:.2f} | {trades} | {win_rate:.2%} | "
            "{profit_factor:.2f} | {max_drawdown_r:.2f} | {fold_pass} |".format(**row)
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    prepared = prepare_folds()
    stage_one = []
    print("Stage 1: filter sweep...")
    for base in BASE_PARAMS:
        for filters in filter_grid():
            stage_one.append(evaluate({**base, **filters}, prepared))
    stage_one = sorted(stage_one, key=lambda row: (row["score"], row["stress_total_r"]), reverse=True)

    print("Stage 2: parameter sweep around top filters...")
    final_results = list(stage_one[:120])
    top_filters = [
        {key: row[key] for key in ["trend_min", "rsi_min", "rsi_max", "vola_max", "spread_atr_max", "macd_min"]}
        for row in stage_one[:16]
    ]
    for filters in top_filters:
        for params in param_grid():
            final_results.append(evaluate({**params, **filters}, prepared))

    write_outputs(final_results)
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_BEST}")
    print(f"Battle-gate candidates: {sum(row['battle_gate'] for row in final_results)}/{len(final_results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
