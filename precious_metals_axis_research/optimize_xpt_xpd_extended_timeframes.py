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
from precious_metals_axis_research.readiness_silver_xaueur import scale_spread  # noqa: E402
from precious_metals_axis_research.walk_forward_long_tf_cost import (  # noqa: E402
    FOLDS,
    compact_period,
    slice_by_ratio,
)


OUTPUT_CSV = RESEARCH_DIR / "xpt_xpd_extended_timeframe_results.csv"
OUTPUT_JSON = RESEARCH_DIR / "xpt_xpd_extended_timeframe_results.json"
OUTPUT_MD = RESEARCH_DIR / "xpt_xpd_extended_timeframe_report.md"
OUTPUT_BEST = RESEARCH_DIR / "xpt_xpd_extended_timeframe_best.json"

SYMBOLS = ["XPTUSD#", "XPDUSD#"]
TIMEFRAMES_BY_SYMBOL = {
    "XPTUSD#": ["H2", "H4", "H12", "Daily"],
    "XPDUSD#": ["H2", "H4", "H12"],
}
PROFILE_NAMES = ["current_symbol", "smooth_more_trees"]
COST_MULTIPLIERS = [1.0, 2.0, 3.0, 4.0, 5.0]
TREND_SUFFIX = "_TREND"

MIN_TOTAL_TRADES = {
    "H1": 32,
    "H2": 28,
    "H4": 20,
    "H12": 12,
    "Daily": 8,
}
MIN_FOLD_TRADES = {
    "H1": 6,
    "H2": 5,
    "H4": 4,
    "H12": 2,
    "Daily": 1,
}

FILTERS = [
    {
        "filter_name": "none",
        "trend_mode": "any",
        "rsi_min": 0.0,
        "rsi_max": 100.0,
        "vola_max": 99.0,
        "spread_atr_max": 99.0,
        "macd_mode": "any",
    },
    {
        "filter_name": "low_vola",
        "trend_mode": "any",
        "rsi_min": 0.0,
        "rsi_max": 100.0,
        "vola_max": 1.2,
        "spread_atr_max": 99.0,
        "macd_mode": "any",
    },
    {
        "filter_name": "aligned_trend",
        "trend_mode": "aligned",
        "rsi_min": 0.0,
        "rsi_max": 100.0,
        "vola_max": 1.6,
        "spread_atr_max": 0.80,
        "macd_mode": "any",
    },
    {
        "filter_name": "counter_trend",
        "trend_mode": "counter",
        "rsi_min": 15.0,
        "rsi_max": 85.0,
        "vola_max": 1.4,
        "spread_atr_max": 0.80,
        "macd_mode": "any",
    },
    {
        "filter_name": "quiet_aligned",
        "trend_mode": "aligned",
        "rsi_min": 25.0,
        "rsi_max": 80.0,
        "vola_max": 1.05,
        "spread_atr_max": 0.65,
        "macd_mode": "any",
    },
]


def timeframe_param_grid(timeframe: str):
    if timeframe in {"H1", "H2"}:
        thresholds = [0.58, 0.64]
        tp_sl_pairs = [(2.4, 4.0), (3.6, 5.8)]
        holds = [120, 240]
    elif timeframe == "H4":
        thresholds = [0.58, 0.64]
        tp_sl_pairs = [(2.0, 3.6), (3.2, 5.4)]
        holds = [48, 96]
    elif timeframe == "H12":
        thresholds = [0.60, 0.66]
        tp_sl_pairs = [(1.6, 3.2), (2.8, 5.4)]
        holds = [24, 48]
    else:
        thresholds = [0.60, 0.66]
        tp_sl_pairs = [(1.4, 2.8), (2.4, 4.8)]
        holds = [10, 20]

    for threshold, edge, pair, hold, direction in product(
        thresholds,
        [0.0],
        tp_sl_pairs,
        holds,
        ["long", "short", "both"],
    ):
        tp, sl = pair
        yield {
            "threshold": threshold,
            "edge_threshold": edge,
            "tp_atr": tp,
            "sl_atr": sl,
            "max_hold": hold,
            "direction_mode": direction,
        }


def trend_score(frame) -> np.ndarray:
    columns = [column for column in frame.columns if column.endswith(TREND_SUFFIX)]
    if not columns:
        return np.zeros(len(frame), dtype=np.float64)
    return frame[columns].mean(axis=1).to_numpy(dtype=np.float64)


def passes_signal_filter(
    signal: int,
    trend_value: float,
    rsi: float,
    vola: float,
    spread_atr: float,
    macd: float,
    params: dict,
) -> bool:
    if not (params["rsi_min"] <= rsi <= params["rsi_max"]):
        return False
    if vola > params["vola_max"] or spread_atr > params["spread_atr_max"]:
        return False

    if params["trend_mode"] == "positive" and trend_value < 0:
        return False
    if params["trend_mode"] == "negative" and trend_value > 0:
        return False
    if params["trend_mode"] == "aligned":
        if signal == 1 and trend_value < 0:
            return False
        if signal == 2 and trend_value > 0:
            return False
    if params["trend_mode"] == "counter":
        if signal == 1 and trend_value > 0:
            return False
        if signal == 2 and trend_value < 0:
            return False

    if params["macd_mode"] == "positive" and macd < 0:
        return False
    if params["macd_mode"] == "negative" and macd > 0:
        return False
    if params["macd_mode"] == "aligned":
        if signal == 1 and macd < 0:
            return False
        if signal == 2 and macd > 0:
            return False

    return True


def prepare(symbol: str, timeframe: str, profile_name: str, profile: dict) -> list[dict]:
    frame, features = load_case(symbol, timeframe)
    point = get_symbol_point(symbol)
    prepared = []
    for fold in FOLDS:
        train_df = slice_by_ratio(frame, *fold["train"])
        test_df = slice_by_ratio(frame, *fold["test"])
        print(
            f"{symbol} {timeframe} {profile_name} {fold['name']}: "
            f"train={len(train_df):,} test={len(test_df):,}",
            flush=True,
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


def simulate(frame, probs: np.ndarray, params: dict, point: float) -> dict:
    close = frame["CLOSE"].to_numpy(dtype=np.float64)
    atr = frame["ATR"].to_numpy(dtype=np.float64)
    spread_points = (
        frame["SPREAD"].fillna(0).to_numpy(dtype=np.float64)
        if "SPREAD" in frame.columns
        else np.zeros(len(frame), dtype=np.float64)
    )
    trend = trend_score(frame)
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
        confidence = buy_prob if signal == 1 else sell_prob
        edge = abs(buy_prob - sell_prob)
        has_signal = (
            confidence >= params["threshold"]
            and edge >= params["edge_threshold"]
            and passes_signal_filter(
                signal,
                trend[i],
                rsi[i],
                vola[i],
                spread_atr[i],
                macd[i],
                params,
            )
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


def fold_pass(stats: dict, timeframe: str) -> bool:
    return (
        stats["pnl_r"] > 0
        and stats["profit_factor"] >= 1.12
        and stats["win_rate"] >= 0.52
        and stats["trades"] >= MIN_FOLD_TRADES[timeframe]
    )


def aggregate(symbol: str, timeframe: str, fold_rows: list[dict]) -> dict:
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
    min_total_trades = MIN_TOTAL_TRADES[timeframe]
    gate = (
        positive == 4
        and passed >= 3
        and total_r >= 8.0
        and total_trades >= min_total_trades
        and recent["pnl_r"] > 0
        and recent["profit_factor"] >= 1.08
        and max_dd >= -10.0
    )
    score = (
        total_r * 180.0
        + positive * 850.0
        + passed * 520.0
        + weighted_win * 700.0
        + min(mean_pf, 4.0) * 220.0
        - abs(max_dd) * 60.0
        + min(worst, 0.0) * 300.0
        + min(total_trades / max(min_total_trades, 1), 2.0) * 120.0
    )
    return {
        "symbol": symbol,
        "base_timeframe": timeframe,
        "total_r": round(total_r, 4),
        "trades": total_trades,
        "positive_folds": positive,
        "passed_folds": passed,
        "weighted_win_rate": round(weighted_win, 4),
        "mean_profit_factor": round(mean_pf, 4),
        "worst_fold_r": round(worst, 4),
        "max_drawdown_r": round(max_dd, 4),
        "recent_paper_r": recent["pnl_r"],
        "min_total_trades": min_total_trades,
        "gate": gate,
        "score": round(score, 4),
        "folds": fold_rows,
    }


def evaluate(
    symbol: str,
    timeframe: str,
    params: dict,
    prepared: list[dict],
    cost_multiplier: float,
) -> dict:
    fold_rows = []
    for fold in prepared:
        stats = simulate(
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
                "fold_pass": fold_pass(stats, timeframe),
            }
        )
    return {
        **params,
        "cost_multiplier": cost_multiplier,
        **aggregate(symbol, timeframe, fold_rows),
    }


def choose_best(rows: list[dict]) -> dict:
    return sorted(
        rows,
        key=lambda row: (
            row["gate"],
            row["positive_folds"],
            row["passed_folds"],
            row["score"],
            row["total_r"],
        ),
        reverse=True,
    )[0]


def flatten_result(row: dict, profile_name: str, group: str) -> dict:
    return {
        key: value
        for key, value in {"profile": profile_name, "group": group, **row}.items()
        if key != "folds"
    }


def write_outputs(rows: list[dict], selected: dict, skipped: list[dict]) -> None:
    flat_rows = [
        flatten_result(row["result"], row["profile"], row["group"])
        for row in rows
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    OUTPUT_JSON.write_text(
        json.dumps({"rows": rows, "selected": selected, "skipped": skipped}, indent=2),
        encoding="utf-8",
    )
    OUTPUT_BEST.write_text(json.dumps(selected, indent=2), encoding="utf-8")

    lines = [
        "# XPT / XPD Extended Timeframe Optimization",
        "",
        "Research-only walk-forward search across H2/H4/H12/Daily with 1x-5x cost stress.",
        "",
        "## Selected",
        "",
        "| Symbol | TF | Profile | Cost Gates | 3x Gate | 3x R | Trades | Win | PF | Worst R | DD | Params |",
        "|---|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for symbol in SYMBOLS:
        item = selected[symbol]
        row = item["best_3x"]
        lines.append(
            "| {symbol} | {base_timeframe} | {profile} | {cost_gate_count}/5 | "
            "{gate} | {total_r:.2f} | {trades} | {weighted_win_rate:.2%} | "
            "{mean_profit_factor:.2f} | {worst_fold_r:.2f} | {max_drawdown_r:.2f} | "
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
            "| Symbol | TF | Profile | Cost | Gate | R | Positive | Passed | Trades | Win | Worst R | DD |",
            "|---|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for symbol in SYMBOLS:
        item = selected[symbol]
        for row in item["cost_stress"]:
            lines.append(
                "| {symbol} | {base_timeframe} | {profile} | {cost_multiplier:.1f}x | "
                "{gate} | {total_r:.2f} | {positive_folds}/4 | {passed_folds}/4 | "
                "{trades} | {weighted_win_rate:.2%} | {worst_fold_r:.2f} | "
                "{max_drawdown_r:.2f} |".format(
                    profile=item["profile"],
                    **row,
                )
            )

    if skipped:
        lines.extend(["", "## Skipped", ""])
        for item in skipped:
            lines.append(
                f"- {item['symbol']} {item['base_timeframe']} {item['profile']}: "
                f"{item['reason']}"
            )

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = []
    skipped = []
    candidate_groups = {symbol: [] for symbol in SYMBOLS}
    profiles = {name: MODEL_PROFILES[name] for name in PROFILE_NAMES}

    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES_BY_SYMBOL[symbol]:
            for profile_name, profile in profiles.items():
                print(f"=== {symbol} {timeframe} profile={profile_name} ===", flush=True)
                try:
                    prepared = prepare(symbol, timeframe, profile_name, profile)
                except Exception as exc:
                    print(f"Skipped {symbol} {timeframe} {profile_name}: {exc}")
                    skipped.append(
                        {
                            "symbol": symbol,
                            "base_timeframe": timeframe,
                            "profile": profile_name,
                            "reason": str(exc),
                        }
                    )
                    continue
                grid_results = []
                for base_params in timeframe_param_grid(timeframe):
                    for filter_params in FILTERS:
                        params = {**base_params, **filter_params}
                        grid_results.append(
                            evaluate(symbol, timeframe, params, prepared, 3.0)
                        )
                best_3x = choose_best(grid_results)
                cost_stress = [
                    evaluate(symbol, timeframe, best_3x, prepared, cost)
                    for cost in COST_MULTIPLIERS
                ]
                candidate_groups[symbol].append(
                    {
                        "profile": profile_name,
                        "base_timeframe": timeframe,
                        "best_3x": best_3x,
                        "cost_stress": cost_stress,
                        "cost_gate_count": sum(row["gate"] for row in cost_stress),
                        "cost_total_r": round(
                            sum(row["total_r"] for row in cost_stress), 4
                        ),
                    }
                )
                rows.extend(
                    {
                        "profile": profile_name,
                        "group": f"{symbol}_{timeframe}_grid_3x",
                        "result": row,
                    }
                    for row in grid_results
                )
                rows.extend(
                    {
                        "profile": profile_name,
                        "group": f"{symbol}_{timeframe}_selected_cost",
                        "result": row,
                    }
                    for row in cost_stress
                )

    selected = {}
    for symbol, items in candidate_groups.items():
        selected[symbol] = sorted(
            items,
            key=lambda item: (
                item["cost_gate_count"],
                item["best_3x"]["gate"],
                item["best_3x"]["positive_folds"],
                item["best_3x"]["passed_folds"],
                item["best_3x"]["total_r"],
                item["cost_total_r"],
            ),
            reverse=True,
        )[0]

    write_outputs(rows, selected, skipped)
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_BEST}")
    for symbol, item in selected.items():
        row = item["best_3x"]
        print(
            f"{symbol}: tf={item['base_timeframe']} profile={item['profile']} "
            f"gates={item['cost_gate_count']}/5 r3x={row['total_r']:.2f} "
            f"gate3x={row['gate']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
