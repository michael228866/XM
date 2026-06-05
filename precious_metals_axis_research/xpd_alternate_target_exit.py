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
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402

from precious_metals_axis_research.axis_timeframe_smoke import load_case  # noqa: E402
from precious_metals_axis_research.cost_aware_xaueur_m5 import get_symbol_point  # noqa: E402
from precious_metals_axis_research.optimize_xpt_xpd_extended_timeframes import (  # noqa: E402
    passes_signal_filter,
    trend_score,
)
from precious_metals_axis_research.readiness_silver_xaueur import scale_spread  # noqa: E402
from precious_metals_axis_research.walk_forward_long_tf_cost import (  # noqa: E402
    FOLDS,
    compact_period,
    slice_by_ratio,
)


OUTPUT_CSV = RESEARCH_DIR / "xpd_alternate_target_exit_results.csv"
OUTPUT_JSON = RESEARCH_DIR / "xpd_alternate_target_exit_results.json"
OUTPUT_MD = RESEARCH_DIR / "xpd_alternate_target_exit_report.md"
OUTPUT_BEST = RESEARCH_DIR / "xpd_alternate_target_exit_best.json"

SYMBOL = "XPDUSD#"
TIMEFRAMES = ["H4", "H12"]
COST_MULTIPLIERS = [1.0, 2.0, 3.0, 4.0, 5.0]

MIN_TOTAL_TRADES = {
    "H4": 24,
    "H12": 12,
}
MIN_FOLD_TRADES = {
    "H4": 4,
    "H12": 2,
}

TARGET_CONFIGS = {
    "H4": [
        {"target_mode": "future_close", "horizon": 8, "label_atr": 0.8},
        {"target_mode": "future_close", "horizon": 16, "label_atr": 1.2},
        {"target_mode": "dominant_swing", "horizon": 16, "label_atr": 1.4, "dominance_atr": 0.4},
        {"target_mode": "dominant_swing", "horizon": 32, "label_atr": 1.8, "dominance_atr": 0.6},
    ],
    "H12": [
        {"target_mode": "future_close", "horizon": 4, "label_atr": 0.8},
        {"target_mode": "future_close", "horizon": 8, "label_atr": 1.2},
        {"target_mode": "dominant_swing", "horizon": 8, "label_atr": 1.2, "dominance_atr": 0.4},
        {"target_mode": "dominant_swing", "horizon": 12, "label_atr": 1.6, "dominance_atr": 0.6},
    ],
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
        "filter_name": "aligned",
        "trend_mode": "aligned",
        "rsi_min": 20.0,
        "rsi_max": 90.0,
        "vola_max": 1.8,
        "spread_atr_max": 0.9,
        "macd_mode": "any",
    },
    {
        "filter_name": "quiet_aligned",
        "trend_mode": "aligned",
        "rsi_min": 25.0,
        "rsi_max": 85.0,
        "vola_max": 1.2,
        "spread_atr_max": 0.8,
        "macd_mode": "aligned",
    },
]


def forward_rolling(series: pd.Series, window: int, op: str) -> pd.Series:
    future = series.shift(-1)
    reversed_future = future.iloc[::-1]
    if op == "max":
        rolled = reversed_future.rolling(window, min_periods=1).max()
    elif op == "min":
        rolled = reversed_future.rolling(window, min_periods=1).min()
    else:
        raise ValueError(op)
    return rolled.iloc[::-1]


def build_alternate_target(frame: pd.DataFrame, config: dict) -> pd.Series:
    atr = frame["ATR"].replace(0, np.nan)
    target = pd.Series(0, index=frame.index, dtype="float64")
    horizon = int(config["horizon"])
    if config["target_mode"] == "future_close":
        future_close = frame["CLOSE"].shift(-horizon)
        score = (future_close - frame["CLOSE"]) / (atr + 1e-9)
        target[score >= config["label_atr"]] = 1
        target[score <= -config["label_atr"]] = 2
    elif config["target_mode"] == "dominant_swing":
        future_high = forward_rolling(frame["HIGH"], horizon, "max")
        future_low = forward_rolling(frame["LOW"], horizon, "min")
        upside = (future_high - frame["CLOSE"]) / (atr + 1e-9)
        downside = (frame["CLOSE"] - future_low) / (atr + 1e-9)
        target[
            (upside >= config["label_atr"])
            & ((upside - downside) >= config["dominance_atr"])
        ] = 1
        target[
            (downside >= config["label_atr"])
            & ((downside - upside) >= config["dominance_atr"])
        ] = 2
    else:
        raise ValueError(config["target_mode"])
    target.iloc[-horizon:] = np.nan
    return target


def train_model(train_df: pd.DataFrame, features: list[str]) -> xgb.XGBClassifier:
    y = train_df["ALT_TARGET"].to_numpy(dtype=np.int8)
    counts = np.bincount(y, minlength=3)
    if np.any(counts == 0):
        raise ValueError(f"target class missing: {counts.tolist()}")
    class_weight = len(y) / (3.0 * np.maximum(counts, 1))
    sample_weight = class_weight[y]
    sample_weight[y == 0] *= 0.7
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        device="cpu",
        n_estimators=160,
        learning_rate=0.045,
        max_depth=4,
        min_child_weight=90,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        reg_alpha=0.05,
        random_state=42,
        verbosity=0,
    )
    model.fit(train_df[features], train_df["ALT_TARGET"], sample_weight=sample_weight)
    return model


def load_target_case(timeframe: str, target_config: dict) -> tuple[pd.DataFrame, list[str]]:
    frame, features = load_case(SYMBOL, timeframe)
    frame = frame.copy()
    frame["ALT_TARGET"] = build_alternate_target(frame, target_config)
    frame = frame.dropna(subset=features + ["ALT_TARGET", "ATR", "CLOSE", "BASE_RSI"])
    frame["ALT_TARGET"] = frame["ALT_TARGET"].astype(np.int8)
    return frame.reset_index(drop=True), features


def param_grid(timeframe: str):
    if timeframe == "H4":
        holds = [12, 24, 36]
        tp_values = [1.8, 2.6, 3.4]
        sl_values = [3.2, 4.8]
    else:
        holds = [6, 10, 14]
        tp_values = [1.4, 2.2, 3.0]
        sl_values = [2.8, 4.2]

    for threshold, edge, exit_mode, tp, sl, hold, direction in product(
        [0.46, 0.50, 0.54, 0.58],
        [0.0, 0.03],
        ["tp_sl", "time_stop"],
        tp_values,
        sl_values,
        holds,
        ["long", "both"],
    ):
        if tp > sl:
            continue
        yield {
            "threshold": threshold,
            "edge_threshold": edge,
            "exit_mode": exit_mode,
            "tp_atr": tp,
            "sl_atr": sl,
            "max_hold": hold,
            "direction_mode": direction,
        }


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
        if params["exit_mode"] == "tp_sl":
            exit_now = (
                gross >= entry_tp
                or gross <= -entry_sl
                or hold >= params["max_hold"]
            )
        elif params["exit_mode"] == "time_stop":
            exit_now = gross <= -entry_sl or hold >= params["max_hold"]
        else:
            raise ValueError(params["exit_mode"])

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


def prepare(timeframe: str, target_config: dict) -> list[dict]:
    frame, features = load_target_case(timeframe, target_config)
    point = get_symbol_point(SYMBOL)
    prepared = []
    for fold in FOLDS:
        train_df = slice_by_ratio(frame, *fold["train"])
        test_df = slice_by_ratio(frame, *fold["test"])
        print(
            f"{SYMBOL} {timeframe} {target_config['target_mode']} "
            f"h={target_config['horizon']} {fold['name']}: "
            f"train={len(train_df):,} test={len(test_df):,}",
            flush=True,
        )
        model = train_model(train_df, features)
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


def fold_pass(stats: dict, timeframe: str) -> bool:
    return (
        stats["pnl_r"] > 0
        and stats["profit_factor"] >= 1.10
        and stats["win_rate"] >= 0.52
        and stats["trades"] >= MIN_FOLD_TRADES[timeframe]
    )


def aggregate(timeframe: str, fold_rows: list[dict]) -> dict:
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
        and total_trades >= MIN_TOTAL_TRADES[timeframe]
        and recent["pnl_r"] > 0
        and max_dd >= -12.0
    )
    score = (
        total_r * 170.0
        + positive * 900.0
        + passed * 550.0
        + weighted_win * 650.0
        + min(mean_pf, 4.0) * 190.0
        - abs(max_dd) * 65.0
        + min(worst, 0.0) * 320.0
    )
    return {
        "symbol": SYMBOL,
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
        "gate": gate,
        "score": round(score, 4),
        "folds": fold_rows,
    }


def evaluate(timeframe: str, params: dict, prepared: list[dict], cost_multiplier: float) -> dict:
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
    return {**params, "cost_multiplier": cost_multiplier, **aggregate(timeframe, fold_rows)}


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


def flat(row: dict, group: str, target_config: dict) -> dict:
    item = {**target_config, "group": group, **row}
    return {key: value for key, value in item.items() if key != "folds"}


def write_outputs(rows: list[dict], selected: dict, skipped: list[dict]) -> None:
    flat_rows = [flat(item["result"], item["group"], item["target_config"]) for item in rows]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        fieldnames = sorted({key for row in flat_rows for key in row})
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)
    OUTPUT_JSON.write_text(
        json.dumps({"rows": rows, "selected": selected, "skipped": skipped}, indent=2),
        encoding="utf-8",
    )
    OUTPUT_BEST.write_text(json.dumps(selected, indent=2), encoding="utf-8")

    best = selected["best_3x"]
    target = selected["target_config"]
    lines = [
        "# XPD Alternate Target / Exit Search",
        "",
        "Research-only test using direction and dominant-swing targets instead of the original clean barrier target.",
        "",
        "## Selected",
        "",
        "| TF | Target | Cost Gates | 3x Gate | 3x R | Trades | Win | PF | Worst R | DD | Params |",
        "|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---|",
        (
            "| {base_timeframe} | {target_mode} h={horizon} | {cost_gate_count}/5 | "
            "{gate} | {total_r:.2f} | {trades} | {weighted_win_rate:.2%} | "
            "{mean_profit_factor:.2f} | {worst_fold_r:.2f} | {max_drawdown_r:.2f} | "
            "{filter_name}: exit={exit_mode}, conf={threshold}, edge={edge_threshold}, "
            "tp/sl={tp_atr}/{sl_atr}, hold={max_hold}, dir={direction_mode} |"
        ).format(cost_gate_count=selected["cost_gate_count"], **target, **best),
        "",
        "## Cost Stress",
        "",
        "| Cost | Gate | R | Positive | Passed | Trades | Win | Worst R | DD |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected["cost_stress"]:
        lines.append(
            "| {cost_multiplier:.1f}x | {gate} | {total_r:.2f} | "
            "{positive_folds}/4 | {passed_folds}/4 | {trades} | "
            "{weighted_win_rate:.2%} | {worst_fold_r:.2f} | {max_drawdown_r:.2f} |".format(
                **row
            )
        )
    if skipped:
        lines.extend(["", "## Skipped", ""])
        for item in skipped:
            lines.append(
                f"- {item['base_timeframe']} {item['target_mode']} h={item['horizon']}: "
                f"{item['reason']}"
            )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = []
    skipped = []
    candidates = []
    for timeframe in TIMEFRAMES:
        for target_config in TARGET_CONFIGS[timeframe]:
            try:
                prepared = prepare(timeframe, target_config)
            except Exception as exc:
                print(
                    f"Skipped {timeframe} {target_config['target_mode']} "
                    f"h={target_config['horizon']}: {exc}"
                )
                skipped.append({**target_config, "base_timeframe": timeframe, "reason": str(exc)})
                continue

            grid_results = []
            for base_params in param_grid(timeframe):
                for filter_params in FILTERS:
                    params = {**base_params, **filter_params}
                    grid_results.append(evaluate(timeframe, params, prepared, 3.0))
            best_3x = choose_best(grid_results)
            cost_stress = [
                evaluate(timeframe, best_3x, prepared, cost)
                for cost in COST_MULTIPLIERS
            ]
            candidates.append(
                {
                    "base_timeframe": timeframe,
                    "target_config": target_config,
                    "best_3x": best_3x,
                    "cost_stress": cost_stress,
                    "cost_gate_count": sum(row["gate"] for row in cost_stress),
                    "cost_total_r": round(sum(row["total_r"] for row in cost_stress), 4),
                }
            )
            rows.extend(
                {"group": f"{timeframe}_grid_3x", "target_config": target_config, "result": row}
                for row in grid_results
            )
            rows.extend(
                {"group": f"{timeframe}_selected_cost", "target_config": target_config, "result": row}
                for row in cost_stress
            )

    if not candidates:
        raise ValueError("No XPD alternate-target candidates were produced.")

    selected = sorted(
        candidates,
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
    best = selected["best_3x"]
    target = selected["target_config"]
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_BEST}")
    print(
        f"Selected tf={selected['base_timeframe']} target={target['target_mode']} "
        f"h={target['horizon']} gates={selected['cost_gate_count']}/5 "
        f"r3x={best['total_r']:.2f} gate3x={best['gate']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
