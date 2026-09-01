from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from barrier_final_train import prepare_barrier_data
from drl_trading_v2 import SPREAD_POINTS
from gold_expected_r_walk_forward import (
    EXTRA_COST_POINTS,
    HORIZON,
    MIN_SL_PRICE,
    SL_ATR,
)
from gold_generation11_execution_aligned import add_targets
from gold_recent_walk_forward import DEFAULT_TERMINAL, build_feature_frame
from gold_regime_experts_walk_forward import RECENT_START, SELECTION_FOLDS


PROJECT_ROOT = Path(__file__).resolve().parent
GEN15_REPORT = PROJECT_ROOT / "gold_generation15_signal_mining.json"
GEN17_REPORT = PROJECT_ROOT / "gold_generation17_cross_regime.json"
REPORT_JSON = PROJECT_ROOT / "gold_generation18_payoff_audit.json"
REPORT_MD = PROJECT_ROOT / "gold_generation18_payoff_audit.md"
RISK_PER_TRADE = 0.014
RECENT_WARMUP_START = datetime(2026, 4, 1, tzinfo=timezone.utc)
DEVELOPMENT_END = datetime(2026, 8, 31, tzinfo=timezone.utc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generation 18 payoff audit")
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def profit_factor(rewards: np.ndarray) -> float | None:
    gains = float(rewards[rewards > 0.0].sum())
    losses = float(-rewards[rewards < 0.0].sum())
    return None if losses <= 0.0 else gains / losses


def safe_mean(values: np.ndarray) -> float | None:
    return None if len(values) == 0 else float(values.mean())


def safe_median(values: np.ndarray) -> float | None:
    return None if len(values) == 0 else float(np.median(values))


def enrich_ledger(ledger: list[dict], frame: pd.DataFrame) -> list[dict]:
    output = []
    total_cost_price = (SPREAD_POINTS + EXTRA_COST_POINTS) * 0.01
    spread_price = SPREAD_POINTS * 0.01
    for record in ledger:
        index = int(record["index"])
        direction = int(record["direction"])
        if index < 0 or index >= len(frame):
            raise IndexError(f"Ledger index {index} outside frame with {len(frame)} rows")
        outcome_column = "LONG_OUTCOME" if direction == 1 else "SHORT_OUTCOME"
        reward_column = "LONG_REWARD" if direction == 1 else "SHORT_REWARD"
        exit_column = "LONG_EXIT_OFFSET" if direction == 1 else "SHORT_EXIT_OFFSET"
        outcome = int(frame[outcome_column].iat[index])
        reward = float(frame[reward_column].iat[index])
        exit_offset = int(frame[exit_column].iat[index])
        if outcome != int(record["outcome"]):
            raise RuntimeError(f"Outcome mismatch for {record['trade_id']}")
        if not math.isclose(
            reward, float(record["reward"]), rel_tol=1e-6, abs_tol=1e-6
        ):
            raise RuntimeError(f"Reward mismatch for {record['trade_id']}")
        if index + exit_offset != int(record["exit_index"]):
            raise RuntimeError(f"Exit offset mismatch for {record['trade_id']}")
        atr = float(frame["ATR"].iat[index])
        stop_loss = max(atr * SL_ATR, MIN_SL_PRICE)
        denominator = stop_loss + spread_price
        cost_r = total_cost_price / denominator
        output.append(
            {
                "outcome": outcome,
                "reward": reward,
                "gross_reward_before_cost": reward + cost_r,
                "spread_r": spread_price / denominator,
                "extra_cost_r": (EXTRA_COST_POINTS * 0.01) / denominator,
                "total_cost_r": cost_r,
            }
        )
    return output


def payoff_metrics(rows: list[dict], evaluated_days: int) -> dict:
    outcomes = np.asarray([row["outcome"] for row in rows], dtype=np.int8)
    rewards = np.asarray([row["reward"] for row in rows], dtype=np.float64)
    gross_rewards = np.asarray(
        [row["gross_reward_before_cost"] for row in rows], dtype=np.float64
    )
    costs = np.asarray([row["total_cost_r"] for row in rows], dtype=np.float64)
    spread = np.asarray([row["spread_r"] for row in rows], dtype=np.float64)
    extra = np.asarray([row["extra_cost_r"] for row in rows], dtype=np.float64)
    positive = rewards[rewards > 0.0]
    nonpositive = rewards[rewards <= 0.0]
    average_winner = safe_mean(positive)
    average_loser = safe_mean(nonpositive)
    payoff_ratio = (
        None
        if average_winner is None or average_loser is None or average_loser == 0.0
        else average_winner / abs(average_loser)
    )
    break_even = None if payoff_ratio is None else 1.0 / (1.0 + payoff_ratio)
    trades = len(rows)
    realized_win_rate = int((rewards > 0.0).sum()) / max(trades, 1)
    timeout_mask = outcomes == 0
    non_timeout_rewards = rewards[~timeout_mask]
    balance = 1000.0
    peak = balance
    maximum_drawdown = 0.0
    pnl = 0.0
    for reward in rewards:
        change = balance * RISK_PER_TRADE * reward
        balance += change
        pnl += change
        peak = max(peak, balance)
        maximum_drawdown = min(maximum_drawdown, balance / peak - 1.0)
    return {
        "trades": trades,
        "evaluated_days": evaluated_days,
        "trades_per_day": trades / max(evaluated_days, 1),
        "tp_first_rate": int((outcomes == 1).sum()) / max(trades, 1),
        "realized_positive_trade_win_rate": realized_win_rate,
        "tp_exits": int((outcomes == 1).sum()),
        "sl_exits": int((outcomes == 2).sum()),
        "timeout_exits": int(timeout_mask.sum()),
        "other_exit_types": int((~np.isin(outcomes, (0, 1, 2))).sum()),
        "realized_positive_trades": int((rewards > 0.0).sum()),
        "realized_nonpositive_trades": int((rewards <= 0.0).sum()),
        "tp_first_but_nonpositive": int(((outcomes == 1) & (rewards <= 0.0)).sum()),
        "non_tp_first_but_positive": int(((outcomes != 1) & (rewards > 0.0)).sum()),
        "average_winning_r": average_winner,
        "median_winning_r": safe_median(positive),
        "average_losing_r": average_loser,
        "median_losing_r": safe_median(nonpositive),
        "gross_profit_r": float(positive.sum()),
        "gross_loss_r": float(-nonpositive.sum()),
        "payoff_ratio": payoff_ratio,
        "realized_break_even_win_rate": break_even,
        "actual_realized_win_rate": realized_win_rate,
        "break_even_adjusted_win_rate_edge": (
            None if break_even is None else realized_win_rate - break_even
        ),
        "profit_factor": profit_factor(rewards),
        "mean_r": safe_mean(rewards) or 0.0,
        "sum_r": float(rewards.sum()),
        "pnl": pnl,
        "max_drawdown_pct": maximum_drawdown,
        "mean_gross_r_before_cost": safe_mean(gross_rewards) or 0.0,
        "profit_factor_before_cost": profit_factor(gross_rewards),
        "average_total_cost_r_per_trade": safe_mean(costs) or 0.0,
        "average_spread_contribution_r_per_trade": safe_mean(spread) or 0.0,
        "average_extra_cost_contribution_r_per_trade": safe_mean(extra) or 0.0,
        "total_cost_contribution_r": float(costs.sum()),
        "timeout_contribution": {
            "trades": int(timeout_mask.sum()),
            "positive": int((timeout_mask & (rewards > 0.0)).sum()),
            "nonpositive": int((timeout_mask & (rewards <= 0.0)).sum()),
            "sum_r": float(rewards[timeout_mask].sum()),
            "mean_r": safe_mean(rewards[timeout_mask]),
            "pf_without_timeouts": profit_factor(non_timeout_rewards),
            "mean_r_without_timeouts": safe_mean(non_timeout_rewards),
            "pf_delta_from_timeouts": (
                None
                if profit_factor(rewards) is None
                or profit_factor(non_timeout_rewards) is None
                else profit_factor(rewards) - profit_factor(non_timeout_rewards)
            ),
            "mean_r_delta_from_timeouts": (
                None
                if safe_mean(non_timeout_rewards) is None
                else (safe_mean(rewards) or 0.0) - safe_mean(non_timeout_rewards)
            ),
        },
    }


def historical_frames(history: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames = {
        name: history[
            (history["TIME_DT"] >= start) & (history["TIME_DT"] < end)
        ].reset_index(drop=True)
        for name, start, end in SELECTION_FOLDS
    }
    frames["2025_2026_05"] = history[
        history["TIME_DT"] >= datetime(2025, 1, 1)
    ].reset_index(drop=True)
    return frames


def recent_frame(report17: dict) -> pd.DataFrame:
    if not mt5.initialize(path=str(DEFAULT_TERMINAL), timeout=10_000):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        frame, _ = build_feature_frame(RECENT_WARMUP_START, DEVELOPMENT_END)
    finally:
        mt5.shutdown()
    frame = add_targets(frame)
    end = pd.Timestamp(report17["data"]["recent_development_end"]) + pd.Timedelta(
        minutes=1
    )
    return frame[
        (frame["TIME_DT"] >= RECENT_START.replace(tzinfo=None))
        & (frame["TIME_DT"] < end)
    ].reset_index(drop=True)


def report_markdown(report: dict) -> str:
    lines = [
        "# Generation 18 - Phase 1 payoff audit",
        "",
        "TP-first labels and positive net-return wins are audited separately.",
        "",
        "| Strategy | Period | Trades | TP-first | Realized win | Avg win R | Avg loss R | Payoff | Break-even | Edge | PF | Mean-R | Avg cost R | Timeouts |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy, periods in report["strategies"].items():
        for period, value in periods.items():
            lines.append(
                f"| {strategy} | {period} | {value['trades']} | "
                f"{value['tp_first_rate']:.2%} | "
                f"{value['realized_positive_trade_win_rate']:.2%} | "
                f"{value['average_winning_r'] or 0.0:.4f} | "
                f"{value['average_losing_r'] or 0.0:.4f} | "
                f"{value['payoff_ratio'] or 0.0:.4f} | "
                f"{value['realized_break_even_win_rate'] or 0.0:.2%} | "
                f"{value['break_even_adjusted_win_rate_edge'] or 0.0:.2%} | "
                f"{value['profit_factor'] or 0.0:.2f} | {value['mean_r']:.4f} | "
                f"{value['average_total_cost_r_per_trade']:.4f} | "
                f"{value['timeout_exits']} |"
            )
    return "\n".join(lines) + "\n"


def self_check() -> None:
    rows = [
        {
            "outcome": 1,
            "reward": 0.6,
            "gross_reward_before_cost": 0.7,
            "spread_r": 0.08,
            "extra_cost_r": 0.02,
            "total_cost_r": 0.1,
        },
        {
            "outcome": 2,
            "reward": -1.0,
            "gross_reward_before_cost": -0.9,
            "spread_r": 0.08,
            "extra_cost_r": 0.02,
            "total_cost_r": 0.1,
        },
    ]
    value = payoff_metrics(rows, 2)
    assert math.isclose(value["payoff_ratio"], 0.6)
    assert math.isclose(value["realized_break_even_win_rate"], 0.625)
    assert value["tp_first_rate"] == value["realized_positive_trade_win_rate"]
    print("generation18_payoff_audit_self_check_ok")


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0
    report15 = json.loads(GEN15_REPORT.read_text(encoding="utf-8"))
    report17 = json.loads(GEN17_REPORT.read_text(encoding="utf-8"))
    history, _ = prepare_barrier_data()
    history = add_targets(history)
    frames = historical_frames(history)
    frames["2026_recent"] = recent_frame(report17)

    ledger_sets = {
        "parent": {
            "2018_2020": report15["baseline_trade_ledgers"]["2018_2020"],
            "2021_2022": report15["baseline_trade_ledgers"]["2021_2022"],
            "2023_2024": report15["baseline_trade_ledgers"]["2023_2024"],
            "2025_2026_05": report15["baseline_trade_ledgers"][
                "2025_2026_05_holdout"
            ],
            "2026_recent": report15["baseline_trade_ledgers"]["2026_recent"],
        },
        "gen17_short_trend_diagnostic": {
            "2018_2020": report17["selected"]["results"]["2018_2020"][
                "trade_ledger"
            ],
            "2021_2022": report17["selected"]["results"]["2021_2022"][
                "trade_ledger"
            ],
            "2023_2024": report17["selected"]["results"]["2023_2024"][
                "trade_ledger"
            ],
            "2025_2026_05": report17["selected"]["results"][
                "2025_2026_05_development"
            ]["trade_ledger"],
            "2026_recent": report17["selected"]["results"][
                "2026_recent_development"
            ]["trade_ledger"],
        },
    }
    strategies = {}
    for strategy, periods in ledger_sets.items():
        enriched = {
            period: enrich_ledger(ledger, frames[period])
            for period, ledger in periods.items()
        }
        selection_rows = [
            row
            for period in ("2018_2020", "2021_2022", "2023_2024")
            for row in enriched[period]
        ]
        selection_days = sum(
            int(
                (
                    report17["selected"]["results"][period]["metrics"]
                    if strategy == "gen17_short_trend_diagnostic"
                    else report15["parent"]["selection_folds"][period]
                )["evaluated_days"]
            )
            for period in ("2018_2020", "2021_2022", "2023_2024")
        )
        strategies[strategy] = {
            period: payoff_metrics(
                rows,
                int(frames[period]["TIME_DT"].dt.date.nunique()),
            )
            for period, rows in enriched.items()
        }
        strategies[strategy]["selection_pooled"] = payoff_metrics(
            selection_rows, selection_days
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generation": "18_payoff_alignment_phase1",
        "status": "research_only",
        "execution_profile": {
            "horizon": HORIZON,
            "tp_atr": 1.3,
            "sl_atr": SL_ATR,
            "minimum_tp_price": 1.5,
            "minimum_sl_price": MIN_SL_PRICE,
            "spread_points": SPREAD_POINTS,
            "extra_cost_points": EXTRA_COST_POINTS,
            "position_limit": 1,
            "same_bar_tp_sl": "stop_first",
        },
        "metric_semantics": {
            "tp_first_rate": "outcome == 1 divided by executable trades",
            "realized_positive_trade_win_rate": "net reward > 0 divided by executable trades",
            "profit_factor": "sum positive net R divided by absolute sum negative net R",
            "break_even": "1 / (1 + average_winner_R / abs(average_loser_R))",
        },
        "strategies": strategies,
        "promotion_pass": False,
    }
    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REPORT_MD.write_text(report_markdown(report), encoding="utf-8")
    print(report_markdown(report), flush=True)
    print(f"Saved {REPORT_JSON.name}, {REPORT_MD.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
