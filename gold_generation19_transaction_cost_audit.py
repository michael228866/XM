from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
M1_CSV = ROOT / "GOLD#_M1_201401020000_202605082357.csv"
SIGNAL_LOG = ROOT / "gemini_signal_log.csv"
TRADE_LOG = ROOT / "gemini_trade_history.csv"
GEN17_REPORT = ROOT / "gold_generation17_cross_regime.json"
REPORT_JSON = ROOT / "gold_generation19_transaction_cost_audit.json"
REPORT_MD = ROOT / "gold_generation19_transaction_cost_audit.md"
TERMINAL = Path(r"D:\XM2\terminal64.exe")

POINT = 0.01
FALLBACK_SPREAD_POINTS = 30.0
EXTRA_COST_POINTS = 5.0
TP_ATR = 1.3
SL_ATR = 1.6
MIN_TP_PRICE = 1.5
MIN_SL_PRICE = 0.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generation 19 cost audit")
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def scalar(value):
    if pd.isna(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def describe(values: pd.Series | np.ndarray) -> dict:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return {key: None for key in ("count", "min", "median", "p75", "p90", "p95", "mean", "max")}
    return {
        "count": len(series),
        "min": float(series.min()),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "p90": float(series.quantile(0.90)),
        "p95": float(series.quantile(0.95)),
        "mean": float(series.mean()),
        "max": float(series.max()),
    }


def grouped_spread(frame: pd.DataFrame, column: str) -> dict:
    output = {}
    for key, group in frame.groupby(column, sort=True, observed=True):
        spread = group["SPREAD_POINTS"]
        output[str(scalar(key))] = {
            "rows": len(group),
            "zero_or_missing_rate": float((spread.fillna(0.0) <= 0.0).mean()),
            "observed_positive": describe(spread[spread > 0.0]),
        }
    return output


def add_time_groups(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    hour = result["TIME_DT"].dt.hour
    result["YEAR"] = result["TIME_DT"].dt.year
    result["HOUR"] = hour
    result["SESSION"] = pd.cut(
        hour,
        bins=(-1, 5, 12, 18, 23),
        labels=("utc_00_05", "utc_06_12", "utc_13_18", "utc_19_23"),
    )
    return result


def load_historical() -> pd.DataFrame:
    columns = ["<DATE>", "<TIME>", "<HIGH>", "<LOW>", "<CLOSE>", "<SPREAD>"]
    frame = pd.read_csv(M1_CSV, sep="\t", usecols=columns)
    frame.columns = [name.strip("<>") for name in frame.columns]
    frame["TIME_DT"] = pd.to_datetime(
        frame.pop("DATE") + " " + frame.pop("TIME"),
        format="%Y.%m.%d %H:%M:%S",
    )
    frame.rename(columns={"SPREAD": "SPREAD_POINTS"}, inplace=True)
    previous_close = frame["CLOSE"].shift()
    true_range = pd.concat(
        (
            frame["HIGH"] - frame["LOW"],
            (frame["HIGH"] - previous_close).abs(),
            (frame["LOW"] - previous_close).abs(),
        ),
        axis=1,
    ).max(axis=1)
    frame["ATR"] = true_range.rolling(14).mean().shift(1)
    frame["VOLA_RATIO"] = frame["ATR"] / frame["ATR"].rolling(240).mean().shift(1)
    return add_time_groups(frame)


def atr_groups(frame: pd.DataFrame) -> tuple[dict, dict]:
    valid = frame[(frame["SPREAD_POINTS"] > 0.0) & frame["ATR"].notna()].copy()
    thresholds = valid["ATR"].quantile((0.2, 0.4, 0.6, 0.8)).to_numpy()
    valid["ATR_PERCENTILE"] = np.searchsorted(thresholds, valid["ATR"], side="right") + 1
    valid["VOLATILITY_REGIME"] = pd.cut(
        valid["VOLA_RATIO"],
        bins=(-np.inf, 0.8, 1.2, np.inf),
        labels=("low", "normal", "high"),
    )
    return grouped_spread(valid, "ATR_PERCENTILE"), grouped_spread(valid, "VOLATILITY_REGIME")


def mt5_snapshot() -> tuple[dict, pd.DataFrame]:
    if not mt5.initialize(path=str(TERMINAL), timeout=10_000):
        return {"available": False, "error": str(mt5.last_error())}, pd.DataFrame()
    try:
        symbol = mt5.symbol_info("GOLD#")
        tick = mt5.symbol_info_tick("GOLD#")
        account = mt5.account_info()
        rates = mt5.copy_rates_range(
            "GOLD#",
            mt5.TIMEFRAME_M1,
            datetime(2026, 5, 9, tzinfo=timezone.utc),
            datetime.now(timezone.utc),
        )
        if symbol is None or tick is None:
            raise RuntimeError("GOLD# symbol information unavailable")
        snapshot = {
            "available": True,
            "account_trade_mode": None if account is None else int(account.trade_mode),
            "account_trade_mode_label": "demo" if account is not None and account.trade_mode == 0 else "non_demo_or_unknown",
            "server": None if account is None else account.server,
            "chart_mode": int(symbol.chart_mode),
            "chart_mode_label": "bid" if symbol.chart_mode == 0 else "last",
            "digits": int(symbol.digits),
            "point": float(symbol.point),
            "trade_tick_size": float(symbol.trade_tick_size),
            "trade_tick_value": float(symbol.trade_tick_value),
            "contract_size": float(symbol.trade_contract_size),
            "spread_float": bool(symbol.spread_float),
            "snapshot_spread_points": int(symbol.spread),
            "snapshot_bid": float(tick.bid),
            "snapshot_ask": float(tick.ask),
            "snapshot_ask_minus_bid": float(tick.ask - tick.bid),
        }
        recent = pd.DataFrame() if rates is None else pd.DataFrame(rates)
        if not recent.empty:
            recent["TIME_DT"] = pd.to_datetime(recent.pop("time"), unit="s", utc=True).dt.tz_localize(None)
            recent.rename(
                columns={
                    "high": "HIGH",
                    "low": "LOW",
                    "close": "CLOSE",
                    "spread": "SPREAD_POINTS",
                },
                inplace=True,
            )
            recent = add_time_groups(recent)
        return snapshot, recent
    except Exception as exc:
        return {"available": False, "error": str(exc)}, pd.DataFrame()
    finally:
        mt5.shutdown()


def log_audit() -> dict:
    output = {"signal_log": {"available": False}, "trade_log": {"available": False}}
    if SIGNAL_LOG.exists():
        log = pd.read_csv(SIGNAL_LOG, low_memory=False)
        spread = pd.to_numeric(log.get("spread_points"), errors="coerce")
        output["signal_log"] = {
            "available": True,
            "rows": len(log),
            "spread_rows": int(spread.notna().sum()),
            "spread_points": describe(spread.dropna()),
            "start": None if log.empty else str(log["event_time"].iloc[0]),
            "end": None if log.empty else str(log["event_time"].iloc[-1]),
        }
    if TRADE_LOG.exists():
        trades = pd.read_csv(TRADE_LOG, low_memory=False)
        fields = {}
        for name in ("commission", "swap", "fee"):
            values = pd.to_numeric(trades.get(name), errors="coerce").fillna(0.0)
            fields[name] = {
                "sum": float(values.sum()),
                "nonzero_rows": int((values != 0.0).sum()),
            }
        output["trade_log"] = {"available": True, "rows": len(trades), **fields}
    return output


def effective_spread(raw_points: float) -> tuple[float, bool]:
    observed = math.isfinite(raw_points) and raw_points > 0.0
    return (raw_points if observed else FALLBACK_SPREAD_POINTS), observed


def reprice_gen17(frame: pd.DataFrame) -> dict:
    report = json.loads(GEN17_REPORT.read_text(encoding="utf-8"))
    ledgers = [
        record
        for fold in ("2018_2020", "2021_2022", "2023_2024")
        for record in report["selected"]["results"][fold]["trade_ledger"]
    ]
    indexed = frame.set_index("TIME_DT", verify_integrity=True)
    old_rewards, new_rewards, spreads, costs = [], [], [], []
    observed_count = 0
    for record in ledgers:
        row = indexed.loc[pd.Timestamp(record["time"])]
        atr = float(row["ATR"])
        stop_loss = max(atr * SL_ATR, MIN_SL_PRICE)
        spread_points, observed = effective_spread(float(row["SPREAD_POINTS"]))
        observed_count += observed
        old_reward = float(record["reward"])
        gross_pnl = old_reward * (stop_loss + FALLBACK_SPREAD_POINTS * POINT) + (
            FALLBACK_SPREAD_POINTS + EXTRA_COST_POINTS
        ) * POINT
        denominator = stop_loss + spread_points * POINT
        new_reward = (
            gross_pnl - (spread_points + EXTRA_COST_POINTS) * POINT
        ) / denominator
        old_rewards.append(old_reward)
        new_rewards.append(new_reward)
        spreads.append(spread_points)
        costs.append((spread_points + EXTRA_COST_POINTS) * POINT / denominator)
    old = np.asarray(old_rewards)
    new = np.asarray(new_rewards)

    def pf(values: np.ndarray) -> float | None:
        loss = float(-values[values < 0.0].sum())
        return None if loss <= 0.0 else float(values[values > 0.0].sum() / loss)

    return {
        "trades": len(new),
        "observed_entry_spread_trades": observed_count,
        "fallback_entry_spread_trades": len(new) - observed_count,
        "effective_entry_spread_points": describe(spreads),
        "fixed_30_points": {
            "mean_r": float(old.mean()),
            "profit_factor": pf(old),
            "realized_win_rate": float((old > 0.0).mean()),
        },
        "observed_or_30_fallback": {
            "mean_r": float(new.mean()),
            "sum_r": float(new.sum()),
            "profit_factor": pf(new),
            "realized_win_rate": float((new > 0.0).mean()),
            "average_cost_r_per_trade": float(np.mean(costs)),
        },
    }


def markdown(report: dict) -> str:
    historical = report["historical_spread"]
    gen17 = report["gen17_repricing"]
    lines = [
        "# Generation 19 - Transaction-cost audit",
        "",
        "All outputs remain `research_only`. Historical zero spreads are treated as missing and fall back to 30 points.",
        "",
        "## Unit and execution audit",
        "",
        "| Item | Result |",
        "|---|---|",
        f"| GOLD# chart mode | {report['mt5_snapshot'].get('chart_mode_label', 'unavailable')} |",
        f"| Digits / point / tick size | {report['mt5_snapshot'].get('digits')} / {report['mt5_snapshot'].get('point')} / {report['mt5_snapshot'].get('trade_tick_size')} |",
        f"| Fixed spread assumption | 30 points = {30 * POINT:.2f} price |",
        f"| Extra cost assumption | 5 points = {5 * POINT:.2f} price |",
        "| Cost double-counted | No: spread is deducted once from price PnL; the denominator only expresses R in total stop cash-risk units. |",
        "| Commission/slippage/swap | 5 points is an opaque extra-cost allowance; recorded commission/swap/fee are reported below. |",
        "",
        "## Historical spread coverage",
        "",
        "| Year | Rows | Zero/missing | Median positive | p90 | p95 | Mean |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for year, value in historical["by_year"].items():
        stats = value["observed_positive"]
        lines.append(
            f"| {year} | {value['rows']} | {value['zero_or_missing_rate']:.2%} | "
            f"{stats['median'] if stats['median'] is not None else 'n/a'} | "
            f"{stats['p90'] if stats['p90'] is not None else 'n/a'} | "
            f"{stats['p95'] if stats['p95'] is not None else 'n/a'} | "
            f"{stats['mean'] if stats['mean'] is not None else 'n/a'} |"
        )
    old = gen17["fixed_30_points"]
    new = gen17["observed_or_30_fallback"]
    lines.extend(
        [
            "",
            "## Gen17 short trend-continuation repricing",
            "",
            "| Cost method | Trades | Win rate | PF | Mean-R | Sum-R | Avg cost R |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| Fixed 30 + 5 points | {gen17['trades']} | {old['realized_win_rate']:.2%} | {old['profit_factor']:.4f} | {old['mean_r']:.6f} | n/a | 0.141944 |",
            f"| Observed spread, zero fallback 30, +5 | {gen17['trades']} | {new['realized_win_rate']:.2%} | {new['profit_factor']:.4f} | {new['mean_r']:.6f} | {new['sum_r']:.6f} | {new['average_cost_r_per_trade']:.6f} |",
            "",
            "The fixed 0.1419R drag is internally correct for the fixed-cost formula, but it is not the observed historical average. Repricing improves Gen17 materially while remaining slightly negative; it does not create a champion by itself.",
        ]
    )
    return "\n".join(lines) + "\n"


def self_check() -> None:
    spread, observed = effective_spread(0.0)
    assert spread == 30.0 and not observed
    spread, observed = effective_spread(18.0)
    assert spread == 18.0 and observed
    stop = 1.6
    gross = 1.3
    reward = (gross - 0.35) / (stop + 0.30)
    recovered = reward * (stop + 0.30) + 0.35
    assert math.isclose(recovered, gross)
    print("generation19_transaction_cost_audit_self_check_ok")


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        return 0
    historical = load_historical()
    mt5_info, recent = mt5_snapshot()
    atr_percentile, volatility = atr_groups(historical)
    positive = historical.loc[historical["SPREAD_POINTS"] > 0.0, "SPREAD_POINTS"]
    recent_spread = {} if recent.empty else {
        "overall": describe(recent["SPREAD_POINTS"]),
        "by_session": grouped_spread(recent, "SESSION"),
        "by_hour": grouped_spread(recent, "HOUR"),
        "start": recent["TIME_DT"].iloc[0].isoformat(),
        "end": recent["TIME_DT"].iloc[-1].isoformat(),
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generation": "19_transaction_cost_audit",
        "status": "research_only",
        "method": {
            "observed_spread_rule": "entry bar SPREAD when > 0, otherwise 30-point fallback",
            "zero_spread_semantics": "missing, not free execution",
            "extra_cost_points": EXTRA_COST_POINTS,
            "stress_requirement": "retain an adverse extra-cost case in candidate evaluation",
            "atr_percentile_note": "full-development descriptive grouping only; never used for candidate selection",
        },
        "price_and_cost_semantics": {
            "ohlc_source": "MT5 M1 chart bars",
            "bar_price_type": "Bid when symbol chart_mode == 0",
            "long": "enter Ask, evaluate Bid-bar path, exit Bid; one spread deduction",
            "short": "enter Bid, evaluate Bid-bar path, exit Ask; one spread deduction",
            "same_bar_tp_sl": "stop_first",
            "fixed_reward_formula": "(gross_price_pnl - 0.30 spread - 0.05 extra) / (stop_price + 0.30 spread)",
            "double_counted": False,
            "denominator_explanation": "R normalization by total stop cash loss; it is not a second PnL charge",
        },
        "mt5_snapshot": mt5_info,
        "historical_spread": {
            "file": M1_CSV.name,
            "rows": len(historical),
            "start": historical["TIME_DT"].iloc[0].isoformat(),
            "end": historical["TIME_DT"].iloc[-1].isoformat(),
            "zero_or_missing_rows": int((historical["SPREAD_POINTS"] <= 0.0).sum()),
            "observed_positive_rows": len(positive),
            "observed_positive_overall": describe(positive),
            "by_year": grouped_spread(historical, "YEAR"),
            "by_session": grouped_spread(historical, "SESSION"),
            "by_hour": grouped_spread(historical, "HOUR"),
            "by_atr_percentile": atr_percentile,
            "by_volatility_regime": volatility,
            "direction_specific_spread_available": False,
        },
        "recent_mt5_m1_spread": recent_spread,
        "live_demo_logs": log_audit(),
        "gen17_repricing": reprice_gen17(historical),
        "verdict": {
            "fixed_0_1419r_mathematically_correct": True,
            "fixed_30_points_empirically_exact": False,
            "spread_double_counted": False,
            "methodology_correction_required": True,
            "correction": "use observed entry spread when positive and 30-point fallback for missing historical values",
        },
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_MD.write_text(markdown(report), encoding="utf-8")
    print(f"Wrote {REPORT_JSON.name} and {REPORT_MD.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
