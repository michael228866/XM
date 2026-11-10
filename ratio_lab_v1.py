"""Gold/silver ratio mean reversion on daily bars: market-neutral, so it does not need gold to trend.

z = (log(gold/silver) - rolling mean) / rolling std at each close. |z| > entry: short the rich leg, buy the cheap
one (equal notional, half each), from the next bar. Exit when z crosses back through 0 or after max_hold days.
Costs: each leg's spread on entry and exit, plus overnight swap on both legs at XM's current rates
(7 charges a week over 5 daily bars; swap_mode 0 on metals is still charged, read as points).

    python -B ratio_lab_v1.py --output forward_reports/20261111_ratio_lab_v1
"""
import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from cross_asset_combo_v1 import bars

START = pd.Timestamp("2014-01-01")
TEST_START = pd.Timestamp("2024-01-01")
GOLD_SPREAD_FLOOR = 0.00008
SWAP_CHARGES_PER_BAR = 7 / 5
# Fixed before any result was seen: (lookback days, entry |z|, max hold days).
VARIANTS = {"L60_z2": (60, 2.0, 20), "L120_z2": (120, 2.0, 40)}


def backtest(close, spread_frac, swap_frac, lookback, entry, max_hold):
    """Trades as (entry date, net return on gross notional). swap_frac[symbol][side] = per night, fraction of price."""
    lr = np.log(close["GOLD#"] / close["SILVER#"])
    z = ((lr - lr.rolling(lookback).mean()) / lr.rolling(lookback).std()).to_numpy()
    ret = close.pct_change().fillna(0.0)
    long_ratio = ((ret["GOLD#"] - ret["SILVER#"]) / 2).to_numpy()
    cost = (spread_frac["GOLD#"] + spread_frac["SILVER#"]) / 2
    carry = {1: (swap_frac["GOLD#"]["long"] + swap_frac["SILVER#"]["short"]) / 2 * SWAP_CHARGES_PER_BAR,
             -1: (swap_frac["GOLD#"]["short"] + swap_frac["SILVER#"]["long"]) / 2 * SWAP_CHARGES_PER_BAR}
    trades, pos, start = [], 0, 0
    for t in range(len(z)):
        if pos == 0:
            if np.isfinite(z[t]) and abs(z[t]) > entry:
                pos, start = (-1 if z[t] > 0 else 1), t  # decided at close t, held from bar t+1
        elif np.sign(z[t]) == pos or t - start >= max_hold:
            days = t - start
            gross = pos * long_ratio[start + 1:t + 1].sum()
            trades.append((close.index[start], gross + carry[pos] * days - 2 * cost, days))
            pos = 0
    return pd.DataFrame(trades, columns=["entry", "ret", "days"])


def summary(t):
    if not len(t):
        return {"trades": 0}
    r = t["ret"].to_numpy()
    years = (t["entry"].iloc[-1] - t["entry"].iloc[0]).days / 365.25 or 1
    return {"trades": len(r), "win_rate": round(float((r > 0).mean()), 3), "avg_trade_pct": round(float(r.mean() * 100), 3),
            "t_stat": round(float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))), 2) if len(r) > 1 else None,
            "avg_days": round(float(t["days"].mean()), 1), "sum_pct_per_year": round(float(r.sum() / years * 100), 2)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    end = datetime.now(timezone.utc)
    if not mt5.initialize(path=os.environ.get("XM_TERMINAL_PATH", r"D:\XM2\terminal64.exe")):
        raise SystemExit(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        start = datetime(2013, 1, 1, tzinfo=timezone.utc)
        frames = {s: bars(s, mt5.TIMEFRAME_D1, start, end) for s in ("GOLD#", "SILVER#")}
        infos = {s: mt5.symbol_info(s) for s in frames}
    finally:
        mt5.shutdown()
    close = pd.concat({s: f.set_index("TIME_DT")["CLOSE"] for s, f in frames.items()}, axis=1).dropna()
    last = close.iloc[-1]
    recent = frames["SILVER#"].tail(250)
    recent = recent[recent["SPREAD"] > 0]
    spread_frac = {"GOLD#": GOLD_SPREAD_FLOOR,
                   "SILVER#": float((recent["SPREAD"] * infos["SILVER#"].point / recent["CLOSE"]).median())}
    # shortcut: today's swap rates applied to all history; upgrade with historical rates if this ever passes.
    swap_frac = {s: {"long": i.swap_long * i.point / last[s], "short": i.swap_short * i.point / last[s]}
                 for s, i in infos.items()}
    report = {"spread_frac": spread_frac, "swap_frac_per_night": swap_frac, "variants": {}}
    for name, (lookback, entry, max_hold) in VARIANTS.items():
        t = backtest(close, spread_frac, swap_frac, lookback, entry, max_hold)
        t = t[t["entry"] >= START]
        report["variants"][name] = res = {"dev_2014_2023": summary(t[t["entry"] < TEST_START]),
                                          "test_2024_on": summary(t[t["entry"] >= TEST_START])}
        print(name, json.dumps(res, ensure_ascii=False), flush=True)
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
