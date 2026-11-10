"""Rules lab: well-known public trading rules on GOLD#, no fitting, same costs and baselines as the H1 lab.

Rules and exits are fixed below before any result was seen (breakouts, RSI(2), candlestick patterns).
Each signal enters at the next bar's open; exits are first-touch ATR TP/SL or a time stop (strategy_lab_v1).
Costs: spread (with the H1 lab floor) plus overnight swap at XM's current GOLD# rates.

    python -B rules_lab_v1.py --output forward_reports/20261108_rules_lab_v1

Reports 2014-2023 (dev) and 2024+ (test) separately; judge on test vs always-in and random entries.
Many rules are tested at once, so one passing by luck is expected; a pass only earns a frozen forward test.
"""
import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from gold_recent_walk_forward import copy_rates
from strategy_lab_v1 import first_touch, stats, trade

START = datetime(2014, 1, 1, tzinfo=timezone.utc)
TEST_START = pd.Timestamp("2024-01-01")
MIN_SPREAD_FRACTION = 0.00008  # same floor as h1_lab_v1
RANDOM_DRAWS = 300
# Swap charges per bar held: 7 nightly charges a week (Wednesday x3) over ~115 H1 bars or 5 D1 bars.
SWAP_CHARGES_PER_BAR = {"H1": 7 / 115, "D1": 7 / 5}
# XM server time is London + 2 h all year. Asia range 02-09, London open bar 10.
ASIA_HOURS, LONDON_OPEN = range(2, 10), 10

# name: (timeframe, rule, side, tp_atr, sl_atr, hold_bars)
STRATEGIES = {
    "D1_DONCH20_L": ("D1", "donchian20", "long", 4.0, 2.0, 20),
    "D1_DONCH20_S": ("D1", "donchian20", "short", 4.0, 2.0, 20),
    "D1_DONCH55_L": ("D1", "donchian55", "long", 4.0, 2.0, 40),
    "D1_DONCH55_S": ("D1", "donchian55", "short", 4.0, 2.0, 40),
    "D1_RSI2_L": ("D1", "rsi2", "long", 1.0, 2.0, 5),
    "D1_RSI2_S": ("D1", "rsi2", "short", 1.0, 2.0, 5),
    "D1_ENGULF_L": ("D1", "engulf", "long", 3.0, 1.5, 10),
    "D1_ENGULF_S": ("D1", "engulf", "short", 3.0, 1.5, 10),
    "D1_PINBAR_L": ("D1", "pinbar", "long", 3.0, 1.5, 10),
    "D1_PINBAR_S": ("D1", "pinbar", "short", 3.0, 1.5, 10),
    "H1_ASIA_BRK_L": ("H1", "asia_break", "long", 3.0, 1.5, 12),
    "H1_ASIA_BRK_S": ("H1", "asia_break", "short", 3.0, 1.5, 12),
    "H1_LONDON_ORB_L": ("H1", "london_orb", "long", 3.0, 1.5, 12),
    "H1_LONDON_ORB_S": ("H1", "london_orb", "short", 3.0, 1.5, 12),
    "H1_RSI2_L": ("H1", "rsi2", "long", 1.0, 2.0, 12),
    "H1_RSI2_S": ("H1", "rsi2", "short", 1.0, 2.0, 12),
    "H1_ENGULF_L": ("H1", "engulf_trend", "long", 3.0, 1.5, 24),
    "H1_ENGULF_S": ("H1", "engulf_trend", "short", 3.0, 1.5, 24),
    "H1_PINBAR_L": ("H1", "pinbar_trend", "long", 3.0, 1.5, 24),
    "H1_PINBAR_S": ("H1", "pinbar_trend", "short", 3.0, 1.5, 24),
}


def prepare(bars):
    frame = bars.sort_values("TIME_DT").reset_index(drop=True)
    tr = pd.concat([frame["HIGH"] - frame["LOW"], (frame["HIGH"] - frame["CLOSE"].shift()).abs(),
                    (frame["LOW"] - frame["CLOSE"].shift()).abs()], axis=1).max(axis=1)
    frame["ATR"] = tr.rolling(14).mean().shift(1)  # known at entry
    frame["SPREAD"] = np.maximum(frame["SPREAD"], frame["OPEN"] * MIN_SPREAD_FRACTION / 0.01)
    return frame


def rsi(close, n):
    diff = close.diff()
    gain = diff.clip(lower=0).rolling(n).mean()
    loss = (-diff.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-9))


def candles(f):
    """Pattern on each closed bar; caller shifts so the trade enters on the next bar's open."""
    o, h, l, c = f["OPEN"], f["HIGH"], f["LOW"], f["CLOSE"]
    rng = (h - l).replace(0, np.nan)
    po, pc = o.shift(), c.shift()
    bull_engulf = (pc < po) & (c > o) & (c >= po) & (o <= pc)
    bear_engulf = (pc > po) & (c < o) & (c <= po) & (o >= pc)
    lower_wick, upper_wick = np.minimum(o, c) - l, h - np.maximum(o, c)
    bull_pin = (lower_wick >= rng * 2 / 3) & (c >= l + rng * 2 / 3)
    bear_pin = (upper_wick >= rng * 2 / 3) & (c <= l + rng / 3)
    return bull_engulf, bear_engulf, bull_pin, bear_pin


def first_per_day(cond, day):
    return cond & (cond.astype(int).groupby(day).cumsum() == 1)


def signals(frame, tf, rule, side, daily_up=None):
    """Boolean per row: enter long/short at this row's open. Uses only bars closed before it."""
    long = side == "long"
    c = frame["CLOSE"]
    if rule.startswith("donchian"):
        n = int(rule[len("donchian"):])
        sig = c > frame["HIGH"].shift(1).rolling(n).max() if long else c < frame["LOW"].shift(1).rolling(n).min()
    elif rule == "rsi2":
        sma = c.rolling(200).mean()
        r2 = rsi(c, 2)
        sig = (c > sma) & (r2 < 10) if long else (c < sma) & (r2 > 90)
    elif rule in ("engulf", "pinbar", "engulf_trend", "pinbar_trend"):
        be, se, bp, sp = candles(frame)
        sig = (be if long else se) if rule.startswith("engulf") else (bp if long else sp)
        if tf == "D1":  # tutorials: trade patterns with the trend
            trend = c > c.rolling(50).mean()
            sig = sig & (trend if long else ~trend)
        else:
            up = frame["TIME_DT"].dt.normalize().map(daily_up)
            sig = sig & (up == (1 if long else 0))
    elif rule in ("asia_break", "london_orb"):
        day, hour = frame["TIME_DT"].dt.normalize(), frame["TIME_DT"].dt.hour
        window = hour.isin(ASIA_HOURS) if rule == "asia_break" else hour == LONDON_OPEN
        hi = frame["HIGH"].where(window).groupby(day).transform("max")
        lo = frame["LOW"].where(window).groupby(day).transform("min")
        # Signal bar must close after the range is complete (asia: from 10, london: from 11) and before 17.
        allowed = hour.between(10, 16) if rule == "asia_break" else hour.between(11, 16)
        sig = first_per_day(allowed & ((c > hi) if long else (c < lo)), day)
    else:
        raise ValueError(rule)
    return sig.fillna(False).shift(1, fill_value=False).to_numpy()


def evaluate(frame, tf, sig, side, tp, sl, hold, swap_per_charge):
    r, after = first_touch(frame, side, tp, sl, hold)
    # swap in R: price units per oz per charge, times charges while held, over the SL distance.
    r = r + swap_per_charge * SWAP_CHARGES_PER_BAR[tf] * after / (frame["ATR"].to_numpy() * sl)
    test = (frame["TIME_DT"] >= TEST_START).to_numpy()
    out = {}
    for name, mask in (("dev_2014_2023", ~test), ("test_2024_on", test)):
        f, s, rr, aa = frame[mask], sig[mask].astype(float), r[mask], after[mask]
        taken = trade(f, s, 0.5, rr, aa)
        out[name] = stats(taken["r"])
        if name == "test_2024_on":
            out["test_always_in"] = stats(trade(f, np.ones(len(f)), 0.5, rr, aa)["r"])
            valid = np.flatnonzero(np.isfinite(rr))
            k = min(int((s.astype(bool) & np.isfinite(rr)).sum()), len(valid))
            rng = np.random.default_rng(0)
            draws = []
            for _ in range(RANDOM_DRAWS if k else 0):
                pick = np.zeros(len(f))
                pick[rng.choice(valid, size=k, replace=False)] = 1
                draws.append(trade(f, pick, 0.5, rr, aa)["r"].mean())
            if draws and len(taken):
                out["test_random_share_at_or_above"] = round(float(np.mean(np.array(draws) >= taken["r"].mean())), 3)
    return out


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
        frames = {"H1": copy_rates("H1", START - timedelta(days=60), end),
                  "D1": copy_rates("Daily", START - timedelta(days=400), end)}
        info = mt5.symbol_info("GOLD#")
    finally:
        mt5.shutdown()
    if info.swap_mode != 0:
        raise SystemExit(f"swap_mode {info.swap_mode}: only points-based swap (0) is handled")
    swap = {"long": info.swap_long * info.point, "short": info.swap_short * info.point}
    frames = {tf: prepare(f) for tf, f in frames.items()}
    d1 = frames["D1"]
    # Daily trend for H1 patterns: last closed day's close vs its 20-day mean, keyed by the next day.
    up = (d1["CLOSE"] > d1["CLOSE"].rolling(20).mean()).astype(int)
    daily_up = pd.Series(up.shift(1).to_numpy(), index=d1["TIME_DT"].dt.normalize())
    for tf in frames:
        f = frames[tf]
        frames[tf] = f[(f["TIME_DT"] >= START.replace(tzinfo=None)) & f["ATR"].notna()].reset_index(drop=True)

    report = {"swap_per_charge_price": swap, "test_from": str(TEST_START.date()), "strategies": {}}
    for name, (tf, rule, side, tp, sl, hold) in STRATEGIES.items():
        frame = frames[tf]
        result = evaluate(frame, tf, signals(frame, tf, rule, side, daily_up), side, tp, sl, hold, swap[side])
        report["strategies"][name] = result
        t = result["test_2024_on"]
        print(f"{name:17s} dev {result['dev_2014_2023'].get('avg_r', float('nan')):+.3f} "
              f"(n={result['dev_2014_2023']['n']})  test {t.get('avg_r', float('nan')):+.3f} (n={t['n']})  "
              f"always {result['test_always_in'].get('avg_r', float('nan')):+.3f}  "
              f"random>= {result.get('test_random_share_at_or_above')}", flush=True)
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
