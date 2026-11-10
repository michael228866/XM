"""Four commonly shared gold strategies, implemented as written and fixed before any result was seen.

1 EMA trend (H1, H4): EMA20 > EMA50 and price above both; enter on a pullback that touches EMA20 and closes
  above it, or on a break of the prior 20-bar high. Shorts mirrored.
2 Range (M15): price pierces the previous day's low/high and the bar closes back inside with a pin bar or
  engulfing candle -> fade it.
3 RSI14 + KD (M15, M5): RSI < 30 and stochastic %K crosses above %D below 20 -> long; mirrored for shorts.
4 News straddle (M1): 10-minute range before a high-impact USD release; buy stop above, sell stop below;
  first trigger within 15 min wins, SL = other side of the range, TP = 2x range, out after 60 min.
  Tested with 0 / 50 / 100 points of extra slippage on the stop fill.

Costs: 32-point spread floor (live median 31) plus overnight swap at XM's current GOLD# rates.

    python -B web_strategies_v1.py --output forward_reports/20261111_web_strategies_v1
"""
import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

import rules_lab_v1 as rl
from calendar_filter_v1 import load_events
from cross_asset_combo_v1 import bars
from strategy_lab_v1 import stats

rl.SWAP_CHARGES_PER_BAR.update({"H4": 7 / 29, "M15": 7 / 460, "M5": 7 / 1380})
TIMEFRAMES = {"H4": mt5.TIMEFRAME_H4, "H1": mt5.TIMEFRAME_H1, "M15": mt5.TIMEFRAME_M15, "M5": mt5.TIMEFRAME_M5}
# name: (timeframe, rule, side, tp_atr, sl_atr, hold_bars)
STRATEGIES = {
    "1_EMA_PULLBACK_H1_L": ("H1", "ema_pullback", "long", 3.0, 1.5, 48),
    "1_EMA_PULLBACK_H1_S": ("H1", "ema_pullback", "short", 3.0, 1.5, 48),
    "1_EMA_BREAKOUT_H1_L": ("H1", "ema_breakout", "long", 3.0, 1.5, 48),
    "1_EMA_BREAKOUT_H1_S": ("H1", "ema_breakout", "short", 3.0, 1.5, 48),
    "1_EMA_PULLBACK_H4_L": ("H4", "ema_pullback", "long", 3.0, 1.5, 24),
    "1_EMA_PULLBACK_H4_S": ("H4", "ema_pullback", "short", 3.0, 1.5, 24),
    "1_EMA_BREAKOUT_H4_L": ("H4", "ema_breakout", "long", 3.0, 1.5, 24),
    "1_EMA_BREAKOUT_H4_S": ("H4", "ema_breakout", "short", 3.0, 1.5, 24),
    "2_RANGE_M15_L": ("M15", "range_fade", "long", 2.0, 1.0, 32),
    "2_RANGE_M15_S": ("M15", "range_fade", "short", 2.0, 1.0, 32),
    "3_RSI_KD_M15_L": ("M15", "rsi_kd", "long", 1.5, 1.5, 16),
    "3_RSI_KD_M15_S": ("M15", "rsi_kd", "short", 1.5, 1.5, 16),
    "3_RSI_KD_M5_L": ("M5", "rsi_kd", "long", 1.5, 1.5, 24),
    "3_RSI_KD_M5_S": ("M5", "rsi_kd", "short", 1.5, 1.5, 24),
}
NEWS_RANGE_MIN, NEWS_TRIGGER_MIN, NEWS_HOLD_MIN = 10, 15, 60
SLIPPAGE_POINTS = (0, 50, 100)


def signals(f, rule, side):
    """Boolean per row: enter at this row's open, using only bars closed before it."""
    long = side == "long"
    o, h, l, c = f["OPEN"], f["HIGH"], f["LOW"], f["CLOSE"]
    if rule.startswith("ema"):
        e20, e50 = c.ewm(span=20, adjust=False).mean(), c.ewm(span=50, adjust=False).mean()
        trend = (e20 > e50) & (c > e50) if long else (e20 < e50) & (c < e50)
        if rule == "ema_pullback":
            touch = (l <= e20) & (c > e20) if long else (h >= e20) & (c < e20)
        else:
            touch = c > h.shift(1).rolling(20).max() if long else c < l.shift(1).rolling(20).min()
        sig = trend & touch
    elif rule == "range_fade":
        day = f["TIME_DT"].dt.normalize()
        daily = f.groupby(day).agg(hi=("HIGH", "max"), lo=("LOW", "min")).shift(1)  # previous day's range
        prev_hi, prev_lo = day.map(daily["hi"]), day.map(daily["lo"])
        be, se, bp, sp = rl.candles(f)
        sig = ((l <= prev_lo) & (c > prev_lo) & (be | bp)) if long else ((h >= prev_hi) & (c < prev_hi) & (se | sp))
    elif rule == "rsi_kd":
        r14 = rl.rsi(c, 14)
        k = 100 * (c - l.rolling(9).min()) / (h.rolling(9).max() - l.rolling(9).min())
        k = k.ewm(alpha=1 / 3, adjust=False).mean()
        d = k.ewm(alpha=1 / 3, adjust=False).mean()
        if long:
            sig = (r14 < 30) & (k > d) & (k.shift(1) <= d.shift(1)) & (k < 20)
        else:
            sig = (r14 > 70) & (k < d) & (k.shift(1) >= d.shift(1)) & (k > 80)
    else:
        raise ValueError(rule)
    return sig.fillna(False).shift(1, fill_value=False).to_numpy()


def straddle(m1, event, spread, slip):
    """One event on M1 bars around it; returns (R, side) or None. Bars: bid OHLC; ask = bid + spread."""
    before = m1[(m1["TIME_DT"] >= event - timedelta(minutes=NEWS_RANGE_MIN)) & (m1["TIME_DT"] < event)]
    after = m1[(m1["TIME_DT"] >= event) & (m1["TIME_DT"] < event + timedelta(minutes=NEWS_HOLD_MIN))]
    if len(before) < NEWS_RANGE_MIN - 2 or len(after) < 10:
        return None
    hi, lo = before["HIGH"].max() + spread, before["LOW"].min()  # buy stop on ask above, sell stop on bid below
    width = hi - lo
    side = entry = None
    for i, bar in enumerate(after.itertuples()):
        if bar.TIME_DT >= event + timedelta(minutes=NEWS_TRIGGER_MIN) and side is None:
            return None
        if side is None:
            up, down = bar.HIGH + spread >= hi, bar.LOW <= lo
            if not (up or down):
                continue
            # Both stops in one bar: assume the worse case, the first fill gets stopped by the other side.
            if up and down:
                return -1.0 - slip / width, "whipsaw"
            side = "long" if up else "short"
            entry = hi + slip if up else lo - slip
            sl, tp = (lo, entry + 2 * width) if up else (hi, entry - 2 * width)
            risk = abs(entry - sl)
            # Same bar after the fill: only a stop is counted (conservative).
            if (side == "long" and bar.LOW <= sl) or (side == "short" and bar.HIGH + spread >= sl):
                return -1.0, side
            continue
        if side == "long":
            if bar.LOW <= sl:
                return (sl - entry) / risk, side
            if bar.HIGH >= tp:
                return (tp - entry) / risk, side
        else:
            if bar.HIGH + spread >= sl:
                return (entry - sl) / risk, side
            if bar.LOW + spread <= tp:
                return (entry - tp) / risk, side
    if side is None:
        return None
    last = after["CLOSE"].iloc[-1]
    return ((last - entry) if side == "long" else (entry - last - spread)) / risk, side


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    end = datetime.now(timezone.utc)
    events = [pd.Timestamp(t) for t in load_events() if rl.START.replace(tzinfo=None) <= pd.Timestamp(t) < end.replace(tzinfo=None)]
    if not mt5.initialize(path=os.environ.get("XM_TERMINAL_PATH", r"D:\XM2\terminal64.exe")):
        raise SystemExit(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        info = mt5.symbol_info("GOLD#")
        frames = {tf: bars("GOLD#", code, rl.START - timedelta(days=60), end) for tf, code in TIMEFRAMES.items()}
        m1 = bars("GOLD#", mt5.TIMEFRAME_M1, rl.START, end)
    finally:
        mt5.shutdown()
    swap = {"long": info.swap_long * info.point, "short": info.swap_short * info.point}
    report = {"swap_per_night_price": swap, "strategies": {}}
    for tf in frames:
        f = rl.prepare(frames[tf])
        frames[tf] = f[(f["TIME_DT"] >= rl.START.replace(tzinfo=None)) & f["ATR"].notna()].reset_index(drop=True)
    for name, (tf, rule, side, tp, sl, hold) in STRATEGIES.items():
        frame = frames[tf]
        result = rl.evaluate(frame, tf, signals(frame, rule, side), side, tp, sl, hold, swap[side])
        report["strategies"][name] = result
        t = result["test_2024_on"]
        print(f"{name:22s} dev {result['dev_2014_2023'].get('avg_r', float('nan')):+.3f} (n={result['dev_2014_2023']['n']})  "
              f"test {t.get('avg_r', float('nan')):+.3f} (n={t['n']})  always {result['test_always_in'].get('avg_r', float('nan')):+.3f}  "
              f"random>= {result.get('test_random_share_at_or_above')}", flush=True)

    m1 = m1.set_index("TIME_DT", drop=False)
    for slip_points in SLIPPAGE_POINTS:
        rows = []
        for event in events:
            window = m1.loc[event - timedelta(minutes=NEWS_RANGE_MIN): event + timedelta(minutes=NEWS_HOLD_MIN)]
            spread = max(float(window["SPREAD"].median()) * info.point, float(window["OPEN"].iloc[0]) * rl.MIN_SPREAD_FRACTION) if len(window) else 0
            out = straddle(window, event, spread, slip_points * info.point)  # <= 60 min, no overnight swap
            if out is not None:
                rows.append({"event": event, "r": out[0], "side": out[1]})
        t = pd.DataFrame(rows)
        name = f"4_NEWS_STRADDLE_slip{slip_points}"
        test = t["event"] >= rl.TEST_START
        report["strategies"][name] = res = {"dev_2014_2023": stats(t.loc[~test, "r"]), "test_2024_on": stats(t.loc[test, "r"]),
                                            "whipsaw_share": round(float((t["side"] == "whipsaw").mean()), 3)}
        print(f"{name:22s} dev {res['dev_2014_2023'].get('avg_r', float('nan')):+.3f} (n={res['dev_2014_2023']['n']})  "
              f"test {res['test_2024_on'].get('avg_r', float('nan')):+.3f} (n={res['test_2024_on']['n']})  "
              f"whipsaw {res['whipsaw_share']}", flush=True)
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
