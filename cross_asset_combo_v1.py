"""Combo A (net long-minus-short rule count >= 2, long, TP 3 / SL 1.5 H1 ATR, 48 h) unchanged on other markets.

Nothing is re-picked: rules, k=2 and exits come from the gold combo lab. Spread floor per symbol = median of
its recent nonzero spreads (older bars often report 0); swap from XM's current per-symbol settings
(swap_mode 0 on metals is still charged, read as points).

    python -B cross_asset_combo_v1.py --output forward_reports/20261111_cross_asset_combo_v1
"""
import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

import combo_lab_v1 as cl
import rules_lab_v1 as rl
from gemini import PRICE_PER_POINT

SYMBOLS = ["GOLD#", "SILVER#", "US500Cash#", "US100Cash#", "US30Cash#", "GER40Cash#", "UK100Cash#", "JP225Cash#",
           "OILCash#", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
K, TP, SL, HOLD = 2, 3.0, 1.5, 48


def bars(symbol, timeframe, start, end):
    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    if rates is None or len(rates) == 0:
        raise SystemExit(f"{symbol}: no rates: {mt5.last_error()}")
    frame = pd.DataFrame(rates).rename(columns={"open": "OPEN", "high": "HIGH", "low": "LOW", "close": "CLOSE",
                                                "spread": "SPREAD"})
    frame["TIME_DT"] = pd.to_datetime(frame["time"], unit="s")
    return frame


def prepare(frame, info, floor_frac):
    """first_touch reads SPREAD in gold points (PRICE_PER_POINT), so convert this symbol's spread to that unit."""
    frame = frame.sort_values("TIME_DT").reset_index(drop=True)
    tr = pd.concat([frame["HIGH"] - frame["LOW"], (frame["HIGH"] - frame["CLOSE"].shift()).abs(),
                    (frame["LOW"] - frame["CLOSE"].shift()).abs()], axis=1).max(axis=1)
    frame["ATR"] = tr.rolling(14).mean().shift(1)
    spread_price = np.maximum(frame["SPREAD"] * info.point, frame["OPEN"] * floor_frac)
    frame["SPREAD"] = spread_price / PRICE_PER_POINT
    return frame


def swap_price_per_unit(info):
    """Long swap per night in price units per 1 unit of the instrument (negative = cost)."""
    # XM reports swap_mode 0 on GOLD#/SILVER# but still charges overnight swap; treat as points like mode 1.
    if info.swap_mode in (0, 1):
        return info.swap_long * info.point
    if info.swap_mode in (2, 3):  # money per lot in symbol/margin currency, assumed = quote currency
        return info.swap_long / info.trade_contract_size
    raise SystemExit(f"{info.name}: swap_mode {info.swap_mode} not handled")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    end = datetime.now(timezone.utc)
    if not mt5.initialize(path=os.environ.get("XM_TERMINAL_PATH", r"D:\XM2\terminal64.exe")):
        raise SystemExit(f"MT5 initialize failed: {mt5.last_error()}")
    data = {}
    try:
        for symbol in SYMBOLS:
            if not mt5.symbol_select(symbol, True):
                raise SystemExit(f"{symbol}: symbol_select failed")
            info = mt5.symbol_info(symbol)
            h1 = bars(symbol, mt5.TIMEFRAME_H1, rl.START - timedelta(days=60), end)
            d1 = bars(symbol, mt5.TIMEFRAME_D1, rl.START - timedelta(days=400), end)
            recent = h1.tail(24 * 250)
            nonzero = recent.loc[recent["SPREAD"] > 0]
            floor_frac = float((nonzero["SPREAD"] * info.point / nonzero["OPEN"]).median()) if len(nonzero) else 0.0
            data[symbol] = (info, h1, d1, floor_frac)
    finally:
        mt5.shutdown()

    report = {"k": K, "exits": {"tp": TP, "sl": SL, "hold": HOLD}, "symbols": {}}
    for symbol, (info, h1, d1, floor_frac) in data.items():
        h1, d1 = prepare(h1, info, floor_frac), prepare(d1, info, floor_frac)
        rules = cl.rule_columns(h1, d1)
        net = rules[[c for c in rules if c.endswith("_L")]].sum(axis=1) - rules[[c for c in rules if c.endswith("_S")]].sum(axis=1)
        keep = ((h1["TIME_DT"] >= rl.START.replace(tzinfo=None)) & h1["ATR"].notna()).to_numpy()
        frame = h1[keep].reset_index(drop=True)
        sig = (net[keep] >= K).to_numpy()
        swap = swap_price_per_unit(info)
        result = rl.evaluate(frame, "H1", sig, "long", TP, SL, HOLD, swap)
        result.update({"first_bar": str(frame["TIME_DT"].iloc[0]), "spread_floor_frac": floor_frac,
                       "swap_long_price_per_night": swap})
        report["symbols"][symbol] = result
        dev, test, always = result["dev_2014_2023"], result["test_2024_on"], result["test_always_in"]
        print(f"{symbol:11s} dev {dev.get('avg_r', float('nan')):+.3f} (n={dev['n']})  test {test.get('avg_r', float('nan')):+.3f} "
              f"(n={test['n']})  always {always.get('avg_r', float('nan')):+.3f}  random>= {result.get('test_random_share_at_or_above')}",
              flush=True)
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
