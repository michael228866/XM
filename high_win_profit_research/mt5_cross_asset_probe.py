from __future__ import annotations

import csv
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd


RESEARCH_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = RESEARCH_DIR / "mt5_cross_asset_probe.csv"

SYMBOLS = [
    "GOLD#",
    "SILVER#",
    "BTCUSD#",
    "ETHUSD#",
    "LTCUSD#",
    "XRPUSD#",
    "SOLUSD#",
    "XPTUSD#",
    "XPDUSD#",
]


def atr_summary(symbol: str, bars: int = 50_000) -> dict:
    if not mt5.symbol_select(symbol, True):
        return {"symbol": symbol, "status": "select_failed"}

    info = mt5.symbol_info(symbol)
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, bars)
    tick = mt5.symbol_info_tick(symbol)
    if info is None or rates is None or len(rates) < 1_000:
        return {"symbol": symbol, "status": "insufficient_data"}

    df = pd.DataFrame(rates)
    true_range = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(14).mean().dropna().tail(10_000)
    mean_atr = float(atr.mean())
    median_atr = float(atr.median())
    spec_spread_price = float(info.spread * info.point)
    current_spread_price = (
        float(tick.ask - tick.bid) if tick is not None else spec_spread_price
    )
    spec_spread_atr = spec_spread_price / mean_atr if mean_atr > 0 else float("inf")
    current_spread_atr = (
        current_spread_price / mean_atr if mean_atr > 0 else float("inf")
    )

    if current_spread_atr <= 0.25:
        verdict = "strong_candidate"
    elif current_spread_atr <= 0.75:
        verdict = "research_candidate"
    elif current_spread_atr <= 1.50:
        verdict = "costly"
    else:
        verdict = "avoid_for_m1"

    return {
        "symbol": symbol,
        "status": "ok",
        "m1_bars": len(df),
        "digits": info.digits,
        "trade_mode": info.trade_mode,
        "mean_atr_m1": round(mean_atr, 6),
        "median_atr_m1": round(median_atr, 6),
        "spec_spread_price": round(spec_spread_price, 6),
        "current_spread_price": round(current_spread_price, 6),
        "spec_spread_atr": round(spec_spread_atr, 4),
        "current_spread_atr": round(current_spread_atr, 4),
        "verdict": verdict,
    }


def main() -> int:
    if not mt5.initialize():
        print(f"MT5 initialize failed: {mt5.last_error()}")
        return 1

    try:
        rows = [atr_summary(symbol) for symbol in SYMBOLS]
    finally:
        mt5.shutdown()

    fieldnames = [
        "symbol",
        "status",
        "m1_bars",
        "digits",
        "trade_mode",
        "mean_atr_m1",
        "median_atr_m1",
        "spec_spread_price",
        "current_spread_price",
        "spec_spread_atr",
        "current_spread_atr",
        "verdict",
    ]
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUTPUT_FILE}")
    for row in rows:
        print(
            f"{row['symbol']}: {row.get('verdict', row['status'])} "
            f"spread/ATR={row.get('current_spread_atr', '')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
