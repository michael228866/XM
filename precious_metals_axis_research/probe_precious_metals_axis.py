from __future__ import annotations

import csv
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd


RESEARCH_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = RESEARCH_DIR / "precious_metals_axis_probe.csv"

SYMBOLS = ["GOLD#", "SILVER#", "XAUEUR#", "GAUUSD#", "XPTUSD#", "XPDUSD#"]
TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
}


def classify(spread_atr: float) -> str:
    if spread_atr <= 0.15:
        return "strong"
    if spread_atr <= 0.35:
        return "usable"
    if spread_atr <= 0.75:
        return "research_only"
    return "too_expensive"


def timeframe_probe(symbol: str, timeframe_name: str, timeframe: int) -> dict:
    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 20_000)
    if info is None or rates is None or len(rates) < 1_000:
        return {
            "symbol": symbol,
            "timeframe": timeframe_name,
            "status": "insufficient_data",
        }

    df = pd.DataFrame(rates)
    true_range = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = float(true_range.rolling(14).mean().dropna().tail(5_000).mean())
    spread_price = (
        float(tick.ask - tick.bid)
        if tick is not None
        else float(info.spread * info.point)
    )
    spread_atr = spread_price / atr if atr > 0 else float("inf")
    return {
        "symbol": symbol,
        "timeframe": timeframe_name,
        "status": "ok",
        "bars": len(df),
        "digits": info.digits,
        "mean_atr": round(atr, 6),
        "current_spread_price": round(spread_price, 6),
        "spread_atr": round(spread_atr, 4),
        "verdict": classify(spread_atr),
    }


def main() -> int:
    if not mt5.initialize():
        print(f"MT5 initialize failed: {mt5.last_error()}")
        return 1

    rows = []
    try:
        for symbol in SYMBOLS:
            mt5.symbol_select(symbol, True)
            for timeframe_name, timeframe in TIMEFRAMES.items():
                rows.append(timeframe_probe(symbol, timeframe_name, timeframe))
    finally:
        mt5.shutdown()

    fieldnames = [
        "symbol",
        "timeframe",
        "status",
        "bars",
        "digits",
        "mean_atr",
        "current_spread_price",
        "spread_atr",
        "verdict",
    ]
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUTPUT_FILE}")
    for row in rows:
        if row["status"] != "ok":
            print(f"{row['symbol']} {row['timeframe']}: {row['status']}")
            continue
        print(
            f"{row['symbol']} {row['timeframe']}: {row['verdict']} "
            f"spread/ATR={row['spread_atr']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
