"""Daily multi-asset trend following (time-series momentum), one fixed rule set, no fitting.

Each week: per asset, signal = average sign of its 63/126/252-day return; size = signal x (target vol / its
60-day realized vol). Costs: spread on every position change, plus overnight swap at XM's current rates.
Reads MT5 daily bars only, never trades.

    python -B trend_daily_v1.py --output forward_reports/20261105_trend_daily_v1
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

# Fixed before any result was seen.
UNIVERSE = ["GOLD#", "SILVER#", "OILCash#", "US500Cash#", "US100Cash#", "US30Cash#", "GER40Cash#", "JP225Cash#",
            "UK100Cash#", "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]
LOOKBACKS = (63, 126, 252)
VOL_WINDOW = 60
ASSET_TARGET_VOL = 0.03          # per asset, annualized; 16 partly correlated assets -> roughly 8-12% portfolio
REBALANCE_WEEKDAY = 4            # Friday close; new weights apply from the next bar
START = datetime(2009, 1, 1, tzinfo=timezone.utc)
REPORT_FROM = "2013-01-01"       # first date with every lookback warmed up for most assets
SPLIT = "2020-01-01"


def load_closes(end):
    closes, costs = {}, {}
    for symbol in UNIVERSE:
        if not mt5.symbol_select(symbol, True):
            raise SystemExit(f"{symbol}: symbol_select failed")
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_D1, START, end)
        if rates is None or len(rates) == 0:
            raise SystemExit(f"{symbol}: no D1 rates: {mt5.last_error()}")
        bars = pd.DataFrame(rates)
        bars.index = pd.to_datetime(bars["time"], unit="s")
        closes[symbol] = bars["close"]
        info = mt5.symbol_info(symbol)
        price = float(bars["close"].iloc[-1])
        recent_spread = bars["spread"].iloc[-250:]
        spread_points = float(recent_spread[recent_spread > 0].median()) if (recent_spread > 0).any() else info.spread
        # Swap per day as a fraction of price. Points-based (modes 0/1) vs money per lot (mode 3, per contract unit).
        unit = info.point if info.swap_mode in (0, 1) else 1.0 / info.trade_contract_size
        costs[symbol] = {"spread_frac": spread_points * info.point / price,
                         "swap_long_frac": info.swap_long * unit / price,
                         "swap_short_frac": info.swap_short * unit / price,
                         "swap_mode": info.swap_mode}
    return pd.DataFrame(closes).sort_index().ffill(), costs


def weights_from(closes):
    """Weight known at day t's close, applied to the return from t to t+1 by the caller."""
    rets = closes.pct_change(fill_method=None)
    signal = sum(np.sign(closes / closes.shift(n) - 1) for n in LOOKBACKS) / len(LOOKBACKS)
    vol = rets.rolling(VOL_WINDOW).std() * np.sqrt(252)
    raw = (signal * ASSET_TARGET_VOL / vol).replace([np.inf, -np.inf], np.nan)
    weekly = raw[raw.index.dayofweek == REBALANCE_WEEKDAY].reindex(raw.index)
    return weekly.ffill().fillna(0.0)


def backtest(closes, costs):
    rets = closes.pct_change(fill_method=None).fillna(0.0)
    w = weights_from(closes)
    held = w.shift(1).fillna(0.0)                       # yesterday's decision earns today's return
    gross = held * rets
    spread = pd.Series({s: c["spread_frac"] for s, c in costs.items()})
    trading = held.diff().abs().fillna(held.abs()) * spread
    # shortcut: today's swap rates applied to all history; rates were near zero 2014-2021, so this overstates costs then.
    long_cost = pd.Series({s: c["swap_long_frac"] for s, c in costs.items()})
    short_cost = pd.Series({s: c["swap_short_frac"] for s, c in costs.items()})
    days = closes.index.to_series().diff().dt.days.fillna(1).clip(upper=4)  # weekends/holidays accrue swap
    swap = (held.clip(lower=0) * long_cost + (-held).clip(lower=0) * short_cost).mul(days, axis=0)
    net = gross - trading + swap
    return gross, net, held


def summarize(daily):
    daily = daily[daily.index >= REPORT_FROM]
    equity = (1 + daily).cumprod()
    years = len(daily) / 252
    ann = equity.iloc[-1] ** (1 / years) - 1
    vol = daily.std() * np.sqrt(252)
    dd = (equity / equity.cummax() - 1).min()
    return {"annual_return": round(float(ann), 4), "annual_vol": round(float(vol), 4),
            "sharpe": round(float(daily.mean() / daily.std() * np.sqrt(252)), 2), "max_drawdown": round(float(dd), 4)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    if not mt5.initialize(path=os.environ.get("XM_TERMINAL_PATH", r"D:\XM2\terminal64.exe")):
        raise SystemExit(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        closes, costs = load_closes(datetime.now(timezone.utc))
    finally:
        mt5.shutdown()

    gross, net, held = backtest(closes, costs)
    port_gross, port_net = gross.sum(axis=1), net.sum(axis=1)
    # Benchmark: same vol-scaled sizes but always long, to separate trend timing from just holding risk assets.
    always = (closes.pct_change(fill_method=None).fillna(0) * held.abs()).sum(axis=1)
    report_net = port_net[port_net.index >= REPORT_FROM]
    report = {
        "universe": UNIVERSE, "lookbacks": LOOKBACKS, "asset_target_vol": ASSET_TARGET_VOL,
        "window": [str(report_net.index[0].date()), str(report_net.index[-1].date())],
        "gross": summarize(port_gross), "net": summarize(port_net),
        "net_before_split": summarize(port_net[port_net.index < SPLIT]),
        "net_from_split": summarize(port_net[port_net.index >= SPLIT]),
        "always_long_same_sizes_gross": summarize(always),
        "net_by_year": {str(y): round(float((1 + g).prod() - 1), 4) for y, g in report_net.groupby(report_net.index.year)},
        "net_contribution_by_asset": {s: round(float(net.loc[net.index >= REPORT_FROM, s].sum()), 4)
                                      for s in UNIVERSE},
        "avg_gross_leverage": round(float(held[held.index >= REPORT_FROM].abs().sum(axis=1).mean()), 2),
        "costs": costs,
    }
    args.output.mkdir(parents=True)
    port_net.to_frame("net_daily_return").to_csv(args.output / "daily_returns.csv")
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "costs"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
