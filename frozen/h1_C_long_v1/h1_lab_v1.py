"""H1 lab: the strategy_lab_v1 method on GOLD# H1 bars, where spread is a much smaller share of TP.

Labels are first-touch TP/SL on H1 high/low with entry spread; features are scale-free (ratios to price)
because gold went from ~1,200 to ~4,000 over the sample. Reads MT5 bars only, never trades.

    python -B h1_lab_v1.py --output forward_reports/20261105_h1_lab_v1

Splits (UTC): fit 2014-01-01..2022-12-31 (purged by horizon), threshold pick 2023, test 2024-01-01..end.
MT5 H1 history is really daily before 2013 and has SPREAD=0 before 2018, hence the start and the cost floor.
"""
import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from drl_trading_v2 import add_indicators
from gold_recent_walk_forward import copy_rates
from strategy_lab_v1 import first_touch, run_strategy, stats, trade

START = datetime(2014, 1, 1, tzinfo=timezone.utc)
VALIDATION_START = pd.Timestamp("2023-01-01")
TEST_START = pd.Timestamp("2024-01-01")
# Cost floor: 0.008% of price (~0.36 at 4,500), above the ~0.30 seen live, since older bars report no spread.
MIN_SPREAD_FRACTION = 0.00008
HIGHER = ["H4", "Daily", "Weekly", "Monthly"]
FEATURES = ["RSI", "ATR_PCT", "MACD_PCT", "BB_WIDTH", "BIAS_20", "BODY_PCT", "ROC_5", "ROC_24",
            "VOLA_RATIO", "HOUR_SIN", "HOUR_COS", "DAY_OF_WEEK"] + [f"{tf}_TREND" for tf in HIGHER]
# --cross: other markets' recent H1 returns. VIX starts 2020; XGBoost treats the earlier gap as missing.
# shortcut: USDX/VIX are futures, so a return spanning a contract roll is off; upgrade with a roll-adjusted series.
CROSS = {"USDX-DEC26": "USDX", "SILVER#": "SILVER", "US500Cash#": "US500", "VIX-NOV26": "VIX"}
CROSS_LAGS = (1, 6, 24)
CROSS_FEATURES = [f"{p}_ROC_{n}" for p in CROSS.values() for n in CROSS_LAGS]
RANDOM_DRAWS = 300

# Fixed before any result was seen. tp/sl in H1 ATR, hold in H1 bars.
STRATEGIES = {
    "A_long_2.0_2.0_24h": {"side": "long", "tp": 2.0, "sl": 2.0, "hold": 24},
    "B_short_2.0_2.0_24h": {"side": "short", "tp": 2.0, "sl": 2.0, "hold": 24},
    "C_long_3.0_1.5_48h": {"side": "long", "tp": 3.0, "sl": 1.5, "hold": 48},    # trend-following payoff
    "D_short_3.0_1.5_48h": {"side": "short", "tp": 3.0, "sl": 1.5, "hold": 48},
}


def make_features(h1, higher, cross=None):
    """Row t holds only information from bars closed before t opens; the trade enters at t's open."""
    frame = add_indicators(h1)
    diff = frame["CLOSE"].diff()
    gain = diff.where(diff > 0, 0).rolling(14).mean()
    loss = (-diff.where(diff < 0, 0)).rolling(14).mean()
    frame["RSI"] = 100 - 100 / (1 + gain / (loss + 1e-9))
    frame["ATR_PCT"] = frame["ATR"] / frame["CLOSE"]
    frame["MACD_PCT"] = frame["MACD_HIST"] / frame["CLOSE"]
    frame["ROC_24"] = frame["CLOSE"].pct_change(24)
    frame["VOLA_RATIO"] = frame["ATR"] / (frame["ATR"].rolling(240).mean() + 1e-9)
    for tf, bars in higher.items():
        bars = bars.sort_values("TIME_DT").copy()
        # Trend of the last closed higher bar: shift so the bar still forming is never used.
        bars[f"{tf}_TREND"] = np.where(bars["CLOSE"] > bars["CLOSE"].rolling(20).mean(), 1, -1)
        bars[f"{tf}_TREND"] = bars[f"{tf}_TREND"].shift(1)
        frame = pd.merge_asof(frame.sort_values("TIME_DT"), bars[["TIME_DT", f"{tf}_TREND"]],
                              on="TIME_DT", direction="backward")
    extra = []
    for prefix, bars in (cross or {}).items():
        bars = bars.sort_values("TIME_DT")[["TIME_DT", "CLOSE"]].copy()
        for n in CROSS_LAGS:
            bars[f"{prefix}_ROC_{n}"] = bars["CLOSE"].pct_change(n)
        cols = [f"{prefix}_ROC_{n}" for n in CROSS_LAGS]
        frame = pd.merge_asof(frame.sort_values("TIME_DT"), bars[["TIME_DT"] + cols],
                              on="TIME_DT", direction="backward")
        extra += cols
    # ATR is shifted with the features: TP/SL distances are set from what was known at entry.
    frame[FEATURES + extra + ["ATR"]] = frame[FEATURES + extra + ["ATR"]].shift(1)
    frame["SPREAD"] = np.maximum(frame["SPREAD"], frame["OPEN"] * MIN_SPREAD_FRACTION / 0.01)
    return frame.dropna(subset=FEATURES + ["ATR"]).reset_index(drop=True)


def copy_symbol_h1(symbol, start, end):
    if not mt5.symbol_select(symbol, True):
        raise SystemExit(f"{symbol}: symbol_select failed: {mt5.last_error()}")
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start, end)
    if rates is None or len(rates) == 0:
        raise SystemExit(f"{symbol}: no H1 rates: {mt5.last_error()}")
    frame = pd.DataFrame(rates).rename(columns={"close": "CLOSE"})
    frame["TIME_DT"] = pd.to_datetime(frame["time"], unit="s")
    return frame[["TIME_DT", "CLOSE"]]


def baselines(frame, scores, threshold, spec, test):
    """Same exits without the model: always in the market, and random entries as often as the model signals."""
    r, after = first_touch(frame, spec["side"], spec["tp"], spec["sl"], spec["hold"])
    f, rr, aa, s = frame[test], r[test], after[test], scores[test]
    model_avg = trade(f, s, threshold, rr, aa)["r"].mean()
    signals = np.flatnonzero((s >= threshold) & np.isfinite(rr))
    valid = np.flatnonzero(np.isfinite(rr))
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(RANDOM_DRAWS):
        pick = np.zeros(len(f))
        pick[rng.choice(valid, size=len(signals), replace=False)] = 1
        draws.append(trade(f, pick, 0.5, rr, aa)["r"].mean())
    draws = np.array(draws)
    return {"always_in": stats(trade(f, np.ones(len(f)), 0.5, rr, aa)["r"]),
            "random_avg_r_median": round(float(np.median(draws)), 3),
            "random_share_at_or_above_model": round(float(np.mean(draws >= model_avg)), 3)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--cross", action="store_true", help="add USDX/silver/S&P 500/VIX H1 returns as features")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    if not mt5.initialize(path=os.environ.get("XM_TERMINAL_PATH", r"D:\XM2\terminal64.exe")):
        raise SystemExit(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        h1 = copy_rates("H1", START - timedelta(days=60), end)
        higher = {tf: copy_rates(tf, START - timedelta(days=900), end) for tf in HIGHER}
        cross = ({prefix: copy_symbol_h1(symbol, START - timedelta(days=60), end) for symbol, prefix in CROSS.items()}
                 if args.cross else None)
    finally:
        mt5.shutdown()
    features = FEATURES + (CROSS_FEATURES if args.cross else [])
    frame = make_features(h1, higher, cross)
    frame = frame[frame["TIME_DT"] >= START.replace(tzinfo=None)].reset_index(drop=True)
    print(f"H1 rows: {len(frame):,} ({frame['TIME_DT'].iloc[0]} -> {frame['TIME_DT'].iloc[-1]})", flush=True)

    args.output.mkdir(parents=True)
    report = {"window": [str(START.date()), args.end], "features": features,
              "splits": {"fit_before": str(VALIDATION_START.date()), "test_from": str(TEST_START.date())},
              "min_spread_fraction": MIN_SPREAD_FRACTION, "strategies": {}}
    test = (frame["TIME_DT"] >= TEST_START).to_numpy()
    for name, spec in STRATEGIES.items():
        print(f"=== {name}", flush=True)
        result, model = run_strategy(frame, features, spec, VALIDATION_START, TEST_START)
        model.save_model(args.output / f"{name}_xgb.json")
        scores = model.predict_proba(frame[features].astype(np.float32))[:, 1]
        result["baselines"] = baselines(frame, scores, result["chosen"]["threshold"], spec, test)
        report["strategies"][name] = result
        print(json.dumps({k: result[k] for k in ("chosen", "test", "baselines")}, ensure_ascii=False), flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
