"""Strategy lab: train a few pre-registered GOLD# strategies with execution-aligned labels, compare out of sample.

Labels are first-touch TP/SL on M1 high/low with entry spread, the same exit rules the runtime uses,
so the model learns what is actually traded. Features are the current model's 31 (gold_recent_walk_forward).

    python -B strategy_lab_v1.py --output forward_reports/20261105_strategy_lab_v1

Splits (UTC): fit < 2026-04-01 (purged by horizon), threshold pick 2026-04-01..06-01, test 2026-06-01..end.
The test window was already inspected for the incumbent, so results are development evidence, not untouched OOS.
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import xgboost as xgb

from gemini import PRICE_PER_POINT
from gold_recent_walk_forward import build_feature_frame

START = datetime(2025, 1, 1, tzinfo=timezone.utc)
VALIDATION_START = pd.Timestamp("2026-04-01")
TEST_START = pd.Timestamp("2026-06-01")
TOP_SHARES = (0.005, 0.01, 0.02, 0.05)  # threshold = score quantile picked on validation
MIN_VALIDATION_TRADES = 30

# Fixed before any result was seen. tp/sl in M1 ATR, hold in M1 bars.
STRATEGIES = {
    "S1_long_1.3_1.6_90m": {"side": "long", "tp": 1.3, "sl": 1.6, "hold": 90},   # runtime exits, aligned label
    "S2_short_1.3_1.6_90m": {"side": "short", "tp": 1.3, "sl": 1.6, "hold": 90},  # untested mirror side
    "S3_long_1.0_1.0_60m": {"side": "long", "tp": 1.0, "sl": 1.0, "hold": 60},    # shorter, symmetric
    "S4_long_4.0_4.0_480m": {"side": "long", "tp": 4.0, "sl": 4.0, "hold": 480},  # longer swing, spread share small
}


def first_touch(frame, side, tp_atr, sl_atr, hold):
    """Per row: R of a trade entered at this bar's open (ask for long), and bars until exit. Stop first on ties."""
    opens, highs, lows, closes = (frame[c].to_numpy(float) for c in ("OPEN", "HIGH", "LOW", "CLOSE"))
    spread = frame["SPREAD"].to_numpy(float) * PRICE_PER_POINT
    atr = frame["ATR"].to_numpy(float)
    n = len(frame)
    sl_dist, tp_dist = atr * sl_atr, atr * tp_atr
    # Long buys at ask and exits on bid bars; short sells at bid and exits on ask bars (bid + entry spread).
    entry = opens + spread if side == "long" else opens
    r = np.full(n, np.nan)
    exit_after = np.full(n, hold)
    open_ = np.ones(n, dtype=bool)
    for k in range(hold):
        idx = np.arange(n) + k
        valid = idx < n
        j = np.where(valid, idx, n - 1)
        if side == "long":
            hit_sl = open_ & valid & (lows[j] <= entry - sl_dist)
            hit_tp = open_ & valid & ~hit_sl & (highs[j] >= entry + tp_dist)
        else:
            hit_sl = open_ & valid & (highs[j] + spread >= entry + sl_dist)
            hit_tp = open_ & valid & ~hit_sl & (lows[j] + spread <= entry - tp_dist)
        r[hit_sl], r[hit_tp] = -1.0, tp_atr / sl_atr
        exit_after[hit_sl | hit_tp] = k + 1
        open_ &= ~(hit_sl | hit_tp) & valid
    last = np.arange(n) + hold - 1
    timed_out = open_ & (last < n)
    exit_price = closes[np.minimum(last, n - 1)]
    move = exit_price - entry if side == "long" else entry - (exit_price + spread)
    r[timed_out] = (move / sl_dist)[timed_out]
    return r, exit_after  # r stays NaN where the data ends before the trade resolves


def trade(frame, scores, threshold, r, exit_after):
    """One position at a time: take a signal only after the previous trade has exited."""
    taken, busy_until = [], -1
    for i in np.flatnonzero((scores >= threshold) & np.isfinite(r)):
        if i >= busy_until:
            taken.append(i)
            busy_until = i + exit_after[i]
    return frame.iloc[taken].assign(r=r[taken])


def stats(r):
    r = np.asarray(r, float)
    if len(r) == 0:
        return {"n": 0}
    loss = -r[r < 0].sum()
    se = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else float("nan")
    return {"n": len(r), "win_rate": round(float((r > 0).mean()), 3), "avg_r": round(float(r.mean()), 3),
            "profit_factor": round(float(r[r > 0].sum() / loss), 3) if loss else None,
            "avg_r_95ci": [round(float(r.mean() - 1.96 * se), 3), round(float(r.mean() + 1.96 * se), 3)]}


def run_strategy(frame, features, spec, validation_start=VALIDATION_START, test_start=TEST_START):
    r, exit_after = first_touch(frame, spec["side"], spec["tp"], spec["sl"], spec["hold"])
    times = frame["TIME_DT"]
    fit = (times < validation_start).to_numpy() & np.isfinite(r)
    fit[np.flatnonzero(fit)[-spec["hold"]:]] = False  # purge labels that reach into validation
    label = (r > 0).astype(np.int8)
    pos = label[fit].mean()
    model = xgb.XGBClassifier(n_estimators=220, learning_rate=0.05, max_depth=4, min_child_weight=80,
                              subsample=0.85, colsample_bytree=0.85, random_state=42, tree_method="hist",
                              scale_pos_weight=(1 - pos) / max(pos, 1e-9), n_jobs=max(1, (os.cpu_count() or 2) - 1))
    model.fit(frame.loc[fit, features].astype(np.float32), label[fit])
    scores = model.predict_proba(frame[features].astype(np.float32))[:, 1]

    val = ((times >= validation_start) & (times < test_start)).to_numpy()
    test = (times >= test_start).to_numpy()
    picks = []
    for share in TOP_SHARES:
        threshold = float(np.quantile(scores[val], 1 - share))
        picks.append({"top_share": share, "threshold": round(threshold, 4),
                      **stats(trade(frame[val], scores[val], threshold, r[val], exit_after[val])["r"])})
    eligible = [p for p in picks if p["n"] >= MIN_VALIDATION_TRADES]
    best = max(eligible or picks, key=lambda p: p.get("avg_r", -9))

    taken = trade(frame[test], scores[test], best["threshold"], r[test], exit_after[test])
    monthly = {m: stats(g["r"]) for m, g in taken.groupby(taken["TIME_DT"].dt.strftime("%Y-%m"))}
    return {"spec": spec, "fit_rows": int(fit.sum()), "fit_positive_rate": round(float(pos), 3),
            "validation_candidates": picks, "chosen": best,
            "validation_viable": bool(eligible) and best["avg_r"] > 0,
            "test": stats(taken["r"]), "test_monthly": monthly}, model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    if not mt5.initialize(path=os.environ.get("XM_TERMINAL_PATH", r"D:\XM2\terminal64.exe")):
        raise SystemExit(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        frame, features = build_feature_frame(START, end)
    finally:
        mt5.shutdown()

    args.output.mkdir(parents=True)
    report = {"window": [str(START.date()), args.end], "features": features,
              "splits": {"fit_before": str(VALIDATION_START.date()), "test_from": str(TEST_START.date())},
              "strategies": {}}
    for name, spec in STRATEGIES.items():
        print(f"=== {name}", flush=True)
        result, model = run_strategy(frame, features, spec)
        model.save_model(args.output / f"{name}_xgb.json")
        report["strategies"][name] = result
        print(json.dumps({k: result[k] for k in ("chosen", "test")}, ensure_ascii=False), flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
