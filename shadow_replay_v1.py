"""Shadow replay: rerun gemini.py's live rules on MT5 history and compare with actual forward fills.

Read-only: fetches bars from MT5, never sends orders. Imports gemini so features and rules exist once.

    python -B shadow_replay_v1.py --start 2026-10-01 --end 2026-11-01 --output forward_reports/shadow_replay_oct

--start/--end use the MT5 server-time clock, the same labels as bar_time in gemini_signal_log.csv.
"""
import argparse
import json
import os
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import xgboost as xgb

import gemini
from demo_runtime_foundation_v1 import intrabar_evidence

ROOT = Path(__file__).resolve().parent
TIMEFRAMES = {"M1": mt5.TIMEFRAME_M1, **gemini.TIMEFRAME_MAP}
TRADE_COLUMNS = ["bar_time", "buy_prob", "entry_price", "sl_distance", "tp_distance", "spread_points",
                 "exit_time", "exit_reason", "exit_price", "r"]


def bar_duration(name):
    if name == "Monthly":
        return pd.DateOffset(months=1)
    if name == "Weekly":
        return pd.Timedelta(days=7)
    if name == "Daily":
        return pd.Timedelta(days=1)
    return pd.Timedelta(minutes=int(name[1:]) * (60 if name[0] == "H" else 1))


def epoch_seconds(times):
    return (times - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)


class ClosedBars:
    """Stands in for gemini.get_mt5_data: only bars already closed at replay time `now`."""

    def __init__(self, frames):
        self.frames = {
            TIMEFRAMES[name]: (df, epoch_seconds(df["TIME_DT"] + bar_duration(name)).to_numpy())
            for name, df in frames.items()
        }
        self.now = None

    def __call__(self, symbol, timeframe, count, start_pos=1):
        df, close_times = self.frames[timeframe]
        end = int(np.searchsorted(close_times, epoch_seconds(self.now), side="right"))
        if end == 0:
            return None
        return df.iloc[max(0, end - count):end].reset_index(drop=True)


def simulate_exit(bars, entry, sl_distance, tp_distance):
    """Long only on bid bars; when one bar touches both, stop first like intrabar_evidence."""
    def record(time, price, reason):
        return {"exit_time": time, "exit_reason": reason, "exit_price": price,
                "r": (price - entry) / sl_distance}

    deadline = bars["TIME_DT"].iloc[0] + pd.Timedelta(minutes=gemini.MAX_HOLD_MINUTES)
    for bar in bars.itertuples(index=False):
        if bar.TIME_DT >= deadline:
            return record(bar.TIME_DT, bar.OPEN, "timeout")
        hit = intrabar_evidence("long", entry, tp_distance, sl_distance, bar.HIGH, bar.LOW)["observed_order"]
        if hit in ("sl", "Unknown"):
            return record(bar.TIME_DT, min(bar.OPEN, entry - sl_distance), "sl")
        if hit == "tp":
            return record(bar.TIME_DT, max(bar.OPEN, entry + tp_distance), "tp")
    return record(bars["TIME_DT"].iloc[-1], np.nan, "data_end")


def load_frames(start, end):
    fetch_end = end + pd.Timedelta(days=1)  # room for the last trade's exit
    frames = {}
    for name, timeframe in TIMEFRAMES.items():
        step = pd.Timedelta(days=30) if name == "Monthly" else bar_duration(name)
        count = int((fetch_end - start) / step) + 500
        rates = mt5.copy_rates_from(gemini.SYMBOL, timeframe, fetch_end.to_pydatetime(), count)
        if rates is None or len(rates) == 0:
            raise SystemExit(f"{name}: copy_rates_from failed: {mt5.last_error()}")
        df = pd.DataFrame(rates)
        df["TIME_DT"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.columns = [col.upper() for col in df.columns]
        df = df.sort_values("TIME_DT").reset_index(drop=True)
        if (df["TIME_DT"] < start).sum() < (400 if name == "M1" else 80):
            raise SystemExit(f"{name}: not enough history before {start}; raise MT5 'Max bars in chart'")
        frames[name] = df
    return frames


def replay(frames, model, start, end):
    bars = ClosedBars(frames)
    gemini.get_mt5_data = bars
    m1 = frames["M1"]
    times = m1["TIME_DT"]
    scores, trades = [], []
    busy_until = last_loss_exit = None
    for i in np.flatnonzero(((times >= start) & (times < end)).to_numpy()):
        bar_time = times.iloc[i]
        if (i + 1 >= len(m1) or bar_time.hour not in gemini.ALLOWED_ENTRY_HOURS
                or bar_time.dayofweek not in gemini.ALLOWED_ENTRY_WEEKDAYS):
            continue
        decision_time = bar_time + pd.Timedelta(minutes=1)
        bars.now = decision_time
        pack = gemini.get_current_features()
        if pack is None:
            continue
        feat = pack[0]
        buy_prob = float(gemini.normalize_model_probs(model.predict_proba(feat)[0])[1])
        scores.append({"bar_time": bar_time, "replay_buy_prob": buy_prob})
        if busy_until is not None and decision_time < busy_until:
            continue
        if (last_loss_exit is not None and decision_time - last_loss_exit
                < pd.Timedelta(minutes=gemini.LOSS_COOLDOWN_MINUTES)):
            continue
        # Mirrors the entry gates written inline in gemini.live_trading_loop; keep in sync.
        # shortcut: no daily loss guard or lot-minimum skip; matters only on loss streaks or tiny balances.
        rsi = float(feat["M1_RSI"].iloc[0])
        if (rsi < gemini.MIN_ENTRY_RSI
                or any(low <= rsi <= high for low, high in gemini.EXCLUDED_RSI_RANGES)
                or gemini.get_entry_mode(True, False, buy_prob, buy_prob, None) != "base"):
            continue
        atr = float(feat["ATR"].iloc[0])
        sl_distance = max(atr * gemini.SL_ATR_MULT, gemini.MIN_SL_PRICE)
        tp_distance = max(atr * gemini.TP_ATR_MULT, gemini.MIN_TP_PRICE)
        entry_bar = m1.iloc[i + 1]
        spread_points = float(entry_bar["SPREAD"])
        if spread_points > gemini.get_spread_limit_points(tp_distance):
            continue
        entry = entry_bar["OPEN"] + spread_points * gemini.PRICE_PER_POINT  # buy at ask
        trade = simulate_exit(m1.iloc[i + 1:], entry, sl_distance, tp_distance)
        trades.append({"bar_time": bar_time, "buy_prob": buy_prob, "entry_price": entry,
                       "sl_distance": sl_distance, "tp_distance": tp_distance,
                       "spread_points": spread_points, **trade})
        busy_until = trade["exit_time"]
        last_loss_exit = trade["exit_time"] if trade["r"] < 0 else None
    return pd.DataFrame(scores, columns=["bar_time", "replay_buy_prob"]), pd.DataFrame(trades, columns=TRADE_COLUMNS)


def load_live(start, end):
    log = pd.read_csv(ROOT / gemini.SIGNAL_LOG_FILE, dtype={"position_ticket": str},
                      usecols=["bar_time", "status", "message", "buy_prob", "sl_distance", "lot", "position_ticket"])
    log["bar_time"] = pd.to_datetime(log["bar_time"], utc=True, format="ISO8601")
    log = log[(log["bar_time"] >= start) & (log["bar_time"] < end)].drop_duplicates("bar_time", keep="last")
    deals = pd.read_csv(ROOT / gemini.TRADE_HISTORY_LOG_FILE, dtype={"position_id": str})
    closes = deals[deals["entry"] == 1]
    pnl = (closes["profit"] + closes["commission"] + closes["swap"]
           + closes["fee"].fillna(0)).groupby(closes["position_id"]).sum()
    opened = log[log["status"] == "order_opened"].copy()
    opened["actual_pnl"] = opened["position_ticket"].map(pnl)
    # 100 oz per lot, the same contract size gemini's stop_cost assumes.
    opened["actual_r"] = opened["actual_pnl"] / (opened["sl_distance"] * opened["lot"] * 100)
    return log, opened


def stats(r):
    r = r.dropna()
    loss = -r[r < 0].sum()
    return {"n": int(len(r)),
            "win_rate": round(float((r > 0).mean()), 4) if len(r) else None,
            "avg_r": round(float(r.mean()), 4) if len(r) else None,
            "profit_factor": round(float(r[r > 0].sum() / loss), 4) if loss else None}


def compare(scores, trades, log, opened):
    live = log.set_index("bar_time")
    table = trades.merge(
        opened[["bar_time", "buy_prob", "actual_pnl", "actual_r"]].rename(columns={"buy_prob": "live_buy_prob"}),
        on="bar_time", how="outer", indicator=True)
    table["match"] = table["_merge"].astype(str).map(
        {"both": "both", "left_only": "replay_only", "right_only": "live_only"})
    table["live_status"] = table["bar_time"].map(live["status"]).fillna("no_live_row")
    table["live_message"] = table["bar_time"].map(live["message"])
    table = table.drop(columns="_merge").sort_values("bar_time")

    paired = scores.merge(log[["bar_time", "buy_prob"]], on="bar_time")
    diff = (paired["replay_buy_prob"] - paired["buy_prob"]).abs()
    both = table[table["match"] == "both"]
    summary = {
        "replay": stats(trades["r"]),
        "live": stats(opened["actual_r"]),
        "same_trades": {"replay": stats(both["r"]), "live": stats(both["actual_r"])},
        "match_counts": table["match"].value_counts().to_dict(),
        "replay_only_live_status": table.loc[table["match"] == "replay_only", "live_status"].value_counts().to_dict(),
        "score_parity": {"n": int(len(diff)),
                         "mean_abs_diff": round(float(diff.mean()), 6) if len(diff) else None,
                         "max_abs_diff": round(float(diff.max()), 6) if len(diff) else None,
                         "share_over_0.01": round(float((diff > 0.01).mean()), 4) if len(diff) else None},
        "simplifications": ["entry at next M1 open plus that bar's spread field",
                            "stop first when one bar touches TP and SL",
                            "no daily loss guard or lot-minimum skip",
                            "live R = PnL / (sl_distance * lot * 100)"],
    }
    return table, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True, type=Path)
    # What-if exits: override gemini's constants in this process only; gemini.py is untouched.
    parser.add_argument("--tp-atr", type=float, help="override gemini.TP_ATR_MULT")
    parser.add_argument("--sl-atr", type=float, help="override gemini.SL_ATR_MULT")
    parser.add_argument("--max-hold", type=int, help="override gemini.MAX_HOLD_MINUTES")
    parser.add_argument("--threshold", type=float, help="override gemini.CONF_THRESHOLD")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    overrides = {name: value for name, value in [
        ("TP_ATR_MULT", args.tp_atr), ("SL_ATR_MULT", args.sl_atr),
        ("MAX_HOLD_MINUTES", args.max_hold), ("CONF_THRESHOLD", args.threshold)] if value is not None}
    for name, value in overrides.items():
        setattr(gemini, name, value)
    start, end = pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC")

    if not mt5.initialize(path=os.environ.get("XM_TERMINAL_PATH", r"D:\XM2\terminal64.exe")):
        raise SystemExit(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        frames = load_frames(start, end)
    finally:
        mt5.shutdown()

    model = xgb.XGBClassifier()
    model.load_model(ROOT / gemini.MODEL_FILE)
    model.set_params(device="cpu")
    scores, trades = replay(frames, model, start, end)
    log, opened = load_live(start, end)
    table, summary = compare(scores, trades, log, opened)
    summary = {"window": [str(start), str(end)], "model_file": gemini.MODEL_FILE,
               "overrides": overrides, **summary}

    args.output.mkdir(parents=True)
    table.to_csv(args.output / "comparison.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
