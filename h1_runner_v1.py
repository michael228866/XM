"""Demo runner for the frozen H1 candidate (frozen/h1_C_long_v1), run beside gemini.py to compare live results.

Each new H1 bar: score it with the frozen model and code; if score >= frozen threshold and no own position,
buy 0.01 lot with SL 1.5 ATR / TP 3.0 ATR (server-side), close after 48 h. Own magic, so gemini's position,
daily-loss and history queries (all filtered by magic) never see these trades. Demo only.

    set XM_EXPECTED_LOGIN=... & set XM_EXPECTED_SERVER=... & python -u h1_runner_v1.py [--dry-run]
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import numpy as np

import h1_forward_check_v1 as fc
from demo_runtime_foundation_v1 import TradingSafetyError, require_demo_account

SYMBOL = "GOLD#"
MAGIC_NUMBER = 20261108
LOT = 0.01
ENTRY_WINDOW_SECONDS = 600   # act only in the first 10 min of a bar; a late start skips that bar
MAX_SPREAD_POINTS = 100      # lab cost floor is ~36 points; skip only abnormal spikes
POLL_SECONDS = 20
LOG_FILE = fc.ROOT / "h1_runner_signal_log.csv"
LOG_FIELDS = ["time_utc", "bar_time", "score", "threshold", "atr", "spread_points", "action", "detail"]


def log(row):
    new = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **row})
    print(row, flush=True)


def load_frozen():
    config = fc.json.loads((fc.FROZEN / "config.json").read_text(encoding="utf-8"))
    for name, digest in config["sha256"].items():
        path = fc.FROZEN / name if (fc.FROZEN / name).exists() else fc.ROOT / name
        if fc.sha256(path) != digest:
            raise SystemExit(f"{path} changed since freeze; refusing to trade a different candidate")
    sys.path.insert(0, str(fc.FROZEN))  # frozen copies win over the project's current files
    import h1_lab_v1 as lab
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(fc.FROZEN / "model_xgb.json")
    return config, lab, model


def score_forming_bar(config, lab, model):
    """Features of the bar that just opened use only closed bars (make_features shifts by one)."""
    end = datetime.now(timezone.utc) + timedelta(hours=1)
    h1 = lab.copy_rates("H1", end - timedelta(days=60), end)
    higher = {tf: lab.copy_rates(tf, end - timedelta(days=900), end) for tf in lab.HIGHER}
    frame = lab.make_features(h1, higher)
    last = frame.iloc[[-1]]
    score = float(model.predict_proba(last[config["features"]].astype(np.float32))[:, 1][0])
    return last["TIME_DT"].iloc[0], score, float(last["ATR"].iloc[0])


def own_positions():
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None:
        raise TradingSafetyError(f"positions_get failed: {mt5.last_error()}")
    return [p for p in positions if p.magic == MAGIC_NUMBER]


def send(request):
    require_demo_account(mt5, os.environ.get("XM_EXPECTED_LOGIN"), os.environ.get("XM_EXPECTED_SERVER"))
    return mt5.order_send({**request, "symbol": SYMBOL, "magic": MAGIC_NUMBER, "action": mt5.TRADE_ACTION_DEAL,
                           "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC})


def close_expired(hold_hours):
    for p in own_positions():
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is not None and tick.time - p.time >= hold_hours * 3600:  # both server time
            result = send({"volume": p.volume, "type": mt5.ORDER_TYPE_SELL, "position": p.ticket,
                           "price": tick.bid, "comment": "H1C timeout"})
            log({"action": "timeout_close", "detail": f"ticket={p.ticket} {getattr(result, 'retcode', None)}"})


def on_new_bar(config, lab, model, dry_run):
    bar_time, score, atr = score_forming_bar(config, lab, model)
    tick, info = mt5.symbol_info_tick(SYMBOL), mt5.symbol_info(SYMBOL)
    spread = (tick.ask - tick.bid) / info.point
    row = {"bar_time": bar_time, "score": round(score, 4), "threshold": config["threshold"],
           "atr": round(atr, 3), "spread_points": round(spread, 1)}
    late = tick.time - int(bar_time.timestamp())
    if score < config["threshold"]:
        return log({**row, "action": "no_signal"})
    if own_positions():
        return log({**row, "action": "skip", "detail": "position open"})
    if late > ENTRY_WINDOW_SECONDS:
        return log({**row, "action": "skip", "detail": f"{late}s after bar open"})
    if spread > MAX_SPREAD_POINTS:
        return log({**row, "action": "skip", "detail": "spread"})
    spec = config["spec"]
    request = {"volume": LOT, "type": mt5.ORDER_TYPE_BUY, "price": tick.ask, "comment": "H1C",
               "sl": round(tick.ask - spec["sl"] * atr, info.digits), "tp": round(tick.ask + spec["tp"] * atr, info.digits)}
    if dry_run:
        return log({**row, "action": "dry_run_buy", "detail": request})
    result = send(request)
    log({**row, "action": "buy", "detail": f"retcode={getattr(result, 'retcode', None)} "
                                           f"price={getattr(result, 'price', None)} sl={request['sl']} tp={request['tp']}"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="score the current bar once, never trade")
    args = parser.parse_args(argv)
    config, lab, model = load_frozen()
    if not mt5.initialize(path=os.environ.get("XM_TERMINAL_PATH", r"D:\XM2\terminal64.exe")):
        raise SystemExit(f"MT5 initialize failed: {mt5.last_error()}")
    require_demo_account(mt5, os.environ.get("XM_EXPECTED_LOGIN"), os.environ.get("XM_EXPECTED_SERVER"))
    if args.dry_run:
        return on_new_bar(config, lab, model, dry_run=True)
    print(f"H1 runner: magic {MAGIC_NUMBER}, threshold {config['threshold']}, lot {LOT}", flush=True)
    last_bar = None
    while True:
        try:
            close_expired(config["spec"]["hold"])
            rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 1)
            if rates is not None and len(rates) and rates[0]["time"] != last_bar:
                if last_bar is not None:  # first pass only learns the current bar; no mid-bar entry on start
                    on_new_bar(config, lab, model, dry_run=False)
                last_bar = rates[0]["time"]
        except Exception as exc:  # keep running; MT5 hiccups are common, and every action is logged
            log({"action": "error", "detail": repr(exc)})
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
