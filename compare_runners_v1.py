"""Compare closed demo trades of gemini.py and h1_runner_v1.py in R (net PnL / initial SL risk), so lot sizes don't matter.

    python -B compare_runners_v1.py [--since 2026-11-08]

Reads MT5 history only. Balance adjustments (SetCustomBalance) have no symbol/magic and are ignored.
"""
import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone

import MetaTrader5 as mt5
import numpy as np

import gemini
import h1_runner_v1 as h1

RUNNERS = {"gemini_M1": gemini.MAGIC_NUMBER, "h1_C": h1.MAGIC_NUMBER}
SYMBOL = "GOLD#"


def trade_r(deals, orders):
    """One R per closed position: (profit + swap + commission + fee) / loss at the opening order's SL."""
    by_position = defaultdict(list)
    for d in deals:
        by_position[d.position_id].append(d)
    sl_by_position = {o.position_id: o.sl for o in orders if o.sl > 0 and o.position_id}
    out = []
    for pos, ds in by_position.items():
        opening = [d for d in ds if d.entry == mt5.DEAL_ENTRY_IN]
        closing = [d for d in ds if d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_OUT_BY)]
        if not opening or not closing or pos not in sl_by_position:
            continue  # still open, or opened before the window
        o = opening[0]
        risk = -mt5.order_calc_profit(o.type, SYMBOL, o.volume, o.price, sl_by_position[pos])
        if not risk or risk <= 0:
            continue
        net = sum(d.profit + d.swap + d.commission + getattr(d, "fee", 0.0) for d in ds)
        out.append({"open_time": datetime.fromtimestamp(o.time, timezone.utc).isoformat(), "r": net / risk, "net": net})
    return out


def stats(trades):
    r = np.array([t["r"] for t in trades])
    if not len(r):
        return {"n": 0}
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    return {"n": len(r), "win_rate": round(float((r > 0).mean()), 3), "avg_r": round(float(r.mean()), 3),
            "profit_factor": round(float(wins / losses), 2) if losses else None,
            "net_usd": round(sum(t["net"] for t in trades), 2)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", default="2026-11-08", help="UTC date; default = H1 runner start")
    args = parser.parse_args(argv)
    since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if not mt5.initialize(path=os.environ.get("XM_TERMINAL_PATH", r"D:\XM2\terminal64.exe")):
        raise SystemExit(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        now = datetime.now(timezone.utc)
        deals = mt5.history_deals_get(since, now) or ()
        orders = mt5.history_orders_get(since, now) or ()
        report = {"since": args.since}
        for name, magic in RUNNERS.items():
            mine = [d for d in deals if d.symbol == SYMBOL and d.magic == magic]
            report[name] = stats(trade_r(mine, [o for o in orders if o.magic == magic]))
    finally:
        mt5.shutdown()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
