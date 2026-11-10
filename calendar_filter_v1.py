"""Do trades near high-impact USD releases do worse? Splits an existing replay's trades by event proximity.

Needs usd_calendar.csv from the MQL5 script ExportUsdCalendar (MT5 Common\\Files). Calendar and replay
bar_time are both MT5 server time, so no timezone conversion is done.

    python -B calendar_filter_v1.py --trades forward_reports/20261103_shadow_replay_v1_thr065_jun_nov/comparison.csv
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from strategy_lab_v1 import stats

CALENDAR = Path(os.environ["APPDATA"]) / "MetaQuotes" / "Terminal" / "Common" / "Files" / "usd_calendar.csv"
HIGH_IMPORTANCE = 3  # ENUM_CALENDAR_EVENT_IMPORTANCE: CALENDAR_IMPORTANCE_HIGH
BEFORE_MIN, AFTER_MIN = 30, 30  # blackout: entries from 30 min before an event until 30 min after it


def load_events(path=CALENDAR):
    cal = pd.read_csv(path, encoding="mbcs", encoding_errors="replace")  # MQL5 FILE_ANSI = Windows code page
    cal = cal[cal["importance"] == HIGH_IMPORTANCE]
    return np.sort(pd.to_datetime(cal["server_time"], unit="s").unique())


def near_event(times, events, before=BEFORE_MIN, after=AFTER_MIN):
    """True where an event falls in [t - after, t + before]: the entry is inside some event's blackout."""
    t = pd.to_datetime(pd.Series(times)).dt.tz_localize(None).to_numpy("datetime64[ns]")
    lo = np.searchsorted(events, t - np.timedelta64(after, "m"), side="left")
    hi = np.searchsorted(events, t + np.timedelta64(before, "m"), side="right")
    return hi > lo


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trades", required=True, type=Path, help="CSV with bar_time and r columns")
    parser.add_argument("--calendar", type=Path, default=CALENDAR)
    args = parser.parse_args(argv)
    trades = pd.read_csv(args.trades)
    trades = trades[trades["r"].notna()]
    events = load_events(args.calendar)
    near = near_event(trades["bar_time"].str.slice(0, 19), events)
    print(json.dumps({"high_impact_events": len(events), "all": stats(trades["r"]),
                      "near_event": stats(trades.loc[near, "r"]), "outside_blackout": stats(trades.loc[~near, "r"])},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
