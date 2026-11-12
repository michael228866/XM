"""GVZ (CBOE Gold ETF Volatility Index, FRED series GVZCLS) for H-014: snapshot, revision check, timing, audit.

The snapshot is committed and its sha256 is listed in the preregistration's code_files, so the runner refuses
to run if the data change. MT5 is only needed for the GOLD# calendar check (`audit --gold`).

Timing (pre-declared, conservative; FRED/ALFRED cannot show when a historical daily value was first published):
- GVZCLS for US date d is the Cboe close, computed from GLD options that stop trading at 16:00 New York
  (20:00 UTC in US summer time, 21:00 UTC in winter); the index close is set by about 16:15.
- A forecast origin is the end of GOLD# server day t (server midnight = 21:00 UTC summer / 22:00 UTC winter,
  since the server is UTC+3/UTC+2 following US daylight saving).
- Rule: at origin t use the latest GVZ with d <= t - 1 calendar day (lag 1). The same-day value is never used,
  even though the Cboe close precedes server midnight, so publication delay up to ~23 hours is covered
  regardless of daylight-saving mismatches. Sensitivity "lag 2": one more GVZ observation back.
- A GVZ value older than MAX_STALE_DAYS calendar days at origin is treated as missing (day dropped for all models).

    python -B research_data_gvz_v1.py fetch          # download snapshot + ALFRED revision check
    python -B research_data_gvz_v1.py audit          # GVZ-only quality report
    python -B research_data_gvz_v1.py audit --gold   # plus alignment against GOLD# complete server days (MT5)
"""
import argparse
import hashlib
import io
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "research_data" / "gvz" / "GVZCLS.csv"
PROVENANCE = DATA.with_name("provenance.json")
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GVZCLS"
ALFRED = "https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=GVZCLS&vintage_date={}"
VINTAGES = ("2020-03-17", "2022-06-15", "2024-08-07")
MAX_STALE_DAYS = 5


def parse(raw):
    """FRED CSV bytes -> float Series indexed by US observation date; '.' (no value) kept as NaN."""
    s = pd.read_csv(io.BytesIO(raw), na_values=".", index_col=0, parse_dates=True).iloc[:, 0]
    return s.astype(float).rename("gvz")


def load(path=DATA):
    return parse(Path(path).read_bytes())


def align(gvz, days, extra_lag=0):
    """Per server day t in `days`: latest GVZ with US date <= t - 1 day (then `extra_lag` observations further back).
    Returns gvz, gvz_date, stale_days; gvz is NaN when stale > MAX_STALE_DAYS."""
    g = gvz.dropna()
    g = pd.DataFrame({"gvz": g.shift(extra_lag), "gvz_date": g.index.to_series().shift(extra_lag)}, index=g.index)
    g = g.dropna()
    left = pd.DataFrame({"cutoff": pd.DatetimeIndex(days) - pd.Timedelta(days=1)}, index=pd.DatetimeIndex(days))
    m = pd.merge_asof(left.reset_index().rename(columns={"index": "day"}).sort_values("cutoff"),
                      g.reset_index(drop=True).assign(key=g.index).sort_values("key"),
                      left_on="cutoff", right_on="key", direction="backward").set_index("day").sort_index()
    stale = (m.index - m["gvz_date"]).dt.days
    out = pd.DataFrame({"gvz": m["gvz"].where(stale <= MAX_STALE_DAYS), "gvz_date": m["gvz_date"],
                        "stale_days": stale}, index=pd.DatetimeIndex(days))
    assert (out["gvz_date"].dropna() < out.index[out["gvz_date"].notna()]).all()  # never same-day or future
    return out


def audit_gvz(gvz):
    v = gvz.dropna()
    bdays = pd.bdate_range(v.index[0], v.index[-1])
    gaps = v.index.to_series().diff().dt.days
    jumps = np.log(v).diff().abs()
    return {"rows": int(len(gvz)), "values": int(len(v)), "first": str(v.index[0].date()), "last": str(v.index[-1].date()),
            "no_value_rows": [str(d.date()) for d in gvz.index[gvz.isna()]][:60],
            "no_value_count": int(gvz.isna().sum()),
            "weekdays_without_row": int(len(bdays.difference(gvz.index))),
            "weekend_rows": int((gvz.index.dayofweek >= 5).sum()), "duplicates": int(gvz.index.duplicated().sum()),
            "gaps_over_4_days": {str(d.date()): int(n) for d, n in gaps[gaps > 4].items()},
            "min": float(v.min()), "max": float(v.max()), "median": float(v.median()),
            "abs_log_change_over_0.4": {str(d.date()): round(float(x), 3) for d, x in jumps[jumps > 0.4].items()}}


def alignment_audit(gvz, days):
    """Calendar-only check against GOLD# complete server days; reads no prices."""
    a, a2 = align(gvz, days), align(gvz, days, extra_lag=1)
    usable = a["gvz"].notna()
    by_year = usable.groupby(a.index.year).agg(["sum", "size"])
    return {"server_days": int(len(a)), "usable_lag1": int(usable.sum()), "usable_lag2": int(a2["gvz"].notna().sum()),
            "stale_days_distribution_lag1": {int(k): int(n) for k, n in a["stale_days"].value_counts().sort_index().items()},
            "stale_by_weekday_lag1": {int(k): float(v) for k, v in a["stale_days"].groupby(a.index.dayofweek).mean().round(2).items()},
            "dropped_days": [str(d.date()) for d in a.index[~usable]][:60],
            "usable_by_year": {int(y): f"{int(r['sum'])}/{int(r['size'])}" for y, r in by_year.iterrows()},
            "same_day_or_future_used": 0}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch():
    # FRED stalls urllib's default client; curl (bundled with Windows) downloads normally
    get = lambda url: subprocess.run(["curl", "-sSf", "-m", "90", url], check=True, capture_output=True).stdout
    raw = get(FRED)
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_bytes(raw)
    cur = parse(raw)
    revisions = {}
    for v in VINTAGES:
        old = parse(get(ALFRED.format(v)))
        j = pd.concat([old, cur], axis=1, join="inner").dropna()
        revisions[v] = {"rows": int(len(old)), "changed_values": int((j.iloc[:, 0] - j.iloc[:, 1]).abs().gt(1e-9).sum())}
    prov = {"source": FRED, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sha256": sha256(DATA), "rows": int(len(cur)), "last_observation": str(cur.dropna().index[-1].date()),
            "alfred_revision_check": revisions,
            "publication_time": "not verifiable historically (ALFRED vintages of this daily series include the vintage "
                                "date's own value); handled by the lag-1 rule in this module's docstring"}
    PROVENANCE.write_text(json.dumps(prov, indent=1), encoding="utf-8")
    print(json.dumps(prov, indent=1))


def gold_days():
    from datetime import datetime as dt
    import MetaTrader5 as mt5
    from exp_vol_forecast_v1 import daily_rv
    if not mt5.initialize(path=r"D:\XM2\terminal64.exe"):
        raise RuntimeError(mt5.last_error())
    try:
        h1 = pd.DataFrame(mt5.copy_rates_range("GOLD#", mt5.TIMEFRAME_H1, dt(2013, 10, 1, tzinfo=timezone.utc),
                                               dt.now(timezone.utc)))
    finally:
        mt5.shutdown()
    h1["TIME_DT"] = pd.to_datetime(h1["time"], unit="s")
    h1 = h1.rename(columns={"high": "HIGH", "low": "LOW", "close": "CLOSE"}).iloc[:-1]
    h1 = h1[h1["TIME_DT"].dt.normalize() < h1["TIME_DT"].dt.normalize().max()]
    return daily_rv(h1).index  # calendar only; the RV column is discarded here


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["fetch", "audit"])
    parser.add_argument("--gold", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "fetch":
        return fetch()
    gvz = load()
    out = {"sha256": sha256(DATA), "gvz": audit_gvz(gvz)}
    if args.gold:
        days = gold_days()
        out["alignment"] = alignment_audit(gvz, days[days >= "2014-01-01"])
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
