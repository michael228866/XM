"""E-003 (H-004): does GOLD# drift concentrate in particular intraday sessions, so a flat-at-rollover
position could earn it without paying swap?

Four sessions fixed in advance, in MT5 server hours (server midnight = NY 17:00 = rollover, so the
boundaries move with US DST): ASIA 01-09, LONDON 09-15, NY 15-19 (US data, London PM fix), LATE 19-24.
The 00-01 bar right after rollover (widest spread) is excluded. A session trade enters at the open of its
first H1 bar and exits at the close of its last bar the same server day, paying max(bar spread, 0.00008 x
price) once. Long and short are both candidates: 4 sessions x 2 sides = 8 trials, Sidak-adjusted.
"""
from datetime import datetime, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

import research_eval_v1 as ev
from research_run_v1 import mt5_manifest_data

SESSIONS = {"ASIA": (1, 9), "LONDON": (9, 15), "NY": (15, 19), "LATE": (19, 24)}


def session_returns(h1, spread_floor_frac, point):
    """One row per server day x session: gross return and round-trip cost fraction."""
    hour, day = h1["TIME_DT"].dt.hour, h1["TIME_DT"].dt.normalize()
    rows = []
    for name, (a, b) in SESSIONS.items():
        g = h1[(hour >= a) & (hour < b)].groupby(day[(hour >= a) & (hour < b)])
        first, last = g.first(), g.last()
        cost = np.maximum(first["SPREAD"] * point / first["OPEN"], spread_floor_frac)
        rows.append(pd.DataFrame({"day": first.index, "session": name, "ret": last["CLOSE"] / first["OPEN"] - 1,
                                  "cost": cost.to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def daily_long(d1, swap_long_price):
    """Always-long close-to-close return per D1 bar net of swap (7 charges per week: triple on Thursday's bar)."""
    nights = np.where(d1["TIME_DT"].dt.dayofweek == 3, 3, 1)
    ret = d1["CLOSE"].pct_change() + swap_long_price * nights / d1["CLOSE"].shift(1)
    return pd.DataFrame({"day": d1["TIME_DT"].dt.normalize(), "ret": ret}).dropna()


def side_stats(net, days, trials):
    t = float(net.mean() / net.std(ddof=1) * np.sqrt(len(net))) if len(net) > 2 else float("nan")
    years = net.groupby(days.dt.year).mean()
    return {"days": int(len(net)), "mean_bp": round(float(net.mean()) * 1e4, 3), "t": round(t, 2),
            "sidak_p": round(ev.sidak_p(t, trials), 4), "annual_pct": round(float(net.mean()) * 252 * 100, 2),
            "positive_years": int((years > 0).sum()), "years": int(len(years)),
            "by_year_bp": {int(y): round(float(v) * 1e4, 2) for y, v in years.items()}}


def run(run_dir, spec, trials):
    start = datetime(2017, 12, 1, tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)
    if not mt5.initialize(path=spec.get("terminal", r"D:\XM2\terminal64.exe")):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        info = mt5.symbol_info("GOLD#")
        h1 = pd.DataFrame(mt5.copy_rates_range("GOLD#", mt5.TIMEFRAME_H1, start, end))
        d1 = pd.DataFrame(mt5.copy_rates_range("GOLD#", mt5.TIMEFRAME_D1, start, end))
        manifest_data = mt5_manifest_data(mt5, ["GOLD#"], start, end, len(h1), spec["windows"],
                                          "MT5 history can be revised by the broker")
    finally:
        mt5.shutdown()
    for f in (h1, d1):
        f["TIME_DT"] = pd.to_datetime(f["time"], unit="s")  # server time, as the rest of the repo uses
        f.rename(columns={"open": "OPEN", "close": "CLOSE", "spread": "SPREAD"}, inplace=True)
    h1, d1 = h1.iloc[:-1], d1.iloc[:-1]  # drop bars still forming
    sess = session_returns(h1, spec["spread_floor_frac"], info.point)
    long_d = daily_long(d1, info.swap_long * info.point)  # swap_mode 0: swap is in points

    n_trials = len(SESSIONS) * 2 * trials
    results, passed = {}, []
    labels = ("discovery_2018_2022", "check_2023_on")
    for name in SESSIONS:
        s = sess[sess["session"] == name]
        for side, sign in (("long", 1), ("short", -1)):
            key, results[f"{name}_{side}"] = f"{name}_{side}", {}
            for label, (a, b) in zip(labels, spec["windows"]):
                w = s[(s["day"] >= a) & (s["day"] < b)]
                results[key][label] = side_stats(sign * w["ret"] - w["cost"], w["day"], n_trials)
            disc, chk = results[key][labels[0]], results[key][labels[1]]
            if disc["positive_years"] >= spec["min_positive_years"] and disc["sidak_p"] < 0.05:
                passed.append((key, chk["mean_bp"] > 0))
            print(key, disc["mean_bp"], disc["t"], chk["mean_bp"], flush=True)
    baseline = {label: side_stats(long_d[(long_d["day"] >= a) & (long_d["day"] < b)]["ret"],
                                  long_d[(long_d["day"] >= a) & (long_d["day"] < b)]["day"], 1)
                for label, (a, b) in zip(labels, spec["windows"])}

    confirmed = [k for k, ok in passed if ok]
    outcome = "supported" if confirmed else ("inconclusive" if passed else "refuted")
    summary = (f"sessions passing 2018-2022 (>= {spec['min_positive_years']}/5 positive years, Sidak p < 0.05 over "
               f"{n_trials} trials): {[k for k, _ in passed] or 'none'}; same sign 2023+: {confirmed or 'none'}")
    rows = [f"| {k} | {lab} | {r['days']} | {r['mean_bp']} | {r['t']} | {r['sidak_p']} | {r['annual_pct']} | "
            f"{r['positive_years']}/{r['years']} |" for k, by in results.items() for lab, r in by.items()]
    rows += [f"| ALWAYS_LONG_D1 (net swap) | {lab} | {r['days']} | {r['mean_bp']} | {r['t']} | - | {r['annual_pct']} | "
             f"{r['positive_years']}/{r['years']} |" for lab, r in baseline.items()]
    report = "\n".join([
        "# E-003 session structure (H-004)", "", f"Outcome: **{outcome}**", "", summary, "",
        "Evidence class: development (gold prices 2018+ were inspected before; sessions were not decomposed).", "",
        "Net of spread (session rows) or swap (always-long row). mean in basis points per day.", "",
        "| candidate | window | days | mean bp/day | t | Sidak p | annual % | positive years |",
        "|---|---|---|---|---|---|---|---|", *rows, "",
        f"Sessions (server hours): {SESSIONS}. Swap long per night (points): {info.swap_long}."])
    metrics = {"candidates": results, "always_long": baseline, "passed": [k for k, _ in passed],
               "confirmed": confirmed, "trials": n_trials}
    return {"outcome": outcome, "evidence_class": "development", "summary": summary,
            "failure_cause": "none" if confirmed else "information",
            "lessons": f"session drift that survives spread: {confirmed}" if confirmed else
                       "no intraday session carries a drift that reliably beats its spread",
            "next_questions": "does the session edge hold on fresh data after 2026-11-12 (confirmatory)?" if confirmed else
                              "intraday structure is not an edge; move to second-moment (H-005) or new information (H-003)",
            "metrics": metrics, "report": report, "manifest_data": manifest_data,
            "model_note": "no model; fixed session definitions"}
