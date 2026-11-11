"""E-005 (H-004, question=direction) and E-006 (H-008, question=cost): GOLD# intraday session structure.

Replaces exp_session_structure_v1 (withdrawn before it ran, D-006). v1 mixed two questions and had no control
for market direction. Here they are separate:

- cost (H-008): is a long held 01-24 server time and flat across rollover (no swap, one spread per day) better
  than always-long paying swap? Controls: ALWAYS_LONG (net swap), OVERNIGHT_LONG (prior close -> 01 open, pays
  swap; what intraday-only gives up).
- direction (H-004): does a session earn more than its proportional share of the month's intraday drift?
  excess = side x (session return - hours/23 x month mean intraday return, the day itself left out) - cost. Being long while gold rose
  makes every session look good; the proportional share removes that. A session must also be profitable as
  traded (raw net), or the excess is not usable.

Sessions are server hours (server midnight = NY 17:00 = rollover): ASIA 01-09, LONDON 09-15, NY 15-19, LATE 19-24.
Cost per round trip = max(spread of the segment's first and last bar, 0.00008 x price) x multiplier, so the
wide 01:00 reopen and pre-rollover spreads count; "live31" floors it at today's live median of 31 points.
Swap is a rate, not fixed points: today's swap_long / today's price per night (a fixed point amount would
charge ~3x more in 2018 when gold was ~1300), x nights (Thursday's segment carries Wednesday's triple) x
multiplier. Today's rate still exceeds 2018-2021 rates, so E-006 must also win at swap x0.5.

Known residual bias (critic note): the proportional share uses hours, not each session's share of variance, so
a high-volatility session keeps ~0.3 bp/day of drift exposure in a rising market, far below the ~4 bp/day
detectable effect.
"""
from datetime import datetime, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

import research_eval_v1 as ev
from research_run_v1 import mt5_manifest_data

SESSIONS = {"ASIA": (1, 9), "LONDON": (9, 15), "NY": (15, 19), "LATE": (19, 24)}
INTRADAY = (1, 24)
LIVE_POINTS = 31


def day_table(h1, point, floor_frac):
    """One row per server day: each segment's open->close return, cost fraction and entry price."""
    hour, day = h1["TIME_DT"].dt.hour, h1["TIME_DT"].dt.normalize()
    out = {}
    for name, (a, b) in {**SESSIONS, "INTRADAY": INTRADAY}.items():
        m = (hour >= a) & (hour < b)
        g = h1[m].groupby(day[m])
        first, last = g.first(), g.last()
        out[f"{name}_ret"] = last["CLOSE"] / first["OPEN"] - 1
        out[f"{name}_cost"] = np.maximum(np.maximum(first["SPREAD"], last["SPREAD"]) * point / first["OPEN"], floor_frac)
        out[f"{name}_open"] = first["OPEN"]
    t = pd.DataFrame(out)
    prev = h1.groupby(day)["CLOSE"].last().shift(1)
    t["ON_ret"] = t["INTRADAY_open"] / prev - 1
    t["CC_ret"] = h1.groupby(day)["CLOSE"].last() / prev - 1
    t["prev_close"] = prev
    t["nights"] = np.where(t.index.dayofweek == 3, 3, 1)
    return t.dropna()  # shortcut: drops holiday days missing a session; fine for daily means


def strategies(t, point, swap_frac_per_night, cost_mult=1.0, swap_mult=1.0, live=False):
    """Net daily returns of every candidate and control under one cost/swap scenario."""
    def cost(name):
        c = t[f"{name}_cost"] * cost_mult
        return np.maximum(c, LIVE_POINTS * point / t[f"{name}_open"]) if live else c
    swap = swap_frac_per_night * swap_mult * t["nights"]
    g = t["INTRADAY_ret"].groupby(t.index.to_period("M"))
    month_drift = (g.transform("sum") - t["INTRADAY_ret"]) / (g.transform("count") - 1)  # leave the day itself out
    out = {"ALWAYS_LONG": t["CC_ret"] + swap, "INTRADAY_LONG": t["INTRADAY_ret"] - cost("INTRADAY"),
           "OVERNIGHT_LONG": t["ON_ret"] - cost("INTRADAY") + swap}
    for name, (a, b) in SESSIONS.items():
        share = (b - a) / (INTRADAY[1] - INTRADAY[0])
        for side, sign in (("long", 1), ("short", -1)):
            out[f"{name}_{side}"] = sign * t[f"{name}_ret"] - cost(name)
            out[f"{name}_{side}_excess"] = sign * (t[f"{name}_ret"] - share * month_drift) - cost(name)
    return pd.DataFrame(out, index=t.index)


def scenarios(spec):
    return [(f"cost x{m}", m, 1.0, False) for m in spec["cost_multipliers"]] + \
           [("cost live31", 1.0, 1.0, True)] + [(f"swap x{s}", 1.0, s, False) for s in spec["swap_multipliers"] if s != 1]


def run(run_dir, spec, trials):
    start = datetime(2017, 12, 1, tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)
    if not mt5.initialize(path=spec.get("terminal", r"D:\XM2\terminal64.exe")):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        info = mt5.symbol_info("GOLD#")
        h1 = pd.DataFrame(mt5.copy_rates_range("GOLD#", mt5.TIMEFRAME_H1, start, end))
        manifest_data = mt5_manifest_data(mt5, ["GOLD#"], start, end, len(h1), spec["windows"],
                                          "MT5 history can be revised by the broker")
    finally:
        mt5.shutdown()
    h1["TIME_DT"] = pd.to_datetime(h1["time"], unit="s")  # server time, as the rest of the repo uses
    h1 = h1.rename(columns={"open": "OPEN", "close": "CLOSE", "spread": "SPREAD"}).iloc[:-1]
    t = day_table(h1, info.point, spec["spread_floor_frac"])
    swap_frac = info.swap_long * info.point / h1["CLOSE"].iloc[-1]  # swap_mode 0: points; as a rate of today's price
    labels = ("discovery_2018_2022", "check_2023_on")
    masks = {lab: (t.index >= a) & (t.index < b) for lab, (a, b) in zip(labels, spec["windows"])}

    table = {}  # scenario -> window -> strategy -> stats
    for scen, cm, sm, live in scenarios(spec):
        s = strategies(t, info.point, swap_frac, cm, sm, live)
        table[scen] = {lab: {c: ev.daily_stats(s.loc[m, c], s.index[m]) for c in s} for lab, m in masks.items()}
    base, stress = table["cost x1.0"], table[f"cost x{float(max(spec['cost_multipliers']))}"]
    d, c = labels
    n_trials = spec["trials_declared"] * trials

    if spec["question"] == "cost":
        def better(tab, lab):
            return tab[lab]["INTRADAY_LONG"]["sharpe"] > tab[lab]["ALWAYS_LONG"]["sharpe"]
        hard = [table["cost x1.5"], table["cost live31"]] + [table[f"swap x{s}"] for s in spec["swap_multipliers"] if s < 1]
        if not better(base, d):
            outcome = "refuted"
        elif all(better(tab, lab) for tab in hard for lab in labels):
            outcome = "supported"
        else:
            outcome = "inconclusive"
        summary = (f"Sharpe INTRADAY_LONG vs ALWAYS_LONG (cost x1): 2018-2022 {base[d]['INTRADAY_LONG']['sharpe']} vs "
                   f"{base[d]['ALWAYS_LONG']['sharpe']}, 2023+ {base[c]['INTRADAY_LONG']['sharpe']} vs "
                   f"{base[c]['ALWAYS_LONG']['sharpe']}; better in both windows at cost x1.5, live31 and low swap: "
                   f"{all(better(tab, lab) for tab in hard for lab in labels)}")
        focus = ["ALWAYS_LONG", "INTRADAY_LONG", "OVERNIGHT_LONG"]
    else:
        passed = []
        for name in SESSIONS:
            for side in ("long", "short"):
                k = f"{name}_{side}"
                e = base[d][f"{k}_excess"]
                ok = (e["positive_years"] >= spec["min_positive_years"] and ev.sidak_p(e["t"], n_trials) < 0.05
                      and base[c][f"{k}_excess"]["net_mean_bp"] > 0 and stress[d][f"{k}_excess"]["net_mean_bp"] > 0
                      and base[d][k]["net_mean_bp"] > 0 and base[c][k]["net_mean_bp"] > 0)
                if ok:
                    passed.append(k)
        best_t = max(base[d][f"{n}_{s}_excess"]["t"] for n in SESSIONS for s in ("long", "short"))
        outcome = "supported" if passed else ("refuted" if ev.sidak_p(best_t, n_trials) >= 0.5 else "not_significant")
        summary = (f"sessions with excess over proportional drift passing all rules: {passed or 'none'}; "
                   f"best 2018-2022 excess t {best_t} (Sidak p over {n_trials}: {round(ev.sidak_p(best_t, n_trials), 3)})")
        focus = [f"{n}_{s}{x}" for n in SESSIONS for s in ("long", "short") for x in ("", "_excess")] + ["ALWAYS_LONG"]

    rows = [f"| {k} | {lab} | {r.get('days')} | {r.get('net_mean_bp')} | {r.get('t')} | {r.get('sharpe')} | "
            f"{r.get('calmar')} | {r.get('max_dd_pct')} | {r.get('annual_pct')} | {r.get('positive_years')}/{r.get('years')} |"
            for k in focus for lab, r in ((lab, base[lab][k]) for lab in labels)]
    sens = [f"| {k} | {scen} | " + " | ".join(f"{table[scen][lab][k].get('net_mean_bp')} / {table[scen][lab][k].get('sharpe')}"
                                            for lab in labels) + " |"
            for k in focus for scen in table]
    ledger_outcome = "inconclusive" if outcome == "not_significant" else outcome
    report = "\n".join([
        f"# {'E-006 swap avoidance (H-008)' if spec['question'] == 'cost' else 'E-005 session direction (H-004)'}", "",
        f"Outcome: **{outcome}**", "", summary, "",
        "Evidence class: development (gold prices 2018+ were inspected before; sessions were not decomposed).", "",
        "## Base scenario (cost x1, swap x1)", "",
        "| strategy | window | days | net bp/day | t | Sharpe | Calmar | max DD % | annual % | positive years |",
        "|---|---|---|---|---|---|---|---|---|---|", *rows, "",
        "## Cost and swap sensitivity (net bp/day / Sharpe)", "", f"| strategy | scenario | {labels[0]} | {labels[1]} |",
        "|---|---|---|---|", *sens, "",
        f"Sessions (server hours): {SESSIONS}; trials {n_trials}; swap long per night (points) {info.swap_long} = "
        f"{swap_frac * 365 * 100:.2f}%/yr of price, applied as a rate in every year."])
    return {"outcome": ledger_outcome, "evidence_class": "development", "summary": f"{outcome}: {summary}",
            "failure_cause": "none" if outcome == "supported" else ("cost" if spec["question"] == "cost" else "information"),
            "lessons": f"{spec['question']} question: {outcome}; see report for cost sensitivity",
            "next_questions": "confirm on data after 2026-11-12 before any live use" if outcome == "supported" else
                              "session structure carries no usable edge at these costs; prefer H-003/H-009+ questions",
            "metrics": {"scenarios": table, "outcome_detail": outcome, "trials": n_trials, "mean_r": "n/a"},
            "report": report, "manifest_data": manifest_data, "model_note": "no model; fixed session definitions"}
