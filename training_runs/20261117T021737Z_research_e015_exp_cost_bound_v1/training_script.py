"""E-015 (H-023): does a trailing estimate of the quoted spread give a conservative per-trade cost upper bound that keeps
its coverage out of sample?

Target: the quoted spread a market order meets, in points (1 point = 0.01 USD/oz; gemini.py uses (ask - bid) / point,
the same unit). Entry target = spread of the first tick quote inside each minute (what a bar-open order sees, latency
aside); exit target = the first quote at or after 60 minutes later (the H-013 horizon), also across the daily break
(the reopen fill), but not across a weekend. On bid-priced bars a round trip pays ONE spread (long: at entry, short:
at exit); charging the spread at both entry and exit double counts. Slippage, commission, fee and swap are separate
components, measured from demo deals only (not live).

Candidates for day d use only data before d (bars and tick days in the previous adaptive_days):
  A_fixed  trailing mean of the M1 SPREAD field x 1.21 (the D-020 rule; 1.21 came from E-014, which saw 2025-2026)
  A_cal    the same mean x k_cal, k_cal = first-quote mean / bar mean on the calibration window only
  A_x1.5, A_x2  A_fixed x 1.5 / x 2 (stress multipliers; tested whether they reach the coverage, not assumed)
           (A_* rules use a constant factor, so they pass only if the factor is stable)
  B95      trailing 95th percentile of the first-quote spread (all hours)
  C95      B95 computed separately for rollover hours (server 23/0/1) and other hours
  D95      level-adaptive C95: trailing 95th percentile of spread / (that day's level), times the current level
           (level = median first-quote spread of the previous 5 sampled tick days)
  exit: B95, C95x / D95x (rollover keyed on the exit time) and E95x (the last quote before the entry minute x
        trailing 95th percentile of exit spread / that quote)
Also: whether the 1.21 factor is stable by year, session and spread level; whether a fixed server-hour rollover flag
holds across DST (spike hour per month). Revised after critic_agent_7 (B1-B5, N2, N5-N7, N9).
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from exp_spread_cost_v1 import ROLLOVER_HOURS, SESSIONS, ci, execution, frame

ENTRY = ["A_fixed", "A_cal", "A_x1.5", "A_x2", "B95", "C95", "D95"]


def session(hours):
    out = np.full(len(hours), "other", dtype=object)
    for name, hs in SESSIONS.items():
        out[np.isin(hours, hs)] = name
    return out


def in_spike_window(times):
    mm = times.hour * 60 + times.minute
    return np.asarray((mm >= 23 * 60 + 55) | (mm <= 65))  # 23:55-01:05 server time, the real rollover spike


def minute_table(ticks, point, horizon_min=60, day_end=None, max_exit_gap_min=180):
    """ticks: time_msc, bid, ask (server-time ms) covering a day plus the next morning. One row per minute with a
    valid quote (before day_end if given): first / last spread (points), tick mean, tick count; quote_before = the last
    quote before the minute (any earlier minute); y_exit = the first quote at or after +horizon, also across the daily
    break (the reopen fill), unless that quote is more than max_exit_gap_min later (weekend) -> no target. The exit's
    rollover flag and spike window are keyed on the actual exit time (critic_agent_7 B2, B3, N9)."""
    tk = ticks[(ticks["bid"] > 0) & (ticks["ask"] > ticks["bid"])]  # a zero or crossed spread is missing, not free
    d = pd.DataFrame({"m": (tk["time_msc"].to_numpy() // 60000) * 60, "sp": np.rint((tk["ask"] - tk["bid"]) / point)})
    g = d.groupby("m", sort=True)["sp"]
    out = pd.DataFrame({"first": g.first(), "last": g.last(), "tmean": g.mean(), "n": g.size()}).reset_index()
    m = out["m"].to_numpy()
    out["quote_before"] = out["last"].shift(1)
    target = m + horizon_min * 60
    jj = np.minimum(np.searchsorted(m, target), len(m) - 1)
    ok = (m[jj] >= target) & (m[jj] - target <= max_exit_gap_min * 60)
    out["y_exit"] = np.where(ok, out["first"].to_numpy()[jj], np.nan)
    exit_time = pd.DatetimeIndex(pd.to_datetime(np.where(ok, m[jj], target), unit="s"))
    out["roll_exit"] = np.asarray(exit_time.hour.isin(ROLLOVER_HOURS))
    out["spike_exit"] = in_spike_window(exit_time)
    out["exit_at"] = pd.Series(exit_time).where(ok).to_numpy()
    out["TIME_DT"] = pd.to_datetime(out["m"], unit="s")
    out["day"] = out["TIME_DT"].dt.normalize()
    out["exit_crosses_day"] = ok & (np.asarray(exit_time.normalize()) != out["day"].to_numpy())
    out["hour"] = out["TIME_DT"].dt.hour
    out["roll"] = out["hour"].isin(ROLLOVER_HOURS)
    out["hour_start"] = out["TIME_DT"].dt.minute == 0
    out["spike"] = in_spike_window(pd.DatetimeIndex(out["TIME_DT"]))
    if day_end is not None:
        out = out[out["TIME_DT"] < pd.Timestamp(day_end)].reset_index(drop=True)
    return out


def day_levels(mt, k=5):
    """Level known at the start of each sampled day: median first-quote spread over the previous k sampled days."""
    daily = mt.groupby("day")["first"].apply(list)
    days, out = list(daily.index), {}
    for i, d in enumerate(days):
        if i >= k:
            out[d] = float(np.median(np.concatenate([daily[x] for x in days[i - k:i]])))
    return pd.Series(out, dtype=float)


def bounds(mt, bars, spec):
    """Per sampled day d: every candidate's parameters from data strictly before d (bars and tick days in the
    previous adaptive_days). Days with fewer than min_trailing_tick_days tick days in the window get no bound."""
    q, span = spec["quantile"], pd.Timedelta(days=spec["adaptive_days"])
    lv = day_levels(mt, spec["level_days"])
    lv_non = day_levels(mt[~mt["roll"]], spec["level_days"])
    mt = mt.assign(L=mt["day"].map(lv))
    mt = mt.assign(ratio=mt["first"] / mt["L"], ratio_e=mt["y_exit"] / mt["quote_before"])
    good = bars[bars["SPREAD"] > 0]
    rows = {}
    for d in sorted(mt["day"].unique()):
        w = mt[(mt["day"] >= d - span) & (mt["day"] < d)]
        if w["day"].nunique() < spec["min_trailing_tick_days"] or d not in lv.index:
            continue
        b = good[(good["TIME_DT"] >= d - span) & (good["TIME_DT"] < d)]["SPREAD"]
        qq = lambda s: float(s.dropna().quantile(q)) if s.notna().sum() >= spec["min_cell_minutes"] else np.nan
        rows[d] = {"A_mean": float(b.mean()), "B95": qq(w["first"]), "B50": float(w["first"].median()),
                   "C95_roll": qq(w.loc[w["roll"], "first"]), "C95_non": qq(w.loc[~w["roll"], "first"]),
                   "L": lv[d], "L_non": float(lv_non.get(d, np.nan)),
                   "D_roll": qq(w.loc[w["roll"], "ratio"]), "D_non": qq(w.loc[~w["roll"], "ratio"]),
                   # an earlier day's exit may land on day d's reopen: only exits completed before d (critic_agent_7 N1)
                   "E_roll": qq(w.loc[w["roll_exit"] & (w["exit_at"] < d), "ratio_e"]),
                   "E_non": qq(w.loc[~w["roll_exit"] & (w["exit_at"] < d), "ratio_e"])}
    return pd.DataFrame(rows).T.astype(float)


def with_bounds(mt, B, k_cal, spec):
    f = mt[mt["day"].isin(B.index)].copy()
    p = B.reindex(f["day"]).reset_index(drop=True)
    f = f.reset_index(drop=True)
    a = p["A_mean"] * spec["a_factor_d020"]
    f["A_fixed"], f["A_cal"], f["A_x1.5"], f["A_x2"] = a, p["A_mean"] * k_cal, a * 1.5, a * 2.0
    f["B95"], f["B50"] = p["B95"], p["B50"]
    f["C95"] = np.where(f["roll"], p["C95_roll"], p["C95_non"])
    f["D95"] = p["L"].to_numpy() * np.where(f["roll"], p["D_roll"], p["D_non"])
    f["C95x"] = np.where(f["roll_exit"], p["C95_roll"], p["C95_non"])  # exit versions keyed on the exit time
    f["D95x"] = p["L"].to_numpy() * np.where(f["roll_exit"], p["D_roll"], p["D_non"])
    f["E95x"] = f["quote_before"] * np.where(f["roll_exit"], p["E_roll"], p["E_non"])
    f["Lday"], f["Lnon"] = p["L"], p["L_non"]
    f["month"] = f["TIME_DT"].dt.to_period("M")
    f["year"] = f["TIME_DT"].dt.year
    return f


def block_ci(cov, month, blocks, spec, level, rng):
    key = np.array([(m.year * 12 + m.month - 1) // blocks for m in month])
    sums = pd.DataFrame({"k": key, "c": cov, "n": 1}).groupby("k")[["c", "n"]].sum().to_numpy(float)
    draws = np.array([s[0] / s[1] for s in (sums[rng.integers(0, len(sums), len(sums))].sum(axis=0)
                                            for _ in range(spec["bootstrap_draws"]))])
    return ci(draws, level), len(sums)


def score(f, col, target, spec, level, rng):
    """Coverage of `target` by bound `col`. For the exit target, rollover, spike window and high-cost events are keyed
    on the exit time; the hour-start sub-check is on the decision minute (H-013 decides at HH:00, critic_agent_7 B5)."""
    exit_ = target == "y_exit"
    g = f[f[col].notna() & f[target].notna()]
    if not len(g):
        return {"n": 0}
    y, b = g[target].to_numpy(), g[col].to_numpy()
    roll = g["roll_exit" if exit_ else "roll"].to_numpy(bool)
    spike = g["spike_exit" if exit_ else "spike"].to_numpy(bool)
    cov = y <= b
    month = list(g["month"])
    ci1, months = block_ci(cov, month, 1, spec, level, rng)
    # high-cost events: non-rollover minutes above 2x the non-rollover level known that morning (critic_agent_7 B1)
    high = ~roll & (y > spec["high_cost_multiple"] * g["Lnon"].to_numpy())
    high_roll = roll & (y > spec["high_cost_multiple"] * g["Lday"].to_numpy())
    rate = lambda mask: round(float(cov[mask].mean()), 4) if mask.any() else None
    miss = lambda mask: round(float((~cov[mask]).mean()), 4) if mask.any() else None
    by = lambda key: {str(k): round(float(c.mean()), 4) for k, c in pd.Series(cov).groupby(np.asarray(key))
                      if len(c) >= spec["min_group_minutes"]}
    monthly = pd.Series(cov).groupby(np.asarray([str(m) for m in month])).mean()
    quarterly = pd.Series(cov).groupby(np.asarray([str(m.asfreq("Q")) for m in month])).mean()
    hs = g["hour_start"].to_numpy(bool)
    return {"n": int(len(g)), "months": int(months), "coverage": round(float(cov.mean()), 4), "coverage_ci": ci1,
            "coverage_ci_3month_blocks": block_ci(cov, month, 3, spec, level, rng)[0],
            "coverage_by_year": by(g["year"].to_numpy()),
            "coverage_rollover": rate(roll), "coverage_other": rate(~roll),
            "coverage_hour_start": rate(hs), "hour_start_minutes": int(hs.sum()),
            "coverage_spike_window_2355_0105": rate(spike),
            "coverage_exit_crossing_break": rate(g["exit_crosses_day"].to_numpy(bool)) if exit_ else None,
            "coverage_by_session": by(session(g["hour"].to_numpy())),
            "worst_month": [monthly.idxmin(), round(float(monthly.min()), 4)],
            "worst_quarter": [quarterly.idxmin(), round(float(quarterly.min()), 4)],
            "uncovered_cost_share": round(float(np.maximum(y - b, 0).sum() / y.sum()), 4),
            "bound_over_cost": round(float(b.mean() / y.mean()), 3),
            "high_cost_events_non_rollover": int(high.sum()), "high_cost_miss_rate": miss(high),
            "high_cost_events_rollover": int(high_roll.sum()), "high_cost_miss_rate_rollover": miss(high_roll),
            "mean_bound": round(float(b.mean()), 2), "mean_cost": round(float(y.mean()), 2),
            "mae": round(float(np.abs(y - b).mean()), 2), "bias_share": round(float((b.mean() - y.mean()) / y.mean()), 4)}


def passes(s, spec, factor_stable=True, candidate=""):
    c = spec["criteria"]
    if not s.get("n"):
        return {"pass": False, "failed": ["no data"]}
    ge = lambda v, lo: v is not None and v >= lo
    checks = {"coverage": s["coverage"] >= c["coverage_min"],
              "coverage_ci_lower": s["coverage_ci"][0] >= c["coverage_ci_lower_min"],
              "every_year": all(v >= c["year_coverage_min"] for v in s["coverage_by_year"].values()),
              "rollover": ge(s["coverage_rollover"], c["rollover_coverage_min"]),
              "other_hours": ge(s["coverage_other"], c["other_coverage_min"]),
              "hour_start": ge(s["coverage_hour_start"], c["hour_start_coverage_min"]),
              "uncovered_cost_share": s["uncovered_cost_share"] <= c["uncovered_cost_share_max"],
              "not_too_loose": s["bound_over_cost"] <= c["bound_over_cost_max"]}
    if candidate.startswith("A_"):  # a constant tick/bar factor is usable only if it is stable (critic_agent_7 N2)
        checks["factor_stable"] = bool(factor_stable)
    failed = [k for k, ok in checks.items() if not ok]
    # A nominal-95% bound cannot cover most > 2x events under any continuous spread distribution, so spike coverage is
    # a binding label, not a pass gate: spikes_covered False obliges H-013 to gate entries on the live quote; their
    # cost is gated by uncovered_cost_share.
    spikes = s["high_cost_miss_rate"] is not None and s["high_cost_miss_rate"] <= c["high_cost_miss_max"]
    return {"pass": not failed, "failed": failed, "spikes_covered": bool(spikes)}


def factor_stability(mt, bars, spec):
    """tick-mean / bar-mean (the 1.21 of E-014) and first-quote / bar-mean, on matching minutes, by window, year,
    session and calibration-tercile spread level. Stable if forward years are within year_tol and forward sessions
    and levels within group_tol of the calibration value."""
    bar = bars.set_index(bars["TIME_DT"].astype("int64") // 10**9)["SPREAD"]
    j = mt.join(bar.rename("bar"), on="m", how="inner")
    j = j[j["bar"] > 0].copy()
    j["year"], j["session"] = j["TIME_DT"].dt.year, session(j["hour"].to_numpy())
    cal = j[(j["TIME_DT"] >= pd.Timestamp(spec["calibration"][0])) & (j["TIME_DT"] < pd.Timestamp(spec["calibration"][1]))]
    cuts = cal["bar"].quantile([1 / 3, 2 / 3]).to_numpy()
    j["level"] = np.select([j["bar"] <= cuts[0], j["bar"] <= cuts[1]], ["low", "mid"], "high")
    fwd = j[j["TIME_DT"] >= pd.Timestamp(spec["forward"][0])]
    r = lambda g, col="tmean": round(float(g[col].mean() / g["bar"].mean()), 4) if len(g) else None
    base = r(cal)
    groups = lambda g, key, col="tmean": {str(k): r(x, col) for k, x in g.groupby(key) if len(x) >= spec["min_group_minutes"]}
    years, sess, lev = groups(fwd, "year"), groups(fwd, "session"), groups(fwd, "level")
    tol_y, tol_g = spec["factor_year_tol"], spec["factor_group_tol"]
    stable = (base is not None and all(abs(v / base - 1) <= tol_y for v in years.values())
              and all(abs(v / base - 1) <= tol_g for v in list(sess.values()) + list(lev.values())))
    return {"calibration_tick_mean_over_bar": base, "forward_tick_mean_over_bar": r(fwd),
            "calibration_first_over_bar": r(cal, "first"), "forward_first_over_bar": r(fwd, "first"),
            "by_year_all": groups(j, "year"), "forward_by_year": years, "forward_by_session": sess,
            "forward_by_level_tercile": lev, "level_cuts_points": cuts.tolist(),
            "level_note": "terciles cut on 2023-24 bar spreads; forward levels drift up, so low/mid groups can be small",
            "first_over_bar_by_year": groups(j, "year", "first"), "stable": bool(stable),
            "k_cal_first_over_bar": r(cal, "first"), "minutes_matched": int(len(j))}


def rollover_flag_check(bars, spec):
    """Spike hour (highest median bar SPREAD, server time) per month: is it inside the fixed flag 23/0/1 across DST?"""
    b = bars[(bars["SPREAD"] > 0) & (bars["TIME_DT"] >= pd.Timestamp(spec["rollover_check_start"]))]
    med = b.groupby([b["TIME_DT"].dt.to_period("M"), b["TIME_DT"].dt.hour])["SPREAD"].median()
    spike = med.groupby(level=0).idxmax().map(lambda x: x[1])
    inside = spike.isin(ROLLOVER_HOURS)
    fwd = spike.index >= pd.Period(spec["forward"][0][:7], "M")
    return {"months": int(len(spike)), "share_spike_in_flag": round(float(inside.mean()), 4),
            "forward_share_spike_in_flag": round(float(inside[fwd].mean()), 4) if fwd.any() else None,
            "months_outside": {str(k): int(v) for k, v in spike[~inside].items()},
            "spike_hour_counts": {str(k): int(v) for k, v in spike.value_counts().items()},
            "flag_holds": bool(inside.mean() >= spec["rollover_flag_share_min"])}


def round_trips(deals, contract_size):
    """deals: position_id, entry (0 in / 1 out), type (0 buy / 1 sell), price, volume, profit, commission, fee, swap.
    Checks that deal profit is the fill-price difference (spread already inside the fill prices, so a separate spread
    charge would count it twice) and sums the other cost components. Assumes a USD account."""
    if not len(deals):
        return {"round_trips": 0}
    rows = []
    for _, g in deals.groupby("position_id"):
        i, o = g[g["entry"] == 0], g[g["entry"] == 1]
        if len(i) != 1 or not len(o):
            continue
        sign = 1 if i["type"].iloc[0] == 0 else -1
        expected = float(((o["price"] - i["price"].iloc[0]) * o["volume"]).sum() * sign * contract_size)
        rows.append({"expected": expected, "profit": float(o["profit"].sum()),
                     "commission": float(g["commission"].sum()), "fee": float(g["fee"].sum()),
                     "swap": float(g["swap"].sum()), "lots": float(i["volume"].iloc[0])})
    t = pd.DataFrame(rows)
    if not len(t):
        return {"round_trips": 0}
    match = (t["profit"] - t["expected"]).abs() <= np.maximum(0.011, 0.001 * t["expected"].abs())
    return {"round_trips": int(len(t)), "profit_equals_price_difference_share": round(float(match.mean()), 4),
            "commission_total": round(float(t["commission"].sum()), 2), "fee_total": round(float(t["fee"].sum()), 2),
            "swap_total": round(float(t["swap"].sum()), 2), "lots_total": round(float(t["lots"].sum()), 2)}


def decide(scores, spec, factor_stable):
    """supported only if an entry bound AND an exit bound pass (a short pays the exit spread; critic_agent_7 B4);
    inconclusive if only one side passes or some candidate fails only the CI lower bound; else refuted."""
    verdict = {c: passes(scores["forward_entry"][c], spec, factor_stable, c) for c in ENTRY}
    exit_verdict = {c: passes(scores["forward_exit"][c], spec, factor_stable, c) for c in spec["exit_preference"]}
    entry = [c for c in spec["preference"] if verdict[c]["pass"]]
    exit_ = [c for c in spec["exit_preference"] if exit_verdict[c]["pass"]]
    near = [c for c, v in list(verdict.items()) + list(exit_verdict.items()) if v["failed"] == ["coverage_ci_lower"]]
    outcome = "supported" if entry and exit_ else "inconclusive" if (entry or exit_ or near) else "refuted"
    return outcome, verdict, exit_verdict, (entry[0] if entry else None), (exit_[0] if exit_ else None)


def demo_minutes(f, deal_minutes, entry_col, exit_col):
    """Coverage of the recommended bounds on minutes where demo deals happened (sampled days only; reported)."""
    if deal_minutes is None or not len(deal_minutes):
        return {"minutes_on_sampled_days": 0}
    g = f[f["m"].isin(set(deal_minutes))]
    out = {"minutes_on_sampled_days": int(len(g))}
    for name, col, target in (("entry", entry_col, "first"), ("exit", exit_col, "y_exit")):
        h = g[g[col].notna() & g[target].notna()] if col else g.iloc[:0]
        out[f"{name}_coverage"] = round(float((h[target] <= h[col]).mean()), 4) if len(h) else None
    return out


def analyse(mt, bars, deals_rt, fills, spec, level, rng, deal_minutes=None):
    fac = factor_stability(mt, bars, spec)
    k_cal = fac["k_cal_first_over_bar"]
    B = bounds(mt, bars, spec)
    f = with_bounds(mt, B, k_cal, spec)
    win = lambda w: f[(f["TIME_DT"] >= pd.Timestamp(w[0])) & (f["TIME_DT"] < pd.Timestamp(w[1]))]
    cal, fwd = win(spec["calibration"]), win(spec["forward"])
    cols = ENTRY + ["B50"]
    scores = {"calibration_entry": {c: score(cal, c, "first", spec, level, rng) for c in cols},
              "forward_entry": {c: score(fwd, c, "first", spec, level, rng) for c in cols},
              "forward_exit": {c: score(fwd, c, "y_exit", spec, level, rng) for c in spec["exit_preference"]}}
    stress = {f"A_fixed_x{m}": scores["forward_entry"][c]["coverage"] for m, c in (("1", "A_fixed"), ("1.5", "A_x1.5"), ("2", "A_x2"))}
    outcome, verdict, exit_verdict, recommended, recommended_exit = decide(scores, spec, fac["stable"])
    return {"outcome": outcome, "verdict": verdict, "exit_verdict": exit_verdict,
            "recommended_entry_bound": recommended, "recommended_exit_bound": recommended_exit,
            "spikes_covered": {"entry": verdict[recommended]["spikes_covered"] if recommended else None,
                               "exit": exit_verdict[recommended_exit]["spikes_covered"] if recommended_exit else None},
            "scores": scores,
            "stress_coverage": stress, "factor": fac, "rollover_flag": rollover_flag_check(bars, spec),
            "demo_deal_minutes": demo_minutes(fwd, deal_minutes, recommended, recommended_exit),
            "round_trips": deals_rt, "execution": execution(fills, spec["point"]),
            "days_with_bound": int(len(B)), "forward_days": int(fwd["day"].nunique()),
            "calibration_days": int(cal["day"].nunique())}


def run(run_dir, spec, trials):
    import MetaTrader5 as mt5
    from research_run_v1 import mt5_manifest_data
    start, end = datetime.fromisoformat(spec["bars_start"]).replace(tzinfo=timezone.utc), datetime.now(timezone.utc)
    if not mt5.initialize(path=spec.get("terminal", r"D:\XM2\terminal64.exe")):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        mt5.symbol_select("GOLD#", True)
        info = mt5.symbol_info("GOLD#")
        bars = frame(mt5.copy_rates_range("GOLD#", mt5.TIMEFRAME_M1, start, end))
        bars = bars[bars["TIME_DT"].dt.normalize() < bars["TIME_DT"].dt.normalize().max()]  # drop the partial day
        days = sorted(d for d in bars["TIME_DT"].dt.normalize().unique() if d >= pd.Timestamp(spec["tick_start"]))
        sample, parts, tick_rows = days[::spec["tick_every_nth_day"]], [], 0
        for d in sample:
            d = pd.Timestamp(d)
            t = mt5.copy_ticks_from("GOLD#", d.to_pydatetime().replace(tzinfo=timezone.utc), 4_000_000, mt5.COPY_TICKS_ALL)
            if t is None or not len(t):
                continue
            t = pd.DataFrame(t)
            # the day plus the next morning, so exits across the daily break get the reopen quote (B3)
            t = t[t["time"] < (d + pd.Timedelta(days=1, hours=spec["exit_lookahead_hours"])).value // 10**9]
            tick_rows += int((t["time"] < (d + pd.Timedelta(days=1)).value // 10**9).sum())
            parts.append(minute_table(t[["time_msc", "bid", "ask"]], info.point, spec["horizon_minutes"],
                                      day_end=d + pd.Timedelta(days=1), max_exit_gap_min=spec["max_exit_gap_minutes"]))
        mt = pd.concat(parts, ignore_index=True)
        raw = mt5.history_deals_get(datetime.fromisoformat(spec["tick_start"]).replace(tzinfo=timezone.utc), end) or []
        deals = pd.DataFrame([{"position_id": d.position_id, "entry": d.entry, "type": d.type, "price": d.price,
                               "volume": d.volume, "profit": d.profit, "commission": d.commission,
                               "fee": getattr(d, "fee", 0.0), "swap": d.swap, "time": d.time, "time_msc": d.time_msc,
                               "reason": d.reason} for d in raw if d.symbol == "GOLD#" and d.type in (0, 1)])
        fills, skipped = [], 0
        for _, dl in (deals[deals["entry"].isin([0, 1])].iterrows() if len(deals) else []):
            t = mt5.copy_ticks_from("GOLD#", datetime.fromtimestamp(int(dl["time"]) - 300, timezone.utc), 20000,
                                    mt5.COPY_TICKS_ALL)
            t = pd.DataFrame(t) if t is not None and len(t) else None
            q = None if t is None else t[(t["time_msc"] <= dl["time_msc"]) & (t["time_msc"] >= dl["time_msc"] - 60_000)]
            if q is None or not len(q):
                skipped += 1
                continue
            last = q.iloc[-1]
            fills.append({"side": "buy" if dl["type"] == 0 else "sell", "price": dl["price"], "bid": last["bid"],
                          "ask": last["ask"], "reason": int(dl["reason"]), "lag_ms": int(dl["time_msc"] - last["time_msc"])})
        manifest_data = mt5_manifest_data(mt5, ["GOLD#"], start, end, len(bars), [spec["forward"]],
                                          "MT5 history can be revised by the broker; demo account deals")
        contract = info.trade_contract_size
    finally:
        mt5.shutdown()
    spec = {**spec, "point": info.point}
    k = spec["trials_declared"] * trials
    level = 0.95 ** (1 / k)
    deal_minutes = (deals.loc[deals["entry"] == 0, "time"] // 60 * 60).tolist() if len(deals) else []
    res = analyse(mt, bars, round_trips(deals, contract), pd.DataFrame(fills), spec, level,
                  np.random.default_rng(spec["seed"]), deal_minutes)
    res["execution"]["skipped_no_quote_within_60s"] = skipped
    res["units"] = {"point": info.point, "digits": info.digits, "contract_size": contract,
                    "spread_unit": "points (1 point = 0.01 USD/oz), same as gemini.py (ask - bid) / point",
                    "tick_days_sampled": len(sample), "tick_rows": tick_rows, "minutes": int(len(mt)), "ci_level": level}
    fe, fx, v, vx = res["scores"]["forward_entry"], res["scores"]["forward_exit"], res["verdict"], res["exit_verdict"]
    line = lambda s, ok, c: (f"{c} cov {s[c].get('coverage')} (CI {s[c].get('coverage_ci')}, years {s[c].get('coverage_by_year')}, "
                             f"rollover {s[c].get('coverage_rollover')}, HH:00 {s[c].get('coverage_hour_start')}, uncovered "
                             f"{s[c].get('uncovered_cost_share')}, bound/cost {s[c].get('bound_over_cost')}, high-cost miss "
                             f"{s[c].get('high_cost_miss_rate')}) {'PASS' if ok[c]['pass'] else 'fail: ' + ','.join(ok[c]['failed'])}")
    summary = (f"{res['outcome']}: recommended entry bound {res['recommended_entry_bound']}, exit bound "
               f"{res['recommended_exit_bound']} (spikes covered {res['spikes_covered']}). Forward {spec['forward'][0]}+ "
               f"({res['forward_days']} sampled days). "
               "ENTRY: " + "; ".join(line(fe, v, c) for c in spec["preference"])
               + ". EXIT (+60 min): " + "; ".join(line(fx, vx, c) for c in spec["exit_preference"])
               + f". Stress coverage {res['stress_coverage']}. Factor tick-mean/bar: calibration "
               f"{res['factor']['calibration_tick_mean_over_bar']}, forward by year {res['factor']['forward_by_year']}, "
               f"stable {res['factor']['stable']}; rollover flag 23/0/1 holds {res['rollover_flag']['flag_holds']} "
               f"({res['rollover_flag']['share_spike_in_flag']} of months); demo round trips "
               f"{res['round_trips'].get('round_trips')} profit = fill-price difference "
               f"{res['round_trips'].get('profit_equals_price_difference_share')}")
    js = lambda x: json.dumps(x, ensure_ascii=False, indent=1, default=str)
    report = "\n".join([
        "# E-015 GOLD# conservative cost upper bound (H-023)", "", f"Outcome: **{res['outcome']}**", "", summary, "",
        "Quoted spreads only; demo deals are a demo account, not live. Stress multipliers are scenarios, not validated "
        "costs. Forward 2025+ spreads were partly seen in E-014, so this is development evidence.", "",
        "## Units", "", js(res["units"]), "", "## Verdict per candidate (forward)", "", js(res["verdict"]), "",
        js(res["exit_verdict"]), "", "## Forward entry", "", js(fe), "", "## Forward exit (+60 min, across the daily break)",
        "", js(fx), "", "## Calibration window (reported, not decisive)", "", js(res["scores"]["calibration_entry"]), "",
        "## Demo deal minutes (reported)", "", js(res["demo_deal_minutes"]), "",
        "## 1.21 factor stability", "", js(res["factor"]), "", "## Rollover flag across DST", "", js(res["rollover_flag"]),
        "", "## Demo round trips: cost components", "", js(res["round_trips"]), "",
        "## Demo fills vs tick quote (slippage)", "", js(res["execution"])])
    metrics = json.loads(js({**res, "mean_r": "n/a"}))
    return {"outcome": res["outcome"], "evidence_class": "development", "summary": summary,
            "failure_cause": "none" if res["outcome"] == "supported" else "cost",
            "lessons": f"entry bound {res['recommended_entry_bound']}, exit {res['recommended_exit_bound']}, factor stable "
                       f"{res['factor']['stable']}, rollover flag {res['rollover_flag']['flag_holds']}",
            "next_questions": "H-013 preregistration v2 with the recommended bounds, or a richer cost source if none passed",
            "metrics": metrics, "report": report, "manifest_data": manifest_data,
            "model_note": "trailing quantiles and means only; no trading model"}
