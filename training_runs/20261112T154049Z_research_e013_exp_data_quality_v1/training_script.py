"""E-013 (H-021, data diagnostic): is XM GOLD# history before 2018 of a different quality from later history, and
does that explain the lower post-2017 realised volatility that broke the 2014-2017 fitted models (E-008/E-010/E-011)?
Nothing is trained and no strategy is evaluated. Periods: P0 2014-2017, P1 2018-2023, P2 2024+.

Sources (all from the XM MT5 terminal; no independent external gold feed was reachable: Stooq requires a browser
challenge, Dukascopy and Yahoo answered 429 on 2026-11-12): GOLD# M1, H1, D1; EURUSD H1; XAUEUR# H1 (from 2017-08-28).
MT5 most likely builds H1/D1 from its stored M1, so M1/H1/D1 agreement is a low-discrimination check, reported but
never used to clear the data.

Primary (critic B1): the variance ratio across time scales. With r_h the H1 log returns (consecutive bars, overnight
included) and r_d = sum of a day's r_h (exact telescoping, so no gaps are lost), VR_day = sum r_d^2 / sum r_h^2 over a
period, and VR_week the same with weekly sums. For a price without noise VR is about 1. Quote noise that reverts within
a day - also persistent noise with a half-life of hours, which a one-lag M1 autocovariance cannot see - inflates
sum r_h^2 far more than r_d^2, so VR_day falls. delta = VR_day(P1) - VR_day(P0) with a month-block bootstrap; the
noise signal must be VR_day(P0) below 1, not VR_day(P1) above 1 (intraday trending after 2018 would raise P1 and
must not be booked as a pre-2018 defect; critic B7). Intraday mean reversion of the market itself also lowers VR and
cannot be told apart from quote noise, so the finding is worded "consistent with quote noise" and the hour-of-day
decomposition and descriptive checks are reported with it.

Descriptive checks, per year / period / month:
A. Completeness and timestamps: H1 bars per server day, hours missing on full days, weekly-open hour, duplicate or
   backward timestamps, M1 coverage per year, M1 zero-return share (stale quotes).
B. Bar integrity: OHLC inconsistency, flat bars, M1 bounce-back spikes (two consecutive returns > 8 sigmas with opposite
   signs; sigma = 1.4826 x median of the day's non-zero |returns|, floored at two ticks, so stale days cannot zero it).
C. Spread: share of bars with SPREAD = 0 (field not recorded; a missing field, not a price defect).
D. Aggregation H1 vs M1 and D1 vs H1 (low discrimination, see above).
E. RV at 1/5/15/60 minutes and H1, D1 Parkinson, Roll noise share of H1 RV (only i.i.d. quote noise).
F. Hour-of-day share of sum r_h^2 per period, and the share carried by hours present before 2018 but not after.
G. Return shape: kurtosis and |z| > 5 of H1 returns standardised by the day's RV; ATR(14)/price; tick volume.
H. Cross-symbol residual e = r(GOLD#) - r(EURUSD) - r(XAUEUR#) on consecutive common H1 bars. XAUEUR# is probably
   synthesised by XM from its own gold and EURUSD, so this checks GOLD# against XM's own gold, not an external feed.
I. Change month: the first month after which the monthly SPREAD = 0 share, median H1 bars/day and M1 zero-return
   share stay on the 2018+ side of the midpoint between the P0 and P1 medians for 6 months. If that month is on or
   before 2017-08, Aug-Dec 2017 is new-vs-new and the cross-symbol check cannot speak about the old data.
"""
import json
import math
from datetime import datetime, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from exp_vol_forecast_v1 import MIN_BARS, daily_rv
from research_run_v1 import mt5_manifest_data

PERIODS = {"P0": ("2014-01-01", "2018-01-01"), "P1": ("2018-01-01", "2024-01-01"), "P2": ("2024-01-01", "2099-01-01")}
SPIKE_SIGMAS = 8
TICK = 0.01  # GOLD# price step


def frame(rates):
    f = pd.DataFrame(rates)
    f["TIME_DT"] = pd.to_datetime(f["time"], unit="s")
    f = f.rename(columns={"open": "OPEN", "high": "HIGH", "low": "LOW", "close": "CLOSE", "spread": "SPREAD",
                          "tick_volume": "TV"})
    return f.sort_values("TIME_DT").reset_index(drop=True)


def period(index):
    lab = pd.Series("", index=index)
    for k, (a, b) in PERIODS.items():
        lab[(index >= a) & (index < b)] = k
    return lab


def clean(x):
    """JSON-safe: NaN/inf -> None, numpy scalars -> Python."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def m1_days(m1):
    """Per server day from M1 (same-day returns only): bars, zero-return share, RV at 1/5/15/60 min, Roll w2, spikes."""
    day = m1["TIME_DT"].dt.normalize()
    lc = np.log(m1["CLOSE"])
    same = day.eq(day.shift())
    r = lc.diff().where(same)
    prev = r.shift().where(same)
    nz = r.abs().where(r.abs() > 0)
    floor = 2 * TICK / m1["CLOSE"]
    sig = np.maximum(nz.groupby(day).transform("median") * 1.4826, floor)
    spike = (r.abs() > SPIKE_SIGMAS * sig) & (r.shift(-1).abs() > SPIKE_SIGMAS * sig) & (np.sign(r) == -np.sign(r.shift(-1)))
    out = pd.DataFrame({"m1_bars": day.groupby(day).size(), "zero_return_share": (r == 0).groupby(day).sum()
                        / r.notna().groupby(day).sum().clip(lower=1),
                        "rv_1m": (r ** 2).groupby(day).sum(), "acov_sum": (r * prev).groupby(day).sum(),
                        "acov_n": (r.notna() & prev.notna()).groupby(day).sum(), "spikes": spike.groupby(day).sum()})
    for k in (5, 15, 60):
        c = m1.groupby(m1["TIME_DT"].dt.floor(f"{k}min"))["CLOSE"].last()
        d = c.index.normalize()
        rk = np.log(c).diff().where(pd.Series(d).eq(pd.Series(d).shift()).to_numpy())
        out[f"rv_{k}m"] = (rk ** 2).groupby(d).sum()
    out["w2"] = -out["acov_sum"] / out["acov_n"].clip(lower=1)
    return out


def hourly_returns(h1):
    """H1 log returns over consecutive bars (overnight and weekend included), tagged by the bar's server day/week/hour."""
    r = np.log(h1["CLOSE"]).diff()
    t = h1["TIME_DT"]
    return pd.DataFrame({"r": r, "day": t.dt.normalize(), "week": t.dt.to_period("W"), "hour": t.dt.hour,
                         "month": t.dt.to_period("M")}).iloc[1:]


def variance_ratios(hr, mask):
    """VR_day and VR_week = sum of squared day/week returns over sum of squared hourly returns (rows in mask)."""
    h = hr[mask]
    rv = (h["r"] ** 2).sum()
    return float((h.groupby("day")["r"].sum() ** 2).sum() / rv), float((h.groupby("week")["r"].sum() ** 2).sum() / rv)


def bootstrap_vr(hr, spec):
    """Month-block bootstrap (months resampled within each period) of VR(P1) - VR(P0), day and week."""
    rng = np.random.default_rng(spec["seed"])
    lab = period(pd.DatetimeIndex(hr["day"]))
    stats = {}
    for p in ("P0", "P1"):
        h = hr[(lab == p).to_numpy()]
        per_month = h.groupby("month")[["r", "day", "week"]].apply(lambda g: pd.Series({
            "rv": (g["r"] ** 2).sum(), "d2": (g.groupby("day")["r"].sum() ** 2).sum(),
            "w2": (g.groupby("week")["r"].sum() ** 2).sum()}))
        stats[p] = per_month.to_numpy()
    draws = []
    for _ in range(spec["bootstrap_draws"]):
        v = {}
        for p, a in stats.items():
            s = a[rng.integers(0, len(a), len(a))].sum(axis=0)
            v[p] = (s[1] / s[0], s[2] / s[0])
        draws.append((v["P1"][0] - v["P0"][0], v["P1"][1] - v["P0"][1], v["P0"][0], v["P1"][0]))
    d = np.array(draws)
    q = lambda x: [round(float(np.quantile(x, 0.025)), 4), round(float(np.quantile(x, 0.975)), 4)]
    return [q(d[:, k]) for k in range(4)]  # weeks crossing a month boundary are split; same in both periods


def hour_shares(hr, h1):
    lab = period(pd.DatetimeIndex(hr["day"]))
    day = h1["TIME_DT"].dt.normalize()
    per_day = day.groupby(day).size()
    present = h1.assign(day=day, hour=h1["TIME_DT"].dt.hour)
    present = present[present["day"].map(per_day) >= MIN_BARS]
    plab = period(pd.DatetimeIndex(present["day"])).to_numpy()
    frac = {p: present[plab == p].groupby("hour")["day"].nunique() / max(present[plab == p]["day"].nunique(), 1)
            for p in ("P0", "P1")}
    extra = [int(h) for h in range(24) if frac["P0"].get(h, 0) > 0.5 and frac["P1"].get(h, 0) <= 0.5]
    out = {"hours_present_before_2018_only": extra}
    for p in PERIODS:
        h = hr[(lab == p).to_numpy()]
        if len(h):
            s = (h["r"] ** 2).groupby(h["hour"]).sum() / (h["r"] ** 2).sum()
            out[p] = {"share_by_hour": {int(k): round(float(v), 4) for k, v in s.items()},
                      "share_of_extra_hours": round(float(s.reindex(extra).fillna(0).sum()), 4)}
    return out


def aggregation_mismatch(fine, coarse, freq):
    """Share of coarse bars whose OHLC differs from the fine bars inside them by more than max(0.05, 2e-5 * price)."""
    key = fine["TIME_DT"].dt.floor(freq)
    agg = fine.groupby(key).agg(o=("OPEN", "first"), h=("HIGH", "max"), l=("LOW", "min"), c=("CLOSE", "last"))
    j = coarse.set_index("TIME_DT")[["OPEN", "HIGH", "LOW", "CLOSE"]].join(agg, how="inner")
    tol = np.maximum(0.05, 2e-5 * j["CLOSE"])
    bad = ((j["OPEN"] - j["o"]).abs() > tol) | ((j["HIGH"] - j["h"]).abs() > tol) | \
          ((j["LOW"] - j["l"]).abs() > tol) | ((j["CLOSE"] - j["c"]).abs() > tol)
    return bad.groupby(bad.index.year).mean()


def bar_integrity(f):
    bad = (f["HIGH"] < f[["OPEN", "CLOSE"]].max(axis=1)) | (f["LOW"] > f[["OPEN", "CLOSE"]].min(axis=1))
    y = f["TIME_DT"].dt.year
    return pd.DataFrame({"ohlc_inconsistent": bad.groupby(y).mean(), "flat": (f["HIGH"] == f["LOW"]).groupby(y).mean(),
                         "spread_zero": (f["SPREAD"] == 0).groupby(y).mean(),
                         "spread_median_nonzero": f["SPREAD"].where(f["SPREAD"] > 0).groupby(y).median(),
                         "duplicate_or_backward_time": (f["TIME_DT"].diff() <= pd.Timedelta(0)).groupby(y).sum()})


def sessions(h1):
    day = h1["TIME_DT"].dt.normalize()
    per_day = day.groupby(day).size()
    hours = h1.assign(day=day, hour=h1["TIME_DT"].dt.hour).pivot_table(index="day", columns="hour", values="CLOSE",
                                                                         aggfunc="size").reindex(columns=range(24))
    full = hours.index[per_day.reindex(hours.index) >= MIN_BARS]
    missing = hours.loc[full].isna().groupby(full.year).mean()
    gap_open = h1["TIME_DT"][h1["TIME_DT"].diff() > pd.Timedelta(hours=24)]
    out = {}
    for y in sorted(set(day.dt.year)):
        m = missing.loc[y] if y in missing.index else pd.Series(dtype=float)
        out[int(y)] = {"h1_bars_per_day_median": float(per_day[per_day.index.year == y].median()),
                       "hours_missing_on_full_days_over_50pct": [int(h) for h, v in m.items() if v > 0.5],
                       "weekly_open_hour_mode": int(gap_open[gap_open.dt.year == y].dt.hour.mode().iloc[0])
                       if (gap_open.dt.year == y).any() else None}
    return out


def change_months(h1, days):
    """First month after which each series stays on the post-2018 side of the P0/P1 midpoint for 6 months."""
    day = h1["TIME_DT"].dt.normalize()
    month = h1["TIME_DT"].dt.to_period("M")
    series = {"spread_zero_share": (h1["SPREAD"] == 0).groupby(month).mean(),
              "h1_bars_per_day": day.groupby(day).size().groupby(day.groupby(day).size().index.to_period("M")).median(),
              "m1_zero_return_share": days["zero_return_share"].groupby(days.index.to_period("M")).mean()}
    out = {}
    for name, s in series.items():
        s = s[s.index >= pd.Period("2014-01", "M")]
        p0, p1 = s[s.index < pd.Period("2018-01", "M")].median(), s[(s.index >= pd.Period("2018-01", "M")) &
                                                                      (s.index < pd.Period("2024-01", "M"))].median()
        if not (np.isfinite(p0) and np.isfinite(p1)) or p0 == p1:
            out[name] = None
            continue
        side = (s > (p0 + p1) / 2) if p1 > p0 else (s < (p0 + p1) / 2)
        stay = side[::-1].rolling(6, min_periods=6).min()[::-1] == 1
        out[name] = str(stay.index[stay.to_numpy().argmax()]) if stay.any() else None
    return out


def return_shape(h1, rv_h1):
    day = h1["TIME_DT"].dt.normalize()
    r = np.log(h1["CLOSE"]).diff()
    n = day.groupby(day).transform("size")
    z = (r / np.sqrt(day.map(rv_h1) / n)).replace([np.inf, -np.inf], np.nan)
    y = h1["TIME_DT"].dt.year
    tr = pd.concat([h1["HIGH"] - h1["LOW"], (h1["HIGH"] - h1["CLOSE"].shift()).abs(),
                    (h1["LOW"] - h1["CLOSE"].shift()).abs()], axis=1).max(axis=1)
    return pd.DataFrame({"z_kurtosis": z.groupby(y).apply(lambda s: float(s.dropna().kurt())),
                         "z_abs_gt5": (z.abs() > 5).groupby(y).mean(),
                         "atr14_pct": (tr.rolling(14).mean() / h1["CLOSE"]).groupby(y).mean() * 100,
                         "median_tick_volume": h1["TV"].groupby(y).median()})


def cross_residual(gold, eur, xaueur):
    j = pd.concat([np.log(s.set_index("TIME_DT")["CLOSE"]).rename(k) for k, s in
                   (("g", gold), ("e", eur), ("x", xaueur))], axis=1, join="inner").diff().dropna()
    j = j[(j.index.to_series().diff() == pd.Timedelta(hours=1)).to_numpy()]  # consecutive common bars only
    res = j["g"] - j["e"] - j["x"]
    keys = {"2017_08_12": ("2017-08-28", "2018-01-01"), **{str(y): (f"{y}-01-01", f"{y + 1}-01-01") for y in range(2018, 2027)}}
    out = {}
    for k, (a, b) in keys.items():
        m = (j.index >= a) & (j.index < b)
        if m.sum() > 200:
            out[k] = {"bars": int(m.sum()), "var_ratio": round(float(res[m].var() / j["x"][m].var()), 4),
                      "corr_g_minus_e_vs_x": round(float(np.corrcoef(j["g"][m] - j["e"][m], j["x"][m])[0, 1]), 4)}
    return out


def evaluate(m1, h1, d1, eur, xaueur, spec):
    """Pure (no MT5): every table and the primary statistics."""
    rvh = daily_rv(h1)
    days = m1_days(m1).join(rvh[["rv", "bars", "park"]].rename(columns={"rv": "rv_h1", "bars": "n_h1"}), how="inner")
    days = days[(days.index >= PERIODS["P0"][0]) & (days["m1_bars"] >= spec["min_m1_bars"])]
    hr = hourly_returns(h1)
    hr = hr[hr["day"] >= PERIODS["P0"][0]]
    hlab = period(pd.DatetimeIndex(hr["day"]))
    lab = period(days.index)
    by_period = {}
    for p in PERIODS:
        m, hm = (lab == p).to_numpy(), (hlab == p).to_numpy()
        if m.sum() < 50:
            continue
        d = days[m]
        vr_d, vr_w = variance_ratios(hr, hm)
        by_period[p] = {"days": int(m.sum()), "mean_rv_h1": float(d["rv_h1"].mean()), "vr_day": round(vr_d, 4),
                        "vr_week": round(vr_w, 4),
                        "roll_noise_share_h1": round(float((2 * d["n_h1"] * d["w2"]).sum() / d["rv_h1"].sum()), 5),
                        "ratio_rv_h1_over_rv_15m": round(float(d["rv_h1"].sum() / d["rv_15m"].sum()), 4),
                        "ratio_rv_h1_over_parkinson": round(float(d["rv_h1"].sum() / d["park"].sum()), 4),
                        "m1_zero_return_share": round(float(d["zero_return_share"].mean()), 4),
                        "spikes_per_1000_days": round(float(d["spikes"].mean() * 1000), 2)}
    by_year = days.groupby(days.index.year).agg(
        days=("rv_h1", "size"), rv_h1=("rv_h1", "mean"), rv_1m=("rv_1m", "mean"), rv_15m=("rv_15m", "mean"),
        rv_60m=("rv_60m", "mean"), park=("park", "mean"), w2=("w2", "mean"), spikes=("spikes", "mean"),
        m1_bars=("m1_bars", "mean"), zero_return_share=("zero_return_share", "mean"))
    by_year["vr_day"] = [variance_ratios(hr, (pd.DatetimeIndex(hr["day"]).year == y))[0] for y in by_year.index]
    ci_day, ci_week, ci_p0, ci_p1 = bootstrap_vr(hr, spec)
    p0, p1 = by_period["P0"], by_period["P1"]
    cross = cross_residual(h1, eur, xaueur) if xaueur is not None and len(xaueur) else {}
    later = [v["var_ratio"] for k, v in cross.items() if k != "2017_08_12" and k < "2024"]
    primary = {"vr_day_P0": p0["vr_day"], "vr_day_P1": p1["vr_day"], "delta_vr_day": round(p1["vr_day"] - p0["vr_day"], 4),
               "delta_vr_day_ci95": ci_day, "vr_day_P0_ci95": ci_p0, "vr_day_P1_ci95": ci_p1,
               "delta_vr_week": round(p1["vr_week"] - p0["vr_week"], 4),
               "delta_vr_week_ci95": ci_week, "rv_level_ratio_P1_over_P0": round(p1["mean_rv_h1"] / p0["mean_rv_h1"], 4),
               "rv_level_ratio_vr_adjusted": round(p1["mean_rv_h1"] * p1["vr_day"] / (p0["mean_rv_h1"] * p0["vr_day"]), 4),
               "cross_residual_ratio_2017H2_over_2018_2023": round(cross["2017_08_12"]["var_ratio"] / np.median(later), 3)
               if "2017_08_12" in cross and later else None,
               "spike_rate_ratio_P0_over_P1": round(p0["spikes_per_1000_days"] / max(p1["spikes_per_1000_days"], 1.0), 3)}
    return {"by_period": by_period, "by_year": by_year, "primary": primary, "hours": hour_shares(hr, h1),
            "integrity": {tf: bar_integrity(f) for tf, f in (("M1", m1), ("H1", h1), ("D1", d1))},
            "mismatch": {"H1_vs_M1": aggregation_mismatch(m1, h1, "h"), "D1_vs_H1": aggregation_mismatch(h1, d1, "D")},
            "sessions": sessions(h1), "change_months": change_months(h1, days), "shape": return_shape(h1, rvh["rv"]),
            "cross": cross}


def classify(out, spec):
    """Pre-declared rules -> (verdict, findings list of (item, class, evidence))."""
    p, f = out["primary"], []
    for tf, t in out["integrity"].items():
        bad_years = [int(y) for y, v in t["spread_zero"].items() if v > 0.5]
        if bad_years:
            f.append((f"{tf} spread field", "confirmed_defect", f"SPREAD = 0 on > 50% of bars in {bad_years} "
                                                                f"(missing cost field, not a price defect)"))
        inc = t["ohlc_inconsistent"]
        if (inc > spec["ohlc_defect_rate"]).any():
            f.append((f"{tf} OHLC integrity", "confirmed_defect", f"inconsistent OHLC rate > {spec['ohlc_defect_rate']} in "
                                                                  f"{[int(y) for y, v in inc.items() if v > spec['ohlc_defect_rate']]}"))
    for name, s in out["mismatch"].items():
        early, late = s[s.index < 2018], s[(s.index >= 2018) & (s.index < 2024)]
        if len(early) and len(late) and early.mean() > spec["mismatch_defect_rate"] and early.mean() > 5 * max(late.mean(), 1e-6):
            f.append((f"{name} aggregation", "confirmed_defect", f"mismatch {early.mean():.4f} before 2018 vs {late.mean():.4f}"))
    sess = out["sessions"]
    pre = {sess[y]["h1_bars_per_day_median"] for y in sess if 2014 <= y < 2018}
    post = {sess[y]["h1_bars_per_day_median"] for y in sess if 2018 <= y < 2024}
    if pre and post and pre.isdisjoint(post):
        f.append(("session length", "confirmed_collection_change", f"median H1 bars/day {sorted(pre)} before vs "
                                                                   f"{sorted(post)} after; extra hours "
                                                                   f"{out['hours']['hours_present_before_2018_only']}"))
    months = [m for m in out["change_months"].values() if m]
    if months and min(months) <= "2017-08":
        f.append(("cross-symbol check", "undetermined", f"data change month {min(months)} is on or before 2017-08, so "
                                                         f"Aug-Dec 2017 compares new data with new data"))
    d, (lo, hi), dw = p["delta_vr_day"], p["delta_vr_day_ci95"], p["delta_vr_week"]
    p0_low = p["vr_day_P0_ci95"][1] < 1  # the noise signal is P0 below 1 ...
    p1_near = p["vr_day_P1_ci95"][0] <= 1 <= p["vr_day_P1_ci95"][1]  # ... not P1 above (or below) 1
    if d >= spec["vr_inflation_delta"] and lo > 0 and dw > 0 and p0_low and p1_near:
        verdict = "supported"
        f.append(("RV level shift", "intraday_reversal_excess_consistent_with_quote_noise",
                  f"VR_day P0 {p['vr_day_P0']} (CI {p['vr_day_P0_ci95']}) vs P1 {p['vr_day_P1']} (CI {p['vr_day_P1_ci95']}), "
                  f"delta CI {p['delta_vr_day_ci95']}, week delta {dw}; real intraday mean reversion is not excluded"))
    elif lo > -spec["vr_equivalence_band"] and hi < spec["vr_equivalence_band"] and abs(dw) < spec["vr_week_tolerance"]:
        verdict = "not_explained_by_quote_noise"
        f.append(("RV level shift", "not_explained_by_quote_noise", f"VR_day delta CI {p['delta_vr_day_ci95']} inside "
                                                                    f"+-{spec['vr_equivalence_band']}, week delta {dw}; a "
                                                                    f"feed-wide difference cannot be excluded"))
    else:
        verdict = "inconclusive"
        f.append(("RV level shift", "undetermined", f"VR_day delta {d} CI {p['delta_vr_day_ci95']}, week delta {dw}, "
                                                    f"P0 CI {p['vr_day_P0_ci95']}, P1 CI {p['vr_day_P1_ci95']}"))
    f.append(("external cross-check", "undetermined", "no independent gold feed reachable; a feed-wide level "
                                                      "difference that is internally consistent cannot be excluded"))
    return verdict, f


def run(run_dir, spec, trials):
    start, end = datetime(2013, 10, 1, tzinfo=timezone.utc), datetime.now(timezone.utc)
    if not mt5.initialize(path=spec.get("terminal", r"D:\XM2\terminal64.exe")):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        for s in ("GOLD#", "EURUSD", "XAUEUR#"):
            mt5.symbol_select(s, True)
        get = lambda s, tf: frame(mt5.copy_rates_range(s, tf, start, end))
        m1, h1, d1 = get("GOLD#", mt5.TIMEFRAME_M1), get("GOLD#", mt5.TIMEFRAME_H1), get("GOLD#", mt5.TIMEFRAME_D1)
        eur, xaueur = get("EURUSD", mt5.TIMEFRAME_H1), get("XAUEUR#", mt5.TIMEFRAME_H1)
        manifest_data = mt5_manifest_data(mt5, ["GOLD#", "EURUSD", "XAUEUR#"], start, end, len(m1), spec["windows"],
                                          "MT5 history can be revised by the broker")
    finally:
        mt5.shutdown()
    if m1["TIME_DT"].iloc[0] > pd.Timestamp(spec["m1_must_start_by"]):
        raise RuntimeError(f"M1 history starts {m1['TIME_DT'].iloc[0]} (terminal max bars?); P0 would be truncated")
    today = lambda f: f[f["TIME_DT"].dt.normalize() < f["TIME_DT"].dt.normalize().max()]  # drop today's partial day
    m1, h1, d1, eur, xaueur = (today(f) for f in (m1, h1, d1, eur, xaueur))
    out = evaluate(m1, h1, d1, eur, xaueur, spec)
    verdict, findings = classify(out, spec)
    p = out["primary"]
    summary = (f"{verdict}: VR_day P0 {p['vr_day_P0']} P1 {p['vr_day_P1']} (delta {p['delta_vr_day']}, CI "
               f"{p['delta_vr_day_ci95']}), VR_week delta {p['delta_vr_week']} (CI {p['delta_vr_week_ci95']}); RV level "
               f"P1/P0 {p['rv_level_ratio_P1_over_P0']}, VR-adjusted {p['rv_level_ratio_vr_adjusted']}; change months "
               f"{out['change_months']}; findings: " + "; ".join(f"{a}: {b}" for a, b, _ in findings))
    tbl = lambda df: "\n".join(["| " + " | ".join(["key", *map(str, df.columns)]) + " |",
                                "|---" * (len(df.columns) + 1) + "|",
                                *["| " + " | ".join([str(i), *[f"{v:.6g}" if isinstance(v, float) else str(v) for v in r]]) + " |"
                                  for i, r in zip(df.index, df.to_numpy())]])
    report = "\n".join([
        "# E-013 GOLD# historical data quality (H-021, diagnostic)", "", f"Outcome: **{verdict}**", "", summary, "",
        "Diagnostic only: no model is trained, no strategy evaluated, and nothing here re-establishes any past "
        "out-of-sample edge. M1/H1/D1 agreement and RV_H1/RV_15m are low-discrimination (same stored M1).", "",
        "## Findings (pre-declared classes)", "", "| item | class | evidence |", "|---|---|---|",
        *[f"| {a} | {b} | {c} |" for a, b, c in findings], "", "## Primary statistics", "", str(p), "",
        "## By period", "", str(out["by_period"]), "", "## By year (daily means; vr_day per year)", "", tbl(out["by_year"]), "",
        "## Hour-of-day share of squared H1 returns", "", str(out["hours"]), "",
        "## Change months", "", str(out["change_months"]), "",
        *[x for tf, t in out["integrity"].items() for x in (f"## Bar integrity {tf}", "", tbl(t), "")],
        "## Aggregation mismatch rate by year (low discrimination)", "", tbl(pd.DataFrame(out["mismatch"])), "",
        "## Sessions", "", str(out["sessions"]), "", "## Return shape", "", tbl(out["shape"]), "",
        "## Cross-symbol residual (GOLD# - EURUSD - XAUEUR#; XM's own gold, not external)", "", str(out["cross"])])
    metrics = clean({"primary": p, "by_period": out["by_period"], "findings": findings, "verdict": verdict,
                     "by_year": json.loads(out["by_year"].to_json(orient="index")), "hours": out["hours"],
                     "change_months": out["change_months"], "cross": out["cross"], "sessions": out["sessions"],
                     "m1_first_bar": str(m1["TIME_DT"].iloc[0]), "mean_r": "n/a"})
    outcome = {"supported": "supported", "not_explained_by_quote_noise": "refuted"}.get(verdict, "inconclusive")
    return {"outcome": outcome, "evidence_class": "development", "summary": summary,
            "failure_cause": "data" if verdict == "supported" else "none",
            "lessons": "data-quality classes recorded: " + "; ".join(f"{a}={b}" for a, b, _ in findings),
            "next_questions": "excluding or re-weighting any period needs its own preregistration; past results stay "
                              "as recorded and reruns on other periods are not confirmatory",
            "metrics": metrics, "report": report, "manifest_data": manifest_data,
            "model_note": "diagnostic; no model trained or evaluated"}
