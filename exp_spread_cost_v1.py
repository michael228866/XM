"""E-014 (H-009): XM GOLD# spread data quality, a time-conditional cost model, and spread anomalies.

Q1 cost model: does an hour-of-week median of the M1 SPREAD field (fit on the train window) predict test-window
   spreads better than one global median, in every test year and without drifting mean cost? A monthly re-estimated
   version (previous 90 days) is scored the same way, so instability of the static model shows.
Q2 H-009 with its own falsification rule: minutes whose spread z (vs the same hour-of-week over the previous 90
   days) is above that window's 90th percentile -- is the next 30 minutes' realized volatility (sqrt of RV, scaled by
   the same window's hour-of-week median) >= 1.2x that after other minutes of the same month? Plus whether the anomaly
   adds anything beyond the past 30 minutes' variance. Norms trail so that level drift across 2023-2026 (price,
   spread, volatility) cannot be read as anomalies (critic_agent_6 B1).
Q3 quote vs cost: what the M1 bar SPREAD field is (min / last / first / max / mean of the tick spreads inside the
   minute, on sampled tick days), and demo fills against the tick quote at the fill time.
Descriptive: zero spreads are missing (never zero cost), units, timestamps, sessions, percentiles, extreme runs.

Spread-only tables start at spread_start (2018-10-01, D-016); everything that touches M1 prices or fits a model
starts at train[0] (2019-01-01). Q1's static model uses the train window; adaptive parts and all Q2 norms use only
the 90 days that end before each test month.
"""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

ROLLOVER_HOURS = (23, 0, 1)  # server time: the last hour and the first bar of the day (daily open is 01:00)
SESSIONS = {"rollover_23_01": ROLLOVER_HOURS, "asia_02_09": tuple(range(2, 10)), "london_10_15": tuple(range(10, 16)),
            "new_york_16_20": tuple(range(16, 21)), "late_21_22": (21, 22)}
EPS = 1e-10  # floor for log realized variance (flat 30-minute windows)


def frame(rates):
    f = pd.DataFrame(rates)
    f["TIME_DT"] = pd.to_datetime(f["time"], unit="s")  # MT5 server time
    f = f.rename(columns={"open": "OPEN", "high": "HIGH", "low": "LOW", "close": "CLOSE", "spread": "SPREAD"})
    return f[["TIME_DT", "OPEN", "HIGH", "LOW", "CLOSE", "SPREAD"]]


def pct(x):
    x = pd.Series(x, dtype=float).dropna()
    if not len(x):
        return {"n": 0}
    q = x.quantile([0.5, 0.9, 0.95, 0.99]).to_numpy()
    return {"n": int(len(x)), "mean": round(float(x.mean()), 2), "median": float(q[0]), "p90": float(q[1]),
            "p95": float(q[2]), "p99": float(q[3])}


def between(f, w):
    return f[(f["TIME_DT"] >= pd.Timestamp(w[0])) & (f["TIME_DT"] < pd.Timestamp(w[1]))]


def with_cells(m1):
    f = m1.copy()
    f["CELL"] = (f["TIME_DT"].dt.dayofweek * 24 + f["TIME_DT"].dt.hour).to_numpy()
    f["S"] = f["SPREAD"].where(f["SPREAD"] > 0)  # zero/negative = missing, never zero cost
    return f


def describe(m1, spec):
    f = with_cells(m1[m1["TIME_DT"] >= pd.Timestamp(spec["spread_start"])])
    f["YEAR"] = f["TIME_DT"].dt.year
    by_year = {}
    for y, g in f.groupby("YEAR"):
        by_year[int(y)] = {"bars": int(len(g)), "missing_share": round(float(g["S"].isna().mean()), 4),
                           "bars_per_day_median": float(g.groupby(g["TIME_DT"].dt.normalize()).size().median()),
                           **pct(g["S"])}
    v = f.dropna(subset=["S"])
    hour = {int(h): pct(g["S"]) for h, g in v.groupby(v["TIME_DT"].dt.hour)}
    sessions = {name: pct(v.loc[v["TIME_DT"].dt.hour.isin(hs), "S"]) for name, hs in SESSIONS.items()}
    # Timestamp check: in server time the rollover spike should sit at the same hour in summer and winter.
    spike = {}
    for (y, summer), g in v.groupby([v["YEAR"], v["TIME_DT"].dt.month.between(4, 10)]):
        med = g.groupby(g["TIME_DT"].dt.hour)["S"].mean()
        spike[f"{y}_{'summer' if summer else 'winter'}"] = int(med.idxmax())
    first = f["TIME_DT"].dt.second.ne(0).mean()
    return {"by_year": by_year, "by_hour": hour, "sessions": sessions, "spike_hour_by_season": spike,
            "bars_not_minute_aligned_share": round(float(first), 6),
            "first_valid_spread_bar": str(v["TIME_DT"].iloc[0]) if len(v) else None}


def extreme_runs(test, p99):
    """Runs of consecutive minutes above the train hour-of-week p99."""
    hit = (test["S"] > test["CELL"].map(p99)).to_numpy()
    t = test["TIME_DT"].to_numpy().astype("datetime64[s]").astype(np.int64)
    runs, n = [], 0
    for i in range(len(hit)):
        if hit[i] and n and t[i] - t[i - 1] == 60:
            n += 1
        else:
            if n:
                runs.append(n)
            n = 1 if hit[i] else 0
    if n:
        runs.append(n)
    hours = len(test) / 60
    r = np.array(runs) if runs else np.array([0])
    return {"minutes_above_p99_share": round(float(hit.mean()), 4), "runs": len(runs),
            "runs_per_100_trading_hours": round(100 * len(runs) / hours, 2) if hours else None,
            "run_minutes_median": float(np.median(r)), "run_minutes_p90": float(np.quantile(r, 0.9)),
            "run_minutes_max": int(r.max())}


def cell_table(f, min_bars):
    g = f.groupby("CELL")["S"]
    n = g.size()
    return g.median().where(n >= min_bars), g.mean().where(n >= min_bars)


def ci(draws, level):
    if not len(draws):
        return [float("nan"), float("nan")]  # e.g. no non-anomalous minutes: the decision falls to inconclusive
    a = (1 - level) / 2
    return [round(float(np.quantile(draws, a)), 4), round(float(np.quantile(draws, 1 - a)), 4)]


def month_bootstrap(groups, stat, spec, rng):
    """groups: array (months, k) of per-month sums; stat: function of the column sums."""
    out = []
    for _ in range(spec["bootstrap_draws"]):
        out.append(stat(groups[rng.integers(0, len(groups), len(groups))].sum(axis=0)))
    return np.array(out)


def trailing(v, start, spec):
    """Rows of the adaptive window before a test month: the previous adaptive_days, ending a margin before the month
    so that no forward 30-minute outcome reaches into it."""
    margin = pd.Timedelta(minutes=spec.get("max_span_minutes", 40))
    return v[(v["TIME_DT"] >= start - pd.Timedelta(days=spec["adaptive_days"])) & (v["TIME_DT"] < start - margin)]


def cost_model(m1, spec, level, rng):
    v = with_cells(m1).dropna(subset=["S"])
    tr, te = between(v, spec["train"]), between(v, spec["test"]).copy()
    c0, c0mean = tr["S"].median(), tr["S"].mean()
    med, mean = cell_table(tr, spec["min_cell_bars"])
    te["C0"], te["C1"], te["C1M"] = c0, te["CELL"].map(med).fillna(c0), te["CELL"].map(mean).fillna(c0mean)
    te["C0A"], te["C2"], te["C2M"] = np.nan, np.nan, np.nan
    for month, idx in te.groupby(te["TIME_DT"].dt.to_period("M")).groups.items():
        w = trailing(v, month.start_time, spec)
        w = w if len(w) else tr
        wm, wmean = cell_table(w, spec["min_cell_bars_adaptive"])
        te.loc[idx, "C0A"] = w["S"].median()  # C2's fair control: the same window without the hour structure
        te.loc[idx, "C2"] = te.loc[idx, "CELL"].map(wm).fillna(w["S"].median()).to_numpy()
        te.loc[idx, "C2M"] = te.loc[idx, "CELL"].map(wmean).fillna(w["S"].mean()).to_numpy()
    for c in ("C0", "C1", "C0A", "C2"):
        te[f"E{c}"] = (te["S"] - te[c]).abs()
    te["MONTH"] = te["TIME_DT"].dt.to_period("M")
    pairs = {"C1": ("EC1", "EC0"), "C2": ("EC2", "EC0A")}
    sums = te.groupby("MONTH")[["EC0", "EC1", "EC0A", "EC2"]].sum()
    out = {"train_global_median": float(c0), "test_bars": int(len(te)), "months": int(len(sums))}
    off = te[~te["TIME_DT"].dt.hour.isin(ROLLOVER_HOURS)]
    for c, (num, den) in pairs.items():
        s = sums[[num, den]].to_numpy()
        out[f"mae_ratio_{c}"] = round(float(s[:, 0].sum() / s[:, 1].sum()), 4)
        out[f"mae_ratio_{c}_ci"] = ci(month_bootstrap(s, lambda x: x[0] / x[1], spec, rng), level)
        out[f"mae_ratio_{c}_excl_rollover"] = round(float(off[num].sum() / off[den].sum()), 4)
    years = {}
    for y, g in te.groupby(te["TIME_DT"].dt.year):
        actual = g["S"].mean()
        years[int(y)] = {"bars": int(len(g)), "actual_mean": round(float(actual), 2),
                         **{f"mae_ratio_{c}": round(float(g[num].sum() / g[den].sum()), 4) for c, (num, den) in pairs.items()},
                         "mean_bias_C1": round(float((g["C1M"].mean() - actual) / actual), 4),
                         "mean_bias_C2": round(float((g["C2M"].mean() - actual) / actual), 4)}
    out["by_year"] = years
    for c in pairs:
        out[f"{c}_passes"] = bool(out[f"mae_ratio_{c}"] <= spec["mae_ratio_max"] and out[f"mae_ratio_{c}_ci"][1] < 1
                                  and out[f"mae_ratio_{c}_excl_rollover"] < 1
                                  and all(y[f"mae_ratio_{c}"] < 1 for y in years.values())
                                  and all(abs(y[f"mean_bias_{c}"]) <= spec["mean_bias_max"] for y in years.values()))
    p99 = tr.groupby("CELL")["S"].quantile(0.99)
    out["extreme_runs_test"] = extreme_runs(te, p99)
    return out


def windows_rv(m1, horizon, max_span_min, price="CLOSE"):
    """Per bar i: realized variance of the next `horizon` M1 returns and of the last `horizon` (incl. bar i)."""
    t = m1["TIME_DT"].to_numpy().astype("datetime64[s]").astype(np.int64)
    r = np.diff(np.log(m1[price].to_numpy()), prepend=np.nan)
    r[0] = 0.0
    c = np.concatenate([[0.0], np.cumsum(r * r)])
    n, idx = len(t), np.arange(len(t))
    nxt, prior = np.full(n, np.nan), np.full(n, np.nan)
    ok = idx + horizon < n
    j = idx[ok]
    span_ok = t[j + horizon] - t[j] <= max_span_min * 60
    nxt[j[span_ok]] = c[j[span_ok] + horizon + 1] - c[j[span_ok] + 1]
    ok = idx - horizon >= 0
    j = idx[ok]
    span_ok = t[j] - t[j - horizon] <= max_span_min * 60
    prior[j[span_ok]] = c[j[span_ok] + 1] - c[j[span_ok] - horizon + 1]
    return nxt, prior


def month_ratio(d, col="NV"):
    """Within-month ratio of mean normalized volatility after anomalies vs other minutes, combined across months as
    an anomaly-weighted geometric mean (month fixed effect: period-level drift cannot create the ratio)."""
    g = d.groupby(["MONTH", "A"])[col].agg(["mean", "size"]).unstack("A")
    g = g.dropna()
    if not len(g):
        return np.empty((0, 2))
    return np.column_stack([np.log(g[("mean", 1.0)] / g[("mean", 0.0)]), g[("size", 1.0)]])


def combine(lw):
    return float(np.exp((lw[:, 0] * lw[:, 1]).sum() / lw[:, 1].sum())) if len(lw) else np.nan


def anomaly(m1, spec, level, rng):
    start = pd.Timestamp(spec["test"][0]) - pd.Timedelta(days=spec["adaptive_days"] + 1)
    f = with_cells(m1[m1["TIME_DT"] >= max(start, pd.Timestamp(spec["train"][0]))]).reset_index(drop=True)
    # bars carry bid; mid as a robustness check (a missing spread carries the last known one, never 0)
    f["MID"] = f["CLOSE"] + f["S"].ffill().fillna(f["S"].median()) * spec["point"] / 2
    f["RV_NEXT"], f["RV_PRIOR"] = windows_rv(f, spec["horizon_minutes"], spec["max_span_minutes"])
    _, f["RV_PRIOR_4H"] = windows_rv(f, 240, 300)  # descriptive control: multi-hour volatility state
    f["RV_NEXT_MID"], _ = windows_rv(f, spec["horizon_minutes"], spec["max_span_minutes"], "MID")
    v = f.dropna(subset=["S"])
    parts, thresholds = [], []
    for month, idx in between(v, spec["test"]).groupby(between(v, spec["test"])["TIME_DT"].dt.to_period("M")).groups.items():
        w = trailing(v, month.start_time, spec).dropna(subset=["RV_NEXT"])
        g = w.groupby("CELL")
        n, mu = g.size(), g["S"].mean()
        sd = g["S"].std().clip(lower=1.0)  # a constant cell still flags any widening (1 point = one quote step)
        keep = n[n >= spec["min_cell_bars_adaptive"]].index
        wz = ((w["S"] - w["CELL"].map(mu)) / w["CELL"].map(sd))[w["CELL"].isin(keep)]
        if not len(wz):
            continue
        thr = float(wz.quantile(spec["anomaly_quantile"]))
        norm = g["RV_NEXT"].median().pipe(np.sqrt)
        norm_mid = g["RV_NEXT_MID"].median().pipe(np.sqrt)
        m = v.loc[idx]
        m = m[m["CELL"].isin(keep) & m["CELL"].isin(norm[norm > 0].index)].dropna(subset=["RV_NEXT"]).copy()
        m["Z"] = (m["S"] - m["CELL"].map(mu)) / m["CELL"].map(sd)
        m["A"] = (m["Z"] > thr).astype(float)  # strict: ties at the threshold are not anomalies
        m["NV"] = np.sqrt(m["RV_NEXT"]) / m["CELL"].map(norm)  # volatility (sqrt of RV), as H-009 states it
        m["NV_MID"] = np.sqrt(m["RV_NEXT_MID"]) / m["CELL"].map(norm_mid)
        thresholds.append(thr)
        parts.append(m)
    te = pd.concat(parts)
    te["MONTH"] = te["TIME_DT"].dt.to_period("M")
    lw = month_ratio(te)
    draws = np.array([combine(lw[rng.integers(0, len(lw), len(lw))]) for _ in range(spec["bootstrap_draws"])])
    off = te[~te["TIME_DT"].dt.hour.isin(ROLLOVER_HOURS)]
    by_year = {int(y): round(combine(month_ratio(g)), 4) for y, g in te.groupby(te["TIME_DT"].dt.year)}
    # Increment beyond recent volatility: log RV_next on log RV_prior and the anomaly flag, demeaned within
    # hour-of-week x month; flat windows (RV = 0) are dropped instead of floored.
    r = te[(te["RV_NEXT"] > 0) & (te["RV_PRIOR"] > 0)].dropna(subset=["RV_PRIOR"])
    ms, beta = increment_sums(r, ["RV_PRIOR"])
    g_draws = month_bootstrap(ms, beta, spec, rng)
    r4 = r[r["RV_PRIOR_4H"] > 0].dropna(subset=["RV_PRIOR_4H"])
    ms4, beta4 = increment_sums(r4, ["RV_PRIOR", "RV_PRIOR_4H"])
    a, o = te[te.A == 1], te[te.A == 0]
    out = {"z_threshold_median": round(float(np.median(thresholds)), 4), "test_minutes": int(len(te)),
           "anomaly_share_test": round(float(te["A"].mean()), 4), "anomalies_test": int(te["A"].sum()),
           "months_with_anomalies": int(len(lw)), "months": int(te["MONTH"].nunique()),
           "vol_ratio": round(combine(lw), 4), "vol_ratio_ci": ci(draws[~np.isnan(draws)], level),
           "vol_ratio_median_based": round(float(a["NV"].median() / o["NV"].median()), 4),
           "vol_ratio_mid_price": round(combine(month_ratio(te, "NV_MID")), 4),
           "vol_ratio_excl_rollover": round(combine(month_ratio(off)), 4), "vol_ratio_by_year": by_year,
           "increment_log_rv": round(float(beta(ms.sum(axis=0))), 4), "increment_ci": ci(g_draws[~np.isnan(g_draws)], level),
           "increment_with_4h_control_descriptive": round(float(beta4(ms4.sum(axis=0))), 4),
           "flat_windows_dropped_from_increment": int(len(te) - len(r))}
    move = np.sqrt(te["RV_NEXT"]) * te["CLOSE"]  # typical absolute 30-minute move in price units
    out["direct_cost"] = {"spread_mean_anomaly": round(float(a["S"].mean()), 2), "spread_mean_other": round(float(o["S"].mean()), 2),
                          "spread_over_move_median_anomaly": round(float((a["S"] * spec["point"] / move[te.A == 1]).median()), 4),
                          "spread_over_move_median_other": round(float((o["S"] * spec["point"] / move[te.A == 0]).median()), 4)}
    return out


def increment_sums(r, controls):
    """Per-month X'X and X'y of log next RV on log controls and the anomaly flag, demeaned within hour-of-week x month;
    returns (month sums, function giving the anomaly coefficient from summed rows)."""
    cols = [*controls, "A"]
    d = pd.DataFrame({"y": np.log(r["RV_NEXT"]), **{c: np.log(r[c]) for c in controls}, "A": r["A"]})
    g = r["CELL"].astype(str) + "_" + r["MONTH"].astype(str)
    d = d - d.groupby(g).transform("mean")
    k = len(cols)
    terms = [d[a] * d[b] for a in cols for b in cols] + [d[a] * d["y"] for a in cols]
    ms = pd.concat(terms, axis=1).groupby(r["MONTH"].to_numpy()).sum().to_numpy()

    def beta(s):
        xtx, xty = s[:k * k].reshape(k, k), s[k * k:]
        return np.linalg.solve(xtx, xty)[-1] if np.linalg.det(xtx) > 0 else np.nan

    return ms, beta


def field_semantics(ticks, m1, point):
    """ticks: DataFrame time_msc, bid, ask (server-time ms). Which per-minute tick statistic equals the bar SPREAD?
    Compared only on minutes whose tick spread changed (min != max), where the candidates can disagree."""
    tk = ticks[(ticks["bid"] > 0) & (ticks["ask"] > 0)]
    d = pd.DataFrame({"m": (tk["time_msc"] // 60000) * 60, "sp": np.rint((tk["ask"] - tk["bid"]) / point)})
    g = d.groupby("m")["sp"]
    agg = pd.DataFrame({"min": g.min(), "max": g.max(), "first": g.first(), "last": g.last(), "mean": g.mean()})
    bar = m1.set_index(m1["TIME_DT"].astype("int64") // 10**9)["SPREAD"]
    j = agg.join(bar.rename("bar"), how="inner")
    j = j[j["bar"] > 0]
    varied = j[j["min"] != j["max"]]
    if not len(varied):
        return {"minutes": int(len(j)), "varied_minutes": 0, "label": "undetermined"}
    share = {c: round(float((varied["bar"] == (np.rint(varied[c]) if c == "mean" else varied[c])).mean()), 4)
             for c in agg.columns}
    ranked = sorted(share.values(), reverse=True)
    best = max(share, key=share.get)
    clear = ranked[0] >= 0.9 and ranked[0] - ranked[1] >= 0.2
    return {"minutes": int(len(j)), "varied_minutes": int(len(varied)), "match_share_varied": share,
            "label": best if clear else "undetermined", "bar_field_mean": round(float(j["bar"].mean()), 2),
            "tick_mean_of_minute_means": round(float(j["mean"].mean()), 2),
            "tick_mean_over_bar_mean": round(float(j["mean"].mean() / j["bar"].mean()), 4)}


def execution(fills, point):
    """fills: side ('buy'/'sell'), price, bid, ask, reason, lag_ms (fill time - quote time). Adverse = paid beyond
    the quoted side, in points."""
    if not len(fills):
        return {"fills": 0}
    buy = fills["side"].eq("buy")
    adverse = np.where(buy, fills["price"] - fills["ask"], fills["bid"] - fills["price"]) / point
    quoted = (fills["ask"] - fills["bid"]) / point
    f = fills.assign(adverse=np.round(adverse, 2), quoted=np.round(quoted, 2))
    out = {"fills": int(len(f)), "adverse_points": pct(f["adverse"]), "share_adverse_gt_0": round(float((f["adverse"] > 0).mean()), 4),
           "quoted_spread_points": pct(f["quoted"]),
           "effective_cost_points_mean": round(float((f["quoted"] / 2 + f["adverse"]).mean()), 2)}
    if "lag_ms" in f:
        out["quote_lag_ms"] = pct(f["lag_ms"])
    out["by_reason"] = {str(r): {"fills": int(len(g)), "adverse_mean": round(float(g["adverse"].mean()), 2)}
                        for r, g in f.groupby("reason")}
    return out


def decide(cm, an, field, exe, spec):
    lo, hi = an["vol_ratio_ci"]
    thr, years = spec["vol_ratio_threshold"], list(an["vol_ratio_by_year"].values())
    robust = (an["vol_ratio_excl_rollover"] >= thr and sum(y >= thr for y in years) >= spec["year_share_min"] * len(years)
              and spec["anomaly_share_band"][0] <= an["anomaly_share_test"] <= spec["anomaly_share_band"][1])
    if an["anomalies_test"] < spec["min_anomalies"] or an["months_with_anomalies"] < spec["min_months"]:
        q2 = "inconclusive_sample"
    elif an["vol_ratio"] >= thr and lo > 1 and robust:
        q2 = "vol_clause_not_refuted" if an["increment_ci"][0] > 0 else "vol_clause_not_refuted_but_redundant"
    elif hi < thr:
        q2 = "refuted"
    else:
        q2 = "inconclusive"
    q1 = "static_generalizes" if cm["C1_passes"] else "adaptive_only" if cm["C2_passes"] else "not_reliable"
    ready = q1 != "not_reliable" and field["label"] != "undetermined" and exe.get("fills", 0) >= spec["min_fills"]
    return q1, q2, ready


def run(run_dir, spec, trials):
    import MetaTrader5 as mt5
    from research_run_v1 import mt5_manifest_data
    start, end = datetime.fromisoformat(spec["spread_start"]).replace(tzinfo=timezone.utc), datetime.now(timezone.utc)
    if not mt5.initialize(path=spec.get("terminal", r"D:\XM2\terminal64.exe")):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        mt5.symbol_select("GOLD#", True)
        info = mt5.symbol_info("GOLD#")
        m1 = frame(mt5.copy_rates_range("GOLD#", mt5.TIMEFRAME_M1, start, end))
        m1 = m1[m1["TIME_DT"].dt.normalize() < m1["TIME_DT"].dt.normalize().max()]  # drop today's partial day
        days = sorted(d for d in m1["TIME_DT"].dt.normalize().unique() if d >= pd.Timestamp(spec["tick_start"]))
        sample = days[::spec["tick_every_nth_day"]]
        parts = []
        for d in sample:
            t = mt5.copy_ticks_from("GOLD#", pd.Timestamp(d).to_pydatetime().replace(tzinfo=timezone.utc), 3_000_000,
                                    mt5.COPY_TICKS_ALL)
            if t is not None and len(t):
                t = pd.DataFrame(t)
                parts.append(t[t["time"] < (pd.Timestamp(d) + pd.Timedelta(days=1)).value // 10**9][["time_msc", "bid", "ask"]])
        ticks = pd.concat(parts) if parts else pd.DataFrame(columns=["time_msc", "bid", "ask"])
        deals = mt5.history_deals_get(datetime.fromisoformat(spec["tick_start"]).replace(tzinfo=timezone.utc), end) or []
        fills, skipped = [], 0
        for dl in deals:
            if dl.symbol != "GOLD#" or dl.type not in (0, 1) or dl.entry not in (0, 1):
                continue
            t = mt5.copy_ticks_from("GOLD#", datetime.fromtimestamp(dl.time - 300, timezone.utc), 20000, mt5.COPY_TICKS_ALL)
            t = pd.DataFrame(t) if t is not None and len(t) else None
            q = None if t is None else t[(t["time_msc"] <= dl.time_msc) & (t["time_msc"] >= dl.time_msc - 60_000)]
            if q is None or not len(q):
                skipped += 1
                continue
            last = q.iloc[-1]
            fills.append({"side": "buy" if dl.type == 0 else "sell", "price": dl.price, "bid": last["bid"],
                          "ask": last["ask"], "reason": int(dl.reason), "entry": int(dl.entry),
                          "lag_ms": int(dl.time_msc - last["time_msc"])})
        manifest_data = mt5_manifest_data(mt5, ["GOLD#"], start, end, len(m1), [spec["test"]],
                                          "MT5 history can be revised by the broker; demo account fills")
    finally:
        mt5.shutdown()
    spec = {**spec, "point": info.point}
    k = spec["trials_declared"] * trials
    level = 0.95 ** (1 / k)  # Sidak over the declared decisions and earlier experiments in the family
    rng = np.random.default_rng(spec["seed"])
    desc = describe(m1, spec)
    cm = cost_model(m1, spec, level, rng)
    an = anomaly(m1, spec, level, rng)
    field = field_semantics(ticks, m1, info.point)
    exe = execution(pd.DataFrame(fills), info.point)
    exe["skipped_no_quote_within_60s"] = skipped
    q1, q2, ready = decide(cm, an, field, exe, spec)
    units = {"point": info.point, "digits": info.digits, "spread_unit": "points (1 point = 0.01 USD per oz)",
             "spread_float": bool(info.spread_float), "tick_days_sampled": len(sample), "tick_rows": int(len(ticks))}
    summary = (f"Q2 (H-009 volatility clause; replay clause untested) {q2}: next-30m volatility ratio after "
               f"top-decile spread z {an['vol_ratio']} (CI {an['vol_ratio_ci']}, threshold {spec['vol_ratio_threshold']}; "
               f"excl. rollover {an['vol_ratio_excl_rollover']}, by year {an['vol_ratio_by_year']}, median-based "
               f"{an['vol_ratio_median_based']}), increment beyond recent variance {an['increment_log_rv']} log RV "
               f"(CI {an['increment_ci']}), anomaly share {an['anomaly_share_test']}; Q1 cost model {q1}: MAE ratio "
               f"hour-of-week/global {cm['mae_ratio_C1']} (CI {cm['mae_ratio_C1_ci']}, excl. rollover "
               f"{cm['mae_ratio_C1_excl_rollover']}), adaptive vs same-window global {cm['mae_ratio_C2']}; Q3 bar SPREAD field = {field['label']} of the minute's tick spreads "
               f"(tick mean / bar mean {field.get('tick_mean_over_bar_mean')}), demo fills {exe.get('fills')} with mean "
               f"adverse {exe.get('adverse_points', {}).get('mean')} points; H-013 cost input ready: {ready}")
    js = lambda x: json.dumps(x, ensure_ascii=False, indent=1, default=str)
    report = "\n".join([
        "# E-014 GOLD# spread data, cost model and spread anomalies (H-009)", "", f"Outcome (Q2): **{q2}**", "", summary, "",
        "Quoted spread (bar field, ticks) is not execution cost; demo fills are a demo account, not live. No strategy is "
        "evaluated: the H-009 replay clause (net R of filtered entries) is not tested here.", "",
        "## Units and collection", "", js(units), "", "## Spread by year (zero = missing)", "", js(desc["by_year"]), "",
        "## Sessions (2018-10+)", "", js(desc["sessions"]), "", "## By server hour", "", js(desc["by_hour"]), "",
        "## Rollover spike hour by season (timestamp consistency)", "", js(desc["spike_hour_by_season"]), "",
        f"Bars not minute-aligned: {desc['bars_not_minute_aligned_share']}; first valid spread bar {desc['first_valid_spread_bar']}", "",
        "## Q1 cost model (C1 vs train global median C0; C2 monthly re-estimated vs the same window's global median)",
        "", js(cm), "", "## Q2 spread anomalies (trailing 90-day norms, within-month ratios)", "", js(an), "", "## Q3 bar field vs ticks", "", js(field), "",
        "## Q3 demo fills vs tick quote", "", js(exe)])
    metrics = json.loads(js({"q1": q1, "q2": q2, "h013_cost_input_ready": ready, "units": units, "describe": desc,
                             "cost_model": cm, "anomaly": an, "field": field, "execution": exe, "ci_level": level,
                             "mean_r": "n/a"}))
    # supported here means only that the volatility clause survived; the replay clause of H-009 is untested
    outcome = {"vol_clause_not_refuted": "supported", "vol_clause_not_refuted_but_redundant": "supported",
               "refuted": "refuted"}.get(q2, "inconclusive")
    return {"outcome": outcome, "evidence_class": "development", "summary": summary,
            "failure_cause": "none" if outcome != "refuted" else "information",
            "lessons": f"Q1 {q1}; Q3 field {field['label']}; H-013 cost input ready {ready}",
            "next_questions": "H-009 replay clause (filtered entries' net R) and H-013 with the cost model recorded here",
            "metrics": metrics, "report": report, "manifest_data": manifest_data,
            "model_note": "hour-of-week medians and thresholds from the train window; no trading model"}
