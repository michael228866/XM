"""E-011 (H-020, descriptive diagnostic): why did models with parameters fixed on 2014-2017 degrade afterwards
(E-008 frozen HAR, E-010 frozen R/A)? No verdict on any hypothesis and no claim of success: it splits the
degradation into five measurable parts so that future, forward-data experiments can test one explanation each.

Periods: P0 2014-2017 (where every frozen model was fit), P1 2018-2023, P2 2024+.
1. Target distribution shift: log daily RV and log 5-day mean RV per period; KS statistic vs P0.
2. Covariate shift: the same for the inputs (log EWMA_094, log RV22, log implied variance from GVZ, log premium
   = log IV - log EWMA); share of P1/P2 values outside P0's 1-99% range.
3. Calibration drift: per year, mean(y / f) of each frozen forecast (1 = calibrated in level; QLIKE-optimal
   scale) and Mincer-Zarnowitz log-regression intercept/slope per period.
4. Concept drift: per year, QLIKE of the frozen forecast (L_frozen), of the same forecast times that year's
   optimal scalar (L_scaled), and of the same model refit inside that year (L_refit, in-sample; a floor, not a
   usable forecast; only for E-010's R and A, HAR has no in-year refit). L_frozen - L_scaled = level
   miscalibration; L_scaled - L_refit = change in the shape of the relationship. With an intercept in
   exp(Xb), L_refit <= L_scaled by construction, so the shape part is always >= 0 and carries in-sample
   optimism. Both use data already seen; they describe, they do not validate anything. Forecast days at the
   HAR floor are excluded and counted.
5. Data quality shift: per year complete days, bars per day, share of raw H1 bars with SPREAD = 0, flat bars
   (high = low), median tick volume, repeated closes, GVZ staleness.
Frozen models are rebuilt exactly as preregistered in E-008/E-010 (same code, same estimation window); the
rebuilt coefficients and the training-row mean(y/f) of R/A (1 by the first-order condition) are reported so
that broker revisions of MT5 history would show up against the values recorded in the E-008/E-010 runs.
"""
from datetime import datetime, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

import exp_gvz_increment_v1 as gi
import exp_vol_forecast_v1 as vf
import research_data_gvz_v1 as gvzdata
from research_run_v1 import mt5_manifest_data

PERIODS = {"P0_2014_2017": ("2014-01-01", "2018-01-01"), "P1_2018_2023": ("2018-01-01", "2024-01-01"),
           "P2_2024_on": ("2024-01-01", "2099-01-01")}


def period_of(index):
    lab = pd.Series("", index=index)
    for k, (a, b) in PERIODS.items():
        lab[(index >= a) & (index < b)] = k
    return lab


def shift_table(series):
    """Per variable: period mean/sd/q05/q95, KS vs P0, share outside P0's 1-99% range."""
    out = {}
    for name, s in series.items():
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        lab = period_of(s.index)
        p0 = s[lab == "P0_2014_2017"]
        lo, hi = p0.quantile(0.01), p0.quantile(0.99)
        out[name] = {}
        for k in PERIODS:
            x = s[lab == k]
            if len(x) < 20:
                continue
            row = {"n": int(len(x)), "mean": round(float(x.mean()), 3), "sd": round(float(x.std()), 3),
                   "q05": round(float(x.quantile(0.05)), 3), "q95": round(float(x.quantile(0.95)), 3)}
            if k != "P0_2014_2017":
                row["ks_vs_P0"] = round(float(ks_2samp(x, p0).statistic), 3)
                row["outside_P0_1_99"] = round(float(((x < lo) | (x > hi)).mean()), 3)
            out[name][k] = row
    return out


def frozen_forecasts(h1, gvz):
    """Daily table with targets and the frozen forecasts of E-008 (HAR, h=1) and E-010 (R, A; general h=5)."""
    days = vf.daily_rv(h1)
    rv, x = days["rv"], vf.design(days["rv"])
    est = (pd.Timestamp("2014-01-01"), pd.Timestamp("2018-01-01"))
    paths = {lam: vf.ewma_path(rv, lam) for lam in vf.LAMBDAS}
    har, har_params = vf.frozen(x, rv, paths, 1, est)
    params = {"HAR_h1": har_params}
    qs, e94, iv = gi.questions(rv, x, gvz)
    q = qs["general"]
    target_end = pd.Series(x.index, index=x.index).shift(-5)
    ok = q["t"].notna() & (q["t"] > 0) & pd.concat(q["regs"].values(), axis=1).notna().all(axis=1) & (x.index >= est[0])
    train = ok & (target_end < est[1])
    fc = {}
    for m in ("R", "A"):
        X = np.column_stack([np.ones(len(x)), q["regs"][m].to_numpy()])
        beta = gi.fit("qlike", X[train.to_numpy()], q["t"][train].to_numpy())[0]
        fc[m] = pd.Series(np.exp(X @ beta), index=x.index)
        params[f"{m}_h5"] = {"beta": [round(float(b), 4) for b in beta],
                             "train_mean_y_over_f": round(float((q["t"][train] / fc[m][train]).mean()), 4)}
    table = pd.DataFrame({"y1": x["y1"], "y5": x["y5"], "HAR_h1": har["HAR"], "R_h5": fc["R"], "A_h5": fc["A"],
                          "log_ewma": np.log(e94), "log_rv22": np.log(x["rv_m"]), "log_iv": np.log(iv),
                          "log_rv": np.log(rv)})
    table["log_premium"] = table["log_iv"] - table["log_ewma"]
    return table, q["regs"], params


def calibration_and_concept(table, regs):
    """Per year and forecast: mean(y/f), L_frozen, L_scaled, L_refit (QLIKE)."""
    out = {}
    for name, ycol, refit_cols in (("HAR_h1", "y1", None), ("R_h5", "y5", "R"), ("A_h5", "y5", "A")):
        d = table[[ycol, name]].dropna()
        d = d[d[ycol] > 0]
        at_floor = d[name] <= vf.FLOOR * 10
        d = d[~at_floor]
        out[name] = {"floor_days_excluded": int(at_floor.sum())}
        for year, g in d.groupby(d.index.year):
            if len(g) < 100:
                continue
            y, f = g[ycol].to_numpy(), g[name].to_numpy()
            k = float(np.mean(y / f))
            row = {"n": int(len(g)), "mean_y_over_f": round(k, 3),
                   "L_frozen": round(float(vf.qlike(y, f).mean()), 4), "L_scaled": round(float(vf.qlike(y, k * f).mean()), 4)}
            if refit_cols:
                X = np.column_stack([np.ones(len(g)), regs[refit_cols].loc[g.index].to_numpy()])
                ok = np.isfinite(X).all(axis=1)
                beta = gi.fit("qlike", X[ok], y[ok])[0]
                row["L_refit_in_year"] = round(float(vf.qlike(y[ok], np.exp(X[ok] @ beta)).mean()), 4)
            out[name][int(year)] = row
        lab = period_of(d.index)
        out[name]["mincer_zarnowitz_log"] = {}
        for k in PERIODS:
            g = d[lab == k]
            if len(g) > 100:
                b = np.polyfit(np.log(g[name]), np.log(g[ycol]), 1)
                out[name]["mincer_zarnowitz_log"][k] = {"slope": round(float(b[0]), 3), "intercept": round(float(b[1]), 3)}
    return out


def data_quality(raw, gvz, days_index):
    raw = raw.assign(day=raw["TIME_DT"].dt.normalize(), year=raw["TIME_DT"].dt.year)
    per_day = raw.groupby("day").size()
    g = raw.groupby("year")
    complete = (per_day >= vf.MIN_BARS).groupby(per_day.index.year)
    stale = gvzdata.align(gvz, days_index)["stale_days"]
    return {int(y): {"bars": int(len(r)), "days": int(per_day[per_day.index.year == y].size),
                     "complete_day_share": round(float(complete.mean()[y]), 3),
                     "bars_per_day": round(float(per_day[per_day.index.year == y].mean()), 2),
                     "spread_zero_share": round(float((r["spread"] == 0).mean()), 3),
                     "flat_bar_share": round(float((r["HIGH"] == r["LOW"]).mean()), 4),
                     "median_tick_volume": float(r["tick_volume"].median()),
                     "repeated_close_share": round(float((r["CLOSE"].diff() == 0).mean()), 4),
                     "gvz_mean_stale_days": round(float(stale[stale.index.year == y].mean()), 2)
                     if (stale.index.year == y).any() else None}
            for y, r in g}


def evaluate(h1, gvz):
    table, regs, params = frozen_forecasts(h1, gvz)
    shifts = shift_table({"log_rv": table["log_rv"], "log_y5": np.log(table["y5"])})
    covs = shift_table({k: table[k] for k in ("log_ewma", "log_rv22", "log_iv", "log_premium")})
    cc = calibration_and_concept(table, regs)
    return {"target_shift": shifts, "covariate_shift": covs, "calibration_concept": cc, "rebuilt_params": params,
            "data_quality": data_quality(h1, gvz, table.index)}


def run(run_dir, spec, trials):
    start = datetime(2013, 10, 1, tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)
    if not mt5.initialize(path=spec.get("terminal", r"D:\XM2\terminal64.exe")):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        h1 = pd.DataFrame(mt5.copy_rates_range("GOLD#", mt5.TIMEFRAME_H1, start, end))
        manifest_data = mt5_manifest_data(mt5, ["GOLD#"], start, end, len(h1), spec["windows"],
                                          "MT5 history can be revised by the broker")
    finally:
        mt5.shutdown()
    manifest_data["data_sources"].append(f"FRED GVZCLS snapshot sha256 {gvzdata.sha256(gvzdata.DATA)}")
    h1["TIME_DT"] = pd.to_datetime(h1["time"], unit="s")
    h1 = h1.rename(columns={"open": "OPEN", "high": "HIGH", "low": "LOW", "close": "CLOSE"}).iloc[:-1]
    h1 = h1[h1["TIME_DT"].dt.normalize() < h1["TIME_DT"].dt.normalize().max()]
    out = evaluate(h1, gvzdata.load())
    cc = out["calibration_concept"]
    lines = ["# E-011 fixed-parameter drift diagnostic (H-020, descriptive)", "",
             "Outcome: **inconclusive** (descriptive by design; no hypothesis is decided).", "",
             "Everything below uses data already inspected; refits are descriptive floors, not forecasts.", "",
             "## 1-2 Target and covariate shift (vs P0 2014-2017)", "",
             "| variable | period | n | mean | sd | q05 | q95 | KS vs P0 | outside P0 1-99% |", "|---|---|---|---|---|---|---|---|---|"]
    for block in ("target_shift", "covariate_shift"):
        for var, by in out[block].items():
            for k, r in by.items():
                lines.append(f"| {var} | {k} | {r['n']} | {r['mean']} | {r['sd']} | {r['q05']} | {r['q95']} | "
                             f"{r.get('ks_vs_P0', '')} | {r.get('outside_P0_1_99', '')} |")
    lines += ["", "## 3-4 Calibration vs concept drift (QLIKE per year)", "",
              "| forecast | year | n | mean y/f | L_frozen | L_scaled | L_refit_in_year |", "|---|---|---|---|---|---|---|"]
    for name, by in cc.items():
        for y, r in by.items():
            if isinstance(y, int):
                lines.append(f"| {name} | {y} | {r['n']} | {r['mean_y_over_f']} | {r['L_frozen']} | {r['L_scaled']} | "
                             f"{r.get('L_refit_in_year', '')} |")
        lines.append(f"| {name} | Mincer-Zarnowitz (log) | {by['mincer_zarnowitz_log']} | | | | |")
        lines.append(f"| {name} | floor days excluded | {by['floor_days_excluded']} | | | | |")
    lines += ["", "## 5 Data quality by year", "", "| year | " + " | ".join(next(iter(out["data_quality"].values()))) + " |",
              "|---|" + "---|" * len(next(iter(out["data_quality"].values())))]
    lines += [f"| {y} | " + " | ".join(str(v) for v in r.values()) + " |" for y, r in out["data_quality"].items()]
    lines += ["", f"Rebuilt frozen parameters (compare with the E-008/E-010 runs): {out['rebuilt_params']}",
              "L_refit <= L_scaled by construction (in-year, in-sample); the shape part is an optimistic floor.",
              "", "Testable follow-ups must use data after 2026-11-12 (see the preregistration's claim scope)."]
    summary = "descriptive diagnostic of fixed-parameter drift; see report tables (no verdict)"
    return {"outcome": "inconclusive", "evidence_class": "development", "summary": summary, "failure_cause": "none",
            "lessons": "drift decomposition recorded for E-008/E-010 frozen models; interpretation goes to observations",
            "next_questions": "pick the dominant measurable part and preregister a forward-data test for it",
            "metrics": {**out, "mean_r": "n/a"}, "report": "\n".join(lines), "manifest_data": manifest_data,
            "model_note": "rebuilds E-008/E-010 frozen forecasts (2014-2017 fit) and descriptive in-year refits; no model file"}
