"""E-008 (H-011, forecasting clause only): do HAR forecasts of GOLD# realised variance beat EWMA and simple
benchmarks, causally and out of sample? Layer 1 of 2: statistical accuracy only. Economic value is a separate
hypothesis (H-018) with its own experiment, so no trading, costs or sizing appear here.

Timing (server time, server midnight = rollover):
- A day is a server calendar day with >= MIN_BARS completed H1 bars; partial days (holidays) are not used as
  origins or targets. The day the script runs and the forming bar are dropped.
- RV_t = sum of squared H1 log returns of day t (each bar against the previous bar's close, so the first bar
  carries the overnight gap). Forecast origin = close of day t, using RV up to and including t only.
- Target for horizon h = mean RV over the next h complete days t+1..t+h (h = 1, 5, 22).

Models (same information set: the daily RV series; every model forecasts every h):
- RW: mean RV of the last h days (h = 1: yesterday's RV). No parameters.
- HIST: expanding mean of all RV up to t. No parameters.
- EWMA_094: f_t = 0.94 f_{t-1} + 0.06 RV_t (RiskMetrics constant on RV). No fitted parameters; flat over h.
- EWMA_FIT: same recursion, lambda in 0.80..0.99 (step 0.01) chosen by mean QLIKE on the estimation window, per h.
- HAR: target_h = b0 + bd RV_t + bw mean(RV_{t-4..t}) + bm mean(RV_{t-21..t}), per h on the estimation window,
  fit by QLIKE-type iteratively reweighted least squares (weights 1/f^2, from OLS start) so that HAR and
  EWMA_FIT are both estimated against the same loss (critic finding: OLS-vs-QLIKE favoured EWMA).

Estimation, two pre-declared protocols (neither touches an evaluation day's target):
- FROZEN (primary): parameters fit once on pairs whose whole target lies in 2014-01-01..2017-12-31, then fixed.
- RECURSIVE (secondary, real-time): refit at each month start on all pairs whose target ended before that
  month's first origin. Uses only the past, but parameters move, so it is reported separately from FROZEN.

Losses: QLIKE = y/f - log(y/f) - 1 (primary; robust to noise in the RV proxy, Patton 2011), MSE/RMSE (robust),
MAE (not robust to proxy noise; reported, never decisive). Diebold-Mariano on the loss difference with a
Newey-West variance (lag max(h, 5), covering overlap of multi-day targets; lag 22 reported as sensitivity).
Proxy check (descriptive only): forecasts rescaled by c = mean(Parkinson)/mean(RV) on the estimation window,
then scored against the Parkinson range proxy, which has no overnight gap and a different scale. Also
reported: QLIKE by weekday (Monday RV carries the weekend gap) and how many HAR forecasts hit the floor.
Stability: QLIKE ratio HAR/benchmark per calendar year and per volatility regime (tercile of the trailing
66-day RV, tercile cut points from the expanding past only).
"""
from datetime import datetime, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

import research_eval_v1 as ev
from research_run_v1 import mt5_manifest_data

MIN_BARS = 20
HORIZONS = (1, 5, 22)
LAMBDAS = np.round(np.arange(0.80, 0.995, 0.01), 2)
FLOOR = 1e-10  # forecasts are variances; HAR in levels can dip below zero
IRLS_STEPS = 10
MODELS = ("RW", "HIST", "EWMA_094", "EWMA_FIT", "HAR")


def daily_rv(h1):
    """Complete server days: RV from H1 log returns, Parkinson proxy from the day's high/low."""
    h1 = h1.sort_values("TIME_DT").reset_index(drop=True)
    r = np.log(h1["CLOSE"]).diff()
    day = h1["TIME_DT"].dt.normalize()
    g = pd.DataFrame({"day": day, "r2": r ** 2, "HIGH": h1["HIGH"], "LOW": h1["LOW"]}).iloc[1:].groupby("day")
    out = pd.DataFrame({"rv": g["r2"].sum(), "bars": g.size(),
                        "park": np.log(g["HIGH"].max() / g["LOW"].min()) ** 2 / (4 * np.log(2))})
    return out[out["bars"] >= MIN_BARS]


def design(rv):
    """Per origin t: HAR regressors and the h-day-ahead targets (NaN when the future is not complete)."""
    x = pd.DataFrame({"rv_d": rv, "rv_w": rv.rolling(5).mean(), "rv_m": rv.rolling(22).mean()}, index=rv.index)
    for h in HORIZONS:
        x[f"y{h}"] = rv[::-1].rolling(h).mean()[::-1].shift(-1)  # mean of t+1..t+h
    return x


def ewma_path(rv, lam):
    out, f = np.empty(len(rv)), rv.iloc[0]
    for i, v in enumerate(rv.to_numpy()):
        f = lam * f + (1 - lam) * v
        out[i] = f  # known at the close of day i
    return pd.Series(out, index=rv.index)


def qlike(y, f):
    ratio = y / np.maximum(f, FLOOR)
    return ratio - np.log(ratio) - 1


def fit(x, paths, h, train_mask):
    """HAR coefficients and EWMA lambda from the rows in train_mask (targets must be complete).
    paths: {lambda: EWMA path}; a path at t uses RV up to t only, so precomputing it leaks nothing."""
    rows = train_mask & x[[f"y{h}", "rv_m"]].notna().all(axis=1).to_numpy()
    a = np.column_stack([np.ones(rows.sum()), x.loc[rows, ["rv_d", "rv_w", "rv_m"]].to_numpy()])
    y = x.loc[rows, f"y{h}"].to_numpy()
    beta = np.linalg.lstsq(a, y, rcond=None)[0]
    for _ in range(IRLS_STEPS):  # QLIKE first-order condition ~ least squares weighted by 1/f^2
        w = 1 / np.maximum(a @ beta, np.median(y) * 0.05)
        beta = np.linalg.lstsq(a * w[:, None], y * w, rcond=None)[0]
    losses = {lam: float(np.mean(qlike(x.loc[rows, f"y{h}"], paths[lam][rows]))) for lam in LAMBDAS}
    return beta, min(losses, key=losses.get)


def forecasts(x, rv, paths, h, beta, lam):
    har = beta[0] + x[["rv_d", "rv_w", "rv_m"]].to_numpy() @ beta[1:]
    return pd.DataFrame({"RW": rv.rolling(h).mean(), "HIST": rv.expanding().mean(), "EWMA_094": paths[0.94],
                         "EWMA_FIT": paths[lam], "HAR": np.maximum(har, FLOOR)}, index=rv.index)


def frozen(x, rv, paths, h, est):
    """est = (start, end): only pairs whose whole target window ends before `end` are used."""
    days = x.index
    target_end = pd.Series(days, index=days).shift(-h)  # day t+h
    mask = ((days >= est[0]) & (target_end < est[1])).to_numpy()
    beta, lam = fit(x, paths, h, mask)
    return forecasts(x, rv, paths, h, beta, lam), {"beta": [round(float(b), 6) for b in beta], "lambda": float(lam)}


def recursive(x, rv, paths, h, start, est_start):
    """Refit at each month start from pairs whose target ended before that month's first origin."""
    days = x.index
    target_end = pd.Series(days, index=days).shift(-h)
    out = pd.DataFrame(index=days, columns=list(MODELS), dtype=float)
    for month, idx in pd.Series(np.arange(len(days)), index=days)[days >= start].groupby(days[days >= start].to_period("M")):
        first = days[idx.iloc[0]]
        mask = ((days >= est_start) & (target_end < first)).to_numpy()
        beta, lam = fit(x, paths, h, mask)
        out.iloc[idx.to_numpy()] = forecasts(x, rv, paths, h, beta, lam).iloc[idx.to_numpy()].to_numpy()
    return out


def dm(d, lag):
    """Diebold-Mariano t of mean(d) with a Newey-West variance; negative t = first model has lower loss."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n, u = len(d), d - d.mean()
    var = u @ u / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / n
    return float(d.mean() / np.sqrt(var / n)) if var > 0 else float("nan")


def regimes(rv, index):
    """Tercile of trailing 66-day RV, cut points from the expanding past only (no look-ahead)."""
    trail = rv.rolling(66).mean()
    lo, hi = trail.expanding(252).quantile(1 / 3).shift(1), trail.expanding(252).quantile(2 / 3).shift(1)
    lab = np.where(trail <= lo, "low", np.where(trail >= hi, "high", "mid"))
    return pd.Series(np.where(lo.isna(), "", lab), index=rv.index).reindex(index)


def score(fc, y, park_y, h, reg, park_scale):
    ok = y.notna() & fc.notna().all(axis=1)
    fc, y, park_y, reg = fc[ok], y[ok], park_y[ok], reg[ok]
    out = {"n": int(ok.sum()), "har_at_floor": int((fc["HAR"] <= FLOOR).sum())}
    for m in MODELS:
        e = y - fc[m]
        out[m] = {"qlike": round(float(qlike(y, fc[m]).mean()), 5), "mse": float((e ** 2).mean()),
                  "rmse": float(np.sqrt((e ** 2).mean())), "mae": float(e.abs().mean()),
                  "qlike_parkinson_scaled": round(float(qlike(park_y, fc[m] * park_scale).mean()), 5)}
    lag = max(h, 5)
    for b in ("EWMA_FIT", "EWMA_094", "RW", "HIST"):
        d = qlike(y, fc["HAR"]) - qlike(y, fc[b])
        years = d.groupby(d.index.year).mean()
        out[f"HAR_vs_{b}"] = {
            "qlike_ratio": round(out["HAR"]["qlike"] / out[b]["qlike"], 4), "dm_t": round(dm(d, lag), 2),
            "dm_p_one_sided": round(ev.sidak_p(-dm(d, lag), 1), 4),
            "dm_p_har_worse": round(ev.sidak_p(dm(d, lag), 1), 4),
            "dm_t_lag22": round(dm(d, 22), 2),
            "mse_dm_t": round(dm((y - fc["HAR"]) ** 2 - (y - fc[b]) ** 2, lag), 2),
            "years_har_better": int((years < 0).sum()), "years": int(len(years)),
            "parkinson_scaled_har_better": bool(out["HAR"]["qlike_parkinson_scaled"] < out[b]["qlike_parkinson_scaled"]),
            "by_weekday_qlike_ratio": {int(k): round(float(v), 4) for k, v in
                                       (qlike(y, fc["HAR"]).groupby(y.index.dayofweek).mean() /
                                        qlike(y, fc[b]).groupby(y.index.dayofweek).mean()).items()},
            "by_regime_qlike_ratio": {r: round(float(qlike(y[reg == r], fc["HAR"][reg == r]).mean() /
                                                     qlike(y[reg == r], fc[b][reg == r]).mean()), 4)
                                      for r in ("low", "mid", "high") if (reg == r).sum() > 20}}
    return out


def evaluate(h1, spec):
    """All tables for E-008; pure (no MT5), so it can be tested on synthetic bars."""
    days = daily_rv(h1)
    rv, x = days["rv"], design(days["rv"])
    park = days["park"]
    est = tuple(pd.Timestamp(d) for d in spec["estimation_window"])
    windows = [tuple(pd.Timestamp(d) for d in w) for w in spec["windows"]]
    reg = regimes(rv, x.index)
    paths = {lam: ewma_path(rv, lam) for lam in LAMBDAS}
    in_est = (days.index >= est[0]) & (days.index < est[1])
    park_scale = float(park[in_est].mean() / rv[in_est].mean())
    results, params = {}, {}
    for h in HORIZONS:
        park_y = park[::-1].rolling(h).mean()[::-1].shift(-1)
        fz, params[h] = frozen(x, rv, paths, h, est)
        rc = recursive(x, rv, paths, h, windows[0][0], est[0])
        for proto, fc in (("FROZEN", fz), ("RECURSIVE", rc)):
            for (a, b), lab in zip(windows, ("eval_2018_2023", "check_2024_on")):
                m = (x.index >= a) & (x.index < b)
                results[f"{proto}|h{h}|{lab}"] = score(fc[m], x[f"y{h}"][m], park_y[m], h, reg[m], park_scale)
    params["parkinson_scale"] = round(park_scale, 4)
    return results, params, days


def decide(results, spec):
    p = results["FROZEN|h1|eval_2018_2023"]
    fit_, rw, e94 = p["HAR_vs_EWMA_FIT"], p["HAR_vs_RW"], p["HAR_vs_EWMA_094"]
    if fit_["dm_p_har_worse"] < 0.05:  # refuted only when HAR is significantly worse, not merely not better
        return "refuted"
    if (fit_["qlike_ratio"] < 1 and fit_["dm_p_one_sided"] < 0.05 and e94["qlike_ratio"] < 1 and rw["qlike_ratio"] < 1
            and fit_["years_har_better"] >= spec["min_years_better"]
            and sum(v < 1 for v in fit_["by_regime_qlike_ratio"].values()) >= 2):
        return "supported"
    return "not_significant"


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
    h1["TIME_DT"] = pd.to_datetime(h1["time"], unit="s")  # server time, as the rest of the repo uses
    h1 = h1.rename(columns={"open": "OPEN", "high": "HIGH", "low": "LOW", "close": "CLOSE"}).iloc[:-1]
    h1 = h1[h1["TIME_DT"].dt.normalize() < h1["TIME_DT"].dt.normalize().max()]  # drop today's incomplete day
    results, params, days = evaluate(h1, spec)
    outcome = decide(results, spec)
    p = results["FROZEN|h1|eval_2018_2023"]["HAR_vs_EWMA_FIT"]
    summary = (f"{outcome}: FROZEN h=1 2018-2023 QLIKE HAR/EWMA_FIT {p['qlike_ratio']} (DM t {p['dm_t']}, "
               f"p {p['dm_p_one_sided']}), HAR better in {p['years_har_better']}/{p['years']} years, "
               f"regimes {p['by_regime_qlike_ratio']}; DM t lag 22 {p['dm_t_lag22']}; scaled-Parkinson agrees "
               f"{p['parkinson_scaled_har_better']} (descriptive)")
    rows = []
    for key, r in results.items():
        q = " | ".join(str(r[m]["qlike"]) for m in MODELS)
        c = r["HAR_vs_EWMA_FIT"]
        rows.append(f"| {key} | {r['n']} | {q} | {c['qlike_ratio']} | {c['dm_t']} | {c['mse_dm_t']} | "
                    f"{c['years_har_better']}/{c['years']} | {c['by_regime_qlike_ratio']} |")
    report = "\n".join([
        "# E-008 volatility forecasting, layer 1 (H-011)", "", f"Outcome: **{outcome}**", "", summary, "",
        "Statistical forecast accuracy only; says nothing about economic value (H-018).",
        "Evidence class: development (prices 2018+ were inspected before; these forecasts were not). The 2024+ "
        "rows are reported and were never used to choose a model.", "",
        "| protocol / horizon / window | days | QLIKE " + " | ".join(MODELS) + " | HAR/EWMA_FIT | DM t (QLIKE) | DM t (MSE) | years HAR better | by regime |",
        "|---|---|" + "---|" * len(MODELS) + "---|---|---|---|---|", *rows, "",
        "Primary-cell detail (FROZEN h=1 2018-2023, HAR vs EWMA_FIT): DM t lag 22 "
        f"{p['dm_t_lag22']}, by weekday {p['by_weekday_qlike_ratio']}, HAR at floor "
        f"{results['FROZEN|h1|eval_2018_2023']['har_at_floor']} days.", "",
        f"FROZEN parameters (fit {spec['estimation_window']}): {params}.",
        f"Days used: {len(days)} complete server days (>= {MIN_BARS} H1 bars). Full MSE/RMSE/MAE in metrics.json."])
    return {"outcome": "inconclusive" if outcome == "not_significant" else outcome, "evidence_class": "development",
            "summary": summary, "failure_cause": "none" if outcome == "supported" else "information",
            "lessons": f"HAR vs EWMA forecasting (layer 1): {outcome}",
            "next_questions": "H-018: does the better forecast change sizing/no-trade decisions net of costs?"
                              if outcome == "supported" else
                              "keep EWMA as the volatility input for H-014/H-018; test whether GVZ adds information",
            "metrics": {"results": results, "frozen_params": {str(k): v for k, v in params.items()},
                        "outcome_detail": outcome, "mean_r": "n/a"},
            "report": report, "manifest_data": manifest_data,
            "model_note": "HAR/EWMA parameters estimated inside the run on 2014-2017 (FROZEN) or the expanding past "
                          "(RECURSIVE); no model file is produced"}
