"""E-009 (H-018): with the entry signal fixed and no new direction forecast, does EWMA_094 volatility information
improve position sizing, i.e. net risk-adjusted return, drawdown and tail loss of the same trades?

Signal and trades (identical in every arm): the frozen H1 C model (frozen/h1_C_long_v1, threshold from its
config), long at the bar open (ask), SL 1.5 ATR / TP 3.0 ATR / 48 h, one position at a time, spread from the
lab frame (bar spread floored at 0.008% of price). H1 C was fit on 2014-2022 and its threshold chosen on 2023,
so 2018-2023 is in sample (descriptive only) and 2024+ is the judged window. 2024+ is not a clean holdout:
H1 C was picked among candidates after its 2024+ test was seen, so the evidence is development class.
first_touch fills a gap through the stop at the stop price, which understates tail losses in every arm alike.

Arms differ only in the notional w (fraction of equity) fixed at entry; every arm uses the same leverage cap:
- FIXED_RISK (baseline): lose RISK_PER_TRADE of equity at the stop, w = b * entry / (1.5 ATR). This is
  already inverse-volatility sizing on the H1 ATR (14 hourly bars, so it carries intraday seasonality).
- EWMA_RISK (primary): FIXED_RISK times sigma_ref / sigma, sigma = sqrt(EWMA_094 of daily RV as of the last
  complete server day before entry), sigma_ref = its median on 2018-2023 signal bars. Nested: the same ATR
  stop sizing plus only the daily EWMA scaling, so the comparison asks whether EWMA adds information beyond
  the ATR (critic B1: replacing ATR by EWMA mostly re-weights by entry hour).
- EWMA_TARGET (descriptive): pure target volatility, w = sigma_target / sigma, sigma_target set so its median
  notional equals FIXED_RISK's on 2018-2023 signal bars (entry information only).
- CONSTANT_NOTIONAL (descriptive): one constant notional (median FIXED_RISK notional on 2018-2023), like the
  runner's fixed lot; the no-volatility-information anchor. EWMA_TARGET vs CONSTANT_NOTIONAL is reported with
  its own CI as "daily vol information vs none".
Ex ante (no outcomes) the report gives the R^2 of log(w_arm / w_FIXED) on entry-hour dummies, i.e. how much of
each comparison is only intraday re-weighting.

Daily P&L: positions marked to the bid close of each H1 bar, exit at the stop/target/timeout price; swap charged
at each server rollover crossed (Wednesday triple) at today's rate as a fraction of notional; returns summed per
complete server day, zero when flat (non-compounding). Swap: the account's actual swap is the base case. The
XM demo account reports GOLD# swap_mode DISABLED and charged 0 swap on every position that crossed a rollover
(O-016), so base swap = 0; the quoted swap_long (as points) x 1.5 is a stress scenario. Cost scenarios: base,
spread x2, quoted swap x1.5.
Raw metrics use each arm's own budget; "vol-matched" rescales an arm ex post to FIXED_RISK's realised daily
volatility in the same window, so drawdown/tail gains from simply holding less cannot pass as skill
(raw max_dd_pct compounds, vol-matched drawdown is a non-compounded sum).
"""
import math
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

import h1_runner_v1
import research_eval_v1 as ev
from exp_vol_forecast_v1 import daily_rv, ewma_path, regimes
from exp_vol_managed_v2 import sharpe_diff_ci
from research_run_v1 import mt5_manifest_data
from strategy_lab_v1 import PRICE_PER_POINT, first_touch, trade

ARMS = ("FIXED_RISK", "EWMA_RISK", "EWMA_TARGET", "CONSTANT_NOTIONAL")
SCENARIOS = {"base": (1.0, "account", 1.0), "spread_x2": (2.0, "account", 1.0), "swap_quoted_x1.5": (1.0, "quoted", 1.5)}


def entry_sigma(frame):
    """sqrt(EWMA_094 daily RV) known before each bar: value at the last complete server day before the bar's day."""
    rv = daily_rv(frame)["rv"]
    path = ewma_path(rv, 0.94)
    day = frame["TIME_DT"].dt.normalize().to_numpy()
    pos = np.searchsorted(path.index.to_numpy(), day, side="left") - 1  # strictly earlier day
    out = np.where(pos >= 0, np.sqrt(path.to_numpy()[np.maximum(pos, 0)]), np.nan)
    return out, rv


def trades(f, signal, spec, spread_mult):
    """Taken trades inside the contiguous slice f; column i is the bar's label (= position in the full frame)."""
    f = f.assign(SPREAD=f["SPREAD"] * spread_mult)
    r, after = first_touch(f, "long", spec["tp"], spec["sl"], spec["hold"])
    taken = trade(f, np.asarray(signal, float), 0.5, r, after)
    p = f.index.get_indexer(taken.index)
    spread = f["SPREAD"].to_numpy()[p] * PRICE_PER_POINT
    entry = f["OPEN"].to_numpy()[p] + spread
    sl_dist = f["ATR"].to_numpy()[p] * spec["sl"]
    t = pd.DataFrame({"i": taken.index.to_numpy(), "after": after[p].astype(int), "r": r[p], "entry": entry,
                      "sl_dist": sl_dist, "exit": entry + r[p] * sl_dist, "spread": spread})
    return t[np.isfinite(t["r"])].reset_index(drop=True)


def weights(t, sigma, spec, calib):
    fixed, s = spec["risk_per_trade"] * t["entry"].to_numpy() / t["sl_dist"].to_numpy(), sigma[t["i"].to_numpy()]
    w = {"FIXED_RISK": fixed, "EWMA_RISK": fixed * calib["sigma_ref"] / s, "EWMA_TARGET": calib["sigma_target"] / s,
         "CONSTANT_NOTIONAL": np.full(len(t), calib["constant_w"])}
    cap = spec["leverage_cap"]
    return ({k: pd.Series(np.minimum(v, cap)) for k, v in w.items()}, {k: int((v > cap).sum()) for k, v in w.items()})


def daily_pnl(frame, t, w, swap_frac, swap_mult):
    """Per server day: summed P&L (fraction of equity) of trades marked at bar closes; per-trade P&L; total cost
    (entry spread + swap) as a fraction of equity."""
    close, day = frame["CLOSE"].to_numpy(), frame["TIME_DT"].dt.normalize().to_numpy()
    pnl = pd.Series(0.0, index=pd.DatetimeIndex(np.unique(day)))
    per_trade = np.zeros(len(t))
    cost = -float((w.to_numpy() * t["spread"].to_numpy() / t["entry"].to_numpy()).sum())
    for n, row in enumerate(t.itertuples()):
        prev, wn = row.entry, float(w.iloc[n])
        for k in range(row.i, row.i + row.after):
            v = row.exit if k == row.i + row.after - 1 else close[k]
            x = wn * (v - prev) / row.entry
            if k > row.i and day[k] != day[k - 1]:  # crossed a rollover; Wednesday night is charged triple
                nights = 3 if pd.Timestamp(day[k - 1]).dayofweek == 2 else 1
                x += wn * swap_frac * swap_mult * nights
                cost += wn * swap_frac * swap_mult * nights
            pnl[day[k]] += x
            per_trade[n] += x
            prev = v
    return pnl, per_trade, cost


def nw_alpha_t(y, X, lag=5):
    """OLS of y on [1, X]; Newey-West t of the intercept."""
    A = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    u = (y - A @ beta)[:, None] * A
    S = u.T @ u
    for k in range(1, lag + 1):
        g = u[k:].T @ u[:-k]
        S += (1 - k / (lag + 1)) * (g + g.T)
    inv = np.linalg.inv(A.T @ A)
    se = np.sqrt(np.diag(inv @ S @ inv))
    return float(beta[0]), float(beta[0] / se[0]) if se[0] > 0 else float("nan"), [round(float(b), 4) for b in beta[1:]]


def arm_metrics(ret, ref_std, per_trade, w, cost):
    s = ret.std(ddof=1)
    vm = ret * (ref_std / s) if s > 0 else ret
    eq = vm.cumsum()
    mdd_vm = float((eq - eq.cummax()).min())
    q = vm.quantile(0.05)
    top = np.sort(np.abs(per_trade))[::-1]
    month = ret.groupby(ret.index.to_period("M")).sum()
    share = (month.abs() / month.abs().sum()) if month.abs().sum() > 0 else month
    return {**ev.daily_stats(ret.to_numpy(), ret.index),
            "vol_matched_max_dd_pct": round(mdd_vm * 100, 3), "vol_matched_cvar5_bp": round(float(vm[vm <= q].mean()) * 1e4, 3),
            "worst_day_bp": round(float(ret.min()) * 1e4, 3),
            "avg_w": round(float(np.mean(w)), 3), "max_w": round(float(np.max(w)), 3),
            "return_per_unit_notional_bp": round(float(ret.sum() / max(np.sum(w), 1e-12)) * 1e4, 3),
            "cost_bp_total": round(float(cost) * 1e4, 2),
            "top10pct_trades_abs_pnl_share": round(float(top[:max(1, len(top) // 10)].sum() / max(top.sum(), 1e-12)), 3),
            "month_pnl_herfindahl": round(float((share ** 2).sum()), 4)}


def window_eval(frame, signal, spec, sigma, rv, calib, swap_frac, a, b):
    m = ((frame["TIME_DT"] >= a) & (frame["TIME_DT"] < b)).to_numpy()
    f = frame[m]
    sig = np.asarray(signal)[m]
    out = {}
    for sc, (spread_mult, swap_source, swap_mult) in SCENARIOS.items():
        t = trades(f, sig, spec, spread_mult)
        w, capped = weights(t, sigma, spec, calib)
        series, per_trade, cost = {}, {}, {}
        for arm in ARMS:
            series[arm], per_trade[arm], cost[arm] = daily_pnl(frame, t, w[arm], swap_frac[swap_source], swap_mult)
        days = series["FIXED_RISK"].index
        days = days[(days >= a) & (days < b)]
        series = {k: v.reindex(days, fill_value=0.0) for k, v in series.items()}
        ref_std = series["FIXED_RISK"].std(ddof=1)
        trade_months = frame["TIME_DT"].iloc[t["i"].to_numpy()].dt.to_period("M").nunique()
        r = {"trades": int(len(t)), "months_with_trades": int(trade_months), "capped": capped,
             "arms": {arm: arm_metrics(series[arm], ref_std, per_trade[arm], w[arm], cost[arm]) for arm in ARMS}}
        e, fx = series["EWMA_RISK"], series["FIXED_RISK"]
        sr = lambda x: float(x.mean() / x.std(ddof=1) * math.sqrt(252)) if x.std(ddof=1) > 0 else float("nan")
        r["delta_sharpe"] = round(sr(e) - sr(fx), 3)
        if sc == "base":
            r["delta_sharpe_ci"] = sharpe_diff_ci(e, fx, days, seed=spec["seed"], draws=spec["bootstrap_draws"])
            hour = pd.get_dummies(frame["TIME_DT"].iloc[t["i"].to_numpy()].dt.hour).to_numpy(float)
            r["hour_r2_log_weight_ratio"] = {arm: hour_r2(np.log(w[arm].to_numpy() / w["FIXED_RISK"].to_numpy()), hour)
                                             for arm in ("EWMA_RISK", "EWMA_TARGET", "CONSTANT_NOTIONAL")}
            for name, (x1, x2) in {"EWMA_TARGET_vs_FIXED_RISK": ("EWMA_TARGET", "FIXED_RISK"),
                                   "EWMA_TARGET_vs_CONSTANT_NOTIONAL": ("EWMA_TARGET", "CONSTANT_NOTIONAL")}.items():
                r[f"descriptive_{name}"] = {"delta_sharpe": round(sr(series[x1]) - sr(series[x2]), 3),
                                            "ci": sharpe_diff_ci(series[x1], series[x2], days, seed=spec["seed"],
                                                                 draws=spec["bootstrap_draws"])}
            close = frame.groupby(frame["TIME_DT"].dt.normalize())["CLOSE"].last()
            mkt = close.pct_change().reindex(days).fillna(0.0)
            a_, t_, betas = nw_alpha_t(e.to_numpy(), np.column_stack([fx.to_numpy(), mkt.to_numpy()]))
            r["alpha_on_fixed_and_market"] = {"alpha_bp": round(a_ * 1e4, 3), "nw_t": round(t_, 2), "betas": betas}
            reg = regimes(rv, rv.index).shift(1).reindex(days)  # label known at the start of the day
            r["by_regime_sharpe"] = {g: {"days": int((reg == g).sum()), "EWMA_RISK": round(sr(e[reg == g]), 3),
                                         "FIXED_RISK": round(sr(fx[reg == g]), 3)}
                                     for g in ("low", "mid", "high") if (reg == g).sum() > 1}
            r["by_year_sharpe"] = {int(y): {"EWMA_RISK": round(sr(e[e.index.year == y]), 3),
                                            "FIXED_RISK": round(sr(fx[fx.index.year == y]), 3)}
                                   for y in sorted(set(days.year))}
        out[sc] = r
    return out


def calibrate(frame, signal, spec, sigma, a, b):
    """Ex-ante budget match on signal bars in [a, b): entry information only (price, ATR, sigma), no outcomes."""
    m = ((frame["TIME_DT"] >= a) & (frame["TIME_DT"] < b)).to_numpy() & np.asarray(signal, bool) & np.isfinite(sigma)
    entry = frame["OPEN"].to_numpy()[m] + frame["SPREAD"].to_numpy()[m] * PRICE_PER_POINT
    w_fixed = spec["risk_per_trade"] * entry / (frame["ATR"].to_numpy()[m] * spec["sl"])
    return {"sigma_target": float(np.median(w_fixed) * np.median(sigma[m])), "constant_w": float(np.median(w_fixed)),
            "sigma_ref": float(np.median(sigma[m])), "bars": int(m.sum())}


def hour_r2(y, dummies):
    """Share of the variance of y explained by entry-hour dummies (ex ante: entry information only)."""
    fit = dummies @ np.linalg.lstsq(dummies, y, rcond=None)[0]
    v = np.var(y)
    return round(float(1 - np.var(y - fit) / v), 3) if v > 0 else 0.0


def evaluate(frame, signal, spec, swap_frac):
    """Pure (no MT5): all windows and scenarios."""
    frame = frame.reset_index(drop=True)
    sigma, rv = entry_sigma(frame)
    calib = calibrate(frame, signal, spec, sigma, *[pd.Timestamp(d) for d in spec["calibration_window"]])
    results = {}
    for lab, (a, b) in spec["labelled_windows"].items():
        results[lab] = window_eval(frame, signal, spec, sigma, rv, calib, swap_frac, pd.Timestamp(a), pd.Timestamp(b))
    return results, calib


def decide(results, spec):
    p = results["oos_2024_on"]
    base = p["base"]
    if base["trades"] < spec["min_trades"] or base["months_with_trades"] < spec["min_months"]:
        return "inconclusive_sample"
    lo, hi = base["delta_sharpe_ci"]
    if hi < 0:
        return "refuted"
    e, f = base["arms"]["EWMA_RISK"], base["arms"]["FIXED_RISK"]
    regs = [v for v in base["by_regime_sharpe"].values() if v["days"] >= spec["min_regime_days"]]
    years = base["by_year_sharpe"].values()
    wins = sum(v["EWMA_RISK"] >= v["FIXED_RISK"] for v in regs)
    regime_ok = wins >= len(regs) - 1 if len(regs) >= 2 else len(regs) == 1 and wins == 1
    if lo > 0:
        robust = (all(p[s]["delta_sharpe"] > 0 for s in ("spread_x2", "swap_quoted_x1.5"))
                  and e["vol_matched_max_dd_pct"] >= f["vol_matched_max_dd_pct"]
                  and e["vol_matched_cvar5_bp"] >= f["vol_matched_cvar5_bp"]
                  and regime_ok and sum(v["EWMA_RISK"] >= v["FIXED_RISK"] for v in years) >= 2)
        return "supported" if robust else "significant_not_robust"  # never not_significant (critic B2)
    mdd_cut = (1 - e["vol_matched_max_dd_pct"] / f["vol_matched_max_dd_pct"]) if f["vol_matched_max_dd_pct"] < 0 else 0.0
    if hi < spec["materiality_sharpe"] and mdd_cut < spec["materiality_mdd_cut"]:
        return "not_significant"
    return "underpowered"


def run(run_dir, spec, trials):
    config, lab, model = h1_runner_v1.load_frozen()  # verifies frozen hashes; does not start the runner
    if any(spec[k] != config["spec"][k] for k in ("tp", "sl", "hold")) or config["spec"]["side"] != "long":
        raise SystemExit(f"spec exits differ from the frozen H1 C config {config['spec']}")
    end = datetime.now(timezone.utc)
    start = datetime(2017, 1, 1, tzinfo=timezone.utc)
    if not mt5.initialize(path=spec.get("terminal", r"D:\XM2\terminal64.exe")):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        h1 = lab.copy_rates("H1", start - timedelta(days=60), end)
        higher = {tf: lab.copy_rates(tf, start - timedelta(days=900), end) for tf in lab.HIGHER}
        info = mt5.symbol_info("GOLD#")
        if info.swap_mode not in (mt5.SYMBOL_SWAP_MODE_DISABLED, mt5.SYMBOL_SWAP_MODE_POINTS):
            raise RuntimeError(f"swap_mode {info.swap_mode}: only DISABLED and POINTS are handled")
        manifest_data = mt5_manifest_data(mt5, ["GOLD#"], start - timedelta(days=900), end, len(h1),
                                          spec["windows"], "MT5 history can be revised by the broker")
    finally:
        mt5.shutdown()
    frame = lab.make_features(h1.iloc[:-1], higher)  # drop the bar still forming
    frame = frame[frame["TIME_DT"].dt.normalize() < frame["TIME_DT"].dt.normalize().max()]  # drop today's day
    frame = frame.reset_index(drop=True)
    signal = model.predict_proba(frame[config["features"]].astype(np.float32))[:, 1] >= config["threshold"]
    quoted = info.swap_long * info.point / float(frame["CLOSE"].iloc[-1])  # per night, fraction of notional
    swap_frac = {"account": 0.0 if info.swap_mode == mt5.SYMBOL_SWAP_MODE_DISABLED else quoted, "quoted": quoted}
    results, calib = evaluate(frame, signal, spec, swap_frac)
    verdict = decide(results, spec)
    outcome = {"supported": "supported", "refuted": "refuted"}.get(verdict, "inconclusive")
    b = results["oos_2024_on"]["base"]
    e, f = b["arms"]["EWMA_RISK"], b["arms"]["FIXED_RISK"]
    summary = (f"{verdict}: 2024+ {b['trades']} trades / {b['months_with_trades']} months with trades; Sharpe "
               f"EWMA_RISK {e.get('sharpe')} vs "
               f"FIXED_RISK {f.get('sharpe')} (delta {b['delta_sharpe']}, 95% month-block CI {b['delta_sharpe_ci']}); "
               f"vol-matched MDD {e['vol_matched_max_dd_pct']}% vs {f['vol_matched_max_dd_pct']}%, CVaR5 "
               f"{e['vol_matched_cvar5_bp']} vs {f['vol_matched_cvar5_bp']} bp; alpha on FIXED+market "
               f"{b['alpha_on_fixed_and_market']}; delta Sharpe spread x2 {results['oos_2024_on']['spread_x2']['delta_sharpe']}, "
               f"quoted swap x1.5 {results['oos_2024_on']['swap_quoted_x1.5']['delta_sharpe']}; descriptive EWMA_TARGET vs FIXED_RISK "
               f"{b['descriptive_EWMA_TARGET_vs_FIXED_RISK']}, vs CONSTANT_NOTIONAL "
               f"{b['descriptive_EWMA_TARGET_vs_CONSTANT_NOTIONAL']}; hour R^2 of weight ratios {b['hour_r2_log_weight_ratio']}")
    keys = ("days", "net_mean_bp", "annual_pct", "sharpe", "max_dd_pct", "calmar", "vol_matched_max_dd_pct",
            "vol_matched_cvar5_bp", "worst_day_bp", "avg_w", "max_w", "return_per_unit_notional_bp", "cost_bp_total",
            "top10pct_trades_abs_pnl_share", "month_pnl_herfindahl")
    rows = [f"| {lab_} | {sc} | {arm} | {r['trades']} | " + " | ".join(str(r["arms"][arm].get(k, "")) for k in keys) + " |"
            for lab_, by in results.items() for sc, r in by.items() for arm in ARMS]
    report = "\n".join([
        "# E-009 volatility-managed sizing on fixed H1 C trades (H-018)", "", f"Outcome: **{verdict}**", "", summary, "",
        "Primary: EWMA_RISK vs FIXED_RISK, oos_2024_on / base. in_sample_2018_2023 is descriptive (H1 C signal fit "
        "on 2014-2023). EWMA_TARGET and CONSTANT_NOTIONAL are descriptive. The delta-Sharpe CI is a percentile "
        "month-block bootstrap over about 34 months, slightly narrow (about 5% per tail instead of 2.5% in the "
        "critic's simulation).",
        "Evidence class: development (2024+ prices were inspected in earlier experiments).", "",
        "| window | scenario | arm | trades | " + " | ".join(keys) + " |", "|---|---|---|---|" + "---|" * len(keys), *rows, "",
        f"By regime (2024+, base): {b['by_regime_sharpe']}", f"By year (2024+, base): {b['by_year_sharpe']}",
        f"Leverage cap binding (2024+, base): {b['capped']}", f"Calibration (2018-2023 signal bars, entry info only): {calib}",
        f"Swap per night (fraction of notional): account {swap_frac['account']:.6f}, quoted {swap_frac['quoted']:.6f}"])
    return {"outcome": outcome, "evidence_class": "development", "summary": summary,
            "failure_cause": "none" if verdict == "supported" else "information" if verdict in ("refuted", "not_significant")
            else "method" if verdict == "significant_not_robust" else "data",  # underpowered / too few trades
            "lessons": f"EWMA_094 scaling on top of fixed-risk sizing, fixed H1 C trades: {verdict}",
            "next_questions": "accumulate forward H1 C trades before any runner change" if verdict == "supported" else
                              "if no sizing value: turn to questions with a different economic mechanism (H-008/H-013/H-009)",
            "metrics": {"results": results, "calibration": calib, "verdict": verdict, "mean_r": "n/a"},
            "report": report, "manifest_data": manifest_data,
            "model_note": "frozen H1 C model evaluated, not trained (frozen/h1_C_long_v1); sizing has no fitted "
                          "parameters except sigma_target/constant_w set on 2018-2023 entry information"}
