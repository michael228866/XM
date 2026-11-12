"""E-010 (H-014, volatility clause only): does implied volatility (GVZ) add information about future GOLD#
realised variance beyond the EWMA_094 price-history benchmark? Statistical only; no trading, sizing or costs.
The return-prediction clause of H-014 (premium vs 20-day return) is not tested here.

Information increment, not lower error: every question compares two models fit the same way on the same rows,
the restricted one using price history only (EWMA_094) and the augmented one adding GVZ. A recalibration gain
therefore cannot be credited to GVZ. EWMA_094 is the development benchmark from E-008 (chosen on 2018-2023,
which if anything favours the restricted side).

Timing: as E-008 (complete server days, origin = close of day t, targets over t+1..t+h). GVZ enters through
research_data_gvz_v1.align: latest GVZ dated <= t - 1 calendar day (lag 1); lag 2 = one more observation back.
IV_t = (GVZ/100)^2 / 252 (daily implied variance).

Three pre-declared questions (one primary test each, Sidak over 3):
- general   (h=5):  QLIKE of f = exp(X b); R: [1, log EWMA]; A: R + log IV.
- extreme   (h=5):  P(mean RV over t+1..t+5 > q90_t), q90_t = expanding 90th percentile of past 5-day means whose
                    window ended by t; logistic; R: [1, log EWMA, log q90]; A: R + log IV. Log loss.
- regime_up (h=22): P(mean RV over t+1..t+22 >= 1.5 x trailing 22-day RV); logistic; R: [1, log EWMA, log RV22];
                    A: R + log IV. Log loss.
Models are nested: R's coefficients are all free, and A adds exactly one column, so a gain can only come from
GVZ (an earlier draft tied R's coefficients and A unlocked a price variable; critic B1). Each question also has
P (R plus one more price variable: log RV22, log RV22, log RV5 respectively; a point-estimate check that the gain
is not just one extra parameter) and A_lag2 (GVZ one observation older).
A refuted question means the frozen GVZ model did worse out of sample (e.g. drift of the implied-realised
premium), not that GVZ carries no information.
All coefficients are fit once on pairs whose whole target lies in 2014-01-01..2017-12-31 (FROZEN), by Newton on
the same loss that scores them. General at h=1 and h=22, the level controls (EWMA_094, RW5, HIST, IV_SCALED),
regime splits and 2024+ are descriptive.
"""
from datetime import datetime, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

import research_data_gvz_v1 as gvzdata
import research_eval_v1 as ev
from exp_vol_forecast_v1 import daily_rv, design, ewma_path, regimes
from research_run_v1 import mt5_manifest_data

QUESTIONS = {"general": {"h": 5, "kind": "qlike", "lag": 5}, "extreme": {"h": 5, "kind": "logit", "lag": 5},
             "regime_up": {"h": 22, "kind": "logit", "lag": 22}}
DESCRIPTIVE = {"general_h1": {"h": 1, "kind": "qlike", "lag": 5}, "general_h22": {"h": 22, "kind": "qlike", "lag": 22}}
MODELS = ("R", "A", "P", "A_lag2")
UP_SHIFT = 1.5
MIN_TRAIN_EVENTS = 30


def loss(kind, z, t):
    """Per-row loss of the linear index z: QLIKE with f = exp(z), or logistic log loss."""
    if kind == "qlike":
        return t * np.exp(-z) + z - np.log(t) - 1  # = t/f - log(t/f) - 1
    return np.logaddexp(0, z) - t * z


def fit(kind, X, t):
    """Newton with step halving; both losses are convex in the coefficients. Returns (beta, iterations,
    max |mean gradient| at the start of the last step) so non-convergence or separation shows in the report."""
    beta = np.linalg.lstsq(X, np.log(t), rcond=None)[0] if kind == "qlike" else np.zeros(X.shape[1])
    for it in range(1, 101):
        z = X @ beta
        if kind == "qlike":
            w = t * np.exp(-z)
            g, hess = X.T @ (1 - w), (X * w[:, None]).T @ X
        else:
            p = 1 / (1 + np.exp(-z))
            g, hess = X.T @ (p - t), (X * (p * (1 - p))[:, None]).T @ X
        step, s, base = np.linalg.solve(hess, g), 1.0, loss(kind, z, t).sum()
        while loss(kind, X @ (beta - s * step), t).sum() > base and s > 1e-8:
            s /= 2
        beta = beta - s * step
        if np.abs(s * step).max() < 1e-10:
            break
    return beta, it, float(np.abs(g).max() / len(t))


def questions(rv, x, gvz):
    """Per question: target, loss kind, and regressor frames per model (constant added later)."""
    e94 = ewma_path(rv, 0.94)
    le, lm, lw = np.log(e94), np.log(x["rv_m"]), np.log(x["rv_w"])
    li = np.log((gvzdata.align(gvz, rv.index)["gvz"] / 100) ** 2 / 252)
    li2 = np.log((gvzdata.align(gvz, rv.index, extra_lag=1)["gvz"] / 100) ** 2 / 252)
    lq = np.log(x["y5"].shift(5).expanding(252).quantile(0.9))  # y5 at t-5 covers t-4..t: known at t
    out = {}
    for name, q in {**QUESTIONS, **DESCRIPTIVE}.items():
        y = x[f"y{q['h']}"]
        # nested: A, P and A_lag2 add exactly one column to R, so only that column can earn the gain (critic B1)
        if name == "extreme":
            t, base, extra = (y > np.exp(lq)).astype(float).where(y.notna() & lq.notna()), [le, lq], lm
        elif name == "regime_up":
            t, base, extra = (y >= UP_SHIFT * x["rv_m"]).astype(float).where(y.notna()), [le, lm], lw
        else:
            t, base, extra = y, [le], lm
        regs = {"R": base, "A": [*base, li], "P": [*base, extra], "A_lag2": [*base, li2]}
        out[name] = {**q, "t": t, "regs": {m: pd.concat(v, axis=1) for m, v in regs.items()}}
    return out, e94, np.exp(li)


def nw_se(d, lag):
    """Newey-West standard error of mean(d)."""
    u = d - d.mean()
    var = u @ u / len(d)
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / len(d)
    return float(np.sqrt(max(var, 0) / len(d)))


def compare(la, lr, lag, trials):
    """A vs R on per-row losses: ratio, DM t, Sidak one-sided p both ways, 95% upper bound of the gain (%)."""
    d = (la - lr).to_numpy()
    se = nw_se(d, lag)
    t = float(d.mean() / se) if se > 0 else 0.0
    gain_ub = (-d.mean() + 1.645 * se) / lr.mean() * 100
    years = (la - lr).groupby(la.index.year).mean()
    return {"ratio": round(float(la.mean() / lr.mean()), 4), "dm_t": round(t, 2),
            "p_better": round(ev.sidak_p(-t, trials), 4), "p_worse": round(ev.sidak_p(t, trials), 4),
            "gain_pct": round(float(-d.mean() / lr.mean() * 100), 2), "gain_upper95_pct": round(float(gain_ub), 2),
            "years_better": int((years < 0).sum()), "years": int(len(years))}


def evaluate(h1, gvz, spec, trials=1):
    """All tables for E-010; pure (no MT5), so it can be tested on synthetic bars."""
    days = daily_rv(h1)
    rv, x = days["rv"], design(days["rv"])
    est = tuple(pd.Timestamp(d) for d in spec["estimation_window"])
    windows = [tuple(pd.Timestamp(d) for d in w) for w in spec["windows"]]
    reg = regimes(rv, x.index)
    qs, e94, iv = questions(rv, x, gvz)
    results, params = {}, {}
    for name, q in qs.items():
        h, kind = q["h"], q["kind"]
        target_end = pd.Series(x.index, index=x.index).shift(-h)
        ok = q["t"].notna() & pd.concat(q["regs"].values(), axis=1).notna().all(axis=1)
        if kind == "qlike":
            ok &= q["t"] > 0
        ok &= x.index >= est[0]
        train = ok & (target_end < est[1])
        z, params[name] = {}, {"train_rows": int(train.sum())}
        if kind == "logit":
            params[name]["train_events"] = int(q["t"][train].sum())
        for m, frame in q["regs"].items():
            X = np.column_stack([np.ones(len(frame)), frame.to_numpy()])
            beta, iters, grad = fit(kind, X[train.to_numpy()], q["t"][train].to_numpy())
            z[m] = pd.Series(X @ beta, index=x.index)
            params[name][m] = {"beta": [round(float(b), 4) for b in beta], "iterations": iters, "grad": float(f"{grad:.1e}")}
        if kind == "qlike":  # level controls, descriptive; IV_SCALED: QLIKE-optimal k on the estimation rows
            k = float((q["t"][train] / iv[train]).mean())
            params[name]["iv_scale"] = round(k, 4)
            for m, f in (("EWMA_094", e94), ("RW5", rv.rolling(5).mean()), ("HIST", rv.expanding().mean()),
                         ("IV_SCALED", k * iv)):
                z[m] = np.log(f)
        n_trials = len(QUESTIONS) * trials
        for (a, b), lab in zip(windows, ("eval_2018_2023", "check_2024_on")):
            rows = ok & (x.index >= a) & (x.index < b)
            t = q["t"][rows]
            losses = {m: pd.Series(loss(kind, z[m][rows].to_numpy(), t.to_numpy()), index=t.index) for m in z}
            fit_ok = (all(params[name][m]["iterations"] < 100 for m in MODELS)
                      and params[name].get("train_events", MIN_TRAIN_EVENTS) >= MIN_TRAIN_EVENTS)
            r = {"n": int(rows.sum()), "fit_ok": fit_ok, "loss": {m: round(float(v.mean()), 5) for m, v in losses.items()},
                 "A_vs_R": compare(losses["A"], losses["R"], q["lag"], n_trials),
                 "A_vs_R_lag22_t": compare(losses["A"], losses["R"], 22, n_trials)["dm_t"],
                 "A_lag2_vs_R": compare(losses["A_lag2"], losses["R"], q["lag"], n_trials),
                 "A_vs_P": compare(losses["A"], losses["P"], q["lag"], n_trials),
                 "by_regime_ratio_A_R": {g: round(float(losses["A"][reg[rows] == g].mean() / losses["R"][reg[rows] == g].mean()), 4)
                                         for g in ("low", "mid", "high") if (reg[rows] == g).sum() > 20}}
            if kind == "logit":
                r["event_rate"] = round(float(t.mean()), 4)
                r["brier"] = {m: round(float(((1 / (1 + np.exp(-z[m][rows])) - t) ** 2).mean()), 5) for m in MODELS}
            results[f"{name}|{lab}"] = r
    return results, params, days


def decide(results, spec):
    """Per primary question: supported / refuted / not_significant / underpowered (see the preregistration)."""
    verdicts = {}
    for name in QUESTIONS:
        r = results[f"{name}|eval_2018_2023"]
        c = r["A_vs_R"]
        if not r["fit_ok"]:  # Newton hit the iteration cap (e.g. separation) or too few training events
            verdicts[name] = "underpowered"
        elif c["p_worse"] < 0.05:
            verdicts[name] = "refuted"
        elif (c["ratio"] < 1 and c["p_better"] < 0.05 and r["A_lag2_vs_R"]["ratio"] < 1 and r["A_vs_P"]["ratio"] < 1
              and c["years_better"] >= spec["min_years_better"]):
            verdicts[name] = "supported"
        elif c["gain_upper95_pct"] < spec["materiality_pct"]:
            verdicts[name] = "not_significant"
        else:
            verdicts[name] = "underpowered"
    v = set(verdicts.values())
    overall = "supported" if "supported" in v else "refuted" if "refuted" in v else "inconclusive"
    return overall, verdicts


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
    manifest_data["data_sources"].append(f"FRED GVZCLS snapshot {gvzdata.DATA.relative_to(gvzdata.ROOT).as_posix()} "
                                         f"sha256 {gvzdata.sha256(gvzdata.DATA)}")
    h1["TIME_DT"] = pd.to_datetime(h1["time"], unit="s")  # server time, as the rest of the repo uses
    h1 = h1.rename(columns={"open": "OPEN", "high": "HIGH", "low": "LOW", "close": "CLOSE"}).iloc[:-1]
    h1 = h1[h1["TIME_DT"].dt.normalize() < h1["TIME_DT"].dt.normalize().max()]  # drop today's incomplete day
    gvz = gvzdata.load()
    results, params, days = evaluate(h1, gvz, spec, trials)
    overall, verdicts = decide(results, spec)
    audit = gvzdata.alignment_audit(gvz, days.index[days.index >= spec["estimation_window"][0]])
    parts = []
    for name in QUESTIONS:
        r = results[f"{name}|eval_2018_2023"]
        c = r["A_vs_R"]
        parts.append(f"{name}: {verdicts[name]} (A/R {c['ratio']}, DM t {c['dm_t']}, Sidak p {c['p_better']}, "
                     f"gain {c['gain_pct']}% ub95 {c['gain_upper95_pct']}%, years {c['years_better']}/{c['years']}, "
                     f"lag2 A/R {r['A_lag2_vs_R']['ratio']}, A/P {r['A_vs_P']['ratio']})")
    summary = f"{overall}: " + "; ".join(parts)
    rows = []
    for key, r in results.items():
        c = r["A_vs_R"]
        rows.append(f"| {key} | {r['n']} | {r['loss']['R']} | {r['loss']['A']} | {r['loss']['P']} | "
                    f"{r['loss']['A_lag2']} | {c['ratio']} | {c['dm_t']} | {r['A_vs_R_lag22_t']} | {c['p_better']} | "
                    f"{c['gain_pct']} | {c['gain_upper95_pct']} | {c['years_better']}/{c['years']} | "
                    f"{r['A_vs_P']['ratio']} | {r['by_regime_ratio_A_R']} | {r.get('event_rate', '')} |")
    controls = [f"| {key} | " + " | ".join(str(r["loss"][m]) for m in ("R", "A", "EWMA_094", "RW5", "HIST", "IV_SCALED")) + " |"
                for key, r in results.items() if "EWMA_094" in r["loss"]]
    report = "\n".join([
        "# E-010 GVZ information increment (H-014, volatility clause)", "", f"Outcome: **{overall}**", "",
        summary, "", "Statistical only; no economic value is claimed (H-018). Primary rows: general, extreme, "
        "regime_up on eval_2018_2023. general_h1, general_h22 and all check_2024_on rows are descriptive.",
        "Evidence class: development (GOLD# 2018+ was inspected before; GVZ was never used).", "",
        "| question / window | n | loss R | A | P | A_lag2 | A/R | DM t | DM t lag22 | Sidak p (A better) | gain % | "
        "gain ub95 % | years A better | A/P | A/R by regime | event rate |",
        "|---|" + "---|" * 15, *rows, "",
        "Level controls (QLIKE; descriptive):", "", "| question / window | R | A | EWMA_094 | RW5 | HIST | IV_SCALED |",
        "|---|---|---|---|---|---|---|", *controls, "",
        f"FROZEN coefficients (fit {spec['estimation_window']}; constant first): {params}.", "",
        f"GVZ alignment (lag 1): {audit['usable_lag1']}/{audit['server_days']} server days usable, staleness "
        f"{audit['stale_days_distribution_lag1']}, same-day or future values used: {audit['same_day_or_future_used']}."])
    return {"outcome": overall, "evidence_class": "development", "summary": summary,
            "failure_cause": "none" if overall == "supported" else "information",
            "lessons": f"GVZ increment over price history: {verdicts}",
            "next_questions": "H-018 with EWMA_094 stays next; a supported question names GVZ only for that question "
                              "and needs forward data before any runner use",
            "metrics": {"results": results, "params": params, "verdicts": verdicts, "alignment": audit, "mean_r": "n/a"},
            "report": report, "manifest_data": manifest_data,
            "model_note": "restricted/augmented coefficients estimated inside the run on 2014-2017 only; no model file"}
