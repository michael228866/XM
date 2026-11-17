"""H-013 building blocks (design_only, D-021): cost contract, ex-post label, past-only features, cost gate, purged
walk-forward splits, a fixed small model set and the power calculation. Offline only: no MT5, no orders, no strategy
replay. There is no run(): a formal run needs the user's choice between the two designs in H013_PREREG_DRAFT_V2.md
section 6, a preregistration v2, the critic and formal approval; research_run_v1.py refuses it until then.

Units: points (1 point = 0.01 USD/oz, gemini.py (ask - bid) / point). Prices on the bid side unless stated.

Cost contract (H013_PREREG_DRAFT_V2.md section 3): one spread per round trip, never two.
  basis "fill": PnL from the actual fill prices; the spread is already inside them -> spread charge 0, only the
                broker's commission / fee / swap (all 0 on this demo account, O-016 / E-015).
  basis "bid":  long buys at ask = bid + entry spread, sells at bid -> pays the ENTRY spread;
                short sells at bid, buys back at ask = bid + exit spread -> pays the EXIT spread.
  basis "mid":  half the entry spread + half the exit spread, either side.
All three give the same PnL for the same quotes (test_research_os_v1.H013BuildingBlockTests).
A missing spread or bound is NaN, never 0 (a missing cost is not a free trade).
"""
import numpy as np
import pandas as pd

from exp_spread_cost_v1 import ROLLOVER_HOURS

LABEL = "y_ex_post"        # question A: uses (t, t+h], description only, never a feature
PRED = "p_pred"            # question B: out-of-sample prediction from past-only features
FEATURES = ["rv_30", "rv_240", "rv_1440", "hour_sin", "hour_cos", "roll_exit_sched", "quote_before", "c95"]


def spread_charge(side, basis, entry_spread, exit_spread):
    """Spread paid by one round trip, points (NaN propagates)."""
    if basis == "fill":
        return 0.0  # the spread is inside the fill prices; charging it again double counts
    if basis == "bid":
        return np.asarray(entry_spread if side == "long" else exit_spread, dtype=float)
    if basis == "mid":
        return 0.5 * (np.asarray(entry_spread, dtype=float) + np.asarray(exit_spread, dtype=float))
    raise ValueError(f"unknown price basis {basis!r}")


def net_points(side, basis, entry_px, exit_px, entry_spread=np.nan, exit_spread=np.nan, extra_points=0.0, point=0.01):
    """Net PnL in points of one round trip. entry_px / exit_px: the fill prices (basis fill), the bid prices (bid) or
    the mid prices (mid). extra_points: commission + fee + swap converted to points (charged on every basis)."""
    sign = 1.0 if side == "long" else -1.0
    gross = sign * (np.asarray(exit_px, dtype=float) - np.asarray(entry_px, dtype=float)) / point
    return gross - spread_charge(side, basis, entry_spread, exit_spread) - extra_points


def label_cost(c95, e95x, exit_stress=2.0):
    """Direction-free conservative round-trip cost for the label: the worse of a long (entry spread, C95) and a short
    (exit spread, E95x x stress; D-022). One spread, not C95 + E95x. NaN if either bound is missing."""
    c, e = np.asarray(c95, dtype=float), np.asarray(e95x, dtype=float) * exit_stress
    return np.where(np.isnan(c) | np.isnan(e), np.nan, np.maximum(c, e))


def roll_exit_scheduled(times, horizon_min):
    """Rollover flag of the SCHEDULED exit time t + h (known at t). E-015 keys E95x on the actual exit quote time,
    which across the daily break is the reopen and only known later: a predictor must use this one."""
    return np.asarray((pd.DatetimeIndex(times) + pd.Timedelta(minutes=horizon_min)).hour.isin(ROLLOVER_HOURS))


def y_ex_post(bars, horizon_min, k, cost):
    """Question A label: 1 if the bid high-low range over minutes (t, t+h] exceeds k x cost(t), else 0; NaN when the
    cost is missing or the window is not complete (fewer than h bars inside, e.g. a weekend). Uses the future."""
    h = horizon_min
    fwd = lambda s, how: getattr(s.shift(-1)[::-1].rolling(h, min_periods=h), how)()[::-1].to_numpy()  # bars i+1..i+h
    hi, lo = fwd(bars["HIGH"].astype(float).reset_index(drop=True), "max"), fwd(bars["LOW"].astype(float).reset_index(drop=True), "min")
    t = bars["TIME_DT"].reset_index(drop=True)
    complete = ((t.shift(-h) - t) == pd.Timedelta(minutes=h)).to_numpy()  # h bars and no gap inside the window
    cost = np.asarray(cost, float)
    rng = (hi - lo) / 0.01
    return np.where(complete & ~np.isnan(cost), (rng > k * cost).astype(float), np.nan)


def past_features(bars, quote_before, c95, horizon_min):
    """Features known at the open of minute t: realized volatility of bid closes strictly before t over the last
    30 / 240 / 1440 bars, server hour, scheduled exit rollover flag, the last quote before t (the live quote a
    decision sees) and the day's C95 (from earlier days only, exp_cost_bound_v1.bounds)."""
    r = np.log(bars["CLOSE"].astype(float)).diff()
    f = pd.DataFrame(index=bars.index)
    for w in (30, 240, 1440):
        f[f"rv_{w}"] = r.rolling(w, min_periods=w // 2).std().shift(1)  # shift: bar t's close is not known at t
    hour = bars["TIME_DT"].dt.hour
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f["roll_exit_sched"] = roll_exit_scheduled(bars["TIME_DT"], horizon_min).astype(float)
    f["quote_before"], f["c95"] = np.asarray(quote_before, float), np.asarray(c95, float)
    return f


def cost_gate(quote_before, c95):
    """Reason per decision minute: 'no_bound' (C95 missing: not evaluable, not free), 'cost_gate' (live quote above
    C95; spikes are not covered, E-015) or 'ok'. Uses the quote visible at decision time only."""
    q, c = np.asarray(quote_before, float), np.asarray(c95, float)
    return np.where(np.isnan(c) | np.isnan(q), "no_bound", np.where(q > c, "cost_gate", "ok"))


def skip_reason(gate, p_pred, threshold):
    """Why a gemini decision would not be taken: the cost gate first, then the model filter (reported separately)."""
    p = np.asarray(p_pred, float)
    return np.where(gate != "ok", gate, np.where(np.isnan(p), "no_prediction", np.where(p < threshold, "model_filter", "trade")))


def walk_forward_folds(times, start, end, purge_min):
    """Monthly folds: test = one calendar month; train = everything before the month minus purge_min minutes (purge
    must cover the label window)."""
    t = pd.DatetimeIndex(times)
    folds = []
    for m in pd.period_range(pd.Timestamp(start), pd.Timestamp(end) - pd.Timedelta(days=1), freq="M"):
        a, b = m.start_time, (m + 1).start_time
        folds.append((np.where(t < a - pd.Timedelta(minutes=purge_min))[0], np.where((t >= a) & (t < b))[0]))
    return folds


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    ok = ~(np.isnan(y) | np.isnan(p))
    return float(roc_auc_score(y[ok], p[ok])) if ok.sum() and len(np.unique(y[ok])) == 2 else float("nan")


def question_b(frame, features, start, end, purge_min, min_train=5000):
    """Fixed model set, refit each month on past rows only: 'base_rate_by_hour' and 'logistic' (standardized
    features, sklearn defaults). Refuses label-derived columns as features (leakage)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    bad = [c for c in features if c == LABEL or c.startswith("y_") or c in ("exit_at", "y_exit", "spike_exit")]
    if bad:
        raise ValueError(f"leakage: label or future columns used as features: {bad}")
    y = frame[LABEL].to_numpy(float)
    X = frame[features].to_numpy(float)
    hour = frame["TIME_DT"].dt.hour.to_numpy()
    preds = {"base_rate_by_hour": np.full(len(frame), np.nan), "logistic": np.full(len(frame), np.nan)}
    for tr, te in walk_forward_folds(frame["TIME_DT"], start, end, purge_min):
        tr = tr[~np.isnan(y[tr]) & ~np.isnan(X[tr]).any(1)]
        if len(tr) < min_train or not len(te) or len(np.unique(y[tr])) < 2:
            continue
        rate = pd.Series(y[tr]).groupby(hour[tr]).mean()
        preds["base_rate_by_hour"][te] = pd.Series(hour[te]).map(rate).fillna(y[tr].mean()).to_numpy()
        ok = te[~np.isnan(X[te]).any(1)]
        if len(ok):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
            preds["logistic"][ok] = m.predict_proba(X[ok])[:, 1]
    return {name: {"auc": auc(y, p), "n_scored": int((~np.isnan(p) & ~np.isnan(y)).sum())} for name, p in preds.items()}, preds


def two_group_se(sd, n, share_low, deff=1.0):
    """Standard error of the mean-R difference between the low-tradeability group (share_low of n) and the rest;
    deff = design effect for dependence between trades (1 + (m - 1) x rho for clusters of m)."""
    return sd * np.sqrt(deff / (n * share_low * (1 - share_low)))


def power(effect, sd, n, share_low, deff=1.0, alpha=0.05):
    """One-sided power of a two-group z test for a difference of `effect` R."""
    from scipy.stats import norm
    return float(norm.cdf(effect / two_group_se(sd, n, share_low, deff) - norm.ppf(1 - alpha)))


def n_required(effect, sd, share_low, deff=1.0, alpha=0.05, target=0.8):
    from scipy.stats import norm
    z = norm.ppf(1 - alpha) + norm.ppf(target)
    return int(np.ceil((z * sd / effect) ** 2 * deff / (share_low * (1 - share_low))))
