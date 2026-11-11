"""Critical evaluation battery for any entry signal: is there timing skill beyond the market's own drift?

Given bars, a 0/1 entry signal and fixed exits, it reports:
- raw stats, and the same exits taken on every bar ("always in");
- drift decomposition: each trade's R minus the always-in mean R of its calendar month. What survives is
  timing skill; what is removed is just being exposed while the market trended that month;
- month-matched random entries (same months, same count) as a null distribution;
- month-block bootstrap CI for raw and timing-residual mean R;
- per-year table, share of positive years, and a Sidak multiple-testing adjustment for the number of
  trials already run in the family (taken from the research ledger by the caller).
"""
import math

import numpy as np
import pandas as pd

from strategy_lab_v1 import first_touch, stats, trade

BOOTSTRAP, RANDOM_DRAWS = 1000, 500


def trade_returns(frame, side, tp, sl, hold, swap_per_bar=0.0):
    """R and bars held for an entry at every row; swap_per_bar is in price units (negative = cost)."""
    r, after = first_touch(frame, side, tp, sl, hold)
    return r + swap_per_bar * after / (frame["ATR"].to_numpy() * sl), after


def block_bootstrap_ci(values, blocks, seed=0):
    values, blocks = np.asarray(values, float), np.asarray(blocks)
    keys = np.unique(blocks)
    if len(keys) < 2:
        return [float("nan"), float("nan")]
    groups = [values[blocks == k] for k in keys]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(BOOTSTRAP):
        pick = rng.integers(0, len(groups), len(groups))
        means.append(np.concatenate([groups[i] for i in pick]).mean())
    return [round(float(np.quantile(means, 0.025)), 4), round(float(np.quantile(means, 0.975)), 4)]


def sidak_p(t_stat, trials):
    """One-sided p of mean > 0 (normal approximation), adjusted for `trials` tries in the family."""
    p = 0.5 * math.erfc(t_stat / math.sqrt(2))
    return 1 - (1 - p) ** max(1, trials)


def evaluate(frame, signal, side, tp, sl, hold, swap_per_bar=0.0, trials=1, seed=0):
    r, after = trade_returns(frame, side, tp, sl, hold, swap_per_bar)
    taken = trade(frame, np.asarray(signal, float), 0.5, r, after)
    if taken.empty:
        return {"trades": stats([])}
    month = frame["TIME_DT"].dt.to_period("M")
    valid = np.isfinite(r)
    month_mean = pd.Series(r[valid]).groupby(month[valid].to_numpy()).mean()
    taken_month = taken["TIME_DT"].dt.to_period("M")
    drift = taken_month.map(month_mean).to_numpy(float)
    resid = taken["r"].to_numpy() - drift

    # Month-matched random entries: for each trade, a random valid bar from the same month.
    rng = np.random.default_rng(seed)
    by_month = {m: np.flatnonzero(valid & (month == m).to_numpy()) for m in taken_month.unique()}
    draws = np.array([np.mean([r[rng.choice(by_month[m])] for m in taken_month]) for _ in range(RANDOM_DRAWS)])

    n = len(resid)
    t_resid = float(resid.mean() / (resid.std(ddof=1) / math.sqrt(n))) if n > 1 and resid.std(ddof=1) > 0 else 0.0
    years = taken.assign(resid=resid).groupby(taken["TIME_DT"].dt.year).agg(
        n=("r", "size"), mean_r=("r", "mean"), timing_resid=("resid", "mean"))
    always = trade(frame, np.ones(len(frame)), 0.5, r, after)
    return {
        "trades": stats(taken["r"]),
        "always_in": stats(always["r"]),
        "drift_component_mean_r": round(float(np.nanmean(drift)), 4),
        "timing_residual_mean_r": round(float(np.nanmean(resid)), 4),
        "timing_residual_t": round(t_resid, 2),
        "timing_residual_p_sidak": round(sidak_p(t_resid, trials), 4),
        "trials_in_family": trials,
        "mean_r_ci_month_bootstrap": block_bootstrap_ci(taken["r"], taken_month.astype(str), seed),
        "timing_residual_ci_month_bootstrap": block_bootstrap_ci(resid, taken_month.astype(str), seed),
        "month_matched_random_mean_r_median": round(float(np.median(draws)), 4),
        "month_matched_random_share_at_or_above": round(float(np.mean(draws >= taken["r"].mean())), 3),
        "share_positive_years": round(float((years["mean_r"] > 0).mean()), 3),
        "share_positive_timing_years": round(float((years["timing_resid"] > 0).mean()), 3),
        "by_year": {int(y): {k: round(float(v), 4) for k, v in row.items()} for y, row in years.iterrows()},
    }


def daily_stats(ret, times):
    """Net expectation, Sharpe, Calmar and max drawdown of a daily return series (fractions, 252 days/yr)."""
    ret = pd.Series(np.asarray(ret, float)).dropna()
    if len(ret) < 3 or ret.std(ddof=1) == 0:
        return {"days": int(len(ret))}
    equity = (1 + ret).cumprod()
    mdd = float((equity / equity.cummax() - 1).min())
    annual = float(ret.mean()) * 252
    years = ret.groupby(pd.Series(pd.DatetimeIndex(times)[ret.index]).dt.year.to_numpy()).mean()
    return {"days": int(len(ret)), "net_mean_bp": round(float(ret.mean()) * 1e4, 3),
            "t": round(float(ret.mean() / ret.std(ddof=1) * math.sqrt(len(ret))), 2),
            "annual_pct": round(annual * 100, 2), "sharpe": round(float(ret.mean() / ret.std(ddof=1)) * math.sqrt(252), 3),
            "max_dd_pct": round(mdd * 100, 2), "calmar": round(annual / abs(mdd), 3) if mdd < 0 else None,
            "positive_years": int((years > 0).sum()), "years": int(len(years))}
