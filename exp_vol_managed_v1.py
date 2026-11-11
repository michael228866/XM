"""E-004 (H-005): does scaling a GOLD# long by inverse forecast volatility beat constant exposure on a
risk-adjusted basis, after spread and swap?

One fixed rule, no fitting: sigma_t = sqrt(EWMA(lambda=0.94) of squared daily returns) known at the close of
day t; tomorrow's weight w = min(cap, target / sigma_t), target = expanding median of sigma up to t (so the
average weight stays near 1 without peeking). Costs: spread fraction x |change in weight| on each rebalance,
swap x weight x nights (7 charges per week: triple on Thursday's bar). Constant exposure holds w = 1.
"""
from datetime import datetime, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from research_run_v1 import mt5_manifest_data


def vol_managed(d1, lam, cap, spread_frac, swap_long_price):
    """Daily net returns of constant (w=1) and vol-managed long, and the weights, indexed like d1."""
    ret = d1["CLOSE"].pct_change()
    sigma = np.sqrt((ret ** 2).ewm(alpha=1 - lam, adjust=False).mean())
    target = sigma.expanding(min_periods=60).median()
    w = (target / sigma).clip(upper=cap).shift(1)  # decided at yesterday's close
    nights = np.where(d1["TIME_DT"].dt.dayofweek == 3, 3, 1)
    swap = swap_long_price * nights / d1["CLOSE"].shift(1)  # negative while long
    managed = w * (ret + swap) - spread_frac * w.diff().abs()
    return pd.DataFrame({"TIME_DT": d1["TIME_DT"], "constant": ret + swap, "managed": managed, "w": w})


def sharpe(x):
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if x.std(ddof=1) > 0 else float("nan")


def max_dd(x):
    equity = (1 + x).cumprod()
    return float((equity / equity.cummax() - 1).min())


def sharpe_diff_ci(frame, seed=0, draws=1000):
    """Month-block bootstrap 95% CI of Sharpe(managed) - Sharpe(constant)."""
    groups = [g for _, g in frame.groupby(frame["TIME_DT"].dt.to_period("M"))]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(draws):
        s = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        diffs.append(sharpe(s["managed"]) - sharpe(s["constant"]))
    return [round(float(np.quantile(diffs, 0.025)), 3), round(float(np.quantile(diffs, 0.975)), 3)]


def window_stats(f):
    out = {"days": int(len(f)), "mean_weight": round(float(f["w"].mean()), 3),
           "sharpe_diff_ci_month_bootstrap": sharpe_diff_ci(f)}
    for col in ("constant", "managed"):
        out[col] = {"sharpe": round(sharpe(f[col]), 3), "annual_pct": round(float(f[col].mean()) * 252 * 100, 2),
                    "annual_vol_pct": round(float(f[col].std(ddof=1)) * np.sqrt(252) * 100, 2),
                    "max_dd_pct": round(max_dd(f[col]) * 100, 2)}
    out["sharpe_diff"] = round(out["managed"]["sharpe"] - out["constant"]["sharpe"], 3)
    return out


def run(run_dir, spec, trials):
    start = datetime(2014, 1, 1, tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)
    if not mt5.initialize(path=spec.get("terminal", r"D:\XM2\terminal64.exe")):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        info = mt5.symbol_info("GOLD#")
        d1 = pd.DataFrame(mt5.copy_rates_range("GOLD#", mt5.TIMEFRAME_D1, start, end))
        manifest_data = mt5_manifest_data(mt5, ["GOLD#"], start, end, len(d1), spec["windows"],
                                          "MT5 history can be revised by the broker")
    finally:
        mt5.shutdown()
    d1["TIME_DT"] = pd.to_datetime(d1["time"], unit="s")
    d1 = d1.rename(columns={"close": "CLOSE"}).iloc[:-1]  # drop the bar still forming
    daily = vol_managed(d1, spec["lambda"], spec["cap"], spec["spread_frac"], info.swap_long * info.point).dropna()

    labels = ("discovery_2018_2023", "check_2024_on")
    results = {lab: window_stats(daily[(daily["TIME_DT"] >= a) & (daily["TIME_DT"] < b)].reset_index(drop=True))
               for lab, (a, b) in zip(labels, spec["windows"])}
    for lab, r in results.items():
        print(lab, r["constant"]["sharpe"], r["managed"]["sharpe"], r["sharpe_diff_ci_month_bootstrap"], flush=True)
    disc, chk = results[labels[0]], results[labels[1]]
    if disc["sharpe_diff"] <= 0:
        outcome = "refuted"
    elif chk["sharpe_diff"] > 0 and disc["sharpe_diff_ci_month_bootstrap"][0] > 0:
        outcome = "supported"
    else:
        outcome = "inconclusive"
    summary = (f"Sharpe managed - constant: 2018-2023 {disc['sharpe_diff']} (CI {disc['sharpe_diff_ci_month_bootstrap']}), "
               f"2024+ {chk['sharpe_diff']}")
    rows = [f"| {lab} | {col} | {r[col]['sharpe']} | {r[col]['annual_pct']} | {r[col]['annual_vol_pct']} | "
            f"{r[col]['max_dd_pct']} | {r['mean_weight'] if col == 'managed' else 1} |"
            for lab, r in results.items() for col in ("constant", "managed")]
    report = "\n".join([
        "# E-004 volatility-managed long (H-005)", "", f"Outcome: **{outcome}**", "", summary, "",
        "Evidence class: development (gold prices were inspected before; this rule was not).", "",
        "| window | exposure | Sharpe | annual % | annual vol % | max DD % | mean weight |",
        "|---|---|---|---|---|---|---|", *rows, "",
        f"EWMA lambda {spec['lambda']}, cap {spec['cap']}, spread {spec['spread_frac']} per unit of weight change, "
        f"swap long per night (points) {info.swap_long}."])
    metrics = {"windows": results, "mean_r": "n/a"}
    return {"outcome": outcome, "evidence_class": "development", "summary": summary,
            "failure_cause": "none" if outcome == "supported" else "information",
            "lessons": "inverse-vol scaling improves gold long risk-adjusted return" if outcome == "supported" else
                       "inverse-vol scaling does not reliably improve gold long risk-adjusted return",
            "next_questions": "apply as a sizing layer to the live runners and confirm forward" if outcome == "supported"
                              else "second-moment sizing alone is not an edge for gold; focus on new information (H-003)",
            "metrics": metrics, "report": report, "manifest_data": manifest_data,
            "model_note": "no model; fixed EWMA volatility rule"}
