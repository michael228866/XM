"""E-007 (H-005): does inverse-volatility sizing of a GOLD# long beat constant exposure at an equal risk budget?

Replaces exp_vol_managed_v1 (withdrawn before it ran, D-006). v1 compared against w = 1, so annual return,
drawdown and Calmar mixed sizing skill with a different average exposure, and it had no control for the
weight distribution itself. Here:

- MANAGED: w_t = min(cap, target / sigma_{t-1}); sigma = sqrt(EWMA(lambda) of squared daily returns),
  target = expanding median of sigma (both known at the previous close).
- CONST_EQUAL_EXPOSURE: constant w = mean MANAGED weight in the window (same average exposure and swap bill).
- CONST_VOL_MATCHED: constant w scaled to MANAGED's realised volatility (same risk budget; compare drawdown and
  Calmar here). Both constants use one in-window scalar: a level, not timing, so it cannot leak skill.
- SHUFFLED: MANAGED's weights permuted across whole months (seeded draws). Same weight distribution, random
  timing; MANAGED must beat it for the timing of the weights to matter.

Costs: spread fraction x cost multiplier x |weight change| on every rebalance (constants pay it once), swap =
today's swap_long / today's price per night (a rate: fixed points would overcharge low-price years, which are also
the low-volatility, high-weight years, biasing against MANAGED) x nights (Thursday triple) x weight x multiplier.
"""
from datetime import datetime, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

import research_eval_v1 as ev
from research_run_v1 import mt5_manifest_data


def managed_weights(close, lam, cap):
    ret = close.pct_change()
    sigma = np.sqrt((ret ** 2).ewm(alpha=1 - lam, adjust=False).mean())
    return (sigma.expanding(min_periods=60).median() / sigma).clip(upper=cap).shift(1)  # decided at the prior close


def net_returns(ret, carry, w, spread_frac):
    """Daily net return for weight path w; carry is the swap return of one unit (negative while long)."""
    w = pd.Series(w, index=ret.index)
    return w * (ret + carry) - spread_frac * w.diff().abs().fillna(w.abs())


def month_shuffle(w, months, rng):
    keys = months.unique()
    blocks = {k: w[months == k].to_numpy() for k in keys}
    order = rng.permutation(keys)
    out = np.concatenate([blocks[k] for k in order])
    return out[:len(w)] if len(out) >= len(w) else np.resize(out, len(w))


def window_eval(f, spec, cost_mult, swap_mult, seed=0):
    spread = spec["spread_frac"] * cost_mult
    carry = f["carry"] * swap_mult
    w = f["w"]
    managed = net_returns(f["ret"], carry, w, spread)
    eq = net_returns(f["ret"], carry, np.full(len(f), w.mean()), spread)
    vm_scale = managed.std(ddof=1) / f["ret"].std(ddof=1)
    vm = net_returns(f["ret"], carry, np.full(len(f), vm_scale), spread)
    months = f["TIME_DT"].dt.to_period("M").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    shuffled = [ev.daily_stats(net_returns(f["ret"], carry, month_shuffle(w.reset_index(drop=True), months, rng), spread),
                               f["TIME_DT"])["sharpe"] for _ in range(spec["shuffle_draws"])]
    out = {k: ev.daily_stats(v, f["TIME_DT"]) for k, v in
           (("MANAGED", managed), ("CONST_EQUAL_EXPOSURE", eq), ("CONST_VOL_MATCHED", vm))}
    out["SHUFFLED"] = {"sharpe_median": round(float(np.median(shuffled)), 3),
                       "share_at_or_above_managed": round(float(np.mean(np.array(shuffled) >= out["MANAGED"]["sharpe"])), 3)}
    out["mean_weight"] = round(float(w.mean()), 3)
    out["sharpe_diff"] = round(out["MANAGED"]["sharpe"] - out["CONST_EQUAL_EXPOSURE"]["sharpe"], 3)
    return out, managed, eq


def sharpe_diff_ci(a, b, times, seed=0, draws=1000):
    """Month-block bootstrap 95% CI of Sharpe(a) - Sharpe(b)."""
    frame = pd.DataFrame({"a": a.to_numpy(), "b": b.to_numpy(), "m": pd.DatetimeIndex(times).to_period("M")})
    groups = [g for _, g in frame.groupby("m")]
    rng = np.random.default_rng(seed)
    sr = lambda x: x.mean() / x.std(ddof=1) * np.sqrt(252)
    diffs = []
    for _ in range(draws):
        s = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        diffs.append(sr(s["a"]) - sr(s["b"]))
    return [round(float(np.quantile(diffs, 0.025)), 3), round(float(np.quantile(diffs, 0.975)), 3)]


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
    nights = np.where(d1["TIME_DT"].dt.dayofweek == 3, 3, 1)
    frame = pd.DataFrame({"TIME_DT": d1["TIME_DT"], "ret": d1["CLOSE"].pct_change(),
                          "carry": info.swap_long * info.point / d1["CLOSE"].iloc[-1] * nights,
                          "w": managed_weights(d1["CLOSE"], spec["lambda"], spec["cap"])}).dropna()

    labels = ("discovery_2018_2023", "check_2024_on")
    table, ci = {}, {}
    for lab, (a, b) in zip(labels, spec["windows"]):
        f = frame[(frame["TIME_DT"] >= a) & (frame["TIME_DT"] < b)].reset_index(drop=True)
        table[lab] = {}
        for m in spec["cost_multipliers"]:
            table[lab][f"cost x{m}"], managed, eq = window_eval(f, spec, m, 1.0)
            if m == 1.0:
                ci[lab] = sharpe_diff_ci(managed, eq, f["TIME_DT"])
        for s in spec["swap_multipliers"]:
            if s != 1.0:
                table[lab][f"swap x{s}"] = window_eval(f, spec, 1.0, s)[0]
        print(lab, table[lab]["cost x1.0"]["sharpe_diff"], ci[lab], flush=True)

    d, c = labels
    base_d, base_c = table[d]["cost x1.0"], table[c]["cost x1.0"]
    stress_d = table[d][f"cost x{float(max(spec['cost_multipliers']))}"]
    if base_d["sharpe_diff"] <= 0:
        outcome = "refuted"
    elif (base_c["sharpe_diff"] > 0 and ci[d][0] > 0 and stress_d["sharpe_diff"] > 0
          and base_d["SHUFFLED"]["share_at_or_above_managed"] <= 0.05):
        outcome = "supported"
    else:
        outcome = "inconclusive"
    summary = (f"Sharpe MANAGED - CONST_EQUAL_EXPOSURE: 2018-2023 {base_d['sharpe_diff']} (CI {ci[d]}), 2024+ "
               f"{base_c['sharpe_diff']}; cost x{max(spec['cost_multipliers'])} 2018-2023 {stress_d['sharpe_diff']}; shuffled >= managed "
               f"{base_d['SHUFFLED']['share_at_or_above_managed']}")
    rows = []
    for lab in labels:
        for scen, r in table[lab].items():
            for k in ("MANAGED", "CONST_EQUAL_EXPOSURE", "CONST_VOL_MATCHED"):
                x = r[k]
                rows.append(f"| {lab} | {scen} | {k} | {x.get('net_mean_bp')} | {x.get('sharpe')} | {x.get('calmar')} | "
                            f"{x.get('max_dd_pct')} | {x.get('annual_pct')} |")
            rows.append(f"| {lab} | {scen} | SHUFFLED | - | {r['SHUFFLED']['sharpe_median']} (share >= managed "
                        f"{r['SHUFFLED']['share_at_or_above_managed']}) | - | - | - |")
    report = "\n".join([
        "# E-007 volatility-managed long at equal risk budget (H-005)", "", f"Outcome: **{outcome}**", "", summary, "",
        "Evidence class: development (gold prices were inspected before; this rule was not).", "",
        "| window | scenario | strategy | net bp/day | Sharpe | Calmar | max DD % | annual % |",
        "|---|---|---|---|---|---|---|---|", *rows, "",
        f"EWMA lambda {spec['lambda']}, cap {spec['cap']}, spread {spec['spread_frac']} per unit of weight change, "
        f"swap long per night (points) {info.swap_long}, mean weight 2018-2023 {base_d['mean_weight']}."])
    return {"outcome": outcome, "evidence_class": "development", "summary": summary,
            "failure_cause": "none" if outcome == "supported" else "information",
            "lessons": f"inverse-vol sizing at equal exposure: {outcome}",
            "next_questions": "use as a sizing layer only after confirmation on data after 2026-11-12"
                              if outcome == "supported" else "second-moment sizing alone is not an edge; test H-011 forecasting quality",
            "metrics": {"windows": table, "sharpe_diff_ci": ci, "mean_r": "n/a"}, "report": report,
            "manifest_data": manifest_data, "model_note": "no model; fixed EWMA volatility rule"}
