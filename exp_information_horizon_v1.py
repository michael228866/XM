"""E-002 (H-002): do real rates and CFTC positioning predict GOLD# at their own (weekly) horizon?

Earlier families (Treasury real rate, CFTC COT) were closed after being added as 5 extra features to the M1
90-minute long classifier. That tested the wrong horizon: weekly positioning and daily rates move gold over
days to weeks. Here each pre-declared signal is tested alone on non-overlapping weekly returns, using the
already certified data files (availability timestamps respected), then translated into a weekly long/flat
rule net of spread and overnight swap and compared with always-long.

Pre-declared signals and expected sign (fixed before results):
  REAL10Y_CHG_20D  -> negative (rising real yields hurt gold)
  COT_MM_NET_PCT_OI -> negative (crowded managed-money longs precede weaker returns)
"""
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from research_memory_v1 import file_sha256
from research_run_v1 import mt5_manifest_data

COT = "training_runs/20260912T171048Z_gemini_cftc_gold_cot_foundation_v1/cftc_gold_cot_features.csv"
TREASURY = "training_runs/20260909T074821Z_gemini_us_treasury_real_rate_foundation_v1/treasury_daily_canonical.csv"
SIGNALS = {"REAL10Y_CHG_20D": -1, "COT_MM_NET_PCT_OI": -1}


def weekly_panel(d1, treasury, cot):
    """One row per week: decision at the first D1 bar of the week (open), return to next week's first open."""
    d1 = d1.sort_values("TIME_DT")
    week = d1["TIME_DT"].dt.to_period("W-SUN")
    first = d1.groupby(week).head(1).reset_index(drop=True)
    first["ret"] = first["OPEN"].shift(-1) / first["OPEN"] - 1
    first["nights"] = (first["TIME_DT"].shift(-1) - first["TIME_DT"]).dt.days  # swap charged every calendar night
    # Server time is ~UTC+2/3; using the bar time as UTC makes availability checks ~2-3 h conservative.
    decision = first["TIME_DT"].dt.tz_localize("UTC")
    t = treasury.copy()
    t["avail"] = pd.to_datetime(t["availability_ns"], utc=True)
    t["REAL10Y_CHG_20D"] = t["real_10y"] - t["real_10y"].shift(20)
    c = cot.copy()
    c["avail"] = pd.to_datetime(c["availability_time_utc"], utc=True)
    panel = pd.merge_asof(first.assign(decision=decision).sort_values("decision"),
                          t[["avail", "REAL10Y_CHG_20D"]].sort_values("avail"),
                          left_on="decision", right_on="avail", direction="backward", allow_exact_matches=False)
    panel = pd.merge_asof(panel.drop(columns="avail"), c[["avail", "COT_MM_NET_PCT_OI"]].sort_values("avail"),
                          left_on="decision", right_on="avail", direction="backward", allow_exact_matches=False)
    return panel.dropna(subset=["ret", *SIGNALS])


def spearman(x, y):
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def long_flat(panel, signal, sign, cost_frac, swap_frac_per_night):
    """Long for the week when the signal points up (sign * value < its expanding median), else flat."""
    s = sign * panel[signal]
    threshold = s.expanding().median().shift(1)  # only past weeks set the cut
    long = (s > threshold).to_numpy()
    changes = np.abs(np.diff(np.r_[0, long.astype(int)]))
    gross = np.where(long, panel["ret"], 0.0)
    net = gross + np.where(long, swap_frac_per_night * panel["nights"], 0.0) - changes * cost_frac
    always = panel["ret"] + swap_frac_per_night * panel["nights"]
    return pd.Series(net, index=panel.index), pd.Series(always.to_numpy(), index=panel.index), float(long.mean())


def sharpe(x):
    return round(float(x.mean() / x.std() * np.sqrt(52)), 2) if x.std() > 0 else None


def run(run_dir, spec, trials):
    from research_memory_v1 import ROOT
    treasury = pd.read_csv(ROOT / TREASURY)
    cot = pd.read_csv(ROOT / COT)
    end = datetime(2025, 1, 6, tzinfo=timezone.utc)  # certified data stop at 2024-12-31
    start = datetime(2015, 10, 1, tzinfo=timezone.utc)
    if not mt5.initialize(path=spec.get("terminal", r"D:\XM2\terminal64.exe")):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        rates = pd.DataFrame(mt5.copy_rates_range("GOLD#", mt5.TIMEFRAME_D1, start, end))
        info = mt5.symbol_info("GOLD#")
        manifest_data = mt5_manifest_data(mt5, ["GOLD#"], start, end, len(rates), spec["windows"],
                                          "certified COT/Treasury files referenced by sha256")
    finally:
        mt5.shutdown()
    d1 = rates.rename(columns={"open": "OPEN", "close": "CLOSE"})
    d1["TIME_DT"] = pd.to_datetime(d1["time"], unit="s")
    panel = weekly_panel(d1, treasury, cot)
    price = float(d1["CLOSE"].iloc[-1])
    swap_frac = info.swap_long * info.point / price      # today's rate applied to all history (shortcut)
    cost_frac = spec["spread_frac"]

    results, rows = {}, []
    for name, sign in SIGNALS.items():
        results[name] = {}
        for label, (a, b) in zip(("discovery_2016_2022", "check_2023_2024"), spec["windows"]):
            p = panel[(panel["TIME_DT"] >= a) & (panel["TIME_DT"] < b)]
            ic = spearman(p[name], p["ret"])
            net, always, share = long_flat(p, name, sign, cost_frac, swap_frac)
            results[name][label] = {"weeks": len(p), "ic": round(ic, 4), "ic_t": round(ic * np.sqrt(len(p)), 2),
                                    "ic_in_expected_direction": bool(np.sign(ic) == sign),
                                    "long_flat_sharpe": sharpe(net), "always_long_sharpe": sharpe(always),
                                    "long_flat_annual": round(float(net.mean() * 52), 4),
                                    "always_long_annual": round(float(always.mean() * 52), 4), "share_long": round(share, 3)}
            r = results[name][label]
            rows.append(f"| {name} | {label} | {r['weeks']} | {r['ic']} | {r['ic_t']} | {r['long_flat_sharpe']} | "
                        f"{r['always_long_sharpe']} | {r['long_flat_annual']} | {r['always_long_annual']} |")

    def verdict(name, sign):
        d, c = results[name]["discovery_2016_2022"], results[name]["check_2023_2024"]
        if d["ic_in_expected_direction"] and abs(d["ic_t"]) >= 2 and c["ic_in_expected_direction"] and \
                (d["long_flat_sharpe"] or -9) > (d["always_long_sharpe"] or 9) and \
                (c["long_flat_sharpe"] or -9) > (c["always_long_sharpe"] or 9):
            return "supported"
        if not d["ic_in_expected_direction"] or abs(d["ic_t"]) < 1:
            return "refuted"
        return "inconclusive"

    verdicts = {n: verdict(n, s) for n, s in SIGNALS.items()}
    outcome = "supported" if "supported" in verdicts.values() else (
        "refuted" if all(v == "refuted" for v in verdicts.values()) else "inconclusive")
    summary = f"weekly-horizon verdicts: {verdicts}"
    report = "\n".join([
        "# E-002 information horizon (H-002)", "", f"Outcome: **{outcome}**", "", summary, "",
        "Evidence class: development (gold prices 2016-2024 were inspected before; the weekly factor tests were not).", "",
        "| signal | window | weeks | Spearman IC | IC t | long/flat Sharpe | always-long Sharpe | long/flat annual | always-long annual |",
        "|---|---|---|---|---|---|---|---|---|", *rows, "",
        f"Costs: spread {cost_frac:.5f} per change, swap {swap_frac:.6f} of price per night while long (today's rate).",
        f"Trials counted in family: {trials}."])
    manifest_data["data_sources"].append("certified CFTC COT and US Treasury real-rate files (training_runs)")
    manifest_data["source_files"] = [{"path": f, "sha256": file_sha256(ROOT / f), "retention_status": "in_repository"}
                                     for f in (COT, TREASURY)]
    return {"outcome": outcome, "evidence_class": "development", "summary": summary,
            "failure_cause": "none" if outcome == "supported" else "information",
            "lessons": "see report: horizon-matched test of previously closed exogenous families",
            "next_questions": ("freeze the supported signal and collect fresh 2025+ data (needs a data refresh)"
                               if outcome == "supported" else
                               "if weekly horizon also fails, these exogenous families carry no usable edge for XM gold CFDs"),
            "metrics": {"signals": results, "verdicts": verdicts, "weeks": int(len(panel))},
            "report": report, "manifest_data": manifest_data}
