"""E-001 (H-001): are the current "best" GOLD# H1 candidates timing skill, or just long exposure in a rising market?

Candidates are fixed definitions from earlier work (no fitting here): the frozen H1 C model, combo A
(net rule count >= 2), the EMA20/50 breakout, plus two controls -- H1 RSI(2) long (expected no skill) and a
seeded random signal (must show ~0 timing). All use the same exits (long, TP 3 / SL 1.5 ATR, 48 h) and pay
spread (32-point floor) and overnight swap at today's rate. Evaluated by research_eval_v1 on 2018-2023
(reliable bars) and 2024+ separately. Both windows were already looked at: evidence is reuse_contaminated.
"""
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

import h1_runner_v1
import research_eval_v1 as ev
from research_run_v1 import mt5_manifest_data

CONTROLS = ("H1_RSI2_L", "RANDOM_2PCT")


def signals(frame, rules, config, model):
    c, h = frame["CLOSE"], frame["HIGH"]
    e20, e50 = c.ewm(span=20, adjust=False).mean(), c.ewm(span=50, adjust=False).mean()
    ema = ((e20 > e50) & (c > e50) & (c > h.shift(1).rolling(20).max())).fillna(False).shift(1, fill_value=False)
    net = rules[[k for k in rules if k.endswith("_L")]].sum(axis=1) - rules[[k for k in rules if k.endswith("_S")]].sum(axis=1)
    score = model.predict_proba(frame[config["features"]].astype(np.float32))[:, 1]
    rng = np.random.default_rng(20261111)
    return {"H1_C_frozen": score >= config["threshold"], "COMBO_A_k2": (net >= 2).to_numpy(),
            "EMA_BREAKOUT_20_50_20": ema.to_numpy(), "H1_RSI2_L": rules["H1_RSI2_L"].to_numpy().astype(bool),
            "RANDOM_2PCT": rng.random(len(frame)) < 0.02}


def run(run_dir, spec, trials):
    config, lab, model = h1_runner_v1.load_frozen()  # verifies frozen hashes; frozen h1_lab_v1 wins the import
    import combo_lab_v1 as cl
    import rules_lab_v1 as rl
    end = datetime.now(timezone.utc)
    start = datetime(2014, 1, 1, tzinfo=timezone.utc)
    if not mt5.initialize(path=spec.get("terminal", r"D:\XM2\terminal64.exe")):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        h1 = lab.copy_rates("H1", start - timedelta(days=60), end)
        higher = {tf: lab.copy_rates(tf, start - timedelta(days=900), end) for tf in lab.HIGHER}
        d1 = rl.prepare(lab.copy_rates("Daily", start - timedelta(days=400), end))
        info = mt5.symbol_info("GOLD#")
        manifest_data = mt5_manifest_data(mt5, ["GOLD#"], start - timedelta(days=900), end, len(h1),
                                          spec["windows"], "MT5 history can be revised by the broker")
    finally:
        mt5.shutdown()
    frame = lab.make_features(h1.iloc[:-1], higher)  # drop the bar still forming
    rules = cl.rule_columns(frame, d1)
    sigs = signals(frame, rules, config, model)
    swap_per_bar = info.swap_long * info.point * spec["swap_charges_per_h1_bar"]

    results = {}
    for name, sig in sigs.items():
        results[name] = {}
        for label, (a, b) in zip(("reliable_2018_2023", "recent_2024_on"), spec["windows"]):
            mask = ((frame["TIME_DT"] >= a) & (frame["TIME_DT"] < b)).to_numpy()
            f = frame[mask].reset_index(drop=True)
            results[name][label] = ev.evaluate(f, sig[mask], "long", spec["tp"], spec["sl"], spec["hold"],
                                               swap_per_bar, trials=trials)
        print(name, {k: (v.get("timing_residual_mean_r"), v.get("timing_residual_ci_month_bootstrap"))
                     for k, v in results[name].items()}, flush=True)

    def skill(name):
        rel, rec = results[name]["reliable_2018_2023"], results[name]["recent_2024_on"]
        ci = rel.get("timing_residual_ci_month_bootstrap", [float("nan")] * 2)
        return bool(ci[0] > 0 and rec.get("timing_residual_mean_r", -1) > 0)

    skilled = [n for n in sigs if n not in CONTROLS and skill(n)]
    control_ok = all(abs(results[c]["reliable_2018_2023"].get("timing_residual_mean_r", 0)) < 0.1 or c == "H1_RSI2_L"
                     for c in CONTROLS)
    outcome = "inconclusive" if not control_ok else ("refuted" if skilled else "supported")
    rows = []
    for name, by in results.items():
        for label, r in by.items():
            t = r["trades"]
            rows.append(f"| {name} | {label} | {t.get('n', 0)} | {t.get('avg_r', '')} | {r.get('always_in', {}).get('avg_r', '')} | "
                        f"{r.get('drift_component_mean_r', '')} | {r.get('timing_residual_mean_r', '')} | "
                        f"{r.get('timing_residual_ci_month_bootstrap', '')} | {r.get('month_matched_random_share_at_or_above', '')} | "
                        f"{r.get('share_positive_timing_years', '')} |")
    summary = (f"timing skill beyond monthly drift (2018-2023 CI > 0 and 2024+ > 0): {skilled or 'none'}; "
               f"controls {'OK' if control_ok else 'FAILED'}")
    report = "\n".join([
        f"# E-001 edge decomposition (H-001)", "", f"Outcome: **{outcome}** (H-001 = candidate edges are drift exposure)",
        "", summary, "", "Evidence class: reuse_contaminated (both windows were inspected in earlier work).", "",
        "| candidate | window | trades | mean R | always-in R | drift part | timing residual | residual 95% CI (month bootstrap) | month-matched random >= | share positive timing years |",
        "|---|---|---|---|---|---|---|---|---|---|", *rows, "",
        f"Trials counted in family for Sidak adjustment: {trials}. Swap per H1 bar (price): {swap_per_bar:.4f}."])
    all_main = [results[n]["recent_2024_on"]["trades"] for n in sigs if n not in CONTROLS]
    metrics = {"candidates": results, "skilled": skilled, "control_ok": control_ok,
               "mean_r": round(float(np.mean([t.get("avg_r", 0) for t in all_main])), 4)}
    return {"outcome": outcome, "evidence_class": "reuse_contaminated", "summary": summary,
            "failure_cause": "none" if skilled else "information",
            "lessons": ("timing skill exists beyond drift for " + ", ".join(skilled)) if skilled else
                       "after removing each month's drift, the candidates' timing adds nothing reliable; "
                       "price-only H1 entries mostly harvest long exposure",
            "next_questions": "is the skill stable forward (fresh data only)?" if skilled else
                              "test information that is not in the price path (H-002/H-003), or stop price-only timing",
            "metrics": metrics, "report": report, "manifest_data": manifest_data,
            "model_note": "frozen H1 C model evaluated, not trained (frozen/h1_C_long_v1)"}
