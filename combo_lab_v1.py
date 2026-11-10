"""Combo lab: do the 20 public rules (rules_lab_v1) help when mixed, on the H1 C setup (long, TP 3 / SL 1.5 ATR, 48 h)?

A  confluence: net count of active long rules minus short rules; threshold k picked on 2023, tested 2024+.
B  stacked:    the 20 rule signals added as features to the H1 lab XGBoost, trained exactly like H1 C.
Both compared with always-in, random entries and the frozen H1 C (dev test avg R +0.161, random >= 9%).

    python -B combo_lab_v1.py --output forward_reports/20261110_combo_lab_v1
"""
import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

import h1_lab_v1 as lab
import rules_lab_v1 as rl
from strategy_lab_v1 import MIN_VALIDATION_TRADES, first_touch, run_strategy, stats, trade

SPEC = lab.STRATEGIES["C_long_3.0_1.5_48h"]
CONFLUENCE_K = (1, 2, 3, 4)


def rule_columns(h1, d1):
    """One 0/1 column per rule on the H1 rows; D1 rules are active all day on the day they would enter."""
    up = (d1["CLOSE"] > d1["CLOSE"].rolling(20).mean()).astype(int)
    daily_up = pd.Series(up.shift(1).to_numpy(), index=d1["TIME_DT"].dt.normalize())
    day = h1["TIME_DT"].dt.normalize()
    cols = {}
    for name, (tf, rule, side, *_exits) in rl.STRATEGIES.items():
        if tf == "H1":
            cols[name] = rl.signals(h1, tf, rule, side, daily_up).astype(int)
        else:
            sig = pd.Series(rl.signals(d1, tf, rule, side).astype(int), index=d1["TIME_DT"].dt.normalize())
            cols[name] = day.map(sig[~sig.index.duplicated()]).fillna(0).astype(int).to_numpy()
    return pd.DataFrame(cols, index=h1.index)


def confluence(frame, rules, r, after):
    longs = [c for c in rules if c.endswith("_L")]
    shorts = [c for c in rules if c.endswith("_S")]
    score = (frame[longs].sum(axis=1) - frame[shorts].sum(axis=1)).to_numpy(float)
    times = frame["TIME_DT"]
    val = ((times >= lab.VALIDATION_START) & (times < lab.TEST_START)).to_numpy()
    test = (times >= lab.TEST_START).to_numpy()
    picks = [{"k": k, **stats(trade(frame[val], score[val], k, r[val], after[val])["r"])} for k in CONFLUENCE_K]
    eligible = [p for p in picks if p["n"] >= MIN_VALIDATION_TRADES]
    best = max(eligible or picks, key=lambda p: p.get("avg_r", -9))
    taken = trade(frame[test], score[test], best["k"], r[test], after[test])
    return {"validation_candidates": picks, "chosen": best, "validation_viable": bool(eligible) and best["avg_r"] > 0,
            "test": stats(taken["r"]), "baselines": lab.baselines(frame, score, best["k"], SPEC, test)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    end = datetime.now(timezone.utc)
    if not mt5.initialize(path=os.environ.get("XM_TERMINAL_PATH", r"D:\XM2\terminal64.exe")):
        raise SystemExit(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        h1 = lab.copy_rates("H1", lab.START - timedelta(days=60), end)
        higher = {tf: lab.copy_rates(tf, lab.START - timedelta(days=900), end) for tf in lab.HIGHER}
        d1 = rl.prepare(lab.copy_rates("Daily", lab.START - timedelta(days=400), end))
    finally:
        mt5.shutdown()
    frame = lab.make_features(h1, higher)
    rules = rule_columns(frame, d1)
    frame = pd.concat([frame, rules], axis=1)
    frame = frame[frame["TIME_DT"] >= lab.START.replace(tzinfo=None)].reset_index(drop=True)
    r, after = first_touch(frame, SPEC["side"], SPEC["tp"], SPEC["sl"], SPEC["hold"])
    test = (frame["TIME_DT"] >= lab.TEST_START).to_numpy()

    report = {"spec": SPEC, "rules": list(rules.columns),
              "h1_C_reference": json.loads((Path(__file__).parent / "frozen" / "h1_C_long_v1" / "config.json")
                                           .read_text(encoding="utf-8"))["development_test"]}
    print("=== A confluence", flush=True)
    report["A_confluence"] = confluence(frame, list(rules.columns), r, after)
    print(json.dumps({k: report["A_confluence"][k] for k in ("chosen", "test", "baselines")}), flush=True)
    print("=== B stacked XGBoost", flush=True)
    result, model = run_strategy(frame, lab.FEATURES + list(rules.columns), SPEC, lab.VALIDATION_START, lab.TEST_START)
    scores = model.predict_proba(frame[lab.FEATURES + list(rules.columns)].astype(np.float32))[:, 1]
    result["baselines"] = lab.baselines(frame, scores, result["chosen"]["threshold"], SPEC, test)
    gain = model.get_booster().get_score(importance_type="gain")
    result["top_features_by_gain"] = dict(sorted(gain.items(), key=lambda kv: -kv[1])[:10])
    report["B_stacked"] = result
    print(json.dumps({k: result[k] for k in ("chosen", "test", "baselines", "top_features_by_gain")}), flush=True)
    args.output.mkdir(parents=True)
    model.save_model(args.output / "B_stacked_xgb.json")
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
