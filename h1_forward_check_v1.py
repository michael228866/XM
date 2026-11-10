"""Forward check for a frozen H1 candidate: score only bars after its freeze cutoff, with the frozen model and code.

    python -B h1_forward_check_v1.py --freeze              # once: snapshot model, threshold, code; cutoff = now
    python -B h1_forward_check_v1.py                       # any time later: trades since cutoff vs always-long

Reads MT5 bars only, never trades. The frozen folder holds copies of the lab code, so later edits to
h1_lab_v1.py / strategy_lab_v1.py cannot change what is being evaluated.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / "frozen" / "h1_C_long_v1"
SOURCE_RUN = ROOT / "forward_reports" / "20261105_h1_lab_v1_gold_only"
CANDIDATE = "C_long_3.0_1.5_48h"
FROZEN_CODE = ("h1_lab_v1.py", "strategy_lab_v1.py")
# Imported from the project, not copied; their hashes are checked so a silent change is caught.
DEPENDENCIES = ("drl_trading_v2.py", "gold_recent_walk_forward.py")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    if FROZEN.exists():
        raise SystemExit(f"Already frozen: {FROZEN}")
    report = json.loads((SOURCE_RUN / "report.json").read_text(encoding="utf-8"))
    result = report["strategies"][CANDIDATE]
    FROZEN.mkdir(parents=True)
    shutil.copy2(SOURCE_RUN / f"{CANDIDATE}_xgb.json", FROZEN / "model_xgb.json")
    for name in FROZEN_CODE:
        shutil.copy2(ROOT / name, FROZEN / name)
    config = {
        "candidate": CANDIDATE, "spec": result["spec"], "threshold": result["chosen"]["threshold"],
        "features": report["features"], "min_spread_fraction": report["min_spread_fraction"],
        "cutoff_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_run": str(SOURCE_RUN.relative_to(ROOT)), "development_test": result["test"],
        "development_baselines": result.get("baselines"),
        "sha256": {"model_xgb.json": sha256(FROZEN / "model_xgb.json"),
                   **{n: sha256(FROZEN / n) for n in FROZEN_CODE},
                   **{n: sha256(ROOT / n) for n in DEPENDENCIES}},
    }
    (FROZEN / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(config, indent=2, ensure_ascii=False))


def check(output):
    config = json.loads((FROZEN / "config.json").read_text(encoding="utf-8"))
    for name, digest in config["sha256"].items():
        path = FROZEN / name if (FROZEN / name).exists() else ROOT / name
        if sha256(path) != digest:
            raise SystemExit(f"{path} changed since freeze; the frozen candidate is no longer what is evaluated")
    sys.path.insert(0, str(FROZEN))  # frozen copies win over the project's current files
    import MetaTrader5 as mt5
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    import h1_lab_v1 as lab
    from strategy_lab_v1 import first_touch, stats, trade

    cutoff = pd.Timestamp(config["cutoff_utc"]).tz_convert(None)
    end = datetime.now(timezone.utc) + timedelta(hours=1)
    if not mt5.initialize(path=os.environ.get("XM_TERMINAL_PATH", r"D:\XM2\terminal64.exe")):
        raise SystemExit(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        start = cutoff.to_pydatetime().replace(tzinfo=timezone.utc) - timedelta(days=60)
        h1 = lab.copy_rates("H1", start, end)
        higher = {tf: lab.copy_rates(tf, start - timedelta(days=900), end) for tf in lab.HIGHER}
    finally:
        mt5.shutdown()
    h1 = h1.iloc[:-1]  # drop the bar still forming
    frame = lab.make_features(h1, higher)
    spec = config["spec"]
    model = xgb.XGBClassifier()
    model.load_model(FROZEN / "model_xgb.json")
    scores = model.predict_proba(frame[config["features"]].astype(np.float32))[:, 1]
    r, after = first_touch(frame, spec["side"], spec["tp"], spec["sl"], spec["hold"])
    live = (frame["TIME_DT"] >= cutoff).to_numpy()
    f, s, rr, aa = frame[live], scores[live], r[live], after[live]
    taken = trade(f, s, config["threshold"], rr, aa)
    open_signals = f[(s >= config["threshold"]) & ~np.isfinite(rr)]  # signals whose trade has not resolved yet
    summary = {"cutoff_utc": config["cutoff_utc"], "bars_since_cutoff": int(live.sum()),
               "last_closed_bar": str(frame["TIME_DT"].iloc[-1]), "model": stats(taken["r"]),
               "always_long": stats(trade(f, np.ones(len(f)), 0.5, rr, aa)["r"]),
               "unresolved_signals": [str(t) for t in open_signals["TIME_DT"]],
               "development_test_for_reference": config["development_test"]}
    output.mkdir(parents=True, exist_ok=True)
    taken.assign(score=s[np.searchsorted(f.index, taken.index)] if len(taken) else [])[
        ["TIME_DT", "OPEN", "ATR", "score", "r"]].to_csv(output / "trades.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "forward_reports" / f"{datetime.now():%Y%m%d_%H%M%S}_h1_C_forward")
    args = parser.parse_args(argv)
    freeze() if args.freeze else check(args.output)


if __name__ == "__main__":
    main()
