from __future__ import annotations

import csv
import json
import os
import sys
from itertools import product
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

RESEARCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RESEARCH_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402

from barrier_classifier_strategy import build_profit_sample_weight  # noqa: E402
from precious_metals_axis_research.axis_timeframe_smoke import load_case  # noqa: E402
from precious_metals_axis_research.cost_aware_xaueur_m5 import (  # noqa: E402
    get_symbol_point,
    simulate_cost_aware,
)


SYMBOLS = ["GOLD#", "SILVER#", "XAUEUR#", "XPTUSD#", "XPDUSD#", "GAUCNH#"]
BASE_TIMEFRAME = "H1"
TRAIN_END_RATIO = 0.70
VALIDATION_END_RATIO = 0.92
MIN_ROWS = 700
MAX_ROWS_PER_SYMBOL = 90_000

OUTPUT_CSV = RESEARCH_DIR / "all_metals_h1_shared_results.csv"
OUTPUT_JSON = RESEARCH_DIR / "all_metals_h1_shared_results.json"
OUTPUT_MD = RESEARCH_DIR / "all_metals_h1_shared_report.md"
OUTPUT_MODEL = RESEARCH_DIR / "all_metals_h1_shared_xgb.json"


def make_param_grid() -> list[dict]:
    params: list[dict] = []
    for threshold, edge, tp, sl, hold, direction in product(
        [0.52, 0.54, 0.56, 0.58, 0.60],
        [0.00, 0.03, 0.06],
        [1.8, 2.2, 2.8, 3.2],
        [2.8, 3.4, 4.0, 4.4],
        [48, 72, 120],
        ["long", "both"],
    ):
        if tp > sl:
            continue
        params.append(
            {
                "threshold": threshold,
                "edge_threshold": edge,
                "tp_atr": tp,
                "sl_atr": sl,
                "max_hold": hold,
                "direction_mode": direction,
            }
        )
    return params


def add_symbol_features(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["SYMBOL_ID"] = float(SYMBOLS.index(symbol))
    for candidate in SYMBOLS:
        clean = candidate.replace("#", "").replace("/", "")
        frame[f"IS_{clean}"] = 1.0 if candidate == symbol else 0.0
    return frame


def load_symbol_splits() -> tuple[pd.DataFrame, dict[str, dict], list[str], list[dict]]:
    splits: dict[str, dict] = {}
    skipped: list[dict] = []
    feature_union: set[str] = set()
    loaded_frames: dict[str, tuple[pd.DataFrame, list[str]]] = {}

    for symbol in SYMBOLS:
        try:
            frame, features = load_case(symbol, BASE_TIMEFRAME)
        except Exception as exc:
            skipped.append({"symbol": symbol, "reason": str(exc)})
            continue
        if len(frame) < MIN_ROWS:
            skipped.append({"symbol": symbol, "reason": f"too few rows: {len(frame)}"})
            continue
        if len(frame) > MAX_ROWS_PER_SYMBOL:
            frame = frame.tail(MAX_ROWS_PER_SYMBOL).reset_index(drop=True)
        frame = add_symbol_features(frame, symbol)
        symbol_features = features + ["SYMBOL_ID"] + [
            f"IS_{candidate.replace('#', '').replace('/', '')}" for candidate in SYMBOLS
        ]
        feature_union.update(symbol_features)
        loaded_frames[symbol] = (frame, symbol_features)

    if not loaded_frames:
        raise ValueError("No symbols were loaded.")

    shared_features = sorted(feature_union)
    train_frames = []
    for symbol, (frame, symbol_features) in loaded_frames.items():
        missing = sorted(set(shared_features) - set(symbol_features))
        for feature in missing:
            frame[feature] = 0.0
        frame = frame.sort_values("TIME_DT").reset_index(drop=True)
        train_end = int(len(frame) * TRAIN_END_RATIO)
        validation_end = int(len(frame) * VALIDATION_END_RATIO)
        splits[symbol] = {
            "train": frame.iloc[:train_end].copy(),
            "validation": frame.iloc[train_end:validation_end].copy(),
            "test": frame.iloc[validation_end:].copy(),
            "point": get_symbol_point(symbol),
            "rows": len(frame),
        }
        train_frames.append(splits[symbol]["train"])
        print(
            f"{symbol}: rows={len(frame):,} train={len(splits[symbol]['train']):,} "
            f"validation={len(splits[symbol]['validation']):,} "
            f"test={len(splits[symbol]['test']):,}"
        )

    train_df = pd.concat(train_frames, ignore_index=True)
    train_df[shared_features] = train_df[shared_features].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return train_df, splits, shared_features, skipped


def train_shared_model(train_df: pd.DataFrame, features: list[str]) -> xgb.XGBClassifier:
    sample_weight = build_profit_sample_weight(
        train_df, train_df["BARRIER_TARGET"].to_numpy(dtype=np.int8)
    )
    sample_weight = np.nan_to_num(sample_weight, nan=1.0, posinf=1.0, neginf=1.0)
    sample_weight = np.maximum(sample_weight, 1e-6)
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        device="cpu",
        n_estimators=240,
        learning_rate=0.04,
        max_depth=4,
        min_child_weight=120,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbosity=0,
    )
    model.fit(train_df[features], train_df["BARRIER_TARGET"], sample_weight=sample_weight)
    model.save_model(OUTPUT_MODEL)
    return model


def scale_spread(frame: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    adjusted = frame.copy()
    if "SPREAD" in adjusted.columns:
        adjusted["SPREAD"] = adjusted["SPREAD"].fillna(0) * multiplier
    return adjusted


def evaluate_grid(model: xgb.XGBClassifier, splits: dict[str, dict], features: list[str]) -> list[dict]:
    rows = []
    grid = make_param_grid()
    for symbol, split in splits.items():
        validation_df = split["validation"].copy()
        test_df = split["test"].copy()
        validation_df[features] = validation_df[features].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        test_df[features] = test_df[features].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        validation_probs = model.predict_proba(validation_df[features]).astype(np.float32)
        test_probs = model.predict_proba(test_df[features]).astype(np.float32)
        stress_test_df = scale_spread(test_df, 3.0)

        for params in grid:
            validation_stats = simulate_cost_aware(
                validation_df, validation_probs, params, split["point"]
            )
            test_stats = simulate_cost_aware(test_df, test_probs, params, split["point"])
            stress_stats = simulate_cost_aware(
                stress_test_df, test_probs, params, split["point"]
            )
            passes = (
                validation_stats["pnl_r"] > 0
                and test_stats["pnl_r"] > 0
                and stress_stats["pnl_r"] > 0
                and test_stats["trades"] >= 12
                and stress_stats["profit_factor"] >= 1.12
                and stress_stats["win_rate"] >= 0.55
            )
            score = (
                stress_stats["pnl_r"] * 140.0
                + test_stats["pnl_r"] * 70.0
                + validation_stats["pnl_r"] * 35.0
                + stress_stats["win_rate"] * 500.0
                + min(stress_stats["profit_factor"], 3.0) * 160.0
                - abs(stress_stats["max_drawdown_r"]) * 45.0
            )
            rows.append(
                {
                    "symbol": symbol,
                    "base_timeframe": BASE_TIMEFRAME,
                    **params,
                    "validation_pnl_r": validation_stats["pnl_r"],
                    "validation_trades": validation_stats["trades"],
                    "validation_win_rate": validation_stats["win_rate"],
                    "validation_profit_factor": validation_stats["profit_factor"],
                    "test_pnl_r": test_stats["pnl_r"],
                    "test_trades": test_stats["trades"],
                    "test_win_rate": test_stats["win_rate"],
                    "test_profit_factor": test_stats["profit_factor"],
                    "test_max_drawdown_r": test_stats["max_drawdown_r"],
                    "stress3_pnl_r": stress_stats["pnl_r"],
                    "stress3_trades": stress_stats["trades"],
                    "stress3_win_rate": stress_stats["win_rate"],
                    "stress3_profit_factor": stress_stats["profit_factor"],
                    "stress3_max_drawdown_r": stress_stats["max_drawdown_r"],
                    "passes_gate": passes,
                    "score": round(score, 4),
                }
            )
    return sorted(
        rows,
        key=lambda row: (row["passes_gate"], row["score"], row["stress3_pnl_r"]),
        reverse=True,
    )


def summarize(rows: list[dict], skipped: list[dict]) -> dict:
    summary = {}
    for symbol in sorted({row["symbol"] for row in rows}):
        subset = [row for row in rows if row["symbol"] == symbol]
        best = subset[0]
        passed = [row for row in subset if row["passes_gate"]]
        summary[symbol] = {
            "passed_candidates": len(passed),
            "best": best,
            "verdict": "shared_model_candidate" if passed else "trained_but_not_passed",
        }
    return {"symbols": summary, "skipped": skipped}


def write_outputs(rows: list[dict], summary: dict) -> None:
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    OUTPUT_JSON.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8"
    )

    lines = [
        "# All Metals H1 Shared Training",
        "",
        "Research-only shared H1 model. All available precious metals participate in training.",
        "",
        "| Symbol | Verdict | Passed | Best Stress R | Stress Win | Stress PF | Stress DD | Params |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for symbol, item in summary["symbols"].items():
        best = item["best"]
        lines.append(
            "| {row_symbol} | {verdict} | {passed_candidates} | {stress3_pnl_r:.2f} | "
            "{stress3_win_rate:.2%} | {stress3_profit_factor:.2f} | "
            "{stress3_max_drawdown_r:.2f} | conf={threshold}, edge={edge_threshold}, "
            "tp/sl={tp_atr}/{sl_atr}, hold={max_hold}, dir={direction_mode} |".format(
                row_symbol=symbol,
                verdict=item["verdict"],
                passed_candidates=item["passed_candidates"],
                **best,
            )
        )
    if summary["skipped"]:
        lines.extend(["", "## Skipped", ""])
        for item in summary["skipped"]:
            lines.append(f"- {item['symbol']}: {item['reason']}")
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    train_df, splits, features, skipped = load_symbol_splits()
    print(f"Training all-metals H1 shared model on {len(train_df):,} rows...")
    model = train_shared_model(train_df, features)
    rows = evaluate_grid(model, splits, features)
    summary = summarize(rows, skipped)
    write_outputs(rows, summary)
    passed = sum(1 for row in rows if row["passes_gate"])
    print(f"Wrote {OUTPUT_MODEL}")
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Passed candidates: {passed}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
