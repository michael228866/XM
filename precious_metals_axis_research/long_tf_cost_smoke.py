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
import xgboost as xgb  # noqa: E402

from barrier_classifier_strategy import build_profit_sample_weight  # noqa: E402
from precious_metals_axis_research.axis_timeframe_smoke import (  # noqa: E402
    TRAIN_END_RATIO,
    VALIDATION_END_RATIO,
    load_case,
)
from precious_metals_axis_research.cost_aware_xaueur_m5 import (  # noqa: E402
    get_symbol_point,
    simulate_cost_aware,
)


CASES = [
    ("SILVER#", "H1"),
    ("XAUEUR#", "H1"),
    ("XPTUSD#", "H1"),
    ("XPDUSD#", "H1"),
]
OUTPUT_CSV = RESEARCH_DIR / "long_tf_cost_smoke_results.csv"
OUTPUT_JSON = RESEARCH_DIR / "long_tf_cost_smoke_results.json"
OUTPUT_MD = RESEARCH_DIR / "long_tf_cost_smoke_report.md"
OUTPUT_BEST = RESEARCH_DIR / "long_tf_cost_best_candidates.json"


def make_grid():
    for threshold, edge, tp, sl, hold, direction in product(
        [0.54, 0.57, 0.60, 0.63, 0.66],
        [0.00, 0.05, 0.10, 0.15],
        [1.0, 1.2, 1.5, 1.8, 2.0],
        [1.8, 2.2, 2.6, 3.0, 3.4],
        [24, 48, 72, 120],
        ["both", "long", "short"],
    ):
        if tp > sl:
            continue
        yield {
            "threshold": threshold,
            "edge_threshold": edge,
            "tp_atr": tp,
            "sl_atr": sl,
            "max_hold": hold,
            "direction_mode": direction,
        }


def passes_gate(validation: dict, test: dict) -> bool:
    return (
        validation["pnl_r"] > 0
        and validation["profit_factor"] >= 1.05
        and validation["trades"] >= 12
        and test["pnl_r"] > 0
        and test["profit_factor"] >= 1.20
        and test["win_rate"] >= 0.55
        and test["trades"] >= 12
        and abs(test["max_drawdown_r"]) <= 15.0
    )


def score_row(row: dict) -> float:
    if row["test_trades"] < 12:
        return -100_000.0 + row["test_pnl_r"]
    return (
        row["test_pnl_r"] * 130.0
        + row["validation_pnl_r"] * 70.0
        + row["test_win_rate"] * 260.0
        + min(row["test_profit_factor"], 3.0) * 160.0
        - abs(row["test_max_drawdown_r"]) * 22.0
    )


def train_safe_model(train_df, features):
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
        n_estimators=170,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=80,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbosity=0,
    )
    model.fit(train_df[features], train_df["BARRIER_TARGET"], sample_weight=sample_weight)
    return model


def run_case(symbol: str, base_tf: str) -> list[dict]:
    point = get_symbol_point(symbol)
    print(f"Loading {symbol} {base_tf}; point={point}...")
    frame, features = load_case(symbol, base_tf)
    train_end = int(len(frame) * TRAIN_END_RATIO)
    validation_end = int(len(frame) * VALIDATION_END_RATIO)
    train_df = frame.iloc[:train_end].copy()
    validation_df = frame.iloc[train_end:validation_end].copy()
    test_df = frame.iloc[validation_end:].copy()
    print(
        f"{symbol} {base_tf}: train={len(train_df):,} "
        f"validation={len(validation_df):,} test={len(test_df):,}"
    )

    model = train_safe_model(train_df, features)
    validation_probs = model.predict_proba(validation_df[features]).astype(np.float32)
    test_probs = model.predict_proba(test_df[features]).astype(np.float32)

    rows = []
    total = 0
    for params in make_grid():
        total += 1
        validation = simulate_cost_aware(validation_df, validation_probs, params, point)
        test = simulate_cost_aware(test_df, test_probs, params, point)
        row = {
            "symbol": symbol,
            "base_timeframe": base_tf,
            **params,
            "validation_pnl_r": validation["pnl_r"],
            "validation_trades": validation["trades"],
            "validation_win_rate": validation["win_rate"],
            "validation_profit_factor": validation["profit_factor"],
            "validation_max_drawdown_r": validation["max_drawdown_r"],
            "test_pnl_r": test["pnl_r"],
            "test_trades": test["trades"],
            "test_win_rate": test["win_rate"],
            "test_profit_factor": test["profit_factor"],
            "test_max_drawdown_r": test["max_drawdown_r"],
            "test_avg_r": test["avg_r"],
            "test_max_loss_streak": test["max_loss_streak"],
            "passes_gate": passes_gate(validation, test),
        }
        row["score"] = round(score_row(row), 4)
        rows.append(row)
    print(f"{symbol} {base_tf}: swept {total}")
    return rows


def write_outputs(rows: list[dict]) -> None:
    rows = sorted(
        rows,
        key=lambda row: (row["passes_gate"], row["score"], row["test_pnl_r"]),
        reverse=True,
    )
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    OUTPUT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    best_by_symbol = {}
    for row in rows:
        best_by_symbol.setdefault(row["symbol"], row)
    OUTPUT_BEST.write_text(json.dumps(best_by_symbol, indent=2), encoding="utf-8")

    lines = [
        "# Long Timeframe Cost-Aware Smoke",
        "",
        "Research-only H1 tests using CSV spread cost and R-multiple evaluation.",
        "",
        "| Rank | Symbol | Pass | Score | Test R | Test Win | Test PF | Trades | Val R | Val PF | Params |",
        "|---:|---|:---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(rows[:24], start=1):
        lines.append(
            "| {rank} | {symbol} | {passes_gate} | {score:.1f} | {test_pnl_r:.2f} | "
            "{test_win_rate:.2%} | {test_profit_factor:.2f} | {test_trades} | "
            "{validation_pnl_r:.2f} | {validation_profit_factor:.2f} | "
            "conf={threshold}, edge={edge_threshold}, tp/sl={tp_atr}/{sl_atr}, "
            "hold={max_hold}, dir={direction_mode} |".format(rank=rank, **row)
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = []
    for symbol, base_tf in CASES:
        rows.extend(run_case(symbol, base_tf))
    write_outputs(rows)
    passed = sum(1 for row in rows if row["passes_gate"])
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_BEST}")
    print(f"Long timeframe candidates passed: {passed}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
