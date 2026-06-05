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

from precious_metals_axis_research.axis_timeframe_smoke import (  # noqa: E402
    TRAIN_END_RATIO,
    VALIDATION_END_RATIO,
    load_case,
)
from precious_metals_axis_research.cost_aware_xaueur_m5 import (  # noqa: E402
    get_symbol_point,
    simulate_cost_aware,
)
from precious_metals_axis_research.long_tf_cost_smoke import train_safe_model  # noqa: E402


SYMBOL = "SILVER#"
BASE_TIMEFRAME = "H1"
OUTPUT_CSV = RESEARCH_DIR / "silver_h1_cost_optimized_results.csv"
OUTPUT_JSON = RESEARCH_DIR / "silver_h1_cost_optimized_results.json"
OUTPUT_MD = RESEARCH_DIR / "silver_h1_cost_optimized_report.md"
OUTPUT_BEST = RESEARCH_DIR / "silver_h1_cost_best_candidate.json"


def make_grid():
    for threshold, edge, tp, sl, hold in product(
        [0.56, 0.58, 0.60, 0.62, 0.64, 0.66],
        [0.00, 0.03, 0.05, 0.08, 0.10],
        [1.6, 1.8, 2.0, 2.2, 2.4],
        [2.8, 3.0, 3.2, 3.4, 3.8],
        [72, 96, 120, 168],
    ):
        if tp > sl:
            continue
        yield {
            "threshold": threshold,
            "edge_threshold": edge,
            "tp_atr": tp,
            "sl_atr": sl,
            "max_hold": hold,
            "direction_mode": "long",
        }


def passes_gate(validation: dict, test: dict) -> bool:
    return (
        validation["pnl_r"] > 0
        and validation["profit_factor"] >= 1.08
        and validation["trades"] >= 20
        and test["pnl_r"] > 0
        and test["profit_factor"] >= 1.25
        and test["win_rate"] >= 0.60
        and test["trades"] >= 30
        and abs(test["max_drawdown_r"]) <= 8.0
    )


def score_row(row: dict) -> float:
    if row["test_trades"] < 30:
        return -100_000.0 + row["test_pnl_r"]
    return (
        row["test_pnl_r"] * 140.0
        + row["validation_pnl_r"] * 75.0
        + row["test_win_rate"] * 300.0
        + min(row["test_profit_factor"], 3.0) * 190.0
        - abs(row["test_max_drawdown_r"]) * 35.0
    )


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
    OUTPUT_BEST.write_text(json.dumps(rows[0], indent=2), encoding="utf-8")

    lines = [
        "# SILVER H1 Cost-Aware Optimization",
        "",
        "Research-only optimization around the current SILVER# H1 long-only candidate.",
        "",
        "| Rank | Pass | Score | Test R | Test Win | Test PF | Test Trades | Val R | Val PF | Params |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(rows[:25], start=1):
        lines.append(
            "| {rank} | {passes_gate} | {score:.1f} | {test_pnl_r:.2f} | "
            "{test_win_rate:.2%} | {test_profit_factor:.2f} | {test_trades} | "
            "{validation_pnl_r:.2f} | {validation_profit_factor:.2f} | "
            "conf={threshold}, edge={edge_threshold}, tp/sl={tp_atr}/{sl_atr}, "
            "hold={max_hold}, dir={direction_mode} |".format(rank=rank, **row)
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    point = get_symbol_point(SYMBOL)
    print(f"Loading {SYMBOL} {BASE_TIMEFRAME}; point={point}...")
    frame, features = load_case(SYMBOL, BASE_TIMEFRAME)
    train_end = int(len(frame) * TRAIN_END_RATIO)
    validation_end = int(len(frame) * VALIDATION_END_RATIO)
    train_df = frame.iloc[:train_end].copy()
    validation_df = frame.iloc[train_end:validation_end].copy()
    test_df = frame.iloc[validation_end:].copy()
    print(
        f"Rows train={len(train_df):,} validation={len(validation_df):,} "
        f"test={len(test_df):,}"
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
            "symbol": SYMBOL,
            "base_timeframe": BASE_TIMEFRAME,
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

    write_outputs(rows)
    passed = sum(1 for row in rows if row["passes_gate"])
    print(f"Swept {total} candidates, passed {passed}.")
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_BEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
