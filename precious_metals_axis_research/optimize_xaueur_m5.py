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
    evaluate_case,
    load_case,
    train_model,
)


SYMBOL = "XAUEUR#"
BASE_TIMEFRAME = "M5"
OUTPUT_CSV = RESEARCH_DIR / "xaueur_m5_optimized_results.csv"
OUTPUT_JSON = RESEARCH_DIR / "xaueur_m5_optimized_results.json"
OUTPUT_MD = RESEARCH_DIR / "xaueur_m5_optimized_report.md"
OUTPUT_BEST = RESEARCH_DIR / "xaueur_m5_best_candidate.json"


def make_grid():
    for threshold, edge, tp, sl, hold, direction in product(
        [0.54, 0.56, 0.575, 0.59, 0.61, 0.63],
        [0.05, 0.08, 0.10, 0.12, 0.15],
        [1.2, 1.3, 1.4, 1.5, 1.6],
        [2.0, 2.2, 2.3, 2.5, 2.8],
        [180, 240, 300, 360],
        ["both", "short"],
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


def score_row(row: dict) -> float:
    if row["test_trades"] < 30:
        return -100_000.0 + row["test_pnl"]
    validation_penalty = max(0.0, 1.0 - row["validation_profit_factor"]) * 700.0
    return (
        row["test_pnl"]
        + row["validation_pnl"] * 0.45
        + row["test_win_rate"] * 650.0
        + min(row["test_profit_factor"], 3.0) * 260.0
        - abs(min(row["test_drawdown_pct"], 0.0)) * 1300.0
        - validation_penalty
    )


def passes_gate(validation_stats: dict, test_stats: dict) -> bool:
    return (
        validation_stats["pnl"] > 0
        and validation_stats["profit_factor"] >= 1.0
        and test_stats["pnl"] > 0
        and test_stats["profit_factor"] >= 1.25
        and test_stats["win_rate"] >= 0.62
        and test_stats["trades"] >= 30
        and abs(min(test_stats["max_drawdown_pct"], 0.0)) <= 0.20
        and not validation_stats["stopped_out"]
        and not test_stats["stopped_out"]
    )


def write_outputs(rows: list[dict]) -> None:
    rows = sorted(
        rows,
        key=lambda row: (row["passes_gate"], row["score"], row["test_pnl"]),
        reverse=True,
    )
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    OUTPUT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    best = rows[0]
    OUTPUT_BEST.write_text(json.dumps(best, indent=2), encoding="utf-8")

    lines = [
        "# XAUEUR M5 Optimization",
        "",
        "Research-only optimization around the best XAUEUR# M5 region.",
        "",
        "| Rank | Pass | Score | Test PnL | Test Win | Test PF | Test Trades | Val PnL | Val PF | Params |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(rows[:20], start=1):
        lines.append(
            "| {rank} | {passes_gate} | {score:.1f} | {test_pnl:.2f} | "
            "{test_win_rate:.2%} | {test_profit_factor:.2f} | {test_trades} | "
            "{validation_pnl:.2f} | {validation_profit_factor:.2f} | "
            "conf={threshold}, edge={edge_threshold}, tp/sl={tp_atr}/{sl_atr}, "
            "hold={max_hold}, dir={direction_mode} |".format(rank=rank, **row)
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    print(f"Loading {SYMBOL} {BASE_TIMEFRAME}...")
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

    model = train_model(train_df, features)
    validation_probs = model.predict_proba(validation_df[features]).astype(np.float32)
    test_probs = model.predict_proba(test_df[features]).astype(np.float32)

    rows = []
    total = 0
    for params in make_grid():
        total += 1
        validation_stats = evaluate_case(params, validation_df, validation_probs)
        test_stats = evaluate_case(params, test_df, test_probs)
        row = {
            "symbol": SYMBOL,
            "base_timeframe": BASE_TIMEFRAME,
            **params,
            "validation_pnl": validation_stats["pnl"],
            "validation_trades": validation_stats["trades"],
            "validation_win_rate": validation_stats["win_rate"],
            "validation_profit_factor": validation_stats["profit_factor"],
            "validation_drawdown_pct": validation_stats["max_drawdown_pct"],
            "test_pnl": test_stats["pnl"],
            "test_trades": test_stats["trades"],
            "test_win_rate": test_stats["win_rate"],
            "test_profit_factor": test_stats["profit_factor"],
            "test_drawdown_pct": test_stats["max_drawdown_pct"],
            "test_max_loss_streak": test_stats["max_consecutive_losses"],
            "passes_gate": passes_gate(validation_stats, test_stats),
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
