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

from precious_metals_axis_research.axis_timeframe_smoke import load_case  # noqa: E402
from precious_metals_axis_research.cost_aware_xaueur_m5 import (  # noqa: E402
    get_symbol_point,
    simulate_cost_aware,
)
from precious_metals_axis_research.walk_forward_long_tf_cost import (  # noqa: E402
    FOLDS,
    compact_period,
    slice_by_ratio,
    train_fold_model,
)


SYMBOL = "SILVER#"
BASE_TIMEFRAME = "H1"
OUTPUT_CSV = RESEARCH_DIR / "silver_h1_stress_fold2_refine_results.csv"
OUTPUT_JSON = RESEARCH_DIR / "silver_h1_stress_fold2_refine_results.json"
OUTPUT_MD = RESEARCH_DIR / "silver_h1_stress_fold2_refine_report.md"
OUTPUT_BEST = RESEARCH_DIR / "silver_h1_stress_fold2_refine_best.json"
SPREAD_MULTIPLIERS = [1.0, 2.0, 3.0]


def make_grid():
    for threshold, edge, tp, sl, hold in product(
        [0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.62],
        [0.00, 0.02, 0.04, 0.06],
        [2.6, 2.8, 3.0, 3.2, 3.4],
        [3.4, 3.6, 3.8, 4.0, 4.4, 4.8],
        [72, 96, 120, 144, 168],
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


def scale_spread(frame, multiplier: float):
    adjusted = frame.copy()
    if "SPREAD" in adjusted.columns:
        adjusted["SPREAD"] = adjusted["SPREAD"].fillna(0) * multiplier
    return adjusted


def prepare_folds():
    point = get_symbol_point(SYMBOL)
    print(f"Loading {SYMBOL} {BASE_TIMEFRAME}; point={point}...")
    frame, features = load_case(SYMBOL, BASE_TIMEFRAME)
    prepared = []
    for fold in FOLDS:
        train_df = slice_by_ratio(frame, *fold["train"])
        test_df = slice_by_ratio(frame, *fold["test"])
        train_period = compact_period(train_df)
        test_period = compact_period(test_df)
        print(
            f"{fold['name']}: train={train_period['rows']:,} "
            f"test={test_period['rows']:,}"
        )
        model = train_fold_model(train_df, features)
        probs = model.predict_proba(test_df[features]).astype(np.float32)
        prepared.append(
            {
                "fold": fold["name"],
                "frames": {
                    multiplier: scale_spread(test_df, multiplier)
                    for multiplier in SPREAD_MULTIPLIERS
                },
                "probs": probs,
                "test_period": test_period,
                "point": point,
            }
        )
    return prepared


def fold_pass(stats: dict) -> bool:
    return (
        stats["pnl_r"] > 0
        and stats["profit_factor"] >= 1.12
        and stats["trades"] >= 8
        and stats["win_rate"] >= 0.55
    )


def summarize(folds: list[dict]) -> dict:
    total_trades = sum(row["test_trades"] for row in folds)
    weighted_win = (
        sum(row["test_win_rate"] * row["test_trades"] for row in folds) / total_trades
        if total_trades
        else 0.0
    )
    return {
        "total_pnl_r": round(sum(row["test_pnl_r"] for row in folds), 4),
        "total_trades": total_trades,
        "positive_folds": sum(row["test_pnl_r"] > 0 for row in folds),
        "passed_folds": sum(row["fold_pass"] for row in folds),
        "weighted_win_rate": round(weighted_win, 4),
        "mean_profit_factor": round(
            sum(row["test_profit_factor"] for row in folds) / len(folds), 4
        ),
        "worst_fold_pnl_r": round(min(row["test_pnl_r"] for row in folds), 4),
        "worst_fold_profit_factor": round(min(row["test_profit_factor"] for row in folds), 4),
        "max_drawdown_r": round(min(row["test_max_drawdown_r"] for row in folds), 4),
    }


def evaluate(params: dict, prepared_folds: list[dict]) -> dict:
    by_multiplier = {}
    for multiplier in SPREAD_MULTIPLIERS:
        fold_rows = []
        for fold in prepared_folds:
            stats = simulate_cost_aware(
                fold["frames"][multiplier], fold["probs"], params, fold["point"]
            )
            fold_rows.append(
                {
                    "fold": fold["fold"],
                    "test_start": fold["test_period"]["start"],
                    "test_end": fold["test_period"]["end"],
                    "test_pnl_r": stats["pnl_r"],
                    "test_trades": stats["trades"],
                    "test_win_rate": stats["win_rate"],
                    "test_profit_factor": stats["profit_factor"],
                    "test_max_drawdown_r": stats["max_drawdown_r"],
                    "test_avg_r": stats["avg_r"],
                    "test_max_loss_streak": stats["max_loss_streak"],
                    "fold_pass": fold_pass(stats),
                }
            )
        by_multiplier[str(multiplier)] = {**summarize(fold_rows), "folds": fold_rows}

    base = by_multiplier["1.0"]
    stress2 = by_multiplier["2.0"]
    stress3 = by_multiplier["3.0"]
    passes_strict_gate = (
        base["passed_folds"] == 4
        and stress2["passed_folds"] == 4
        and stress3["passed_folds"] == 4
        and stress3["total_pnl_r"] >= 12.0
        and stress3["weighted_win_rate"] >= 0.58
        and stress3["max_drawdown_r"] >= -16.0
        and stress3["total_trades"] >= 100
    )
    score = (
        stress3["passed_folds"] * 1200.0
        + stress3["total_pnl_r"] * 150.0
        + stress2["total_pnl_r"] * 60.0
        + base["total_pnl_r"] * 25.0
        + stress3["weighted_win_rate"] * 700.0
        + min(stress3["mean_profit_factor"], 2.8) * 260.0
        + stress3["worst_fold_profit_factor"] * 500.0
        - abs(stress3["max_drawdown_r"]) * 50.0
        + min(stress3["worst_fold_pnl_r"], 0.0) * 150.0
    )
    return {
        **params,
        "passes_strict_gate": passes_strict_gate,
        "score": round(score, 4),
        "base_total_pnl_r": base["total_pnl_r"],
        "stress2_total_pnl_r": stress2["total_pnl_r"],
        "stress3_total_pnl_r": stress3["total_pnl_r"],
        "stress3_total_trades": stress3["total_trades"],
        "stress3_positive_folds": stress3["positive_folds"],
        "stress3_passed_folds": stress3["passed_folds"],
        "stress3_weighted_win_rate": stress3["weighted_win_rate"],
        "stress3_mean_profit_factor": stress3["mean_profit_factor"],
        "stress3_worst_fold_pnl_r": stress3["worst_fold_pnl_r"],
        "stress3_worst_fold_profit_factor": stress3["worst_fold_profit_factor"],
        "stress3_max_drawdown_r": stress3["max_drawdown_r"],
        "multipliers": by_multiplier,
    }


def write_outputs(results: list[dict]) -> None:
    results = sorted(
        results,
        key=lambda row: (
            row["passes_strict_gate"],
            row["stress3_passed_folds"],
            row["score"],
            row["stress3_total_pnl_r"],
        ),
        reverse=True,
    )
    flat_rows = []
    for rank, row in enumerate(results, start=1):
        flat = {key: value for key, value in row.items() if key != "multipliers"}
        flat["rank"] = rank
        flat_rows.append(flat)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    OUTPUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    OUTPUT_BEST.write_text(json.dumps(results[0], indent=2), encoding="utf-8")

    lines = [
        "# SILVER H1 Fold-2 Stress Refinement",
        "",
        "Searches around the stress-aware area and prioritizes 3x spread fold-pass consistency.",
        "",
        "| Rank | Strict | Score | 1x R | 2x R | 3x R | 3x Passed | 3x Trades | 3x Win | 3x PF | 3x Worst PF | 3x Worst R | 3x DD | Params |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(results[:25], start=1):
        lines.append(
            "| {rank} | {passes_strict_gate} | {score:.1f} | {base_total_pnl_r:.2f} | "
            "{stress2_total_pnl_r:.2f} | {stress3_total_pnl_r:.2f} | "
            "{stress3_passed_folds}/4 | {stress3_total_trades} | "
            "{stress3_weighted_win_rate:.2%} | {stress3_mean_profit_factor:.2f} | "
            "{stress3_worst_fold_profit_factor:.2f} | {stress3_worst_fold_pnl_r:.2f} | "
            "{stress3_max_drawdown_r:.2f} | conf={threshold}, edge={edge_threshold}, "
            "tp/sl={tp_atr}/{sl_atr}, hold={max_hold} |".format(rank=rank, **row)
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    prepared_folds = prepare_folds()
    results = []
    total = 0
    for params in make_grid():
        total += 1
        results.append(evaluate(params, prepared_folds))
    write_outputs(results)
    strict = sum(1 for row in results if row["passes_strict_gate"])
    print(f"Swept {total} candidates, strict passed {strict}.")
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_BEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
