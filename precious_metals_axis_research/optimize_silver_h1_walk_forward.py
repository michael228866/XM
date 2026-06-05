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
OUTPUT_CSV = RESEARCH_DIR / "silver_h1_walk_forward_optimized_results.csv"
OUTPUT_JSON = RESEARCH_DIR / "silver_h1_walk_forward_optimized_results.json"
OUTPUT_MD = RESEARCH_DIR / "silver_h1_walk_forward_optimized_report.md"
OUTPUT_BEST = RESEARCH_DIR / "silver_h1_walk_forward_best_candidate.json"


def make_grid():
    for threshold, edge, tp, sl, hold in product(
        [0.54, 0.56, 0.58, 0.60, 0.62],
        [0.00, 0.03, 0.05, 0.08],
        [2.0, 2.2, 2.4, 2.6],
        [2.8, 3.0, 3.2, 3.4],
        [48, 72, 96, 120],
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
                "test_df": test_df,
                "probs": probs,
                "test_period": test_period,
                "point": point,
            }
        )
    return prepared


def evaluate_candidate(params: dict, prepared_folds: list[dict]) -> dict:
    fold_rows = []
    for fold in prepared_folds:
        stats = simulate_cost_aware(
            fold["test_df"], fold["probs"], params, fold["point"]
        )
        fold_pass = (
            stats["pnl_r"] > 0
            and stats["profit_factor"] >= 1.12
            and stats["trades"] >= 8
            and stats["win_rate"] >= 0.55
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
                "fold_pass": fold_pass,
            }
        )

    total_r = sum(row["test_pnl_r"] for row in fold_rows)
    total_trades = sum(row["test_trades"] for row in fold_rows)
    positive_folds = sum(row["test_pnl_r"] > 0 for row in fold_rows)
    passed_folds = sum(row["fold_pass"] for row in fold_rows)
    weighted_win = (
        sum(row["test_win_rate"] * row["test_trades"] for row in fold_rows)
        / total_trades
        if total_trades
        else 0.0
    )
    worst_fold_r = min(row["test_pnl_r"] for row in fold_rows)
    max_dd_r = min(row["test_max_drawdown_r"] for row in fold_rows)
    mean_pf = sum(row["test_profit_factor"] for row in fold_rows) / len(fold_rows)

    score = (
        total_r * 120.0
        + positive_folds * 350.0
        + passed_folds * 180.0
        + weighted_win * 500.0
        + min(mean_pf, 2.5) * 160.0
        - abs(max_dd_r) * 45.0
        + min(worst_fold_r, 0.0) * 90.0
    )
    passes_gate = (
        positive_folds == 4
        and total_r > 18.0
        and weighted_win >= 0.58
        and max_dd_r >= -10.0
        and total_trades >= 120
    )

    return {
        **params,
        "total_pnl_r": round(total_r, 4),
        "total_trades": total_trades,
        "positive_folds": positive_folds,
        "passed_folds": passed_folds,
        "weighted_win_rate": round(weighted_win, 4),
        "mean_profit_factor": round(mean_pf, 4),
        "worst_fold_pnl_r": round(worst_fold_r, 4),
        "max_drawdown_r": round(max_dd_r, 4),
        "passes_gate": passes_gate,
        "score": round(score, 4),
        "folds": fold_rows,
    }


def write_outputs(results: list[dict]) -> None:
    results = sorted(
        results,
        key=lambda row: (row["passes_gate"], row["score"], row["total_pnl_r"]),
        reverse=True,
    )
    flat_rows = []
    for rank, row in enumerate(results, start=1):
        flat = {key: value for key, value in row.items() if key != "folds"}
        flat["rank"] = rank
        flat_rows.append(flat)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    OUTPUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    OUTPUT_BEST.write_text(json.dumps(results[0], indent=2), encoding="utf-8")

    lines = [
        "# SILVER H1 Walk-Forward Optimization",
        "",
        "Optimizes directly on cost-aware walk-forward folds.",
        "",
        "| Rank | Pass | Score | Total R | Positive | Passed | Trades | Win | Mean PF | Worst R | Max DD | Params |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(results[:25], start=1):
        lines.append(
            "| {rank} | {passes_gate} | {score:.1f} | {total_pnl_r:.2f} | "
            "{positive_folds}/4 | {passed_folds}/4 | {total_trades} | "
            "{weighted_win_rate:.2%} | {mean_profit_factor:.2f} | "
            "{worst_fold_pnl_r:.2f} | {max_drawdown_r:.2f} | "
            "conf={threshold}, edge={edge_threshold}, tp/sl={tp_atr}/{sl_atr}, "
            "hold={max_hold} |".format(rank=rank, **row)
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    prepared_folds = prepare_folds()
    results = []
    total = 0
    for params in make_grid():
        total += 1
        results.append(evaluate_candidate(params, prepared_folds))
    write_outputs(results)
    passed = sum(1 for row in results if row["passes_gate"])
    print(f"Swept {total} candidates, passed {passed}.")
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_BEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
