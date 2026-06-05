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

from precious_metals_axis_research.optimize_silver_regime_readiness import (  # noqa: E402
    evaluate,
    prepare_folds,
)


OUTPUT_CSV = RESEARCH_DIR / "silver_regime_refine_results.csv"
OUTPUT_JSON = RESEARCH_DIR / "silver_regime_refine_results.json"
OUTPUT_MD = RESEARCH_DIR / "silver_regime_refine_report.md"
OUTPUT_BEST = RESEARCH_DIR / "silver_regime_refine_best.json"

FILTERS = [
    {
        "filter_name": "trend_high_r",
        "trend_min": 0.34,
        "rsi_min": 0.0,
        "rsi_max": 100.0,
        "vola_max": 2.5,
        "spread_atr_max": 1.25,
        "macd_min": -999.0,
    },
    {
        "filter_name": "trend_high_r_tighter_spread",
        "trend_min": 0.34,
        "rsi_min": 0.0,
        "rsi_max": 100.0,
        "vola_max": 2.5,
        "spread_atr_max": 0.75,
        "macd_min": -999.0,
    },
    {
        "filter_name": "balanced_rsi75",
        "trend_min": 0.0,
        "rsi_min": 0.0,
        "rsi_max": 75.0,
        "vola_max": 2.5,
        "spread_atr_max": 0.45,
        "macd_min": -999.0,
    },
    {
        "filter_name": "balanced_low_vola",
        "trend_min": 0.0,
        "rsi_min": 0.0,
        "rsi_max": 100.0,
        "vola_max": 1.2,
        "spread_atr_max": 0.75,
        "macd_min": -999.0,
    },
    {
        "filter_name": "trend_rsi85",
        "trend_min": 0.34,
        "rsi_min": 0.0,
        "rsi_max": 85.0,
        "vola_max": 2.5,
        "spread_atr_max": 1.25,
        "macd_min": -999.0,
    },
    {
        "filter_name": "trend_low_vola",
        "trend_min": 0.34,
        "rsi_min": 0.0,
        "rsi_max": 100.0,
        "vola_max": 1.6,
        "spread_atr_max": 1.25,
        "macd_min": -999.0,
    },
]


def params_grid():
    for threshold, edge, tp, sl, hold in product(
        [0.50, 0.52, 0.54, 0.56, 0.58, 0.60],
        [0.0, 0.05],
        [2.6, 3.2, 3.8, 4.4, 5.2, 6.0],
        [5.2, 6.0, 7.0, 8.5, 10.0],
        [216, 336, 504, 720],
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


def write_outputs(results: list[dict]) -> None:
    results = sorted(
        results,
        key=lambda row: (
            row["battle_gate"],
            row["stress_total_r"],
            row["stress_total_trades"],
            row["score"],
        ),
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

    best = results[0]
    lines = [
        "# SILVER Regime Refine",
        "",
        "Focused refinement around the best SILVER# regime filters.",
        "",
        "| Gate | R | Positive | Passed | Trades | Win | PF | Worst R | DD | Recent R | Params |",
        "|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        (
            "| {battle_gate} | {stress_total_r:.2f} | {stress_positive_folds}/5 | "
            "{stress_passed_folds}/5 | {stress_total_trades} | "
            "{stress_weighted_win_rate:.2%} | {stress_mean_profit_factor:.2f} | "
            "{stress_worst_fold_r:.2f} | {stress_max_drawdown_r:.2f} | {recent_paper_r:.2f} | "
            "{filter_name}: conf={threshold}, edge={edge_threshold}, "
            "tp/sl={tp_atr}/{sl_atr}, hold={max_hold} |"
        ).format(**best),
        "",
        "## Fold Detail",
        "",
        "| Fold | R | Trades | Win | PF | DD | Pass |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for fold in best["folds"]:
        lines.append(
            "| {fold} | {pnl_r:.2f} | {trades} | {win_rate:.2%} | "
            "{profit_factor:.2f} | {max_drawdown_r:.2f} | {fold_pass} |".format(**fold)
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    prepared = prepare_folds()
    results = []
    print("Refining SILVER regime candidates...")
    for filters in FILTERS:
        filter_name = filters["filter_name"]
        filter_params = {key: value for key, value in filters.items() if key != "filter_name"}
        for params in params_grid():
            result = evaluate({**params, **filter_params}, prepared)
            result["filter_name"] = filter_name
            results.append(result)
    write_outputs(results)
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_BEST}")
    print(f"Battle-gate candidates: {sum(row['battle_gate'] for row in results)}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
