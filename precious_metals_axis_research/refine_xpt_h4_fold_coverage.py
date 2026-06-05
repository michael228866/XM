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

from precious_metals_axis_research.optimize_training_profiles_silver_xaueur import (  # noqa: E402
    MODEL_PROFILES,
)
from precious_metals_axis_research.optimize_xpt_xpd_extended_timeframes import (  # noqa: E402
    COST_MULTIPLIERS,
    evaluate,
    prepare,
)


OUTPUT_CSV = RESEARCH_DIR / "xpt_h4_fold_coverage_results.csv"
OUTPUT_JSON = RESEARCH_DIR / "xpt_h4_fold_coverage_results.json"
OUTPUT_MD = RESEARCH_DIR / "xpt_h4_fold_coverage_report.md"
OUTPUT_BEST = RESEARCH_DIR / "xpt_h4_fold_coverage_best.json"

SYMBOL = "XPTUSD#"
TIMEFRAME = "H4"
PROFILE_NAMES = ["current_symbol", "smooth_more_trees"]

FILTERS = [
    {
        "filter_name": "none",
        "trend_mode": "any",
        "rsi_min": 0.0,
        "rsi_max": 100.0,
        "vola_max": 99.0,
        "spread_atr_max": 99.0,
        "macd_mode": "any",
    },
    {
        "filter_name": "quiet",
        "trend_mode": "any",
        "rsi_min": 0.0,
        "rsi_max": 100.0,
        "vola_max": 1.2,
        "spread_atr_max": 99.0,
        "macd_mode": "any",
    },
    {
        "filter_name": "aligned",
        "trend_mode": "aligned",
        "rsi_min": 20.0,
        "rsi_max": 90.0,
        "vola_max": 1.8,
        "spread_atr_max": 0.9,
        "macd_mode": "any",
    },
    {
        "filter_name": "aligned_momentum",
        "trend_mode": "aligned",
        "rsi_min": 35.0,
        "rsi_max": 85.0,
        "vola_max": 1.8,
        "spread_atr_max": 0.85,
        "macd_mode": "aligned",
    },
    {
        "filter_name": "counter_quiet",
        "trend_mode": "counter",
        "rsi_min": 15.0,
        "rsi_max": 85.0,
        "vola_max": 1.2,
        "spread_atr_max": 0.85,
        "macd_mode": "any",
    },
]


def param_grid():
    for threshold, edge, tp, sl, hold, direction in product(
        [0.50, 0.52, 0.54, 0.56, 0.58, 0.60],
        [0.0, 0.02],
        [2.4, 2.8, 3.2, 3.6],
        [4.4, 5.4, 6.4],
        [48, 72, 96, 120, 144],
        ["long", "both"],
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


def add_coverage_metrics(row: dict) -> dict:
    folds = row["folds"]
    min_fold_trades = min(item["trades"] for item in folds)
    all_folds_positive = all(item["pnl_r"] > 0 for item in folds)
    all_folds_pass = all(item["fold_pass"] for item in folds)
    strict_gate = (
        all_folds_positive
        and all_folds_pass
        and row["total_r"] >= 8.0
        and row["trades"] >= 32
        and row["max_drawdown_r"] >= -10.0
        and row["recent_paper_r"] > 0
    )
    coverage_score = (
        row["total_r"] * 150.0
        + row["positive_folds"] * 900.0
        + row["passed_folds"] * 700.0
        + min(min_fold_trades, 8) * 220.0
        + row["weighted_win_rate"] * 650.0
        + min(row["mean_profit_factor"], 4.0) * 160.0
        - abs(row["max_drawdown_r"]) * 70.0
        + min(row["worst_fold_r"], 0.0) * 350.0
    )
    return {
        **row,
        "min_fold_trades_observed": min_fold_trades,
        "all_folds_positive": all_folds_positive,
        "all_folds_pass": all_folds_pass,
        "strict_coverage_gate": strict_gate,
        "coverage_score": round(coverage_score, 4),
    }


def choose_best(rows: list[dict]) -> dict:
    return sorted(
        rows,
        key=lambda row: (
            row["strict_coverage_gate"],
            row["all_folds_positive"],
            row["passed_folds"],
            row["min_fold_trades_observed"],
            row["coverage_score"],
            row["total_r"],
        ),
        reverse=True,
    )[0]


def flatten(row: dict, profile: str, group: str) -> dict:
    return {
        key: value
        for key, value in {"profile": profile, "group": group, **row}.items()
        if key != "folds"
    }


def write_outputs(rows: list[dict], selected: dict) -> None:
    flat_rows = [flatten(item["result"], item["profile"], item["group"]) for item in rows]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    OUTPUT_JSON.write_text(
        json.dumps({"rows": rows, "selected": selected}, indent=2),
        encoding="utf-8",
    )
    OUTPUT_BEST.write_text(json.dumps(selected, indent=2), encoding="utf-8")

    best = selected["best_3x"]
    lines = [
        "# XPTUSD H4 Fold-Coverage Refinement",
        "",
        "Research-only refinement focused on increasing fold-level trade coverage.",
        "",
        "## Selected",
        "",
        "| Profile | Strict Gate | 3x R | Trades | Min Fold Trades | Positive | Passed | Win | PF | Worst R | DD | Params |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        (
            "| {profile} | {strict_coverage_gate} | {total_r:.2f} | {trades} | "
            "{min_fold_trades_observed} | {positive_folds}/4 | {passed_folds}/4 | "
            "{weighted_win_rate:.2%} | {mean_profit_factor:.2f} | "
            "{worst_fold_r:.2f} | {max_drawdown_r:.2f} | "
            "{filter_name}: conf={threshold}, edge={edge_threshold}, "
            "tp/sl={tp_atr}/{sl_atr}, hold={max_hold}, dir={direction_mode} |"
        ).format(profile=selected["profile"], **best),
        "",
        "## Cost Stress",
        "",
        "| Cost | Gate | Strict Gate | R | Trades | Min Fold Trades | Positive | Passed | Win | Worst R | DD |",
        "|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected["cost_stress"]:
        lines.append(
            "| {cost_multiplier:.1f}x | {gate} | {strict_coverage_gate} | "
            "{total_r:.2f} | {trades} | {min_fold_trades_observed} | "
            "{positive_folds}/4 | {passed_folds}/4 | {weighted_win_rate:.2%} | "
            "{worst_fold_r:.2f} | {max_drawdown_r:.2f} |".format(**row)
        )

    lines.extend(["", "## Fold Details", ""])
    for fold in best["folds"]:
        lines.append(
            "- {fold}: R={pnl_r:.2f}, trades={trades}, win={win_rate:.2%}, "
            "PF={profit_factor:.2f}, pass={fold_pass}".format(**fold)
        )

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = []
    candidates = []
    profiles = {name: MODEL_PROFILES[name] for name in PROFILE_NAMES}

    for profile_name, profile in profiles.items():
        print(f"=== {SYMBOL} {TIMEFRAME} profile={profile_name} ===", flush=True)
        prepared = prepare(SYMBOL, TIMEFRAME, profile_name, profile)
        grid_results = []
        for base_params in param_grid():
            for filter_params in FILTERS:
                params = {**base_params, **filter_params}
                grid_results.append(
                    add_coverage_metrics(
                        evaluate(SYMBOL, TIMEFRAME, params, prepared, 3.0)
                    )
                )

        best_3x = choose_best(grid_results)
        cost_stress = [
            add_coverage_metrics(evaluate(SYMBOL, TIMEFRAME, best_3x, prepared, cost))
            for cost in COST_MULTIPLIERS
        ]
        candidates.append(
            {
                "profile": profile_name,
                "best_3x": best_3x,
                "cost_stress": cost_stress,
                "strict_gate_count": sum(row["strict_coverage_gate"] for row in cost_stress),
                "cost_gate_count": sum(row["gate"] for row in cost_stress),
                "cost_total_r": round(sum(row["total_r"] for row in cost_stress), 4),
            }
        )
        rows.extend(
            {"profile": profile_name, "group": "grid_3x", "result": row}
            for row in grid_results
        )
        rows.extend(
            {"profile": profile_name, "group": "selected_cost", "result": row}
            for row in cost_stress
        )

    selected = sorted(
        candidates,
        key=lambda item: (
            item["strict_gate_count"],
            item["best_3x"]["strict_coverage_gate"],
            item["best_3x"]["min_fold_trades_observed"],
            item["cost_gate_count"],
            item["best_3x"]["total_r"],
            item["cost_total_r"],
        ),
        reverse=True,
    )[0]

    write_outputs(rows, selected)
    best = selected["best_3x"]
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_BEST}")
    print(
        f"Selected profile={selected['profile']} strict_gates="
        f"{selected['strict_gate_count']}/5 cost_gates={selected['cost_gate_count']}/5 "
        f"r3x={best['total_r']:.2f} min_fold_trades={best['min_fold_trades_observed']} "
        f"strict3x={best['strict_coverage_gate']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
