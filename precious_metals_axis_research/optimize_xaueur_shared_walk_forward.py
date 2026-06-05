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

from precious_metals_axis_research.cost_aware_xaueur_m5 import (  # noqa: E402
    get_symbol_point,
    simulate_cost_aware,
)
from precious_metals_axis_research.walk_forward_all_metals_shared import (  # noqa: E402
    FOLDS,
    load_frames,
    slice_by_ratio,
    stress_frame,
    train_fold_model,
)


SYMBOL = "XAUEUR#"
OUTPUT_CSV = RESEARCH_DIR / "xaueur_shared_walk_forward_optimized_results.csv"
OUTPUT_JSON = RESEARCH_DIR / "xaueur_shared_walk_forward_optimized_results.json"
OUTPUT_MD = RESEARCH_DIR / "xaueur_shared_walk_forward_optimized_report.md"
OUTPUT_BEST = RESEARCH_DIR / "xaueur_shared_walk_forward_best.json"


def make_grid():
    for threshold, edge, tp, sl, hold, direction in product(
        [0.52, 0.54, 0.56, 0.58, 0.60, 0.62],
        [0.00, 0.02, 0.04, 0.06],
        [1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
        [3.4, 3.6, 4.0, 4.4, 4.8],
        [72, 96, 120, 168],
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


def prepare_folds():
    frames, features = load_frames()
    point = get_symbol_point(SYMBOL)
    prepared = []
    for fold in FOLDS:
        print(f"Training {fold['name']} shared model...")
        train_frames = [slice_by_ratio(frame, *fold["train"]) for frame in frames.values()]
        model = train_fold_model(train_frames, features)
        test_df = slice_by_ratio(frames[SYMBOL], *fold["test"])
        probs = model.predict_proba(test_df[features]).astype(np.float32)
        prepared.append(
            {
                "fold": fold["name"],
                "test_df": test_df,
                "stress_df": stress_frame(test_df),
                "probs": probs,
                "point": point,
            }
        )
    return prepared


def evaluate_candidate(params: dict, prepared_folds: list[dict]) -> dict:
    fold_rows = []
    for fold in prepared_folds:
        normal = simulate_cost_aware(fold["test_df"], fold["probs"], params, fold["point"])
        stress = simulate_cost_aware(fold["stress_df"], fold["probs"], params, fold["point"])
        fold_pass = (
            stress["pnl_r"] > 0
            and stress["profit_factor"] >= 1.15
            and stress["win_rate"] >= 0.58
            and stress["trades"] >= 6
        )
        fold_rows.append(
            {
                "fold": fold["fold"],
                "test_start": fold["test_df"]["TIME_DT"].iloc[0].isoformat(),
                "test_end": fold["test_df"]["TIME_DT"].iloc[-1].isoformat(),
                "normal_pnl_r": normal["pnl_r"],
                "normal_trades": normal["trades"],
                "normal_win_rate": normal["win_rate"],
                "normal_profit_factor": normal["profit_factor"],
                "normal_max_drawdown_r": normal["max_drawdown_r"],
                "stress_pnl_r": stress["pnl_r"],
                "stress_trades": stress["trades"],
                "stress_win_rate": stress["win_rate"],
                "stress_profit_factor": stress["profit_factor"],
                "stress_max_drawdown_r": stress["max_drawdown_r"],
                "fold_pass": fold_pass,
            }
        )

    stress_trades = sum(row["stress_trades"] for row in fold_rows)
    weighted_win = (
        sum(row["stress_win_rate"] * row["stress_trades"] for row in fold_rows)
        / stress_trades
        if stress_trades
        else 0.0
    )
    stress_total = sum(row["stress_pnl_r"] for row in fold_rows)
    stress_mean_pf = sum(row["stress_profit_factor"] for row in fold_rows) / len(fold_rows)
    stress_worst = min(row["stress_pnl_r"] for row in fold_rows)
    stress_dd = min(row["stress_max_drawdown_r"] for row in fold_rows)
    passed_folds = sum(row["fold_pass"] for row in fold_rows)
    positive_folds = sum(row["stress_pnl_r"] > 0 for row in fold_rows)
    passes_gate = (
        positive_folds == len(fold_rows)
        and passed_folds == len(fold_rows)
        and stress_total >= 6.0
        and stress_trades >= 18
        and weighted_win >= 0.65
        and stress_dd >= -4.0
    )
    score = (
        stress_total * 180.0
        + positive_folds * 450.0
        + passed_folds * 300.0
        + weighted_win * 700.0
        + min(stress_mean_pf, 4.0) * 220.0
        - abs(stress_dd) * 55.0
        + min(stress_worst, 0.0) * 150.0
    )
    return {
        **params,
        "stress_total_r": round(stress_total, 4),
        "stress_total_trades": stress_trades,
        "stress_positive_folds": positive_folds,
        "stress_passed_folds": passed_folds,
        "stress_weighted_win_rate": round(weighted_win, 4),
        "stress_mean_profit_factor": round(stress_mean_pf, 4),
        "stress_worst_fold_r": round(stress_worst, 4),
        "stress_max_drawdown_r": round(stress_dd, 4),
        "passes_gate": passes_gate,
        "score": round(score, 4),
        "folds": fold_rows,
    }


def write_outputs(results: list[dict]) -> None:
    results = sorted(
        results,
        key=lambda row: (row["passes_gate"], row["score"], row["stress_total_r"]),
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
        "# XAUEUR Shared Walk-Forward Optimization",
        "",
        "Optimizes XAUEUR# parameters on the all-metals shared H1 model under 3x spread stress.",
        "",
        "| Rank | Pass | Score | Stress R | Positive | Passed | Trades | Win | PF | Worst R | DD | Params |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(results[:25], start=1):
        lines.append(
            "| {rank} | {passes_gate} | {score:.1f} | {stress_total_r:.2f} | "
            "{stress_positive_folds}/2 | {stress_passed_folds}/2 | "
            "{stress_total_trades} | {stress_weighted_win_rate:.2%} | "
            "{stress_mean_profit_factor:.2f} | {stress_worst_fold_r:.2f} | "
            "{stress_max_drawdown_r:.2f} | conf={threshold}, edge={edge_threshold}, "
            "tp/sl={tp_atr}/{sl_atr}, hold={max_hold}, dir={direction_mode} |".format(
                rank=rank, **row
            )
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    prepared = prepare_folds()
    results = []
    total = 0
    for params in make_grid():
        total += 1
        results.append(evaluate_candidate(params, prepared))
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
