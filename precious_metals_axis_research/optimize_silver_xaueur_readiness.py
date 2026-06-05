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
from precious_metals_axis_research.readiness_silver_xaueur import (  # noqa: E402
    SILVER_FOLDS,
    XAUEUR_FOLDS,
    fold_pass,
    scale_spread,
)
from precious_metals_axis_research.walk_forward_all_metals_shared import (  # noqa: E402
    load_frames,
    slice_by_ratio,
    train_fold_model as train_shared_fold_model,
)
from precious_metals_axis_research.walk_forward_long_tf_cost import (  # noqa: E402
    compact_period,
    train_fold_model as train_symbol_fold_model,
)


OUTPUT_CSV = RESEARCH_DIR / "silver_xaueur_readiness_optimized_results.csv"
OUTPUT_JSON = RESEARCH_DIR / "silver_xaueur_readiness_optimized_results.json"
OUTPUT_MD = RESEARCH_DIR / "silver_xaueur_readiness_optimized_report.md"
OUTPUT_BEST = RESEARCH_DIR / "silver_xaueur_readiness_optimized_best.json"


def silver_grid():
    for threshold, edge, tp, sl, hold in product(
        [0.56, 0.58, 0.60, 0.62, 0.64, 0.66],
        [0.00, 0.03, 0.06, 0.10],
        [2.0, 2.4, 2.8, 3.2, 3.6],
        [3.6, 4.4, 5.2, 6.0],
        [72, 120, 168, 216],
    ):
        if tp > sl:
            continue
        yield {
            "symbol": "SILVER#",
            "threshold": threshold,
            "edge_threshold": edge,
            "tp_atr": tp,
            "sl_atr": sl,
            "max_hold": hold,
            "direction_mode": "long",
        }


def xaueur_grid():
    for threshold, edge, tp, sl, hold, direction in product(
        [0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64],
        [0.00, 0.02, 0.04, 0.08],
        [1.6, 1.8, 2.0, 2.2, 2.4, 2.6],
        [3.0, 3.4, 3.8, 4.2, 4.8],
        [72, 96, 120, 168, 216],
        ["both", "short"],
    ):
        if tp > sl:
            continue
        yield {
            "symbol": "XAUEUR#",
            "threshold": threshold,
            "edge_threshold": edge,
            "tp_atr": tp,
            "sl_atr": sl,
            "max_hold": hold,
            "direction_mode": direction,
        }


def prepare_silver():
    symbol = "SILVER#"
    frame, features = load_case(symbol, "H1")
    point = get_symbol_point(symbol)
    prepared = []
    for fold in SILVER_FOLDS:
        train_df = slice_by_ratio(frame, *fold["train"])
        test_df = slice_by_ratio(frame, *fold["test"])
        print(f"SILVER {fold['name']}: train={len(train_df):,} test={len(test_df):,}")
        model = train_symbol_fold_model(train_df, features)
        probs = model.predict_proba(test_df[features]).astype(np.float32)
        prepared.append(
            {
                "fold": fold["name"],
                "test_period": compact_period(test_df),
                "stress_df": scale_spread(test_df, 3.0),
                "probs": probs,
                "point": point,
            }
        )
    return prepared


def prepare_xaueur():
    symbol = "XAUEUR#"
    frames, features = load_frames()
    point = get_symbol_point(symbol)
    prepared = []
    for fold in XAUEUR_FOLDS:
        train_frames = [slice_by_ratio(frame, *fold["train"]) for frame in frames.values()]
        test_df = slice_by_ratio(frames[symbol], *fold["test"])
        print(f"XAUEUR {fold['name']}: train_shared={sum(len(df) for df in train_frames):,} test={len(test_df):,}")
        model = train_shared_fold_model(train_frames, features)
        probs = model.predict_proba(test_df[features]).astype(np.float32)
        prepared.append(
            {
                "fold": fold["name"],
                "test_period": compact_period(test_df),
                "stress_df": scale_spread(test_df, 3.0),
                "probs": probs,
                "point": point,
            }
        )
    return prepared


def evaluate(params: dict, prepared: list[dict]) -> dict:
    fold_rows = []
    for fold in prepared:
        stats = simulate_cost_aware(
            fold["stress_df"], fold["probs"], params, fold["point"]
        )
        fold_rows.append(
            {
                "fold": fold["fold"],
                "test_start": fold["test_period"]["start"],
                "test_end": fold["test_period"]["end"],
                "pnl_r": stats["pnl_r"],
                "trades": stats["trades"],
                "win_rate": stats["win_rate"],
                "profit_factor": stats["profit_factor"],
                "max_drawdown_r": stats["max_drawdown_r"],
                "avg_r": stats["avg_r"],
                "max_loss_streak": stats["max_loss_streak"],
                "fold_pass": fold_pass(stats),
            }
        )
    total_trades = sum(row["trades"] for row in fold_rows)
    weighted_win = (
        sum(row["win_rate"] * row["trades"] for row in fold_rows) / total_trades
        if total_trades
        else 0.0
    )
    total_r = sum(row["pnl_r"] for row in fold_rows)
    passed = sum(row["fold_pass"] for row in fold_rows)
    positive = sum(row["pnl_r"] > 0 for row in fold_rows)
    mean_pf = sum(row["profit_factor"] for row in fold_rows) / len(fold_rows)
    worst = min(row["pnl_r"] for row in fold_rows)
    max_dd = min(row["max_drawdown_r"] for row in fold_rows)
    symbol = params["symbol"]
    if symbol == "SILVER#":
        battle_gate = (
            positive >= 4
            and passed >= 4
            and total_r >= 12
            and total_trades >= 80
            and fold_rows[-1]["pnl_r"] > 0
            and fold_rows[-1]["profit_factor"] >= 1.15
        )
    else:
        battle_gate = (
            positive == 3
            and passed >= 2
            and total_r >= 6
            and total_trades >= 24
            and fold_rows[-1]["pnl_r"] > 0
            and fold_rows[-1]["profit_factor"] >= 1.15
        )
    score = (
        total_r * 160.0
        + positive * 700.0
        + passed * 500.0
        + weighted_win * 650.0
        + min(mean_pf, 4.0) * 220.0
        - abs(max_dd) * 50.0
        + min(worst, 0.0) * 150.0
        + min(fold_rows[-1]["pnl_r"], 0.0) * 250.0
    )
    return {
        **params,
        "stress_total_r": round(total_r, 4),
        "stress_total_trades": total_trades,
        "stress_positive_folds": positive,
        "stress_passed_folds": passed,
        "stress_weighted_win_rate": round(weighted_win, 4),
        "stress_mean_profit_factor": round(mean_pf, 4),
        "stress_worst_fold_r": round(worst, 4),
        "stress_max_drawdown_r": round(max_dd, 4),
        "recent_paper_r": fold_rows[-1]["pnl_r"],
        "battle_gate": battle_gate,
        "score": round(score, 4),
        "folds": fold_rows,
    }


def write_outputs(results: list[dict]) -> None:
    results = sorted(
        results,
        key=lambda row: (row["battle_gate"], row["score"], row["stress_total_r"]),
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
    best = {}
    for symbol in ["SILVER#", "XAUEUR#"]:
        best[symbol] = next(row for row in results if row["symbol"] == symbol)
    OUTPUT_BEST.write_text(json.dumps(best, indent=2), encoding="utf-8")

    lines = [
        "# SILVER / XAUEUR Readiness Optimization",
        "",
        "Optimizes directly on the extended readiness folds under 3x spread.",
        "",
        "| Symbol | Gate | R | Positive | Passed | Trades | Win | PF | Worst R | Recent R | Params |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for symbol in ["SILVER#", "XAUEUR#"]:
        row = best[symbol]
        lines.append(
            "| {symbol} | {battle_gate} | {stress_total_r:.2f} | "
            "{stress_positive_folds}/{fold_count} | {stress_passed_folds}/{fold_count} | "
            "{stress_total_trades} | {stress_weighted_win_rate:.2%} | "
            "{stress_mean_profit_factor:.2f} | {stress_worst_fold_r:.2f} | "
            "{recent_paper_r:.2f} | conf={threshold}, edge={edge_threshold}, "
            "tp/sl={tp_atr}/{sl_atr}, hold={max_hold}, dir={direction_mode} |".format(
                fold_count=len(row["folds"]), **row
            )
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    results = []
    print("Preparing SILVER readiness folds...")
    silver_prepared = prepare_silver()
    print("Sweeping SILVER...")
    for params in silver_grid():
        results.append(evaluate(params, silver_prepared))
    print("Preparing XAUEUR readiness folds...")
    xaueur_prepared = prepare_xaueur()
    print("Sweeping XAUEUR...")
    for params in xaueur_grid():
        results.append(evaluate(params, xaueur_prepared))
    write_outputs(results)
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_BEST}")
    print(f"Battle-gate candidates: {sum(row['battle_gate'] for row in results)}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
