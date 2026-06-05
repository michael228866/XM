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
from precious_metals_axis_research.optimize_silver_regime_readiness import (  # noqa: E402
    simulate_filtered as simulate_silver_filtered,
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


OUTPUT_CSV = RESEARCH_DIR / "final_silver_xaueur_robustness.csv"
OUTPUT_JSON = RESEARCH_DIR / "final_silver_xaueur_robustness.json"
OUTPUT_MD = RESEARCH_DIR / "final_silver_xaueur_robustness.md"

COST_MULTIPLIERS = [1.0, 2.0, 3.0, 4.0, 5.0]

SILVER_BASE = {
    "symbol": "SILVER#",
    "threshold": 0.56,
    "edge_threshold": 0.0,
    "tp_atr": 5.2,
    "sl_atr": 5.2,
    "max_hold": 216,
    "direction_mode": "long",
    "trend_min": 0.0,
    "rsi_min": 0.0,
    "rsi_max": 100.0,
    "vola_max": 1.2,
    "spread_atr_max": 0.75,
    "macd_min": -999.0,
}

XAUEUR_BASE = {
    "symbol": "XAUEUR#",
    "threshold": 0.56,
    "edge_threshold": 0.0,
    "tp_atr": 2.6,
    "sl_atr": 4.8,
    "max_hold": 216,
    "direction_mode": "both",
}


def prepare_silver():
    frame, features = load_case("SILVER#", "H1")
    point = get_symbol_point("SILVER#")
    folds = []
    for fold in SILVER_FOLDS:
        train_df = slice_by_ratio(frame, *fold["train"])
        test_df = slice_by_ratio(frame, *fold["test"])
        print(f"SILVER {fold['name']}: train={len(train_df):,} test={len(test_df):,}")
        model = train_symbol_fold_model(train_df, features)
        probs = model.predict_proba(test_df[features]).astype(np.float32)
        folds.append(
            {
                "fold": fold["name"],
                "test_period": compact_period(test_df),
                "test_df": test_df,
                "probs": probs,
                "point": point,
            }
        )
    return folds


def prepare_xaueur():
    frames, features = load_frames()
    point = get_symbol_point("XAUEUR#")
    folds = []
    for fold in XAUEUR_FOLDS:
        train_frames = [slice_by_ratio(frame, *fold["train"]) for frame in frames.values()]
        test_df = slice_by_ratio(frames["XAUEUR#"], *fold["test"])
        print(f"XAUEUR {fold['name']}: train_shared={sum(len(df) for df in train_frames):,} test={len(test_df):,}")
        model = train_shared_fold_model(train_frames, features)
        probs = model.predict_proba(test_df[features]).astype(np.float32)
        folds.append(
            {
                "fold": fold["name"],
                "test_period": compact_period(test_df),
                "test_df": test_df,
                "probs": probs,
                "point": point,
            }
        )
    return folds


def aggregate(symbol: str, fold_rows: list[dict]) -> dict:
    total_trades = sum(row["trades"] for row in fold_rows)
    total_r = sum(row["pnl_r"] for row in fold_rows)
    weighted_win = (
        sum(row["win_rate"] * row["trades"] for row in fold_rows) / total_trades
        if total_trades
        else 0.0
    )
    positive = sum(row["pnl_r"] > 0 for row in fold_rows)
    passed = sum(row["fold_pass"] for row in fold_rows)
    mean_pf = sum(row["profit_factor"] for row in fold_rows) / len(fold_rows)
    worst = min(row["pnl_r"] for row in fold_rows)
    max_dd = min(row["max_drawdown_r"] for row in fold_rows)
    recent = fold_rows[-1]
    if symbol == "SILVER#":
        gate = (
            positive == 5
            and passed >= 5
            and total_r >= 12.0
            and total_trades >= 60
            and recent["pnl_r"] > 0
            and recent["profit_factor"] >= 1.15
            and max_dd >= -10.0
        )
    else:
        gate = (
            positive == 3
            and passed >= 3
            and total_r >= 6.0
            and total_trades >= 24
            and recent["pnl_r"] > 0
            and recent["profit_factor"] >= 1.15
            and max_dd >= -6.0
        )
    return {
        "symbol": symbol,
        "total_r": round(total_r, 4),
        "trades": total_trades,
        "positive_folds": positive,
        "passed_folds": passed,
        "weighted_win_rate": round(weighted_win, 4),
        "mean_profit_factor": round(mean_pf, 4),
        "worst_fold_r": round(worst, 4),
        "max_drawdown_r": round(max_dd, 4),
        "recent_paper_r": recent["pnl_r"],
        "gate": gate,
        "folds": fold_rows,
    }


def evaluate_silver(params: dict, folds: list[dict], cost_multiplier: float) -> dict:
    fold_rows = []
    for fold in folds:
        stats = simulate_silver_filtered(
            scale_spread(fold["test_df"], cost_multiplier),
            fold["probs"],
            params,
            fold["point"],
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
    summary = aggregate("SILVER#", fold_rows)
    return {**params, **summary, "cost_multiplier": cost_multiplier}


def evaluate_xaueur(params: dict, folds: list[dict], cost_multiplier: float) -> dict:
    fold_rows = []
    for fold in folds:
        stats = simulate_cost_aware(
            scale_spread(fold["test_df"], cost_multiplier),
            fold["probs"],
            params,
            fold["point"],
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
    summary = aggregate("XAUEUR#", fold_rows)
    return {**params, **summary, "cost_multiplier": cost_multiplier}


def silver_neighborhood():
    for threshold, edge, tp, sl, hold, vola_max in product(
        [0.54, 0.56, 0.58],
        [0.0, 0.05],
        [4.4, 5.2, 6.0],
        [5.2, 6.0, 7.0],
        [216, 336],
        [1.0, 1.2, 1.4],
    ):
        if tp > sl:
            continue
        yield {
            **SILVER_BASE,
            "threshold": threshold,
            "edge_threshold": edge,
            "tp_atr": tp,
            "sl_atr": sl,
            "max_hold": hold,
            "vola_max": vola_max,
        }


def xaueur_neighborhood():
    for threshold, edge, tp, sl, hold in product(
        [0.54, 0.56, 0.58, 0.60],
        [0.0, 0.02],
        [2.2, 2.6, 3.0],
        [4.2, 4.8, 5.4],
        [168, 216, 288],
    ):
        if tp > sl:
            continue
        yield {
            **XAUEUR_BASE,
            "threshold": threshold,
            "edge_threshold": edge,
            "tp_atr": tp,
            "sl_atr": sl,
            "max_hold": hold,
        }


def flatten(row: dict, group: str, variant: str) -> dict:
    flat = {key: value for key, value in row.items() if key != "folds"}
    flat["group"] = group
    flat["variant"] = variant
    return flat


def summarize_neighborhood(rows: list[dict]) -> dict:
    if not rows:
        return {}
    passed = [row for row in rows if row["gate"]]
    return {
        "variants": len(rows),
        "gate_passed": len(passed),
        "pass_rate": round(len(passed) / len(rows), 4),
        "median_total_r": round(float(np.median([row["total_r"] for row in rows])), 4),
        "median_trades": round(float(np.median([row["trades"] for row in rows])), 2),
        "best_total_r": round(max(row["total_r"] for row in rows), 4),
        "worst_total_r": round(min(row["total_r"] for row in rows), 4),
    }


def write_outputs(cost_rows: list[dict], neighborhood_rows: list[dict], report: dict) -> None:
    flat_rows = []
    for row in cost_rows:
        flat_rows.append(flatten(row, "cost_stress", "base"))
    for index, row in enumerate(neighborhood_rows, start=1):
        flat_rows.append(flatten(row, "neighborhood_3x", f"variant_{index}"))
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "cost_stress": cost_rows,
                "neighborhood_3x": neighborhood_rows,
                "summary": report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Final SILVER / XAUEUR Robustness",
        "",
        "Fixed candidates tested across 1x-5x spread cost plus 3x-cost parameter neighborhoods.",
        "",
        "## Cost Stress",
        "",
        "| Symbol | Cost | Gate | R | Positive | Passed | Trades | Win | PF | Worst R | DD | Recent R |",
        "|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cost_rows:
        fold_count = 5 if row["symbol"] == "SILVER#" else 3
        lines.append(
            "| {symbol} | {cost_multiplier:.1f}x | {gate} | {total_r:.2f} | "
            "{positive_folds}/{fold_count} | {passed_folds}/{fold_count} | {trades} | "
            "{weighted_win_rate:.2%} | {mean_profit_factor:.2f} | {worst_fold_r:.2f} | "
            "{max_drawdown_r:.2f} | {recent_paper_r:.2f} |".format(
                fold_count=fold_count, **row
            )
        )
    lines.extend(
        [
            "",
            "## Neighborhood",
            "",
            "| Symbol | Variants | Gate Passed | Pass Rate | Median R | Best R | Worst R |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for symbol in ["SILVER#", "XAUEUR#"]:
        item = report["neighborhood"][symbol]
        lines.append(
            "| {symbol} | {variants} | {gate_passed} | {pass_rate:.2%} | "
            "{median_total_r:.2f} | {best_total_r:.2f} | {worst_total_r:.2f} |".format(
                symbol=symbol, **item
            )
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            report["verdict"],
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    print("Preparing SILVER folds...")
    silver_folds = prepare_silver()
    print("Preparing XAUEUR folds...")
    xaueur_folds = prepare_xaueur()

    cost_rows = []
    for cost in COST_MULTIPLIERS:
        cost_rows.append(evaluate_silver(SILVER_BASE, silver_folds, cost))
        cost_rows.append(evaluate_xaueur(XAUEUR_BASE, xaueur_folds, cost))

    neighborhood_rows = []
    print("Evaluating SILVER parameter neighborhood at 3x cost...")
    silver_neighbors = [evaluate_silver(params, silver_folds, 3.0) for params in silver_neighborhood()]
    print("Evaluating XAUEUR parameter neighborhood at 3x cost...")
    xaueur_neighbors = [evaluate_xaueur(params, xaueur_folds, 3.0) for params in xaueur_neighborhood()]
    neighborhood_rows.extend(silver_neighbors)
    neighborhood_rows.extend(xaueur_neighbors)

    report = {
        "cost_stress_passed": {
            symbol: sum(
                row["gate"] for row in cost_rows if row["symbol"] == symbol
            )
            for symbol in ["SILVER#", "XAUEUR#"]
        },
        "neighborhood": {
            "SILVER#": summarize_neighborhood(silver_neighbors),
            "XAUEUR#": summarize_neighborhood(xaueur_neighbors),
        },
    }
    silver_3x = next(row for row in cost_rows if row["symbol"] == "SILVER#" and row["cost_multiplier"] == 3.0)
    xaueur_3x = next(row for row in cost_rows if row["symbol"] == "XAUEUR#" and row["cost_multiplier"] == 3.0)
    if silver_3x["gate"] and xaueur_3x["gate"]:
        report["verdict"] = (
            "Both candidates pass the 3x-cost final gate. Keep 4x/5x results as risk limits; "
            "promote only after live-paper logging from MT5 is clean."
        )
    else:
        report["verdict"] = (
            "At least one candidate fails the 3x-cost final gate. Do not promote failed symbols."
        )

    write_outputs(cost_rows, neighborhood_rows, report)
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(report["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
