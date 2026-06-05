from __future__ import annotations

import csv
import json
import os
import sys
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
from precious_metals_axis_research.walk_forward_all_metals_shared import (  # noqa: E402
    load_frames,
    slice_by_ratio,
    stress_frame,
    train_fold_model as train_shared_fold_model,
)
from precious_metals_axis_research.walk_forward_long_tf_cost import (  # noqa: E402
    compact_period,
    train_fold_model as train_symbol_fold_model,
)


OUTPUT_CSV = RESEARCH_DIR / "silver_xaueur_readiness_folds.csv"
OUTPUT_JSON = RESEARCH_DIR / "silver_xaueur_readiness.json"
OUTPUT_MD = RESEARCH_DIR / "silver_xaueur_readiness.md"

SILVER_PARAMS = {
    "threshold": 0.56,
    "edge_threshold": 0.0,
    "tp_atr": 3.2,
    "sl_atr": 4.4,
    "max_hold": 120,
    "direction_mode": "long",
}

XAUEUR_PARAMS = {
    "threshold": 0.56,
    "edge_threshold": 0.0,
    "tp_atr": 2.4,
    "sl_atr": 3.4,
    "max_hold": 168,
    "direction_mode": "both",
}

SILVER_FOLDS = [
    {"name": "fold_1", "train": (0.00, 0.50), "test": (0.50, 0.60)},
    {"name": "fold_2", "train": (0.10, 0.60), "test": (0.60, 0.70)},
    {"name": "fold_3", "train": (0.20, 0.70), "test": (0.70, 0.80)},
    {"name": "fold_4", "train": (0.30, 0.80), "test": (0.80, 0.90)},
    {"name": "fold_5", "train": (0.40, 0.90), "test": (0.90, 1.00)},
]

XAUEUR_FOLDS = [
    {"name": "fold_1", "train": (0.00, 0.55), "test": (0.55, 0.70)},
    {"name": "fold_2", "train": (0.15, 0.70), "test": (0.70, 0.85)},
    {"name": "fold_3", "train": (0.30, 0.85), "test": (0.85, 1.00)},
]


def scale_spread(frame, multiplier: float):
    adjusted = frame.copy()
    if "SPREAD" in adjusted.columns:
        adjusted["SPREAD"] = adjusted["SPREAD"].fillna(0) * multiplier
    return adjusted


def fold_pass(stats: dict, min_trades: int = 6) -> bool:
    return (
        stats["pnl_r"] > 0
        and stats["profit_factor"] >= 1.15
        and stats["win_rate"] >= 0.55
        and stats["trades"] >= min_trades
    )


def evaluate_stats(symbol: str, fold_name: str, test_period: dict, stats: dict, cost_label: str) -> dict:
    return {
        "symbol": symbol,
        "fold": fold_name,
        "cost": cost_label,
        "test_start": test_period["start"],
        "test_end": test_period["end"],
        "pnl_r": stats["pnl_r"],
        "trades": stats["trades"],
        "win_rate": stats["win_rate"],
        "profit_factor": stats["profit_factor"],
        "max_drawdown_r": stats["max_drawdown_r"],
        "avg_r": stats["avg_r"],
        "max_loss_streak": stats["max_loss_streak"],
        "fold_pass": fold_pass(stats),
    }


def summarize_cost(rows: list[dict], symbol: str, cost: str) -> dict:
    subset = [row for row in rows if row["symbol"] == symbol and row["cost"] == cost]
    total_trades = sum(row["trades"] for row in subset)
    weighted_win = (
        sum(row["win_rate"] * row["trades"] for row in subset) / total_trades
        if total_trades
        else 0.0
    )
    return {
        "symbol": symbol,
        "cost": cost,
        "folds": len(subset),
        "total_pnl_r": round(sum(row["pnl_r"] for row in subset), 4),
        "total_trades": total_trades,
        "positive_folds": sum(row["pnl_r"] > 0 for row in subset),
        "passed_folds": sum(row["fold_pass"] for row in subset),
        "weighted_win_rate": round(weighted_win, 4),
        "mean_profit_factor": round(
            sum(row["profit_factor"] for row in subset) / len(subset), 4
        ),
        "worst_fold_pnl_r": round(min(row["pnl_r"] for row in subset), 4),
        "max_drawdown_r": round(min(row["max_drawdown_r"] for row in subset), 4),
    }


def run_silver() -> list[dict]:
    symbol = "SILVER#"
    print("Loading SILVER# H1...")
    frame, features = load_case(symbol, "H1")
    point = get_symbol_point(symbol)
    rows = []
    for fold in SILVER_FOLDS:
        train_df = slice_by_ratio(frame, *fold["train"])
        test_df = slice_by_ratio(frame, *fold["test"])
        test_period = compact_period(test_df)
        print(f"SILVER {fold['name']}: train={len(train_df):,} test={len(test_df):,}")
        model = train_symbol_fold_model(train_df, features)
        probs = model.predict_proba(test_df[features]).astype(np.float32)
        for multiplier in [1.0, 2.0, 3.0]:
            stats = simulate_cost_aware(
                scale_spread(test_df, multiplier), probs, SILVER_PARAMS, point
            )
            rows.append(
                evaluate_stats(symbol, fold["name"], test_period, stats, f"{multiplier:.1f}x")
            )
            print(
                f"  {multiplier:.1f}x R={stats['pnl_r']:.2f} "
                f"trades={stats['trades']} win={stats['win_rate']:.2%}"
            )
    return rows


def run_xaueur() -> list[dict]:
    symbol = "XAUEUR#"
    print("Loading all-metals shared frames for XAUEUR#...")
    frames, features = load_frames()
    point = get_symbol_point(symbol)
    rows = []
    for fold in XAUEUR_FOLDS:
        train_frames = [slice_by_ratio(frame, *fold["train"]) for frame in frames.values()]
        test_df = slice_by_ratio(frames[symbol], *fold["test"])
        test_period = compact_period(test_df)
        print(f"XAUEUR {fold['name']}: train_shared={sum(len(df) for df in train_frames):,} test={len(test_df):,}")
        model = train_shared_fold_model(train_frames, features)
        probs = model.predict_proba(test_df[features]).astype(np.float32)
        for multiplier in [1.0, 2.0, 3.0]:
            df = test_df if multiplier == 1.0 else scale_spread(test_df, multiplier)
            stats = simulate_cost_aware(df, probs, XAUEUR_PARAMS, point)
            rows.append(
                evaluate_stats(symbol, fold["name"], test_period, stats, f"{multiplier:.1f}x")
            )
            print(
                f"  {multiplier:.1f}x R={stats['pnl_r']:.2f} "
                f"trades={stats['trades']} win={stats['win_rate']:.2%}"
            )
    return rows


def paper_holdout(rows: list[dict]) -> dict:
    silver = summarize_cost(rows, "SILVER#", "3.0x")
    xaueur = summarize_cost(rows, "XAUEUR#", "3.0x")
    return {
        "silver_recent_fold": [
            row for row in rows if row["symbol"] == "SILVER#" and row["fold"] == "fold_5" and row["cost"] == "3.0x"
        ][0],
        "xaueur_recent_fold": [
            row for row in rows if row["symbol"] == "XAUEUR#" and row["fold"] == "fold_3" and row["cost"] == "3.0x"
        ][0],
        "silver_summary": silver,
        "xaueur_summary": xaueur,
    }


def verdicts(summary: dict, holdout: dict) -> dict:
    silver = summary["SILVER#"]["3.0x"]
    xaueur = summary["XAUEUR#"]["3.0x"]
    silver_recent = holdout["silver_recent_fold"]
    xaueur_recent = holdout["xaueur_recent_fold"]
    return {
        "SILVER#": {
            "verdict": (
                "battle_ready_paper"
                if silver["positive_folds"] >= 4
                and silver["passed_folds"] >= 4
                and silver["total_pnl_r"] >= 12
                and silver["total_trades"] >= 80
                and silver_recent["pnl_r"] > 0
                and silver_recent["profit_factor"] >= 1.15
                else "not_battle_ready"
            ),
            "reason": "Requires 4/5 stressed folds passing, positive recent paper holdout, and enough trades.",
        },
        "XAUEUR#": {
            "verdict": (
                "battle_ready_paper"
                if xaueur["positive_folds"] == 3
                and xaueur["passed_folds"] >= 2
                and xaueur["total_pnl_r"] >= 6
                and xaueur["total_trades"] >= 24
                and xaueur_recent["pnl_r"] > 0
                and xaueur_recent["profit_factor"] >= 1.15
                else "not_battle_ready"
            ),
            "reason": "Requires 3/3 stressed folds positive, at least 2/3 passing, positive recent paper holdout, and enough trades.",
        },
    }


def write_outputs(rows: list[dict], summary: dict, holdout: dict, verdict: dict) -> None:
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "summary": summary,
                "paper_holdout": holdout,
                "verdicts": verdict,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    lines = [
        "# SILVER / XAUEUR Readiness",
        "",
        "Research-only readiness gate. No live files are modified.",
        "",
        "## 3x Spread Summary",
        "",
        "| Symbol | Verdict | R | Positive | Passed | Trades | Win | PF | Worst R | Max DD | Recent Paper R |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for symbol in ["SILVER#", "XAUEUR#"]:
        item = summary[symbol]["3.0x"]
        recent = holdout["silver_recent_fold"] if symbol == "SILVER#" else holdout["xaueur_recent_fold"]
        lines.append(
            "| {row_symbol} | {verdict} | {total_pnl_r:.2f} | {positive_folds}/{folds} | "
            "{passed_folds}/{folds} | {total_trades} | {weighted_win_rate:.2%} | "
            "{mean_profit_factor:.2f} | {worst_fold_pnl_r:.2f} | {max_drawdown_r:.2f} | "
            "{recent_r:.2f} |".format(
                row_symbol=symbol,
                verdict=verdict[symbol]["verdict"],
                recent_r=recent["pnl_r"],
                **item,
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `Recent Paper R` is the latest rolling test fold under 3x spread.",
            "- `battle_ready_paper` means ready for dry-run/paper deployment, not a guarantee of live profit.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = []
    rows.extend(run_silver())
    rows.extend(run_xaueur())
    summary = {
        symbol: {cost: summarize_cost(rows, symbol, cost) for cost in ["1.0x", "2.0x", "3.0x"]}
        for symbol in ["SILVER#", "XAUEUR#"]
    }
    holdout = paper_holdout(rows)
    verdict = verdicts(summary, holdout)
    write_outputs(rows, summary, holdout, verdict)
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    for symbol, item in verdict.items():
        print(f"{symbol}: {item['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
