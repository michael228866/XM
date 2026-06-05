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
from precious_metals_axis_research.walk_forward_long_tf_cost import (  # noqa: E402
    FOLDS,
    compact_period,
    slice_by_ratio,
    train_fold_model,
)


SYMBOL = "SILVER#"
BASE_TIMEFRAME = "H1"
BEST_CANDIDATE = RESEARCH_DIR / "silver_h1_walk_forward_best_candidate.json"
OUTPUT_CSV = RESEARCH_DIR / "silver_h1_cost_stress.csv"
OUTPUT_JSON = RESEARCH_DIR / "silver_h1_cost_stress.json"
OUTPUT_MD = RESEARCH_DIR / "silver_h1_cost_stress.md"

SPREAD_MULTIPLIERS = [1.0, 1.5, 2.0, 3.0]
GATE = {
    "min_total_r": 18.0,
    "min_positive_folds": 4,
    "min_passed_folds": 3,
    "min_weighted_win": 0.56,
    "max_drawdown_r": -14.0,
    "min_trades": 120,
}


def load_candidate() -> dict:
    candidate = json.loads(BEST_CANDIDATE.read_text(encoding="utf-8"))
    return {
        "threshold": candidate["threshold"],
        "edge_threshold": candidate["edge_threshold"],
        "tp_atr": candidate["tp_atr"],
        "sl_atr": candidate["sl_atr"],
        "max_hold": candidate["max_hold"],
        "direction_mode": candidate["direction_mode"],
    }


def scale_spread(frame, multiplier: float):
    adjusted = frame.copy()
    if "SPREAD" in adjusted.columns:
        adjusted["SPREAD"] = adjusted["SPREAD"].fillna(0) * multiplier
    return adjusted


def fold_pass(stats: dict) -> bool:
    return (
        stats["pnl_r"] > 0
        and stats["profit_factor"] >= 1.15
        and stats["trades"] >= 8
        and stats["win_rate"] >= 0.55
    )


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for multiplier in SPREAD_MULTIPLIERS:
        subset = [row for row in rows if row["spread_multiplier"] == multiplier]
        total_trades = sum(int(row["test_trades"]) for row in subset)
        total_r = sum(float(row["test_pnl_r"]) for row in subset)
        weighted_win = (
            sum(float(row["test_win_rate"]) * int(row["test_trades"]) for row in subset)
            / total_trades
            if total_trades
            else 0.0
        )
        passed_folds = sum(bool(row["fold_pass"]) for row in subset)
        positive_folds = sum(float(row["test_pnl_r"]) > 0 for row in subset)
        max_drawdown = min(float(row["test_max_drawdown_r"]) for row in subset)
        item = {
            "spread_multiplier": multiplier,
            "total_pnl_r": round(total_r, 4),
            "total_trades": total_trades,
            "weighted_win_rate": round(weighted_win, 4),
            "mean_profit_factor": round(
                sum(float(row["test_profit_factor"]) for row in subset) / len(subset), 4
            ),
            "worst_fold_pnl_r": round(min(float(row["test_pnl_r"]) for row in subset), 4),
            "max_drawdown_r": round(max_drawdown, 4),
            "positive_folds": positive_folds,
            "passed_folds": passed_folds,
        }
        item["passes_stress_gate"] = (
            item["total_pnl_r"] >= GATE["min_total_r"]
            and item["positive_folds"] >= GATE["min_positive_folds"]
            and item["passed_folds"] >= GATE["min_passed_folds"]
            and item["weighted_win_rate"] >= GATE["min_weighted_win"]
            and item["max_drawdown_r"] >= GATE["max_drawdown_r"]
            and item["total_trades"] >= GATE["min_trades"]
        )
        summary.append(item)
    return summary


def write_outputs(rows: list[dict], summary: list[dict], candidate: dict) -> None:
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    OUTPUT_JSON.write_text(
        json.dumps(
            {"candidate": candidate, "gate": GATE, "summary": summary, "folds": rows},
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# SILVER H1 Cost Stress Test",
        "",
        "Best direct walk-forward candidate tested with inflated CSV spread cost.",
        "",
        "| Spread x | Pass | Total R | Positive | Passed | Trades | Win | Mean PF | Worst R | Max DD |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            "| {spread_multiplier:.1f} | {passes_stress_gate} | {total_pnl_r:.2f} | "
            "{positive_folds}/4 | {passed_folds}/4 | {total_trades} | "
            "{weighted_win_rate:.2%} | {mean_profit_factor:.2f} | "
            "{worst_fold_pnl_r:.2f} | {max_drawdown_r:.2f} |".format(**item)
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    candidate = load_candidate()
    point = get_symbol_point(SYMBOL)
    print(f"Loading {SYMBOL} {BASE_TIMEFRAME}; point={point}...")
    frame, features = load_case(SYMBOL, BASE_TIMEFRAME)
    rows = []

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

        for multiplier in SPREAD_MULTIPLIERS:
            adjusted_test = scale_spread(test_df, multiplier)
            stats = simulate_cost_aware(adjusted_test, probs, candidate, point)
            rows.append(
                {
                    "symbol": SYMBOL,
                    "base_timeframe": BASE_TIMEFRAME,
                    "fold": fold["name"],
                    "spread_multiplier": multiplier,
                    "test_start": test_period["start"],
                    "test_end": test_period["end"],
                    **candidate,
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

    summary = summarize(rows)
    write_outputs(rows, summary, candidate)
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    for item in summary:
        print(
            "spread={spread_multiplier:.1f} pass={passes_stress_gate} "
            "R={total_pnl_r:.2f} win={weighted_win_rate:.2%} "
            "pf={mean_profit_factor:.2f} dd={max_drawdown_r:.2f}".format(**item)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
