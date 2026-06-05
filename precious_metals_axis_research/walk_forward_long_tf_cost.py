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
import xgboost as xgb  # noqa: E402

from barrier_classifier_strategy import build_profit_sample_weight  # noqa: E402
from precious_metals_axis_research.axis_timeframe_smoke import load_case  # noqa: E402
from precious_metals_axis_research.cost_aware_xaueur_m5 import (  # noqa: E402
    get_symbol_point,
    simulate_cost_aware,
)


OUTPUT_CSV = RESEARCH_DIR / "long_tf_cost_walk_forward.csv"
OUTPUT_JSON = RESEARCH_DIR / "long_tf_cost_walk_forward.json"
OUTPUT_MD = RESEARCH_DIR / "long_tf_cost_walk_forward.md"

FOLDS = [
    {"name": "fold_1", "train": (0.00, 0.55), "test": (0.55, 0.66)},
    {"name": "fold_2", "train": (0.11, 0.66), "test": (0.66, 0.77)},
    {"name": "fold_3", "train": (0.22, 0.77), "test": (0.77, 0.88)},
    {"name": "fold_4", "train": (0.33, 0.88), "test": (0.88, 1.00)},
]

CASES = [
    {
        "symbol": "XPTUSD#",
        "base_timeframe": "H1",
        "candidates": [
            {
                "name": "xpt_best",
                "threshold": 0.54,
                "edge_threshold": 0.0,
                "tp_atr": 2.0,
                "sl_atr": 3.0,
                "max_hold": 48,
                "direction_mode": "long",
            },
            {
                "name": "xpt_wide_sl",
                "threshold": 0.54,
                "edge_threshold": 0.0,
                "tp_atr": 2.0,
                "sl_atr": 3.4,
                "max_hold": 48,
                "direction_mode": "long",
            },
            {
                "name": "xpt_long_hold",
                "threshold": 0.54,
                "edge_threshold": 0.0,
                "tp_atr": 2.0,
                "sl_atr": 3.0,
                "max_hold": 120,
                "direction_mode": "long",
            },
        ],
    },
    {
        "symbol": "SILVER#",
        "base_timeframe": "H1",
        "candidates": [
            {
                "name": "silver_best",
                "threshold": 0.60,
                "edge_threshold": 0.0,
                "tp_atr": 2.0,
                "sl_atr": 3.4,
                "max_hold": 120,
                "direction_mode": "long",
            }
            ,
            {
                "name": "silver_optimized",
                "threshold": 0.56,
                "edge_threshold": 0.0,
                "tp_atr": 2.4,
                "sl_atr": 2.8,
                "max_hold": 72,
                "direction_mode": "long",
            }
        ],
    },
]


def slice_by_ratio(frame, start: float, end: float):
    return frame.iloc[int(len(frame) * start) : int(len(frame) * end)].copy()


def compact_period(df):
    return {
        "start": df["TIME_DT"].iloc[0].isoformat(),
        "end": df["TIME_DT"].iloc[-1].isoformat(),
        "rows": len(df),
    }


def train_fold_model(train_df, features):
    sample_weight = build_profit_sample_weight(
        train_df, train_df["BARRIER_TARGET"].to_numpy(dtype=np.int8)
    )
    sample_weight = np.nan_to_num(sample_weight, nan=1.0, posinf=1.0, neginf=1.0)
    sample_weight = np.maximum(sample_weight, 1e-6)
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        device="cpu",
        n_estimators=170,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=80,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbosity=0,
    )
    model.fit(train_df[features], train_df["BARRIER_TARGET"], sample_weight=sample_weight)
    return model


def summarize(rows: list[dict]) -> dict[str, dict]:
    summary = {}
    for key in sorted({(row["symbol"], row["candidate"]) for row in rows}):
        symbol, candidate = key
        subset = [row for row in rows if row["symbol"] == symbol and row["candidate"] == candidate]
        total_r = sum(float(row["test_pnl_r"]) for row in subset)
        total_trades = sum(int(row["test_trades"]) for row in subset)
        weighted_win = (
            sum(float(row["test_win_rate"]) * int(row["test_trades"]) for row in subset)
            / total_trades
            if total_trades
            else 0.0
        )
        summary[f"{symbol}:{candidate}"] = {
            "symbol": symbol,
            "candidate": candidate,
            "folds": len(subset),
            "positive_folds": sum(float(row["test_pnl_r"]) > 0 for row in subset),
            "passed_folds": sum(bool(row["fold_pass"]) for row in subset),
            "total_pnl_r": round(total_r, 4),
            "total_trades": total_trades,
            "weighted_win_rate": round(weighted_win, 4),
            "min_fold_pnl_r": round(min(float(row["test_pnl_r"]) for row in subset), 4),
            "max_fold_drawdown_r": round(min(float(row["test_max_drawdown_r"]) for row in subset), 4),
        }
    return summary


def write_outputs(rows: list[dict], summary: dict[str, dict]) -> None:
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    OUTPUT_JSON.write_text(
        json.dumps({"folds": rows, "summary": summary}, indent=2), encoding="utf-8"
    )

    lines = [
        "# Long TF Cost-Aware Walk-Forward",
        "",
        "Each fold retrains the model and evaluates the next window with CSV spread cost.",
        "",
        "## Summary",
        "",
        "| Symbol | Candidate | Positive | Passed | Total R | Trades | Win | Worst Fold R | Max DD R |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in sorted(summary.values(), key=lambda value: value["total_pnl_r"], reverse=True):
        lines.append(
            "| {symbol} | {candidate} | {positive_folds}/{folds} | {passed_folds}/{folds} | "
            "{total_pnl_r:.2f} | {total_trades} | {weighted_win_rate:.2%} | "
            "{min_fold_pnl_r:.2f} | {max_fold_drawdown_r:.2f} |".format(**item)
        )

    lines.extend(
        [
            "",
            "## Folds",
            "",
            "| Symbol | Fold | Candidate | Period | R | Win | PF | Trades | DD R | Pass |",
            "|---|---|---|---|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {symbol} | {fold} | {candidate} | {test_start} -> {test_end} | "
            "{test_pnl_r:.2f} | {test_win_rate:.2%} | {test_profit_factor:.2f} | "
            "{test_trades} | {test_max_drawdown_r:.2f} | {fold_pass} |".format(**row)
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = []
    for case in CASES:
        symbol = case["symbol"]
        base_tf = case["base_timeframe"]
        point = get_symbol_point(symbol)
        print(f"Loading {symbol} {base_tf}; point={point}...")
        frame, features = load_case(symbol, base_tf)
        for fold in FOLDS:
            train_df = slice_by_ratio(frame, *fold["train"])
            test_df = slice_by_ratio(frame, *fold["test"])
            train_period = compact_period(train_df)
            test_period = compact_period(test_df)
            print(
                f"{symbol} {fold['name']}: train={train_period['rows']:,} "
                f"test={test_period['rows']:,}"
            )
            model = train_fold_model(train_df, features)
            test_probs = model.predict_proba(test_df[features]).astype(np.float32)
            for candidate in case["candidates"]:
                stats = simulate_cost_aware(test_df, test_probs, candidate, point)
                row = {
                    "symbol": symbol,
                    "base_timeframe": base_tf,
                    "fold": fold["name"],
                    "candidate": candidate["name"],
                    "train_start": train_period["start"],
                    "train_end": train_period["end"],
                    "test_start": test_period["start"],
                    "test_end": test_period["end"],
                    **{key: value for key, value in candidate.items() if key != "name"},
                    "test_pnl_r": stats["pnl_r"],
                    "test_trades": stats["trades"],
                    "test_win_rate": stats["win_rate"],
                    "test_profit_factor": stats["profit_factor"],
                    "test_max_drawdown_r": stats["max_drawdown_r"],
                    "test_avg_r": stats["avg_r"],
                    "test_max_loss_streak": stats["max_loss_streak"],
                    "fold_pass": (
                        stats["pnl_r"] > 0
                        and stats["profit_factor"] >= 1.15
                        and stats["trades"] >= 8
                        and stats["win_rate"] >= 0.55
                    ),
                }
                rows.append(row)

    summary = summarize(rows)
    write_outputs(rows, summary)
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    for item in sorted(summary.values(), key=lambda value: value["total_pnl_r"], reverse=True):
        print(
            f"{item['symbol']} {item['candidate']}: total_r={item['total_pnl_r']:.2f}, "
            f"positive={item['positive_folds']}/{item['folds']}, "
            f"passed={item['passed_folds']}/{item['folds']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
