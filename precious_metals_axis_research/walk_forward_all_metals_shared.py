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
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402

from barrier_classifier_strategy import build_profit_sample_weight  # noqa: E402
from precious_metals_axis_research.cost_aware_xaueur_m5 import (  # noqa: E402
    get_symbol_point,
    simulate_cost_aware,
)
from precious_metals_axis_research.train_all_metals_h1_shared import (  # noqa: E402
    BASE_TIMEFRAME,
    MAX_ROWS_PER_SYMBOL,
    SYMBOLS,
    add_symbol_features,
    load_case,
)


OUTPUT_CSV = RESEARCH_DIR / "all_metals_h1_shared_walk_forward.csv"
OUTPUT_JSON = RESEARCH_DIR / "all_metals_h1_shared_walk_forward.json"
OUTPUT_MD = RESEARCH_DIR / "all_metals_h1_shared_walk_forward.md"

FOLDS = [
    {"name": "fold_1", "train": (0.00, 0.70), "test": (0.70, 0.84)},
    {"name": "fold_2", "train": (0.14, 0.84), "test": (0.84, 1.00)},
]

CANDIDATES = [
    {
        "name": "xpt_shared_best",
        "symbol": "XPTUSD#",
        "threshold": 0.52,
        "edge_threshold": 0.0,
        "tp_atr": 3.2,
        "sl_atr": 3.4,
        "max_hold": 120,
        "direction_mode": "long",
    },
    {
        "name": "xaueur_shared_best",
        "symbol": "XAUEUR#",
        "threshold": 0.56,
        "edge_threshold": 0.0,
        "tp_atr": 2.2,
        "sl_atr": 4.0,
        "max_hold": 120,
        "direction_mode": "both",
    },
    {
        "name": "gold_shared_best",
        "symbol": "GOLD#",
        "threshold": 0.60,
        "edge_threshold": 0.0,
        "tp_atr": 2.8,
        "sl_atr": 2.8,
        "max_hold": 72,
        "direction_mode": "both",
    },
    {
        "name": "xpd_watch",
        "symbol": "XPDUSD#",
        "threshold": 0.58,
        "edge_threshold": 0.0,
        "tp_atr": 2.2,
        "sl_atr": 3.4,
        "max_hold": 72,
        "direction_mode": "long",
    },
]


def slice_by_ratio(frame: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    return frame.iloc[int(len(frame) * start) : int(len(frame) * end)].copy()


def load_frames() -> tuple[dict[str, pd.DataFrame], list[str]]:
    loaded = {}
    feature_union: set[str] = set()
    symbol_features = {}
    for symbol in SYMBOLS:
        frame, features = load_case(symbol, BASE_TIMEFRAME)
        if len(frame) > MAX_ROWS_PER_SYMBOL:
            frame = frame.tail(MAX_ROWS_PER_SYMBOL).reset_index(drop=True)
        frame = add_symbol_features(frame, symbol)
        features = features + ["SYMBOL_ID"] + [
            f"IS_{candidate.replace('#', '').replace('/', '')}" for candidate in SYMBOLS
        ]
        loaded[symbol] = frame
        symbol_features[symbol] = features
        feature_union.update(features)

    shared_features = sorted(feature_union)
    for symbol, frame in loaded.items():
        missing = sorted(set(shared_features) - set(symbol_features[symbol]))
        for feature in missing:
            frame[feature] = 0.0
        frame[shared_features] = frame[shared_features].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        loaded[symbol] = frame.reset_index(drop=True)
    return loaded, shared_features


def train_fold_model(train_frames: list[pd.DataFrame], features: list[str]) -> xgb.XGBClassifier:
    train_df = pd.concat(train_frames, ignore_index=True)
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
        n_estimators=180,
        learning_rate=0.045,
        max_depth=4,
        min_child_weight=120,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbosity=0,
    )
    model.fit(train_df[features], train_df["BARRIER_TARGET"], sample_weight=sample_weight)
    return model


def stress_frame(frame: pd.DataFrame) -> pd.DataFrame:
    adjusted = frame.copy()
    if "SPREAD" in adjusted.columns:
        adjusted["SPREAD"] = adjusted["SPREAD"].fillna(0) * 3.0
    return adjusted


def summarize(rows: list[dict]) -> dict[str, dict]:
    summary = {}
    for candidate in sorted({row["candidate"] for row in rows}):
        subset = [row for row in rows if row["candidate"] == candidate]
        total_trades = sum(row["stress_trades"] for row in subset)
        weighted_win = (
            sum(row["stress_win_rate"] * row["stress_trades"] for row in subset) / total_trades
            if total_trades
            else 0.0
        )
        item = {
            "symbol": subset[0]["symbol"],
            "candidate": candidate,
            "folds": len(subset),
            "normal_total_r": round(sum(row["normal_pnl_r"] for row in subset), 4),
            "stress_total_r": round(sum(row["stress_pnl_r"] for row in subset), 4),
            "stress_total_trades": total_trades,
            "stress_weighted_win_rate": round(weighted_win, 4),
            "stress_mean_profit_factor": round(
                sum(row["stress_profit_factor"] for row in subset) / len(subset), 4
            ),
            "stress_positive_folds": sum(row["stress_pnl_r"] > 0 for row in subset),
            "stress_passed_folds": sum(row["fold_pass"] for row in subset),
            "stress_worst_fold_r": round(min(row["stress_pnl_r"] for row in subset), 4),
            "stress_max_drawdown_r": round(min(row["stress_max_drawdown_r"] for row in subset), 4),
        }
        item["verdict"] = (
            "walk_forward_candidate"
            if item["stress_positive_folds"] == item["folds"]
            and item["stress_passed_folds"] >= item["folds"] - 1
            and item["stress_total_r"] > 0
            and item["stress_total_trades"] >= 16
            else "failed_walk_forward"
        )
        summary[candidate] = item
    return summary


def write_outputs(rows: list[dict], summary: dict[str, dict]) -> None:
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    OUTPUT_JSON.write_text(
        json.dumps({"summary": summary, "folds": rows}, indent=2), encoding="utf-8"
    )
    lines = [
        "# All Metals Shared Walk-Forward",
        "",
        "Research-only. Retrains the shared H1 model on rolling folds and stress-tests candidate parameters with 3x spread.",
        "",
        "| Symbol | Candidate | Verdict | Stress R | Positive | Passed | Trades | Win | PF | Worst R | Max DD |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in sorted(summary.values(), key=lambda row: row["stress_total_r"], reverse=True):
        lines.append(
            "| {symbol} | {candidate} | {verdict} | {stress_total_r:.2f} | "
            "{stress_positive_folds}/{folds} | {stress_passed_folds}/{folds} | "
            "{stress_total_trades} | {stress_weighted_win_rate:.2%} | "
            "{stress_mean_profit_factor:.2f} | {stress_worst_fold_r:.2f} | "
            "{stress_max_drawdown_r:.2f} |".format(**item)
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    frames, features = load_frames()
    rows = []
    for fold in FOLDS:
        print(f"Training {fold['name']}...")
        train_frames = [slice_by_ratio(frame, *fold["train"]) for frame in frames.values()]
        model = train_fold_model(train_frames, features)
        for candidate in CANDIDATES:
            symbol = candidate["symbol"]
            test_df = slice_by_ratio(frames[symbol], *fold["test"])
            probs = model.predict_proba(test_df[features]).astype(np.float32)
            point = get_symbol_point(symbol)
            normal = simulate_cost_aware(test_df, probs, candidate, point)
            stress = simulate_cost_aware(stress_frame(test_df), probs, candidate, point)
            row = {
                "fold": fold["name"],
                "symbol": symbol,
                "candidate": candidate["name"],
                "test_start": test_df["TIME_DT"].iloc[0].isoformat(),
                "test_end": test_df["TIME_DT"].iloc[-1].isoformat(),
                **{key: value for key, value in candidate.items() if key not in {"name", "symbol"}},
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
                "fold_pass": (
                    stress["pnl_r"] > 0
                    and stress["profit_factor"] >= 1.12
                    and stress["win_rate"] >= 0.55
                    and stress["trades"] >= 6
                ),
            }
            rows.append(row)
            print(
                f"{fold['name']} {candidate['name']}: "
                f"stress_R={stress['pnl_r']:.2f} trades={stress['trades']} "
                f"win={stress['win_rate']:.2%}"
            )
    summary = summarize(rows)
    write_outputs(rows, summary)
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
