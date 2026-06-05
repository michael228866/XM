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
from precious_metals_axis_research.axis_symbol_smoke import (  # noqa: E402
    evaluate_params,
    load_symbol_frame,
)


SYMBOLS = [
    symbol.strip()
    for symbol in os.environ.get("AXIS_SYMBOLS", "GOLD#,SILVER#,XAUEUR#").split(",")
    if symbol.strip()
]
SYMBOL_ID = {symbol: idx for idx, symbol in enumerate(SYMBOLS)}
RUN_NAME = os.environ.get(
    "AXIS_RUN_NAME",
    "axis_shared_" + "_".join(symbol.replace("#", "") for symbol in SYMBOLS).lower(),
)
OUTPUT_CSV = RESEARCH_DIR / f"{RUN_NAME}_results.csv"
OUTPUT_JSON = RESEARCH_DIR / f"{RUN_NAME}_results.json"
OUTPUT_MD = RESEARCH_DIR / f"{RUN_NAME}_report.md"
OUTPUT_MODEL = RESEARCH_DIR / f"{RUN_NAME}_xgb.json"

TRAIN_END_RATIO = 0.70
VALIDATION_END_RATIO = 0.92
MAX_ROWS_PER_SYMBOL = 260_000

PARAM_GRID = [
    {
        "threshold": 0.525,
        "edge_threshold": 0.00,
        "tp_atr": 1.1,
        "sl_atr": 2.0,
        "max_hold": 180,
        "direction_mode": "both",
    },
    {
        "threshold": 0.55,
        "edge_threshold": 0.05,
        "tp_atr": 1.3,
        "sl_atr": 2.0,
        "max_hold": 180,
        "direction_mode": "both",
    },
    {
        "threshold": 0.575,
        "edge_threshold": 0.10,
        "tp_atr": 1.3,
        "sl_atr": 2.3,
        "max_hold": 240,
        "direction_mode": "both",
    },
    {
        "threshold": 0.575,
        "edge_threshold": 0.10,
        "tp_atr": 1.3,
        "sl_atr": 2.3,
        "max_hold": 240,
        "direction_mode": "short",
    },
    {
        "threshold": 0.525,
        "edge_threshold": 0.00,
        "tp_atr": 1.3,
        "sl_atr": 2.0,
        "max_hold": 180,
        "direction_mode": "long",
    },
]


def add_shared_features(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["SYMBOL_ID"] = float(SYMBOL_ID[symbol])
    frame["IS_GOLD"] = 1.0 if symbol == "GOLD#" else 0.0
    frame["IS_SILVER"] = 1.0 if symbol == "SILVER#" else 0.0
    frame["IS_XAUEUR"] = 1.0 if symbol == "XAUEUR#" else 0.0
    frame["IS_XPTUSD"] = 1.0 if symbol == "XPTUSD#" else 0.0
    frame["IS_XPDUSD"] = 1.0 if symbol == "XPDUSD#" else 0.0
    return frame


def load_splits() -> tuple[pd.DataFrame, dict[str, dict], list[str]]:
    splits: dict[str, dict] = {}
    shared_features: list[str] | None = None

    for symbol in SYMBOLS:
        print(f"Loading {symbol}...")
        frame, features = load_symbol_frame(symbol)
        if len(frame) > MAX_ROWS_PER_SYMBOL:
            frame = frame.tail(MAX_ROWS_PER_SYMBOL).reset_index(drop=True)
        frame = add_shared_features(frame, symbol)
        symbol_features = features + [
            "SYMBOL_ID",
            "IS_GOLD",
            "IS_SILVER",
            "IS_XAUEUR",
            "IS_XPTUSD",
            "IS_XPDUSD",
        ]
        if shared_features is None:
            shared_features = symbol_features
        else:
            missing = sorted(set(shared_features) - set(symbol_features))
            if missing:
                raise ValueError(f"{symbol} missing shared features: {missing}")

        train_end = int(len(frame) * TRAIN_END_RATIO)
        validation_end = int(len(frame) * VALIDATION_END_RATIO)
        splits[symbol] = {
            "train": frame.iloc[:train_end].copy(),
            "validation": frame.iloc[train_end:validation_end].copy(),
            "test": frame.iloc[validation_end:].copy(),
        }
        print(
            f"{symbol}: train={len(splits[symbol]['train']):,} "
            f"validation={len(splits[symbol]['validation']):,} "
            f"test={len(splits[symbol]['test']):,}"
        )

    if shared_features is None:
        raise ValueError("No symbols loaded.")
    train_df = pd.concat([splits[symbol]["train"] for symbol in SYMBOLS], ignore_index=True)
    return train_df, splits, shared_features


def train_shared_model(train_df: pd.DataFrame, features: list[str]) -> xgb.XGBClassifier:
    sample_weight = build_profit_sample_weight(
        train_df, train_df["BARRIER_TARGET"].to_numpy(dtype=np.int8)
    )
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        device="cpu",
        n_estimators=220,
        learning_rate=0.045,
        max_depth=4,
        min_child_weight=120,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbosity=0,
    )
    model.fit(train_df[features], train_df["BARRIER_TARGET"], sample_weight=sample_weight)
    model.save_model(OUTPUT_MODEL)
    return model


def score_row(row: dict) -> float:
    if row["test_trades"] < 20:
        return -1_000_000.0 + row["test_pnl"]
    return (
        row["test_pnl"]
        + row["validation_pnl"] * 0.4
        + row["test_win_rate"] * 700.0
        + min(row["test_profit_factor"], 3.0) * 180.0
        - abs(min(row["test_drawdown_pct"], 0.0)) * 1200.0
    )


def evaluate_symbol_grid(
    model: xgb.XGBClassifier,
    splits: dict[str, dict],
    features: list[str],
) -> list[dict]:
    rows: list[dict] = []
    for symbol in SYMBOLS:
        validation_df = splits[symbol]["validation"]
        test_df = splits[symbol]["test"]
        validation_probs = model.predict_proba(validation_df[features]).astype("float32")
        test_probs = model.predict_proba(test_df[features]).astype("float32")

        for params in PARAM_GRID:
            validation_stats = evaluate_params(params, validation_df, validation_probs)
            test_stats = evaluate_params(params, test_df, test_probs)
            row = {
                "symbol": symbol,
                **params,
                "validation_pnl": validation_stats["pnl"],
                "validation_trades": validation_stats["trades"],
                "validation_win_rate": validation_stats["win_rate"],
                "validation_profit_factor": validation_stats["profit_factor"],
                "validation_drawdown_pct": validation_stats["max_drawdown_pct"],
                "test_pnl": test_stats["pnl"],
                "test_trades": test_stats["trades"],
                "test_win_rate": test_stats["win_rate"],
                "test_profit_factor": test_stats["profit_factor"],
                "test_drawdown_pct": test_stats["max_drawdown_pct"],
                "test_max_loss_streak": test_stats["max_consecutive_losses"],
                "passes_smoke": (
                    validation_stats["pnl"] > 0
                    and test_stats["pnl"] > 0
                    and test_stats["profit_factor"] >= 1.20
                    and test_stats["trades"] >= 20
                    and not test_stats["stopped_out"]
                ),
            }
            row["score"] = round(score_row(row), 4)
            rows.append(row)
    return sorted(
        rows,
        key=lambda item: (item["passes_smoke"], item["score"], item["test_pnl"]),
        reverse=True,
    )


def write_outputs(rows: list[dict]) -> None:
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    OUTPUT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        "# Shared Precious Metals Model",
        "",
        "Research-only shared model.",
        "",
        "Symbols: " + ", ".join(SYMBOLS),
        "",
        "| Symbol | Pass | Score | Test PnL | Test Win | Test PF | Test Trades | Params |",
        "|---|:---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows[:15]:
        lines.append(
            "| {symbol} | {passes_smoke} | {score:.1f} | {test_pnl:.2f} | "
            "{test_win_rate:.2%} | {test_profit_factor:.2f} | {test_trades} | "
            "conf={threshold}, edge={edge_threshold}, tp/sl={tp_atr}/{sl_atr}, "
            "hold={max_hold}, dir={direction_mode} |".format(**row)
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    train_df, splits, features = load_splits()
    print(f"Training shared model on {len(train_df):,} rows...")
    model = train_shared_model(train_df, features)
    rows = evaluate_symbol_grid(model, splits, features)
    write_outputs(rows)
    passed = sum(1 for row in rows if row["passes_smoke"])
    print(f"Wrote {OUTPUT_MODEL}")
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Shared model smoke candidates passed: {passed}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
