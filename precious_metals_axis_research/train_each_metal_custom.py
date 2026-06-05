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
import xgboost as xgb  # noqa: E402

from barrier_classifier_strategy import build_profit_sample_weight  # noqa: E402
from precious_metals_axis_research.axis_timeframe_smoke import (  # noqa: E402
    TRAIN_END_RATIO,
    VALIDATION_END_RATIO,
    load_case,
)
from precious_metals_axis_research.cost_aware_xaueur_m5 import (  # noqa: E402
    get_symbol_point,
    simulate_cost_aware,
)


OUTPUT_CSV = RESEARCH_DIR / "each_metal_custom_results.csv"
OUTPUT_JSON = RESEARCH_DIR / "each_metal_custom_results.json"
OUTPUT_MD = RESEARCH_DIR / "each_metal_custom_report.md"
OUTPUT_BEST = RESEARCH_DIR / "each_metal_custom_best_by_symbol.json"
FAST_MODE = os.environ.get("EACH_METAL_FAST", "1") != "0"

SYMBOL_CONFIG = {
    "GOLD#": {
        "role": "anchor",
        "timeframes": ["M15", "H1"],
        "directions": ["long", "both"],
    },
    "SILVER#": {
        "role": "core_satellite",
        "timeframes": ["M30", "H1"],
        "directions": ["long", "both"],
    },
    "XAUEUR#": {
        "role": "gold_cross",
        "timeframes": ["M15", "H1"],
        "directions": ["short", "both"],
    },
    "XPTUSD#": {
        "role": "platinum",
        "timeframes": ["H1"],
        "directions": ["long", "both"],
    },
    "XPDUSD#": {
        "role": "palladium",
        "timeframes": ["H1"],
        "directions": ["long", "both"],
    },
    "GAUCNH#": {
        "role": "short_history_gold_cross",
        "timeframes": ["H1", "H4", "Daily"],
        "directions": ["long", "both"],
    },
}

MIN_ROWS_BY_TIMEFRAME = {
    "M1": 20_000,
    "M5": 8_000,
    "M15": 4_000,
    "M30": 2_000,
    "H1": 700,
    "H4": 250,
    "Daily": 120,
}
MAX_ROWS_BY_TIMEFRAME = {
    "M1": 90_000,
    "M5": 90_000,
    "M15": 15_000,
    "M30": 15_000,
    "H1": 15_000,
    "H4": 20_000,
    "Daily": 5_000,
}


def train_safe_model(train_df, features):
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
        n_estimators=50,
        learning_rate=0.09,
        max_depth=4,
        min_child_weight=80,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbosity=0,
    )
    model.fit(train_df[features], train_df["BARRIER_TARGET"], sample_weight=sample_weight)
    return model


def param_grid_for_timeframe(timeframe: str, directions: list[str]) -> list[dict]:
    if FAST_MODE:
        if timeframe in {"M1", "M5", "M15"}:
            thresholds = [0.56]
            edges = [0.00]
            tps = [1.3, 1.6]
            sls = [2.3, 2.8]
            holds = [180, 240]
        elif timeframe in {"M30", "H1"}:
            thresholds = [0.56]
            edges = [0.00]
            tps = [2.4, 3.2]
            sls = [3.6, 4.4]
            holds = [72, 120]
        else:
            thresholds = [0.58]
            edges = [0.00]
            tps = [2.6, 3.2]
            sls = [4.0, 4.8]
            holds = [30, 45]
        params = []
        for threshold, edge, tp, sl, hold, direction in product(
            thresholds, edges, tps, sls, holds, directions[:1]
        ):
            if tp > sl:
                continue
            params.append(
                {
                    "threshold": threshold,
                    "edge_threshold": edge,
                    "tp_atr": tp,
                    "sl_atr": sl,
                    "max_hold": hold,
                    "direction_mode": direction,
                }
            )
        return params

    if timeframe in {"M1", "M5", "M15"}:
        thresholds = [0.52, 0.55, 0.575, 0.60]
        edges = [0.00, 0.05, 0.10]
        tps = [1.1, 1.3, 1.6]
        sls = [2.0, 2.3, 2.8]
        holds = [120, 180, 240]
    elif timeframe in {"M30", "H1"}:
        thresholds = [0.54, 0.56, 0.58, 0.60, 0.62]
        edges = [0.00, 0.03, 0.06]
        tps = [1.8, 2.4, 3.2]
        sls = [3.0, 3.6, 4.4]
        holds = [48, 72, 120]
    else:
        thresholds = [0.54, 0.58, 0.62]
        edges = [0.00, 0.03]
        tps = [1.8, 2.6, 3.2]
        sls = [3.2, 4.0, 4.8]
        holds = [20, 30, 45]

    params = []
    for threshold, edge, tp, sl, hold, direction in product(
        thresholds, edges, tps, sls, holds, directions
    ):
        if tp > sl:
            continue
        params.append(
            {
                "threshold": threshold,
                "edge_threshold": edge,
                "tp_atr": tp,
                "sl_atr": sl,
                "max_hold": hold,
                "direction_mode": direction,
            }
        )
    return params


def scale_spread(frame, multiplier: float):
    adjusted = frame.copy()
    if "SPREAD" in adjusted.columns:
        adjusted["SPREAD"] = adjusted["SPREAD"].fillna(0) * multiplier
    return adjusted


def pass_gate(validation: dict, test: dict, stress: dict, timeframe: str) -> bool:
    min_trades = 12 if timeframe in {"H1", "H4", "Daily"} else 30
    return (
        validation["pnl_r"] > 0
        and test["pnl_r"] > 0
        and stress["pnl_r"] > 0
        and stress["trades"] >= min_trades
        and stress["win_rate"] >= 0.55
        and stress["profit_factor"] >= 1.12
        and stress["max_drawdown_r"] >= -18.0
    )


def score_row(row: dict) -> float:
    trade_penalty = 0.0 if row["stress3_trades"] >= row["min_required_trades"] else 1000.0
    return round(
        row["stress3_pnl_r"] * 150.0
        + row["test_pnl_r"] * 70.0
        + row["validation_pnl_r"] * 35.0
        + row["stress3_win_rate"] * 600.0
        + min(row["stress3_profit_factor"], 3.0) * 180.0
        - abs(row["stress3_max_drawdown_r"]) * 45.0
        - trade_penalty,
        4,
    )


def evaluate_symbol_timeframe(symbol: str, timeframe: str, directions: list[str]) -> list[dict]:
    print(f"Preparing {symbol} {timeframe}...")
    frame, features = load_case(symbol, timeframe)
    min_rows = MIN_ROWS_BY_TIMEFRAME.get(timeframe, 700)
    if len(frame) < min_rows:
        raise ValueError(f"too few rows for {symbol} {timeframe}: {len(frame)} < {min_rows}")
    max_rows = MAX_ROWS_BY_TIMEFRAME.get(timeframe)
    if max_rows is not None and len(frame) > max_rows:
        frame = frame.tail(max_rows).reset_index(drop=True)

    train_end = int(len(frame) * TRAIN_END_RATIO)
    validation_end = int(len(frame) * VALIDATION_END_RATIO)
    train_df = frame.iloc[:train_end].copy()
    validation_df = frame.iloc[train_end:validation_end].copy()
    test_df = frame.iloc[validation_end:].copy()
    print(
        f"{symbol} {timeframe}: train={len(train_df):,} "
        f"validation={len(validation_df):,} test={len(test_df):,}",
        flush=True,
    )

    model = train_safe_model(train_df, features)
    validation_probs = model.predict_proba(validation_df[features]).astype(np.float32)
    test_probs = model.predict_proba(test_df[features]).astype(np.float32)
    point = get_symbol_point(symbol)
    stress_test_df = scale_spread(test_df, 3.0)
    min_required_trades = 12 if timeframe in {"H1", "H4", "Daily"} else 30

    rows = []
    for params in param_grid_for_timeframe(timeframe, directions):
        validation_stats = simulate_cost_aware(validation_df, validation_probs, params, point)
        test_stats = simulate_cost_aware(test_df, test_probs, params, point)
        stress_stats = simulate_cost_aware(stress_test_df, test_probs, params, point)
        row = {
            "symbol": symbol,
            "base_timeframe": timeframe,
            **params,
            "min_required_trades": min_required_trades,
            "validation_pnl_r": validation_stats["pnl_r"],
            "validation_trades": validation_stats["trades"],
            "validation_win_rate": validation_stats["win_rate"],
            "validation_profit_factor": validation_stats["profit_factor"],
            "test_pnl_r": test_stats["pnl_r"],
            "test_trades": test_stats["trades"],
            "test_win_rate": test_stats["win_rate"],
            "test_profit_factor": test_stats["profit_factor"],
            "test_max_drawdown_r": test_stats["max_drawdown_r"],
            "stress3_pnl_r": stress_stats["pnl_r"],
            "stress3_trades": stress_stats["trades"],
            "stress3_win_rate": stress_stats["win_rate"],
            "stress3_profit_factor": stress_stats["profit_factor"],
            "stress3_max_drawdown_r": stress_stats["max_drawdown_r"],
        }
        row["passes_gate"] = pass_gate(validation_stats, test_stats, stress_stats, timeframe)
        row["score"] = score_row(row)
        rows.append(row)
    return rows


def best_by_symbol(rows: list[dict], skipped: list[dict]) -> dict:
    summary = {}
    for symbol in SYMBOL_CONFIG:
        subset = [row for row in rows if row["symbol"] == symbol]
        if not subset:
            summary[symbol] = {
                "verdict": "skipped",
                "skipped": [item for item in skipped if item["symbol"] == symbol],
            }
            continue
        subset = sorted(
            subset,
            key=lambda row: (row["passes_gate"], row["score"], row["stress3_pnl_r"]),
            reverse=True,
        )
        passed = [row for row in subset if row["passes_gate"]]
        summary[symbol] = {
            "verdict": "passed" if passed else "trained_but_failed_gate",
            "passed_candidates": len(passed),
            "best": subset[0],
        }
    return summary


def write_outputs(rows: list[dict], summary: dict, skipped: list[dict]) -> None:
    rows = sorted(
        rows,
        key=lambda row: (row["passes_gate"], row["score"], row["stress3_pnl_r"]),
        reverse=True,
    )
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    OUTPUT_JSON.write_text(
        json.dumps({"summary": summary, "skipped": skipped, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    OUTPUT_BEST.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Each Metal Custom Training",
        "",
        "Research-only. Every metal is trained separately and searched on its own candidate timeframes/conditions.",
        "",
        "| Symbol | Verdict | Passed | Best TF | Stress R | Stress Win | Stress PF | Stress DD | Params |",
        "|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for symbol, item in summary.items():
        if item["verdict"] == "skipped":
            lines.append(f"| {symbol} | skipped | 0 | - | - | - | - | - | no usable timeframe |")
            continue
        best = item["best"]
        lines.append(
            "| {row_symbol} | {verdict} | {passed_candidates} | {base_timeframe} | "
            "{stress3_pnl_r:.2f} | {stress3_win_rate:.2%} | "
            "{stress3_profit_factor:.2f} | {stress3_max_drawdown_r:.2f} | "
            "conf={threshold}, edge={edge_threshold}, tp/sl={tp_atr}/{sl_atr}, "
            "hold={max_hold}, dir={direction_mode} |".format(
                row_symbol=symbol, **item, **best
            )
        )
    if skipped:
        lines.extend(["", "## Skipped Cases", ""])
        for item in skipped:
            lines.append(f"- {item['symbol']} {item['timeframe']}: {item['reason']}")
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = []
    skipped = []
    for symbol, config in SYMBOL_CONFIG.items():
        for timeframe in config["timeframes"]:
            try:
                rows.extend(
                    evaluate_symbol_timeframe(symbol, timeframe, config["directions"])
                )
            except Exception as exc:
                print(f"Skipped {symbol} {timeframe}: {exc}")
                skipped.append(
                    {"symbol": symbol, "timeframe": timeframe, "reason": str(exc)}
                )
    if not rows:
        raise ValueError("No training rows were produced.")
    summary = best_by_symbol(rows, skipped)
    write_outputs(rows, summary, skipped)
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_BEST}")
    print(f"Passed candidates: {sum(1 for row in rows if row['passes_gate'])}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
