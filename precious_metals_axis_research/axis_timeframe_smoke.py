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

from barrier_classifier_strategy import (  # noqa: E402
    HORIZON,
    build_barrier_target,
    build_profit_sample_weight,
    evaluate,
)
from precious_metals_axis_research.axis_symbol_smoke import (  # noqa: E402
    DATA_DIRS,
    read_price_csv,
)


CASES = [
    ("SILVER#", "M5"),
    ("SILVER#", "M15"),
    ("XAUEUR#", "M5"),
    ("XAUEUR#", "M15"),
    ("XPTUSD#", "M30"),
    ("XPDUSD#", "M30"),
]
OUTPUT_CSV = RESEARCH_DIR / "axis_timeframe_smoke_results.csv"
OUTPUT_JSON = RESEARCH_DIR / "axis_timeframe_smoke_results.json"
OUTPUT_MD = RESEARCH_DIR / "axis_timeframe_smoke_report.md"

TRAIN_END_RATIO = 0.70
VALIDATION_END_RATIO = 0.92
MAX_ROWS = 220_000

PARAM_GRID = [
    {"threshold": 0.525, "edge_threshold": 0.00, "tp_atr": 1.1, "sl_atr": 2.0, "max_hold": 120, "direction_mode": "both"},
    {"threshold": 0.55, "edge_threshold": 0.05, "tp_atr": 1.3, "sl_atr": 2.0, "max_hold": 180, "direction_mode": "both"},
    {"threshold": 0.575, "edge_threshold": 0.10, "tp_atr": 1.3, "sl_atr": 2.3, "max_hold": 240, "direction_mode": "both"},
    {"threshold": 0.575, "edge_threshold": 0.10, "tp_atr": 1.3, "sl_atr": 2.3, "max_hold": 240, "direction_mode": "short"},
    {"threshold": 0.525, "edge_threshold": 0.00, "tp_atr": 1.3, "sl_atr": 2.0, "max_hold": 180, "direction_mode": "long"},
]

BASE_FEATURES = [
    "BASE_RSI",
    "ATR_PCT",
    "MACD_ATR",
    "BB_WIDTH",
    "BIAS_20",
    "BODY_PCT",
    "ROC_5",
    "VOLA_RATIO",
    "HOUR_SIN",
    "HOUR_COS",
    "DAY_OF_WEEK",
]


def find_files(symbol: str) -> dict[str, Path]:
    files = {}
    for data_dir in DATA_DIRS:
        if not data_dir.exists():
            continue
        for path in data_dir.glob(f"{symbol}_*.csv"):
            parts = path.stem.rsplit("_", 3)
            if len(parts) == 4:
                files[parts[1]] = path
    return files


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    true_range = pd.concat(
        [
            (df["HIGH"] - df["LOW"]).abs(),
            (df["HIGH"] - df["CLOSE"].shift()).abs(),
            (df["LOW"] - df["CLOSE"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR"] = true_range.rolling(14).mean()
    df["HOUR_SIN"] = np.sin(2 * np.pi * df["TIME_DT"].dt.hour / 24)
    df["HOUR_COS"] = np.cos(2 * np.pi * df["TIME_DT"].dt.hour / 24)
    df["DAY_OF_WEEK"] = df["TIME_DT"].dt.dayofweek / 7.0
    ema12 = df["CLOSE"].ewm(span=12, adjust=False).mean()
    ema26 = df["CLOSE"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df["MACD_HIST"] = macd - macd.ewm(span=9, adjust=False).mean()
    ma20 = df["CLOSE"].rolling(20).mean()
    df["BB_WIDTH"] = (df["CLOSE"].rolling(20).std() * 4) / (ma20 + 1e-9)
    df["BIAS_20"] = (df["CLOSE"] - ma20) / (ma20 + 1e-9)
    df["ROC_5"] = df["CLOSE"].pct_change(5)
    candle_range = df["HIGH"] - df["LOW"] + 1e-9
    df["BODY_PCT"] = (df["CLOSE"] - df["OPEN"]).abs() / candle_range
    df["ATR_PCT"] = df["ATR"] / (df["CLOSE"].abs() + 1e-9)
    df["MACD_ATR"] = df["MACD_HIST"] / (df["ATR"] + 1e-9)
    return df


def load_case(symbol: str, base_tf: str) -> tuple[pd.DataFrame, list[str]]:
    files = find_files(symbol)
    if base_tf not in files:
        raise FileNotFoundError(f"{symbol} {base_tf} file was not found.")

    base = read_price_csv(files[base_tf])
    if base is None:
        raise ValueError(f"Unable to read {files[base_tf]}")
    base = add_indicators(base)
    diff = base["CLOSE"].diff()
    gain = diff.where(diff > 0, 0).rolling(14).mean()
    loss = (-diff.where(diff < 0, 0)).rolling(14).mean()
    base["BASE_RSI"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    base["VOLA_MA"] = base["ATR"].rolling(240).mean()
    base["VOLA_RATIO"] = base["ATR"] / (base["VOLA_MA"] + 1e-9)
    frame = base.sort_values("TIME_DT").reset_index(drop=True)

    mtf_features = []
    for timeframe, path in sorted(files.items()):
        if timeframe == base_tf:
            continue
        tdf = read_price_csv(path)
        if tdf is None:
            continue
        tdf = add_indicators(tdf)
        trend_col = f"{timeframe}_TREND"
        tdf[trend_col] = np.where(tdf["CLOSE"] > tdf["CLOSE"].rolling(20).mean(), 1, -1)
        tdf[trend_col] = tdf[trend_col].shift(1)
        frame = pd.merge_asof(frame, tdf[["TIME_DT", trend_col]], on="TIME_DT")
        mtf_features.append(trend_col)

    features = BASE_FEATURES + mtf_features
    frame[features] = frame[features].shift(1)
    frame["BARRIER_TARGET"] = build_barrier_target(frame)
    frame = frame.iloc[:-HORIZON].dropna(
        subset=features + ["BARRIER_TARGET", "ATR", "CLOSE", "BASE_RSI"]
    )
    if len(frame) > MAX_ROWS:
        frame = frame.tail(MAX_ROWS)
    return frame.reset_index(drop=True), features


def train_model(train_df: pd.DataFrame, features: list[str]) -> xgb.XGBClassifier:
    sample_weight = build_profit_sample_weight(
        train_df, train_df["BARRIER_TARGET"].to_numpy(dtype=np.int8)
    )
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


def evaluate_case(params: dict, df: pd.DataFrame, probs: np.ndarray) -> dict:
    eval_params = {
        "threshold": params["threshold"],
        "edge_threshold": params["edge_threshold"],
        "tp_atr": params["tp_atr"],
        "sl_atr": params["sl_atr"],
        "min_tp_price": 0.0,
        "min_sl_price": 0.0,
        "max_hold": params["max_hold"],
        "cooldown_ticks": 0,
        "close_on_opposite": False,
        "direction_mode": params["direction_mode"],
        "initial_balance": 1000,
        "stop_out_balance": 0,
        "risk_per_trade": 0.02,
        "allowed_entry_hours": None,
        "allowed_entry_weekdays": None,
        "excluded_rsi_ranges": [],
        "max_daily_loss_pct": 0.05,
        "max_daily_trades": None,
        "extra_cost_points": 5.0,
        "drawdown_guard_start_pct": 0.08,
        "drawdown_guard_full_pct": 0.35,
        "drawdown_guard_min_risk_mult": 0.50,
        "loss_streak_threshold": 3,
        "loss_streak_risk_mult": 0.55,
        "loss_streak_pause_threshold": 3,
        "loss_streak_pause_ticks": 120,
        "rolling_guard_window": 30,
        "rolling_guard_min_trades": 18,
        "rolling_guard_min_profit_factor": 1.15,
        "rolling_guard_min_win_rate": None,
        "rolling_guard_risk_mult": 0.50,
        "rolling_guard_pause_ticks": 0,
    }
    stats = evaluate(
        eval_params,
        df["CLOSE"].to_numpy(dtype=np.float64),
        df["ATR"].to_numpy(dtype=np.float64),
        probs,
        dates=df["TIME_DT"].dt.date.to_numpy(),
        rsi_values=df["BASE_RSI"].to_numpy(dtype=np.float64),
    )
    return {
        "pnl": round(float(stats["pnl"]), 2),
        "trades": int(stats["trades"]),
        "win_rate": round(float(stats["win_rate"]), 4),
        "profit_factor": round(float(stats["profit_factor"]), 4),
        "max_drawdown_pct": round(float(stats["max_drawdown_pct"]), 4),
        "max_consecutive_losses": int(stats["max_consecutive_losses"]),
        "stopped_out": bool(stats["stopped_out"]),
    }


def run_case(symbol: str, base_tf: str) -> list[dict]:
    print(f"Preparing {symbol} {base_tf}...")
    frame, features = load_case(symbol, base_tf)
    train_end = int(len(frame) * TRAIN_END_RATIO)
    validation_end = int(len(frame) * VALIDATION_END_RATIO)
    train_df = frame.iloc[:train_end].copy()
    validation_df = frame.iloc[train_end:validation_end].copy()
    test_df = frame.iloc[validation_end:].copy()
    print(
        f"{symbol} {base_tf}: train={len(train_df):,} "
        f"validation={len(validation_df):,} test={len(test_df):,}"
    )
    model = train_model(train_df, features)
    validation_probs = model.predict_proba(validation_df[features]).astype("float32")
    test_probs = model.predict_proba(test_df[features]).astype("float32")

    rows = []
    for params in PARAM_GRID:
        validation_stats = evaluate_case(params, validation_df, validation_probs)
        test_stats = evaluate_case(params, test_df, test_probs)
        row = {
            "symbol": symbol,
            "base_timeframe": base_tf,
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
                and test_stats["trades"] >= 10
                and not test_stats["stopped_out"]
            ),
        }
        row["score"] = round(
            row["test_pnl"]
            + row["validation_pnl"] * 0.35
            + row["test_win_rate"] * 500.0
            + min(row["test_profit_factor"], 3.0) * 120.0
            - abs(min(row["test_drawdown_pct"], 0.0)) * 900.0,
            4,
        )
        rows.append(row)
    return rows


def write_outputs(rows: list[dict]) -> None:
    rows = sorted(
        rows,
        key=lambda row: (row["passes_smoke"], row["score"], row["test_pnl"]),
        reverse=True,
    )
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    OUTPUT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        "# Axis Timeframe Smoke Results",
        "",
        "Research-only test for symbol-specific base timeframes.",
        "",
        "| Symbol | TF | Pass | Test PnL | Test Win | Test PF | Test Trades | Params |",
        "|---|---|:---:|---:|---:|---:|---:|---|",
    ]
    for row in rows[:16]:
        lines.append(
            "| {symbol} | {base_timeframe} | {passes_smoke} | {test_pnl:.2f} | "
            "{test_win_rate:.2%} | {test_profit_factor:.2f} | {test_trades} | "
            "conf={threshold}, edge={edge_threshold}, tp/sl={tp_atr}/{sl_atr}, "
            "hold={max_hold}, dir={direction_mode} |".format(**row)
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = []
    for symbol, base_tf in CASES:
        rows.extend(run_case(symbol, base_tf))
    write_outputs(rows)
    passed = sum(1 for row in rows if row["passes_smoke"])
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Timeframe smoke candidates passed: {passed}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
