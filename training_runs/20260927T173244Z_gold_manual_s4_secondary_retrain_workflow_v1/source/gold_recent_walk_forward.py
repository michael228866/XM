from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import xgboost as xgb

from barrier_classifier_strategy import (
    HORIZON,
    build_barrier_target,
    build_profit_sample_weight,
    evaluate,
)
from barrier_final_train import FINAL_PARAMS
from drl_trading_v2 import BASE_FEATURES, add_indicators
from gemini import MTF_ORDER, TIMEFRAME_MAP


PROJECT_ROOT = Path(__file__).resolve().parent
SYMBOL = "GOLD#"
INCUMBENT_MODEL = PROJECT_ROOT / "gold_barrier_final_xgb.json"
CANDIDATE_MODEL = PROJECT_ROOT / "gold_barrier_recent_candidate_xgb.json"
REPORT_JSON = PROJECT_ROOT / "gold_recent_walk_forward.json"
REPORT_MD = PROJECT_ROOT / "gold_recent_walk_forward.md"
DEFAULT_TERMINAL = Path(
    os.environ.get("XM_TERMINAL_PATH", r"D:\XM2\terminal64.exe")
)

DEFAULT_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
DEFAULT_VALIDATION_START = datetime(2026, 4, 1, tzinfo=timezone.utc)
DEFAULT_TEST_START = datetime(2026, 6, 1, tzinfo=timezone.utc)
THRESHOLDS = (0.45, 0.475, 0.50, 0.525, 0.55)
TP_ATR_VALUES = (1.1, 1.3)


def parse_utc_date(value: str) -> datetime:
    parsed = datetime.strptime(value, "%Y-%m-%d")
    return parsed.replace(tzinfo=timezone.utc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward comparison of the incumbent and a recent GOLD model."
    )
    parser.add_argument("--terminal", type=Path, default=DEFAULT_TERMINAL)
    parser.add_argument("--start", type=parse_utc_date, default=DEFAULT_START)
    parser.add_argument(
        "--validation-start",
        type=parse_utc_date,
        default=DEFAULT_VALIDATION_START,
    )
    parser.add_argument(
        "--test-start", type=parse_utc_date, default=DEFAULT_TEST_START
    )
    parser.add_argument("--estimators", type=int, default=220)
    return parser.parse_args()


def timeframe_warmup_start(start: datetime, timeframe: str) -> datetime:
    if timeframe == "Monthly":
        return start - timedelta(days=760)
    if timeframe == "Weekly":
        return start - timedelta(days=210)
    if timeframe == "Daily":
        return start - timedelta(days=45)
    return start - timedelta(days=30)


def copy_rates(timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    rates = mt5.copy_rates_range(
        SYMBOL,
        mt5.TIMEFRAME_M1 if timeframe == "M1" else TIMEFRAME_MAP[timeframe],
        start,
        end,
    )
    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"No {SYMBOL} {timeframe} rates: {mt5.last_error()}"
        )

    frame = pd.DataFrame(rates).rename(
        columns={
            "open": "OPEN",
            "high": "HIGH",
            "low": "LOW",
            "close": "CLOSE",
            "tick_volume": "TICKVOL",
            "real_volume": "VOL",
            "spread": "SPREAD",
        }
    )
    frame["TIME_DT"] = pd.to_datetime(frame["time"], unit="s", utc=True).dt.tz_localize(
        None
    )
    return frame.sort_values("TIME_DT").drop_duplicates("TIME_DT")


def build_feature_frame(start: datetime, end: datetime) -> tuple[pd.DataFrame, list[str]]:
    m1 = copy_rates("M1", start - timedelta(days=7), end)
    print(
        f"Loaded M1: {len(m1):,} rows "
        f"({m1['TIME_DT'].iloc[0]} -> {m1['TIME_DT'].iloc[-1]})",
        flush=True,
    )
    m1 = add_indicators(m1)
    diff = m1["CLOSE"].diff()
    gain = diff.where(diff > 0, 0).rolling(14).mean()
    loss = (-diff.where(diff < 0, 0)).rolling(14).mean()
    m1["M1_RSI"] = 100 - (100 / (1 + (gain / (loss + 1e-6))))
    m1["VOLA_MA"] = m1["ATR"].rolling(240).mean()
    m1["VOLA_RATIO"] = m1["ATR"] / (m1["VOLA_MA"] + 1e-6)

    frame = m1.sort_values("TIME_DT")
    mtf_features: list[str] = []
    for timeframe in MTF_ORDER:
        higher = copy_rates(
            timeframe,
            timeframe_warmup_start(start, timeframe),
            end,
        )
        higher = add_indicators(higher)
        trend_col = f"{timeframe}_TREND"
        higher[trend_col] = np.where(
            higher["CLOSE"] > higher["CLOSE"].rolling(20).mean(), 1, -1
        )
        higher[trend_col] = higher[trend_col].shift(1)
        frame = pd.merge_asof(
            frame,
            higher[["TIME_DT", trend_col]],
            on="TIME_DT",
            direction="backward",
        )
        mtf_features.append(trend_col)
        print(f"Loaded {timeframe}: {len(higher):,} rows", flush=True)

    features = BASE_FEATURES + mtf_features
    frame[features] = frame[features].shift(1)
    frame["BARRIER_TARGET"] = build_barrier_target(frame)
    frame = frame.iloc[:-HORIZON]
    frame = frame[frame["TIME_DT"] >= start.replace(tzinfo=None)]
    frame = frame.dropna(subset=features + ["BARRIER_TARGET", "ATR"])
    return frame.reset_index(drop=True), features


def train_model(
    frame: pd.DataFrame, features: list[str], estimators: int
) -> xgb.XGBClassifier:
    target = frame["BARRIER_TARGET"].to_numpy(dtype=np.int8)
    weights = build_profit_sample_weight(frame, target)
    invalid_weights = ~np.isfinite(weights) | (weights <= 0)
    if invalid_weights.any():
        print(
            f"Replacing {int(invalid_weights.sum()):,} invalid sample weights "
            "with neutral weight 1.0",
            flush=True,
        )
        weights[invalid_weights] = 1.0
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        device="cpu",
        n_estimators=estimators,
        learning_rate=0.04,
        max_depth=5,
        min_child_weight=80,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=max(1, (os.cpu_count() or 2) - 1),
        verbosity=0,
    )
    model.fit(
        frame[features].astype(np.float32),
        target,
        sample_weight=weights,
    )
    return model


def strategy_params(threshold: float, tp_atr: float) -> dict:
    params = dict(FINAL_PARAMS)
    params["threshold"] = threshold
    params["tp_atr"] = tp_atr
    return params


def evaluate_model(
    model: xgb.XGBClassifier,
    frame: pd.DataFrame,
    features: list[str],
    params: dict,
) -> dict:
    probs = model.predict_proba(frame[features].astype(np.float32)).astype(np.float32)
    return evaluate(
        params,
        frame["CLOSE"].to_numpy(dtype=np.float64),
        frame["ATR"].to_numpy(dtype=np.float64),
        probs,
        hours=frame["TIME_DT"].dt.hour.to_numpy(dtype=np.int16),
        weekdays=frame["TIME_DT"].dt.dayofweek.to_numpy(dtype=np.int8),
        dates=frame["TIME_DT"].dt.date.to_numpy(),
        rsi_values=frame["M1_RSI"].to_numpy(dtype=np.float64),
    )


def compact_stats(stats: dict) -> dict:
    keys = (
        "pnl",
        "trades",
        "win_rate",
        "profit_factor",
        "max_drawdown_pct",
        "max_consecutive_losses",
        "take_profit_exits",
        "stop_loss_exits",
        "timeout_exits",
    )
    result = {}
    for key in keys:
        value = stats[key]
        if isinstance(value, (np.integer, int)):
            result[key] = int(value)
        elif np.isfinite(value):
            result[key] = round(float(value), 6)
        else:
            result[key] = None
    return result


def candidate_score(validation: dict) -> float:
    if validation["trades"] < 20:
        return -1e12 + validation["trades"]
    if validation["win_rate"] < 0.65 or validation["profit_factor"] < 1.10:
        return -1e9 + validation["pnl"]
    return (
        validation["pnl"]
        + validation["trades"] * 8.0
        + validation["win_rate"] * 2_000.0
        + validation["profit_factor"] * 150.0
    )


def markdown_report(report: dict) -> str:
    lines = [
        "# GOLD recent walk-forward",
        "",
        "This is research-only. It does not replace the live model.",
        "",
        "| Model / fold | Trades | Win | PF | PnL | DD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    rows = [
        ("Incumbent validation", report["incumbent"]["validation"]),
        ("Recent validation", report["recent"]["validation"]),
        ("Incumbent test", report["incumbent"]["test"]),
        ("Recent test", report["recent"]["test"]),
    ]
    for label, stats in rows:
        pf = "inf" if stats["profit_factor"] is None else f"{stats['profit_factor']:.2f}"
        lines.append(
            f"| {label} | {stats['trades']} | {stats['win_rate']:.2%} | "
            f"{pf} | {stats['pnl']:.2f} | {stats['max_drawdown_pct']:.2%} |"
        )
    lines.extend(
        [
            "",
            f"Selected threshold: `{report['selected_params']['threshold']}`",
            f"Selected TP/SL: `{report['selected_params']['tp_atr']}/"
            f"{report['selected_params']['sl_atr']}` ATR",
            f"Promotion gate: `{'PASS' if report['promotion_pass'] else 'FAIL'}`",
            "",
            "Promotion requires both recent folds to remain profitable, test win rate "
            "at least 65%, PF at least 1.15, at least 20 test trades, and more test "
            "trades than the incumbent.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if not (args.start < args.validation_start < args.test_start):
        raise ValueError("Expected start < validation-start < test-start")
    if args.estimators < 50:
        raise ValueError("--estimators must be at least 50")
    if not args.terminal.exists():
        raise FileNotFoundError(f"MT5 terminal not found: {args.terminal}")
    if not INCUMBENT_MODEL.exists():
        raise FileNotFoundError(f"Incumbent model not found: {INCUMBENT_MODEL}")

    if not mt5.initialize(path=str(args.terminal), timeout=10_000):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            raise RuntimeError(f"No {SYMBOL} tick: {mt5.last_error()}")
        end = datetime.fromtimestamp(tick.time, tz=timezone.utc)
        frame, features = build_feature_frame(args.start, end)
    finally:
        mt5.shutdown()

    validation_start = args.validation_start.replace(tzinfo=None)
    test_start = args.test_start.replace(tzinfo=None)
    train_validation = frame[frame["TIME_DT"] < validation_start].iloc[:-HORIZON]
    validation = frame[
        (frame["TIME_DT"] >= validation_start) & (frame["TIME_DT"] < test_start)
    ]
    train_test = frame[frame["TIME_DT"] < test_start].iloc[:-HORIZON]
    test = frame[frame["TIME_DT"] >= test_start]
    if min(map(len, (train_validation, validation, train_test, test))) == 0:
        raise RuntimeError("One or more walk-forward folds are empty")

    print(
        "Rows | "
        f"train_validation={len(train_validation):,} validation={len(validation):,} "
        f"train_test={len(train_test):,} test={len(test):,}",
        flush=True,
    )

    incumbent = xgb.XGBClassifier()
    incumbent.load_model(INCUMBENT_MODEL)
    incumbent.set_params(device="cpu")
    baseline_params = strategy_params(0.525, 1.3)
    incumbent_validation = evaluate_model(
        incumbent, validation, features, baseline_params
    )
    incumbent_test = evaluate_model(incumbent, test, features, baseline_params)

    print("Training recent validation model...", flush=True)
    validation_model = train_model(train_validation, features, args.estimators)
    sweep = []
    for threshold in THRESHOLDS:
        for tp_atr in TP_ATR_VALUES:
            params = strategy_params(threshold, tp_atr)
            stats = evaluate_model(validation_model, validation, features, params)
            sweep.append(
                {
                    "threshold": threshold,
                    "tp_atr": tp_atr,
                    "score": candidate_score(stats),
                    "validation": compact_stats(stats),
                }
            )
    selected = max(sweep, key=lambda item: item["score"])
    selected_params = strategy_params(selected["threshold"], selected["tp_atr"])

    print("Training recent test model...", flush=True)
    test_model = train_model(train_test, features, args.estimators)
    recent_test = evaluate_model(test_model, test, features, selected_params)
    test_model.save_model(CANDIDATE_MODEL)

    recent_validation = selected["validation"]
    compact_incumbent_validation = compact_stats(incumbent_validation)
    compact_incumbent_test = compact_stats(incumbent_test)
    compact_recent_test = compact_stats(recent_test)
    promotion_pass = bool(
        recent_validation["pnl"] > 0
        and recent_validation["win_rate"] >= 0.65
        and recent_validation["profit_factor"] >= 1.15
        and compact_recent_test["pnl"] > 0
        and compact_recent_test["win_rate"] >= 0.65
        and compact_recent_test["profit_factor"] >= 1.15
        and compact_recent_test["trades"] >= 20
        and compact_recent_test["trades"] > compact_incumbent_test["trades"]
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_only",
        "data": {
            "symbol": SYMBOL,
            "start": args.start.isoformat(),
            "validation_start": args.validation_start.isoformat(),
            "test_start": args.test_start.isoformat(),
            "end": frame["TIME_DT"].iloc[-1].isoformat(),
            "rows": len(frame),
            "features": features,
            "estimators": args.estimators,
        },
        "selected_params": {
            "threshold": selected_params["threshold"],
            "tp_atr": selected_params["tp_atr"],
            "sl_atr": selected_params["sl_atr"],
            "max_hold": selected_params["max_hold"],
        },
        "incumbent": {
            "model": INCUMBENT_MODEL.name,
            "validation": compact_incumbent_validation,
            "test": compact_incumbent_test,
        },
        "recent": {
            "model": CANDIDATE_MODEL.name,
            "validation": recent_validation,
            "test": compact_recent_test,
        },
        "promotion_pass": promotion_pass,
        "validation_sweep": sorted(sweep, key=lambda item: item["score"], reverse=True),
    }
    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REPORT_MD.write_text(markdown_report(report), encoding="utf-8")
    print(markdown_report(report), flush=True)
    print(f"Saved {REPORT_JSON.name}, {REPORT_MD.name}, {CANDIDATE_MODEL.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
