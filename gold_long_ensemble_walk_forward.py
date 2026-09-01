from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np

from barrier_classifier_strategy import HORIZON
from barrier_meta_overlay import (
    add_overlay_regime_features,
    load_final_model,
    load_meta_overlay_model,
    predict_overlay_risk_mult,
)
from barrier_research_suite import (
    make_direction_probs,
    predict_positive,
    train_binary_model,
)
from gold_long_recent_walk_forward import (
    make_params,
    public_params,
)
from gold_recent_walk_forward import (
    DEFAULT_START,
    DEFAULT_TERMINAL,
    DEFAULT_TEST_START,
    DEFAULT_VALIDATION_START,
    build_feature_frame,
)
from gold_short_rule_research import compact_stats, evaluate_frame


PROJECT_ROOT = Path(__file__).resolve().parent
CANDIDATE_MODEL = PROJECT_ROOT / "gold_long_ensemble_candidate_xgb.json"
REPORT_JSON = PROJECT_ROOT / "gold_long_ensemble_walk_forward.json"
REPORT_MD = PROJECT_ROOT / "gold_long_ensemble_walk_forward.md"
N_ESTIMATORS = 220

BINARY_THRESHOLDS = (0.60, 0.65, 0.70, 0.75)
INCUMBENT_BUY_THRESHOLDS = (0.40, 0.45, 0.50)
META_QUALITY_FLOORS = (None, 0.50, 0.60)
TP_ATR_VALUES = (1.1, 1.3)
SESSION_PROFILES = ("current", "current_all_weekdays", "expanded")


def is_frequency_candidate(stats: dict) -> bool:
    profit_factor = stats["profit_factor"]
    return bool(
        not stats["stopped_out"]
        and stats["trades"] >= 25
        and stats["pnl"] > 0
        and stats["win_rate"] >= 0.62
        and (profit_factor is None or profit_factor >= 1.25)
        and stats["max_drawdown_pct"] >= -0.15
    )


def score(stats: dict) -> float:
    if not is_frequency_candidate(stats):
        return -1e12 + float(stats["pnl"])
    profit_factor = stats["profit_factor"]
    finite_pf = 3.0 if profit_factor is None else min(float(profit_factor), 3.0)
    return (
        float(stats["pnl"])
        + float(stats["trades"]) * 1_000.0
        + float(stats["win_rate"]) * 600.0
        + finite_pf * 100.0
        + float(stats["max_drawdown_pct"]) * 400.0
    )


def ensemble_probs(
    binary_prob: np.ndarray,
    incumbent_probs: np.ndarray,
    quality: np.ndarray,
    incumbent_threshold: float,
    quality_floor: float | None,
) -> np.ndarray:
    probs = make_direction_probs(binary_prob, "long")
    mask = (
        (incumbent_probs[:, 1] >= incumbent_threshold)
        & (incumbent_probs[:, 1] >= incumbent_probs[:, 2])
    )
    if quality_floor is not None:
        mask &= quality >= quality_floor
    probs[~mask, 1:] = 0.0
    return probs


def candidate_params(values: tuple) -> tuple[dict, dict]:
    (
        binary_threshold,
        incumbent_threshold,
        quality_floor,
        tp_atr,
        session_profile,
    ) = values
    params = make_params(binary_threshold, tp_atr, 1.6, 90, session_profile)
    public = public_params(params, session_profile)
    public.update(
        {
            "incumbent_buy_threshold": incumbent_threshold,
            "meta_quality_floor": quality_floor,
        }
    )
    return params, public


def evaluate_candidate(
    frame,
    binary_prob,
    incumbent_probs,
    quality,
    values,
    extra_cost_points=5.0,
):
    params, public = candidate_params(values)
    params["extra_cost_points"] = extra_cost_points
    probs = ensemble_probs(
        binary_prob,
        incumbent_probs,
        quality,
        public["incumbent_buy_threshold"],
        public["meta_quality_floor"],
    )
    return evaluate_frame(params, frame, probs), public


def fold_pass(stats: dict, min_trades: int) -> bool:
    profit_factor = stats["profit_factor"]
    return bool(
        not stats["stopped_out"]
        and stats["trades"] >= min_trades
        and stats["pnl"] > 0
        and stats["win_rate"] >= 0.60
        and (profit_factor is None or profit_factor >= 1.15)
        and stats["max_drawdown_pct"] >= -0.20
    )


def positive_subperiod(stats: dict) -> bool:
    profit_factor = stats["profit_factor"]
    return bool(
        stats["trades"] >= 3
        and stats["pnl"] > 0
        and (profit_factor is None or profit_factor >= 1.0)
    )


def markdown_report(report: dict) -> str:
    lines = [
        "# GOLD long ensemble walk-forward",
        "",
        "Research-only unless the promotion gate passes.",
        "",
        "| Fold | Trades | Win | PF | PnL | DD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("validation", "test", "june_july", "august", "test_cost_10"):
        stats = report["selected"][name]
        pf = "inf" if stats["profit_factor"] is None else f"{stats['profit_factor']:.2f}"
        lines.append(
            f"| {name} | {stats['trades']} | {stats['win_rate']:.2%} | "
            f"{pf} | {stats['pnl']:.2f} | {stats['max_drawdown_pct']:.2%} |"
        )
    lines.extend(
        [
            "",
            f"Promotion gate: `{'PASS' if report['promotion_pass'] else 'FAIL'}`",
            "",
            f"Selected parameters: `{json.dumps(report['selected']['params'])}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    if not DEFAULT_TERMINAL.exists():
        raise FileNotFoundError(f"MT5 terminal not found: {DEFAULT_TERMINAL}")
    if not mt5.initialize(path=str(DEFAULT_TERMINAL), timeout=10_000):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        tick = mt5.symbol_info_tick("GOLD#")
        if tick is None:
            raise RuntimeError(f"No GOLD# tick: {mt5.last_error()}")
        end = datetime.fromtimestamp(tick.time, tz=timezone.utc)
        frame, features = build_feature_frame(DEFAULT_START, end)
    finally:
        mt5.shutdown()

    frame, regime_features = add_overlay_regime_features(frame)
    frame = frame.dropna(subset=regime_features).reset_index(drop=True)
    validation_start = DEFAULT_VALIDATION_START.replace(tzinfo=None)
    test_start = DEFAULT_TEST_START.replace(tzinfo=None)
    august_start = datetime(2026, 8, 1)
    train_validation = frame[frame["TIME_DT"] < validation_start].iloc[:-HORIZON]
    validation = frame[
        (frame["TIME_DT"] >= validation_start) & (frame["TIME_DT"] < test_start)
    ].copy().reset_index(drop=True)
    train_test = frame[frame["TIME_DT"] < test_start].iloc[:-HORIZON]
    test = frame[frame["TIME_DT"] >= test_start].copy().reset_index(drop=True)
    june_july = test[test["TIME_DT"] < august_start].copy().reset_index(drop=True)
    august = test[test["TIME_DT"] >= august_start].copy().reset_index(drop=True)
    if min(map(len, (train_validation, validation, train_test, test, june_july, august))) == 0:
        raise RuntimeError("One or more walk-forward folds are empty")

    incumbent = load_final_model()
    meta_model, meta_config = load_meta_overlay_model()
    rule = tuple(float(value) for value in meta_config["risk_rule"])
    validation_incumbent = incumbent.predict_proba(validation[features]).astype(np.float32)
    test_incumbent = incumbent.predict_proba(test[features]).astype(np.float32)
    _, validation_quality = predict_overlay_risk_mult(
        meta_model, validation, validation_incumbent, regime_features, rule
    )
    _, test_quality = predict_overlay_risk_mult(
        meta_model, test, test_incumbent, regime_features, rule
    )

    validation_model = train_binary_model(train_validation, features, 1, N_ESTIMATORS)
    validation_binary = predict_positive(validation_model, validation, features)
    candidates = []
    grid = product(
        BINARY_THRESHOLDS,
        INCUMBENT_BUY_THRESHOLDS,
        META_QUALITY_FLOORS,
        TP_ATR_VALUES,
        SESSION_PROFILES,
    )
    for values in grid:
        stats, public = evaluate_candidate(
            validation,
            validation_binary,
            validation_incumbent,
            validation_quality,
            values,
        )
        if is_frequency_candidate(stats):
            candidates.append((score(stats), values, public, stats))
    if not candidates:
        raise RuntimeError("No ensemble candidate passed the validation gate")
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, selected_values, selected_public, validation_stats = candidates[0]

    test_model = train_binary_model(train_test, features, 1, N_ESTIMATORS)
    test_model.save_model(CANDIDATE_MODEL)
    test_binary = predict_positive(test_model, test, features)
    test_stats, _ = evaluate_candidate(
        test,
        test_binary,
        test_incumbent,
        test_quality,
        selected_values,
    )
    split = len(june_july)
    june_july_stats, _ = evaluate_candidate(
        june_july,
        test_binary[:split],
        test_incumbent[:split],
        test_quality[:split],
        selected_values,
    )
    august_stats, _ = evaluate_candidate(
        august,
        test_binary[split:],
        test_incumbent[split:],
        test_quality[split:],
        selected_values,
    )
    cost_stats, _ = evaluate_candidate(
        test,
        test_binary,
        test_incumbent,
        test_quality,
        selected_values,
        extra_cost_points=10.0,
    )
    compact_validation = compact_stats(validation_stats)
    compact_test = compact_stats(test_stats)
    compact_june_july = compact_stats(june_july_stats)
    compact_august = compact_stats(august_stats)
    compact_cost = compact_stats(cost_stats)
    promotion_pass = bool(
        is_frequency_candidate(compact_validation)
        and fold_pass(compact_test, 20)
        and positive_subperiod(compact_june_july)
        and positive_subperiod(compact_august)
        and compact_cost["pnl"] > 0
        and (
            compact_cost["profit_factor"] is None
            or compact_cost["profit_factor"] >= 1.05
        )
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "promotion_pass" if promotion_pass else "research_only",
        "model": CANDIDATE_MODEL.name,
        "qualified_count": len(candidates),
        "data": {
            "validation_start": DEFAULT_VALIDATION_START.isoformat(),
            "test_start": DEFAULT_TEST_START.isoformat(),
            "august_start": august_start.isoformat(),
            "end": frame["TIME_DT"].iloc[-1].isoformat(),
        },
        "selected": {
            "params": selected_public,
            "validation": compact_validation,
            "test": compact_test,
            "june_july": compact_june_july,
            "august": compact_august,
            "test_cost_10": compact_cost,
        },
        "promotion_pass": promotion_pass,
    }
    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REPORT_MD.write_text(markdown_report(report), encoding="utf-8")
    print(markdown_report(report), flush=True)
    print(f"Saved {CANDIDATE_MODEL.name}, {REPORT_JSON.name}, {REPORT_MD.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
