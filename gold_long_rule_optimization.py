from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np

from barrier_classifier_strategy import evaluate
from barrier_final_train import FINAL_PARAMS, TRAIN_END_RATIO, prepare_barrier_data
from barrier_meta_overlay import (
    add_overlay_regime_features,
    load_final_model,
    load_meta_overlay_model,
    predict_overlay_risk_mult,
)
from gold_recent_walk_forward import (
    DEFAULT_TERMINAL,
    DEFAULT_TEST_START,
    DEFAULT_VALIDATION_START,
    SYMBOL,
    build_feature_frame,
)


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_JSON = PROJECT_ROOT / "gold_long_rule_optimization.json"
REPORT_MD = PROJECT_ROOT / "gold_long_rule_optimization.md"

CURRENT_HOURS = frozenset(FINAL_PARAMS["allowed_entry_hours"])
CURRENT_WEEKDAYS = frozenset(FINAL_PARAMS["allowed_entry_weekdays"])
EXPANDED_HOURS = frozenset({2, 4})
THURSDAY = frozenset({3})
ALL_WEEKDAYS = frozenset(range(5))

AUGMENT_THRESHOLDS = (0.45, 0.475, 0.50, 0.52, 0.525)
AUGMENT_QUALITY_FLOORS = (0.55, 0.60, 0.65, 0.70)
AUGMENT_EDGE_THRESHOLDS = (0.0, 0.10, 0.20, 0.30)
TP_ATR_VALUES = (1.1, 1.3)
MAX_HOLD_VALUES = (120, 180)
SESSION_PROFILES = (
    "base_only",
    "approved_expanded",
    "hour18_only",
    "approved_plus_hour18",
    "all_weekdays_current_hours",
    "all_weekdays_expanded_hours",
    "all_weekdays_all_hours",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize GOLD long entry rules without retraining the live model."
    )
    parser.add_argument("--terminal", type=Path, default=DEFAULT_TERMINAL)
    parser.add_argument(
        "--recent-start",
        type=lambda value: datetime.strptime(value, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        ),
        default=DEFAULT_VALIDATION_START,
    )
    parser.add_argument(
        "--validation-start",
        type=lambda value: datetime.strptime(value, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        ),
        default=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )
    parser.add_argument(
        "--forward-start",
        type=lambda value: datetime.strptime(value, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        ),
        default=DEFAULT_TEST_START,
    )
    return parser.parse_args()


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
        "stopped_out",
    )
    result = {}
    for key in keys:
        value = stats[key]
        if isinstance(value, (bool, np.bool_)):
            result[key] = bool(value)
        elif isinstance(value, (int, np.integer)):
            result[key] = int(value)
        elif np.isfinite(value):
            result[key] = round(float(value), 6)
        else:
            result[key] = None
    return result


def finite_profit_factor(stats: dict) -> float:
    value = stats["profit_factor"]
    return 99.0 if not np.isfinite(value) else float(value)


def prepare_historical_reference():
    frame, features = prepare_barrier_data()
    frame, regime_features = add_overlay_regime_features(frame)
    out_of_sample = frame.iloc[int(len(frame) * TRAIN_END_RATIO) :].copy()
    out_of_sample = out_of_sample.dropna(subset=regime_features).reset_index(drop=True)
    if out_of_sample.empty:
        raise RuntimeError("Historical out-of-sample reference is empty")
    return features, regime_features, out_of_sample


def prepare_recent_frame(terminal: Path, start: datetime):
    if not terminal.exists():
        raise FileNotFoundError(f"MT5 terminal not found: {terminal}")
    if not mt5.initialize(path=str(terminal), timeout=10_000):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            raise RuntimeError(f"No {SYMBOL} tick: {mt5.last_error()}")
        end = datetime.fromtimestamp(tick.time, tz=timezone.utc)
        frame, features = build_feature_frame(start, end)
    finally:
        mt5.shutdown()
    frame, regime_features = add_overlay_regime_features(frame)
    frame = frame.dropna(subset=regime_features).reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("Recent MT5 frame is empty after feature warm-up")
    return frame, features, regime_features


def predict_fold(model, meta_model, config, frame, features, regime_features):
    probs = model.predict_proba(frame[features].astype(np.float32)).astype(np.float32)
    rule = tuple(float(value) for value in config["risk_rule"])
    risk_mult, quality = predict_overlay_risk_mult(
        meta_model,
        frame,
        probs,
        regime_features,
        rule,
    )
    return probs, risk_mult, quality


def base_and_expanded_sessions(frame):
    hours = frame["TIME_DT"].dt.hour.to_numpy(dtype=np.int16)
    weekdays = frame["TIME_DT"].dt.dayofweek.to_numpy(dtype=np.int8)
    base = np.isin(hours, tuple(CURRENT_HOURS)) & np.isin(
        weekdays, tuple(CURRENT_WEEKDAYS)
    )
    expanded = (
        np.isin(hours, tuple(EXPANDED_HOURS))
        & np.isin(weekdays, tuple(CURRENT_WEEKDAYS))
    ) | (
        np.isin(hours, tuple(CURRENT_HOURS))
        & np.isin(weekdays, tuple(THURSDAY))
    )
    return hours, weekdays, base, expanded


def augmentation_session(profile: str, hours, weekdays, base, expanded):
    if profile == "base_only":
        return base
    if profile == "approved_expanded":
        return base | expanded
    hour18 = (hours == 18) & np.isin(weekdays, tuple(ALL_WEEKDAYS))
    if profile == "hour18_only":
        return hour18
    if profile == "approved_plus_hour18":
        return base | expanded | hour18
    if profile == "all_weekdays_current_hours":
        return np.isin(hours, tuple(CURRENT_HOURS)) & np.isin(
            weekdays, tuple(ALL_WEEKDAYS)
        )
    if profile == "all_weekdays_expanded_hours":
        return np.isin(hours, tuple(CURRENT_HOURS | EXPANDED_HOURS)) & np.isin(
            weekdays, tuple(ALL_WEEKDAYS)
        )
    if profile == "all_weekdays_all_hours":
        return np.isin(weekdays, tuple(ALL_WEEKDAYS))
    raise ValueError(f"Unknown session profile: {profile}")


def eligible_mask(frame, probs, quality, candidate=None):
    buy_prob = probs[:, 1]
    sell_prob = probs[:, 2]
    edge = np.abs(buy_prob - sell_prob)
    hours, weekdays, base, expanded = base_and_expanded_sessions(frame)
    rsi = frame["M1_RSI"].to_numpy(dtype=np.float64)
    rsi_ok = (rsi >= 22.0) & ~((rsi >= 35.0) & (rsi <= 45.0))
    long_side = buy_prob >= sell_prob

    current_entry = (
        (base & (buy_prob >= 0.525))
        | (expanded & (buy_prob >= 0.525) & (quality >= 0.55))
        | (
            base
            & (buy_prob >= 0.52)
            & (edge >= 0.30)
            & (quality >= 0.50)
        )
    )
    if candidate is None:
        return rsi_ok & long_side & current_entry

    session = augmentation_session(
        candidate["session_profile"], hours, weekdays, base, expanded
    )
    augmentation = (
        session
        & (buy_prob >= candidate["augment_threshold"])
        & (edge >= candidate["augment_edge_threshold"])
        & (quality >= candidate["augment_quality_floor"])
    )
    return rsi_ok & long_side & (current_entry | augmentation)


def strategy_params(candidate=None, extra_cost_points=5.0):
    params = dict(FINAL_PARAMS)
    params.update(
        {
            "threshold": 1e-6,
            "edge_threshold": 0.0,
            "direction_mode": "long",
            "allowed_entry_hours": None,
            "allowed_entry_weekdays": None,
            "excluded_rsi_ranges": [],
            "tp_atr": 1.3 if candidate is None else candidate["tp_atr"],
            "max_hold": 180 if candidate is None else candidate["max_hold"],
            "extra_cost_points": extra_cost_points,
        }
    )
    return params


def evaluate_policy(
    frame,
    probs,
    risk_mult,
    quality,
    candidate=None,
    extra_cost_points=5.0,
):
    mask = eligible_mask(frame, probs, quality, candidate)
    filtered_probs = probs.copy()
    filtered_probs[~mask, 1:] = 0.0
    return evaluate(
        strategy_params(candidate, extra_cost_points),
        frame["CLOSE"].to_numpy(dtype=np.float64),
        frame["ATR"].to_numpy(dtype=np.float64),
        filtered_probs,
        dates=frame["TIME_DT"].dt.date.to_numpy(),
        entry_risk_mult=risk_mult,
    )


def score(stats: dict) -> float:
    return (
        float(stats["pnl"])
        + float(stats["trades"]) * 5.0
        + float(stats["win_rate"]) * 600.0
        + min(finite_profit_factor(stats), 3.0) * 100.0
        + float(stats["max_drawdown_pct"]) * 500.0
    )


def passes_selection(stats: dict, baseline: dict) -> bool:
    baseline_pf = finite_profit_factor(baseline)
    return bool(
        not stats["stopped_out"]
        and stats["trades"] >= max(20, int(np.ceil(baseline["trades"] * 1.05)))
        and stats["pnl"] > 0
        and stats["win_rate"] >= max(0.58, baseline["win_rate"] - 0.08)
        and finite_profit_factor(stats) >= max(1.10, baseline_pf * 0.85)
        and stats["max_drawdown_pct"] >= baseline["max_drawdown_pct"] - 0.06
    )


def passes_untouched(stats: dict, baseline: dict) -> bool:
    return bool(
        not stats["stopped_out"]
        and stats["trades"] > baseline["trades"]
        and stats["pnl"] > 0
        and stats["win_rate"] >= max(0.60, baseline["win_rate"] - 0.05)
        and finite_profit_factor(stats) >= 1.15
        and stats["max_drawdown_pct"] >= baseline["max_drawdown_pct"] - 0.05
    )


def candidate_grid():
    for values in product(
        AUGMENT_THRESHOLDS,
        AUGMENT_QUALITY_FLOORS,
        AUGMENT_EDGE_THRESHOLDS,
        TP_ATR_VALUES,
        MAX_HOLD_VALUES,
        SESSION_PROFILES,
    ):
        threshold, quality, edge, tp_atr, max_hold, session_profile = values
        yield {
            "augment_threshold": threshold,
            "augment_quality_floor": quality,
            "augment_edge_threshold": edge,
            "tp_atr": tp_atr,
            "sl_atr": 2.0,
            "max_hold": max_hold,
            "session_profile": session_profile,
        }


def fold_period(frame):
    return [
        frame["TIME_DT"].iloc[0].isoformat(),
        frame["TIME_DT"].iloc[-1].isoformat(),
    ]


def markdown_report(report: dict) -> str:
    lines = [
        "# GOLD long-rule optimization",
        "",
        "The incumbent model is unchanged. Candidate selection uses development and validation only.",
        "",
        "| Fold | Version | Trades | Win | PF | PnL | DD |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for fold in (
        "recent_development",
        "recent_validation",
        "historical_reference",
        "recent_forward",
    ):
        for version in ("baseline", "candidate"):
            stats = report["results"][fold][version]
            pf = "inf" if stats["profit_factor"] is None else f"{stats['profit_factor']:.2f}"
            lines.append(
                f"| {fold} | {version} | {stats['trades']} | "
                f"{stats['win_rate']:.2%} | {pf} | {stats['pnl']:.2f} | "
                f"{stats['max_drawdown_pct']:.2%} |"
            )
    lines.extend(
        [
            "",
            f"Historical 10-point cost stress: `{'PASS' if report['cost_stress']['historical_pass'] else 'FAIL'}`",
            f"Recent 10-point cost stress: `{'PASS' if report['cost_stress']['recent_pass'] else 'FAIL'}`",
            f"Promotion gate: `{'PASS' if report['promotion_pass'] else 'FAIL'}`",
            "",
            f"Selected parameters: `{json.dumps(report['selected_params'], ensure_ascii=False)}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if not (args.recent_start < args.validation_start < args.forward_start):
        raise ValueError(
            "Expected recent-start < validation-start < forward-start"
        )
    features, historical_regime_features, historical_reference = (
        prepare_historical_reference()
    )
    recent, recent_features, recent_regime_features = prepare_recent_frame(
        args.terminal, args.recent_start
    )
    if features != recent_features:
        raise RuntimeError("Historical and recent model feature lists differ")
    if historical_regime_features != recent_regime_features:
        raise RuntimeError("Historical and recent regime feature lists differ")

    model = load_final_model()
    meta_model, config = load_meta_overlay_model()
    expected_regime_features = list(config["regime_features"])
    if historical_regime_features != expected_regime_features:
        raise RuntimeError("Meta-model regime features differ from research data")

    validation_start = args.validation_start.replace(tzinfo=None)
    forward_start = args.forward_start.replace(tzinfo=None)
    data = {
        "recent_development": recent[
            recent["TIME_DT"] < validation_start
        ].copy().reset_index(drop=True),
        "recent_validation": recent[
            (recent["TIME_DT"] >= validation_start)
            & (recent["TIME_DT"] < forward_start)
        ].copy().reset_index(drop=True),
        "historical_reference": historical_reference,
        "recent_forward": recent[
            recent["TIME_DT"] >= forward_start
        ].copy().reset_index(drop=True),
    }
    if min(map(len, data.values())) == 0:
        raise RuntimeError("One or more optimization folds are empty")
    predictions = {
        name: predict_fold(
            model,
            meta_model,
            config,
            frame,
            features,
            historical_regime_features,
        )
        for name, frame in data.items()
    }
    baselines = {
        name: evaluate_policy(frame, *predictions[name])
        for name, frame in data.items()
    }

    development = data["recent_development"]
    development_predictions = predictions["recent_development"]
    candidates = []
    for index, candidate in enumerate(candidate_grid(), start=1):
        stats = evaluate_policy(development, *development_predictions, candidate)
        if passes_selection(stats, baselines["recent_development"]):
            candidates.append((score(stats), candidate, stats))
        if index % 200 == 0:
            print(f"Evaluated {index} development candidates", flush=True)
    if not candidates:
        raise RuntimeError("No long candidate passed the development gate")
    candidates.sort(key=lambda item: item[0], reverse=True)

    validation = data["recent_validation"]
    validation_predictions = predictions["recent_validation"]
    finalists = []
    for _, candidate, development_stats in candidates[:40]:
        validation_stats = evaluate_policy(
            validation, *validation_predictions, candidate
        )
        if passes_selection(validation_stats, baselines["recent_validation"]):
            finalists.append(
                (score(validation_stats), candidate, development_stats, validation_stats)
            )
    if not finalists:
        raise RuntimeError("No long candidate passed the validation gate")
    finalists.sort(key=lambda item: item[0], reverse=True)
    _, selected, development_stats, validation_stats = finalists[0]

    candidate_results = {
        "recent_development": development_stats,
        "recent_validation": validation_stats,
    }
    for name in ("historical_reference", "recent_forward"):
        candidate_results[name] = evaluate_policy(
            data[name], *predictions[name], selected
        )

    historical_cost = evaluate_policy(
        data["historical_reference"],
        *predictions["historical_reference"],
        selected,
        extra_cost_points=10.0,
    )
    recent_cost = evaluate_policy(
        data["recent_forward"],
        *predictions["recent_forward"],
        selected,
        extra_cost_points=10.0,
    )
    historical_cost_pass = bool(
        historical_cost["pnl"] > 0 and finite_profit_factor(historical_cost) >= 1.05
    )
    recent_cost_pass = bool(
        recent_cost["pnl"] > 0 and finite_profit_factor(recent_cost) >= 1.05
    )
    promotion_pass = bool(
        passes_selection(validation_stats, baselines["recent_validation"])
        and passes_untouched(
            candidate_results["historical_reference"],
            baselines["historical_reference"],
        )
        and passes_untouched(
            candidate_results["recent_forward"], baselines["recent_forward"]
        )
        and historical_cost_pass
        and recent_cost_pass
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "promotion_pass" if promotion_pass else "research_only",
        "model": Path(config["main_model_path"]).name,
        "development_qualified_count": len(candidates),
        "validation_qualified_count": len(finalists),
        "data": {name: fold_period(frame) for name, frame in data.items()},
        "selected_params": selected,
        "results": {
            name: {
                "baseline": compact_stats(baselines[name]),
                "candidate": compact_stats(candidate_results[name]),
            }
            for name in data
        },
        "cost_stress": {
            "historical_test_10_points": compact_stats(historical_cost),
            "recent_forward_10_points": compact_stats(recent_cost),
            "historical_pass": historical_cost_pass,
            "recent_pass": recent_cost_pass,
        },
        "promotion_pass": promotion_pass,
    }
    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REPORT_MD.write_text(markdown_report(report), encoding="utf-8")
    print(markdown_report(report), flush=True)
    print(f"Saved {REPORT_JSON.name} and {REPORT_MD.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
