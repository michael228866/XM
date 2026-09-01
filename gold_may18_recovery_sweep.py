from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np

from barrier_final_train import FINAL_PARAMS, TRAIN_END_RATIO, prepare_barrier_data
from barrier_meta_overlay import (
    add_overlay_regime_features,
    load_meta_overlay_model,
    predict_overlay_risk_mult,
)
from gold_recent_walk_forward import DEFAULT_TERMINAL, build_feature_frame
from gold_regime_experts_walk_forward import (
    CURRENT_MODEL_FILE,
    MAY_HOURS,
    MAY_MODEL_FILE,
    MAY_WEEKDAYS,
    RECENT_START,
    benchmark_current,
    evaluate_frame,
    load_model,
)
from gold_short_rule_research import compact_stats


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_JSON = PROJECT_ROOT / "gold_may18_recovery_sweep.json"
REPORT_MD = PROJECT_ROOT / "gold_may18_recovery_sweep.md"
SELECTED_CONFIG = PROJECT_ROOT / "gold_may18_recovery_candidate.json"

THRESHOLDS = (0.50, 0.525, 0.55)
TP_ATR_VALUES = (1.1, 1.2, 1.3)
SL_ATR_VALUES = (2.0,)
MAX_HOLD_VALUES = (180,)
META_QUALITY_FLOORS = (None, 0.20, 0.30, 0.40, 0.50, 0.60)
SESSION_PROFILES = {
    "may_baseline": (MAY_HOURS, MAY_WEEKDAYS),
    "controlled_expanded": (
        tuple(sorted(set(MAY_HOURS) | {2, 4})),
        (0, 1, 2, 3, 4),
    ),
}
RISK_PER_TRADE = 0.014


def make_params(candidate, extra_cost_points=5.0):
    hours, weekdays = SESSION_PROFILES[candidate["session_profile"]]
    params = dict(FINAL_PARAMS)
    params.update(
        {
            "threshold": candidate["threshold"],
            "edge_threshold": 0.0,
            "tp_atr": candidate["tp_atr"],
            "sl_atr": candidate["sl_atr"],
            "max_hold": candidate["max_hold"],
            "direction_mode": "long",
            "risk_per_trade": RISK_PER_TRADE,
            "allowed_entry_hours": list(hours),
            "allowed_entry_weekdays": list(weekdays),
            "excluded_rsi_ranges": [(35.0, 45.0)],
            "extra_cost_points": extra_cost_points,
        }
    )
    return params


def candidate_stats(frame, probs, quality, candidate, cost=5.0):
    filtered = probs.copy()
    quality_floor = candidate["meta_quality_floor"]
    if quality_floor is not None:
        rejected = quality < quality_floor
        filtered[rejected, 1:] = 0.0
        filtered[rejected, 0] = 1.0
    return evaluate_frame(make_params(candidate, cost), frame, filtered)


def fold_pass(stats, min_trades):
    profit_factor = stats["profit_factor"]
    return bool(
        not stats["stopped_out"]
        and stats["trades"] >= min_trades
        and stats["pnl"] > 0
        and stats["win_rate"] >= 0.65
        and (profit_factor is None or profit_factor >= 1.15)
        and stats["max_drawdown_pct"] >= -0.15
    )


def pf_value(stats):
    value = stats["profit_factor"]
    return 3.0 if value is None else min(float(value), 3.0)


def candidate_score(folds):
    if not all(stats["trades"] >= 50 for stats in folds):
        return -2e12 + sum(float(stats["trades"]) for stats in folds)
    if not all(fold_pass(stats, 50) for stats in folds):
        return -1e12 + sum(
            float(stats["pnl"])
            + float(stats["trades"])
            + float(stats["win_rate"]) * 500.0
            + pf_value(stats) * 300.0
            + float(stats["max_drawdown_pct"]) * 500.0
            for stats in folds
        )
    return sum(
        float(stats["pnl"])
        + float(stats["trades"]) * 3.0
        + float(stats["win_rate"]) * 1000.0
        + pf_value(stats) * 500.0
        + float(stats["max_drawdown_pct"]) * 500.0
        for stats in folds
    )


def public_candidate(candidate):
    return {
        "model_file": MAY_MODEL_FILE.name,
        "model_output_mode": "three_class",
        "use_meta_overlay": True,
        "risk_per_trade": RISK_PER_TRADE,
        "threshold": candidate["threshold"],
        "tp_atr": candidate["tp_atr"],
        "sl_atr": candidate["sl_atr"],
        "max_hold": candidate["max_hold"],
        "session_profile": candidate["session_profile"],
        "meta_quality_floor": candidate["meta_quality_floor"],
        "allowed_entry_hours": list(SESSION_PROFILES[candidate["session_profile"]][0]),
        "allowed_entry_weekdays": list(
            SESSION_PROFILES[candidate["session_profile"]][1]
        ),
    }


def markdown_report(report):
    lines = [
        "# GOLD May-18 recovery sweep",
        "",
        "The saved May-18 model is evaluated only on data after its training cutoff.",
        "All exits use intrabar HIGH/LOW with conservative same-bar stop priority.",
        "",
        "| Fold | Trades | Win | PF | PnL | DD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, stats in report["selected"]["folds"].items():
        pf = "inf" if stats["profit_factor"] is None else f"{stats['profit_factor']:.2f}"
        lines.append(
            f"| {name} | {stats['trades']} | {stats['win_rate']:.2%} | "
            f"{pf} | {stats['pnl']:.2f} | {stats['max_drawdown_pct']:.2%} |"
        )
    lines.extend(
        [
            "",
            f"Current benchmark: `{json.dumps(report['current_benchmark'])}`",
            "",
            f"Promotion gate: `{'PASS' if report['promotion_pass'] else 'FAIL'}`",
            "",
            f"Selected: `{json.dumps(report['selected']['params'])}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    history, features = prepare_barrier_data()
    test_start = int(len(history) * TRAIN_END_RATIO)
    out_of_sample = history.iloc[test_start:].copy().reset_index(drop=True)
    out_of_sample, regime_features = add_overlay_regime_features(out_of_sample)
    out_of_sample = out_of_sample.dropna(subset=regime_features).reset_index(drop=True)
    split_one = len(out_of_sample) // 3
    split_two = split_one * 2
    selection_folds = {
        "historical_selection_1": out_of_sample.iloc[:split_one].reset_index(drop=True),
        "historical_selection_2": out_of_sample.iloc[split_one:split_two].reset_index(
            drop=True
        ),
    }
    historical_holdout = out_of_sample.iloc[split_two:].reset_index(drop=True)
    model = load_model(MAY_MODEL_FILE)
    meta_model, meta_config = load_meta_overlay_model()
    risk_rule = tuple(float(value) for value in meta_config["risk_rule"])
    selection_probs = {
        name: model.predict_proba(frame[features]).astype(np.float32)
        for name, frame in selection_folds.items()
    }
    selection_quality = {
        name: predict_overlay_risk_mult(
            meta_model,
            frame,
            selection_probs[name],
            regime_features,
            risk_rule,
        )[1]
        for name, frame in selection_folds.items()
    }
    print(
        f"OOS rows={len(out_of_sample):,} {out_of_sample['TIME_DT'].iloc[0]} -> "
        f"{out_of_sample['TIME_DT'].iloc[-1]}",
        flush=True,
    )

    candidates = [
        {
            "threshold": threshold,
            "tp_atr": tp_atr,
            "sl_atr": sl_atr,
            "max_hold": max_hold,
            "session_profile": session,
            "meta_quality_floor": quality_floor,
        }
        for threshold, tp_atr, sl_atr, max_hold, session, quality_floor in product(
            THRESHOLDS,
            TP_ATR_VALUES,
            SL_ATR_VALUES,
            MAX_HOLD_VALUES,
            SESSION_PROFILES,
            META_QUALITY_FLOORS,
        )
    ]
    ranked = []
    for index, candidate in enumerate(candidates, start=1):
        folds = {
            name: candidate_stats(
                frame,
                selection_probs[name],
                selection_quality[name],
                candidate,
            )
            for name, frame in selection_folds.items()
        }
        fold_values = list(folds.values())
        ranked.append(
            {
                **candidate,
                "qualified": all(fold_pass(stats, 50) for stats in fold_values),
                "score": candidate_score(fold_values),
                "folds": {name: compact_stats(stats) for name, stats in folds.items()},
            }
        )
        if index % 20 == 0:
            print(f"Evaluated {index}/{len(candidates)} candidates", flush=True)
    ranked.sort(key=lambda item: item["score"], reverse=True)
    qualified = [item for item in ranked if item["qualified"]]
    covered = [
        item
        for item in ranked
        if all(stats["trades"] >= 50 for stats in item["folds"].values())
    ]
    selected = qualified[0] if qualified else covered[0] if covered else ranked[0]

    holdout_probs = model.predict_proba(historical_holdout[features]).astype(np.float32)
    holdout_quality = predict_overlay_risk_mult(
        meta_model,
        historical_holdout,
        holdout_probs,
        regime_features,
        risk_rule,
    )[1]
    historical_holdout_stats = candidate_stats(
        historical_holdout, holdout_probs, holdout_quality, selected
    )

    if not DEFAULT_TERMINAL.exists():
        raise FileNotFoundError(DEFAULT_TERMINAL)
    if not mt5.initialize(path=str(DEFAULT_TERMINAL), timeout=10_000):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        tick = mt5.symbol_info_tick("GOLD#")
        if tick is None:
            raise RuntimeError(f"No GOLD# tick: {mt5.last_error()}")
        recent_end = datetime.fromtimestamp(tick.time, tz=timezone.utc)
        recent, recent_features = build_feature_frame(RECENT_START, recent_end)
    finally:
        mt5.shutdown()
    if set(features) != set(recent_features):
        raise RuntimeError("Historical and recent feature sets differ")

    recent, recent_regime_features = add_overlay_regime_features(recent)
    if recent_regime_features != regime_features:
        raise RuntimeError("Historical and recent regime feature sets differ")
    recent = recent.dropna(subset=regime_features).reset_index(drop=True)
    recent_probs = model.predict_proba(recent[features]).astype(np.float32)
    recent_quality = predict_overlay_risk_mult(
        meta_model,
        recent,
        recent_probs,
        regime_features,
        risk_rule,
    )[1]
    recent_stats = candidate_stats(
        recent, recent_probs, recent_quality, selected
    )
    recent_cost = candidate_stats(
        recent, recent_probs, recent_quality, selected, cost=10.0
    )
    current_stats = benchmark_current(recent, features)

    compact_holdout = compact_stats(historical_holdout_stats)
    compact_recent = compact_stats(recent_stats)
    compact_cost = compact_stats(recent_cost)
    compact_current = compact_stats(current_stats)
    promotion_pass = bool(
        qualified
        and fold_pass(historical_holdout_stats, 50)
        and fold_pass(recent_stats, 50)
        and compact_recent["trades"] > compact_current["trades"]
        and compact_recent["win_rate"] > compact_current["win_rate"]
        and compact_recent["profit_factor"] is not None
        and compact_recent["profit_factor"] >= 1.15
        and compact_cost["pnl"] > 0
        and compact_cost["profit_factor"] is not None
        and compact_cost["profit_factor"] >= 1.05
    )
    selected_params = public_candidate(selected)
    selected_config = {**selected_params, "promotion_pass": promotion_pass}
    SELECTED_CONFIG.write_text(
        json.dumps(selected_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "promotion_pass" if promotion_pass else "research_only",
        "data": {
            "history_rows": len(history),
            "model_training_cutoff_index": test_start,
            "oos_start": out_of_sample["TIME_DT"].iloc[0].isoformat(),
            "oos_end": out_of_sample["TIME_DT"].iloc[-1].isoformat(),
            "recent_start": recent["TIME_DT"].iloc[0].isoformat(),
            "recent_end": recent["TIME_DT"].iloc[-1].isoformat(),
        },
        "qualified_count": len(qualified),
        "selected": {
            "params": selected_params,
            "folds": {
                **selected["folds"],
                "historical_holdout": compact_holdout,
                "2026_recent": compact_recent,
                "2026_recent_cost_10": compact_cost,
            },
        },
        "current_benchmark": compact_current,
        "promotion_pass": promotion_pass,
        "ranked_selection": ranked,
        "current_model_file": CURRENT_MODEL_FILE.name,
    }
    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REPORT_MD.write_text(markdown_report(report), encoding="utf-8")
    print(markdown_report(report), flush=True)
    print(f"Saved {REPORT_JSON.name}, {REPORT_MD.name}, {SELECTED_CONFIG.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
