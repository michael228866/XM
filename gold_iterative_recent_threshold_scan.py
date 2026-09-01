from __future__ import annotations

from datetime import datetime, timezone
from itertools import product

import MetaTrader5 as mt5
import xgboost as xgb

from gold_recent_walk_forward import DEFAULT_TERMINAL, build_feature_frame
from gold_regime_experts_iterative import (
    MODEL_FILES,
    RECENT_START,
    SESSION_PROFILES,
    candidate_stats,
    predict_experts,
)
from gold_short_rule_research import compact_stats


def load_models(generation):
    models = {}
    for name, path in MODEL_FILES[generation].items():
        if not path.exists():
            raise FileNotFoundError(path)
        model = xgb.XGBClassifier()
        model.load_model(path)
        model.set_params(device="cpu")
        models[name] = model
    return models


def main():
    if not mt5.initialize(path=str(DEFAULT_TERMINAL), timeout=10_000):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        tick = mt5.symbol_info_tick("GOLD#")
        if tick is None:
            raise RuntimeError(f"No GOLD# tick: {mt5.last_error()}")
        end = datetime.fromtimestamp(tick.time, tz=timezone.utc)
        recent, features = build_feature_frame(RECENT_START, end)
    finally:
        mt5.shutdown()

    balanced, strength = predict_experts(
        load_models("balanced"), recent, features
    )
    time_decay, time_strength = predict_experts(
        load_models("time_decay"), recent, features
    )
    if not (strength == time_strength).all():
        raise RuntimeError("Trend routing differs between generations")
    record = {
        "balanced_probs": balanced,
        "time_decay_probs": time_decay,
        "trend_strength": strength,
    }
    ranked = []
    for weight, threshold, trend_floor, session, direction_mode in product(
        (0.0, 0.25, 0.5, 0.75, 1.0),
        tuple(round(0.75 + index * 0.005, 3) for index in range(12)),
        (0.0, 0.2, 0.4),
        SESSION_PROFILES,
        ("long", "short"),
    ):
        candidate = {
            "generation": "recent_threshold_scan",
            "balanced_weight": weight,
            "threshold": threshold,
            "min_trend_strength": trend_floor,
            "session_profile": session,
            "direction_mode": direction_mode,
        }
        stats = compact_stats(candidate_stats(recent, record, candidate))
        profit_factor = stats["profit_factor"] or 0.0
        qualified = bool(
            stats["trades"] >= 40
            and stats["win_rate"] >= 0.60
            and profit_factor >= 1.15
            and stats["pnl"] > 0
        )
        score = (
            (1_000_000.0 if qualified else 0.0)
            + stats["pnl"]
            + stats["trades"] * 2.0
            + stats["win_rate"] * 500.0
            + min(profit_factor, 3.0) * 250.0
        )
        ranked.append((score, candidate, stats, qualified))
    ranked.sort(key=lambda item: item[0], reverse=True)
    for score, candidate, stats, qualified in ranked[:20]:
        print(
            f"qualified={qualified} score={score:.2f} candidate={candidate} "
            f"stats={stats}",
            flush=True,
        )
    print("Best candidate by threshold", flush=True)
    for threshold in sorted({item[1]["threshold"] for item in ranked}):
        best = max(
            (item for item in ranked if item[1]["threshold"] == threshold),
            key=lambda item: item[0],
        )
        _, candidate, stats, qualified = best
        print(
            f"threshold={threshold:.3f} qualified={qualified} "
            f"candidate={candidate} stats={stats}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
