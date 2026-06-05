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
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402

from barrier_classifier_strategy import build_profit_sample_weight  # noqa: E402
from precious_metals_axis_research.axis_timeframe_smoke import load_case  # noqa: E402
from precious_metals_axis_research.cost_aware_xaueur_m5 import (  # noqa: E402
    get_symbol_point,
    simulate_cost_aware,
)
from precious_metals_axis_research.optimize_silver_regime_readiness import (  # noqa: E402
    simulate_filtered as simulate_silver_filtered,
)
from precious_metals_axis_research.readiness_silver_xaueur import (  # noqa: E402
    SILVER_FOLDS,
    XAUEUR_FOLDS,
    fold_pass,
    scale_spread,
)
from precious_metals_axis_research.walk_forward_all_metals_shared import (  # noqa: E402
    load_frames,
    slice_by_ratio,
)
from precious_metals_axis_research.walk_forward_long_tf_cost import compact_period  # noqa: E402


OUTPUT_CSV = RESEARCH_DIR / "training_profile_optimization_results.csv"
OUTPUT_JSON = RESEARCH_DIR / "training_profile_optimization_results.json"
OUTPUT_MD = RESEARCH_DIR / "training_profile_optimization_report.md"
OUTPUT_BEST = RESEARCH_DIR / "training_profile_optimization_best.json"

COST_MULTIPLIERS = [1.0, 2.0, 3.0, 4.0, 5.0]

MODEL_PROFILES = {
    "current_symbol": {
        "n_estimators": 170,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 80,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
    },
    "conservative_shallow": {
        "n_estimators": 180,
        "learning_rate": 0.035,
        "max_depth": 3,
        "min_child_weight": 140,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_lambda": 2.0,
        "reg_alpha": 0.10,
    },
    "balanced_regularized": {
        "n_estimators": 240,
        "learning_rate": 0.035,
        "max_depth": 4,
        "min_child_weight": 110,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 1.8,
        "reg_alpha": 0.05,
    },
    "smooth_more_trees": {
        "n_estimators": 320,
        "learning_rate": 0.025,
        "max_depth": 3,
        "min_child_weight": 100,
        "subsample": 0.88,
        "colsample_bytree": 0.88,
        "reg_lambda": 2.5,
        "reg_alpha": 0.05,
    },
}


def train_model(train_df: pd.DataFrame, features: list[str], profile: dict) -> xgb.XGBClassifier:
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
        random_state=42,
        verbosity=0,
        **profile,
    )
    model.fit(train_df[features], train_df["BARRIER_TARGET"], sample_weight=sample_weight)
    return model


def train_shared_model(train_frames: list[pd.DataFrame], features: list[str], profile: dict) -> xgb.XGBClassifier:
    return train_model(pd.concat(train_frames, ignore_index=True), features, profile)


def prepare_silver(profile_name: str, profile: dict) -> list[dict]:
    frame, features = load_case("SILVER#", "H1")
    point = get_symbol_point("SILVER#")
    prepared = []
    for fold in SILVER_FOLDS:
        train_df = slice_by_ratio(frame, *fold["train"])
        test_df = slice_by_ratio(frame, *fold["test"])
        print(f"{profile_name} SILVER {fold['name']}: train={len(train_df):,} test={len(test_df):,}")
        model = train_model(train_df, features, profile)
        probs = model.predict_proba(test_df[features]).astype(np.float32)
        prepared.append(
            {
                "fold": fold["name"],
                "test_period": compact_period(test_df),
                "test_df": test_df,
                "probs": probs,
                "point": point,
            }
        )
    return prepared


def prepare_xaueur(profile_name: str, profile: dict) -> list[dict]:
    frames, features = load_frames()
    point = get_symbol_point("XAUEUR#")
    prepared = []
    for fold in XAUEUR_FOLDS:
        train_frames = [slice_by_ratio(frame, *fold["train"]) for frame in frames.values()]
        test_df = slice_by_ratio(frames["XAUEUR#"], *fold["test"])
        print(
            f"{profile_name} XAUEUR {fold['name']}: "
            f"train_shared={sum(len(df) for df in train_frames):,} test={len(test_df):,}"
        )
        model = train_shared_model(train_frames, features, profile)
        probs = model.predict_proba(test_df[features]).astype(np.float32)
        prepared.append(
            {
                "fold": fold["name"],
                "test_period": compact_period(test_df),
                "test_df": test_df,
                "probs": probs,
                "point": point,
            }
        )
    return prepared


def aggregate(symbol: str, fold_rows: list[dict]) -> dict:
    total_trades = sum(row["trades"] for row in fold_rows)
    total_r = sum(row["pnl_r"] for row in fold_rows)
    weighted_win = (
        sum(row["win_rate"] * row["trades"] for row in fold_rows) / total_trades
        if total_trades
        else 0.0
    )
    positive = sum(row["pnl_r"] > 0 for row in fold_rows)
    passed = sum(row["fold_pass"] for row in fold_rows)
    mean_pf = sum(row["profit_factor"] for row in fold_rows) / len(fold_rows)
    worst = min(row["pnl_r"] for row in fold_rows)
    max_dd = min(row["max_drawdown_r"] for row in fold_rows)
    recent = fold_rows[-1]
    if symbol == "SILVER#":
        gate = (
            positive == 5
            and passed >= 5
            and total_r >= 12.0
            and total_trades >= 60
            and recent["pnl_r"] > 0
            and recent["profit_factor"] >= 1.15
            and max_dd >= -10.0
        )
    else:
        gate = (
            positive == 3
            and passed >= 3
            and total_r >= 6.0
            and total_trades >= 24
            and recent["pnl_r"] > 0
            and recent["profit_factor"] >= 1.15
            and max_dd >= -6.0
        )
    score = (
        total_r * 180.0
        + positive * 800.0
        + passed * 650.0
        + weighted_win * 700.0
        + min(mean_pf, 4.0) * 240.0
        - abs(max_dd) * 45.0
        + min(worst, 0.0) * 240.0
    )
    return {
        "total_r": round(total_r, 4),
        "trades": total_trades,
        "positive_folds": positive,
        "passed_folds": passed,
        "weighted_win_rate": round(weighted_win, 4),
        "mean_profit_factor": round(mean_pf, 4),
        "worst_fold_r": round(worst, 4),
        "max_drawdown_r": round(max_dd, 4),
        "recent_paper_r": recent["pnl_r"],
        "gate": gate,
        "score": round(score, 4),
        "folds": fold_rows,
    }


def evaluate_silver(params: dict, prepared: list[dict], cost_multiplier: float) -> dict:
    fold_rows = []
    for fold in prepared:
        stats = simulate_silver_filtered(
            scale_spread(fold["test_df"], cost_multiplier),
            fold["probs"],
            params,
            fold["point"],
        )
        fold_rows.append(
            {
                "fold": fold["fold"],
                "test_start": fold["test_period"]["start"],
                "test_end": fold["test_period"]["end"],
                "pnl_r": stats["pnl_r"],
                "trades": stats["trades"],
                "win_rate": stats["win_rate"],
                "profit_factor": stats["profit_factor"],
                "max_drawdown_r": stats["max_drawdown_r"],
                "avg_r": stats["avg_r"],
                "max_loss_streak": stats["max_loss_streak"],
                "fold_pass": fold_pass(stats),
            }
        )
    return {
        "symbol": "SILVER#",
        **params,
        "cost_multiplier": cost_multiplier,
        **aggregate("SILVER#", fold_rows),
    }


def evaluate_xaueur(params: dict, prepared: list[dict], cost_multiplier: float) -> dict:
    fold_rows = []
    for fold in prepared:
        stats = simulate_cost_aware(
            scale_spread(fold["test_df"], cost_multiplier),
            fold["probs"],
            params,
            fold["point"],
        )
        fold_rows.append(
            {
                "fold": fold["fold"],
                "test_start": fold["test_period"]["start"],
                "test_end": fold["test_period"]["end"],
                "pnl_r": stats["pnl_r"],
                "trades": stats["trades"],
                "win_rate": stats["win_rate"],
                "profit_factor": stats["profit_factor"],
                "max_drawdown_r": stats["max_drawdown_r"],
                "avg_r": stats["avg_r"],
                "max_loss_streak": stats["max_loss_streak"],
                "fold_pass": fold_pass(stats),
            }
        )
    return {
        "symbol": "XAUEUR#",
        **params,
        "cost_multiplier": cost_multiplier,
        **aggregate("XAUEUR#", fold_rows),
    }


def silver_grid():
    base = {
        "direction_mode": "long",
        "trend_min": 0.0,
        "rsi_min": 0.0,
        "rsi_max": 100.0,
        "spread_atr_max": 0.75,
        "macd_min": -999.0,
    }
    for threshold, edge, tp, sl, hold, vola in product(
        [0.52, 0.54, 0.56, 0.58],
        [0.0, 0.05],
        [4.4, 5.2, 6.0],
        [5.2, 6.0, 7.0],
        [216, 336],
        [1.0, 1.2, 1.4],
    ):
        if tp > sl:
            continue
        yield {
            **base,
            "threshold": threshold,
            "edge_threshold": edge,
            "tp_atr": tp,
            "sl_atr": sl,
            "max_hold": hold,
            "vola_max": vola,
        }


def xaueur_grid():
    for threshold, edge, tp, sl, hold, direction in product(
        [0.54, 0.56, 0.58, 0.60],
        [0.0, 0.02],
        [2.2, 2.6, 3.0],
        [4.2, 4.8, 5.4],
        [168, 216, 288],
        ["both", "short"],
    ):
        if tp > sl:
            continue
        yield {
            "threshold": threshold,
            "edge_threshold": edge,
            "tp_atr": tp,
            "sl_atr": sl,
            "max_hold": hold,
            "direction_mode": direction,
        }


def select_best(results: list[dict]) -> dict:
    return sorted(
        results,
        key=lambda row: (row["gate"], row["score"], row["total_r"], row["trades"]),
        reverse=True,
    )[0]


def cost_stress(symbol: str, params: dict, prepared: list[dict]) -> list[dict]:
    rows = []
    for cost in COST_MULTIPLIERS:
        if symbol == "SILVER#":
            rows.append(evaluate_silver(params, prepared, cost))
        else:
            rows.append(evaluate_xaueur(params, prepared, cost))
    return rows


def flat_row(profile_name: str, group: str, row: dict) -> dict:
    return {
        key: value
        for key, value in {
            "profile": profile_name,
            "group": group,
            **row,
        }.items()
        if key != "folds"
    }


def write_outputs(all_rows: list[dict], best: dict) -> None:
    flat_rows = [
        flat_row(row["profile"], row["group"], row["result"])
        for row in all_rows
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    OUTPUT_JSON.write_text(
        json.dumps({"rows": all_rows, "best": best}, indent=2),
        encoding="utf-8",
    )
    OUTPUT_BEST.write_text(json.dumps(best, indent=2), encoding="utf-8")

    lines = [
        "# Training Profile Optimization",
        "",
        "Compares XGBoost training profiles and then retunes execution parameters at 3x spread cost.",
        "",
        "## Best By Symbol",
        "",
        "| Symbol | Profile | Gate | R | Trades | Win | PF | Worst R | DD | Params |",
        "|---|---|:---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for symbol in ["SILVER#", "XAUEUR#"]:
        row = best[symbol]["best_3x"]
        profile = best[symbol]["profile"]
        lines.append(
            "| {symbol} | {profile} | {gate} | {total_r:.2f} | {trades} | "
            "{weighted_win_rate:.2%} | {mean_profit_factor:.2f} | "
            "{worst_fold_r:.2f} | {max_drawdown_r:.2f} | "
            "conf={threshold}, edge={edge_threshold}, tp/sl={tp_atr}/{sl_atr}, "
            "hold={max_hold}, dir={direction_mode} |".format(
                profile=profile, **row
            )
        )

    lines.extend(
        [
            "",
            "## Cost Stress Of Selected Profiles",
            "",
            "| Symbol | Profile | Cost | Gate | R | Positive | Passed | Trades | Win | Worst R | DD |",
            "|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for symbol in ["SILVER#", "XAUEUR#"]:
        profile = best[symbol]["profile"]
        for row in best[symbol]["cost_stress"]:
            fold_count = 5 if symbol == "SILVER#" else 3
            lines.append(
                "| {symbol} | {profile} | {cost_multiplier:.1f}x | {gate} | "
                "{total_r:.2f} | {positive_folds}/{fold_count} | {passed_folds}/{fold_count} | "
                "{trades} | {weighted_win_rate:.2%} | {worst_fold_r:.2f} | "
                "{max_drawdown_r:.2f} |".format(
                    profile=profile,
                    fold_count=fold_count,
                    **row,
                )
            )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    all_rows = []
    profile_best = {"SILVER#": [], "XAUEUR#": []}
    for profile_name, profile in MODEL_PROFILES.items():
        print(f"=== Training profile: {profile_name} ===")
        silver_prepared = prepare_silver(profile_name, profile)
        silver_results = [evaluate_silver(params, silver_prepared, 3.0) for params in silver_grid()]
        silver_best = select_best(silver_results)
        silver_cost = cost_stress("SILVER#", silver_best, silver_prepared)
        profile_best["SILVER#"].append(
            {"profile": profile_name, "best_3x": silver_best, "cost_stress": silver_cost}
        )
        all_rows.extend(
            {"profile": profile_name, "group": "silver_grid_3x", "result": row}
            for row in silver_results
        )
        all_rows.extend(
            {"profile": profile_name, "group": "silver_selected_cost", "result": row}
            for row in silver_cost
        )

        xaueur_prepared = prepare_xaueur(profile_name, profile)
        xaueur_results = [evaluate_xaueur(params, xaueur_prepared, 3.0) for params in xaueur_grid()]
        xaueur_best = select_best(xaueur_results)
        xaueur_cost = cost_stress("XAUEUR#", xaueur_best, xaueur_prepared)
        profile_best["XAUEUR#"].append(
            {"profile": profile_name, "best_3x": xaueur_best, "cost_stress": xaueur_cost}
        )
        all_rows.extend(
            {"profile": profile_name, "group": "xaueur_grid_3x", "result": row}
            for row in xaueur_results
        )
        all_rows.extend(
            {"profile": profile_name, "group": "xaueur_selected_cost", "result": row}
            for row in xaueur_cost
        )

    best = {}
    for symbol, rows in profile_best.items():
        best[symbol] = sorted(
            rows,
            key=lambda item: (
                sum(row["gate"] for row in item["cost_stress"]),
                item["best_3x"]["gate"],
                item["best_3x"]["total_r"],
                item["best_3x"]["score"],
            ),
            reverse=True,
        )[0]
    write_outputs(all_rows, best)
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_BEST}")
    for symbol, item in best.items():
        row = item["best_3x"]
        print(
            f"{symbol}: profile={item['profile']} R={row['total_r']:.2f} "
            f"trades={row['trades']} gate={row['gate']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
