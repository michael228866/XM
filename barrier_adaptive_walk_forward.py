import os
from dataclasses import dataclass

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import torch
import xgboost as xgb

from barrier_classifier_strategy import HORIZON, evaluate
from barrier_final_train import FINAL_PARAMS, prepare_barrier_data
from barrier_classifier_strategy import build_profit_sample_weight
from drl_train_candidate import format_stats


N_ESTIMATORS = 260
MIN_VALIDATION_TRADES = 20


@dataclass
class Candidate:
    name: str
    params: dict


def train_classifier(train_df, features):
    target = train_df["BARRIER_TARGET"].to_numpy(dtype=np.int8)
    sample_weight = build_profit_sample_weight(train_df, target)
    sample_weight = np.nan_to_num(sample_weight, nan=1.0, posinf=1.0, neginf=1.0)
    sample_weight = np.maximum(sample_weight, 1e-6)
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        device="cuda" if torch.cuda.is_available() else "cpu",
        n_estimators=N_ESTIMATORS,
        learning_rate=0.045,
        max_depth=5,
        min_child_weight=80,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbosity=0,
    )
    model.fit(train_df[features], train_df["BARRIER_TARGET"], sample_weight=sample_weight)
    model.set_params(device="cpu")
    return model


def with_overrides(name, **overrides):
    params = dict(FINAL_PARAMS)
    params.update(overrides)
    return Candidate(name, params)


def candidates():
    return [
        with_overrides("current_dynamic"),
        with_overrides(
            "current_risk_040",
            risk_per_trade=0.040,
        ),
        with_overrides(
            "quality_month_filter",
            risk_per_trade=0.040,
            allowed_entry_months=[1, 2, 3, 4, 5, 6, 9, 10, 11, 12],
        ),
        with_overrides(
            "aggressive_previous_guarded",
            tp_atr=1.1,
            risk_per_trade=0.028,
            max_daily_loss_pct=0.05,
            max_daily_trades=None,
        ),
        with_overrides(
            "robust_low_risk",
            tp_atr=1.1,
            risk_per_trade=0.018,
            max_daily_loss_pct=0.04,
            max_daily_trades=2,
        ),
        with_overrides(
            "high_conf",
            threshold=0.545,
            tp_atr=1.1,
            risk_per_trade=0.028,
            max_daily_loss_pct=0.05,
        ),
        with_overrides(
            "short_probe",
            direction_mode="short",
            threshold=0.545,
            tp_atr=1.0,
            sl_atr=2.0,
            risk_per_trade=0.018,
            allowed_entry_weekdays=[0, 1, 2, 3, 4],
            excluded_rsi_ranges=[],
            max_daily_loss_pct=0.04,
            max_daily_trades=2,
        ),
        with_overrides(
            "cash_filter",
            threshold=2.0,
            risk_per_trade=0.0,
            max_daily_trades=0,
        ),
    ]


def evaluate_df(params, df, probs):
    return evaluate(
        params,
        df["CLOSE"].to_numpy(dtype=np.float64),
        df["ATR"].to_numpy(dtype=np.float64),
        probs,
        hours=df["TIME_DT"].dt.hour.to_numpy(dtype=np.int16),
        weekdays=df["TIME_DT"].dt.dayofweek.to_numpy(dtype=np.int8),
        dates=df["TIME_DT"].dt.date.to_numpy(),
        months=df["TIME_DT"].dt.month.to_numpy(dtype=np.int8),
        rsi_values=df["M1_RSI"].to_numpy(dtype=np.float64),
        vola_ratio_values=df["VOLA_RATIO"].to_numpy(dtype=np.float64),
    )


def period_years(df):
    elapsed = (df["TIME_DT"].iloc[-1] - df["TIME_DT"].iloc[0]).total_seconds() / (365.25 * 86400.0)
    return max(elapsed, 1e-9)


def trades_per_year(stats, df):
    return stats["trades"] / period_years(df)


def selection_score(stats, df):
    tpy = trades_per_year(stats, df)
    if stats["trades"] == 0 and stats["pnl"] == 0:
        return -50.0
    if stats["stopped_out"] or stats["pnl"] <= 0:
        return -1_000_000.0 + stats["pnl"]
    dd = abs(min(stats["max_drawdown_pct"], 0.0))
    dd_penalty = max(0.0, dd - 0.35) * 12_000.0
    pf_penalty = max(0.0, 1.35 - stats["profit_factor"]) * 6_000.0
    low_trade_penalty = max(0.0, MIN_VALIDATION_TRADES - stats["trades"]) * 80.0
    return stats["pnl"] - dd_penalty - pf_penalty - low_trade_penalty + min(tpy, 180.0) * 3.0


def print_candidate(prefix, name, stats, df):
    print(f"   {prefix} {name} trades/year={trades_per_year(stats, df):.1f} | " + format_stats(name, stats))


def main():
    print("Adaptive walk-forward using current barrier training style...")
    df, features = prepare_barrier_data()
    years = sorted(int(year) for year in df["TIME_DT"].dt.year.unique())
    test_years = [year for year in years if 2021 <= year <= years[-1]]
    results = []

    for test_year in test_years:
        val_year = test_year - 1
        train_df = df[df["TIME_DT"].dt.year < val_year].copy()
        val_df = df[df["TIME_DT"].dt.year == val_year].copy().reset_index(drop=True)
        test_df = df[df["TIME_DT"].dt.year == test_year].copy().reset_index(drop=True)
        if len(train_df) <= HORIZON or len(val_df) == 0 or len(test_df) == 0:
            continue

        train_df = train_df.iloc[:-HORIZON].copy()
        print(
            f"Fold test={test_year} | train<={val_year - 1} rows={len(train_df):,} "
            f"validation={val_year} rows={len(val_df):,} test rows={len(test_df):,}"
        )
        model = train_classifier(train_df, features)
        val_probs = model.predict_proba(val_df[features]).astype(np.float32)
        test_probs = model.predict_proba(test_df[features]).astype(np.float32)

        evaluated = []
        for candidate in candidates():
            val_stats = evaluate_df(candidate.params, val_df, val_probs)
            score = selection_score(val_stats, val_df)
            evaluated.append((score, candidate, val_stats))
        evaluated.sort(key=lambda item: item[0], reverse=True)

        for score, candidate, val_stats in evaluated[:3]:
            print_candidate(f"validation score={score:.2f}", candidate.name, val_stats, val_df)

        _, selected, selected_val_stats = evaluated[0]
        test_stats = evaluate_df(selected.params, test_df, test_probs)
        print(f"Selected for {test_year}: {selected.name}")
        print_candidate("selected validation", selected.name, selected_val_stats, val_df)
        print_candidate("test", selected.name, test_stats, test_df)
        results.append((test_year, selected.name, selected_val_stats, test_stats, test_df))

    print("Adaptive walk-forward summary:")
    if not results:
        print("   No completed folds.")
        return
    profitable = sum(1 for _, _, _, stats, _ in results if stats["pnl"] > 0)
    avg_roi = float(np.mean([stats["roi"] for _, _, _, stats, _ in results]))
    worst_dd = float(np.min([stats["max_drawdown_pct"] for _, _, _, stats, _ in results]))
    avg_pf = float(
        np.mean(
            [
                stats["profit_factor"]
                for _, _, _, stats, _ in results
                if np.isfinite(stats["profit_factor"])
            ]
        )
    )
    avg_tpy = float(np.mean([trades_per_year(stats, test_df) for _, _, _, stats, test_df in results]))
    total_pnl = float(sum(stats["pnl"] for _, _, _, stats, _ in results))
    print(
        f"   profitable_folds={profitable}/{len(results)}, avg_roi={avg_roi:.2%}, "
        f"worst_dd={worst_dd:.2%}, avg_pf={avg_pf:.2f}, avg_trades_per_year={avg_tpy:.1f}, "
        f"sum_pnl={total_pnl:.2f}"
    )
    print("Selections:")
    for test_year, name, _, stats, test_df in results:
        print(
            f"   {test_year}: {name}, pnl={stats['pnl']:.2f}, "
            f"roi={stats['roi']:.2%}, dd={stats['max_drawdown_pct']:.2%}, "
            f"trades/year={trades_per_year(stats, test_df):.1f}"
        )


if __name__ == "__main__":
    main()
