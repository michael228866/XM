import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import xgboost as xgb

from barrier_classifier_strategy import evaluate
from barrier_final_train import FINAL_PARAMS, MODEL_PATH, TRAIN_END_RATIO, prepare_barrier_data
from drl_train_candidate import format_stats


STRESS_EXTRA_COST_POINTS = [0.0, 5.0, 10.0, 20.0]
MIN_TRADES_PER_YEAR = 100.0
MIN_ACTIVE_MONTH_RATIO = 0.70
ACCEPTED_DRAWDOWN_PCT = 0.50


def evaluate_df(params, df, probs):
    return evaluate(
        params,
        df["CLOSE"].to_numpy(dtype=np.float64),
        df["ATR"].to_numpy(dtype=np.float64),
        probs,
        hours=df["TIME_DT"].dt.hour.to_numpy(dtype=np.int16),
        weekdays=df["TIME_DT"].dt.dayofweek.to_numpy(dtype=np.int8),
        dates=df["TIME_DT"].dt.date.to_numpy(),
        rsi_values=df["M1_RSI"].to_numpy(dtype=np.float64),
    )


def activity_metrics(params, test_df, test_probs):
    start_time = test_df["TIME_DT"].iloc[0]
    end_time = test_df["TIME_DT"].iloc[-1]
    elapsed_years = max((end_time - start_time).total_seconds() / (365.25 * 86400.0), 1e-9)
    active_months = 0
    total_months = 0
    profitable_months = 0

    for _, month_df in test_df.groupby(test_df["TIME_DT"].dt.to_period("M")):
        total_months += 1
        idx = month_df.index.to_numpy()
        month_stats = evaluate_df(params, month_df.reset_index(drop=True), test_probs[idx])
        if month_stats["trades"] > 0:
            active_months += 1
        if month_stats["pnl"] > 0:
            profitable_months += 1

    return {
        "elapsed_years": elapsed_years,
        "trades_per_year": 0.0,
        "active_month_ratio": active_months / total_months if total_months else 0.0,
        "profitable_month_ratio": profitable_months / total_months if total_months else 0.0,
    }


def risk_grade(stats, activity):
    trades_per_year = stats["trades"] / activity["elapsed_years"]
    activity["trades_per_year"] = trades_per_year
    if (
        stats["stopped_out"]
        or stats["pnl"] <= 0
        or trades_per_year < MIN_TRADES_PER_YEAR
        or activity["active_month_ratio"] < MIN_ACTIVE_MONTH_RATIO
    ):
        return 0.0
    score = 5.0
    score += min(stats["roi"], 1.0) * 1.0
    score += min(stats["profit_factor"] / 2.0, 1.0) * 1.0
    score += min(stats["win_rate"] / 0.65, 1.0) * 0.8
    score += max(0.0, 1.0 - abs(stats["max_drawdown_pct"]) / ACCEPTED_DRAWDOWN_PCT) * 1.0
    score += max(0.0, 1.0 - stats["max_consecutive_losses"] / 8.0) * 0.7
    score += max(0.0, min(stats["sharpe_like"] / 3.0, 1.0)) * 0.5
    score += min((trades_per_year - MIN_TRADES_PER_YEAR) / 80.0, 1.0) * 0.5
    score += activity["active_month_ratio"] * 0.3
    score += activity["profitable_month_ratio"] * 0.2
    drawdown_pct = abs(stats["max_drawdown_pct"])
    if drawdown_pct > 0.70:
        score = min(score, 6.5)
    elif drawdown_pct > 0.60:
        score = min(score, 7.2)
    elif drawdown_pct > 0.50:
        score = min(score, 7.5)
    return min(score, 10.0)


def make_variants():
    base = dict(FINAL_PARAMS)
    variants = {
        "current": {},
        "aggressive_previous": {
            "tp_atr": 1.1,
            "risk_per_trade": 0.028,
            "max_daily_loss_pct": 0.05,
            "max_daily_trades": None,
        },
        "robust_1_8pct": {
            "tp_atr": 1.1,
            "risk_per_trade": 0.018,
            "max_daily_loss_pct": 0.04,
            "max_daily_trades": 2,
        },
        "risk_1_6pct": {
            "risk_per_trade": 0.016,
            "max_daily_loss_pct": 0.035,
            "max_daily_trades": 1,
        },
        "risk_1_8pct_max2": {
            "risk_per_trade": 0.018,
            "max_daily_loss_pct": 0.04,
            "max_daily_trades": 2,
        },
        "risk_1_4pct": {
            "risk_per_trade": 0.014,
            "max_daily_loss_pct": 0.03,
            "max_daily_trades": 1,
        },
        "high_conf_1_6pct": {
            "threshold": 0.545,
            "risk_per_trade": 0.016,
            "max_daily_loss_pct": 0.035,
            "max_daily_trades": 1,
        },
        "high_conf_1_4pct": {
            "threshold": 0.545,
            "risk_per_trade": 0.014,
            "max_daily_loss_pct": 0.03,
            "max_daily_trades": 1,
        },
        "very_high_conf_1_4pct": {
            "threshold": 0.555,
            "risk_per_trade": 0.014,
            "max_daily_loss_pct": 0.03,
            "max_daily_trades": 1,
        },
    }
    for name, overrides in variants.items():
        params = dict(base)
        params.update(overrides)
        yield name, params


def print_yearly_breakdown(params, test_df, test_probs):
    print("Yearly breakdown:")
    for year, year_df in test_df.groupby(test_df["TIME_DT"].dt.year):
        idx = year_df.index.to_numpy()
        stats = evaluate_df(params, year_df.reset_index(drop=True), test_probs[idx])
        print("   " + format_stats(str(year), stats))


def print_cost_stress(params, test_df, test_probs):
    print("Cost stress:")
    for extra_cost in STRESS_EXTRA_COST_POINTS:
        stressed = dict(params)
        stressed["extra_cost_points"] = extra_cost
        stats = evaluate_df(stressed, test_df, test_probs)
        activity = activity_metrics(stressed, test_df, test_probs)
        print(
            f"   extra_cost={extra_cost:.0f} points | "
            f"grade={risk_grade(stats, activity):.2f} | "
            f"trades/year={activity['trades_per_year']:.1f} | "
            + format_stats("test", stats)
        )


def main():
    print("Loading final barrier model for robust report...")
    df, features = prepare_barrier_data()
    train_end = int(len(df) * TRAIN_END_RATIO)
    test_df = df.iloc[train_end:].copy().reset_index(drop=True)

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    model.set_params(device="cpu")
    test_probs = model.predict_proba(test_df[features]).astype(np.float32)

    ranked = []
    for name, params in make_variants():
        stats = evaluate_df(params, test_df, test_probs)
        activity = activity_metrics(params, test_df, test_probs)
        grade = risk_grade(stats, activity)
        ranked.append((grade, name, params, stats, activity))

    ranked.sort(key=lambda item: item[0], reverse=True)
    print("Variant ranking:")
    for grade, name, _, stats, activity in ranked:
        print(
            f"   grade={grade:.2f} | {name} | "
            f"trades/year={activity['trades_per_year']:.1f}, "
            f"active_months={activity['active_month_ratio']:.1%}, "
            f"profitable_months={activity['profitable_month_ratio']:.1%} | "
            + format_stats("test", stats)
        )

    current = next(item for item in ranked if item[1] == "current")
    _, _, current_params, _, _ = current
    print("Current final params evidence:")
    print_cost_stress(current_params, test_df, test_probs)
    print_yearly_breakdown(current_params, test_df, test_probs)

    best_grade, best_name, best_params, _, _ = ranked[0]
    print(f"Selected robust variant: {best_name} grade={best_grade:.2f}")
    print_cost_stress(best_params, test_df, test_probs)
    print_yearly_breakdown(best_params, test_df, test_probs)
    print(f"Selected params: {best_params}")


if __name__ == "__main__":
    main()
