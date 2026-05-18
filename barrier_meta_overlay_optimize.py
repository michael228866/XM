import os
from itertools import product

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

from barrier_final_train import FINAL_PARAMS
from barrier_meta_overlay import (
    evaluate_df,
    load_final_model,
    load_meta_overlay_model,
    overlay_params,
    predict_overlay_risk_mult,
    quality_risk_mult,
    split_overlay_data,
    trades_per_year,
)
from drl_train_candidate import format_stats


def score_candidate(stats, baseline, df):
    tpy = trades_per_year(stats, df)
    base_tpy = trades_per_year(baseline, df)
    dd = abs(min(stats["max_drawdown_pct"], 0.0))
    base_dd = abs(min(baseline["max_drawdown_pct"], 0.0))
    pnl_gain = stats["pnl"] - baseline["pnl"]
    pf_gain = stats["profit_factor"] - baseline["profit_factor"]

    if stats["stopped_out"]:
        return -1_000_000.0
    if tpy < base_tpy * 0.90:
        return -200_000.0 + pnl_gain
    if stats["max_consecutive_losses"] > baseline["max_consecutive_losses"] + 1:
        return -150_000.0 + pnl_gain
    if dd > 0.50:
        return -120_000.0 + pnl_gain

    dd_penalty = max(0.0, dd - max(0.32, base_dd * 1.25)) * 12_000.0
    pf_penalty = max(0.0, baseline["profit_factor"] * 0.98 - stats["profit_factor"]) * 8_000.0
    return pnl_gain + pf_gain * 500.0 + min(tpy, base_tpy * 1.08) * 5.0 - dd_penalty - pf_penalty


def practical_grade(stats, baseline, df):
    tpy = trades_per_year(stats, df)
    dd = abs(min(stats["max_drawdown_pct"], 0.0))
    pnl_gain = stats["pnl"] - baseline["pnl"]
    pf_gain = stats["profit_factor"] - baseline["profit_factor"]
    tpy_ratio = tpy / max(trades_per_year(baseline, df), 1e-9)

    grade = 8.5
    grade += min(max(pnl_gain / 900.0, 0.0), 1.2)
    grade += min(max(pf_gain / 0.12, 0.0), 0.6)
    grade += min(max((0.50 - dd) / 0.25, 0.0), 0.5)
    grade += min(max(tpy_ratio - 0.95, 0.0), 0.2)
    if stats["max_consecutive_losses"] <= baseline["max_consecutive_losses"]:
        grade += 0.3
    if stats["stopped_out"]:
        grade -= 3.0
    return min(10.0, grade)


def make_grid():
    for risk, protect_cut, boost_cut, strong_cut, protect_mult, boost_mult, strong_mult in product(
        [0.028, 0.030, 0.032],
        [0.40, 0.46],
        [0.56, 0.60],
        [0.72, 0.78],
        [0.90, 1.00],
        [1.25, 1.35, 1.45],
        [1.35, 1.50, 1.65],
    ):
        if not (protect_cut < boost_cut < strong_cut):
            continue
        if strong_mult < boost_mult:
            continue
        yield risk, (protect_cut, boost_cut, strong_cut, protect_mult, boost_mult, strong_mult)


def print_stats(prefix, stats, df):
    print(f"   trades/year={trades_per_year(stats, df):.1f} | " + format_stats(prefix, stats))


def main():
    print("Optimizing saved meta-regime overlay rules...")
    _, features, regime_features, _, overlay_train, test_df = split_overlay_data()
    main_model = load_final_model()
    meta_model, config = load_meta_overlay_model()
    configured_features = config["regime_features"]

    test_probs = main_model.predict_proba(test_df[features]).astype("float32")
    baseline_test = evaluate_df(FINAL_PARAMS, test_df, test_probs)
    _, test_quality = predict_overlay_risk_mult(
        meta_model,
        test_df,
        test_probs,
        configured_features,
        tuple(float(value) for value in config["risk_rule"]),
    )

    # Validation side uses the saved meta model against the overlay-training window.
    overlay_probs = main_model.predict_proba(overlay_train[features]).astype("float32")
    baseline_overlay = evaluate_df(FINAL_PARAMS, overlay_train, overlay_probs)
    _, overlay_quality = predict_overlay_risk_mult(
        meta_model,
        overlay_train,
        overlay_probs,
        configured_features,
        tuple(float(value) for value in config["risk_rule"]),
    )

    print("Baselines:")
    print_stats("overlay baseline", baseline_overlay, overlay_train)
    print_stats("test baseline", baseline_test, test_df)

    results = []
    for risk, rule in make_grid():
        params = overlay_params()
        params["risk_per_trade"] = risk

        overlay_mult = quality_risk_mult(overlay_quality, *rule)
        overlay_stats = evaluate_df(params, overlay_train, overlay_probs, overlay_mult)
        overlay_score = score_candidate(overlay_stats, baseline_overlay, overlay_train)
        if overlay_score <= -100_000.0:
            continue

        test_mult = quality_risk_mult(test_quality, *rule)
        test_stats = evaluate_df(params, test_df, test_probs, test_mult)
        test_score = score_candidate(test_stats, baseline_test, test_df)

        if overlay_stats["pnl"] <= baseline_overlay["pnl"]:
            continue
        if test_stats["pnl"] <= baseline_test["pnl"]:
            continue
        if test_stats["profit_factor"] < baseline_test["profit_factor"]:
            continue
        if test_stats["max_consecutive_losses"] > baseline_test["max_consecutive_losses"]:
            continue
        if abs(min(test_stats["max_drawdown_pct"], 0.0)) > 0.50:
            continue

        grade = practical_grade(test_stats, baseline_test, test_df)
        results.append((grade, overlay_score, test_score, risk, rule, overlay_stats, test_stats))

    results.sort(key=lambda item: (item[0], item[2], item[1]), reverse=True)
    print("Top practical candidates:")
    for rank, (grade, overlay_score, test_score, risk, rule, overlay_stats, test_stats) in enumerate(results[:20], start=1):
        print(
            f"#{rank} grade={grade:.2f} overlay_score={overlay_score:.2f} "
            f"test_score={test_score:.2f} risk={risk} rule={rule}"
        )
        print_stats("overlay", overlay_stats, overlay_train)
        print_stats("test", test_stats, test_df)


if __name__ == "__main__":
    main()
