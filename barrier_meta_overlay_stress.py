import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

from barrier_final_train import FINAL_PARAMS
from barrier_meta_overlay import (
    evaluate_df,
    load_final_model,
    load_meta_overlay_model,
    overlay_params,
    predict_overlay_risk_mult,
    split_overlay_data,
    trades_per_year,
)
from drl_train_candidate import format_stats


def print_stats(prefix, stats, df):
    print(f"   trades/year={trades_per_year(stats, df):.1f} | " + format_stats(prefix, stats))


def main():
    print("Running meta-regime overlay cost stress...")
    _, features, _, _, _, test_df = split_overlay_data()
    main_model = load_final_model()
    meta_model, config = load_meta_overlay_model()
    probs = main_model.predict_proba(test_df[features]).astype("float32")
    rule = tuple(float(value) for value in config["risk_rule"])
    risk_mult, _ = predict_overlay_risk_mult(
        meta_model,
        test_df,
        probs,
        config["regime_features"],
        rule,
    )

    for extra_cost in [5.0, 7.5, 10.0, 12.5]:
        baseline_params = dict(FINAL_PARAMS)
        overlay_params_ = overlay_params()
        baseline_params["extra_cost_points"] = extra_cost
        overlay_params_["extra_cost_points"] = extra_cost
        overlay_params_["risk_per_trade"] = float(config["risk_per_trade"])

        baseline_stats = evaluate_df(baseline_params, test_df, probs)
        overlay_stats = evaluate_df(overlay_params_, test_df, probs, risk_mult)
        print(f"Extra cost points={extra_cost}")
        print_stats("baseline", baseline_stats, test_df)
        print_stats("meta_overlay", overlay_stats, test_df)


if __name__ == "__main__":
    main()
