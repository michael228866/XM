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


def print_period_and_stats(name, stats, df):
    start_time = df["TIME_DT"].iloc[0]
    end_time = df["TIME_DT"].iloc[-1]
    elapsed_days = max((end_time - start_time).total_seconds() / 86400.0, 1e-9)
    print(
        f"{name} period: {start_time} -> {end_time} "
        f"({elapsed_days:.1f} days, {stats['trades'] / elapsed_days:.2f} trades/day, "
        f"{trades_per_year(stats, df):.1f} trades/year)"
    )
    print(
        f"{name} capital: {stats['initial_balance']:.2f} -> {stats['balance']:.2f} "
        f"| stopped_out={stats['stopped_out']}"
    )
    print(format_stats(name, stats))


def main():
    print("Starting final barrier + meta-regime overlay backtest...")
    _, features, regime_features, _, _, test_df = split_overlay_data()

    main_model = load_final_model()
    meta_model, config = load_meta_overlay_model()
    configured_features = config["regime_features"]
    missing_features = [
        feature for feature in configured_features if feature not in regime_features
    ]
    if missing_features:
        raise ValueError(f"Configured regime features unavailable: {missing_features}")

    probs = main_model.predict_proba(test_df[features]).astype("float32")
    baseline_stats = evaluate_df(FINAL_PARAMS, test_df, probs)

    rule = tuple(float(value) for value in config["risk_rule"])
    params = overlay_params()
    params["risk_per_trade"] = float(config["risk_per_trade"])
    risk_mult, quality = predict_overlay_risk_mult(
        meta_model,
        test_df,
        probs,
        configured_features,
        rule,
    )
    overlay_stats = evaluate_df(params, test_df, probs, risk_mult)

    print(f"Loaded meta config version={config['version']} rule={rule}")
    print(
        f"Meta quality: min={quality.min():.4f}, "
        f"mean={quality.mean():.4f}, max={quality.max():.4f}"
    )
    print_period_and_stats("Formal baseline", baseline_stats, test_df)
    print_period_and_stats("Meta overlay", overlay_stats, test_df)


if __name__ == "__main__":
    main()
