import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

from barrier_meta_overlay import (
    META_CONFIG_PATH,
    META_MODEL_PATH,
    RECOMMENDED_RISK_PER_TRADE,
    RECOMMENDED_RULE,
    evaluate_df,
    overlay_params,
    predict_overlay_risk_mult,
    save_meta_overlay_model,
    split_overlay_data,
    trades_per_year,
    train_meta_overlay_model,
)
from drl_train_candidate import format_stats


def print_stats(prefix, stats, df):
    print(f"   trades/year={trades_per_year(stats, df):.1f} | " + format_stats(prefix, stats))


def main():
    print("Training and saving meta-regime overlay model...")
    _, features, regime_features, base_train, overlay_train, _ = split_overlay_data()
    print(
        f"Rows | base_train={len(base_train):,} "
        f"overlay_train={len(overlay_train):,}"
    )

    meta_model, overlay_probs = train_meta_overlay_model(
        base_train,
        overlay_train,
        features,
        regime_features,
    )
    config = save_meta_overlay_model(meta_model, regime_features)

    params = overlay_params()
    risk_mult, _ = predict_overlay_risk_mult(
        meta_model,
        overlay_train,
        overlay_probs,
        regime_features,
        RECOMMENDED_RULE,
    )
    stats = evaluate_df(params, overlay_train, overlay_probs, risk_mult)

    print(
        f"Saved meta model: {META_MODEL_PATH} | config: {META_CONFIG_PATH} | "
        f"risk_per_trade={RECOMMENDED_RISK_PER_TRADE} | rule={RECOMMENDED_RULE}"
    )
    print(f"Config version: {config['version']}")
    print_stats("overlay_train saved_meta", stats, overlay_train)


if __name__ == "__main__":
    main()
