import json
import os

import numpy as np
import xgboost as xgb

from barrier_classifier_strategy import evaluate
from barrier_final_train import (
    FINAL_PARAMS,
    MODEL_PATH,
    MODEL_SELECTION_END_RATIO,
    TRAIN_END_RATIO,
    prepare_barrier_data,
)
from barrier_meta_regime_classifier import (
    BASE_THRESHOLD,
    HORIZON,
    make_meta_frame,
    predict_meta,
    train_base_model,
    train_meta_regime,
)


META_MODEL_PATH = "gold_meta_regime_xgb.json"
META_CONFIG_PATH = "gold_meta_regime_overlay.json"
META_CONFIG_VERSION = 1
RECOMMENDED_RISK_PER_TRADE = 0.028
RECOMMENDED_RULE = (0.40, 0.56, 0.72, 1.00, 1.45, 1.65)


def add_overlay_regime_features(df):
    df = df.copy()
    close = df["CLOSE"]
    high = df["HIGH"]
    low = df["LOW"]
    open_ = df["OPEN"]
    candle_range = (high - low).replace(0, np.nan)
    trend_cols = [col for col in df.columns if col.endswith("_TREND")]

    if trend_cols:
        df["REG_TREND_SCORE"] = (df[trend_cols] > 0).sum(axis=1).astype(np.float32)
    else:
        df["REG_TREND_SCORE"] = 0.0
    df["REG_RET_5"] = close.pct_change(5)
    df["REG_RET_15"] = close.pct_change(15)
    df["REG_RET_60"] = close.pct_change(60)
    df["REG_RANGE_ATR"] = (high - low) / (df["ATR"] + 1e-6)
    df["REG_BODY_SIGNED"] = (close - open_) / (candle_range + 1e-6)
    df["REG_RET_STD_30"] = close.pct_change().rolling(30).std()
    df["REG_RET_STD_120"] = close.pct_change().rolling(120).std()
    df["REG_VOLA_BURST"] = df["REG_RET_STD_30"] / (df["REG_RET_STD_120"] + 1e-9)
    df["REG_ATR_SLOPE_60"] = df["ATR"].pct_change(60)
    df["REG_RSI_SLOPE_10"] = df["M1_RSI"].diff(10)
    df["REG_HOUR"] = df["TIME_DT"].dt.hour.astype(np.float32) / 23.0
    df["REG_MONTH"] = df["TIME_DT"].dt.month.astype(np.float32) / 12.0
    df["REG_VOLA_RATIO"] = df["VOLA_RATIO"]
    df["REG_M1_RSI"] = df["M1_RSI"]
    df["REG_ATR"] = df["ATR"]
    df["REG_HOUR_SIN"] = df["HOUR_SIN"]
    df["REG_HOUR_COS"] = df["HOUR_COS"]
    df["REG_DAY_OF_WEEK"] = df["DAY_OF_WEEK"]

    regime_features = [
        "REG_TREND_SCORE",
        "REG_RET_5",
        "REG_RET_15",
        "REG_RET_60",
        "REG_RANGE_ATR",
        "REG_BODY_SIGNED",
        "REG_RET_STD_30",
        "REG_RET_STD_120",
        "REG_VOLA_BURST",
        "REG_ATR_SLOPE_60",
        "REG_RSI_SLOPE_10",
        "REG_HOUR",
        "REG_MONTH",
        "REG_VOLA_RATIO",
        "REG_M1_RSI",
        "REG_ATR",
        "REG_HOUR_SIN",
        "REG_HOUR_COS",
        "REG_DAY_OF_WEEK",
    ]
    df[regime_features] = df[regime_features].shift(1)
    return df, regime_features


def quality_risk_mult(
    quality,
    protect_cut,
    boost_cut,
    strong_cut,
    protect_mult,
    boost_mult,
    strong_mult,
):
    mult = np.ones(len(quality), dtype=np.float32)
    mult[quality < protect_cut] = protect_mult
    mult[quality >= boost_cut] = boost_mult
    mult[quality >= strong_cut] = strong_mult
    return mult


def overlay_params():
    params = dict(FINAL_PARAMS)
    params["risk_per_trade"] = RECOMMENDED_RISK_PER_TRADE
    return params


def evaluate_df(params, df, probs, entry_risk_mult=None):
    return evaluate(
        params,
        df["CLOSE"].to_numpy(dtype=np.float64),
        df["ATR"].to_numpy(dtype=np.float64),
        probs,
        hours=df["TIME_DT"].dt.hour.to_numpy(dtype=np.int16),
        weekdays=df["TIME_DT"].dt.dayofweek.to_numpy(dtype=np.int8),
        dates=df["TIME_DT"].dt.date.to_numpy(),
        rsi_values=df["M1_RSI"].to_numpy(dtype=np.float64),
        entry_risk_mult=entry_risk_mult,
    )


def trades_per_year(stats, df):
    elapsed_years = max(
        (df["TIME_DT"].iloc[-1] - df["TIME_DT"].iloc[0]).total_seconds()
        / (365.25 * 86400.0),
        1e-9,
    )
    return stats["trades"] / elapsed_years


def load_final_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Main model not found: {model_path}")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    model.set_params(device="cpu")
    return model


def split_overlay_data():
    df, features = prepare_barrier_data()
    df, regime_features = add_overlay_regime_features(df)
    base_end = int(len(df) * MODEL_SELECTION_END_RATIO)
    final_train_end = int(len(df) * TRAIN_END_RATIO)

    base_train = df.iloc[: max(0, base_end - HORIZON)].copy()
    overlay_train = df.iloc[base_end:final_train_end].copy()
    overlay_train = overlay_train.dropna(subset=regime_features).reset_index(drop=True)
    test_df = df.iloc[final_train_end:].copy().reset_index(drop=True)
    return df, features, regime_features, base_train, overlay_train, test_df


def train_meta_overlay_model(base_train, overlay_train, features, regime_features):
    validation_base = train_base_model(base_train, features)
    overlay_probs = validation_base.predict_proba(overlay_train[features]).astype(np.float32)
    overlay_meta_frame = make_meta_frame(overlay_train, overlay_probs, regime_features)
    candidate_mask = (
        (overlay_probs[:, 1] >= BASE_THRESHOLD)
        & (overlay_probs[:, 1] >= overlay_probs[:, 2])
    )
    model = train_meta_regime(
        overlay_meta_frame,
        overlay_train["BARRIER_TARGET"].to_numpy(dtype=np.int8),
        candidate_mask,
    )
    return model, overlay_probs


def save_meta_overlay_model(
    model,
    regime_features,
    model_path=META_MODEL_PATH,
    config_path=META_CONFIG_PATH,
):
    model.save_model(model_path)
    config = {
        "version": META_CONFIG_VERSION,
        "model_path": model_path,
        "main_model_path": MODEL_PATH,
        "risk_per_trade": RECOMMENDED_RISK_PER_TRADE,
        "risk_rule": list(RECOMMENDED_RULE),
        "regime_features": list(regime_features),
        "model_selection_end_ratio": MODEL_SELECTION_END_RATIO,
        "train_end_ratio": TRAIN_END_RATIO,
        "base_threshold": BASE_THRESHOLD,
    }
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    return config


def load_meta_overlay_model(
    model_path=META_MODEL_PATH,
    config_path=META_CONFIG_PATH,
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Meta-regime model not found: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Meta-regime config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    if int(config.get("version", -1)) != META_CONFIG_VERSION:
        raise ValueError(
            f"Unsupported meta config version: {config.get('version')}"
        )
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    model.set_params(device="cpu")
    return model, config


def predict_overlay_risk_mult(meta_model, df, main_probs, regime_features, rule):
    missing = [feature for feature in regime_features if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing regime features: {missing}")
    meta_frame = make_meta_frame(df, main_probs, regime_features)
    quality = predict_meta(meta_model, meta_frame)
    return quality_risk_mult(quality, *rule), quality
