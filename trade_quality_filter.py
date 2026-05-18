import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import xgboost as xgb

from barrier_classifier_strategy import evaluate
from barrier_final_train import FINAL_PARAMS, MODEL_PATH, prepare_barrier_data
from drl_trading_v2 import EXPERT_TRAIN_END_RATIO, SPREAD_POINTS
from drl_train_candidate import format_stats


FILTER_MODEL_PATH = "gold_trade_quality_xgb.json"
FILTER_THRESHOLD = 0.48

QUALITY_FEATURES = [
    "buy_prob",
    "sell_prob",
    "edge",
    "ATR",
    "M1_RSI",
    "MACD_HIST",
    "BB_WIDTH",
    "BIAS_20",
    "BODY_PCT",
    "ROC_5",
    "VOLA_RATIO",
    "HOUR_SIN",
    "HOUR_COS",
    "DAY_OF_WEEK",
    "H1_TREND",
    "H4_TREND",
    "H8_TREND",
    "H12_TREND",
    "Daily_TREND",
    "Weekly_TREND",
]


def candidate_mask(df, probs, params):
    buy_prob = probs[:, 1]
    sell_prob = probs[:, 2]
    edge = np.abs(buy_prob - sell_prob)
    mask = (
        (buy_prob >= sell_prob)
        & (buy_prob >= params["threshold"])
        & (edge >= params["edge_threshold"])
        & df["TIME_DT"].dt.hour.isin(params["allowed_entry_hours"]).to_numpy()
        & df["TIME_DT"].dt.dayofweek.isin(params["allowed_entry_weekdays"]).to_numpy()
    )
    for low, high in params.get("excluded_rsi_ranges", []):
        mask &= ~df["M1_RSI"].between(low, high).to_numpy()
    prev = np.empty(len(mask), dtype=np.int8)
    signal = np.where(buy_prob >= sell_prob, 1, 2).astype(np.int8)
    prev[0] = 0
    prev[1:] = np.where(mask[:-1], signal[:-1], 0)
    return mask & (signal != prev)


def build_quality_frame(df, probs, rows):
    out = df.iloc[rows][[c for c in QUALITY_FEATURES if c in df.columns]].copy()
    out["buy_prob"] = probs[rows, 1]
    out["sell_prob"] = probs[rows, 2]
    out["edge"] = np.abs(probs[rows, 1] - probs[rows, 2])
    return out[QUALITY_FEATURES]


def label_candidates(df, rows, params):
    close = df["CLOSE"].to_numpy(dtype=np.float64)
    atr = df["ATR"].to_numpy(dtype=np.float64)
    labels = []
    rewards = []
    for row in rows:
        entry = close[row]
        tp = max(atr[row] * params["tp_atr"], params["min_tp_price"])
        sl = max(atr[row] * params["sl_atr"], params["min_sl_price"])
        end = min(len(df) - 1, row + params["max_hold"])
        if row >= end:
            reward = -1.0
        else:
            path = close[row + 1 : end + 1] - entry
            hit_tp = np.flatnonzero(path >= tp)
            hit_sl = np.flatnonzero(path <= -sl)
            tp_i = hit_tp[0] if len(hit_tp) else 10**9
            sl_i = hit_sl[0] if len(hit_sl) else 10**9
            if tp_i == 10**9 and sl_i == 10**9:
                exit_idx = end
            elif tp_i <= sl_i:
                exit_idx = row + 1 + tp_i
            else:
                exit_idx = row + 1 + sl_i
            reward = ((close[exit_idx] - entry) * 100.0) - (
                SPREAD_POINTS * 0.01 * 100.0
            )
        rewards.append(reward)
        labels.append(int(reward > 0.0))
    return np.asarray(labels, dtype=np.int8), np.asarray(rewards, dtype=np.float64)


def train_quality_model(x_train, y_train, rewards):
    pos = max(int(y_train.sum()), 1)
    neg = max(len(y_train) - pos, 1)
    sample_weight = np.where(y_train == 1, 1.0 + np.clip(rewards, 0, 500) / 250, 1.0)
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        n_estimators=220,
        learning_rate=0.04,
        max_depth=3,
        min_child_weight=20,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=neg / pos,
        random_state=42,
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)
    model.save_model(FILTER_MODEL_PATH)
    return model


def evaluate_with_filter(df, base_probs, quality_model, threshold):
    q = quality_model.predict_proba(build_quality_frame(df, base_probs, np.arange(len(df))))[:, 1]
    params = dict(FINAL_PARAMS)
    params["min_entry_quality"] = threshold
    return evaluate(
        params,
        df["CLOSE"].to_numpy(dtype=np.float64),
        df["ATR"].to_numpy(dtype=np.float64),
        base_probs,
        hours=df["TIME_DT"].dt.hour.to_numpy(dtype=np.int16),
        weekdays=df["TIME_DT"].dt.dayofweek.to_numpy(dtype=np.int8),
        dates=df["TIME_DT"].dt.date.to_numpy(),
        rsi_values=df["M1_RSI"].to_numpy(dtype=np.float64),
        entry_quality=q,
    )


def main():
    print("Training second-stage trade quality filter...")
    df, features = prepare_barrier_data()
    base_model = xgb.XGBClassifier()
    base_model.load_model(MODEL_PATH)
    base_model.set_params(device="cpu")

    train_end = int(len(df) * EXPERT_TRAIN_END_RATIO)
    val_end = int(len(df) * 0.85)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)

    val_probs = base_model.predict_proba(val_df[features]).astype(np.float32)
    test_probs = base_model.predict_proba(test_df[features]).astype(np.float32)
    rows = np.flatnonzero(candidate_mask(val_df, val_probs, FINAL_PARAMS))
    x_train = build_quality_frame(val_df, val_probs, rows)
    y_train, rewards = label_candidates(val_df, rows, FINAL_PARAMS)
    print(
        f"Quality training candidates={len(rows)} "
        f"win_rate={y_train.mean() if len(y_train) else 0:.2%}"
    )

    quality_model = train_quality_model(x_train, y_train, rewards)
    baseline = evaluate(
        dict(FINAL_PARAMS),
        test_df["CLOSE"].to_numpy(dtype=np.float64),
        test_df["ATR"].to_numpy(dtype=np.float64),
        test_probs,
        hours=test_df["TIME_DT"].dt.hour.to_numpy(dtype=np.int16),
        weekdays=test_df["TIME_DT"].dt.dayofweek.to_numpy(dtype=np.int8),
        dates=test_df["TIME_DT"].dt.date.to_numpy(),
        rsi_values=test_df["M1_RSI"].to_numpy(dtype=np.float64),
    )
    print(format_stats("baseline", baseline))

    for threshold in [0.42, 0.45, 0.48, 0.52, 0.56, 0.60]:
        stats = evaluate_with_filter(test_df, test_probs, quality_model, threshold)
        print(format_stats(f"quality_filter_{threshold:.2f}", stats))

    print(f"Saved quality filter as {FILTER_MODEL_PATH}")


if __name__ == "__main__":
    main()
