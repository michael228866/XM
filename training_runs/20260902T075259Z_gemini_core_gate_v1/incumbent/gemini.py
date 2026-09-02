import csv
import json
import math
import os
import time
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import xgboost as xgb


SYMBOL = "GOLD#"
MAGIC_NUMBER = 20260514
MODEL_FILE = "gold_long_recent_candidate_xgb.json"
MODEL_OUTPUT_MODE = "long_binary"
META_MODEL_FILE = "gold_meta_regime_xgb.json"
META_CONFIG_FILE = "gold_meta_regime_overlay.json"
USE_META_OVERLAY = False
DRY_RUN = False
SIGNAL_LOG_FILE = "gemini_signal_log.csv"
TRADE_HISTORY_LOG_FILE = "gemini_trade_history.csv"
TRADE_HISTORY_LOOKBACK_DAYS = 14
LOG_TIMEZONE = timezone.utc

RISK_PER_TRADE = 0.014
CONF_THRESHOLD = 0.75
EDGE_THRESHOLD = 0.0
LOWER_CONF_THRESHOLD = 1.01
LOWER_CONF_MIN_EDGE = 0.30
EXPANDED_CONF_THRESHOLD = 0.75
MIN_ENTRY_RSI = 22.0
MIN_EXPANDED_META_QUALITY = 0.55
MIN_LOWER_CONF_META_QUALITY = 0.50
TP_ATR_MULT = 1.3
SL_ATR_MULT = 1.6
MIN_TP_PRICE = 1.5
MIN_SL_PRICE = 0.6
SPREAD_POINTS = 30
PRICE_PER_POINT = 0.01
MAX_SPREAD_POINTS = 45
HARD_MAX_SPREAD_POINTS = 100
MAX_SPREAD_TO_TP_RATIO = 0.25
MAX_HOLD_MINUTES = 90
USE_TRAILING_STOP = False
TRAILING_MULT = 1.2
MAX_DAILY_LOSS_PCT = 0.05
MAX_DAILY_TRADES = None
LOSS_COOLDOWN_MINUTES = 15
POST_LOSS_RISK_CAP_MINUTES = 120
POST_LOSS_MAX_RISK_MULT = 1.0
ROLLING_GUARD_WINDOW = 30
ROLLING_GUARD_MIN_TRADES = 18
ROLLING_GUARD_MIN_PROFIT_FACTOR = 1.15
ROLLING_GUARD_RISK_MULT = 0.50
MAX_OPEN_POSITIONS = 1
MAX_ADDON_OPEN_POSITIONS = 1
ADDON_MIN_BUY_PROB = 0.56
ADDON_MIN_META_QUALITY = 0.55
ENABLE_EXPANDED_ENTRY = False

ALLOWED_ENTRY_HOURS = {0, 1, 2, 3, 4, 8, 9, 11, 12, 17, 18, 19, 20, 22, 23}
ALLOWED_ENTRY_WEEKDAYS = {0, 1, 2, 3, 4}
EXPANDED_ENTRY_HOURS = {2, 4}
EXPANDED_ENTRY_WEEKDAYS = {3}
EXCLUDED_RSI_RANGES = [(35.0, 45.0)]

BASE_FEATURES = [
    "M1_RSI",
    "ATR",
    "MACD_HIST",
    "BB_WIDTH",
    "BIAS_20",
    "BODY_PCT",
    "ROC_5",
    "VOLA_RATIO",
    "HOUR_SIN",
    "HOUR_COS",
    "DAY_OF_WEEK",
]

TIMEFRAME_MAP = {
    "M2": mt5.TIMEFRAME_M2,
    "M3": mt5.TIMEFRAME_M3,
    "M4": mt5.TIMEFRAME_M4,
    "M5": mt5.TIMEFRAME_M5,
    "M6": mt5.TIMEFRAME_M6,
    "M10": mt5.TIMEFRAME_M10,
    "M12": mt5.TIMEFRAME_M12,
    "M15": mt5.TIMEFRAME_M15,
    "M20": mt5.TIMEFRAME_M20,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H2": mt5.TIMEFRAME_H2,
    "H3": mt5.TIMEFRAME_H3,
    "H4": mt5.TIMEFRAME_H4,
    "H6": mt5.TIMEFRAME_H6,
    "H8": mt5.TIMEFRAME_H8,
    "H12": mt5.TIMEFRAME_H12,
    "Daily": mt5.TIMEFRAME_D1,
    "Weekly": mt5.TIMEFRAME_W1,
    "Monthly": mt5.TIMEFRAME_MN1,
}

MTF_ORDER = [
    "Daily",
    "H12",
    "H1",
    "H2",
    "H3",
    "H4",
    "H6",
    "H8",
    "M10",
    "M12",
    "M15",
    "M20",
    "M2",
    "M30",
    "M3",
    "M4",
    "M5",
    "M6",
    "Monthly",
    "Weekly",
]

FEATURE_COLUMNS = BASE_FEATURES + [f"{tf}_TREND" for tf in MTF_ORDER]

SIGNAL_LOG_FIELDS = [
    "event_time",
    "bar_time",
    "status",
    "buy_prob",
    "sell_prob",
    "edge",
    "meta_quality",
    "risk_mult",
    "meta_risk_mult",
    "runtime_risk_mult",
    "loss_streak",
    "minutes_since_loss",
    "rolling_profit_factor",
    "risk_guard",
    "hour",
    "weekday",
    "rsi",
    "in_session",
    "base_session",
    "expanded_session",
    "rsi_ok",
    "rsi_floor_ok",
    "valid",
    "entry_mode",
    "position_gate",
    "position_profit",
    "spread_points",
    "spread_limit_points",
    "spread_to_tp_ratio",
    "balance",
    "daily_realized_pnl",
    "floating_pnl",
    "daily_total_pnl",
    "risk_budget",
    "raw_lot",
    "lot",
    "sl_distance",
    "tp_distance",
    "retcode",
    "position_ticket",
    "order_id",
    "deal_id",
    "request_id",
    "broker_price",
    "broker_bid",
    "broker_ask",
    "broker_comment",
    "message",
]

TRADE_HISTORY_FIELDS = [
    "recorded_time",
    "deal_id",
    "order_id",
    "position_id",
    "deal_time",
    "symbol",
    "magic",
    "type",
    "entry",
    "reason",
    "volume",
    "price",
    "profit",
    "commission",
    "swap",
    "fee",
    "comment",
]


def utc_now():
    return datetime.now(LOG_TIMEZONE)


def format_log_time(value):
    if value is None:
        return ""
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            value = value.tz_localize(LOG_TIMEZONE)
        else:
            value = value.tz_convert(LOG_TIMEZONE)
        value = value.to_pydatetime()
    elif isinstance(value, (int, float, np.integer, np.floating)):
        value = datetime.fromtimestamp(value, LOG_TIMEZONE)
    elif not isinstance(value, datetime):
        raise TypeError(f"Unsupported log time value: {value!r}")

    if value.tzinfo is None:
        value = value.replace(tzinfo=LOG_TIMEZONE)
    else:
        value = value.astimezone(LOG_TIMEZONE)
    return value.isoformat(timespec="seconds")


def parse_log_time(value):
    if value is None or value == "":
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=LOG_TIMEZONE)
    return parsed.astimezone(LOG_TIMEZONE)


def parse_float(value, default=0.0):
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed):
        return default
    return parsed


def parse_int(value, default=None):
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def ensure_signal_log_schema():
    if not os.path.exists(SIGNAL_LOG_FILE) or os.path.getsize(SIGNAL_LOG_FILE) == 0:
        return

    with open(SIGNAL_LOG_FILE, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames == SIGNAL_LOG_FIELDS:
            return
        rows = list(reader)

    tmp_file = f"{SIGNAL_LOG_FILE}.tmp"
    with open(tmp_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SIGNAL_LOG_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SIGNAL_LOG_FIELDS})
    os.replace(tmp_file, SIGNAL_LOG_FILE)


def append_signal_log(row):
    ensure_signal_log_schema()
    needs_header = (
        not os.path.exists(SIGNAL_LOG_FILE)
        or os.path.getsize(SIGNAL_LOG_FILE) == 0
    )
    with open(SIGNAL_LOG_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SIGNAL_LOG_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in SIGNAL_LOG_FIELDS})


def fill_trade_result_log(row, result):
    if result is None:
        row["message"] = "mt5.order_send returned None"
        return

    row["retcode"] = getattr(result, "retcode", "")
    row["order_id"] = getattr(result, "order", "")
    row["deal_id"] = getattr(result, "deal", "")
    row["request_id"] = getattr(result, "request_id", "")
    row["broker_price"] = getattr(result, "price", "")
    row["broker_bid"] = getattr(result, "bid", "")
    row["broker_ask"] = getattr(result, "ask", "")
    row["broker_comment"] = getattr(result, "comment", "")
    row["message"] = getattr(result, "comment", "")


def fill_open_position_ticket(row):
    positions = get_strategy_positions()
    if not positions:
        return
    latest_position = max(
        positions,
        key=lambda position: getattr(
            position, "time_msc", getattr(position, "time", 0)
        ),
    )
    row["position_ticket"] = latest_position.ticket


def get_recorded_deal_ids():
    if not os.path.exists(TRADE_HISTORY_LOG_FILE) or os.path.getsize(TRADE_HISTORY_LOG_FILE) == 0:
        return set()

    with open(TRADE_HISTORY_LOG_FILE, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return {row["deal_id"] for row in reader if row.get("deal_id")}


def append_trade_history_rows(rows):
    if not rows:
        return

    needs_header = (
        not os.path.exists(TRADE_HISTORY_LOG_FILE)
        or os.path.getsize(TRADE_HISTORY_LOG_FILE) == 0
    )
    with open(TRADE_HISTORY_LOG_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=TRADE_HISTORY_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def sync_trade_history():
    end_time = utc_now()
    start_time = end_time - timedelta(days=TRADE_HISTORY_LOOKBACK_DAYS)
    deals = mt5.history_deals_get(start_time, end_time)
    if deals is None:
        return

    recorded_ids = get_recorded_deal_ids()
    now_text = format_log_time(end_time)
    rows = []
    for deal in deals:
        if deal.symbol != SYMBOL or deal.magic != MAGIC_NUMBER:
            continue

        deal_id = str(getattr(deal, "ticket", ""))
        if not deal_id or deal_id in recorded_ids:
            continue

        deal_time = format_log_time(deal.time)
        rows.append(
            {
                "recorded_time": now_text,
                "deal_id": deal_id,
                "order_id": getattr(deal, "order", ""),
                "position_id": getattr(deal, "position_id", ""),
                "deal_time": deal_time,
                "symbol": getattr(deal, "symbol", ""),
                "magic": getattr(deal, "magic", ""),
                "type": getattr(deal, "type", ""),
                "entry": getattr(deal, "entry", ""),
                "reason": getattr(deal, "reason", ""),
                "volume": getattr(deal, "volume", ""),
                "price": getattr(deal, "price", ""),
                "profit": getattr(deal, "profit", ""),
                "commission": getattr(deal, "commission", ""),
                "swap": getattr(deal, "swap", ""),
                "fee": getattr(deal, "fee", ""),
                "comment": getattr(deal, "comment", ""),
            }
        )
        recorded_ids.add(deal_id)

    append_trade_history_rows(rows)


def get_closed_trade_results():
    if not os.path.exists(TRADE_HISTORY_LOG_FILE):
        return []

    by_position = {}
    try:
        with open(TRADE_HISTORY_LOG_FILE, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if parse_int(row.get("entry")) != 1:
                    continue

                position_id = row.get("position_id")
                exit_time = parse_log_time(row.get("deal_time"))
                if not position_id or exit_time is None:
                    continue

                net = (
                    parse_float(row.get("profit"))
                    + parse_float(row.get("commission"))
                    + parse_float(row.get("swap"))
                    + parse_float(row.get("fee"))
                )
                trade = by_position.setdefault(
                    position_id,
                    {"exit_time": exit_time, "pnl": 0.0},
                )
                trade["pnl"] += net
                if exit_time >= trade["exit_time"]:
                    trade["exit_time"] = exit_time
    except (OSError, csv.Error):
        return []

    return sorted(by_position.values(), key=lambda trade: trade["exit_time"])


def get_consecutive_losses(closed_trades):
    streak = 0
    for trade in reversed(closed_trades):
        if trade["pnl"] < 0:
            streak += 1
            continue
        break
    return streak


def get_profit_factor(closed_trades):
    gross_win = sum(trade["pnl"] for trade in closed_trades if trade["pnl"] > 0)
    gross_loss = -sum(trade["pnl"] for trade in closed_trades if trade["pnl"] < 0)
    if gross_loss <= 0:
        return math.inf
    return gross_win / gross_loss


def format_guard_value(value, digits=2):
    if value is None:
        return ""
    if math.isinf(value):
        return "inf"
    return round(value, digits)


def get_runtime_risk_guard(now):
    closed_trades = get_closed_trade_results()
    guard = {
        "allow_entry": True,
        "risk_mult": 1.0,
        "max_risk_mult": None,
        "loss_streak": 0,
        "minutes_since_loss": None,
        "rolling_profit_factor": None,
        "reason": "",
    }
    if not closed_trades:
        return guard

    reasons = []
    guard["loss_streak"] = get_consecutive_losses(closed_trades)
    last_trade = closed_trades[-1]
    if last_trade["pnl"] < 0:
        minutes_since_loss = max(
            0.0,
            (now - last_trade["exit_time"]).total_seconds() / 60.0,
        )
        guard["minutes_since_loss"] = minutes_since_loss
        if minutes_since_loss < LOSS_COOLDOWN_MINUTES:
            guard["allow_entry"] = False
            reasons.append(
                f"loss_cooldown={minutes_since_loss:.1f}"
                f"<{LOSS_COOLDOWN_MINUTES}"
            )
        if minutes_since_loss < POST_LOSS_RISK_CAP_MINUTES:
            guard["max_risk_mult"] = POST_LOSS_MAX_RISK_MULT
            reasons.append(f"post_loss_cap={POST_LOSS_MAX_RISK_MULT:.2f}")

    recent_trades = closed_trades[-ROLLING_GUARD_WINDOW:]
    if len(recent_trades) >= ROLLING_GUARD_MIN_TRADES:
        profit_factor = get_profit_factor(recent_trades)
        guard["rolling_profit_factor"] = profit_factor
        if profit_factor < ROLLING_GUARD_MIN_PROFIT_FACTOR:
            guard["risk_mult"] *= ROLLING_GUARD_RISK_MULT
            reasons.append(
                f"rolling_pf={profit_factor:.2f}"
                f"<{ROLLING_GUARD_MIN_PROFIT_FACTOR:.2f}"
            )

    guard["reason"] = ";".join(reasons)
    return guard


def get_effective_risk_mult(meta_risk_mult, runtime_guard):
    effective = meta_risk_mult * runtime_guard["risk_mult"]
    max_risk_mult = runtime_guard.get("max_risk_mult")
    if max_risk_mult is not None:
        effective = min(effective, max_risk_mult)
    return effective


def add_indicators(df):
    df = df.copy()
    high_low = df["HIGH"] - df["LOW"]
    tr = pd.concat(
        [
            high_low,
            np.abs(df["HIGH"] - df["CLOSE"].shift()),
            np.abs(df["LOW"] - df["CLOSE"].shift()),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["HOUR_SIN"] = np.sin(2 * np.pi * df["TIME_DT"].dt.hour / 24)
    df["HOUR_COS"] = np.cos(2 * np.pi * df["TIME_DT"].dt.hour / 24)
    df["DAY_OF_WEEK"] = df["TIME_DT"].dt.dayofweek / 7.0

    ema12 = df["CLOSE"].ewm(span=12, adjust=False).mean()
    ema26 = df["CLOSE"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df["MACD_HIST"] = macd - macd.ewm(span=9, adjust=False).mean()

    ma20 = df["CLOSE"].rolling(20).mean()
    df["BB_WIDTH"] = (df["CLOSE"].rolling(20).std() * 4) / (ma20 + 1e-6)
    df["BIAS_20"] = (df["CLOSE"] - ma20) / (ma20 + 1e-6)
    df["ROC_5"] = df["CLOSE"].pct_change(5)
    candle_range = df["HIGH"] - df["LOW"] + 1e-6
    df["BODY_PCT"] = np.abs(df["CLOSE"] - df["OPEN"]) / candle_range
    return df


def build_live_regime_frame(m1, feat_df, regime_features):
    df = m1.copy()
    close = df["CLOSE"]
    high = df["HIGH"]
    low = df["LOW"]
    open_ = df["OPEN"]
    candle_range = (high - low).replace(0, np.nan)

    trend_cols = [col for col in feat_df.columns if col.endswith("_TREND")]
    trend_score = (
        int((feat_df[trend_cols].iloc[0] > 0).sum()) if trend_cols else 0
    )

    df["REG_TREND_SCORE"] = float(trend_score)
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
    df[regime_features] = df[regime_features].shift(1)

    if len(df) == 0:
        return None
    row = df.iloc[-1]
    if row[regime_features].isna().any():
        return None
    return pd.DataFrame([{name: row[name] for name in regime_features}])


def get_mt5_data(symbol, timeframe, count, start_pos=1):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["TIME_DT"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.columns = [col.upper() for col in df.columns]
    return df.sort_values("TIME_DT").reset_index(drop=True)


def get_trend(symbol, timeframe):
    df = get_mt5_data(symbol, timeframe, 80)
    if df is None or len(df) < 25:
        return None
    close = df["CLOSE"]
    trend = 1 if close.iloc[-1] > close.rolling(20).mean().iloc[-1] else -1
    return trend


def get_current_features(regime_features=None):
    m1 = get_mt5_data(SYMBOL, mt5.TIMEFRAME_M1, 400)
    if m1 is None or len(m1) < 260:
        return None

    m1 = add_indicators(m1)
    diff = m1["CLOSE"].diff()
    gain = diff.where(diff > 0, 0).rolling(14).mean()
    loss = (-diff.where(diff < 0, 0)).rolling(14).mean()
    m1["M1_RSI"] = 100 - (100 / (1 + (gain / (loss + 1e-6))))
    m1["VOLA_MA"] = m1["ATR"].rolling(240).mean()
    m1["VOLA_RATIO"] = m1["ATR"] / (m1["VOLA_MA"] + 1e-6)

    last_row = m1.iloc[-1]
    features = {name: last_row[name] for name in BASE_FEATURES}
    for tf_name in MTF_ORDER:
        trend = get_trend(SYMBOL, TIMEFRAME_MAP[tf_name])
        if trend is None:
            return None
        features[f"{tf_name}_TREND"] = trend

    feat_df = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    if feat_df.isna().any(axis=None):
        return None

    regime_df = None
    if regime_features is not None:
        regime_df = build_live_regime_frame(m1, feat_df, regime_features)
        if regime_df is None or regime_df.isna().any(axis=None):
            return None
    return feat_df, regime_df, last_row["TIME_DT"]


def normalize_model_probs(raw_probs):
    probs = np.asarray(raw_probs, dtype=np.float32).reshape(-1)
    if MODEL_OUTPUT_MODE == "three_class" and len(probs) == 3:
        return probs
    if MODEL_OUTPUT_MODE == "long_binary" and len(probs) == 2:
        return np.asarray([probs[0], probs[1], 0.0], dtype=np.float32)
    raise ValueError(
        f"Unexpected model output: mode={MODEL_OUTPUT_MODE} classes={len(probs)}"
    )


def load_meta_overlay():
    if not USE_META_OVERLAY:
        return None, None
    meta_model = xgb.XGBClassifier()
    meta_model.load_model(META_MODEL_FILE)
    meta_model.set_params(device="cpu")
    with open(META_CONFIG_FILE, "r", encoding="utf-8") as file:
        config = json.load(file)
    return meta_model, config


def get_meta_quality_and_risk_mult(meta_model, meta_config, regime_df, probs):
    if meta_model is None or meta_config is None or regime_df is None:
        return 1.0, None

    meta_df = regime_df.copy()
    buy_prob = float(probs[1])
    sell_prob = float(probs[2])
    meta_df["BASE_BUY_PROB"] = buy_prob
    meta_df["BASE_SELL_PROB"] = sell_prob
    meta_df["BASE_EDGE"] = buy_prob - sell_prob
    meta_df["BASE_CONF"] = max(buy_prob, sell_prob)
    meta_df["BASE_NO_TRADE_PROB"] = float(probs[0])

    meta_probs = meta_model.predict_proba(meta_df)[0]
    class_to_index = {int(cls): idx for idx, cls in enumerate(meta_model.classes_)}
    quality = float(meta_probs[class_to_index.get(1, len(meta_probs) - 1)])
    protect_cut, boost_cut, strong_cut, protect_mult, boost_mult, strong_mult = (
        float(value) for value in meta_config["risk_rule"]
    )

    risk_mult = 1.0
    if quality < protect_cut:
        risk_mult = protect_mult
    if quality >= boost_cut:
        risk_mult = boost_mult
    if quality >= strong_cut:
        risk_mult = strong_mult
    return risk_mult, quality


def get_strategy_positions():
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None:
        return []
    return [pos for pos in positions if pos.magic == MAGIC_NUMBER]


def is_meta_quality_at_least(meta_quality, threshold):
    return meta_quality is not None and meta_quality >= threshold


def get_entry_mode(base_session, expanded_session, buy_prob, edge, meta_quality):
    if (
        base_session
        and buy_prob >= CONF_THRESHOLD
        and edge >= EDGE_THRESHOLD
    ):
        return "base"
    if (
        expanded_session
        and buy_prob >= EXPANDED_CONF_THRESHOLD
        and edge >= EDGE_THRESHOLD
        and is_meta_quality_at_least(meta_quality, MIN_EXPANDED_META_QUALITY)
    ):
        return "expanded_time"
    if (
        base_session
        and buy_prob >= LOWER_CONF_THRESHOLD
        and edge >= LOWER_CONF_MIN_EDGE
        and is_meta_quality_at_least(meta_quality, MIN_LOWER_CONF_META_QUALITY)
    ):
        return "lower_conf_quality"
    return ""


def get_position_gate(positions, buy_prob, meta_quality):
    open_count = len(positions)
    if open_count == 0:
        return True, "base_position"
    if open_count >= MAX_ADDON_OPEN_POSITIONS:
        return False, "max_positions"

    position_profits = [
        float(getattr(position, "profit", 0.0))
        for position in positions
    ]
    if any(profit < 0 for profit in position_profits):
        return False, "losing_position_block"

    can_add_on = (
        position_profits
        and sum(position_profits) > 0
        and all(profit >= 0 for profit in position_profits)
        and buy_prob >= ADDON_MIN_BUY_PROB
        and is_meta_quality_at_least(meta_quality, ADDON_MIN_META_QUALITY)
    )
    if can_add_on:
        if open_count < MAX_OPEN_POSITIONS:
            return True, "profit_second_position"
        return True, "profit_addon"
    if open_count < MAX_OPEN_POSITIONS:
        return False, "second_position_quality"
    return False, "position_limit"


def get_position_profit(positions):
    return sum(float(getattr(position, "profit", 0.0)) for position in positions)


def get_unrealized_pnl(positions=None):
    if positions is None:
        positions = get_strategy_positions()
    return get_position_profit(positions)


def get_daily_realized_pnl():
    now = datetime.now()
    day_start = datetime(now.year, now.month, now.day)
    deals = mt5.history_deals_get(day_start, now)
    if deals is None:
        return 0.0
    return sum(
        deal.profit
        + deal.swap
        + deal.commission
        + getattr(deal, "fee", 0.0)
        for deal in deals
        if deal.symbol == SYMBOL and deal.magic == MAGIC_NUMBER
    )


def get_daily_entry_count():
    now = datetime.now()
    day_start = datetime(now.year, now.month, now.day)
    deals = mt5.history_deals_get(day_start, now)
    if deals is None:
        return 0
    return sum(
        1
        for deal in deals
        if deal.symbol == SYMBOL
        and deal.magic == MAGIC_NUMBER
        and deal.entry == mt5.DEAL_ENTRY_IN
    )


def normalize_lot(raw_lot):
    info = mt5.symbol_info(SYMBOL)
    if info is None:
        return None
    min_lot = info.volume_min
    max_lot = info.volume_max
    step = info.volume_step
    if raw_lot < min_lot:
        return None
    steps = math.floor(min(raw_lot, max_lot) / step)
    lot = steps * step
    precision = max(0, int(round(-math.log10(step)))) if step < 1 else 0
    return round(lot, precision)


def execute_trade(order_type, lot, sl_distance, tp_distance):
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return None
    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
    sl = price - sl_distance if order_type == mt5.ORDER_TYPE_BUY else price + sl_distance
    tp = price + tp_distance if order_type == mt5.ORDER_TYPE_BUY else price - tp_distance
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": MAGIC_NUMBER,
        "comment": "BarrierLive",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    return mt5.order_send(request)


def get_current_spread_points():
    tick = mt5.symbol_info_tick(SYMBOL)
    info = mt5.symbol_info(SYMBOL)
    if tick is None or info is None or info.point <= 0:
        return None
    return (tick.ask - tick.bid) / info.point


def get_spread_limit_points(tp_distance):
    if tp_distance <= 0:
        return MAX_SPREAD_POINTS
    ratio_limit = (tp_distance * MAX_SPREAD_TO_TP_RATIO) / PRICE_PER_POINT
    return min(HARD_MAX_SPREAD_POINTS, max(MAX_SPREAD_POINTS, ratio_limit))


def get_spread_to_tp_ratio(spread_points, tp_distance):
    if tp_distance <= 0:
        return float("inf")
    return (spread_points * PRICE_PER_POINT) / tp_distance


def close_position(position):
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return False
    if position.type == mt5.POSITION_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": position.volume,
        "type": order_type,
        "position": position.ticket,
        "price": price,
        "magic": MAGIC_NUMBER,
        "comment": "BarrierLive Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE


def manage_open_positions():
    for position in get_strategy_positions():
        held_seconds = time.time() - position.time
        if held_seconds >= MAX_HOLD_MINUTES * 60:
            if close_position(position):
                print(f"Timeout close: ticket={position.ticket}")
            continue
        if not USE_TRAILING_STOP:
            continue

        tick = mt5.symbol_info_tick(SYMBOL)
        rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 1, 20)
        if tick is None or rates is None or len(rates) < 14:
            continue
        df = pd.DataFrame(rates)
        curr_atr = (df["high"] - df["low"]).abs().rolling(14).mean().iloc[-1]
        if position.type == mt5.POSITION_TYPE_BUY:
            float_profit = tick.bid - position.price_open
        else:
            float_profit = position.price_open - tick.ask
        if float_profit >= TRAILING_MULT * curr_atr:
            close_position(position)


def live_trading_loop():
    if not mt5.initialize():
        print("MT5 initialize failed")
        return

    model = xgb.XGBClassifier()
    model.load_model(MODEL_FILE)
    model.set_params(device="cpu")
    meta_model, meta_config = load_meta_overlay()
    regime_features = None if meta_config is None else meta_config["regime_features"]
    print(
        "Barrier live strategy active. Monitoring GOLD... "
        f"model={MODEL_FILE} output={MODEL_OUTPUT_MODE} "
        f"meta_overlay={USE_META_OVERLAY and meta_model is not None} "
        f"dry_run={DRY_RUN}"
    )

    last_bar_time = None
    last_history_sync_time = 0
    while True:
        try:
            manage_open_positions()
            now = utc_now()
            if time.time() - last_history_sync_time >= 60:
                sync_trade_history()
                last_history_sync_time = time.time()
            if now.second != 0:
                time.sleep(0.5)
                continue

            feature_pack = get_current_features(regime_features)
            if feature_pack is None:
                print(f"[{now:%H:%M:%S}] Feature data not ready")
                time.sleep(1)
                continue

            feat, regime_df, bar_time = feature_pack
            if last_bar_time == bar_time:
                time.sleep(1)
                continue
            last_bar_time = bar_time

            hour = int(bar_time.hour)
            weekday = int(bar_time.dayofweek)
            probs = normalize_model_probs(model.predict_proba(feat)[0])
            buy_prob = float(probs[1])
            sell_prob = float(probs[2])
            edge = abs(buy_prob - sell_prob)
            meta_risk_mult, meta_quality = get_meta_quality_and_risk_mult(
                meta_model,
                meta_config,
                regime_df,
                probs,
            )
            positions = get_strategy_positions()
            base_session = (
                hour in ALLOWED_ENTRY_HOURS and weekday in ALLOWED_ENTRY_WEEKDAYS
            )
            expanded_session = (
                ENABLE_EXPANDED_ENTRY
                and (
                    (
                        hour in EXPANDED_ENTRY_HOURS
                        and weekday in ALLOWED_ENTRY_WEEKDAYS
                    )
                    or (
                        hour in ALLOWED_ENTRY_HOURS
                        and weekday in EXPANDED_ENTRY_WEEKDAYS
                    )
                )
            )
            in_session = base_session or expanded_session
            rsi = float(feat["M1_RSI"].iloc[0])
            rsi_ok = not any(low <= rsi <= high for low, high in EXCLUDED_RSI_RANGES)
            rsi_floor_ok = rsi >= MIN_ENTRY_RSI
            open_position_count = len(positions)
            entry_mode = get_entry_mode(
                base_session,
                expanded_session,
                buy_prob,
                edge,
                meta_quality,
            )
            position_ok, position_gate = get_position_gate(
                positions,
                buy_prob,
                meta_quality,
            )
            position_profit = get_position_profit(positions)
            is_valid = (
                position_ok
                and in_session
                and rsi_ok
                and rsi_floor_ok
                and entry_mode != ""
                and buy_prob >= sell_prob
            )
            print(
                f"[{now:%H:%M:%S}] buy={buy_prob:.3f} sell={sell_prob:.3f} "
                f"edge={edge:.3f} hour={hour} weekday={weekday} "
                f"rsi={rsi:.1f} session={in_session} rsi_ok={rsi_ok} "
                f"meta_q={meta_quality if meta_quality is not None else -1:.3f} "
                f"risk_mult={meta_risk_mult:.2f} positions="
                f"{open_position_count}/{MAX_OPEN_POSITIONS} "
                f"mode={entry_mode or '-'} gate={position_gate} valid={is_valid}"
            )
            log_row = {
                "event_time": format_log_time(now),
                "bar_time": format_log_time(bar_time),
                "buy_prob": round(buy_prob, 6),
                "sell_prob": round(sell_prob, 6),
                "edge": round(edge, 6),
                "meta_quality": (
                    "" if meta_quality is None else round(meta_quality, 6)
                ),
                "risk_mult": round(meta_risk_mult, 4),
                "meta_risk_mult": round(meta_risk_mult, 4),
                "runtime_risk_mult": "",
                "loss_streak": "",
                "minutes_since_loss": "",
                "rolling_profit_factor": "",
                "risk_guard": "",
                "hour": hour,
                "weekday": weekday,
                "rsi": round(rsi, 4),
                "in_session": in_session,
                "base_session": base_session,
                "expanded_session": expanded_session,
                "rsi_ok": rsi_ok,
                "rsi_floor_ok": rsi_floor_ok,
                "valid": is_valid,
                "entry_mode": entry_mode,
                "position_gate": position_gate,
                "position_profit": round(position_profit, 2),
            }

            if not is_valid:
                log_row["status"] = "not_valid"
                invalid_reasons = []
                if not position_ok:
                    invalid_reasons.append(position_gate)
                if not in_session:
                    invalid_reasons.append("session")
                if not rsi_ok:
                    invalid_reasons.append("rsi_range")
                if not rsi_floor_ok:
                    invalid_reasons.append("rsi_floor")
                if entry_mode == "":
                    invalid_reasons.append("entry_mode")
                if buy_prob < sell_prob:
                    invalid_reasons.append("buy_below_sell")
                log_row["message"] = ";".join(invalid_reasons)
                append_signal_log(log_row)
                time.sleep(1)
                continue

            runtime_guard = get_runtime_risk_guard(now)
            effective_risk_mult = get_effective_risk_mult(
                meta_risk_mult,
                runtime_guard,
            )
            log_row.update(
                {
                    "risk_mult": round(effective_risk_mult, 4),
                    "runtime_risk_mult": round(runtime_guard["risk_mult"], 4),
                    "loss_streak": runtime_guard["loss_streak"],
                    "minutes_since_loss": format_guard_value(
                        runtime_guard["minutes_since_loss"],
                        1,
                    ),
                    "rolling_profit_factor": format_guard_value(
                        runtime_guard["rolling_profit_factor"],
                        2,
                    ),
                    "risk_guard": runtime_guard["reason"],
                }
            )
            if not runtime_guard["allow_entry"]:
                print(f"Skip: runtime risk guard ({runtime_guard['reason']})")
                log_row["status"] = "loss_cooldown"
                log_row["message"] = runtime_guard["reason"]
                append_signal_log(log_row)
                time.sleep(1)
                continue

            atr = float(feat["ATR"].iloc[0])
            sl_distance = max(atr * SL_ATR_MULT, MIN_SL_PRICE)
            tp_distance = max(atr * TP_ATR_MULT, MIN_TP_PRICE)
            log_row.update(
                {
                    "sl_distance": round(sl_distance, 4),
                    "tp_distance": round(tp_distance, 4),
                }
            )

            spread_points = get_current_spread_points()
            if spread_points is None:
                print("Skip: spread unavailable")
                log_row["status"] = "spread_unavailable"
                append_signal_log(log_row)
                time.sleep(1)
                continue
            spread_limit_points = get_spread_limit_points(tp_distance)
            spread_to_tp_ratio = get_spread_to_tp_ratio(
                spread_points,
                tp_distance,
            )
            log_row["spread_points"] = round(spread_points, 2)
            log_row["spread_limit_points"] = round(spread_limit_points, 2)
            log_row["spread_to_tp_ratio"] = round(spread_to_tp_ratio, 4)
            if spread_points > spread_limit_points:
                print(
                    f"Skip: spread too wide "
                    f"({spread_points:.1f} > {spread_limit_points:.1f} points)"
                )
                log_row["status"] = "spread_too_wide"
                log_row["message"] = (
                    f"{spread_points:.1f} > {spread_limit_points:.1f}; "
                    f"spread/tp={spread_to_tp_ratio:.3f}"
                )
                append_signal_log(log_row)
                time.sleep(1)
                continue

            account = mt5.account_info()
            if account is None or account.balance <= 0:
                print("Account info unavailable or invalid balance")
                log_row["status"] = "account_unavailable"
                append_signal_log(log_row)
                time.sleep(1)
                continue
            log_row["balance"] = round(float(account.balance), 2)
            daily_realized_pnl = get_daily_realized_pnl()
            floating_pnl = get_unrealized_pnl(positions)
            daily_total_pnl = daily_realized_pnl + floating_pnl
            log_row.update(
                {
                    "daily_realized_pnl": round(daily_realized_pnl, 2),
                    "floating_pnl": round(floating_pnl, 2),
                    "daily_total_pnl": round(daily_total_pnl, 2),
                }
            )
            if daily_total_pnl <= -account.balance * MAX_DAILY_LOSS_PCT:
                print(
                    f"Skip: daily loss guard active "
                    f"({daily_total_pnl:.2f} <= "
                    f"{-account.balance * MAX_DAILY_LOSS_PCT:.2f})"
                )
                log_row["status"] = "daily_loss_guard"
                log_row["message"] = (
                    f"{daily_total_pnl:.2f} <= "
                    f"{-account.balance * MAX_DAILY_LOSS_PCT:.2f}; "
                    f"realized={daily_realized_pnl:.2f}; "
                    f"floating={floating_pnl:.2f}"
                )
                append_signal_log(log_row)
                time.sleep(1)
                continue
            if MAX_DAILY_TRADES is not None and get_daily_entry_count() >= MAX_DAILY_TRADES:
                print("Skip: max daily trade count reached")
                log_row["status"] = "max_daily_trades"
                append_signal_log(log_row)
                time.sleep(1)
                continue

            stop_cost = (
                sl_distance * 100.0
                + spread_points * PRICE_PER_POINT * 100.0
            )
            risk_budget = account.balance * RISK_PER_TRADE * effective_risk_mult
            raw_lot = risk_budget / max(stop_cost, 1e-9)
            lot = normalize_lot(raw_lot)
            log_row.update(
                {
                    "risk_budget": round(risk_budget, 4),
                    "raw_lot": round(raw_lot, 6),
                    "lot": "" if lot is None else lot,
                }
            )
            if lot is None:
                print(
                    f"Skip: raw lot {raw_lot:.4f} is below broker minimum "
                    "for current risk settings"
                )
                log_row["status"] = "lot_below_minimum"
                append_signal_log(log_row)
                time.sleep(1)
                continue

            if DRY_RUN:
                print(
                    f"DRY RUN BUY | lot={lot} raw_lot={raw_lot:.4f} "
                    f"risk_budget={risk_budget:.2f} meta_q="
                    f"{meta_quality if meta_quality is not None else -1:.3f} "
                    f"risk_mult={effective_risk_mult:.2f} "
                    f"buy_prob={buy_prob:.2%} "
                    f"SL={sl_distance:.2f} TP={tp_distance:.2f}"
                )
                log_row["status"] = "dry_run_buy"
                append_signal_log(log_row)
                time.sleep(1)
                continue

            result = execute_trade(mt5.ORDER_TYPE_BUY, lot, sl_distance, tp_distance)
            fill_trade_result_log(log_row, result)
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                print(
                    f"BUY opened | lot={lot} risk_mult={effective_risk_mult:.2f} "
                    f"meta_q={meta_quality if meta_quality is not None else -1:.3f} "
                    f"buy_prob={buy_prob:.2%} "
                    f"SL={sl_distance:.2f} TP={tp_distance:.2f}"
                )
                log_row["status"] = "order_opened"
                fill_open_position_ticket(log_row)
            else:
                code = None if result is None else result.retcode
                print(f"Order failed | retcode={code}")
                log_row["status"] = "order_failed"
            append_signal_log(log_row)

            time.sleep(1)
        except Exception as exc:
            print(f"Error: {exc}")
            time.sleep(5)


if __name__ == "__main__":
    live_trading_loop()
