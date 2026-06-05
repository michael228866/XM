from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

RESEARCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RESEARCH_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import MetaTrader5 as mt5  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402

LOG_FILE = RESEARCH_DIR / "multi_metal_trader_signal_log.csv"
SILVER_MODEL_FILE = RESEARCH_DIR / "silver_h1_regime_selected_xgb.json"
SHARED_MODEL_FILE = RESEARCH_DIR / "all_metals_h1_smooth_more_trees_xgb.json"
GOLD_MODEL_FILE = PROJECT_ROOT / "gold_barrier_final_xgb.json"
GOLD_META_MODEL_FILE = PROJECT_ROOT / "gold_meta_regime_xgb.json"
GOLD_META_CONFIG_FILE = PROJECT_ROOT / "gold_meta_regime_overlay.json"

BASE_MAGIC = 202605310
SLEEP_SECONDS = 20
MAX_TOTAL_POSITIONS = 4
MAX_POSITIONS_PER_SYMBOL = 1
MAX_DAILY_LOSS_PCT = 0.06

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
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

GOLD_MTF_ORDER = [
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
GOLD_BASE_FEATURES = [
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
GOLD_FEATURE_COLUMNS = GOLD_BASE_FEATURES + [f"{tf}_TREND" for tf in GOLD_MTF_ORDER]
SHARED_SYMBOLS = ["GOLD#", "SILVER#", "XAUEUR#", "XPTUSD#", "XPDUSD#", "GAUCNH#"]
REGIME_TREND_COLUMNS = [
    "H4_TREND",
    "H8_TREND",
    "H12_TREND",
    "Daily_TREND",
    "Weekly_TREND",
    "Monthly_TREND",
]

LOG_FIELDS = [
    "time",
    "strategy",
    "symbol",
    "status",
    "signal",
    "buy_prob",
    "sell_prob",
    "confidence",
    "edge",
    "threshold",
    "spread_points",
    "spread_atr",
    "atr",
    "sl_distance",
    "tp_distance",
    "risk_budget",
    "raw_volume",
    "volume",
    "balance",
    "equity",
    "message",
    "retcode",
    "order",
    "deal",
]


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    symbol: str
    model_kind: str
    model_path: Path
    enabled: bool
    threshold: float
    edge_threshold: float
    tp_atr: float
    sl_atr: float
    max_hold_bars: int
    direction_mode: str
    risk_per_trade: float
    max_spread_atr: float
    max_spread_points: float | None
    magic: int
    bar_minutes: int
    regime_trend_min: float | None = None
    regime_rsi_min: float | None = None
    regime_rsi_max: float | None = None
    regime_vola_max: float | None = None
    regime_spread_atr_stress_max: float | None = None
    regime_macd_min: float | None = None


STRATEGIES = [
    StrategyConfig(
        name="gold_dedicated_m1",
        symbol="GOLD#",
        model_kind="gold_m1",
        model_path=GOLD_MODEL_FILE,
        enabled=True,
        threshold=0.525,
        edge_threshold=0.0,
        tp_atr=1.3,
        sl_atr=2.0,
        max_hold_bars=180,
        direction_mode="long",
        risk_per_trade=0.010,
        max_spread_atr=0.35,
        max_spread_points=45.0,
        magic=BASE_MAGIC + 1,
        bar_minutes=1,
    ),
    StrategyConfig(
        name="silver_h1_regime",
        symbol="SILVER#",
        model_kind="symbol_h1",
        model_path=SILVER_MODEL_FILE,
        enabled=True,
        threshold=0.52,
        edge_threshold=0.0,
        tp_atr=6.0,
        sl_atr=6.0,
        max_hold_bars=336,
        direction_mode="long",
        risk_per_trade=0.004,
        max_spread_atr=0.25,
        max_spread_points=None,
        magic=BASE_MAGIC + 2,
        bar_minutes=60,
        regime_trend_min=0.0,
        regime_rsi_min=0.0,
        regime_rsi_max=100.0,
        regime_vola_max=1.0,
        regime_spread_atr_stress_max=0.75,
    ),
    StrategyConfig(
        name="xaueur_shared_h1",
        symbol="XAUEUR#",
        model_kind="shared_h1",
        model_path=SHARED_MODEL_FILE,
        enabled=True,
        threshold=0.54,
        edge_threshold=0.0,
        tp_atr=2.2,
        sl_atr=4.2,
        max_hold_bars=288,
        direction_mode="both",
        risk_per_trade=0.004,
        max_spread_atr=0.18,
        max_spread_points=None,
        magic=BASE_MAGIC + 3,
        bar_minutes=60,
    ),
    StrategyConfig(
        name="xpt_shared_watch_h1",
        symbol="XPTUSD#",
        model_kind="shared_h1",
        model_path=SHARED_MODEL_FILE,
        enabled=False,
        threshold=0.52,
        edge_threshold=0.0,
        tp_atr=3.2,
        sl_atr=3.4,
        max_hold_bars=120,
        direction_mode="long",
        risk_per_trade=0.004,
        max_spread_atr=0.15,
        max_spread_points=None,
        magic=BASE_MAGIC + 4,
        bar_minutes=60,
    ),
]


def append_log(row: dict) -> None:
    needs_header = not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0
    with LOG_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LOG_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in LOG_FIELDS})


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    high_low = df["HIGH"] - df["LOW"]
    true_range = pd.concat(
        [
            high_low,
            (df["HIGH"] - df["CLOSE"].shift()).abs(),
            (df["LOW"] - df["CLOSE"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR"] = true_range.rolling(14).mean()
    df["HOUR_SIN"] = np.sin(2 * np.pi * df["TIME_DT"].dt.hour / 24)
    df["HOUR_COS"] = np.cos(2 * np.pi * df["TIME_DT"].dt.hour / 24)
    df["DAY_OF_WEEK"] = df["TIME_DT"].dt.dayofweek / 7.0
    ema12 = df["CLOSE"].ewm(span=12, adjust=False).mean()
    ema26 = df["CLOSE"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df["MACD_HIST"] = macd - macd.ewm(span=9, adjust=False).mean()
    ma20 = df["CLOSE"].rolling(20).mean()
    df["BB_WIDTH"] = (df["CLOSE"].rolling(20).std() * 4) / (ma20 + 1e-9)
    df["BIAS_20"] = (df["CLOSE"] - ma20) / (ma20 + 1e-9)
    df["ROC_5"] = df["CLOSE"].pct_change(5)
    candle_range = df["HIGH"] - df["LOW"] + 1e-9
    df["BODY_PCT"] = (df["CLOSE"] - df["OPEN"]).abs() / candle_range
    df["ATR_PCT"] = df["ATR"] / (df["CLOSE"].abs() + 1e-9)
    df["MACD_ATR"] = df["MACD_HIST"] / (df["ATR"] + 1e-9)
    return df


def add_rsi_and_volatility(df: pd.DataFrame, rsi_name: str) -> pd.DataFrame:
    df = add_indicators(df)
    diff = df["CLOSE"].diff()
    gain = diff.where(diff > 0, 0).rolling(14).mean()
    loss = (-diff.where(diff < 0, 0)).rolling(14).mean()
    df[rsi_name] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    df["VOLA_MA"] = df["ATR"].rolling(240).mean()
    df["VOLA_RATIO"] = df["ATR"] / (df["VOLA_MA"] + 1e-9)
    return df


def get_mt5_data(symbol: str, timeframe: int, count: int, start_pos: int = 1):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["TIME_DT"] = pd.to_datetime(df["time"], unit="s")
    df.columns = [col.upper() for col in df.columns]
    return df.sort_values("TIME_DT").reset_index(drop=True)


def get_trend(symbol: str, timeframe_name: str) -> int | None:
    timeframe = TIMEFRAME_MAP.get(timeframe_name)
    if timeframe is None:
        return None
    df = get_mt5_data(symbol, timeframe, 90)
    if df is None or len(df) < 25:
        return None
    close = df["CLOSE"]
    return 1 if close.iloc[-1] > close.rolling(20).mean().iloc[-1] else -1


def build_gold_features(symbol: str):
    m1 = get_mt5_data(symbol, mt5.TIMEFRAME_M1, 420)
    if m1 is None or len(m1) < 260:
        return None
    m1 = add_rsi_and_volatility(m1, "M1_RSI")
    row = m1.iloc[-1]
    features = {name: row[name] for name in GOLD_BASE_FEATURES}
    for timeframe in GOLD_MTF_ORDER:
        trend = get_trend(symbol, timeframe)
        if trend is None:
            return None
        features[f"{timeframe}_TREND"] = trend
    feature_df = pd.DataFrame([features], columns=GOLD_FEATURE_COLUMNS)
    if feature_df.isna().any(axis=None):
        return None
    return feature_df, float(row["ATR"]), row["TIME_DT"]


def symbol_id(symbol: str) -> float:
    return float(SHARED_SYMBOLS.index(symbol)) if symbol in SHARED_SYMBOLS else -1.0


def build_h1_features(symbol: str, feature_names: list[str]):
    h1 = get_mt5_data(symbol, mt5.TIMEFRAME_H1, 460)
    if h1 is None or len(h1) < 260:
        return None
    h1 = add_rsi_and_volatility(h1, "BASE_RSI")
    row = h1.iloc[-1]
    values = {}
    context = {}
    base_values = {
        "BASE_RSI": row["BASE_RSI"],
        "ATR_PCT": row["ATR_PCT"],
        "MACD_ATR": row["MACD_ATR"],
        "BB_WIDTH": row["BB_WIDTH"],
        "BIAS_20": row["BIAS_20"],
        "BODY_PCT": row["BODY_PCT"],
        "ROC_5": row["ROC_5"],
        "VOLA_RATIO": row["VOLA_RATIO"],
        "HOUR_SIN": row["HOUR_SIN"],
        "HOUR_COS": row["HOUR_COS"],
        "DAY_OF_WEEK": row["DAY_OF_WEEK"],
        "SYMBOL_ID": symbol_id(symbol),
    }
    context.update(base_values)
    for name in feature_names:
        if name in base_values:
            values[name] = base_values[name]
        elif name.startswith("IS_"):
            clean = symbol.replace("#", "").replace("/", "")
            values[name] = 1.0 if name == f"IS_{clean}" else 0.0
        elif name.endswith("_TREND"):
            timeframe = name.removesuffix("_TREND")
            trend = get_trend(symbol, timeframe)
            if trend is None:
                return None
            values[name] = trend
            context[name] = trend
        else:
            values[name] = 0.0
    feature_df = pd.DataFrame([values], columns=feature_names)
    if feature_df.isna().any(axis=None):
        return None
    return feature_df, float(row["ATR"]), row["TIME_DT"], context


def load_xgb_model(path: Path):
    if not path.exists():
        return None
    model = xgb.XGBClassifier()
    model.load_model(str(path))
    model.set_params(device="cpu")
    return model


def prepare_missing_models() -> None:
    if SILVER_MODEL_FILE.exists() and SHARED_MODEL_FILE.exists():
        print(f"Silver model exists: {SILVER_MODEL_FILE}")
        print(f"Shared model exists: {SHARED_MODEL_FILE}")
        return
    from precious_metals_axis_research.train_selected_profile_models import (  # noqa: PLC0415
        train_silver,
        train_xaueur_shared,
    )

    if not SILVER_MODEL_FILE.exists():
        train_silver()
    if not SHARED_MODEL_FILE.exists():
        train_xaueur_shared()


def get_strategy_positions(strategy: StrategyConfig | None = None):
    positions = mt5.positions_get()
    if positions is None:
        return []
    allowed_magics = {item.magic for item in STRATEGIES}
    rows = [pos for pos in positions if getattr(pos, "magic", None) in allowed_magics]
    if strategy is not None:
        rows = [
            pos
            for pos in rows
            if pos.symbol == strategy.symbol and pos.magic == strategy.magic
        ]
    return rows


def get_daily_realized_pnl() -> float:
    now = datetime.now()
    day_start = datetime(now.year, now.month, now.day)
    deals = mt5.history_deals_get(day_start, now)
    if deals is None:
        return 0.0
    magics = {item.magic for item in STRATEGIES}
    return float(
        sum(
            deal.profit + deal.swap + deal.commission
            for deal in deals
            if getattr(deal, "magic", None) in magics
        )
    )


def current_spread(symbol: str):
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if tick is None or info is None or info.point <= 0:
        return None
    spread_price = max(float(tick.ask - tick.bid), 0.0)
    return spread_price, spread_price / float(info.point)


def normalize_volume(symbol: str, raw_volume: float):
    info = mt5.symbol_info(symbol)
    if info is None:
        return None
    if raw_volume < info.volume_min:
        return None
    capped = min(raw_volume, info.volume_max)
    steps = math.floor(capped / info.volume_step)
    volume = steps * info.volume_step
    precision = max(0, int(round(-math.log10(info.volume_step)))) if info.volume_step < 1 else 0
    return round(volume, precision)


def calc_volume_by_risk(symbol: str, risk_budget: float, sl_distance: float):
    info = mt5.symbol_info(symbol)
    if info is None or info.trade_tick_size <= 0 or info.trade_tick_value <= 0:
        return None, 0.0
    money_per_lot = (sl_distance / info.trade_tick_size) * info.trade_tick_value
    if money_per_lot <= 0:
        return None, money_per_lot
    raw_volume = risk_budget / money_per_lot
    return normalize_volume(symbol, raw_volume), raw_volume


def close_position(position) -> bool:
    tick = mt5.symbol_info_tick(position.symbol)
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
        "symbol": position.symbol,
        "volume": position.volume,
        "type": order_type,
        "position": position.ticket,
        "price": price,
        "magic": position.magic,
        "comment": "MultiMetal close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE


def manage_positions() -> None:
    by_magic = {strategy.magic: strategy for strategy in STRATEGIES}
    for position in get_strategy_positions():
        strategy = by_magic.get(position.magic)
        if strategy is None:
            continue
        held_seconds = time.time() - position.time
        if held_seconds >= strategy.max_hold_bars * strategy.bar_minutes * 60:
            if close_position(position):
                print(f"Timeout close: {position.symbol} ticket={position.ticket}")


def choose_signal(strategy: StrategyConfig, probs: np.ndarray):
    buy_prob = float(probs[1])
    sell_prob = float(probs[2])
    signal = 1 if buy_prob >= sell_prob else 2
    confidence = buy_prob if signal == 1 else sell_prob
    edge = abs(buy_prob - sell_prob)
    if strategy.direction_mode == "long" and signal != 1:
        return None, buy_prob, sell_prob, confidence, edge
    if strategy.direction_mode == "short" and signal != 2:
        return None, buy_prob, sell_prob, confidence, edge
    if confidence < strategy.threshold or edge < strategy.edge_threshold:
        return None, buy_prob, sell_prob, confidence, edge
    return signal, buy_prob, sell_prob, confidence, edge


def check_regime_filter(strategy: StrategyConfig, context: dict, spread_atr: float):
    if strategy.regime_trend_min is not None:
        trend_values = [context.get(name) for name in REGIME_TREND_COLUMNS]
        if any(value is None for value in trend_values):
            return False, "regime trend unavailable"
        trend_score = float(np.mean(trend_values))
        if trend_score < strategy.regime_trend_min:
            return False, f"trend_score {trend_score:.2f} < {strategy.regime_trend_min:.2f}"
    if strategy.regime_rsi_min is not None:
        rsi = float(context.get("BASE_RSI", np.nan))
        if not np.isfinite(rsi) or rsi < strategy.regime_rsi_min:
            return False, f"rsi {rsi:.2f} < {strategy.regime_rsi_min:.2f}"
    if strategy.regime_rsi_max is not None:
        rsi = float(context.get("BASE_RSI", np.nan))
        if not np.isfinite(rsi) or rsi > strategy.regime_rsi_max:
            return False, f"rsi {rsi:.2f} > {strategy.regime_rsi_max:.2f}"
    if strategy.regime_vola_max is not None:
        vola = float(context.get("VOLA_RATIO", np.nan))
        if not np.isfinite(vola) or vola > strategy.regime_vola_max:
            return False, f"vola_ratio {vola:.2f} > {strategy.regime_vola_max:.2f}"
    if strategy.regime_macd_min is not None:
        macd = float(context.get("MACD_ATR", np.nan))
        if not np.isfinite(macd) or macd < strategy.regime_macd_min:
            return False, f"macd_atr {macd:.2f} < {strategy.regime_macd_min:.2f}"
    if strategy.regime_spread_atr_stress_max is not None:
        stressed_spread_atr = spread_atr * 3.0
        if stressed_spread_atr > strategy.regime_spread_atr_stress_max:
            return (
                False,
                f"stressed_spread_atr {stressed_spread_atr:.4f} "
                f"> {strategy.regime_spread_atr_stress_max:.4f}",
            )
    return True, ""


def build_features(strategy: StrategyConfig, model):
    if strategy.model_kind == "gold_m1":
        return build_gold_features(strategy.symbol)
    feature_names = model.get_booster().feature_names
    if not feature_names:
        return None
    return build_h1_features(strategy.symbol, feature_names)


def execute_entry(strategy: StrategyConfig, signal: int, volume: float, sl_distance: float, tp_distance: float):
    tick = mt5.symbol_info_tick(strategy.symbol)
    if tick is None:
        return None
    order_type = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL
    price = tick.ask if signal == 1 else tick.bid
    sl = price - sl_distance if signal == 1 else price + sl_distance
    tp = price + tp_distance if signal == 1 else price - tp_distance
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": strategy.symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": strategy.magic,
        "comment": f"MultiMetal {strategy.name}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    return mt5.order_send(request)


def evaluate_strategy(strategy: StrategyConfig, model, account, live: bool) -> None:
    now = datetime.now()
    row = {
        "time": now.isoformat(timespec="seconds"),
        "strategy": strategy.name,
        "symbol": strategy.symbol,
        "threshold": strategy.threshold,
        "balance": round(float(account.balance), 2),
        "equity": round(float(account.equity), 2),
    }

    if not strategy.enabled:
        row["status"] = "disabled"
        append_log(row)
        return
    if model is None:
        row["status"] = "model_missing"
        row["message"] = str(strategy.model_path)
        append_log(row)
        print(f"Skip {strategy.name}: model missing {strategy.model_path}")
        return
    symbol_positions = get_strategy_positions(strategy)
    if len(get_strategy_positions()) >= MAX_TOTAL_POSITIONS:
        row["status"] = "max_total_positions"
        append_log(row)
        return
    if len(symbol_positions) >= MAX_POSITIONS_PER_SYMBOL:
        row["status"] = "position_exists"
        append_log(row)
        return

    feature_pack = build_features(strategy, model)
    if feature_pack is None:
        row["status"] = "features_unavailable"
        append_log(row)
        return
    if len(feature_pack) == 3:
        feature_df, atr, bar_time = feature_pack
        context = {}
    else:
        feature_df, atr, bar_time, context = feature_pack
    probs = model.predict_proba(feature_df)[0]
    signal, buy_prob, sell_prob, confidence, edge = choose_signal(strategy, probs)
    row.update(
        {
            "buy_prob": round(buy_prob, 6),
            "sell_prob": round(sell_prob, 6),
            "confidence": round(confidence, 6),
            "edge": round(edge, 6),
            "atr": round(atr, 6),
        }
    )
    if signal is None:
        row["status"] = "no_signal"
        append_log(row)
        print(
            f"{strategy.symbol} {strategy.name}: no signal "
            f"buy={buy_prob:.3f} sell={sell_prob:.3f} edge={edge:.3f}"
        )
        return

    spread = current_spread(strategy.symbol)
    if spread is None:
        row["status"] = "spread_unavailable"
        append_log(row)
        return
    spread_price, spread_points = spread
    spread_atr = spread_price / max(atr, 1e-9)
    row["spread_points"] = round(spread_points, 2)
    row["spread_atr"] = round(spread_atr, 6)
    if strategy.max_spread_points is not None and spread_points > strategy.max_spread_points:
        row["status"] = "spread_points_too_wide"
        row["message"] = f"{spread_points:.1f} > {strategy.max_spread_points:.1f}"
        append_log(row)
        return
    if spread_atr > strategy.max_spread_atr:
        row["status"] = "spread_atr_too_wide"
        row["message"] = f"{spread_atr:.4f} > {strategy.max_spread_atr:.4f}"
        append_log(row)
        return

    regime_ok, regime_message = check_regime_filter(strategy, context, spread_atr)
    if not regime_ok:
        row["status"] = "regime_filter_blocked"
        row["message"] = regime_message
        append_log(row)
        print(f"{strategy.symbol} {strategy.name}: regime blocked ({regime_message})")
        return

    sl_distance = max(atr * strategy.sl_atr, spread_price * 2.0)
    tp_distance = max(atr * strategy.tp_atr, spread_price * 2.5)
    risk_budget = float(account.balance) * strategy.risk_per_trade
    volume, raw_volume = calc_volume_by_risk(strategy.symbol, risk_budget, sl_distance)
    row.update(
        {
            "signal": "BUY" if signal == 1 else "SELL",
            "sl_distance": round(sl_distance, 6),
            "tp_distance": round(tp_distance, 6),
            "risk_budget": round(risk_budget, 4),
            "raw_volume": round(raw_volume, 6),
            "volume": "" if volume is None else volume,
        }
    )
    if volume is None:
        row["status"] = "volume_below_minimum"
        append_log(row)
        print(f"{strategy.symbol} {strategy.name}: volume below minimum")
        return

    if not live:
        row["status"] = "dry_run_signal"
        append_log(row)
        print(
            f"DRY {row['signal']} {strategy.symbol} {strategy.name} "
            f"vol={volume} conf={confidence:.3f} edge={edge:.3f} "
            f"SL={sl_distance:.4f} TP={tp_distance:.4f} bar={bar_time}"
        )
        return

    result = execute_entry(strategy, signal, volume, sl_distance, tp_distance)
    row["retcode"] = "" if result is None else getattr(result, "retcode", "")
    row["order"] = "" if result is None else getattr(result, "order", "")
    row["deal"] = "" if result is None else getattr(result, "deal", "")
    row["message"] = "mt5.order_send returned None" if result is None else getattr(result, "comment", "")
    row["status"] = (
        "order_opened"
        if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        else "order_failed"
    )
    append_log(row)
    print(f"{row['status']}: {row['signal']} {strategy.symbol} vol={volume} ret={row['retcode']}")


def load_models():
    models = {}
    for strategy in STRATEGIES:
        if strategy.name in models:
            continue
        models[strategy.name] = load_xgb_model(strategy.model_path)
    return models


def run_once(live: bool) -> int:
    if not mt5.initialize():
        print("MT5 initialize failed")
        return 1
    try:
        account = mt5.account_info()
        if account is None or account.balance <= 0:
            print("Account unavailable or invalid.")
            return 1
        daily_pnl = get_daily_realized_pnl()
        if daily_pnl <= -float(account.balance) * MAX_DAILY_LOSS_PCT:
            print(f"Daily loss guard active: {daily_pnl:.2f}")
            return 0
        manage_positions()
        models = load_models()
        for strategy in STRATEGIES:
            evaluate_strategy(strategy, models.get(strategy.name), account, live=live)
        return 0
    finally:
        mt5.shutdown()


def run_loop(live: bool) -> int:
    print(
        "Multi precious metals trader active. "
        f"live={live}; log={LOG_FILE}"
    )
    while True:
        run_once(live=live)
        time.sleep(SLEEP_SECONDS)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Research/live candidate multi precious metals trader."
    )
    parser.add_argument("--prepare-models", action="store_true", help="Train missing live-candidate models.")
    parser.add_argument("--once", action="store_true", help="Evaluate all strategies once and exit.")
    parser.add_argument("--live", action="store_true", help="Send real MT5 orders. Omit for dry-run.")
    args = parser.parse_args()

    if args.prepare_models:
        prepare_missing_models()
    if args.once:
        return run_once(live=args.live)
    if args.prepare_models:
        return 0
    return run_loop(live=args.live)


if __name__ == "__main__":
    raise SystemExit(main())
