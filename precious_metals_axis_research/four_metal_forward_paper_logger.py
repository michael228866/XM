from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
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


SIGNAL_LOG_FILE = RESEARCH_DIR / "four_metal_forward_signal_log.csv"
TRADE_LOG_FILE = RESEARCH_DIR / "four_metal_forward_trade_log.csv"
STATE_FILE = RESEARCH_DIR / "four_metal_forward_state.json"

POLL_SECONDS = 5
LOG_TIMEZONE = timezone.utc

GOLD_MODEL_FILE = PROJECT_ROOT / "gold_short_recent_candidate_xgb.json"
GOLD_CANDIDATE_FILE = PROJECT_ROOT / "gold_short_recent_walk_forward.json"
SILVER_MODEL_FILE = RESEARCH_DIR / "silver_h1_regime_selected_xgb.json"
SILVER_METADATA_FILE = RESEARCH_DIR / "silver_h1_regime_selected_xgb.metadata.json"
XPT_MODEL_FILE = RESEARCH_DIR / "xpt_h4_fold_coverage_xgb.json"
XPT_METADATA_FILE = RESEARCH_DIR / "xpt_h4_fold_coverage_xgb.metadata.json"
XPD_MODEL_FILE = RESEARCH_DIR / "xpd_h12_alternate_target_xgb.json"
XPD_METADATA_FILE = RESEARCH_DIR / "xpd_h12_alternate_target_xgb.metadata.json"

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
GOLD_FEATURES = GOLD_BASE_FEATURES + [f"{timeframe}_TREND" for timeframe in MTF_ORDER]

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

SIGNAL_FIELDS = [
    "event_time",
    "symbol",
    "timeframe",
    "bar_time",
    "status",
    "reason",
    "direction",
    "confidence",
    "buy_prob",
    "sell_prob",
    "edge",
    "threshold",
    "edge_threshold",
    "close",
    "atr",
    "base_rsi",
    "vola_ratio",
    "macd_atr",
    "trend_score",
    "spread_points",
    "spread_price",
    "spread_atr",
    "tp_distance",
    "sl_distance",
    "max_hold",
    "open_position_id",
]

TRADE_FIELDS = [
    "event_time",
    "paper_trade_id",
    "symbol",
    "timeframe",
    "action",
    "reason",
    "direction",
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "tp_distance",
    "sl_distance",
    "spread_price",
    "hold_bars",
    "reward_r",
    "confidence",
    "buy_prob",
    "sell_prob",
    "edge",
]


def utc_now() -> datetime:
    return datetime.now(LOG_TIMEZONE)


def format_time(value) -> str:
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
    if value.tzinfo is None:
        value = value.replace(tzinfo=LOG_TIMEZONE)
    else:
        value = value.astimezone(LOG_TIMEZONE)
    return value.isoformat(timespec="seconds")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def append_csv(path: Path, fields: list[str], row: dict) -> None:
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if needs_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def load_state() -> dict:
    if not STATE_FILE.exists() or STATE_FILE.stat().st_size == 0:
        return {"last_bar_times": {}, "positions": []}
    with STATE_FILE.open("r", encoding="utf-8") as file:
        state = json.load(file)
    state.setdefault("last_bar_times", {})
    state.setdefault("positions", [])
    return state


def save_state(state: dict) -> None:
    tmp_file = STATE_FILE.with_suffix(".tmp")
    tmp_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp_file, STATE_FILE)


def load_model(path: Path) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier()
    model.load_model(str(path))
    model.set_params(device="cpu")
    return model


def build_configs() -> list[dict]:
    gold_params = load_json(GOLD_CANDIDATE_FILE)["selected"]["params"]
    silver_meta = load_json(SILVER_METADATA_FILE)
    xpt_meta = load_json(XPT_METADATA_FILE)
    xpd_meta = load_json(XPD_METADATA_FILE)
    return [
        {
            "symbol": "GOLD#",
            "timeframe": "M1",
            "feature_mode": "gold",
            "model_path": GOLD_MODEL_FILE,
            "features": GOLD_FEATURES,
            "params": {
                **gold_params,
                "direction_mode": "short",
                "exit_mode": "tp_sl",
            },
        },
        {
            "symbol": "SILVER#",
            "timeframe": silver_meta["timeframe"],
            "feature_mode": "axis",
            "model_path": SILVER_MODEL_FILE,
            "features": silver_meta["features"],
            "params": {
                **silver_meta["strategy_params"],
                "exit_mode": "tp_sl",
            },
        },
        {
            "symbol": xpt_meta["symbol"],
            "timeframe": xpt_meta["timeframe"],
            "feature_mode": "axis",
            "model_path": XPT_MODEL_FILE,
            "features": xpt_meta["features"],
            "params": {
                **xpt_meta["strategy_params"],
                "exit_mode": "tp_sl",
            },
        },
        {
            "symbol": xpd_meta["symbol"],
            "timeframe": xpd_meta["timeframe"],
            "feature_mode": "axis",
            "model_path": XPD_MODEL_FILE,
            "features": xpd_meta["features"],
            "params": xpd_meta["strategy_params"],
        },
    ]


def copy_rates(symbol: str, timeframe: str, count: int, start_pos: int = 1):
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME_MAP[timeframe], start_pos, count)
    if rates is None or len(rates) == 0:
        return None
    frame = pd.DataFrame(rates)
    frame["TIME_DT"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    frame.columns = [column.upper() for column in frame.columns]
    return frame.sort_values("TIME_DT").reset_index(drop=True)


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    true_range = pd.concat(
        [
            (frame["HIGH"] - frame["LOW"]).abs(),
            (frame["HIGH"] - frame["CLOSE"].shift()).abs(),
            (frame["LOW"] - frame["CLOSE"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["ATR"] = true_range.rolling(14).mean()
    frame["HOUR_SIN"] = np.sin(2 * np.pi * frame["TIME_DT"].dt.hour / 24)
    frame["HOUR_COS"] = np.cos(2 * np.pi * frame["TIME_DT"].dt.hour / 24)
    frame["DAY_OF_WEEK"] = frame["TIME_DT"].dt.dayofweek / 7.0
    ema12 = frame["CLOSE"].ewm(span=12, adjust=False).mean()
    ema26 = frame["CLOSE"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    frame["MACD_HIST"] = macd - macd.ewm(span=9, adjust=False).mean()
    ma20 = frame["CLOSE"].rolling(20).mean()
    frame["BB_WIDTH"] = (frame["CLOSE"].rolling(20).std() * 4) / (ma20 + 1e-9)
    frame["BIAS_20"] = (frame["CLOSE"] - ma20) / (ma20 + 1e-9)
    frame["ROC_5"] = frame["CLOSE"].pct_change(5)
    candle_range = frame["HIGH"] - frame["LOW"] + 1e-9
    frame["BODY_PCT"] = (frame["CLOSE"] - frame["OPEN"]).abs() / candle_range
    frame["ATR_PCT"] = frame["ATR"] / (frame["CLOSE"].abs() + 1e-9)
    frame["MACD_ATR"] = frame["MACD_HIST"] / (frame["ATR"] + 1e-9)
    return frame


def add_rsi(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    diff = frame["CLOSE"].diff()
    gain = diff.where(diff > 0, 0).rolling(14).mean()
    loss = (-diff.where(diff < 0, 0)).rolling(14).mean()
    frame[column] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    return frame


def latest_trend(symbol: str, timeframe: str, bar_time: pd.Timestamp) -> float:
    frame = copy_rates(symbol, timeframe, 80)
    if frame is None or len(frame) < 20:
        return 0.0
    frame = add_indicators(frame)
    frame["TREND"] = np.where(
        frame["CLOSE"] > frame["CLOSE"].rolling(20).mean(),
        1.0,
        -1.0,
    )
    frame["TREND"] = frame["TREND"].shift(1)
    frame = frame[frame["TIME_DT"] <= bar_time].dropna(subset=["TREND"])
    if frame.empty:
        return 0.0
    return float(frame["TREND"].iloc[-1])


def build_gold_snapshot(config: dict) -> dict | None:
    base = copy_rates(config["symbol"], "M1", 420)
    if base is None or len(base) < 260:
        return None
    base = add_indicators(base)
    base = add_rsi(base, "M1_RSI")
    base["VOLA_MA"] = base["ATR"].rolling(240).mean()
    base["VOLA_RATIO"] = base["ATR"] / (base["VOLA_MA"] + 1e-9)
    row = base.iloc[-1].copy()
    feature_row = {name: row[name] for name in GOLD_BASE_FEATURES}
    for timeframe in MTF_ORDER:
        feature_row[f"{timeframe}_TREND"] = latest_trend(
            config["symbol"],
            timeframe,
            row["TIME_DT"],
        )
    frame = pd.DataFrame([feature_row], columns=config["features"])
    if frame.isna().any(axis=None):
        return None
    return {
        "features": frame,
        "bar_time": row["TIME_DT"],
        "close": float(row["CLOSE"]),
        "atr": float(row["ATR"]),
        "base_rsi": float(row["M1_RSI"]),
        "vola_ratio": float(row["VOLA_RATIO"]),
        "macd_atr": float(row["MACD_HIST"] / (row["ATR"] + 1e-9)),
        "trend_score": float(np.mean([feature_row[f"{tf}_TREND"] for tf in MTF_ORDER])),
    }


def build_axis_snapshot(config: dict) -> dict | None:
    symbol = config["symbol"]
    base = copy_rates(symbol, config["timeframe"], 520)
    if base is None or len(base) < 260:
        return None
    base = add_indicators(base)
    base = add_rsi(base, "BASE_RSI")
    base["VOLA_MA"] = base["ATR"].rolling(240).mean()
    base["VOLA_RATIO"] = base["ATR"] / (base["VOLA_MA"] + 1e-9)
    features = config["features"]
    base_feature_names = [name for name in features if not name.endswith("_TREND")]
    base[base_feature_names] = base[base_feature_names].shift(1)
    row = base.iloc[-1].copy()
    feature_row = {name: row[name] for name in base_feature_names}
    trend_values = []
    for name in features:
        if not name.endswith("_TREND"):
            continue
        timeframe = name[: -len("_TREND")]
        trend = latest_trend(symbol, timeframe, row["TIME_DT"])
        feature_row[name] = trend
        trend_values.append(trend)
    frame = pd.DataFrame([feature_row], columns=features)
    if frame.isna().any(axis=None):
        return None
    return {
        "features": frame,
        "bar_time": row["TIME_DT"],
        "close": float(row["CLOSE"]),
        "atr": float(row["ATR"]),
        "base_rsi": float(row["BASE_RSI"]),
        "vola_ratio": float(row["VOLA_RATIO"]),
        "macd_atr": float(row["MACD_ATR"]),
        "trend_score": float(np.mean(trend_values)) if trend_values else 0.0,
    }


def get_spread(symbol: str, fallback_frame: pd.DataFrame | None = None) -> tuple[float, float]:
    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    point = float(info.point) if info is not None and info.point > 0 else 0.01
    if tick is not None and tick.ask > 0 and tick.bid > 0:
        spread_price = max(float(tick.ask - tick.bid), 0.0)
        return spread_price / point, spread_price
    if fallback_frame is not None and "SPREAD" in fallback_frame.columns:
        spread_points = float(fallback_frame["SPREAD"].iloc[-1])
        return spread_points, spread_points * point
    return 0.0, 0.0


def direction_from_probs(probs: np.ndarray, params: dict) -> tuple[int, float, str]:
    buy_prob = float(probs[1])
    sell_prob = float(probs[2])
    signal = 1 if buy_prob >= sell_prob else 2
    confidence = buy_prob if signal == 1 else sell_prob
    direction = "long" if signal == 1 else "short"
    mode = params.get("direction_mode", "both")
    if mode == "long" and signal == 2:
        return signal, confidence, "short_blocked"
    if mode == "short" and signal == 1:
        return signal, confidence, "long_blocked"
    return signal, confidence, direction


def passes_filter(signal: int, snapshot: dict, spread_atr: float, params: dict) -> tuple[bool, str]:
    regime = params.get("regime_filter")
    if regime:
        if snapshot["trend_score"] < float(regime.get("trend_min", -99.0)):
            return False, "trend_filter"
        if not (
            float(regime.get("rsi_min", 0.0))
            <= snapshot["base_rsi"]
            <= float(regime.get("rsi_max", 100.0))
        ):
            return False, "rsi_filter"
        if snapshot["vola_ratio"] > float(regime.get("vola_max", 99.0)):
            return False, "vola_filter"
        if spread_atr > float(regime.get("stress_spread_atr_max", 99.0)):
            return False, "spread_atr_filter"

    if "rsi_min" not in params:
        return True, ""
    if not (params["rsi_min"] <= snapshot["base_rsi"] <= params["rsi_max"]):
        return False, "rsi_filter"
    if snapshot["vola_ratio"] > params["vola_max"]:
        return False, "vola_filter"
    if spread_atr > params["spread_atr_max"]:
        return False, "spread_atr_filter"

    trend = snapshot["trend_score"]
    if params["trend_mode"] == "aligned":
        if signal == 1 and trend < 0:
            return False, "trend_filter"
        if signal == 2 and trend > 0:
            return False, "trend_filter"
    elif params["trend_mode"] == "counter":
        if signal == 1 and trend > 0:
            return False, "trend_filter"
        if signal == 2 and trend < 0:
            return False, "trend_filter"

    macd = snapshot["macd_atr"]
    if params["macd_mode"] == "aligned":
        if signal == 1 and macd < 0:
            return False, "macd_filter"
        if signal == 2 and macd > 0:
            return False, "macd_filter"
    return True, ""


def find_position(state: dict, symbol: str, timeframe: str) -> dict | None:
    for position in state["positions"]:
        if position["symbol"] == symbol and position["timeframe"] == timeframe:
            return position
    return None


def update_position(state: dict, position: dict, snapshot: dict) -> bool:
    direction = position["direction"]
    gross = (
        snapshot["close"] - position["entry_price"]
        if direction == "long"
        else position["entry_price"] - snapshot["close"]
    )
    position["hold_bars"] += 1
    exit_mode = position.get("exit_mode", "tp_sl")
    if exit_mode == "time_stop":
        exit_now = gross <= -position["sl_distance"] or position["hold_bars"] >= position["max_hold"]
    else:
        exit_now = (
            gross >= position["tp_distance"]
            or gross <= -position["sl_distance"]
            or position["hold_bars"] >= position["max_hold"]
        )
    if not exit_now:
        return False

    if gross >= position["tp_distance"] and exit_mode != "time_stop":
        reason = "tp"
    elif gross <= -position["sl_distance"]:
        reason = "sl"
    else:
        reason = "time_stop"
    reward_r = (gross - position["spread_price"]) / max(position["sl_distance"], 1e-9)
    append_csv(
        TRADE_LOG_FILE,
        TRADE_FIELDS,
        {
            "event_time": format_time(utc_now()),
            "paper_trade_id": position["id"],
            "symbol": position["symbol"],
            "timeframe": position["timeframe"],
            "action": "close",
            "reason": reason,
            "direction": direction,
            "entry_time": position["entry_time"],
            "exit_time": format_time(snapshot["bar_time"]),
            "entry_price": round(position["entry_price"], 6),
            "exit_price": round(snapshot["close"], 6),
            "tp_distance": round(position["tp_distance"], 6),
            "sl_distance": round(position["sl_distance"], 6),
            "spread_price": round(position["spread_price"], 6),
            "hold_bars": position["hold_bars"],
            "reward_r": round(reward_r, 6),
            "confidence": position.get("confidence", ""),
            "buy_prob": position.get("buy_prob", ""),
            "sell_prob": position.get("sell_prob", ""),
            "edge": position.get("edge", ""),
        },
    )
    state["positions"] = [item for item in state["positions"] if item["id"] != position["id"]]
    return True


def open_position(
    state: dict,
    config: dict,
    snapshot: dict,
    direction: str,
    spread_price: float,
    probs: np.ndarray,
) -> dict:
    params = config["params"]
    trade_id = (
        f"{config['symbol'].replace('#', '')}-"
        f"{config['timeframe']}-{format_time(snapshot['bar_time']).replace(':', '')}"
    )
    position = {
        "id": trade_id,
        "symbol": config["symbol"],
        "timeframe": config["timeframe"],
        "direction": direction,
        "entry_time": format_time(snapshot["bar_time"]),
        "entry_price": snapshot["close"],
        "tp_distance": max(snapshot["atr"] * float(params["tp_atr"]), 1e-9),
        "sl_distance": max(snapshot["atr"] * float(params["sl_atr"]), 1e-9),
        "spread_price": spread_price,
        "max_hold": int(params["max_hold"]),
        "exit_mode": params.get("exit_mode", "tp_sl"),
        "hold_bars": 0,
        "confidence": round(max(float(probs[1]), float(probs[2])), 6),
        "buy_prob": round(float(probs[1]), 6),
        "sell_prob": round(float(probs[2]), 6),
        "edge": round(abs(float(probs[1]) - float(probs[2])), 6),
    }
    state["positions"].append(position)
    append_csv(
        TRADE_LOG_FILE,
        TRADE_FIELDS,
        {
            "event_time": format_time(utc_now()),
            "paper_trade_id": trade_id,
            "symbol": config["symbol"],
            "timeframe": config["timeframe"],
            "action": "open",
            "reason": "signal",
            "direction": direction,
            "entry_time": position["entry_time"],
            "entry_price": round(position["entry_price"], 6),
            "tp_distance": round(position["tp_distance"], 6),
            "sl_distance": round(position["sl_distance"], 6),
            "spread_price": round(spread_price, 6),
            "hold_bars": 0,
            "confidence": position["confidence"],
            "buy_prob": position["buy_prob"],
            "sell_prob": position["sell_prob"],
            "edge": position["edge"],
        },
    )
    return position


def process_config(config: dict, model, state: dict) -> None:
    snapshot = (
        build_gold_snapshot(config)
        if config["feature_mode"] == "gold"
        else build_axis_snapshot(config)
    )
    event_time = format_time(utc_now())
    if snapshot is None:
        append_csv(
            SIGNAL_LOG_FILE,
            SIGNAL_FIELDS,
            {
                "event_time": event_time,
                "symbol": config["symbol"],
                "timeframe": config["timeframe"],
                "status": "feature_unavailable",
                "reason": "feature_snapshot_none",
            },
        )
        return

    key = f"{config['symbol']}|{config['timeframe']}"
    bar_time_text = format_time(snapshot["bar_time"])
    if state["last_bar_times"].get(key) == bar_time_text:
        return
    state["last_bar_times"][key] = bar_time_text

    existing = find_position(state, config["symbol"], config["timeframe"])
    had_position = existing is not None
    closed = False
    if existing is not None:
        closed = update_position(state, existing, snapshot)

    probs = model.predict_proba(snapshot["features"])[0]
    signal, confidence, direction = direction_from_probs(probs, config["params"])
    buy_prob = float(probs[1])
    sell_prob = float(probs[2])
    edge = abs(buy_prob - sell_prob)
    spread_points, spread_price = get_spread(config["symbol"])
    spread_atr = spread_price / max(snapshot["atr"], 1e-9)
    filter_ok, filter_reason = passes_filter(signal, snapshot, spread_atr, config["params"])
    threshold = float(config["params"]["threshold"])
    edge_threshold = float(config["params"]["edge_threshold"])
    has_signal = (
        direction in {"long", "short"}
        and confidence >= threshold
        and edge >= edge_threshold
        and filter_ok
    )

    open_position_id = ""
    if had_position:
        status = "position_closed" if closed else "position_open"
        reason = "closed_this_bar" if closed else "already_open"
    elif has_signal:
        position = open_position(
            state,
            config,
            snapshot,
            direction,
            spread_price,
            probs,
        )
        open_position_id = position["id"]
        status = "paper_opened"
        reason = "signal"
    elif direction not in {"long", "short"}:
        status = "direction_blocked"
        reason = direction
    elif confidence < threshold:
        status = "no_signal"
        reason = "threshold"
    elif edge < edge_threshold:
        status = "no_signal"
        reason = "edge"
    else:
        status = "filter_blocked"
        reason = filter_reason

    append_csv(
        SIGNAL_LOG_FILE,
        SIGNAL_FIELDS,
        {
            "event_time": event_time,
            "symbol": config["symbol"],
            "timeframe": config["timeframe"],
            "bar_time": bar_time_text,
            "status": status,
            "reason": reason,
            "direction": direction,
            "confidence": round(confidence, 6),
            "buy_prob": round(buy_prob, 6),
            "sell_prob": round(sell_prob, 6),
            "edge": round(edge, 6),
            "threshold": threshold,
            "edge_threshold": edge_threshold,
            "close": round(snapshot["close"], 6),
            "atr": round(snapshot["atr"], 6),
            "base_rsi": round(snapshot["base_rsi"], 4),
            "vola_ratio": round(snapshot["vola_ratio"], 4),
            "macd_atr": round(snapshot["macd_atr"], 6),
            "trend_score": round(snapshot["trend_score"], 4),
            "spread_points": round(spread_points, 2),
            "spread_price": round(spread_price, 6),
            "spread_atr": round(spread_atr, 6),
            "tp_distance": round(snapshot["atr"] * float(config["params"]["tp_atr"]), 6),
            "sl_distance": round(snapshot["atr"] * float(config["params"]["sl_atr"]), 6),
            "max_hold": int(config["params"]["max_hold"]),
            "open_position_id": open_position_id,
        },
    )
    print(
        f"{bar_time_text} {config['symbol']} {config['timeframe']} "
        f"{status} {reason} buy={buy_prob:.3f} sell={sell_prob:.3f}",
        flush=True,
    )


def main() -> int:
    run_once = "--once" in sys.argv
    configs = build_configs()
    missing = [
        str(config["model_path"])
        for config in configs
        if not Path(config["model_path"]).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing model files. Run "
            "python .\\precious_metals_axis_research\\train_four_metal_paper_models.py "
            f"first. Missing: {missing}"
        )
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    for config in configs:
        if not mt5.symbol_select(config["symbol"], True):
            print(f"Warning: unable to select {config['symbol']}", flush=True)

    models = {config["symbol"]: load_model(config["model_path"]) for config in configs}
    state = load_state()
    print("Four-metal forward paper logger active. No live orders will be sent.", flush=True)
    while True:
        try:
            for config in configs:
                process_config(config, models[config["symbol"]], state)
            save_state(state)
            if run_once:
                return 0
            time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            save_state(state)
            return 0
        except Exception as exc:
            print(f"Error: {exc}", flush=True)
            save_state(state)
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
