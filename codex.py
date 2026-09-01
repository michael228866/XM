from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RESEARCH_DIR = PROJECT_ROOT / "precious_metals_axis_research"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import MetaTrader5 as mt5  # noqa: E402
import numpy as np  # noqa: E402

from precious_metals_axis_research.four_metal_forward_paper_logger import (  # noqa: E402
    TIMEFRAME_MAP,
    build_axis_snapshot,
    build_configs,
    build_gold_snapshot,
    direction_from_probs,
    format_time,
    get_spread,
    load_model,
    passes_filter,
)


STRATEGY_NAME = "Codex Precious Portfolio v1"
MAGIC_NUMBER = 20260609
LOG_TIMEZONE = timezone.utc

SIGNAL_LOG_FILE = PROJECT_ROOT / "codex_signal_log.csv"
TRADE_LOG_FILE = PROJECT_ROOT / "codex_trade_log.csv"
STATE_FILE = PROJECT_ROOT / "codex_state.json"
PROMOTION_PLAN_FILE = RESEARCH_DIR / "four_metal_main_promotion_plan.json"

POLL_SECONDS = 5
PORTFOLIO_RISK_PER_SIGNAL = 0.028
MAX_DAILY_LOSS_PCT = 0.05
MAX_TOTAL_OPEN_POSITIONS = 4
MAX_SYMBOL_OPEN_POSITIONS = 1
LIVE_CONFIRM_ENV = "CODEX_LIVE_CONFIRM"
LIVE_CONFIRM_VALUE = "I_UNDERSTAND"
MT5_TERMINAL_PATH = Path(
    os.environ.get("XM_TERMINAL_PATH", r"D:\XM2\terminal64.exe")
)
MT5_CONNECT_TIMEOUT_MS = 10_000
MAX_SNAPSHOT_AGE_BARS = 3
TIMEFRAME_SECONDS = {
    "M1": 60,
    "H1": 60 * 60,
    "H4": 4 * 60 * 60,
    "H12": 12 * 60 * 60,
}

SIGNAL_FIELDS = [
    "event_time",
    "mode",
    "strategy",
    "symbol",
    "timeframe",
    "role",
    "weight",
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
    "exit_mode",
    "position_id",
    "balance",
    "daily_pnl",
    "risk_budget",
    "risk_per_lot",
    "raw_lot",
    "lot",
    "retcode",
    "order_id",
    "deal_id",
    "request_id",
    "broker_price",
    "broker_bid",
    "broker_ask",
    "broker_comment",
    "message",
]

TRADE_FIELDS = [
    "event_time",
    "mode",
    "strategy",
    "trade_id",
    "symbol",
    "timeframe",
    "role",
    "weight",
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
    "lot",
    "ticket",
    "order_id",
    "deal_id",
    "retcode",
    "broker_comment",
]


def utc_now() -> datetime:
    return datetime.now(LOG_TIMEZONE)


def append_csv(path: Path, fields: list[str], row: dict) -> None:
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if needs_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_state() -> dict:
    if not STATE_FILE.exists() or STATE_FILE.stat().st_size == 0:
        return {"last_bar_times": {}, "positions": []}
    state = load_json(STATE_FILE)
    state.setdefault("last_bar_times", {})
    state.setdefault("positions", [])
    return state


def save_state(state: dict) -> None:
    tmp_file = STATE_FILE.with_suffix(".tmp")
    tmp_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp_file, STATE_FILE)


def config_key(config: dict) -> str:
    return f"{config['symbol']}|{config['timeframe']}"


def build_portfolio_configs() -> list[dict]:
    promotion = load_json(PROMOTION_PLAN_FILE)["symbols"]
    configs = build_configs()
    for config in configs:
        promoted = promotion[config["symbol"]]
        config["role"] = promoted["role"]
        config["weight"] = float(promoted["allocation_weight"])
        config["status"] = promoted["status"]
    return sorted(configs, key=lambda item: item["weight"], reverse=True)


def build_snapshot(config: dict) -> dict | None:
    if config["feature_mode"] == "gold":
        return build_gold_snapshot(config)
    return build_axis_snapshot(config)


def normalize_direction_probs(probs: np.ndarray, direction_mode: str) -> np.ndarray:
    if len(probs) == 3:
        return probs
    if len(probs) == 2 and direction_mode == "short":
        short_prob = float(probs[1])
        return np.array([1.0 - short_prob, 0.0, short_prob], dtype=np.float32)
    raise ValueError(
        f"Unsupported probability shape={len(probs)} for direction={direction_mode}"
    )


def find_state_position(state: dict, config: dict, mode: str) -> dict | None:
    key = config_key(config)
    for position in state["positions"]:
        if position.get("key") == key and position.get("mode") == mode:
            return position
    return None


def remove_state_position(state: dict, trade_id: str) -> None:
    state["positions"] = [
        position for position in state["positions"] if position["id"] != trade_id
    ]


def state_position_count(state: dict, mode: str) -> int:
    return sum(1 for position in state["positions"] if position.get("mode") == mode)


def state_symbol_position_count(state: dict, config: dict, mode: str) -> int:
    key = config_key(config)
    return sum(
        1
        for position in state["positions"]
        if position.get("key") == key and position.get("mode") == mode
    )


def get_strategy_positions(symbol: str | None = None):
    positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    if positions is None:
        return []
    return [position for position in positions if position.magic == MAGIC_NUMBER]


def live_total_position_count() -> int:
    return len(get_strategy_positions())


def live_symbol_position_count(symbol: str) -> int:
    return len(get_strategy_positions(symbol))


def paper_position_gate(state: dict, config: dict) -> tuple[bool, str]:
    if state_symbol_position_count(state, config, "paper") >= MAX_SYMBOL_OPEN_POSITIONS:
        return False, "symbol_position_limit"
    if state_position_count(state, "paper") >= MAX_TOTAL_OPEN_POSITIONS:
        return False, "portfolio_position_limit"
    return True, "ok"


def live_position_gate(config: dict) -> tuple[bool, str]:
    if live_symbol_position_count(config["symbol"]) >= MAX_SYMBOL_OPEN_POSITIONS:
        return False, "symbol_position_limit"
    if live_total_position_count() >= MAX_TOTAL_OPEN_POSITIONS:
        return False, "portfolio_position_limit"
    return True, "ok"


def reward_r(position: dict, close_price: float) -> float:
    if position["direction"] == "long":
        gross = close_price - position["entry_price"]
    else:
        gross = position["entry_price"] - close_price
    return (gross - position["spread_price"]) / max(position["sl_distance"], 1e-9)


def should_close_position(position: dict, close_price: float) -> tuple[bool, str]:
    if position["direction"] == "long":
        gross = close_price - position["entry_price"]
    else:
        gross = position["entry_price"] - close_price

    exit_mode = position.get("exit_mode", "tp_sl")
    hit_sl = gross <= -position["sl_distance"]
    hit_tp = gross >= position["tp_distance"] and exit_mode != "time_stop"
    hit_time = position["hold_bars"] >= position["max_hold"]

    if hit_tp:
        return True, "tp"
    if hit_sl:
        return True, "sl"
    if hit_time:
        return True, "time_stop"
    return False, ""


def update_paper_position(state: dict, position: dict, snapshot: dict) -> bool:
    position["hold_bars"] += 1
    close_now, reason = should_close_position(position, snapshot["close"])
    if not close_now:
        return False

    append_csv(
        TRADE_LOG_FILE,
        TRADE_FIELDS,
        {
            "event_time": format_time(utc_now()),
            "mode": "paper",
            "strategy": STRATEGY_NAME,
            "trade_id": position["id"],
            "symbol": position["symbol"],
            "timeframe": position["timeframe"],
            "role": position["role"],
            "weight": position["weight"],
            "action": "close",
            "reason": reason,
            "direction": position["direction"],
            "entry_time": position["entry_time"],
            "exit_time": format_time(snapshot["bar_time"]),
            "entry_price": round(position["entry_price"], 6),
            "exit_price": round(snapshot["close"], 6),
            "tp_distance": round(position["tp_distance"], 6),
            "sl_distance": round(position["sl_distance"], 6),
            "spread_price": round(position["spread_price"], 6),
            "hold_bars": position["hold_bars"],
            "reward_r": round(reward_r(position, snapshot["close"]), 6),
            "confidence": position.get("confidence", ""),
            "buy_prob": position.get("buy_prob", ""),
            "sell_prob": position.get("sell_prob", ""),
            "edge": position.get("edge", ""),
        },
    )
    remove_state_position(state, position["id"])
    return True


def open_paper_position(
    state: dict,
    config: dict,
    snapshot: dict,
    direction: str,
    spread_price: float,
    probs: np.ndarray,
) -> dict:
    params = config["params"]
    trade_id = (
        f"codex-{config['symbol'].replace('#', '')}-"
        f"{config['timeframe']}-{format_time(snapshot['bar_time']).replace(':', '')}"
    )
    position = {
        "id": trade_id,
        "mode": "paper",
        "key": config_key(config),
        "symbol": config["symbol"],
        "timeframe": config["timeframe"],
        "role": config["role"],
        "weight": config["weight"],
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
            "mode": "paper",
            "strategy": STRATEGY_NAME,
            "trade_id": trade_id,
            "symbol": config["symbol"],
            "timeframe": config["timeframe"],
            "role": config["role"],
            "weight": config["weight"],
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


def normalize_lot(symbol: str, raw_lot: float) -> float | None:
    info = mt5.symbol_info(symbol)
    if info is None or raw_lot <= 0:
        return None
    min_lot = float(info.volume_min)
    max_lot = float(info.volume_max)
    step = float(info.volume_step)
    if step <= 0 or raw_lot < min_lot:
        return None
    steps = math.floor(min(raw_lot, max_lot) / step)
    lot = steps * step
    precision = max(0, int(round(-math.log10(step)))) if step < 1 else 0
    lot = round(lot, precision)
    return lot if lot >= min_lot else None


def live_order_type(direction: str) -> int:
    return mt5.ORDER_TYPE_BUY if direction == "long" else mt5.ORDER_TYPE_SELL


def live_entry_price(symbol: str, direction: str) -> float | None:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None
    return float(tick.ask if direction == "long" else tick.bid)


def stop_price(entry_price: float, direction: str, sl_distance: float) -> float:
    if direction == "long":
        return entry_price - sl_distance
    return entry_price + sl_distance


def target_price(entry_price: float, direction: str, tp_distance: float) -> float:
    if direction == "long":
        return entry_price + tp_distance
    return entry_price - tp_distance


def calculate_live_lot(
    config: dict,
    direction: str,
    sl_distance: float,
) -> tuple[float | None, dict]:
    account = mt5.account_info()
    if account is None or account.balance <= 0:
        return None, {"message": "account_unavailable"}

    entry = live_entry_price(config["symbol"], direction)
    if entry is None or entry <= 0:
        return None, {"message": "tick_unavailable", "balance": account.balance}

    order_type = live_order_type(direction)
    sl = stop_price(entry, direction, sl_distance)
    risk_per_lot = mt5.order_calc_profit(order_type, config["symbol"], 1.0, entry, sl)
    if risk_per_lot is None:
        return None, {
            "message": f"order_calc_profit_failed: {mt5.last_error()}",
            "balance": account.balance,
        }
    risk_per_lot = abs(float(risk_per_lot))
    risk_budget = float(account.balance) * PORTFOLIO_RISK_PER_SIGNAL * config["weight"]
    raw_lot = risk_budget / max(risk_per_lot, 1e-9)
    return normalize_lot(config["symbol"], raw_lot), {
        "balance": round(float(account.balance), 2),
        "risk_budget": round(risk_budget, 4),
        "risk_per_lot": round(risk_per_lot, 4),
        "raw_lot": round(raw_lot, 6),
    }


def current_day_range() -> tuple[datetime, datetime]:
    now = datetime.now()
    return datetime(now.year, now.month, now.day), now


def get_daily_realized_pnl() -> float:
    start, end = current_day_range()
    deals = mt5.history_deals_get(start, end)
    if deals is None:
        return 0.0
    return float(
        sum(
            deal.profit + deal.commission + deal.swap + getattr(deal, "fee", 0.0)
            for deal in deals
            if deal.magic == MAGIC_NUMBER
        )
    )


def execute_live_order(
    config: dict,
    direction: str,
    lot: float,
    snapshot: dict,
) -> object | None:
    symbol = config["symbol"]
    params = config["params"]
    entry = live_entry_price(symbol, direction)
    if entry is None:
        return None

    info = mt5.symbol_info(symbol)
    digits = int(info.digits) if info is not None else 3
    tp_distance = max(snapshot["atr"] * float(params["tp_atr"]), 1e-9)
    sl_distance = max(snapshot["atr"] * float(params["sl_atr"]), 1e-9)
    exit_mode = params.get("exit_mode", "tp_sl")
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": live_order_type(direction),
        "price": round(entry, digits),
        "sl": round(stop_price(entry, direction, sl_distance), digits),
        "tp": (
            0.0
            if exit_mode == "time_stop"
            else round(target_price(entry, direction, tp_distance), digits)
        ),
        "magic": MAGIC_NUMBER,
        "comment": "CodexPortfolio",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    return mt5.order_send(request)


def fill_live_result(row: dict, result: object | None) -> None:
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


def live_position_ticket(symbol: str) -> int | str:
    positions = get_strategy_positions(symbol)
    if not positions:
        return ""
    return getattr(positions[-1], "ticket", "")


def open_live_state_position(
    state: dict,
    config: dict,
    snapshot: dict,
    direction: str,
    spread_price: float,
    probs: np.ndarray,
    lot: float,
    result: object,
) -> dict:
    params = config["params"]
    ticket = live_position_ticket(config["symbol"])
    trade_id = (
        f"codex-live-{config['symbol'].replace('#', '')}-"
        f"{config['timeframe']}-{format_time(snapshot['bar_time']).replace(':', '')}"
    )
    position = {
        "id": trade_id,
        "mode": "live",
        "key": config_key(config),
        "symbol": config["symbol"],
        "timeframe": config["timeframe"],
        "role": config["role"],
        "weight": config["weight"],
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
        "lot": lot,
        "ticket": ticket,
        "order_id": getattr(result, "order", ""),
        "deal_id": getattr(result, "deal", ""),
    }
    state["positions"].append(position)
    append_csv(
        TRADE_LOG_FILE,
        TRADE_FIELDS,
        {
            "event_time": format_time(utc_now()),
            "mode": "live",
            "strategy": STRATEGY_NAME,
            "trade_id": trade_id,
            "symbol": config["symbol"],
            "timeframe": config["timeframe"],
            "role": config["role"],
            "weight": config["weight"],
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
            "lot": lot,
            "ticket": ticket,
            "order_id": getattr(result, "order", ""),
            "deal_id": getattr(result, "deal", ""),
            "retcode": getattr(result, "retcode", ""),
            "broker_comment": getattr(result, "comment", ""),
        },
    )
    return position


def close_live_position(position: dict) -> bool:
    ticket = position.get("ticket")
    mt5_positions = get_strategy_positions(position["symbol"])
    candidates = [
        item for item in mt5_positions if str(getattr(item, "ticket", "")) == str(ticket)
    ]
    if not candidates:
        return False
    mt5_position = candidates[0]
    tick = mt5.symbol_info_tick(position["symbol"])
    if tick is None:
        return False
    if mt5_position.type == mt5.POSITION_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position["symbol"],
        "volume": mt5_position.volume,
        "type": order_type,
        "position": mt5_position.ticket,
        "price": price,
        "magic": MAGIC_NUMBER,
        "comment": "CodexPortfolio Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE


def update_live_state_position(state: dict, position: dict, snapshot: dict) -> bool:
    position["hold_bars"] += 1
    ticket = str(position.get("ticket", ""))
    still_open = any(
        str(getattr(item, "ticket", "")) == ticket
        for item in get_strategy_positions(position["symbol"])
    )
    if not still_open:
        append_csv(
            TRADE_LOG_FILE,
            TRADE_FIELDS,
            {
                "event_time": format_time(utc_now()),
                "mode": "live",
                "strategy": STRATEGY_NAME,
                "trade_id": position["id"],
                "symbol": position["symbol"],
                "timeframe": position["timeframe"],
                "role": position["role"],
                "weight": position["weight"],
                "action": "close",
                "reason": "server_closed",
                "direction": position["direction"],
                "entry_time": position["entry_time"],
                "exit_time": format_time(snapshot["bar_time"]),
                "entry_price": round(position["entry_price"], 6),
                "exit_price": round(snapshot["close"], 6),
                "hold_bars": position["hold_bars"],
                "reward_r": round(reward_r(position, snapshot["close"]), 6),
                "lot": position.get("lot", ""),
                "ticket": position.get("ticket", ""),
            },
        )
        remove_state_position(state, position["id"])
        return True

    if position.get("exit_mode") != "time_stop":
        return False

    close_now, reason = should_close_position(position, snapshot["close"])
    if not close_now:
        return False
    if not close_live_position(position):
        return False
    append_csv(
        TRADE_LOG_FILE,
        TRADE_FIELDS,
        {
            "event_time": format_time(utc_now()),
            "mode": "live",
            "strategy": STRATEGY_NAME,
            "trade_id": position["id"],
            "symbol": position["symbol"],
            "timeframe": position["timeframe"],
            "role": position["role"],
            "weight": position["weight"],
            "action": "close",
            "reason": reason,
            "direction": position["direction"],
            "entry_time": position["entry_time"],
            "exit_time": format_time(snapshot["bar_time"]),
            "entry_price": round(position["entry_price"], 6),
            "exit_price": round(snapshot["close"], 6),
            "hold_bars": position["hold_bars"],
            "reward_r": round(reward_r(position, snapshot["close"]), 6),
            "lot": position.get("lot", ""),
            "ticket": position.get("ticket", ""),
        },
    )
    remove_state_position(state, position["id"])
    return True


def process_config(config: dict, model, state: dict, mode: str) -> None:
    snapshot = build_snapshot(config)
    event_time = format_time(utc_now())
    base_log = {
        "event_time": event_time,
        "mode": mode,
        "strategy": STRATEGY_NAME,
        "symbol": config["symbol"],
        "timeframe": config["timeframe"],
        "role": config["role"],
        "weight": config["weight"],
    }
    if snapshot is None:
        append_csv(
            SIGNAL_LOG_FILE,
            SIGNAL_FIELDS,
            {**base_log, "status": "feature_unavailable", "reason": "snapshot_none"},
        )
        return

    tick = mt5.symbol_info_tick(config["symbol"])
    market_time = (
        datetime.fromtimestamp(tick.time, tz=timezone.utc)
        if tick is not None
        else utc_now()
    )
    snapshot_age = (market_time - snapshot["bar_time"]).total_seconds()
    max_snapshot_age = (
        TIMEFRAME_SECONDS[config["timeframe"]] * MAX_SNAPSHOT_AGE_BARS
    )
    if snapshot_age > max_snapshot_age:
        append_csv(
            SIGNAL_LOG_FILE,
            SIGNAL_FIELDS,
            {
                **base_log,
                "bar_time": format_time(snapshot["bar_time"]),
                "status": "stale_snapshot",
                "reason": f"age_seconds={int(snapshot_age)}",
            },
        )
        return

    key = config_key(config)
    state_key = f"{mode}|{key}"
    bar_time_text = format_time(snapshot["bar_time"])
    if state["last_bar_times"].get(state_key) == bar_time_text:
        return
    state["last_bar_times"][state_key] = bar_time_text

    existing = find_state_position(state, config, mode)
    had_position = existing is not None
    closed = False
    if existing is not None:
        if mode == "paper":
            closed = update_paper_position(state, existing, snapshot)
        else:
            closed = update_live_state_position(state, existing, snapshot)

    probs = normalize_direction_probs(
        model.predict_proba(snapshot["features"])[0],
        config["params"].get("direction_mode", "both"),
    )
    signal, confidence, direction = direction_from_probs(probs, config["params"])
    buy_prob = float(probs[1])
    sell_prob = float(probs[2])
    edge = abs(buy_prob - sell_prob)
    spread_points, spread_price = get_spread(config["symbol"])
    spread_atr = spread_price / max(snapshot["atr"], 1e-9)
    filter_ok, filter_reason = passes_filter(signal, snapshot, spread_atr, config["params"])
    threshold = float(config["params"]["threshold"])
    edge_threshold = float(config["params"]["edge_threshold"])
    tp_distance = max(snapshot["atr"] * float(config["params"]["tp_atr"]), 1e-9)
    sl_distance = max(snapshot["atr"] * float(config["params"]["sl_atr"]), 1e-9)

    has_signal = (
        direction in {"long", "short"}
        and confidence >= threshold
        and edge >= edge_threshold
        and filter_ok
    )
    gate_ok, gate_reason = (
        paper_position_gate(state, config)
        if mode == "paper"
        else live_position_gate(config)
    )

    log_row = {
        **base_log,
        "bar_time": bar_time_text,
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
        "tp_distance": round(tp_distance, 6),
        "sl_distance": round(sl_distance, 6),
        "max_hold": int(config["params"]["max_hold"]),
        "exit_mode": config["params"].get("exit_mode", "tp_sl"),
    }

    if had_position:
        log_row["status"] = "position_closed" if closed else "position_open"
        log_row["reason"] = "closed_this_bar" if closed else "already_open"
    elif not has_signal:
        if direction not in {"long", "short"}:
            log_row["status"] = "direction_blocked"
            log_row["reason"] = direction
        elif confidence < threshold:
            log_row["status"] = "no_signal"
            log_row["reason"] = "threshold"
        elif edge < edge_threshold:
            log_row["status"] = "no_signal"
            log_row["reason"] = "edge"
        else:
            log_row["status"] = "filter_blocked"
            log_row["reason"] = filter_reason
    elif not gate_ok:
        log_row["status"] = "position_gate_blocked"
        log_row["reason"] = gate_reason
    elif mode == "paper":
        position = open_paper_position(
            state,
            config,
            snapshot,
            direction,
            spread_price,
            probs,
        )
        log_row["status"] = "paper_opened"
        log_row["reason"] = "signal"
        log_row["position_id"] = position["id"]
    else:
        account = mt5.account_info()
        daily_pnl = get_daily_realized_pnl()
        log_row["daily_pnl"] = round(daily_pnl, 2)
        if account is None or account.balance <= 0:
            log_row["status"] = "account_unavailable"
            log_row["reason"] = "account_info"
        elif daily_pnl <= -float(account.balance) * MAX_DAILY_LOSS_PCT:
            log_row["status"] = "daily_loss_guard"
            log_row["reason"] = "max_daily_loss"
            log_row["balance"] = round(float(account.balance), 2)
        else:
            lot, lot_info = calculate_live_lot(config, direction, sl_distance)
            log_row.update(lot_info)
            log_row["lot"] = "" if lot is None else lot
            if lot is None:
                log_row["status"] = "lot_below_minimum"
                log_row["reason"] = lot_info.get("message", "lot_unavailable")
            else:
                result = execute_live_order(config, direction, lot, snapshot)
                fill_live_result(log_row, result)
                if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                    position = open_live_state_position(
                        state,
                        config,
                        snapshot,
                        direction,
                        spread_price,
                        probs,
                        lot,
                        result,
                    )
                    log_row["status"] = "order_opened"
                    log_row["reason"] = "signal"
                    log_row["position_id"] = position["id"]
                else:
                    log_row["status"] = "order_failed"
                    log_row["reason"] = "broker_rejected"

    append_csv(SIGNAL_LOG_FILE, SIGNAL_FIELDS, log_row)
    print(
        f"{bar_time_text} {config['symbol']} {config['timeframe']} "
        f"{mode} {log_row['status']} {log_row.get('reason', '')} "
        f"buy={buy_prob:.3f} sell={sell_prob:.3f}",
        flush=True,
    )


def parse_args(argv: list[str]) -> tuple[str, bool]:
    if "--help" in argv or "-h" in argv:
        print(
            "Usage: python codex.py [--paper|--live] [--once]\n"
            "Default mode is --paper. Live mode requires "
            f"{LIVE_CONFIRM_ENV}={LIVE_CONFIRM_VALUE}."
        )
        raise SystemExit(0)
    mode = "live" if "--live" in argv else "paper"
    if "--paper" in argv:
        mode = "paper"
    return mode, "--once" in argv


def validate_live_mode(mode: str) -> None:
    if mode != "live":
        return
    if os.environ.get(LIVE_CONFIRM_ENV) != LIVE_CONFIRM_VALUE:
        raise RuntimeError(
            "Live mode is locked. Set "
            f"{LIVE_CONFIRM_ENV}={LIVE_CONFIRM_VALUE} only after paper validation."
        )


def initialize_mt5(configs: list[dict]) -> bool:
    if not MT5_TERMINAL_PATH.exists():
        raise FileNotFoundError(f"MT5 terminal not found: {MT5_TERMINAL_PATH}")

    mt5.shutdown()
    if not mt5.initialize(
        path=str(MT5_TERMINAL_PATH), timeout=MT5_CONNECT_TIMEOUT_MS
    ):
        return False

    for config in configs:
        symbol = config["symbol"]
        if not mt5.symbol_select(symbol, True):
            print(f"Warning: unable to select {symbol}", flush=True)
            continue

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"Warning: no tick available for {symbol}", flush=True)
            continue

        mt5.copy_rates_from(
            symbol,
            TIMEFRAME_MAP[config["timeframe"]],
            datetime.fromtimestamp(tick.time, tz=timezone.utc),
            520,
        )
    return True


def mt5_connection_ready() -> bool:
    terminal = mt5.terminal_info()
    return bool(
        terminal is not None
        and terminal.connected
        and mt5.account_info() is not None
    )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    mode, run_once = parse_args(argv)
    validate_live_mode(mode)

    configs = build_portfolio_configs()
    missing = [
        str(config["model_path"])
        for config in configs
        if not Path(config["model_path"]).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing model files: {missing}")

    if not initialize_mt5(configs):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    models = {config_key(config): load_model(config["model_path"]) for config in configs}
    state = load_state()
    print(
        f"{STRATEGY_NAME} active. mode={mode} "
        "No live orders are sent unless --live and confirmation env are set.",
        flush=True,
    )

    while True:
        try:
            if not mt5_connection_ready():
                print("MT5 disconnected; reconnecting...", flush=True)
                if not initialize_mt5(configs):
                    print(f"MT5 reconnect failed: {mt5.last_error()}", flush=True)
                    time.sleep(POLL_SECONDS)
                    continue
            for config in configs:
                process_config(config, models[config_key(config)], state, mode)
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
