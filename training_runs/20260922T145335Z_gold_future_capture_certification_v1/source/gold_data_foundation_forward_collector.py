from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from gold_data_foundation1_audit import (
    broker_epoch_to_utc,
    utc,
    utc_to_broker_query,
)


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "gold_generation21_forward_protocol.json"
CONFIG_PATH = ROOT / "data_foundation1_collector_config.json"
OUTPUT_ROOT = ROOT / "untouched_forward" / "generation21" / "data"
STATE_PATH = OUTPUT_ROOT / "collector_state.json"
MANIFEST_PATH = OUTPUT_ROOT / "collector_manifest.json"
TICK_CHUNK = timedelta(minutes=15)
M1_CHUNK = timedelta(days=7)

TICK_FIELDS = (
    "timestamp_utc",
    "raw_broker_time_msc",
    "bid",
    "ask",
    "last",
    "volume",
    "volume_real",
    "flags",
    "observed_spread_price",
    "source_symbol",
    "timestamp_basis",
)
M1_FIELDS = (
    "bar_open_utc",
    "raw_broker_epoch_seconds",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread_points",
    "real_volume",
    "source_symbol",
    "timestamp_basis",
)
EVENT_FIELDS = (
    "release_timestamp_utc",
    "event_type",
    "source_url",
    "original_timezone",
    "source_release_id",
    "scheduled_or_actual",
)
EVENT_OUTPUT_FIELDS = EVENT_FIELDS + ("collected_at_utc", "source_file_sha256")


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {path}: {exc}") from exc


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_symbol(symbol: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in symbol)


def append_rows(path: Path, fields: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> int:
    materialized = list(rows)
    if not materialized:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        if new_file:
            writer.writeheader()
        writer.writerows(materialized)
        handle.flush()
    return len(materialized)


def initialise_mt5(terminal_path: Path):
    import MetaTrader5 as mt5

    if not terminal_path.is_file():
        raise RuntimeError(f"MT5 terminal not found: {terminal_path}")
    if not mt5.initialize(path=str(terminal_path), timeout=10_000):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    terminal = mt5.terminal_info()
    if terminal is None or not terminal.connected:
        mt5.shutdown()
        raise RuntimeError("MT5 terminal is not connected")
    return mt5


def tick_rows(symbol: str, ticks) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    names = set(ticks.dtype.names or ())
    for item in ticks:
        raw_msc = int(item["time_msc"]) if "time_msc" in names else int(item["time"]) * 1000
        timestamp = broker_epoch_to_utc(raw_msc / 1000.0)
        bid = float(item["bid"]) if "bid" in names else math.nan
        ask = float(item["ask"]) if "ask" in names else math.nan
        observed_spread = ask - bid if bid > 0 and ask >= bid else ""
        groups[timestamp.date().isoformat()].append(
            {
                "timestamp_utc": timestamp.isoformat(),
                "raw_broker_time_msc": raw_msc,
                "bid": bid if math.isfinite(bid) else "",
                "ask": ask if math.isfinite(ask) else "",
                "last": float(item["last"]) if "last" in names else "",
                "volume": float(item["volume"]) if "volume" in names else "",
                "volume_real": float(item["volume_real"]) if "volume_real" in names else "",
                "flags": int(item["flags"]) if "flags" in names else "",
                "observed_spread_price": observed_spread,
                "source_symbol": symbol,
                "timestamp_basis": "XM EET/EEST server epoch converted to UTC",
            }
        )
    return groups


def m1_rows(symbol: str, rates) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    names = set(rates.dtype.names or ())
    for item in rates:
        raw_epoch = int(item["time"])
        timestamp = broker_epoch_to_utc(raw_epoch)
        groups[timestamp.date().isoformat()].append(
            {
                "bar_open_utc": timestamp.isoformat(),
                "raw_broker_epoch_seconds": raw_epoch,
                "open": float(item["open"]),
                "high": float(item["high"]),
                "low": float(item["low"]),
                "close": float(item["close"]),
                "tick_volume": int(item["tick_volume"]),
                "spread_points": int(item["spread"]),
                "real_volume": int(item["real_volume"]) if "real_volume" in names else "",
                "source_symbol": symbol,
                "timestamp_basis": "XM EET/EEST server epoch converted to UTC",
            }
        )
    return groups


def append_grouped(kind: str, symbol: str, fields: tuple[str, ...], groups: dict[str, list[dict[str, Any]]]) -> int:
    total = 0
    directory = OUTPUT_ROOT / kind / safe_symbol(symbol)
    for day, rows in groups.items():
        total += append_rows(directory / f"{day}.csv", fields, rows)
    return total


def collect_ticks(
    mt5,
    symbol: str,
    start: datetime,
    end: datetime,
    progress: Callable[[datetime], None],
) -> tuple[int, datetime]:
    total = 0
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + TICK_CHUNK - timedelta(milliseconds=1), end)
        ticks = mt5.copy_ticks_range(
            symbol,
            utc_to_broker_query(cursor),
            utc_to_broker_query(chunk_end),
            mt5.COPY_TICKS_ALL,
        )
        if ticks is None:
            raise RuntimeError(f"copy_ticks_range failed for {symbol}: {mt5.last_error()}")
        total += append_grouped("ticks", symbol, TICK_FIELDS, tick_rows(symbol, ticks))
        cursor = chunk_end + timedelta(milliseconds=1)
        progress(cursor)
    return total, cursor


def collect_m1(
    mt5,
    symbol: str,
    start: datetime,
    last_closed_open: datetime,
    progress: Callable[[datetime], None],
) -> tuple[int, datetime]:
    total = 0
    cursor = start.replace(second=0, microsecond=0)
    if cursor < start:
        cursor += timedelta(minutes=1)
    while cursor <= last_closed_open:
        chunk_end = min(cursor + M1_CHUNK - timedelta(minutes=1), last_closed_open)
        rates = mt5.copy_rates_range(
            symbol,
            mt5.TIMEFRAME_M1,
            utc_to_broker_query(cursor),
            utc_to_broker_query(chunk_end),
        )
        if rates is None:
            raise RuntimeError(f"copy_rates_range failed for {symbol}: {mt5.last_error()}")
        total += append_grouped("m1", symbol, M1_FIELDS, m1_rows(symbol, rates))
        cursor = chunk_end + timedelta(minutes=1)
        progress(cursor)
    return total, cursor


def validated_events(path: Path, cutoff: datetime, allowed: set[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EVENT_FIELDS:
            raise ValueError(f"Event file must have exact fields: {', '.join(EVENT_FIELDS)}")
        for line_number, row in enumerate(reader, start=2):
            timestamp = utc(row["release_timestamp_utc"])
            if timestamp < cutoff:
                raise ValueError(f"Event line {line_number} precedes untouched cutoff")
            if row["event_type"] not in allowed:
                raise ValueError(f"Event line {line_number} has unsupported type: {row['event_type']}")
            if not row["source_url"].startswith("https://"):
                raise ValueError(f"Event line {line_number} lacks HTTPS provenance")
            if row["scheduled_or_actual"] not in {"scheduled", "actual_release"}:
                raise ValueError(f"Event line {line_number} has invalid scheduled_or_actual")
            normalized = {field: row[field].strip() for field in EVENT_FIELDS}
            normalized["release_timestamp_utc"] = timestamp.isoformat()
            rows.append(normalized)
    return rows


def import_events(path: Path, cutoff: datetime, allowed: set[str]) -> int:
    rows = validated_events(path, cutoff, allowed)
    collected_at = datetime.now(timezone.utc).isoformat()
    source_hash = sha256(path)
    existing_ids: set[tuple[str, str, str]] = set()
    output = OUTPUT_ROOT / "economic_events.csv"
    if output.exists():
        with output.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                existing_ids.add((row["release_timestamp_utc"], row["event_type"], row["source_release_id"]))
    unique = [
        {**row, "collected_at_utc": collected_at, "source_file_sha256": source_hash}
        for row in rows
        if (row["release_timestamp_utc"], row["event_type"], row["source_release_id"]) not in existing_ids
    ]
    return append_rows(output, EVENT_OUTPUT_FIELDS, unique)


def csv_data_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def validate_configuration(protocol: dict[str, Any], config: dict[str, Any]) -> datetime:
    protocol_cutoff = utc(protocol["untouched_forward_cutoff_utc"])
    config_cutoff = utc(config["cutoff_utc"])
    if protocol_cutoff != config_cutoff:
        raise RuntimeError(f"Cutoff mismatch: protocol={protocol_cutoff}, config={config_cutoff}")
    expected_storage = (ROOT / protocol["storage_path"]).resolve()
    if expected_storage != OUTPUT_ROOT.resolve():
        raise RuntimeError(f"Storage mismatch: protocol={expected_storage}, collector={OUTPUT_ROOT}")
    return protocol_cutoff


def self_check() -> int:
    protocol = load_json(PROTOCOL_PATH)
    config = load_json(CONFIG_PATH)
    cutoff = validate_configuration(protocol, config)
    summer = datetime(2026, 9, 1, 2, tzinfo=timezone.utc)
    winter = datetime(2026, 1, 2, 2, tzinfo=timezone.utc)
    assert broker_epoch_to_utc(utc_to_broker_query(summer).timestamp()) == summer
    assert broker_epoch_to_utc(utc_to_broker_query(winter).timestamp()) == winter
    assert safe_symbol("GOLD#") == "GOLD_"
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory) / "events.csv"
        fixture.write_text(
            ",".join(EVENT_FIELDS)
            + "\n2026-09-02T12:30:00Z,CPI,https://www.bls.gov/,America/New_York,test,scheduled\n",
            encoding="utf-8",
        )
        rows = validated_events(
            fixture,
            cutoff,
            set(config["economic_event_input"]["allowed_event_types"]),
        )
        assert len(rows) == 1 and rows[0]["release_timestamp_utc"].endswith("+00:00")
    print("SELF_CHECK_OK: cutoff, EET/EEST conversion, symbol paths, and event provenance validation")
    return 0


def verify_output(cutoff: datetime) -> int:
    forbidden = {"pnl", "return", "net_r", "label", "prediction", "probability", "position", "trade_id"}
    checked_rows = {"ticks": 0, "m1": 0, "economic_events": 0}
    errors: list[str] = []
    for kind, timestamp_field in (("ticks", "timestamp_utc"), ("m1", "bar_open_utc")):
        directory = OUTPUT_ROOT / kind
        for path in directory.rglob("*.csv") if directory.exists() else ():
            with path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                headers = set(reader.fieldnames or ())
                found_forbidden = sorted(headers & forbidden)
                if found_forbidden:
                    errors.append(f"{path}: forbidden outcome fields {found_forbidden}")
                if timestamp_field not in headers:
                    errors.append(f"{path}: missing {timestamp_field}")
                    continue
                for line_number, row in enumerate(reader, start=2):
                    checked_rows[kind] += 1
                    try:
                        timestamp = utc(row[timestamp_field])
                    except (KeyError, ValueError) as exc:
                        errors.append(f"{path}:{line_number}: invalid timestamp ({exc})")
                        continue
                    if timestamp < cutoff:
                        errors.append(f"{path}:{line_number}: timestamp precedes cutoff")
    event_path = OUTPUT_ROOT / "economic_events.csv"
    if event_path.exists():
        with event_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != EVENT_OUTPUT_FIELDS:
                errors.append(f"{event_path}: unexpected event schema")
            for line_number, row in enumerate(reader, start=2):
                checked_rows["economic_events"] += 1
                try:
                    release = utc(row["release_timestamp_utc"])
                    collected = utc(row["collected_at_utc"])
                except (KeyError, ValueError) as exc:
                    errors.append(f"{event_path}:{line_number}: invalid timestamp ({exc})")
                    continue
                if release < cutoff:
                    errors.append(f"{event_path}:{line_number}: release precedes cutoff")
                if collected < cutoff:
                    errors.append(f"{event_path}:{line_number}: collection precedes cutoff")
    manifest = load_json(MANIFEST_PATH)
    if manifest.get("contains_strategy_outcomes") is not False:
        errors.append("manifest does not explicitly prohibit strategy outcomes")
    if manifest.get("contains_predictions") is not False or manifest.get("contains_labels") is not False:
        errors.append("manifest permits predictions or labels")
    if manifest.get("contamination_status") != "untouched":
        errors.append("manifest contamination_status is not untouched")
    payload = {"status": "PASS" if not errors else "FAIL", "checked_rows": checked_rows, "errors": errors[:100]}
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    return 0 if not errors else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Append-only untouched forward raw-data collector")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--import-events", type=Path)
    args = parser.parse_args()
    if args.self_check:
        return self_check()

    protocol = load_json(PROTOCOL_PATH)
    config = load_json(CONFIG_PATH)
    cutoff = validate_configuration(protocol, config)
    if args.verify:
        return verify_output(cutoff)
    now = datetime.now(timezone.utc)
    state = load_json(STATE_PATH) if STATE_PATH.exists() else {"symbols": {}, "event_imports": []}
    state.setdefault("symbols", {})
    state.setdefault("event_imports", [])
    summary: dict[str, Any] = {"market": {}, "events_added": 0}

    if args.import_events:
        event_path = args.import_events.resolve()
        summary["events_added"] = import_events(
            event_path,
            cutoff,
            set(config["economic_event_input"]["allowed_event_types"]),
        )
        state["event_imports"].append(
            {
                "imported_at_utc": now.isoformat(),
                "source_path": str(event_path),
                "source_sha256": sha256(event_path),
                "rows_added": summary["events_added"],
            }
        )

    if now >= cutoff:
        terminal_path = Path(config["terminal_path"])
        mt5 = initialise_mt5(terminal_path)
        restore_hidden: list[str] = []
        try:
            for item in config["market_symbols"]:
                symbol = item["symbol"]
                symbol_state = state["symbols"].setdefault(symbol, {})
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is not None and not symbol_info.visible:
                    restore_hidden.append(symbol)
                if not mt5.symbol_select(symbol, True):
                    summary["market"][symbol] = {"status": "select_failed", "error": list(mt5.last_error())}
                    continue
                symbol_summary = {"status": "ok", "ticks_added": 0, "m1_added": 0}
                try:
                    if item.get("collect_ticks"):
                        tick_start = utc(symbol_state.get("next_tick_utc", cutoff.isoformat()))
                        if tick_start <= now:
                            def save_tick_progress(value: datetime) -> None:
                                symbol_state["next_tick_utc"] = value.isoformat()
                                write_json(STATE_PATH, state)

                            count, next_tick = collect_ticks(
                                mt5, symbol, tick_start, now, save_tick_progress
                            )
                            symbol_summary["ticks_added"] = count
                            symbol_state["next_tick_utc"] = next_tick.isoformat()
                    if item.get("collect_m1"):
                        last_closed_open = now.replace(second=0, microsecond=0) - timedelta(minutes=1)
                        m1_start = utc(symbol_state.get("next_m1_utc", cutoff.isoformat()))
                        if m1_start <= last_closed_open:
                            def save_m1_progress(value: datetime) -> None:
                                symbol_state["next_m1_utc"] = value.isoformat()
                                write_json(STATE_PATH, state)

                            count, next_m1 = collect_m1(
                                mt5, symbol, m1_start, last_closed_open, save_m1_progress
                            )
                            symbol_summary["m1_added"] = count
                            symbol_state["next_m1_utc"] = next_m1.isoformat()
                except Exception as exc:
                    symbol_summary["status"] = "collection_failed"
                    symbol_summary["error"] = f"{type(exc).__name__}: {exc}"
                summary["market"][symbol] = symbol_summary
                write_json(STATE_PATH, state)
        finally:
            for symbol in restore_hidden:
                mt5.symbol_select(symbol, False)
            mt5.shutdown()
    else:
        summary["waiting_for_cutoff"] = True

    state["last_successful_run_utc"] = datetime.now(timezone.utc).isoformat()
    state["cutoff_utc"] = cutoff.isoformat()
    write_json(STATE_PATH, state)
    manifest = {
        "dataset": "Generation 21 untouched forward raw information",
        "status": "untouched_raw_only",
        "cutoff_utc": cutoff.isoformat(),
        "last_collector_run_utc": state["last_successful_run_utc"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "config_sha256": sha256(CONFIG_PATH),
        "collector_sha256": sha256(Path(__file__).resolve()),
        "timestamp_basis": "UTC after explicit XM EET/EEST server-time conversion",
        "contains_strategy_outcomes": False,
        "contains_predictions": False,
        "contains_labels": False,
        "stored_economic_event_rows": csv_data_row_count(OUTPUT_ROOT / "economic_events.csv"),
        "contamination_status": "untouched",
        "contamination_rule": protocol["contamination_rule"],
        "latest_run_summary": summary,
    }
    write_json(MANIFEST_PATH, manifest)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    print(f"Manifest: {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
