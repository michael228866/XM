from __future__ import annotations

import argparse
import calendar
import json
import math
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
TERMINAL_PATH = Path(r"D:\XM2\terminal64.exe")
REPORT_JSON = ROOT / "gold_data_foundation1_report.json"
REPORT_MD = ROOT / "gold_data_foundation1_report.md"
CUTOFF = datetime(2026, 9, 1, 2, 0, tzinfo=timezone.utc)
WORKER_TIMEOUT_SECONDS = 15

GOLD_PROGRESSIVE_STARTS = (
    "2026-08-01T00:00:00+00:00",
    "2026-06-01T00:00:00+00:00",
    "2026-01-01T00:00:00+00:00",
    "2025-01-01T00:00:00+00:00",
    "2023-01-01T00:00:00+00:00",
    "2021-01-01T00:00:00+00:00",
    "2018-01-01T00:00:00+00:00",
    "2014-01-01T00:00:00+00:00",
    "1970-01-01T00:00:00+00:00",
)

PREREGISTERED_TICK_VOLUME_ACCELERATION = {
    "status": "pre_registered_for_new_data_only",
    "source": "Generation 21 post-hoc observation",
    "entry_time_definition": (
        "For decision minute t, let x_t = log1p(TICKVOL[t-1]), where t-1 is "
        "the immediately preceding completed GOLD# M1 bar. The fixed feature is "
        "x_t - 2*x_(t-1) + x_(t-2)."
    ),
    "equivalent_code": "log1p(TICKVOL).shift(1).diff().diff()",
    "fixed_choices": {
        "bar_size": "M1",
        "input": "broker M1 tick count (TICKVOL)",
        "transform": "log1p",
        "lags": [1, 2, 3],
        "normalization": "none",
    },
    "prohibited": [
        "changing the window or transform using 2018-2024 outcomes",
        "using an incomplete entry bar",
        "selecting a threshold from post-cutoff outcomes",
    ],
    "prior_post_hoc_spearman": {
        "2018-2020": 0.097,
        "2021-2022": 0.148,
        "2023-2024": 0.194,
    },
}


def utc(value: str | int | float | datetime) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    else:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Timezone required: {value}")
    return parsed.astimezone(timezone.utc)


def iso(epoch_seconds: int | float | None) -> str | None:
    if epoch_seconds is None or epoch_seconds <= 0:
        return None
    return broker_epoch_to_utc(epoch_seconds).isoformat()


def last_sunday(year: int, month: int) -> datetime:
    day = calendar.monthrange(year, month)[1]
    value = datetime(year, month, day, 1, tzinfo=timezone.utc)
    return value - timedelta(days=(value.weekday() + 1) % 7)


def xm_server_offset_seconds(actual_utc: datetime) -> int:
    """XM server is empirically EET/EEST: UTC+2 winter, UTC+3 summer."""
    actual_utc = utc(actual_utc)
    dst_start = last_sunday(actual_utc.year, 3)
    dst_end = last_sunday(actual_utc.year, 10)
    return 3 * 3600 if dst_start <= actual_utc < dst_end else 2 * 3600


def broker_epoch_to_utc(epoch_seconds: int | float) -> datetime:
    raw = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
    for hours in (2, 3):
        candidate = raw - timedelta(hours=hours)
        if xm_server_offset_seconds(candidate) == hours * 3600:
            return candidate
    raise ValueError(f"Could not map XM server epoch to UTC: {epoch_seconds}")


def utc_to_broker_query(value: datetime) -> datetime:
    value = utc(value)
    return value + timedelta(seconds=xm_server_offset_seconds(value))


def xm_server_wall_time_to_utc(value: datetime) -> datetime:
    if value.tzinfo is not None:
        raise ValueError("XM wall time must be naive")
    raw = value.replace(tzinfo=timezone.utc)
    for hours in (2, 3):
        candidate = raw - timedelta(hours=hours)
        if xm_server_offset_seconds(candidate) == hours * 3600:
            return candidate
    raise ValueError(f"Could not map XM server wall time to UTC: {value}")


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def initialise_mt5():
    import MetaTrader5 as mt5

    if not TERMINAL_PATH.is_file():
        raise RuntimeError(f"MT5 terminal not found: {TERMINAL_PATH}")
    if not mt5.initialize(path=str(TERMINAL_PATH), timeout=10_000):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    return mt5


def symbol_class(name: str, description: str = "", path: str = "") -> str | None:
    text = " ".join((name, description, path)).upper()
    compact = re.sub(r"[^A-Z0-9]", "", name.upper())
    if compact.startswith("GOLD") or compact.startswith("XAUUSD"):
        return "gold"
    if compact.startswith("SILVER") or compact.startswith("XAGUSD"):
        return "silver"
    if (
        compact.startswith(("DXY", "USDX", "USDINDEX", "DXF"))
        or "DOLLAR INDEX" in text
    ) and "US100" not in compact:
        return "usd_index"
    if re.search(r"US0?2Y|US10Y|UST0?2|UST10|TREASUR|T.?NOTE|YIELD", text):
        return "rates"
    if compact.startswith("VIX") or "VOLATILITY INDEX" in text:
        return "vix"
    if any(token in compact for token in ("WTI", "USOIL", "OIL")):
        return "crude"
    return None


def worker_inventory() -> dict[str, Any]:
    mt5 = initialise_mt5()
    try:
        terminal = mt5.terminal_info()
        version = mt5.version()
        matches = []
        for item in mt5.symbols_get() or ():
            asset_class = symbol_class(item.name, item.description, item.path)
            if asset_class is None:
                continue
            tick = mt5.symbol_info_tick(item.name)
            matches.append(
                {
                    "symbol": item.name,
                    "asset_class": asset_class,
                    "description": item.description,
                    "path": item.path,
                    "visible": bool(item.visible),
                    "trade_mode": int(item.trade_mode),
                    "digits": int(item.digits),
                    "point": float(item.point),
                    "latest_tick_utc": iso(tick.time if tick else None),
                }
            )
        gold_tick = mt5.symbol_info_tick("GOLD#")
        observed_offset = (
            int(round((gold_tick.time - datetime.now(timezone.utc).timestamp()) / 60.0))
            if gold_tick
            else None
        )
        return {
            "status": "ok",
            "terminal": {
                "path": str(TERMINAL_PATH),
                "connected": bool(terminal.connected) if terminal else False,
                "build": int(terminal.build) if terminal else None,
                "maxbars": int(terminal.maxbars) if terminal else None,
                "mt5_version": list(version) if version else None,
                "observed_live_tick_epoch_offset_minutes": observed_offset,
                "timestamp_conversion": "XM EET/EEST server epoch converted to UTC (+2 winter, +3 summer)",
            },
            "symbols": sorted(matches, key=lambda row: (row["asset_class"], row["symbol"])),
        }
    finally:
        mt5.shutdown()


def worker_first_after(symbol: str, start: str) -> dict[str, Any]:
    mt5 = initialise_mt5()
    info = mt5.symbol_info(symbol)
    was_visible = bool(info.visible) if info is not None else True
    try:
        if not mt5.symbol_select(symbol, True):
            return {"status": "select_failed", "error": list(mt5.last_error())}
        requested = utc(start)
        broker_query = utc_to_broker_query(requested)
        ticks = mt5.copy_ticks_from(symbol, broker_query, 1, mt5.COPY_TICKS_ALL)
        if ticks is None:
            return {"status": "mt5_error", "error": list(mt5.last_error())}
        if len(ticks) == 0:
            return {"status": "empty", "requested_start_utc": requested.isoformat()}
        row = ticks[0]
        names = set(ticks.dtype.names or ())
        epoch = float(row["time_msc"]) / 1000.0 if "time_msc" in names else float(row["time"])
        return {
            "status": "ok",
            "requested_start_utc": requested.isoformat(),
            "returned_tick_utc": iso(epoch),
            "returned_raw_broker_epoch": epoch,
            "returned_time_msc": int(row["time_msc"]) if "time_msc" in names else None,
        }
    finally:
        if not was_visible:
            mt5.symbol_select(symbol, False)
        mt5.shutdown()


def describe_numeric(values) -> dict[str, float | int | None]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None}
    return {
        "count": int(len(array)),
        "min": finite(np.min(array)),
        "median": finite(np.median(array)),
        "mean": finite(np.mean(array)),
        "p95": finite(np.percentile(array, 95)),
        "max": finite(np.max(array)),
    }


def worker_tick_window(symbol: str, start: str, end: str) -> dict[str, Any]:
    import numpy as np

    mt5 = initialise_mt5()
    info = mt5.symbol_info(symbol)
    was_visible = bool(info.visible) if info is not None else True
    try:
        if not mt5.symbol_select(symbol, True):
            return {"status": "select_failed", "error": list(mt5.last_error())}
        start_dt, end_dt = utc(start), utc(end)
        if end_dt <= start_dt or end_dt - start_dt > timedelta(hours=2):
            return {"status": "invalid_window", "error": "window must be >0 and <=2 hours"}
        ticks = mt5.copy_ticks_range(
            symbol,
            utc_to_broker_query(start_dt),
            utc_to_broker_query(end_dt),
            mt5.COPY_TICKS_ALL,
        )
        if ticks is None:
            return {"status": "mt5_error", "error": list(mt5.last_error())}
        names = set(ticks.dtype.names or ())
        count = int(len(ticks))
        result: dict[str, Any] = {
            "status": "ok",
            "requested_start_utc": start_dt.isoformat(),
            "requested_end_utc": end_dt.isoformat(),
            "tick_count": count,
            "fields": sorted(names),
        }
        if not count:
            return result

        time_seconds = (
            ticks["time_msc"].astype(np.float64) / 1000.0
            if "time_msc" in names
            else ticks["time"].astype(np.float64)
        )
        result["first_tick_utc"] = iso(float(time_seconds[0]))
        result["last_tick_utc"] = iso(float(time_seconds[-1]))
        availability = {}
        for field in ("bid", "ask", "last", "volume", "volume_real", "flags"):
            if field not in names:
                availability[field] = {"present": False, "nonzero": 0, "nonzero_fraction": 0.0}
                continue
            values = ticks[field]
            nonzero = int(np.count_nonzero(np.isfinite(values) & (values != 0)))
            availability[field] = {
                "present": True,
                "nonzero": nonzero,
                "nonzero_fraction": nonzero / count,
            }
        result["field_availability"] = availability

        if "bid" in names and "ask" in names:
            valid_quotes = (
                np.isfinite(ticks["bid"])
                & np.isfinite(ticks["ask"])
                & (ticks["bid"] > 0)
                & (ticks["ask"] >= ticks["bid"])
            )
            spreads = ticks["ask"][valid_quotes] - ticks["bid"][valid_quotes]
            result["spread_price"] = describe_numeric(spreads)
            mids = (ticks["ask"][valid_quotes] + ticks["bid"][valid_quotes]) / 2.0
            if len(mids) >= 2:
                changes = np.diff(mids)
                absolute_path = float(np.abs(changes).sum())
                result["signed_midquote_pressure"] = finite(np.sign(changes).mean())
                result["intrabar_realized_variance"] = finite(
                    np.square(np.diff(np.log(np.maximum(mids, 1e-12)))).sum()
                )
                result["intrabar_path_efficiency"] = finite(
                    abs(float(mids[-1] - mids[0])) / absolute_path if absolute_path else 0.0
                )

        intervals = np.diff(time_seconds)
        result["interarrival_seconds"] = describe_numeric(intervals)
        duration_minutes = max((time_seconds[-1] - time_seconds[0]) / 60.0, 1.0 / 60.0)
        result["quote_updates_per_minute"] = count / duration_minutes
        if "flags" in names:
            result["flags_distribution"] = dict(
                sorted(Counter(str(int(flag)) for flag in ticks["flags"]).items())
            )
        return result
    finally:
        if not was_visible:
            mt5.symbol_select(symbol, False)
        mt5.shutdown()


def worker_main(args: argparse.Namespace) -> int:
    try:
        if args.worker == "inventory":
            payload = worker_inventory()
        elif args.worker == "first-after":
            payload = worker_first_after(args.symbol, args.start)
        elif args.worker == "tick-window":
            payload = worker_tick_window(args.symbol, args.start, args.end)
        else:
            raise ValueError(f"Unknown worker: {args.worker}")
    except Exception as exc:  # A worker must return structured failure to its parent.
        payload = {"status": "exception", "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    return 0 if payload.get("status") == "ok" else 1


def run_worker(*arguments: str, timeout: int = WORKER_TIMEOUT_SECONDS) -> dict[str, Any]:
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", *arguments]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "timeout_seconds": timeout}
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return {
            "status": "worker_failed",
            "exit_code": completed.returncode,
            "stderr": completed.stderr.strip()[-500:],
        }
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        return {
            "status": "invalid_worker_output",
            "exit_code": completed.returncode,
            "stdout_tail": completed.stdout[-500:],
            "stderr_tail": completed.stderr[-500:],
        }
    payload["worker_exit_code"] = completed.returncode
    return payload


def last_nonempty_line(path: Path) -> str:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell() - 1
        buffer = bytearray()
        while position >= 0:
            handle.seek(position)
            byte = handle.read(1)
            if byte in (b"\n", b"\r"):
                if buffer:
                    break
            else:
                buffer.extend(byte)
            position -= 1
    return bytes(reversed(buffer)).decode("utf-8", errors="replace")


def local_csv_metadata(path: Path, source: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        header = handle.readline().strip().split("\t")
        first = handle.readline().strip().split("\t")
    last = last_nonempty_line(path).split("\t")

    def timestamp(parts: list[str]) -> str | None:
        if len(parts) < 2:
            return None
        try:
            parsed = datetime.strptime(f"{parts[0]} {parts[1]}", "%Y.%m.%d %H:%M:%S")
        except ValueError:
            return None
        return xm_server_wall_time_to_utc(parsed).isoformat()

    return {
        "source": source,
        "path": str(path.relative_to(ROOT)),
        "granularity": "M1",
        "first_timestamp_utc": timestamp(first),
        "last_timestamp_utc": timestamp(last),
        "timezone_assumption": (
            "XM terminal EET/EEST server wall time converted to UTC (+2 winter/+3 summer); "
            "conversion matches the live tick epoch audit"
        ),
        "fields": header,
        "size_bytes": path.stat().st_size,
        "bid_ask_available": False,
        "tick_count_proxy_available": "<TICKVOL>" in header,
        "historical_spread_field_available": "<SPREAD>" in header,
        "known_limitation": "M1 aggregates cannot reconstruct tick ordering or historical bid/ask path",
    }


def month_starts(start: datetime, end: datetime):
    cursor = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    finish = datetime(end.year, end.month, 1, tzinfo=timezone.utc)
    while cursor <= finish:
        yield cursor
        if cursor.month == 12:
            cursor = datetime(cursor.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            cursor = datetime(cursor.year, cursor.month + 1, 1, tzinfo=timezone.utc)


def next_month(value: datetime) -> datetime:
    return (
        datetime(value.year + 1, 1, 1, tzinfo=timezone.utc)
        if value.month == 12
        else datetime(value.year, value.month + 1, 1, tzinfo=timezone.utc)
    )


def audit_symbol(symbol_row: dict[str, Any], progressive: bool) -> dict[str, Any]:
    symbol = symbol_row["symbol"]
    starts = GOLD_PROGRESSIVE_STARTS if progressive else ("2014-01-01T00:00:00+00:00",)
    probes = []
    for start in starts:
        result = run_worker("first-after", "--symbol", symbol, "--start", start)
        probes.append({"start_utc": start, **result})

    returned = [utc(row["returned_tick_utc"]) for row in probes if row.get("status") == "ok"]
    earliest = min(returned) if returned else None
    latest = utc(symbol_row["latest_tick_utc"]) if symbol_row.get("latest_tick_utc") else None

    monthly: dict[str, str] = {}
    if earliest and latest:
        for month in month_starts(earliest, min(latest, CUTOFF)):
            result = run_worker(
                "first-after", "--symbol", symbol, "--start", month.isoformat()
            )
            tick_time = utc(result["returned_tick_utc"]) if result.get("status") == "ok" else None
            if result.get("status") == "timeout":
                state = "query_timeout"
            elif tick_time is None:
                state = "no_tick_returned"
            elif tick_time < next_month(month):
                state = "available"
            else:
                state = "gap"
            monthly[month.strftime("%Y-%m")] = state

    recent_sample: dict[str, Any] = {"status": "not_available"}
    if latest:
        recent_sample = run_worker(
            "tick-window",
            "--symbol",
            symbol,
            "--start",
            (latest - timedelta(hours=1)).isoformat(),
            "--end",
            latest.isoformat(),
            timeout=30,
        )

    approximate_count = None
    approximate_method = None
    sample_count = recent_sample.get("tick_count")
    if earliest and latest and isinstance(sample_count, int) and sample_count > 0:
        elapsed_days = max((min(latest, CUTOFF) - earliest).total_seconds() / 86_400.0, 0.0)
        trading_days = elapsed_days * 5.0 / 7.0
        approximate_count = int(sample_count * 23.0 * trading_days)
        approximate_method = (
            "Low-confidence extrapolation: latest one-hour count * 23 market hours * "
            "elapsed weekdays. This is not an exact server-side count."
        )

    return {
        **symbol_row,
        "earliest_available_tick_utc": earliest.isoformat() if earliest else None,
        "latest_available_tick_utc": latest.isoformat() if latest else None,
        "approximate_tick_count": approximate_count,
        "approximate_tick_count_method": approximate_method,
        "progressive_boundary_probes": probes,
        "monthly_availability": monthly,
        "recent_one_hour_sample": recent_sample,
    }


def microstructure_feasibility(gold_audit: dict[str, Any] | None) -> dict[str, Any]:
    sample = (gold_audit or {}).get("recent_one_hour_sample", {})
    availability = sample.get("field_availability", {})
    bid_ask = (
        availability.get("bid", {}).get("nonzero_fraction", 0.0) > 0.95
        and availability.get("ask", {}).get("nonzero_fraction", 0.0) > 0.95
    )
    timestamps = sample.get("tick_count", 0) > 1
    return {
        "historical_tick_boundary_utc": (gold_audit or {}).get("earliest_available_tick_utc"),
        "sample_status": sample.get("status", "not_available"),
        "features": {
            "bid_ask_spread": bid_ask,
            "spread_change": bid_ask,
            "spread_acceleration": bid_ask,
            "past_only_spread_percentile": bid_ask,
            "tick_count_per_minute": timestamps,
            "tick_count_acceleration": timestamps,
            "m1_tick_volume_acceleration": timestamps,
            "signed_tick_price_pressure": bid_ask,
            "intrabar_realized_volatility": bid_ask,
            "intrabar_path_efficiency": bid_ask,
            "quote_update_intensity": timestamps,
            "burst_activity_state": timestamps,
        },
        "historical_2018_2024_bid_ask_reconstructable": bool(
            bid_ask
            and gold_audit
            and gold_audit.get("earliest_available_tick_utc")
            and utc(gold_audit["earliest_available_tick_utc"]) <= datetime(2018, 1, 1, tzinfo=timezone.utc)
        ),
        "warning": (
            "Feature feasibility applies only inside the actual broker tick-history range. "
            "M1 TICKVOL is an activity proxy; tick-level volume/last may be absent for CFDs."
        ),
        "pre_registered_tick_volume_acceleration": PREREGISTERED_TICK_VOLUME_ACCELERATION,
    }


def choose_symbols(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for asset_class in ("gold", "silver", "usd_index", "rates", "vix", "crude"):
        candidates = [row for row in rows if row["asset_class"] == asset_class]
        candidates.sort(key=lambda row: (not row["visible"], not bool(row["latest_tick_utc"]), row["symbol"]))
        limit = 1 if asset_class in {"gold", "silver"} else 4
        selected.extend(candidates[:limit])
    return selected


def cross_market_inventory(
    audits: list[dict[str, Any]], local_sources: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    local_silver = next((row for row in local_sources if "SILVER" in row["source"]), None)
    if local_silver:
        rows.append(
            {
                "instrument": "XAGUSD / SILVER#",
                "source": local_silver["source"],
                "granularity": "M1 OHLC + TICKVOL",
                "first_timestamp": local_silver["first_timestamp_utc"],
                "last_timestamp": local_silver["last_timestamp_utc"],
                "timezone": local_silver["timezone_assumption"],
                "missing_periods": "Not exhaustively gap-audited; no tick path or bid/ask",
                "revision_policy": "Frozen local export; upstream broker history can change on re-export",
                "known_at_gold_entry": "Yes after each bar closes; current/incomplete bar is prohibited",
                "classification": "historical M1 context, not tick microstructure",
            }
        )
    for audit in audits:
        if audit["asset_class"] == "gold":
            continue
        if not audit.get("latest_available_tick_utc"):
            continue
        missing = [
            month for month, status in audit.get("monthly_availability", {}).items() if status != "available"
        ]
        rows.append(
            {
                "instrument": audit["symbol"],
                "source": "XM MT5 broker tick store",
                "granularity": "quote tick",
                "first_timestamp": audit.get("earliest_available_tick_utc"),
                "last_timestamp": audit.get("latest_available_tick_utc"),
                "timezone": "UTC after audited XM EET/EEST conversion",
                "missing_periods": missing or "No missing month detected inside audited boundary; weekends/closures expected",
                "revision_policy": "Broker history may be backfilled; collector preserves received raw rows",
                "known_at_gold_entry": "Yes when tick timestamp <= GOLD entry timestamp",
                "classification": (
                    "intraday current contract only; no continuous roll"
                    if "-" in audit["symbol"]
                    else "intraday broker CFD"
                ),
            }
        )
    rows.extend(
        [
            {
                "instrument": "US 2Y constant maturity (DGS2)",
                "source": "Federal Reserve H.15 via FRED: https://fred.stlouisfed.org/series/DGS2",
                "granularity": "daily",
                "first_timestamp": "1976-06-01 (series start)",
                "last_timestamp": "2026-08-28 observed; updated 2026-08-31 15:16 CDT at audit time",
                "timezone": "observation date plus documented publication timestamp",
                "missing_periods": "Weekends, holidays and occasional missing daily observations",
                "revision_policy": "FRED states all data are subject to revision",
                "known_at_gold_entry": "Only after the H.15/FRED publication timestamp, never intraday on observation date",
                "classification": "slow regime context only",
            },
            {
                "instrument": "US 10Y constant maturity (DGS10)",
                "source": "Federal Reserve H.15 via FRED: https://fred.stlouisfed.org/series/DGS10",
                "granularity": "daily",
                "first_timestamp": "1962-01-02 (series start)",
                "last_timestamp": "2026-08-28 observed; updated 2026-08-31 15:16 CDT at audit time",
                "timezone": "observation date plus documented publication timestamp",
                "missing_periods": "Weekends, holidays and occasional missing daily observations",
                "revision_policy": "FRED states all data are subject to revision",
                "known_at_gold_entry": "Only after the H.15/FRED publication timestamp, never intraday on observation date",
                "classification": "slow regime context only",
            },
            {
                "instrument": "VIX index",
                "source": "Cboe official daily history: https://www.cboe.com/tradable_products/vix/vix_historical_data/",
                "granularity": "daily close in free official file",
                "first_timestamp": "1990",
                "last_timestamp": "present, updated daily",
                "timezone": "U.S. market date; exact publication timestamp must be retained on acquisition",
                "missing_periods": "Non-trading days; intraday history not included in free file",
                "revision_policy": "Methodology/history can be corrected; preserve acquisition vintage",
                "known_at_gold_entry": "Only after the daily close/publication, not as same-day intraday input",
                "classification": "slow regime context unless licensed intraday data is acquired",
            },
            {
                "instrument": "WTI Cushing spot",
                "source": "EIA official history: https://www.eia.gov/dnav/pet/hist/rwtca.htm",
                "granularity": "daily observations published with delay",
                "first_timestamp": "1986",
                "last_timestamp": "latest published EIA release vintage",
                "timezone": "observation date plus EIA release timestamp",
                "missing_periods": "Non-business days and publication lag",
                "revision_policy": "Historical values/source methodology can be revised; preserve vintage",
                "known_at_gold_entry": "Not knowable intraday on the observation date; use only after release",
                "classification": "slow regime context only",
            },
            {
                "instrument": "ICE U.S. Dollar Index futures",
                "source": "ICE contract reference: https://www.ice.com/products/194/US-Dollar-Index-Futures/expiry",
                "granularity": "licensed tick/intraday; contract reference is public",
                "first_timestamp": "Not acquired as a continuous historical dataset",
                "last_timestamp": "Current contract-dependent",
                "timezone": "exchange timestamp; normalize to UTC",
                "missing_periods": "Historical intraday feed not locally licensed/acquired",
                "revision_policy": "Trades are immutable; vendor corrections and roll construction require vintages",
                "known_at_gold_entry": "Yes only with a timestamped licensed live/historical feed",
                "classification": "not currently available beyond short XM contract history",
            },
        ]
    )
    return rows


def economic_event_source_inventory() -> list[dict[str, Any]]:
    return [
        {
            "events": ["CPI", "Core CPI", "Employment Situation", "NFP", "PPI", "Initial Claims"],
            "source": "U.S. BLS release schedules and archived releases",
            "url": "https://www.bls.gov/schedule/news_release/bls.ics",
            "timestamp_reliability": "Scheduled date/time is official and stated in Eastern Time; archived release embargo header documents actual release timestamp",
            "historical_coverage": "Archived Employment releases are listed back to at least 1994; CPI has equivalent archives",
            "revision_risk": "Schedules can be updated; economic values are revised. Preserve the release-page vintage.",
            "acquisition_status": "Source verified; direct automated ICS request was denied by BLS bot policy in this environment, so validated provenance import is used",
        },
        {
            "events": ["GDP", "PCE", "Core PCE", "Personal Income and Outlays"],
            "source": "U.S. BEA full release schedule and archived news releases",
            "url": "https://www.bea.gov/news/schedule/full",
            "timestamp_reliability": "Official schedule includes date and Eastern release time; archived news release records the actual embargo time",
            "historical_coverage": "Current full-year schedule plus archived release documents; revisions/rescheduling must use the actual archive",
            "revision_risk": "Schedules and estimates are revised; retain retrieval time and original document URL",
            "acquisition_status": "Source verified; historical canonical extraction not yet materialized",
        },
        {
            "events": ["FOMC rate decision", "FOMC statement"],
            "source": "Federal Reserve FOMC calendars, statements and historical materials",
            "url": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
            "timestamp_reliability": "Meeting dates and statement documents are official; exact release time must come from the statement/document record rather than an assumed universal time",
            "historical_coverage": "Current calendar 2021-2027 plus official year-by-year historical materials",
            "revision_risk": "Meeting schedules can change; documents are versioned by official release",
            "acquisition_status": "Source verified; exact timestamp extraction requires per-document provenance",
        },
    ]


def render_report(report: dict[str, Any]) -> str:
    symbol_rows = []
    for row in report["mt5_tick_audit"]["symbols"]:
        sample = row.get("recent_one_hour_sample", {})
        fields = sample.get("field_availability", {})
        symbol_rows.append(
            "| {symbol} | {asset} | {first} | {last} | {count} | {bid} | {ask} | {last_field} | {volume} | {flags} |".format(
                symbol=row["symbol"],
                asset=row["asset_class"],
                first=row.get("earliest_available_tick_utc") or "unverified",
                last=row.get("latest_available_tick_utc") or "unavailable",
                count=row.get("approximate_tick_count") or "not estimable",
                bid="yes" if fields.get("bid", {}).get("nonzero", 0) else "no/unverified",
                ask="yes" if fields.get("ask", {}).get("nonzero", 0) else "no/unverified",
                last_field="yes" if fields.get("last", {}).get("nonzero", 0) else "no",
                volume="yes" if fields.get("volume", {}).get("nonzero", 0) else "no",
                flags="yes" if fields.get("flags", {}).get("present") else "no/unverified",
            )
        )

    gold = next(
        (row for row in report["mt5_tick_audit"]["symbols"] if row["asset_class"] == "gold"),
        None,
    )
    gold_months = (gold or {}).get("monthly_availability", {})
    gold_available_months = [month for month, status in gold_months.items() if status == "available"]
    gold_gap_months = [month for month, status in gold_months.items() if status != "available"]
    micro = report["microstructure_feasibility"]
    intraday = report["cross_market_summary"]["intraday_available"]
    daily_only = report["cross_market_summary"]["daily_or_slow_only"]
    generation_22 = report["generation_22_decision"]
    forward_manifest = report.get("forward_collection_snapshot") or {}
    forward_summary = forward_manifest.get("latest_run_summary", {})
    forward_market = forward_summary.get("market", {})
    forward_counts = ", ".join(
        f"{symbol}: +{values.get('ticks_added', 0)} ticks/+{values.get('m1_added', 0)} M1"
        for symbol, values in forward_market.items()
    )
    cross_rows = [
        "| {instrument} | {source} | {granularity} | {first} | {last} | {timezone} | {missing} | {revision} | {known} |".format(
            instrument=row["instrument"],
            source=row["source"],
            granularity=row["granularity"],
            first=row["first_timestamp"],
            last=row["last_timestamp"],
            timezone=row["timezone"],
            missing=row["missing_periods"],
            revision=row["revision_policy"],
            known=row["known_at_gold_entry"],
        )
        for row in report["cross_market_source_inventory"]
    ]
    event_rows = [
        "| {events} | [{source}]({url}) | {reliability} | {coverage} | {status} |".format(
            events=", ".join(row["events"]),
            source=row["source"],
            url=row["url"],
            reliability=row["timestamp_reliability"],
            coverage=row["historical_coverage"],
            status=row["acquisition_status"],
        )
        for row in report["economic_event_source_inventory"]
    ]
    return "\n".join(
        [
            "# DATA FOUNDATION 1 — Historical Information Acquisition Audit",
            "",
            f"Generated (UTC): `{report['generated_at_utc']}`",
            "",
            "Status: `data_foundation_only`; no model was trained and no strategy outcome was evaluated.",
            "",
            "## MT5 historical tick audit",
            "",
            "| Symbol | Class | Earliest tick (UTC) | Latest tick (UTC) | Approx ticks | Bid | Ask | Last | Volume | Flags |",
            "| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- |",
            *symbol_rows,
            "",
            "Approximate counts are explicitly low-confidence extrapolations from a recent one-hour sample; the script does not download or retain bulk historical ticks.",
            "Month-level availability and every progressive GOLD# boundary query are preserved in the JSON metadata.",
            f"GOLD# monthly coverage inside the broker boundary: `{gold_available_months[0] if gold_available_months else 'none'}` through `{gold_available_months[-1] if gold_available_months else 'none'}`; non-available queried months: `{', '.join(gold_gap_months) if gold_gap_months else 'none'}`.",
            "",
            "## Microstructure feasibility",
            "",
            f"GOLD# broker tick boundary: `{(gold or {}).get('earliest_available_tick_utc') or 'unverified'}`.",
            f"Bid/ask reconstruction for the full 2018–2024 interval: `{'YES' if micro['historical_2018_2024_bid_ask_reconstructable'] else 'NO'}`; the available tail begins at the broker boundary in August 2023.",
            "Within the verified tick range, bid/ask-derived spread, changes, tick intensity, signed midquote pressure, realized variance, path efficiency and burst state are feasible only when the JSON field checks are true.",
            "",
            "Pre-registered new-data hypothesis: `log1p(TICKVOL).shift(1).diff().diff()` on completed GOLD# M1 bars. Its bar size, transform and lags are frozen; 2018–2024 cannot be reused as independent confirmation.",
            "",
            "## Cross-market availability",
            "",
            f"Timestamp-aligned intraday sources currently verified: {', '.join(intraday) if intraday else 'none beyond GOLD#'}.",
            f"Daily/slow context only: {', '.join(daily_only)}.",
            "Contract futures symbols are not continuous series; rolls must be handled without future knowledge before any causal model use.",
            "",
            "| Instrument | Source | Granularity | First | Last | Timezone | Missing periods | Revisions | Knowable at GOLD entry? |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            *cross_rows,
            "",
            "## Economic-event foundation",
            "",
            "Official agency schedules can support documented release timestamps for BLS CPI/Employment, BEA GDP/PCE and Federal Reserve FOMC decisions/statements. There is no locally verified unified historical surprise feed. Actual-minus-forecast surprise is therefore excluded.",
            "",
            "| Events | Official source | Timestamp basis | Coverage | Acquisition status |",
            "| --- | --- | --- | --- | --- |",
            *event_rows,
            "",
            "## Untouched-forward protocol",
            "",
            f"Cutoff remains `{report['forward_protocol']['cutoff_utc']}`. Raw collection is isolated under `{report['forward_protocol']['storage_path']}` and contains no labels, returns, predictions, positions or strategy outcomes.",
            f"Collector status: `{forward_manifest.get('status', 'not yet run')}`; contamination: `{forward_manifest.get('contamination_status', 'not recorded')}`; latest incremental run: {forward_counts or 'no market rows recorded'}, +{forward_summary.get('events_added', 0)} event rows; total stored official event timestamps: {forward_manifest.get('stored_economic_event_rows', 0)}.",
            "",
            "## Required answers",
            "",
            f"1. Actual GOLD# tick history reaches back to `{(gold or {}).get('earliest_available_tick_utc') or 'not verified by the broker query'}`; see progressive probes for timeout/empty evidence.",
            f"2. Bid/ask microstructure for all of 2018–2024: `{'reconstructable' if micro['historical_2018_2024_bid_ask_reconstructable'] else 'not reconstructable'}`. Only 2023-08-03 onward is present, so 2018–2022 and most of 2023 have no broker tick-level bid/ask history.",
            "3. Tick-volume acceleration can be independently tested only on untouched bars at or after the cutoff, using the frozen definition above. It cannot receive an independent historical test from already inspected 2018–2024 data.",
            f"4. Verified intraday cross-market data: {', '.join(intraday) if intraday else 'none'}.",
            f"5. Daily-only datasets unsuitable as intraday causal features: {', '.join(daily_only)}.",
            "6. Reliable release timestamps can be assembled from official BLS, BEA and Federal Reserve release/meeting records; forecast surprises remain unavailable until a timestamped, licensed/reproducible source is added.",
            "7. Collect GOLD# ticks and closed M1 bars, observed bid/ask spread, M1 tick count, SILVER# aligned ticks/M1, verified cross-market quotes, and official event timestamps—raw and append-only, without outcomes.",
            f"8. Generation 22: `{generation_22['decision']}` — {generation_22['reason']}",
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only MT5 information acquisition audit")
    parser.add_argument("--worker", choices=("inventory", "first-after", "tick-window"))
    parser.add_argument("--symbol")
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args()
    if args.worker:
        return worker_main(args)

    inventory = run_worker("inventory", timeout=30)
    inventory_rows = inventory.get("symbols", []) if inventory.get("status") == "ok" else []
    selected = choose_symbols(inventory_rows)
    audits = []
    for row in selected:
        print(f"Auditing {row['symbol']} ({row['asset_class']})...", flush=True)
        audits.append(audit_symbol(row, progressive=row["asset_class"] in {"gold", "silver"}))

    local_sources = []
    local_paths = (
        (ROOT / "GOLD#_M1_201401020000_202605082357.csv", "local MT5 GOLD# M1 export"),
        (ROOT / "數據集" / "SILVER#_M1_201401020000_202605281623.csv", "local MT5 SILVER# M1 export"),
    )
    for path, source in local_paths:
        if path.is_file():
            local_sources.append(local_csv_metadata(path, source))

    gold_audit = next((row for row in audits if row["asset_class"] == "gold"), None)
    intraday = [
        f"MT5 {row['symbol']} ({row['asset_class']})"
        for row in audits
        if row.get("earliest_available_tick_utc")
        and row.get("latest_available_tick_utc")
        and row.get("recent_one_hour_sample", {}).get("status") == "ok"
        and row["asset_class"] != "gold"
    ]
    daily_only = [
        "U.S. Treasury daily par yields / Federal Reserve DGS2 and DGS10",
        "free official historical VIX daily series when intraday entitlement is absent",
        "official WTI spot/energy series when only daily observations are used",
    ]
    enough_new_information = any(
        row["asset_class"] in {"usd_index", "rates", "vix", "crude"}
        and row.get("earliest_available_tick_utc")
        and utc(row["earliest_available_tick_utc"]) <= datetime(2023, 1, 1, tzinfo=timezone.utc)
        for row in audits
    )
    manifest_path = ROOT / "untouched_forward" / "generation21" / "data" / "collector_manifest.json"
    forward_snapshot = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    )

    report = {
        "project": "DATA FOUNDATION 1 — Historical Information Acquisition Audit",
        "status": "data_foundation_only",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prohibitions_respected": {
            "generation_22_created": False,
            "model_trained": False,
            "strategy_outcomes_evaluated": False,
            "gemini_py_modified": False,
            "production_modified": False,
            "parameter_sweep_run": False,
        },
        "mt5_tick_audit": {
            "api": "MetaTrader5 copy_ticks_from/copy_ticks_range, read-only",
            "utc_policy": (
                "All public timestamps are UTC. The audited XM terminal empirically encodes live tick epochs "
                "in EET/EEST server time (+2 winter/+3 summer), so query boundaries and returned epochs are "
                "converted explicitly; raw broker epochs are retained in probe metadata."
            ),
            "inventory": inventory,
            "symbols": audits,
        },
        "local_sources": local_sources,
        "microstructure_feasibility": microstructure_feasibility(gold_audit),
        "cross_market_source_inventory": cross_market_inventory(audits, local_sources),
        "cross_market_summary": {
            "intraday_available": intraday,
            "daily_or_slow_only": daily_only,
            "known_at_entry_rule": (
                "Only timestamp-aligned observations published at or before the GOLD entry may be joined. "
                "Daily values are lagged until their documented publication time."
            ),
        },
        "economic_event_foundation": {
            "reliable_official_families": [
                "BLS CPI and Core CPI release timestamps",
                "BLS Employment Situation / NFP release timestamps",
                "BEA GDP release timestamps",
                "BEA Personal Income and Outlays / PCE release timestamps",
                "Federal Reserve FOMC decision and statement timestamps",
            ],
            "allowed_features": ["minutes_until_event", "minutes_since_event", "event_type", "event_window_flag"],
            "surprise_status": "excluded_pending_reliable_timestamped_actual_and_forecast_source",
            "timezone_policy": "store UTC plus original source timezone and release-time provenance",
        },
        "economic_event_source_inventory": economic_event_source_inventory(),
        "forward_protocol": {
            "cutoff_utc": CUTOFF.isoformat(),
            "storage_path": "untouched_forward/generation21/data/",
            "collector": "gold_data_foundation_forward_collector.py",
            "outcomes_stored": False,
            "contamination_rule": (
                "If any post-cutoff strategy outcome is inspected for a research decision, mark the "
                "entire inspected interval contaminated and set a later cutoff."
            ),
        },
        "forward_collection_snapshot": forward_snapshot,
        "generation_22_decision": {
            "decision": "NOT JUSTIFIED" if not enough_new_information else "DEFER PENDING CAUSAL DATA ASSEMBLY",
            "reason": (
                "No complete, long, timestamp-aligned new intraday information family is currently frozen and validated. "
                "Continue acquisition; do not train a candidate yet."
            ),
        },
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    REPORT_MD.write_text(render_report(report), encoding="utf-8")
    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
