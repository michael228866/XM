"""Bounded read-only MT5 timing diagnostics; never persist prices or account data."""
import argparse
import calendar
import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

SYMBOL = "GOLD#"
SERVER = "XMGlobal-MT5 6"
COMPANY = "XM Global Limited"
TERMINAL = r"D:\XM2\terminal64.exe"


def iso(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def legacy(epoch):
    """Exact legacy hypothesis, not an authoritative conversion rule."""
    raw = datetime.fromtimestamp(epoch, timezone.utc)
    for hours in (2, 3):
        candidate = raw - timedelta(hours=hours)
        bounds = []
        for month in (3, 10):
            last = datetime(candidate.year, month, calendar.monthrange(candidate.year, month)[1], 1, tzinfo=timezone.utc)
            bounds.append(last - timedelta(days=(last.weekday() + 1) % 7))
        offset = 3 if bounds[0] <= candidate < bounds[1] else 2
        if offset == hours:
            return candidate
    raise ValueError("Ambiguous legacy transform")


@contextmanager
def session():
    import MetaTrader5 as mt5
    if not mt5.initialize(TERMINAL, timeout=10000):
        raise RuntimeError("MT5 unavailable")
    try:
        info = mt5.terminal_info()
        symbol = mt5.symbol_info(SYMBOL)
        # No repr, asdict, serialization or exception text of this structure.
        account = mt5.account_info()
        if account is None:
            raise RuntimeError("Source identity unavailable")
        company, server, mode = account.company, account.server, account.trade_mode
        del account
        if company != COMPANY or server != SERVER:
            raise RuntimeError("Unexpected broker source")
        if info is None or not info.connected or symbol is None or symbol.name != SYMBOL:
            raise RuntimeError("Source not connected")
        if symbol.digits != 2 or symbol.point != 0.01:
            raise RuntimeError("Symbol precision changed")
        if not symbol.visible:
            raise RuntimeError("Symbol not visible; no terminal mutation performed")
        metadata = {"terminal_build": int(info.build), "broker_company": company,
            "broker_server": server, "symbol": SYMBOL, "digits": int(symbol.digits), "point": float(symbol.point),
            "source_account_environment": "demo" if mode == 0 else "live" if mode == 2 else "unknown"}
        del info, symbol
        yield mt5, metadata
    finally:
        mt5.shutdown()


def classify(raw_time, raw_msc, epochs, system_epoch):
    if not epochs or raw_time <= 0 or raw_msc // 1000 != raw_time:
        return "INCONSISTENT"
    spacing = [b - a for a, b in zip(epochs, epochs[1:])]
    if any(x <= 0 or x % 60 for x in spacing) or any(x % 60 for x in epochs):
        return "INCONSISTENT"
    if not 0 <= raw_time - epochs[-1] <= 180:
        return "UNRESOLVED"
    direct = abs(raw_time - system_epoch) <= 180
    converted = abs(legacy(raw_time).timestamp() - system_epoch) <= 180
    if direct and not converted:
        return "DIRECT_UTC_SEMANTICS_SUPPORTED"
    if converted and not direct:
        return "LEGACY_CONVERSION_SUPPORTED"
    return "UNRESOLVED"


def diagnose():
    generated = datetime.now(timezone.utc)
    report = {"generated_at_utc": generated.isoformat(), "symbol": SYMBOL,
        "diagnostic_classification": "UNRESOLVED", "tick_sample": None, "m1_sample": [],
        "bar_spacing_seconds": [], "timestamp_monotonic": None,
        "limitations": "One small current sample cannot certify historical API paths, DST, daily rollover or calendar; clock difference alone is not authority.",
        "strategy_outcome_inspected": False}
    try:
        with session() as (mt5, metadata):
            report.update(metadata)
            tick = mt5.symbol_info_tick(SYMBOL)
            rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, 5)
            system = datetime.now(timezone.utc)
            if tick is None or rates is None or len(rates) == 0:
                raise RuntimeError("No diagnostic sample")
            raw_time, raw_msc = int(tick.time), int(tick.time_msc)
            del tick
            epochs = [int(row["time"]) for row in rates]
            del rates
            report["system_current_utc"] = system.isoformat()
            report["tick_sample"] = {"raw_time": raw_time, "raw_time_msc": raw_msc,
                "interpreted_as_utc": iso(raw_time), "delta_from_system_utc_seconds": raw_time - system.timestamp(),
                "legacy_eet_eest_converted": legacy(raw_time).isoformat(),
                "legacy_delta_from_system_utc_seconds": legacy(raw_time).timestamp() - system.timestamp()}
            report["m1_sample"] = [{"raw_epoch": epoch, "interpreted_as_utc": iso(epoch),
                "legacy_eet_eest_converted": legacy(epoch).isoformat(),
                "legacy_minus_direct_seconds": legacy(epoch).timestamp() - epoch} for epoch in epochs]
            report["bar_spacing_seconds"] = [b - a for a, b in zip(epochs, epochs[1:])]
            report["timestamp_monotonic"] = all(x > 0 for x in report["bar_spacing_seconds"])
            report["diagnostic_classification"] = classify(raw_time, raw_msc, epochs, system.timestamp())
            # A second documented API path, only a five-minute direct-UTC window.
            ranged = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M1, system - timedelta(minutes=5), system)
            report["direct_utc_range_query"] = {"start_utc": (system - timedelta(minutes=5)).isoformat(),
                "end_utc": system.isoformat(), "returned_epochs": [] if ranged is None else [int(r["time"]) for r in ranged]}
            del ranged
    except Exception as error:
        report["diagnostic_classification"] = "UNRESOLVED"
        report["diagnostic_error_type"] = type(error).__name__
    return report


def self_test():
    stamp = int(datetime(2030, 7, 1, 12, tzinfo=timezone.utc).timestamp())
    epochs = [stamp - 60, stamp]
    assert classify(stamp, stamp * 1000 + 200, epochs, stamp + 1) == "DIRECT_UTC_SEMANTICS_SUPPORTED"
    assert classify(stamp, stamp * 1000, epochs, stamp - 10800) == "LEGACY_CONVERSION_SUPPORTED"
    assert classify(stamp, stamp * 1000, epochs[::-1], stamp) == "INCONSISTENT"
    assert classify(stamp, 0, epochs, stamp) == "INCONSISTENT"
    assert classify(stamp, stamp * 1000, epochs, stamp + 100000) == "UNRESOLVED"
    assert legacy(stamp).timestamp() - stamp == -10800
    winter = int(datetime(2030, 1, 1, tzinfo=timezone.utc).timestamp())
    assert legacy(winter).timestamp() - winter == -7200
    print("SELF_TEST_PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if args.output.exists():
            raise FileExistsError("Diagnostic output already exists")
        result = diagnose()
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        print(result["diagnostic_classification"])


if __name__ == "__main__":
    main()
