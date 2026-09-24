"""Version 2 adds October/November windows and explicit per-sample metadata; timestamps only."""
import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from gold_mt5_timestamp_semantics_diagnostic_v1 import iso, legacy

TERMINAL = r"D:\XM2\terminal64.exe"
SYMBOL = "GOLD#"
# These are sampling windows, not asserted broker transition dates.
WINDOWS = [("winter", "2026-01-15T12:00:00+00:00"),
           ("summer", "2026-07-15T12:00:00+00:00"),
           ("US_transition_before", "2026-03-06T12:00:00+00:00"),
           ("US_transition_after", "2026-03-09T12:00:00+00:00"),
           ("EU_transition_before", "2026-03-27T12:00:00+00:00"),
           ("EU_transition_after", "2026-03-30T12:00:00+00:00"),
           ("October_before", "2025-10-24T12:00:00+00:00"),
           ("October_after", "2025-10-27T12:00:00+00:00"),
           ("US_autumn_before", "2025-10-31T12:00:00+00:00"),
           ("US_autumn_after", "2025-11-03T12:00:00+00:00")]


def summarize(epochs):
    spacing = [b - a for a, b in zip(epochs, epochs[1:])]
    return {"raw_epochs": epochs, "interpreted_as_utc": [iso(t) for t in epochs],
            "hypothesis_converted": [legacy(t).isoformat() for t in epochs],
            "spacing_seconds": spacing, "monotonic": all(t > 0 for t in spacing),
            "offset_inference": None,
            "sample_source": "GOLD# M1 copy_rates_range",
            "classification": "INSUFFICIENT_DATA",
            "timestamp_rows": [{"raw_epoch": t, "raw_epoch_interpreted_direct_utc": iso(t),
                "candidate_server_offset_seconds": int(t - legacy(t).timestamp()),
                "legacy_conversion_utc": legacy(t).isoformat(),
                "difference_from_independent_anchor_if_available": None} for t in epochs],
            "reason": "Historical epochs alone have no independent UTC anchor; conversion is a hypothesis"}


def current_classification(raw_epoch, observed_utc):
    delta = legacy(raw_epoch).timestamp() - observed_utc
    if abs(delta) <= 180:
        return "SUPPORTS_BROKER_SERVER_RULE"
    if abs(raw_epoch - observed_utc) <= 180:
        return "CONTRADICTS_BROKER_SERVER_RULE"
    return "INSUFFICIENT_DATA"  # A stale tick is not evidence of another offset.


def diagnose():
    report = {"classification": "INSUFFICIENT_DATA", "samples": [], "current": None,
              "symbol": SYMBOL, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_identity_currently_verified": False,
              "limitations": ["No account_info call is permitted; exact current server/environment is not reverified",
                  "Prior archived demo identity is historical evidence only",
                  "No independent historical UTC anchor or authoritative recurrence; no annual certification"],
              "price_fields_inspected": False, "strategy_outcome_inspected": False}
    import MetaTrader5 as mt5
    initialized = False
    try:
        initialized = mt5.initialize(TERMINAL, timeout=10000)
        if not initialized:
            raise RuntimeError("Terminal unavailable")
        terminal = mt5.terminal_info()
        symbol = mt5.symbol_info(SYMBOL)
        if terminal is None or not terminal.connected or symbol is None:
            raise RuntimeError("Source unavailable")
        if symbol.name != SYMBOL or symbol.digits != 2 or symbol.point != 0.01:
            raise ValueError("Symbol identity mismatch")
        del terminal, symbol
        tick = mt5.symbol_info_tick(SYMBOL)
        now = datetime.now(timezone.utc)
        if tick is not None:
            stamp = int(tick.time)
            del tick
            report["current"] = {"raw_epoch": stamp, "observed_utc": now.isoformat(),
                "hypothesis_converted": legacy(stamp).isoformat(),
                "raw_minus_system_seconds": stamp - now.timestamp(),
                "hypothesis_minus_system_seconds": legacy(stamp).timestamp() - now.timestamp(),
                "classification": current_classification(stamp, now.timestamp())}
        for label, text in WINDOWS + [("current", now.isoformat())]:
            start = datetime.fromisoformat(text)
            end = start + timedelta(minutes=5)
            if label == "current":
                start, end = now - timedelta(minutes=5), now
            rates = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M1, start, end)
            epochs = [] if rates is None else [int(row["time"]) for row in rates]
            del rates
            if len(epochs) > 6:
                raise ValueError("Bound exceeded")
            report["samples"].append({"label": label, "query_start_utc": start.isoformat(),
                "query_end_utc": end.isoformat(), "requested_utc": [start.isoformat(), end.isoformat()], **summarize(epochs)})
        # Current support does not establish consistency across historical periods.
        if report["current"] and report["current"]["classification"] == "CONTRADICTS_BROKER_SERVER_RULE":
            report["classification"] = "CONTRADICTS_BROKER_SERVER_RULE"
    except Exception as error:
        report["error_type"] = type(error).__name__  # Never print terminal structures.
        report["classification"] = "UNRESOLVED"
    finally:
        if initialized:
            mt5.shutdown()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or (args.output.parent / "FINALIZED.json").exists():
        raise FileExistsError("Immutable output")
    result = diagnose()
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(result["classification"])
