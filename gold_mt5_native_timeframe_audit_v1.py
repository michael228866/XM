"""Read three closed native bars per timeframe; persist timing/schema only."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from gold_mt5_timestamp_semantics_diagnostic_v1 import session

FRAMES = {"M1": "TIMEFRAME_M1", "Daily": "TIMEFRAME_D1", "H12": "TIMEFRAME_H12",
    "H8": "TIMEFRAME_H8", "H6": "TIMEFRAME_H6", "H4": "TIMEFRAME_H4", "H3": "TIMEFRAME_H3",
    "H2": "TIMEFRAME_H2", "H1": "TIMEFRAME_H1", "M30": "TIMEFRAME_M30", "M20": "TIMEFRAME_M20",
    "M15": "TIMEFRAME_M15", "M12": "TIMEFRAME_M12", "M10": "TIMEFRAME_M10", "M6": "TIMEFRAME_M6",
    "M5": "TIMEFRAME_M5", "M4": "TIMEFRAME_M4", "M3": "TIMEFRAME_M3", "M2": "TIMEFRAME_M2",
    "Weekly": "TIMEFRAME_W1", "Monthly": "TIMEFRAME_MN1"}
REQUIRED = {"time", "open", "high", "low", "close", "spread", "tick_volume", "real_volume"}


def row_status(names, epochs):
    if not epochs:
        return "NO_DATA_RETURNED"
    if not REQUIRED <= set(names) or any(b <= a for a, b in zip(epochs, epochs[1:])):
        return "SCHEMA_MISMATCH"
    return "AVAILABLE_NATIVE"


def audit():
    results = [{"timeframe": frame, "mt5_constant": constant, "api_supported": None,
        "sample_returned": False, "row_count": 0, "first_timestamp": None, "last_timestamp": None,
        "raw_epochs": [], "monotonic": None, "source_symbol": "GOLD#", "schema": [],
        "spread_available": None, "status": "UNRESOLVED"} for frame, constant in FRAMES.items()]
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "timeframes": results,
        "candidate_policy": "UNRESOLVED", "status": "PARTIAL", "strategy_outcome_inspected": False,
        "limitations": "Three-bar structural availability is not history continuity or timezone certification."}
    try:
        with session() as (mt5, metadata):
            report["source_metadata"] = metadata
            for item in results:
                constant = getattr(mt5, item["mt5_constant"], None)
                item["api_supported"] = constant is not None
                if constant is None:
                    item["status"] = "API_CONSTANT_MISSING"
                    continue
                rates = mt5.copy_rates_from_pos("GOLD#", constant, 1, 3)
                names = [] if rates is None else list(rates.dtype.names)
                epochs = [] if rates is None or "time" not in names else [int(row["time"]) for row in rates]
                del rates
                item.update(schema=names, raw_epochs=epochs, sample_returned=bool(epochs), row_count=len(epochs),
                    first_timestamp=datetime.fromtimestamp(epochs[0], timezone.utc).isoformat() if epochs else None,
                    last_timestamp=datetime.fromtimestamp(epochs[-1], timezone.utc).isoformat() if epochs else None,
                    monotonic=all(b > a for a, b in zip(epochs, epochs[1:])) if epochs else None,
                    spread_available="spread" in names, status=row_status(names, epochs))
    except Exception as error:
        report["diagnostic_error_type"] = type(error).__name__
    if all(item["status"] == "AVAILABLE_NATIVE" for item in results):
        report.update(status="PASS", candidate_policy="NATIVE_20TF_REQUIRED")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        assert len(FRAMES) == 21
        assert row_status(REQUIRED, [1, 2, 3]) == "AVAILABLE_NATIVE"
        assert row_status(REQUIRED - {"spread"}, [1]) == "SCHEMA_MISMATCH"
        assert row_status(REQUIRED, [2, 1]) == "SCHEMA_MISMATCH"
        assert row_status(REQUIRED, []) == "NO_DATA_RETURNED"
        print("SELF_TEST_PASS")
    else:
        if args.output.exists():
            raise FileExistsError("Audit output already exists")
        result = audit()
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        print(result["status"])


if __name__ == "__main__":
    main()
