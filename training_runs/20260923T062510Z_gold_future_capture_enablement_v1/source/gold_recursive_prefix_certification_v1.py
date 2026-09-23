"""Document-only recursive prefix investigation. No data/model execution."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def investigate(pipeline):
    raw = pipeline.read_bytes()
    text = raw.decode("utf-8-sig")
    recursive = all(value in text for value in ("ewm(span=12, adjust=False)", "ewm(span=26, adjust=False)", "ewm(span=9, adjust=False)"))
    return {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "source_id": "XMGlobal-MT5-6_GOLD#",
        "symbol": "GOLD#", "required_history_start": None, "available_history_start": None,
        "continuous_history": False, "same_timestamp_semantics": False, "same_timeframe_semantics": False,
        "same_feature_initialization": False, "recursive_state_equivalent": False,
        "prefix_policy": "UNRESOLVED", "status": "PARTIAL", "pipeline_sha256": hashlib.sha256(raw).hexdigest(),
        "recursive_ema_macd_detected": recursive, "strategy_outcome_inspected": False,
        "blockers": ["Original native-file first-row identity and exact recursive state are not supplied as a certified source-bound prefix",
            "Small current API samples do not prove full historical continuity or identical native bars",
            "254 finite warmup rows cannot reproduce exact recursive EMA/MACD state",
            "Historical context is development-only; reinitialization requires a separately frozen protocol change"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = investigate(Path(__file__).resolve().parent / "drl_trading_v2.py")
    if args.self_test:
        assert result["recursive_ema_macd_detected"] and not result["recursive_state_equivalent"]
        assert result["prefix_policy"] == "UNRESOLVED" and result["required_history_start"] is None
        print("SELF_TEST_PASS")
    else:
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        print(result["prefix_policy"])


if __name__ == "__main__":
    main()
