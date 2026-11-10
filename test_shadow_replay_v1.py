"""Offline tests for shadow_replay_v1; MT5 and XGBoost are faked like the gemini foundation tests."""
import sys
import types
import unittest
from unittest.mock import patch

import pandas as pd

from test_demo_trading_foundation_v1 import FakeMT5

with patch.dict(sys.modules, {"MetaTrader5": FakeMT5(), "xgboost": types.ModuleType("xgboost")}):
    import shadow_replay_v1 as replay


def frame(times, opens, highs, lows):
    return pd.DataFrame({"TIME_DT": pd.to_datetime(times, utc=True), "OPEN": opens,
                         "HIGH": highs, "LOW": lows, "CLOSE": opens})


class ShadowReplayTests(unittest.TestCase):
    def test_forming_bars_are_never_visible(self):
        m1 = frame(["2026-10-01 11:58", "2026-10-01 11:59", "2026-10-01 12:00"], [1, 2, 3], [1, 2, 3], [1, 2, 3])
        h1 = frame(["2026-10-01 10:00", "2026-10-01 11:00", "2026-10-01 12:00"], [1, 2, 3], [1, 2, 3], [1, 2, 3])
        bars = replay.ClosedBars({"M1": m1, "H1": h1})
        bars.now = pd.Timestamp("2026-10-01 12:00", tz="UTC")  # decision for the 11:59 M1 bar
        self.assertEqual(bars("GOLD#", replay.TIMEFRAMES["M1"], 400)["OPEN"].tolist(), [1, 2])
        self.assertEqual(bars("GOLD#", replay.TIMEFRAMES["H1"], 80)["OPEN"].tolist(), [1, 2])
        bars.now = pd.Timestamp("2026-10-01 11:30", tz="UTC")
        self.assertEqual(bars("GOLD#", replay.TIMEFRAMES["H1"], 80)["OPEN"].tolist(), [1])
        bars.now = pd.Timestamp("2026-10-01 10:30", tz="UTC")
        self.assertIsNone(bars("GOLD#", replay.TIMEFRAMES["H1"], 80))

    def test_exit_rules(self):
        times = pd.date_range("2026-10-01 10:00", periods=100, freq="min", tz="UTC")
        flat = [100.0] * 100
        cases = {  # bar index -> (high, low, open) override, expected (reason, exit price)
            "tp": ({3: (102.5, 99.5, 100.0)}, ("tp", 102.0)),
            "sl": ({3: (100.5, 98.0, 100.0)}, ("sl", 99.0)),
            "both_same_bar_stop_first": ({3: (103.0, 98.0, 100.0)}, ("sl", 99.0)),
            "gap_through_sl_fills_at_open": ({3: (97.0, 96.0, 96.5)}, ("sl", 96.5)),
            "timeout_at_open_after_90_min": ({}, ("timeout", 100.0)),
        }
        for name, (overrides, expected) in cases.items():
            highs, lows, opens = list(flat), list(flat), list(flat)
            for i, (high, low, open_) in overrides.items():
                highs[i], lows[i], opens[i] = high, low, open_
            result = replay.simulate_exit(frame(times, opens, highs, lows), 100.0, 1.0, 2.0)
            self.assertEqual((result["exit_reason"], result["exit_price"]), expected, name)
        self.assertEqual(result["exit_time"], times[90])
        self.assertEqual(replay.simulate_exit(frame(times[:5], flat[:5], flat[:5], flat[:5]),
                                              100.0, 1.0, 2.0)["exit_reason"], "data_end")


if __name__ == "__main__":
    unittest.main()
