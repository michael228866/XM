import unittest

import numpy as np
import pandas as pd

from strategy_lab_v1 import first_touch, trade


def bars(rows):
    frame = pd.DataFrame(rows, columns=["OPEN", "HIGH", "LOW", "CLOSE"])
    frame["SPREAD"] = 0
    frame["ATR"] = 1.0
    frame["TIME_DT"] = pd.date_range("2026-01-01", periods=len(frame), freq="min")
    return frame


class FirstTouchTests(unittest.TestCase):
    def test_long_tp_sl_tie_and_timeout(self):
        frame = bars([(100, 101.5, 99.9, 101), (101, 101, 98.3, 99), (99, 99.2, 98.9, 99.1), (99, 99, 99, 99)])
        r, after = first_touch(frame, "long", 1.0, 1.0, 2)
        self.assertEqual((r[0], after[0]), (1.0, 1))      # TP on entry bar
        self.assertEqual((r[1], after[1]), (-1.0, 1))     # SL
        self.assertAlmostEqual(r[2], 0.0)                 # timeout at close of last held bar: 99 - 99
        self.assertTrue(np.isnan(r[3]))                   # data ends before the hold resolves

    def test_short_side_and_tie_counts_as_stop(self):
        frame = bars([(100, 101.2, 98.8, 100), (100, 100, 100, 100)])
        r, _ = first_touch(frame, "short", 1.0, 1.0, 2)
        self.assertEqual(r[0], -1.0)                      # both touched in one bar -> stop first

    def test_spread_charged_on_long_entry(self):
        frame = bars([(100, 101.4, 99.6, 100.5), (100.5, 100.5, 100.5, 100.5)])
        frame["SPREAD"] = 50                              # 0.50 price
        r, _ = first_touch(frame, "long", 1.0, 1.0, 1)
        self.assertAlmostEqual(r[0], 0.0)                 # entry 100.5, no touch, exits at close 100.5

    def test_one_position_at_a_time(self):
        frame = bars([(100, 100, 100, 100)] * 5)
        taken = trade(frame, np.array([.9, .9, .9, .9, .9]), .5, np.zeros(5), np.array([2, 2, 2, 2, 2]))
        self.assertEqual(list(taken.index), [0, 2, 4])


if __name__ == "__main__":
    unittest.main()
