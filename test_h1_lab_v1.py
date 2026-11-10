import unittest

import numpy as np
import pandas as pd

from h1_lab_v1 import FEATURES, MIN_SPREAD_FRACTION, make_features


def bars(n, freq, start="2020-01-01", seed=0):
    rng = np.random.default_rng(seed)
    close = 1500 + rng.normal(0, 2, n).cumsum()
    return pd.DataFrame({"TIME_DT": pd.date_range(start, periods=n, freq=freq), "OPEN": close,
                         "HIGH": close + 1, "LOW": close - 1, "CLOSE": close, "SPREAD": 0})


class MakeFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.h1 = bars(600, "h")
        self.higher = {"H4": bars(300, "4h", "2019-12-20"), "Daily": bars(100, "D", "2019-10-01"),
                       "Weekly": bars(60, "W", "2018-12-01"), "Monthly": bars(40, "MS", "2017-01-01")}

    def test_no_lookahead_into_entry_bar(self):
        base = make_features(self.h1, self.higher)
        changed = self.h1.copy()
        changed.loc[len(changed) - 1, ["HIGH", "LOW", "CLOSE"]] += 50   # alter the last (entry) bar only
        after = make_features(changed, self.higher)
        pd.testing.assert_frame_equal(base[FEATURES + ["ATR"]].tail(1), after[FEATURES + ["ATR"]].tail(1))

    def test_cross_features_use_only_closed_bars(self):
        cross = {"USDX": bars(600, "h", seed=1)}
        base = make_features(self.h1, self.higher, cross)
        changed = {"USDX": cross["USDX"].copy()}
        changed["USDX"].loc[len(changed["USDX"]) - 1, "CLOSE"] += 50    # the other market's bar at entry time
        after = make_features(self.h1, self.higher, changed)
        cols = [c for c in base.columns if c.startswith("USDX_ROC_")]
        self.assertEqual(len(cols), 3)
        pd.testing.assert_frame_equal(base[cols].tail(1), after[cols].tail(1))

    def test_spread_floor(self):
        frame = make_features(self.h1, self.higher)
        floor = frame["OPEN"] * MIN_SPREAD_FRACTION / 0.01
        self.assertTrue(np.allclose(frame["SPREAD"], floor))


if __name__ == "__main__":
    unittest.main()
