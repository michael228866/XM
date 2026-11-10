import unittest

import numpy as np
import pandas as pd

import trend_daily_v1 as td


def closes(values):
    idx = pd.bdate_range("2020-01-01", periods=len(values))
    return pd.DataFrame({"A": values}, index=idx)


class TrendDailyTests(unittest.TestCase):
    def test_uptrend_goes_long_downtrend_short(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 0.002, 400)
        up = closes(100 * np.cumprod(1 + 0.001 + noise))
        down = closes(100 * np.cumprod(1 - 0.001 + noise))
        self.assertGreater(td.weights_from(up)["A"].iloc[-1], 0)
        self.assertLess(td.weights_from(down)["A"].iloc[-1], 0)

    def test_no_lookahead(self):
        rng = np.random.default_rng(1)
        base = closes(100 * np.cumprod(1 + rng.normal(0, 0.01, 400)))
        bumped = base.copy()
        bumped.iloc[-1] *= 1.5                      # change only the last day's close
        costs = {"A": {"spread_frac": 0.0, "swap_long_frac": 0.0, "swap_short_frac": 0.0}}
        held_a = td.backtest(base, costs)[2]
        held_b = td.backtest(bumped, costs)[2]
        # The position held on the last day was decided before that close was known.
        self.assertEqual(held_a["A"].iloc[-1], held_b["A"].iloc[-1])

    def test_costs_reduce_returns(self):
        rng = np.random.default_rng(2)
        c = closes(100 * np.cumprod(1 + 0.0008 + rng.normal(0, 0.01, 400)))
        free = {"A": {"spread_frac": 0.0, "swap_long_frac": 0.0, "swap_short_frac": 0.0}}
        paid = {"A": {"spread_frac": 0.001, "swap_long_frac": -0.0002, "swap_short_frac": -0.0002}}
        self.assertLess(td.backtest(c, paid)[1]["A"].sum(), td.backtest(c, free)[1]["A"].sum())


if __name__ == "__main__":
    unittest.main()
