import unittest

import numpy as np
import pandas as pd

import rules_lab_v1 as rl


def h1_frame(days=3, seed=0):
    rng = np.random.default_rng(seed)
    times = pd.date_range("2026-06-01", periods=24 * days, freq="h")
    close = 2000 + np.cumsum(rng.normal(0, 2, len(times)))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"TIME_DT": times, "OPEN": open_, "CLOSE": close, "SPREAD": 30,
                         "HIGH": np.maximum(open_, close) + 1, "LOW": np.minimum(open_, close) - 1})


class RulesLabTests(unittest.TestCase):
    def test_no_lookahead(self):
        base = rl.prepare(h1_frame())
        bumped = base.copy()
        bumped.loc[len(bumped) - 1, ["CLOSE", "HIGH"]] += 500  # change only the last bar
        up = pd.Series(1, index=base["TIME_DT"].dt.normalize().unique())
        for rule in ("donchian20", "rsi2", "engulf_trend", "pinbar_trend", "asia_break", "london_orb"):
            for side in ("long", "short"):
                a = rl.signals(base, "H1", rule, side, up)
                b = rl.signals(bumped, "H1", rule, side, up)
                self.assertEqual(a[-1], b[-1], f"{rule} {side} used the bar it enters on")

    def test_session_breakout_once_per_day(self):
        frame = rl.prepare(h1_frame(days=5, seed=3))
        sig = rl.signals(frame, "H1", "asia_break", "long")
        per_day = pd.Series(sig).groupby(frame["TIME_DT"].dt.normalize()).sum()
        self.assertTrue((per_day <= 1).all())
        self.assertTrue(frame.loc[sig, "TIME_DT"].dt.hour.between(11, 17).all())


if __name__ == "__main__":
    unittest.main()
