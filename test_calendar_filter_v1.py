import unittest

import numpy as np

from calendar_filter_v1 import near_event


class NearEventTests(unittest.TestCase):
    def test_blackout_window(self):
        events = np.array(["2026-06-05T12:30"], dtype="datetime64[ns]")
        times = ["2026-06-05 11:59:00", "2026-06-05 12:00:00", "2026-06-05 13:00:00", "2026-06-05 13:01:00"]
        # 30 min before through 30 min after the release, inclusive.
        self.assertEqual(list(near_event(times, events)), [False, True, True, False])


if __name__ == "__main__":
    unittest.main()
