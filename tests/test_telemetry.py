from __future__ import annotations

import unittest

from core.telemetry import increment, record_step, reset, snapshot


class TelemetryTests(unittest.TestCase):
    def setUp(self) -> None:
        reset()

    def test_increment_and_snapshot_are_recorded(self) -> None:
        increment("launches")
        increment("launches", 2)

        data = snapshot()

        self.assertEqual(data["counters"]["launches"], 3)

    def test_record_step_upserts_by_step_and_player(self) -> None:
        record_step(4, 0, {"value": 1})
        record_step(4, 0, {"value": 2})
        record_step(5, 0, {"value": 3})

        data = snapshot()

        self.assertEqual(len(data["steps"]), 2)
        self.assertEqual(data["steps"][0]["value"], 2)
        self.assertEqual(data["steps"][1]["step"], 5)


if __name__ == "__main__":
    unittest.main()
