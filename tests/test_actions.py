from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

from core.actions import (
    ActionConversionError,
    candidate_to_action,
    safe_candidate_to_action,
    safe_noop,
    to_kaggle_action,
)


@dataclass(slots=True)
class ObjectCandidate:
    source_id: int
    angle: float
    ships: int


class ActionConversionTests(unittest.TestCase):
    def test_converts_valid_fields_to_kaggle_action_shape(self) -> None:
        self.assertEqual(to_kaggle_action(source_id=4, angle=1.25, ships=17), [4, 1.25, 17])

    def test_preserves_finite_angle_without_silent_normalization(self) -> None:
        action = to_kaggle_action(source_id=4, angle=math.tau + 0.5, ships=17)

        self.assertEqual(action, [4, math.tau + 0.5, 17])

    def test_converts_dict_candidate(self) -> None:
        candidate = {"source_id": 3, "angle": 2.0, "ships": 11}

        self.assertEqual(candidate_to_action(candidate), [3, 2.0, 11])

    def test_converts_object_candidate(self) -> None:
        candidate = ObjectCandidate(source_id=2, angle=0.75, ships=9)

        self.assertEqual(candidate_to_action(candidate), [2, 0.75, 9])

    def test_stop_candidate_converts_to_noop(self) -> None:
        self.assertEqual(candidate_to_action({"kind": "STOP"}), [])

    def test_none_candidate_converts_to_noop(self) -> None:
        self.assertEqual(candidate_to_action(None), [])

    def test_safe_noop_returns_empty_action(self) -> None:
        self.assertEqual(safe_noop(), [])

    def test_rejects_invalid_source_id(self) -> None:
        with self.assertRaises(ActionConversionError):
            to_kaggle_action(source_id=-1, angle=1.0, ships=5)

    def test_rejects_invalid_ships(self) -> None:
        with self.assertRaises(ActionConversionError):
            to_kaggle_action(source_id=1, angle=1.0, ships=0)

    def test_rejects_non_finite_angle(self) -> None:
        with self.assertRaises(ActionConversionError):
            to_kaggle_action(source_id=1, angle=math.inf, ships=5)

    def test_safe_conversion_returns_noop_for_invalid_candidate(self) -> None:
        candidate = {"source_id": 1, "angle": math.nan, "ships": 5}

        self.assertEqual(safe_candidate_to_action(candidate), [])


if __name__ == "__main__":
    unittest.main()
