from __future__ import annotations

import math
import unittest

from core.types import GameState, Planet
from rl.local_state import build_local_turn_state
from rl.masks import validate_and_mask
from rl.types import Candidate, CandidateType


class CandidateMaskTests(unittest.TestCase):
    def _local(self):
        return build_local_turn_state(
            GameState(
                step=0,
                player_id=1,
                planets=[
                    Planet(id=1, owner=1, ships=20, x=0.0, y=0.0),
                    Planet(id=2, owner=2, ships=8, x=20.0, y=0.0),
                    Planet(id=3, owner=-1, ships=4, x=0.0, y=20.0),
                    Planet(id=4, owner=1, ships=5, x=10.0, y=10.0),
                ],
                fleets=[],
            )
        )

    def _attack(self, **overrides):
        values = {
            "type": CandidateType.ATTACK,
            "source_id": 1,
            "target_id": 2,
            "ships": 5,
            "angle": 0.0,
            "eta": 20,
        }
        values.update(overrides)
        return Candidate(**values)

    def test_stop_and_valid_launch_are_legal_and_mask_aligned(self) -> None:
        candidates = [Candidate.stop(), self._attack()]

        batch = validate_and_mask(self._local(), candidates)

        self.assertEqual(batch.mask, (True, True))
        self.assertEqual([candidate.legal for candidate in batch.candidates], [True, True])
        self.assertEqual([candidate.reject_reason for candidate in batch.candidates], [None, None])
        self.assertEqual(batch.selected(1).type, CandidateType.ATTACK)

    def test_invalid_source_rejected(self) -> None:
        batch = validate_and_mask(self._local(), [self._attack(source_id=2)])

        self.assertEqual(batch.mask, (False,))
        self.assertEqual(batch.candidates[0].reject_reason, "invalid_source")

    def test_insufficient_ships_rejected(self) -> None:
        local = self._local()
        local.source_commitments[1] = 18

        batch = validate_and_mask(local, [self._attack(ships=5)])

        self.assertEqual(batch.mask, (False,))
        self.assertEqual(batch.candidates[0].reject_reason, "insufficient_ships")

    def test_invalid_ships_rejected(self) -> None:
        batch = validate_and_mask(self._local(), [self._attack(ships=0)])

        self.assertEqual(batch.mask, (False,))
        self.assertEqual(batch.candidates[0].reject_reason, "invalid_ships")

    def test_invalid_angle_rejected(self) -> None:
        batch = validate_and_mask(self._local(), [self._attack(angle=math.inf)])

        self.assertEqual(batch.mask, (False,))
        self.assertEqual(batch.candidates[0].reject_reason, "invalid_angle")

    def test_invalid_eta_rejected(self) -> None:
        batch = validate_and_mask(self._local(), [self._attack(eta=-1)])

        self.assertEqual(batch.mask, (False,))
        self.assertEqual(batch.candidates[0].reject_reason, "invalid_eta")

    def test_invalid_target_rejected(self) -> None:
        batch = validate_and_mask(self._local(), [self._attack(target_id=1)])

        self.assertEqual(batch.mask, (False,))
        self.assertEqual(batch.candidates[0].reject_reason, "invalid_target")

    def test_invalid_action_rejected(self) -> None:
        batch = validate_and_mask(self._local(), [self._attack(source_id=None)])

        self.assertEqual(batch.mask, (False,))
        self.assertEqual(batch.candidates[0].reject_reason, "invalid_action")

    def test_sun_collision_rejected(self) -> None:
        local = build_local_turn_state(
            GameState(
                step=0,
                player_id=1,
                planets=[
                    Planet(id=1, owner=1, ships=20, x=0.0, y=50.0),
                    Planet(id=2, owner=2, ships=8, x=100.0, y=50.0),
                ],
                fleets=[],
            )
        )
        candidate = Candidate(CandidateType.ATTACK, source_id=1, target_id=2, ships=5, angle=0.0, eta=100)

        batch = validate_and_mask(local, [candidate])

        self.assertEqual(batch.mask, (False,))
        self.assertEqual(batch.candidates[0].reject_reason, "sun_collision")

    def test_hold_source_is_legal_without_positive_ships_or_target(self) -> None:
        candidate = Candidate(CandidateType.HOLD_SOURCE, source_id=1, ships=0, target_id=None, eta=0)

        batch = validate_and_mask(self._local(), [candidate])

        self.assertEqual(batch.mask, (True,))
        self.assertIs(batch.candidates[0].legal, True)
        self.assertIsNone(batch.candidates[0].reject_reason)

    def test_input_candidates_are_not_mutated(self) -> None:
        candidate = self._attack(ships=0)

        batch = validate_and_mask(self._local(), [candidate])

        self.assertIs(candidate.legal, True)
        self.assertIsNone(candidate.reject_reason)
        self.assertIs(batch.candidates[0].legal, False)
        self.assertEqual(batch.candidates[0].reject_reason, "invalid_ships")

    def test_malformed_hold_source_is_masked_without_crashing(self) -> None:
        candidate = Candidate(CandidateType.HOLD_SOURCE, source_id="bad")

        batch = validate_and_mask(self._local(), [candidate])

        self.assertEqual(batch.mask, (False,))
        self.assertEqual(batch.candidates[0].reject_reason, "invalid_source")

    def test_malformed_launch_numbers_are_masked_without_crashing(self) -> None:
        candidate = self._attack(ships=None)

        batch = validate_and_mask(self._local(), [candidate])

        self.assertEqual(batch.mask, (False,))
        self.assertEqual(batch.candidates[0].reject_reason, "invalid_ships")


if __name__ == "__main__":
    unittest.main()
