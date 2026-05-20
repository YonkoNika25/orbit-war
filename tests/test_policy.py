from __future__ import annotations

import random
import unittest

from core.actions import safe_candidate_to_action
from core.types import GameState, Planet
from rl.candidates import generate_candidates
from rl.features import encode_candidates, encode_global
from rl.local_state import build_local_turn_state
from rl.masks import validate_and_mask
from rl.policy import PolicySelectionError, RandomMaskedPolicy
from rl.types import Candidate, CandidateBatch, CandidateType


class RandomMaskedPolicyTests(unittest.TestCase):
    def test_select_index_only_returns_legal_mask_indexes(self) -> None:
        policy = RandomMaskedPolicy(rng=random.Random(7))
        mask = (False, True, False, True, False)
        candidate_features = tuple((0.0,) for _ in mask)

        selections = {
            policy.select_index((0.0,), candidate_features, mask)
            for _ in range(50)
        }

        self.assertLessEqual(selections, {1, 3})

    def test_all_illegal_mask_falls_back_to_stop_index(self) -> None:
        policy = RandomMaskedPolicy(rng=random.Random(1), stop_index=0)

        selected = policy.select_index((0.0,), ((0.0,), (0.0,)), (False, False))

        self.assertEqual(selected, 0)

    def test_rejects_feature_mask_alignment_mismatch(self) -> None:
        policy = RandomMaskedPolicy(rng=random.Random(1))

        with self.assertRaisesRegex(PolicySelectionError, "candidate feature rows"):
            policy.select_index((0.0,), ((0.0,),), (True, False))

    def test_rejects_out_of_range_stop_index_when_fallback_needed(self) -> None:
        policy = RandomMaskedPolicy(rng=random.Random(1), stop_index=2)

        with self.assertRaisesRegex(PolicySelectionError, "stop_index"):
            policy.select_index((0.0,), ((0.0,),), (False,))

    def test_selected_index_is_consumed_through_candidate_batch_selected(self) -> None:
        batch = CandidateBatch(
            [
                Candidate.stop(),
                Candidate(CandidateType.ATTACK, source_id=1, target_id=2, ships=5),
                Candidate(CandidateType.HARASS, source_id=1, target_id=2, ships=1),
            ],
            [True, False, True],
        )
        policy = RandomMaskedPolicy(rng=random.Random(0))

        index = policy.select_index((0.0,), ((0.0,), (0.0,), (0.0,)), batch.mask)
        selected = batch.selected(index)

        self.assertIn(index, (0, 2))
        self.assertIn(selected.type, (CandidateType.STOP, CandidateType.HARASS))

    def test_shared_runtime_smoke_path_selects_action_without_training_dependencies(self) -> None:
        local = build_local_turn_state(
            GameState(
                step=0,
                player_id=1,
                planets=[
                    Planet(id=1, owner=1, ships=20, x=0.0, y=0.0),
                    Planet(id=2, owner=2, ships=6, x=3.0, y=4.0),
                    Planet(id=3, owner=-1, ships=3, x=0.0, y=20.0),
                ],
                fleets=[],
            )
        )
        candidates = generate_candidates(local)
        batch = validate_and_mask(local, candidates)
        global_features = encode_global(local)
        candidate_features = encode_candidates(local, batch)
        policy = RandomMaskedPolicy(rng=random.Random(2))

        selected_index = policy.select_index(global_features, candidate_features, batch.mask)
        selected_candidate = batch.selected(selected_index)
        action = safe_candidate_to_action(selected_candidate)

        self.assertTrue(batch.mask[selected_index])
        self.assertIsInstance(action, list)


if __name__ == "__main__":
    unittest.main()
