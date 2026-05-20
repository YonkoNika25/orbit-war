from __future__ import annotations

import unittest

from core.actions import candidate_to_action
from core.types import GameState, Planet
from rl.candidates import generate_candidates
from rl.local_state import build_local_turn_state
from rl.types import Candidate, CandidateBatch, CandidateType


class CandidateTypeTests(unittest.TestCase):
    def test_candidate_contains_required_runtime_fields(self) -> None:
        candidate = Candidate(
            type=CandidateType.ATTACK,
            source_id=1,
            target_id=2,
            ships=12,
            angle=1.25,
            eta=4,
            estimated_owner=1,
            estimated_ships=3,
            score=0.75,
        )

        self.assertEqual(candidate.type, CandidateType.ATTACK)
        self.assertEqual(candidate.source_id, 1)
        self.assertEqual(candidate.target_id, 2)
        self.assertEqual(candidate.ships, 12)
        self.assertEqual(candidate.angle, 1.25)
        self.assertEqual(candidate.eta, 4)
        self.assertIs(candidate.legal, True)
        self.assertIsNone(candidate.reject_reason)
        self.assertEqual(candidate.estimated_owner, 1)
        self.assertEqual(candidate.estimated_ships, 3)
        self.assertEqual(candidate.score, 0.75)

    def test_stop_candidate_is_explicit_and_legal(self) -> None:
        candidate = Candidate.stop()

        self.assertEqual(candidate.type, CandidateType.STOP)
        self.assertIsNone(candidate.source_id)
        self.assertIsNone(candidate.target_id)
        self.assertEqual(candidate.ships, 0)
        self.assertEqual(candidate.angle, 0.0)
        self.assertEqual(candidate.eta, 0)
        self.assertIs(candidate.legal, True)
        self.assertEqual(candidate_to_action(candidate), [])

    def test_candidate_batch_preserves_order_and_mask_alignment(self) -> None:
        stop = Candidate.stop()
        attack = Candidate(CandidateType.ATTACK, source_id=1, target_id=2, ships=10, angle=0.5, eta=3)
        reinforce = Candidate(
            CandidateType.REINFORCE,
            source_id=1,
            target_id=3,
            ships=5,
            angle=1.5,
            eta=2,
            legal=False,
            reject_reason="insufficient_ships",
        )

        batch = CandidateBatch([stop, attack, reinforce], mask=[True, True, False])

        self.assertEqual(batch.candidates, (stop, attack, reinforce))
        self.assertEqual(batch.mask, (True, True, False))
        self.assertEqual(batch.selected(0), stop)
        self.assertEqual(batch.selected(1), attack)
        self.assertEqual(batch.selected(2), reinforce)
        self.assertEqual(batch.legal_candidates(), (stop, attack))

    def test_candidate_batch_rejects_mask_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            CandidateBatch([Candidate.stop()], mask=[])

    def test_selected_index_lookup_rejects_out_of_range_index(self) -> None:
        batch = CandidateBatch([Candidate.stop()], mask=[True])

        with self.assertRaises(IndexError):
            batch.selected(1)
        with self.assertRaises(IndexError):
            batch.selected(-1)

    def test_local_turn_state_apply_accepts_candidate_dataclass(self) -> None:
        state = GameState(
            step=0,
            player_id=1,
            planets=[Planet(id=1, owner=1, ships=20), Planet(id=2, owner=2, ships=10)],
            fleets=[],
        )
        local = build_local_turn_state(state)
        candidate = Candidate(CandidateType.ATTACK, source_id=1, target_id=2, ships=12, angle=0.0, eta=3)

        self.assertTrue(local.apply(candidate))
        self.assertEqual(local.committed_ships(1), 12)
        self.assertEqual(local.target_timelines[2].projected_owner, 1)


class CandidateGenerationTests(unittest.TestCase):
    def test_generate_candidates_includes_stop_first(self) -> None:
        local = build_local_turn_state(
            GameState(step=0, player_id=1, planets=[Planet(id=1, owner=1, ships=10)], fleets=[])
        )

        candidates = generate_candidates(local)

        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0].type, CandidateType.STOP)

    def test_generate_candidates_uses_only_owned_available_sources(self) -> None:
        local = build_local_turn_state(
            GameState(
                step=0,
                player_id=1,
                planets=[
                    Planet(id=1, owner=1, ships=10, x=0.0, y=0.0),
                    Planet(id=2, owner=2, ships=10, x=10.0, y=0.0),
                    Planet(id=3, owner=-1, ships=5, x=0.0, y=10.0),
                ],
                fleets=[],
            )
        )
        local.source_commitments[1] = 10

        candidates = generate_candidates(local)

        launch_sources = {
            candidate.source_id
            for candidate in candidates
            if candidate.source_id is not None and candidate.type != CandidateType.HOLD_SOURCE
        }
        self.assertEqual(launch_sources, set())

    def test_generate_candidates_covers_neutral_enemy_friendly_and_stop(self) -> None:
        local = build_local_turn_state(
            GameState(
                step=0,
                player_id=1,
                planets=[
                    Planet(id=1, owner=1, ships=40, x=0.0, y=0.0),
                    Planet(id=2, owner=2, ships=12, x=12.0, y=0.0),
                    Planet(id=3, owner=-1, ships=6, x=0.0, y=8.0),
                    Planet(id=4, owner=1, ships=8, x=6.0, y=6.0),
                ],
                fleets=[],
            )
        )

        candidates = generate_candidates(local)
        by_type = {candidate.type for candidate in candidates}

        self.assertIn(CandidateType.STOP, by_type)
        self.assertIn(CandidateType.ATTACK, by_type)
        self.assertIn(CandidateType.HARASS, by_type)
        self.assertIn(CandidateType.EXPAND_NEUTRAL, by_type)
        self.assertIn(CandidateType.REINFORCE, by_type)
        self.assertIn(CandidateType.HOLD_SOURCE, by_type)

        attack = next(candidate for candidate in candidates if candidate.type == CandidateType.ATTACK)
        self.assertEqual(attack.source_id, 1)
        self.assertEqual(attack.target_id, 2)
        self.assertGreater(attack.ships, 0)
        self.assertIsInstance(attack.angle, float)
        self.assertGreaterEqual(attack.eta, 1)
        self.assertIsNotNone(attack.estimated_owner)
        self.assertIsNotNone(attack.estimated_ships)

    def test_generate_candidates_defends_friendly_target_with_enemy_arrivals(self) -> None:
        local = build_local_turn_state(
            GameState(
                step=0,
                player_id=1,
                planets=[
                    Planet(id=1, owner=1, ships=40, x=0.0, y=0.0),
                    Planet(id=4, owner=1, ships=8, x=6.0, y=6.0),
                ],
                fleets=[],
            )
        )
        local.target_timelines[4].enemy_by_eta[3] = 20

        candidates = generate_candidates(local)

        defend = [candidate for candidate in candidates if candidate.type == CandidateType.DEFEND]
        self.assertEqual(len(defend), 1)
        self.assertEqual(defend[0].source_id, 1)
        self.assertEqual(defend[0].target_id, 4)

    def test_generate_candidates_does_not_mutate_local_state(self) -> None:
        local = build_local_turn_state(
            GameState(
                step=0,
                player_id=1,
                planets=[
                    Planet(id=1, owner=1, ships=30, x=0.0, y=0.0),
                    Planet(id=2, owner=2, ships=10, x=10.0, y=0.0),
                ],
                fleets=[],
            )
        )
        before_commitments = dict(local.source_commitments)
        before_friendly = {
            target_id: dict(timeline.friendly_by_eta)
            for target_id, timeline in local.target_timelines.items()
        }
        before_projected = {
            target_id: (timeline.projected_owner, timeline.projected_ships)
            for target_id, timeline in local.target_timelines.items()
        }

        generate_candidates(local)

        self.assertEqual(local.source_commitments, before_commitments)
        self.assertEqual(
            {target_id: dict(timeline.friendly_by_eta) for target_id, timeline in local.target_timelines.items()},
            before_friendly,
        )
        self.assertEqual(
            {
                target_id: (timeline.projected_owner, timeline.projected_ships)
                for target_id, timeline in local.target_timelines.items()
            },
            before_projected,
        )

    def test_generate_candidates_skips_enemy_target_already_projected_owned(self) -> None:
        local = build_local_turn_state(
            GameState(
                step=0,
                player_id=1,
                planets=[
                    Planet(id=1, owner=1, ships=40, x=0.0, y=0.0),
                    Planet(id=2, owner=2, ships=10, x=10.0, y=0.0),
                ],
                fleets=[],
            )
        )
        self.assertTrue(
            local.apply(Candidate(CandidateType.ATTACK, source_id=1, target_id=2, ships=12, angle=0.0, eta=3))
        )

        candidates = generate_candidates(local)

        repeated_attacks = [
            candidate
            for candidate in candidates
            if candidate.target_id == 2 and candidate.type in {CandidateType.ATTACK, CandidateType.HARASS}
        ]
        self.assertEqual(repeated_attacks, [])


if __name__ == "__main__":
    unittest.main()
