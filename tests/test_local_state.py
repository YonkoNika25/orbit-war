from __future__ import annotations

import unittest
from dataclasses import dataclass

from core.types import Fleet, GameState, Planet
from rl.local_state import build_local_turn_state


@dataclass(slots=True)
class ObjectCandidate:
    source_id: int
    target_id: int
    ships: int
    eta: int


class LocalTurnStateTests(unittest.TestCase):
    def test_indexes_planets_and_tracks_initial_source_availability(self) -> None:
        state = GameState(
            step=3,
            player_id=1,
            planets=[
                Planet(id=10, owner=1, ships=50),
                Planet(id=20, owner=2, ships=40),
                Planet(id=30, owner=-1, ships=15),
            ],
            fleets=[],
        )

        local = build_local_turn_state(state)

        self.assertIs(local.state, state)
        self.assertEqual(set(local.planets_by_id), {10, 20, 30})
        self.assertEqual(local.owned_planet_ids, (10,))
        self.assertEqual(local.enemy_planet_ids, (20,))
        self.assertEqual(local.neutral_planet_ids, (30,))
        self.assertEqual(local.available_ships(10), 50)
        self.assertEqual(local.available_ships(20), 0)
        self.assertEqual(local.available_ships(30), 0)
        self.assertEqual(local.available_ships(999), 0)
        self.assertEqual(local.committed_ships(10), 0)

    def test_empty_fleets_have_empty_arrivals_and_target_timelines(self) -> None:
        state = GameState(
            step=0,
            player_id=0,
            planets=[Planet(id=1, owner=0, ships=12), Planet(id=2, owner=1, ships=9)],
            fleets=[],
        )

        local = build_local_turn_state(state)

        self.assertEqual(local.fleets, ())
        self.assertEqual(local.friendly_arrivals, {})
        self.assertEqual(local.enemy_arrivals, {})
        self.assertEqual(local.target_timelines[1].current_owner, 0)
        self.assertEqual(local.target_timelines[1].current_ships, 12)
        self.assertEqual(local.target_timelines[1].friendly_by_eta, {})
        self.assertEqual(local.target_timelines[1].enemy_by_eta, {})
        self.assertEqual(local.target_timelines[2].current_owner, 1)
        self.assertEqual(local.target_timelines[2].current_ships, 9)

    def test_friendly_arrivals_are_grouped_by_explicit_target_and_eta(self) -> None:
        state = GameState(
            step=5,
            player_id=1,
            planets=[Planet(id=1, owner=1, ships=30), Planet(id=2, owner=2, ships=20)],
            fleets=[
                Fleet(id=100, owner=1, ships=7, eta=3, extras={"target_id": 2}),
                Fleet(id=101, owner=1, ships=5, eta=3, extras={"to_planet_id": 2}),
                Fleet(id=102, owner=1, ships=4, eta=4, extras={"destination": 2}),
            ],
        )

        local = build_local_turn_state(state)

        self.assertEqual(local.friendly_arrivals[2][3].ships, 12)
        self.assertEqual(local.friendly_arrivals[2][3].fleet_ids, (100, 101))
        self.assertEqual(local.friendly_arrivals[2][4].ships, 4)
        self.assertEqual(local.target_timelines[2].friendly_by_eta, {3: 12, 4: 4})
        self.assertEqual(local.target_timelines[2].enemy_by_eta, {})

    def test_enemy_arrivals_are_grouped_by_explicit_target_and_eta(self) -> None:
        state = GameState(
            step=5,
            player_id=0,
            planets=[Planet(id=1, owner=0, ships=30), Planet(id=2, owner=2, ships=20)],
            fleets=[
                Fleet(id=200, owner=2, ships=6, eta=2, extras={"target": 1}),
                Fleet(id=201, owner=3, ships=8, eta=2, extras={"target_id": 1}),
            ],
        )

        local = build_local_turn_state(state)

        self.assertEqual(local.enemy_arrivals[1][2].ships, 14)
        self.assertEqual(local.enemy_arrivals[1][2].fleet_ids, (200, 201))
        self.assertEqual(local.target_timelines[1].enemy_by_eta, {2: 14})
        self.assertEqual(local.target_timelines[1].friendly_by_eta, {})

    def test_ignores_fleets_without_explicit_target_or_valid_eta(self) -> None:
        state = GameState(
            step=5,
            player_id=1,
            planets=[Planet(id=1, owner=1, ships=30), Planet(id=2, owner=2, ships=20)],
            fleets=[
                Fleet(id=300, owner=1, ships=6, eta=None, extras={"target_id": 2}),
                Fleet(id=301, owner=1, ships=8, eta=2, extras={}),
                Fleet(id=302, owner=2, ships=5, eta=float("inf"), extras={"target_id": 1}),
            ],
        )

        local = build_local_turn_state(state)

        self.assertEqual(local.friendly_arrivals, {})
        self.assertEqual(local.enemy_arrivals, {})
        self.assertEqual(local.target_timelines[1].enemy_by_eta, {})
        self.assertEqual(local.target_timelines[2].friendly_by_eta, {})

    def test_uses_later_target_alias_when_earlier_alias_is_invalid(self) -> None:
        state = GameState(
            step=5,
            player_id=1,
            planets=[Planet(id=1, owner=1, ships=30), Planet(id=2, owner=2, ships=20)],
            fleets=[
                Fleet(
                    id=400,
                    owner=1,
                    ships=6,
                    eta=2,
                    extras={"target_id": "unknown", "to_planet_id": 2},
                )
            ],
        )

        local = build_local_turn_state(state)

        self.assertEqual(local.friendly_arrivals[2][2].ships, 6)
        self.assertEqual(local.target_timelines[2].friendly_by_eta, {2: 6})

    def test_commitment_and_rejection_debug_helpers(self) -> None:
        state = GameState(
            step=0,
            player_id=1,
            planets=[Planet(id=1, owner=1, ships=20)],
            fleets=[],
        )

        local = build_local_turn_state(state)
        local.source_commitments[1] = 6
        local.record_rejection(candidate_index=4, reason="insufficient_ships")

        self.assertEqual(local.committed_ships(1), 6)
        self.assertEqual(local.available_ships(1), 14)
        self.assertEqual(local.rejections[4], "insufficient_ships")

    def test_apply_commits_source_ships_and_reduces_availability(self) -> None:
        state = GameState(
            step=0,
            player_id=1,
            planets=[Planet(id=1, owner=1, ships=20), Planet(id=2, owner=2, ships=30)],
            fleets=[],
        )

        local = build_local_turn_state(state)

        self.assertTrue(local.apply({"source_id": 1, "target_id": 2, "ships": 8, "eta": 3}))
        self.assertEqual(local.committed_ships(1), 8)
        self.assertEqual(local.available_ships(1), 12)

    def test_apply_updates_target_arrival_timeline_and_projection(self) -> None:
        state = GameState(
            step=0,
            player_id=1,
            planets=[Planet(id=1, owner=1, ships=20), Planet(id=2, owner=2, ships=10)],
            fleets=[],
        )

        local = build_local_turn_state(state)

        self.assertTrue(local.apply(ObjectCandidate(source_id=1, target_id=2, ships=14, eta=4)))

        target = local.target_timelines[2]
        self.assertEqual(target.friendly_by_eta, {4: 14})
        self.assertEqual(target.projected_owner, 1)
        self.assertEqual(target.projected_ships, 4)

    def test_apply_reinforces_owned_target_projection(self) -> None:
        state = GameState(
            step=0,
            player_id=1,
            planets=[Planet(id=1, owner=1, ships=20), Planet(id=2, owner=1, ships=10)],
            fleets=[],
        )

        local = build_local_turn_state(state)

        self.assertTrue(local.apply({"source": 1, "target": 2, "ship_count": 5, "eta": 2}))

        target = local.target_timelines[2]
        self.assertEqual(target.friendly_by_eta, {2: 5})
        self.assertEqual(target.projected_owner, 1)
        self.assertEqual(target.projected_ships, 15)

    def test_apply_rejects_duplicate_launch_that_exceeds_available_ships(self) -> None:
        state = GameState(
            step=0,
            player_id=1,
            planets=[Planet(id=1, owner=1, ships=10), Planet(id=2, owner=2, ships=30)],
            fleets=[],
        )

        local = build_local_turn_state(state)

        self.assertTrue(local.apply({"source_id": 1, "target_id": 2, "ships": 7, "eta": 3}))
        self.assertFalse(
            local.apply({"source_id": 1, "target_id": 2, "ships": 4, "eta": 4}, candidate_index=9)
        )

        self.assertEqual(local.committed_ships(1), 7)
        self.assertEqual(local.target_timelines[2].friendly_by_eta, {3: 7})
        self.assertEqual(local.rejections[9], "insufficient_ships")

    def test_apply_rejects_repeated_attack_into_already_winning_target(self) -> None:
        state = GameState(
            step=0,
            player_id=1,
            planets=[Planet(id=1, owner=1, ships=30), Planet(id=2, owner=2, ships=10)],
            fleets=[],
        )

        local = build_local_turn_state(state)

        self.assertTrue(local.apply({"source_id": 1, "target_id": 2, "ships": 12, "eta": 3}))
        self.assertFalse(
            local.apply({"source_id": 1, "target_id": 2, "ships": 5, "eta": 4}, candidate_index=10)
        )

        self.assertEqual(local.committed_ships(1), 12)
        self.assertEqual(local.target_timelines[2].friendly_by_eta, {3: 12})
        self.assertEqual(local.rejections[10], "already_winning_target")

    def test_apply_rejects_malformed_candidate_without_mutation(self) -> None:
        state = GameState(
            step=0,
            player_id=1,
            planets=[Planet(id=1, owner=1, ships=30), Planet(id=2, owner=2, ships=10)],
            fleets=[],
        )

        local = build_local_turn_state(state)

        self.assertFalse(local.apply({"source_id": 1, "target_id": 2, "ships": 5}, candidate_index=11))
        self.assertEqual(local.committed_ships(1), 0)
        self.assertEqual(local.target_timelines[2].friendly_by_eta, {})
        self.assertEqual(local.rejections[11], "invalid_candidate")

    def test_apply_rejects_non_finite_candidate_numbers_without_mutation(self) -> None:
        state = GameState(
            step=0,
            player_id=1,
            planets=[Planet(id=1, owner=1, ships=30), Planet(id=2, owner=2, ships=10)],
            fleets=[],
        )

        local = build_local_turn_state(state)

        self.assertFalse(
            local.apply(
                {"source_id": 1, "target_id": 2, "ships": 5, "eta": float("inf")},
                candidate_index=12,
            )
        )
        self.assertEqual(local.committed_ships(1), 0)
        self.assertEqual(local.target_timelines[2].friendly_by_eta, {})
        self.assertEqual(local.rejections[12], "invalid_candidate")


if __name__ == "__main__":
    unittest.main()
