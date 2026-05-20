from __future__ import annotations

import re
import unittest

from rl.features import (
    CANDIDATE_FEATURE_SCHEMA,
    FEATURE_SCHEMA_VERSION,
    GLOBAL_FEATURE_SCHEMA,
    FeatureField,
    FeatureSchema,
    FeatureSchemaError,
    encode_candidates,
    encode_global,
    schema_fingerprint,
)
from rl.local_state import build_local_turn_state
from rl.types import Candidate, CandidateBatch, CandidateType
from core.types import Fleet, GameState, Planet


EXPECTED_GLOBAL_KEYS = (
    "step_norm",
    "player_id",
    "owned_planets",
    "enemy_planets",
    "neutral_planets",
    "owned_production",
    "enemy_production",
    "neutral_production",
    "owned_ships_planets",
    "enemy_ships_planets",
    "neutral_ships_planets",
    "owned_ships_fleets",
    "enemy_ships_fleets",
    "score_advantage",
    "phase_early",
    "phase_mid",
    "phase_late",
)

EXPECTED_CANDIDATE_KEYS = (
    "type_stop",
    "type_attack",
    "type_expand_neutral",
    "type_reinforce",
    "type_defend",
    "type_harass",
    "type_hold_source",
    "source_ships_available",
    "ships_sent",
    "target_owner",
    "target_ships",
    "target_production",
    "eta",
    "distance",
    "projected_owner",
    "projected_ships",
    "overkill",
    "underkill",
    "reserve",
    "sun_safe",
    "friendly_arrivals_committed",
)


class FeatureSchemaTests(unittest.TestCase):
    def test_global_schema_has_stable_version_length_and_order(self) -> None:
        self.assertEqual(FEATURE_SCHEMA_VERSION, "3.1.0")
        self.assertEqual(GLOBAL_FEATURE_SCHEMA.version, FEATURE_SCHEMA_VERSION)
        self.assertEqual(GLOBAL_FEATURE_SCHEMA.keys, EXPECTED_GLOBAL_KEYS)
        self.assertEqual(GLOBAL_FEATURE_SCHEMA.length, len(EXPECTED_GLOBAL_KEYS))

    def test_candidate_schema_has_stable_version_length_and_order(self) -> None:
        self.assertEqual(CANDIDATE_FEATURE_SCHEMA.version, FEATURE_SCHEMA_VERSION)
        self.assertEqual(CANDIDATE_FEATURE_SCHEMA.keys, EXPECTED_CANDIDATE_KEYS)
        self.assertEqual(CANDIDATE_FEATURE_SCHEMA.length, len(EXPECTED_CANDIDATE_KEYS))

    def test_all_schema_keys_are_snake_case(self) -> None:
        snake_case = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")

        for schema in (GLOBAL_FEATURE_SCHEMA, CANDIDATE_FEATURE_SCHEMA):
            for key in schema.keys:
                self.assertRegex(key, snake_case)

    def test_schema_rejects_non_snake_case_keys(self) -> None:
        with self.assertRaisesRegex(FeatureSchemaError, "snake_case"):
            FeatureSchema(
                name="bad",
                version="1.0.0",
                fields=(FeatureField("notSnake"),),
            )

    def test_schema_rejects_duplicate_keys(self) -> None:
        with self.assertRaisesRegex(FeatureSchemaError, "duplicate"):
            FeatureSchema(
                name="bad",
                version="1.0.0",
                fields=(FeatureField("same_key"), FeatureField("same_key")),
            )

    def test_validate_vector_requires_matching_length(self) -> None:
        valid = GLOBAL_FEATURE_SCHEMA.validate_vector([0.0] * GLOBAL_FEATURE_SCHEMA.length)

        self.assertEqual(valid, (0.0,) * GLOBAL_FEATURE_SCHEMA.length)
        with self.assertRaisesRegex(FeatureSchemaError, "length"):
            GLOBAL_FEATURE_SCHEMA.validate_vector([0.0])

    def test_metadata_contains_checkpoint_compatibility_values(self) -> None:
        metadata = CANDIDATE_FEATURE_SCHEMA.metadata()

        self.assertEqual(metadata["name"], "candidate")
        self.assertEqual(metadata["version"], FEATURE_SCHEMA_VERSION)
        self.assertEqual(metadata["length"], len(EXPECTED_CANDIDATE_KEYS))
        self.assertEqual(metadata["keys"], EXPECTED_CANDIDATE_KEYS)
        self.assertEqual(metadata["fingerprint"], schema_fingerprint(CANDIDATE_FEATURE_SCHEMA))

    def test_fingerprint_changes_when_version_or_order_changes(self) -> None:
        base = FeatureSchema(
            name="candidate",
            version="1.0.0",
            fields=(FeatureField("first_key"), FeatureField("second_key")),
        )
        reordered = FeatureSchema(
            name="candidate",
            version="1.0.0",
            fields=(FeatureField("second_key"), FeatureField("first_key")),
        )
        bumped = FeatureSchema(
            name="candidate",
            version="1.0.1",
            fields=(FeatureField("first_key"), FeatureField("second_key")),
        )

        self.assertNotEqual(schema_fingerprint(base), schema_fingerprint(reordered))
        self.assertNotEqual(schema_fingerprint(base), schema_fingerprint(bumped))


class GlobalFeatureEncodingTests(unittest.TestCase):
    def _feature_map(self, values):
        return dict(zip(GLOBAL_FEATURE_SCHEMA.keys, values))

    def test_encode_global_matches_schema_length_and_order(self) -> None:
        local = build_local_turn_state(
            GameState(
                step=250,
                player_id=1,
                config={"episodeSteps": 500},
                planets=[
                    Planet(id=1, owner=1, ships=20, production=2.0),
                    Planet(id=2, owner=2, ships=10, production=1.5),
                    Planet(id=3, owner=-1, ships=5, production=0.5),
                ],
                fleets=[],
            )
        )

        values = encode_global(local)

        self.assertIsInstance(values, tuple)
        self.assertEqual(len(values), GLOBAL_FEATURE_SCHEMA.length)
        self.assertEqual(tuple(self._feature_map(values).keys()), GLOBAL_FEATURE_SCHEMA.keys)

    def test_encode_global_counts_planets_production_and_planet_ships_by_owner(self) -> None:
        local = build_local_turn_state(
            GameState(
                step=0,
                player_id=1,
                planets=[
                    Planet(id=1, owner=1, ships=20, production=2.0),
                    Planet(id=2, owner=1, ships=7, production=1.0),
                    Planet(id=3, owner=2, ships=10, production=1.5),
                    Planet(id=4, owner=3, ships=4, production=0.25),
                    Planet(id=5, owner=-1, ships=5, production=0.5),
                ],
                fleets=[],
            )
        )

        features = self._feature_map(encode_global(local))

        self.assertEqual(features["owned_planets"], 2.0)
        self.assertEqual(features["enemy_planets"], 2.0)
        self.assertEqual(features["neutral_planets"], 1.0)
        self.assertEqual(features["owned_production"], 3.0)
        self.assertEqual(features["enemy_production"], 1.75)
        self.assertEqual(features["neutral_production"], 0.5)
        self.assertEqual(features["owned_ships_planets"], 27.0)
        self.assertEqual(features["enemy_ships_planets"], 14.0)
        self.assertEqual(features["neutral_ships_planets"], 5.0)

    def test_encode_global_counts_fleet_ships_and_score_advantage(self) -> None:
        local = build_local_turn_state(
            GameState(
                step=0,
                player_id=1,
                planets=[
                    Planet(id=1, owner=1, ships=20, production=2.0),
                    Planet(id=2, owner=2, ships=10, production=1.0),
                ],
                fleets=[
                    Fleet(id=1, owner=1, ships=5),
                    Fleet(id=2, owner=2, ships=3),
                    Fleet(id=3, owner=3, ships=2),
                    Fleet(id=4, owner=-1, ships=99),
                ],
            )
        )

        features = self._feature_map(encode_global(local))

        self.assertEqual(features["owned_ships_fleets"], 5.0)
        self.assertEqual(features["enemy_ships_fleets"], 5.0)
        self.assertAlmostEqual(features["score_advantage"], 11.0 / 43.0)

    def test_encode_global_handles_empty_state(self) -> None:
        local = build_local_turn_state(
            GameState(step=0, player_id=0, planets=[], fleets=[])
        )

        features = self._feature_map(encode_global(local))

        self.assertEqual(features["step_norm"], 0.0)
        self.assertEqual(features["player_id"], 0.0)
        self.assertEqual(features["owned_planets"], 0.0)
        self.assertEqual(features["enemy_planets"], 0.0)
        self.assertEqual(features["neutral_planets"], 0.0)
        self.assertEqual(features["score_advantage"], 0.0)
        self.assertEqual(features["phase_early"], 1.0)
        self.assertEqual(features["phase_mid"], 0.0)
        self.assertEqual(features["phase_late"], 0.0)

    def test_encode_global_normalizes_step_and_phase_indicators(self) -> None:
        early = self._feature_map(
            encode_global(
                build_local_turn_state(
                    GameState(step=50, player_id=1, config={"episodeSteps": 300}, planets=[], fleets=[])
                )
            )
        )
        mid = self._feature_map(
            encode_global(
                build_local_turn_state(
                    GameState(step=150, player_id=1, config={"episodeSteps": 300}, planets=[], fleets=[])
                )
            )
        )
        late = self._feature_map(
            encode_global(
                build_local_turn_state(
                    GameState(step=999, player_id=1, config={"episodeSteps": 300}, planets=[], fleets=[])
                )
            )
        )

        self.assertAlmostEqual(early["step_norm"], 50.0 / 300.0)
        self.assertEqual((early["phase_early"], early["phase_mid"], early["phase_late"]), (1.0, 0.0, 0.0))
        self.assertEqual((mid["phase_early"], mid["phase_mid"], mid["phase_late"]), (0.0, 1.0, 0.0))
        self.assertEqual(late["step_norm"], 1.0)
        self.assertEqual((late["phase_early"], late["phase_mid"], late["phase_late"]), (0.0, 0.0, 1.0))


class CandidateFeatureEncodingTests(unittest.TestCase):
    def _local(self):
        return build_local_turn_state(
            GameState(
                step=0,
                player_id=1,
                planets=[
                    Planet(id=1, owner=1, ships=20, production=2.0, x=0.0, y=0.0),
                    Planet(id=2, owner=2, ships=8, production=1.0, x=3.0, y=4.0),
                    Planet(id=3, owner=-1, ships=4, production=0.5, x=0.0, y=20.0),
                ],
                fleets=[],
            )
        )

    def _row_map(self, row):
        return dict(zip(CANDIDATE_FEATURE_SCHEMA.keys, row))

    def test_encode_candidates_matches_batch_order_and_schema_shape(self) -> None:
        batch = CandidateBatch(
            [
                Candidate.stop(),
                Candidate(CandidateType.HOLD_SOURCE, source_id=1),
                Candidate(CandidateType.ATTACK, source_id=1, target_id=2, ships=9, eta=5),
            ],
            [True, True, False],
        )

        rows = encode_candidates(self._local(), batch)

        self.assertIsInstance(rows, tuple)
        self.assertEqual(len(rows), 3)
        self.assertEqual(batch.mask, (True, True, False))
        for row in rows:
            self.assertIsInstance(row, tuple)
            self.assertEqual(len(row), CANDIDATE_FEATURE_SCHEMA.length)
        self.assertEqual(self._row_map(rows[0])["type_stop"], 1.0)
        self.assertEqual(self._row_map(rows[1])["type_hold_source"], 1.0)
        self.assertEqual(self._row_map(rows[2])["type_attack"], 1.0)

    def test_encode_candidates_sets_basic_source_target_values(self) -> None:
        batch = CandidateBatch(
            [
                Candidate(
                    CandidateType.ATTACK,
                    source_id=1,
                    target_id=2,
                    ships=9,
                    eta=5,
                    estimated_owner=1,
                    estimated_ships=1,
                )
            ],
            [True],
        )

        features = self._row_map(encode_candidates(self._local(), batch)[0])

        self.assertEqual(features["source_ships_available"], 20.0)
        self.assertEqual(features["ships_sent"], 9.0)
        self.assertEqual(features["target_owner"], 2.0)
        self.assertEqual(features["target_ships"], 8.0)
        self.assertEqual(features["target_production"], 1.0)
        self.assertEqual(features["eta"], 5.0)
        self.assertEqual(features["distance"], 5.0)
        self.assertEqual(features["projected_owner"], 1.0)
        self.assertEqual(features["projected_ships"], 1.0)

    def test_encode_candidates_sets_overkill_underkill_reserve_and_sun_safe(self) -> None:
        local = self._local()
        local.source_commitments[1] = 5
        batch = CandidateBatch(
            [
                Candidate(CandidateType.ATTACK, source_id=1, target_id=2, ships=9, eta=5),
                Candidate(CandidateType.HARASS, source_id=1, target_id=2, ships=3, eta=5),
            ],
            [True, True],
        )

        attack = self._row_map(encode_candidates(local, batch)[0])
        harass = self._row_map(encode_candidates(local, batch)[1])

        self.assertEqual(attack["source_ships_available"], 15.0)
        self.assertEqual(attack["overkill"], 1.0)
        self.assertEqual(attack["underkill"], 0.0)
        self.assertEqual(attack["reserve"], 6.0)
        self.assertEqual(attack["sun_safe"], 1.0)
        self.assertEqual(harass["overkill"], 0.0)
        self.assertEqual(harass["underkill"], 5.0)
        self.assertEqual(harass["reserve"], 12.0)

    def test_encode_candidates_uses_timeline_projection_and_friendly_arrivals(self) -> None:
        local = self._local()
        timeline = local.target_timelines[2]
        timeline.projected_owner = 1
        timeline.projected_ships = 4
        timeline.friendly_by_eta[5] = 6
        batch = CandidateBatch(
            [Candidate(CandidateType.ATTACK, source_id=1, target_id=2, ships=2, eta=5)],
            [True],
        )

        features = self._row_map(encode_candidates(local, batch)[0])

        self.assertEqual(features["projected_owner"], 1.0)
        self.assertEqual(features["projected_ships"], 4.0)
        self.assertEqual(features["friendly_arrivals_committed"], 6.0)

    def test_encode_candidates_handles_stop_and_missing_target_as_zero_values(self) -> None:
        batch = CandidateBatch([Candidate.stop()], [True])

        features = self._row_map(encode_candidates(self._local(), batch)[0])

        self.assertEqual(features["type_stop"], 1.0)
        self.assertEqual(features["ships_sent"], 0.0)
        self.assertEqual(features["target_owner"], 0.0)
        self.assertEqual(features["target_ships"], 0.0)
        self.assertEqual(features["sun_safe"], 1.0)

    def test_encode_candidates_marks_unsafe_sun_route(self) -> None:
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
        batch = CandidateBatch(
            [Candidate(CandidateType.ATTACK, source_id=1, target_id=2, ships=9, eta=100)],
            [False],
        )

        features = self._row_map(encode_candidates(local, batch)[0])

        self.assertEqual(features["sun_safe"], 0.0)


if __name__ == "__main__":
    unittest.main()
