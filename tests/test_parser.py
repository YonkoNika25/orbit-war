from __future__ import annotations

import unittest
from dataclasses import dataclass

from core.parser import parse_observation


class Obj:
    def __init__(self, **values):
        self.__dict__.update(values)


@dataclass(slots=True)
class SlottedPlanet:
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: float
    terrain: str = "rock"


@dataclass(slots=True)
class SlottedFleet:
    id: int
    owner: int
    x: float
    y: float
    angle: float
    from_planet_id: int
    ships: int
    tag: str = "scout"


class ParseObservationTests(unittest.TestCase):
    def test_parse_dict_observation_preserves_state_and_extras(self):
        obs = {
            "step": 7,
            "player_id": 1,
            "planets": [
                {
                    "id": 3,
                    "owner": 1,
                    "x": 10,
                    "y": 20,
                    "radius": 2,
                    "ships": 40,
                    "production": 5,
                    "biome": "ice",
                }
            ],
            "fleets": [
                {
                    "id": 9,
                    "owner": 2,
                    "x": 15,
                    "y": 25,
                    "angle": 0.5,
                    "from_planet_id": 3,
                    "ships": 12,
                    "mission": "raid",
                }
            ],
            "boardSize": 120,
            "match_id": "abc",
        }

        state = parse_observation(obs, {"actTimeout": 1, "custom_config": "kept"})

        self.assertEqual(state.step, 7)
        self.assertEqual(state.player_id, 1)
        self.assertEqual(state.config["actTimeout"], 1)
        self.assertEqual(state.config["custom_config"], "kept")
        self.assertEqual(state.config["boardSize"], 120)
        self.assertEqual(state.extras["match_id"], "abc")
        self.assertEqual(state.planets[0].extras["biome"], "ice")
        self.assertEqual(state.fleets[0].extras["mission"], "raid")

    def test_parse_tuple_root_and_tuple_planet_fleet_ordering(self):
        planet = [5, 2, 11.5, 22.5, 3.5, 44, 6.5]
        fleet = [8, 1, 30.0, 31.0, 1.25, 5, 17]

        state = parse_observation((12, 2, [planet], [fleet]))

        self.assertEqual(state.step, 12)
        self.assertEqual(state.player_id, 2)

        parsed_planet = state.planets[0]
        self.assertEqual(parsed_planet.id, 5)
        self.assertEqual(parsed_planet.owner, 2)
        self.assertEqual(parsed_planet.x, 11.5)
        self.assertEqual(parsed_planet.y, 22.5)
        self.assertEqual(parsed_planet.radius, 3.5)
        self.assertEqual(parsed_planet.ships, 44)
        self.assertEqual(parsed_planet.production, 6.5)

        parsed_fleet = state.fleets[0]
        self.assertEqual(parsed_fleet.id, 8)
        self.assertEqual(parsed_fleet.owner, 1)
        self.assertEqual(parsed_fleet.x, 30.0)
        self.assertEqual(parsed_fleet.y, 31.0)
        self.assertEqual(parsed_fleet.angle, 1.25)
        self.assertEqual(parsed_fleet.from_planet_id, 5)
        self.assertEqual(parsed_fleet.ships, 17)

    def test_parse_two_item_tuple_root_defaults_step_and_player(self):
        state = parse_observation(([[1, -1, 2, 3, 4, 5, 6]], []))

        self.assertEqual(state.step, 0)
        self.assertEqual(state.player_id, 0)
        self.assertEqual(len(state.planets), 1)
        self.assertEqual(state.fleets, [])

    def test_parse_object_like_observation_and_config(self):
        obs = Obj(
            turn=4,
            agent_id=3,
            planets=[Obj(id=1, owner=3, x=1.5, y=2.5, radius=1.0, ships=20, production=2)],
            fleets=[Obj(id=7, owner=2, x=4.0, y=5.0, angle=0.75, from_planet_id=1, ships=9)],
            scenario="object-input",
        )
        config = Obj(boardSize=80, sunRadius=8)

        state = parse_observation(obs, config)

        self.assertEqual(state.step, 4)
        self.assertEqual(state.player_id, 3)
        self.assertEqual(state.config["boardSize"], 80)
        self.assertEqual(state.config["sunRadius"], 8)
        self.assertEqual(state.extras["scenario"], "object-input")
        self.assertEqual(state.planets[0].id, 1)
        self.assertEqual(state.planets[0].owner, 3)
        self.assertEqual(state.fleets[0].id, 7)
        self.assertEqual(state.fleets[0].from_planet_id, 1)

    def test_parse_slotted_dataclass_planet_and_fleet_records(self):
        state = parse_observation(
            Obj(
                planets=[SlottedPlanet(2, 1, 10.0, 20.0, 2.5, 33, 4.0)],
                fleets=[SlottedFleet(4, 1, 12.0, 22.0, 1.5, 2, 8)],
            )
        )

        self.assertEqual(state.planets[0].id, 2)
        self.assertEqual(state.planets[0].extras["terrain"], "rock")
        self.assertEqual(state.fleets[0].id, 4)
        self.assertEqual(state.fleets[0].extras["tag"], "scout")

    def test_config_aliases_are_preserved_as_config_not_root_extras(self):
        state = parse_observation(
            {
                "planets": [],
                "fleets": [],
                "board_size": 88,
                "sun_radius": 9,
                "sunX": 44,
                "sunY": 45,
                "angularVelocity": 0.02,
            }
        )

        self.assertEqual(state.config["board_size"], 88)
        self.assertEqual(state.config["sun_radius"], 9)
        self.assertEqual(state.config["sunX"], 44)
        self.assertEqual(state.config["sunY"], 45)
        self.assertEqual(state.config["angularVelocity"], 0.02)
        self.assertNotIn("board_size", state.extras)
        self.assertNotIn("angularVelocity", state.extras)

    def test_object_like_initial_planets_feed_motion_metadata(self):
        state = parse_observation(
            Obj(
                planets=[Obj(id=1, owner=0, x=55.0, y=50.0, radius=2.0, ships=10, production=1)],
                fleets=[],
                initial_planets=[Obj(id=1, owner=0, x=50.0, y=50.0, radius=2.0, ships=10, production=1)],
                angular_velocity=0.1,
            )
        )

        planet = state.planets[0]
        self.assertIs(planet.extras["is_moving"], True)
        self.assertEqual(planet.orbit_center_x, 50.0)
        self.assertEqual(planet.orbit_center_y, 50.0)

    def test_identical_initial_planets_remain_stationary_with_angular_velocity(self):
        state = parse_observation(
            {
                "planets": [[1, 0, 55.0, 50.0, 2.0, 10, 1]],
                "fleets": [],
                "initial_planets": [[1, 0, 55.0, 50.0, 2.0, 10, 1]],
                "angular_velocity": 0.1,
            }
        )

        planet = state.planets[0]
        self.assertIs(planet.extras["is_moving"], False)
        self.assertEqual(planet.orbit_radius, 0.0)

    def test_moved_from_initial_planets_sets_coherent_orbit_metadata(self):
        state = parse_observation(
            {
                "planets": [[1, 0, 55.0, 50.0, 2.0, 10, 1]],
                "fleets": [],
                "initial_planets": [[1, 0, 50.0, 50.0, 2.0, 10, 1]],
                "angular_velocity": 0.1,
                "sunX": 50.0,
                "sunY": 50.0,
            }
        )

        planet = state.planets[0]
        self.assertIs(planet.extras["is_moving"], True)
        self.assertEqual(planet.orbit_center_x, 50.0)
        self.assertEqual(planet.orbit_center_y, 50.0)
        self.assertEqual(planet.orbit_radius, 5.0)
        self.assertEqual(planet.orbit_angle, 0.0)
        self.assertEqual(planet.orbit_speed, 0.1)

    def test_valid_explicit_orbit_metadata_marks_planet_moving_without_inference(self):
        state = parse_observation(
            {
                "planets": [
                    {
                        "id": 2,
                        "owner": -1,
                        "x": 60.0,
                        "y": 50.0,
                        "radius": 2.0,
                        "ships": 5,
                        "production": 1,
                        "orbit_center_x": 50.0,
                        "orbit_center_y": 50.0,
                        "orbit_radius": 10.0,
                        "orbit_angle": 0.0,
                        "orbit_speed": 0.05,
                    }
                ],
                "fleets": [],
            }
        )

        planet = state.planets[0]
        self.assertIs(planet.extras["is_moving"], True)
        self.assertEqual(planet.orbit_radius, 10.0)
        self.assertEqual(planet.orbit_speed, 0.05)

    def test_incomplete_explicit_orbit_metadata_does_not_mark_moving(self):
        state = parse_observation(
            {
                "planets": [
                    {
                        "id": 2,
                        "owner": -1,
                        "x": 60.0,
                        "y": 50.0,
                        "radius": 2.0,
                        "ships": 5,
                        "production": 1,
                        "orbit_speed": 0.05,
                    }
                ],
                "fleets": [],
            }
        )

        planet = state.planets[0]
        self.assertIs(planet.extras["is_moving"], False)
        self.assertEqual(planet.orbit_radius, 0.0)

    def test_radius_and_speed_without_center_and_angle_is_not_valid_explicit_orbit(self):
        state = parse_observation(
            {
                "planets": [
                    {
                        "id": 2,
                        "owner": -1,
                        "x": 60.0,
                        "y": 50.0,
                        "radius": 2.0,
                        "ships": 5,
                        "production": 1,
                        "orbit_radius": 10.0,
                        "orbit_speed": 0.05,
                    }
                ],
                "fleets": [],
            }
        )

        planet = state.planets[0]
        self.assertIs(planet.extras["is_moving"], False)
        self.assertEqual(planet.orbit_radius, 0.0)
        self.assertEqual(planet.orbit_speed, 0.0)

    def test_non_finite_orbit_metadata_is_normalized_to_stationary(self):
        state = parse_observation(
            {
                "planets": [
                    {
                        "id": 2,
                        "owner": -1,
                        "x": 60.0,
                        "y": 50.0,
                        "radius": 2.0,
                        "ships": 5,
                        "production": 1,
                        "orbit_center_x": 50.0,
                        "orbit_center_y": 50.0,
                        "orbit_radius": float("nan"),
                        "orbit_angle": 0.0,
                        "orbit_speed": float("inf"),
                    }
                ],
                "fleets": [],
            }
        )

        planet = state.planets[0]
        self.assertIs(planet.extras["is_moving"], False)
        self.assertEqual(planet.orbit_radius, 0.0)
        self.assertEqual(planet.orbit_speed, 0.0)

    def test_incomplete_initial_planets_do_not_verify_movement(self):
        state = parse_observation(
            {
                "planets": [[1, 0, 55.0, 50.0, 2.0, 10, 1]],
                "fleets": [],
                "initial_planets": [{"id": 1, "owner": 0}],
                "angular_velocity": 0.1,
            }
        )

        planet = state.planets[0]
        self.assertIs(planet.extras["is_moving"], False)
        self.assertEqual(planet.orbit_radius, 0.0)

    def test_comet_metadata_marks_only_mapped_planets_moving(self):
        path = [(10.0, 10.0), (12.0, 12.0)]
        state = parse_observation(
            {
                "planets": [
                    [1, -1, 10.0, 10.0, 2.0, 5, 1],
                    [2, -1, 20.0, 20.0, 2.0, 5, 1],
                ],
                "fleets": [],
                "comets": [{"planet_ids": [1], "paths": [path], "path_index": 0}],
            }
        )

        comet = state.planets[0]
        stationary = state.planets[1]
        self.assertIs(comet.extras["is_comet"], True)
        self.assertIs(comet.extras["is_moving"], True)
        self.assertEqual(comet.extras["comet_path"], path)
        self.assertEqual(comet.extras["comet_path_index"], 0)
        self.assertIs(stationary.extras["is_moving"], False)

    def test_object_like_comet_metadata_marks_mapped_planet_moving(self):
        path = [(10.0, 10.0), (12.0, 12.0)]
        state = parse_observation(
            Obj(
                planets=[
                    [1, -1, 10.0, 10.0, 2.0, 5, 1],
                    [2, -1, 20.0, 20.0, 2.0, 5, 1],
                ],
                fleets=[],
                comets=[Obj(planet_ids=[1], paths=[path], path_index=0)],
            )
        )

        comet = state.planets[0]
        stationary = state.planets[1]
        self.assertIs(comet.extras["is_comet"], True)
        self.assertIs(comet.extras["is_moving"], True)
        self.assertEqual(comet.extras["comet_path"], path)
        self.assertIs(stationary.extras["is_moving"], False)

    def test_single_comet_direct_path_shape_is_preserved_as_full_path(self):
        path = [(10.0, 10.0), (12.0, 12.0)]
        state = parse_observation(
            {
                "planets": [[1, -1, 10.0, 10.0, 2.0, 5, 1]],
                "fleets": [],
                "comets": [{"planet_ids": [1], "paths": path, "path_index": 0}],
            }
        )

        comet = state.planets[0]
        self.assertIs(comet.extras["is_moving"], True)
        self.assertEqual(comet.extras["comet_path"], path)

    def test_parse_none_and_malformed_scalar_return_empty_state(self):
        none_state = parse_observation(None)
        scalar_state = parse_observation(42)

        self.assertEqual(none_state.step, 0)
        self.assertEqual(none_state.player_id, 0)
        self.assertEqual(none_state.planets, [])
        self.assertEqual(none_state.fleets, [])

        self.assertEqual(scalar_state.step, 0)
        self.assertEqual(scalar_state.player_id, 0)
        self.assertEqual(scalar_state.planets, [])
        self.assertEqual(scalar_state.fleets, [])

    def test_parser_does_not_mark_stationary_planet_moving_from_angular_velocity_only(self):
        state = parse_observation(
            {
                "planets": [[1, 0, 40.0, 50.0, 3.0, 20, 1]],
                "fleets": [],
                "angular_velocity": 0.03,
            }
        )

        planet = state.planets[0]
        self.assertEqual(planet.orbit_radius, 0.0)
        self.assertIs(planet.extras["is_moving"], False)


if __name__ == "__main__":
    unittest.main()
