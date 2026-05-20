from __future__ import annotations

import os
import math
import unittest

from core.actions import candidate_to_action, safe_candidate_to_action, safe_noop, to_kaggle_action
from core.geometry import (
    angle_to,
    distance,
    estimate_eta,
    find_intercept_angle,
    fleet_speed_for_ships,
    is_sun_safe_route,
    line_segment_min_distance,
    path_crosses_sun,
    ray_circle_hit_distance,
    predict_position,
)
from core.parser import parse_observation
from rl.candidates import generate_candidates
from rl.local_state import build_local_turn_state
from rl.masks import validate_and_mask
from rl.types import CandidateType


RUN_KAGGLE_SMOKE = os.environ.get("RUN_KAGGLE_SMOKE") == "1"


@unittest.skipUnless(RUN_KAGGLE_SMOKE, "set RUN_KAGGLE_SMOKE=1 to run Kaggle environment smoke tests")
class KaggleEnvironmentSmokeTests(unittest.TestCase):
    def _epic2_batch_for_observation(self, obs, config):
        parsed = parse_observation(obs, config)
        local = build_local_turn_state(parsed)
        candidates = generate_candidates(local)
        batch = validate_and_mask(local, candidates)
        return parsed, local, candidates, batch

    def _epic2_agent_action(self, obs, config, max_actions: int = 2):
        parsed = parse_observation(obs, config)
        local = build_local_turn_state(parsed)
        actions = []

        for _ in range(max_actions):
            candidates = generate_candidates(local)
            batch = validate_and_mask(local, candidates)
            selectable = [
                candidate
                for candidate, legal in zip(batch.candidates, batch.mask)
                if legal and candidate.type not in {CandidateType.STOP, CandidateType.HOLD_SOURCE}
            ]
            if not selectable:
                break
            selected = selectable[0]
            action = safe_candidate_to_action(selected)
            if action == []:
                break
            actions.append(action)
            local.apply(selected)

        return actions if actions else safe_noop()

    def _find_clear_launch_angle(self, source, planets, speed: float, ticks: int = 10) -> float:
        for degrees in range(0, 360, 5):
            angle = math.radians(degrees)
            current = (
                source.x + math.cos(angle) * (source.radius + 0.1),
                source.y + math.sin(angle) * (source.radius + 0.1),
            )
            clear = True
            for _ in range(ticks):
                previous = current
                current = (
                    previous[0] + math.cos(angle) * speed,
                    previous[1] + math.sin(angle) * speed,
                )
                if not (0.0 <= current[0] <= 100.0 and 0.0 <= current[1] <= 100.0):
                    clear = False
                    break
                if line_segment_min_distance(previous[0], previous[1], current[0], current[1], 50.0, 50.0) <= 10.25:
                    clear = False
                    break
                for planet in planets:
                    if planet.id == source.id:
                        continue
                    if (
                        line_segment_min_distance(previous[0], previous[1], current[0], current[1], planet.x, planet.y)
                        <= planet.radius + 0.25
                    ):
                        clear = False
                        break
                if not clear:
                    break
            if clear:
                return angle
        raise AssertionError("no clear launch angle found for smoke test")

    def _fleet_expected_position(self, source, angle: float, speed: float, ticks: int) -> tuple[float, float]:
        travel = source.radius + 0.1 + speed * ticks
        return source.x + math.cos(angle) * travel, source.y + math.sin(angle) * travel

    def _find_static_hit_target(self, source, planets, ships: int, speed: float):
        for target in planets:
            if target.id == source.id:
                continue
            angle = angle_to((source.x, source.y), (target.x, target.y))
            current = (
                source.x + math.cos(angle) * (source.radius + 0.1),
                source.y + math.sin(angle) * (source.radius + 0.1),
            )
            for tick in range(1, 40):
                previous = current
                current = (
                    previous[0] + math.cos(angle) * speed,
                    previous[1] + math.sin(angle) * speed,
                )
                if not (0.0 <= current[0] <= 100.0 and 0.0 <= current[1] <= 100.0):
                    break
                if line_segment_min_distance(previous[0], previous[1], current[0], current[1], 50.0, 50.0) < 10.0:
                    break
                hit_other = False
                for planet in planets:
                    if planet.id in {source.id, target.id}:
                        continue
                    if line_segment_min_distance(previous[0], previous[1], current[0], current[1], planet.x, planet.y) < planet.radius:
                        hit_other = True
                        break
                if hit_other:
                    break
                if line_segment_min_distance(previous[0], previous[1], current[0], current[1], target.x, target.y) < target.radius:
                    return target, angle, tick
        raise AssertionError("no static hit target found for smoke test")

    def _find_moving_intercept_target(self, source, planets, speed: float):
        moving_targets = [
            planet
            for planet in planets
            if planet.id != source.id and planet.extras.get("is_moving")
        ]
        for target in moving_targets:
            intercept = find_intercept_angle(source, target, speed, max_steps=60)
            if intercept is None:
                continue
            angle, hit_tick = intercept
            current = (
                source.x + math.cos(angle) * (source.radius + 0.1),
                source.y + math.sin(angle) * (source.radius + 0.1),
            )
            clear = True
            for tick in range(1, hit_tick + 1):
                previous = current
                current = (
                    previous[0] + math.cos(angle) * speed,
                    previous[1] + math.sin(angle) * speed,
                )
                if line_segment_min_distance(previous[0], previous[1], current[0], current[1], 50.0, 50.0) < 10.0:
                    clear = False
                    break
                for planet in planets:
                    if planet.id in {source.id, target.id}:
                        continue
                    planet_x, planet_y = predict_position(planet, tick)
                    if (
                        line_segment_min_distance(previous[0], previous[1], current[0], current[1], planet_x, planet_y)
                        < planet.radius
                    ):
                        clear = False
                        break
                if not clear:
                    break
            if clear:
                return target, angle, hit_tick
        raise AssertionError("no moving intercept target found for smoke test")

    def test_real_orbit_wars_reset_observation_parses(self) -> None:
        from kaggle_environments import make

        env = make("orbit_wars", configuration={"seed": 7}, debug=True)
        state = env.reset()
        raw_observation = state[0].observation

        parsed = parse_observation(raw_observation, env.configuration)

        self.assertEqual(parsed.step, 0)
        self.assertEqual(parsed.player_id, 0)
        self.assertGreater(len(parsed.planets), 0)
        self.assertEqual(len(parsed.fleets), 0)
        self.assertIn("angular_velocity", parsed.config)
        self.assertIn("initial_planets", parsed.config)
        self.assertIn("remainingOverageTime", parsed.extras)
        self.assertGreaterEqual(len(parsed.my_planets), 1)

    def test_real_orbit_wars_observation_exercises_geometry_helpers(self) -> None:
        from kaggle_environments import make

        env = make("orbit_wars", configuration={"seed": 7}, debug=True)
        state = env.reset()
        parsed = parse_observation(state[0].observation, env.configuration)
        source = parsed.my_planets[0]
        target = next(planet for planet in parsed.planets if planet.id != source.id)

        source_point = (source.x, source.y)
        target_point = (target.x, target.y)
        launch_angle = angle_to(source_point, target_point)
        route_distance = distance(source, target)
        sun_distance = line_segment_min_distance(
            source.x,
            source.y,
            target.x,
            target.y,
            50.0,
            50.0,
        )
        crosses_sun = path_crosses_sun(source_point, target_point)
        ray_hit = ray_circle_hit_distance(source_point, launch_angle, (50.0, 50.0), 10.0)

        self.assertGreater(route_distance, 0.0)
        self.assertTrue(-3.2 <= launch_angle <= 3.2)
        self.assertGreaterEqual(sun_distance, 0.0)
        self.assertNotEqual(crosses_sun, is_sun_safe_route(source_point, target_point))
        self.assertTrue(ray_hit is None or ray_hit >= 0.0)
        self.assertGreater(fleet_speed_for_ships(5, max_speed=env.configuration.shipSpeed), 1.0)
        self.assertGreaterEqual(estimate_eta(source, target, speed=2.0), 1)

    def test_parse_only_agent_runs_in_real_orbit_wars_environment(self) -> None:
        from kaggle_environments import make

        parse_calls = {"count": 0}

        def parse_only_agent(obs, config):
            parse_calls["count"] += 1
            parsed = parse_observation(obs, config)
            self.assertGreater(len(parsed.planets), 0)
            return safe_noop()

        env = make("orbit_wars", configuration={"seed": 7}, debug=True)
        result = env.run([parse_only_agent, "random"])
        final = result[-1]

        self.assertGreater(parse_calls["count"], 0)
        self.assertEqual([item.status for item in final], ["DONE", "DONE"])
        self.assertEqual(len(final), 2)

    def test_action_and_geometry_helpers_match_real_launched_fleet(self) -> None:
        from kaggle_environments import make

        launch = {}

        def one_launch_agent(obs, config):
            parsed = parse_observation(obs, config)
            if parsed.step > 0 or not parsed.my_planets:
                return safe_noop()

            source = parsed.my_planets[0]
            targets = [planet for planet in parsed.planets if planet.id != source.id]
            safe_targets = [
                planet
                for planet in targets
                if is_sun_safe_route((source.x, source.y), (planet.x, planet.y))
            ]
            target = safe_targets[0] if safe_targets else targets[0]
            ships = max(1, min(5, source.ships - 1))
            angle = angle_to((source.x, source.y), (target.x, target.y))
            action = to_kaggle_action(source.id, angle, ships)

            self.assertEqual(
                candidate_to_action({"source_id": source.id, "angle": angle, "ships": ships}),
                action,
            )
            launch.update(
                {
                    "source_id": source.id,
                    "source_x": source.x,
                    "source_y": source.y,
                    "source_radius": source.radius,
                    "angle": angle,
                    "ships": ships,
                    "speed": fleet_speed_for_ships(ships, max_speed=config.shipSpeed),
                }
            )
            return [action]

        def noop_agent(obs, config):
            return safe_noop()

        env = make("orbit_wars", configuration={"seed": 7}, debug=True)
        result = env.run([one_launch_agent, noop_agent])
        first_turn_fleets = result[1][0].observation.fleets

        self.assertEqual([item.status for item in result[-1]], ["DONE", "DONE"])
        self.assertEqual(len(first_turn_fleets), 1)

        fleet = first_turn_fleets[0]
        self.assertEqual(fleet[1], 0)
        self.assertEqual(fleet[4], launch["angle"])
        self.assertEqual(fleet[5], launch["source_id"])
        self.assertEqual(fleet[6], launch["ships"])
        self.assertAlmostEqual(
            ((fleet[2] - launch["source_x"]) ** 2 + (fleet[3] - launch["source_y"]) ** 2) ** 0.5,
            launch["source_radius"] + 0.1 + launch["speed"],
        )

    def test_real_fleet_position_matches_geometry_after_5_and_10_ticks(self) -> None:
        from kaggle_environments import make

        env = make("orbit_wars", configuration={"seed": 7}, debug=True)
        state = env.reset()
        parsed = parse_observation(state[0].observation, env.configuration)
        source = parsed.my_planets[0]
        ships = 5
        speed = fleet_speed_for_ships(ships, max_speed=env.configuration.shipSpeed)
        angle = self._find_clear_launch_angle(source, parsed.planets, speed, ticks=10)

        state = env.step([[[source.id, angle, ships]], []])
        for expected_tick in range(1, 11):
            fleet = state[0].observation.fleets[0]
            expected_x, expected_y = self._fleet_expected_position(source, angle, speed, expected_tick)
            if expected_tick in {5, 10}:
                self.assertAlmostEqual(fleet[2], expected_x)
                self.assertAlmostEqual(fleet[3], expected_y)
            if expected_tick < 10:
                state = env.step([[], []])

    def test_real_orbiting_planet_matches_predict_position_after_5_and_10_steps(self) -> None:
        from kaggle_environments import make

        env = make("orbit_wars", configuration={"seed": 1}, debug=True)
        env.reset()
        env.step([[], []])
        state = env.step([[], []])
        parsed = parse_observation(state[0].observation, env.configuration)
        moving_planet = next(planet for planet in parsed.planets if planet.extras.get("is_moving"))

        snapshots = {0: parsed}
        for tick in range(1, 11):
            state = env.step([[], []])
            if tick in {5, 10}:
                snapshots[tick] = parse_observation(state[0].observation, env.configuration)

        for future_step in (5, 10):
            actual_planet = next(planet for planet in snapshots[future_step].planets if planet.id == moving_planet.id)
            predicted_x, predicted_y = predict_position(moving_planet, future_step)
            self.assertAlmostEqual(actual_planet.x, predicted_x)
            self.assertAlmostEqual(actual_planet.y, predicted_y)

    def test_real_fleet_hits_static_target_on_predicted_collision_tick(self) -> None:
        from kaggle_environments import make

        env = make("orbit_wars", configuration={"seed": 7}, debug=True)
        state = env.reset()
        parsed = parse_observation(state[0].observation, env.configuration)
        source = parsed.my_planets[0]
        ships = max(1, source.ships - 1)
        speed = fleet_speed_for_ships(ships, max_speed=env.configuration.shipSpeed)
        target, angle, hit_tick = self._find_static_hit_target(source, parsed.planets, ships, speed)

        state = env.step([[[source.id, angle, ships]], []])
        for tick in range(1, hit_tick):
            self.assertTrue(
                any(fleet[5] == source.id and fleet[6] == ships for fleet in state[0].observation.fleets),
                f"fleet disappeared before predicted hit tick {hit_tick}",
            )
            state = env.step([[], []])

        self.assertFalse(
            any(fleet[5] == source.id and fleet[6] == ships for fleet in state[0].observation.fleets),
            "fleet should be removed after hitting target",
        )
        actual_target = next(planet for planet in state[0].observation.planets if planet[0] == target.id)
        self.assertIn(actual_target[1], {-1, 0, 1})

    def test_real_fleet_intercepts_moving_planet_on_predicted_tick(self) -> None:
        from kaggle_environments import make

        env = make("orbit_wars", configuration={"seed": 1}, debug=True)
        env.reset()
        env.step([[], []])
        state = env.step([[], []])
        parsed = parse_observation(state[0].observation, env.configuration)
        source = parsed.my_planets[0]
        ships = max(1, source.ships - 1)
        speed = fleet_speed_for_ships(ships, max_speed=env.configuration.shipSpeed)
        target, angle, hit_tick = self._find_moving_intercept_target(source, parsed.planets, speed)

        state = env.step([[[source.id, angle, ships]], []])
        for _ in range(1, hit_tick):
            self.assertTrue(
                any(fleet[5] == source.id and fleet[6] == ships for fleet in state[0].observation.fleets),
                f"fleet disappeared before predicted moving intercept tick {hit_tick}",
            )
            state = env.step([[], []])

        self.assertFalse(
            any(fleet[5] == source.id and fleet[6] == ships for fleet in state[0].observation.fleets),
            "fleet should be removed after intercepting moving target",
        )
        actual_target = next(planet for planet in state[0].observation.planets if planet[0] == target.id)
        self.assertIn(actual_target[1], {-1, 0, 1})

    def test_real_sun_safe_route_stays_active_for_10_ticks(self) -> None:
        from kaggle_environments import make

        env = make("orbit_wars", configuration={"seed": 7}, debug=True)
        state = env.reset()
        parsed = parse_observation(state[0].observation, env.configuration)
        source = parsed.my_planets[0]
        ships = 5
        speed = fleet_speed_for_ships(ships, max_speed=env.configuration.shipSpeed)
        angle = self._find_clear_launch_angle(source, parsed.planets, speed, ticks=10)

        state = env.step([[[source.id, angle, ships]], []])
        previous = None
        for _ in range(10):
            self.assertEqual(len(state[0].observation.fleets), 1)
            fleet = state[0].observation.fleets[0]
            current = (fleet[2], fleet[3])
            if previous is not None:
                self.assertFalse(path_crosses_sun(previous, current, margin=0.0))
            previous = current
            state = env.step([[], []])

    def test_epic2_pipeline_builds_candidates_and_masks_on_real_observations_over_25_steps(self) -> None:
        from kaggle_environments import make

        env = make("orbit_wars", configuration={"seed": 7}, debug=True)
        state = env.reset()

        saw_non_stop_candidate = False
        saw_rejection = False
        for _ in range(25):
            parsed, local, candidates, batch = self._epic2_batch_for_observation(
                state[0].observation,
                env.configuration,
            )

            self.assertGreater(len(parsed.planets), 0)
            self.assertEqual(batch.candidates[0].type, CandidateType.STOP)
            self.assertEqual(batch.mask[0], True)
            self.assertEqual(len(batch.candidates), len(candidates))
            self.assertEqual(len(batch.mask), len(candidates))
            self.assertEqual([candidate.type for candidate in batch.candidates], [candidate.type for candidate in candidates])

            for original, validated, legal in zip(candidates, batch.candidates, batch.mask):
                self.assertIs(original.legal, True)
                self.assertIsNone(original.reject_reason)
                self.assertEqual(validated.legal, legal)
                if legal:
                    self.assertIsNone(validated.reject_reason)
                else:
                    saw_rejection = True
                    self.assertIsNotNone(validated.reject_reason)
                    self.assertEqual(validated.reject_reason, validated.reject_reason.lower())
                    self.assertNotIn(" ", validated.reject_reason)

            launch_candidates = [
                candidate
                for candidate, legal in zip(batch.candidates, batch.mask)
                if legal and candidate.type not in {CandidateType.STOP, CandidateType.HOLD_SOURCE}
            ]
            saw_non_stop_candidate = saw_non_stop_candidate or bool(launch_candidates)
            self.assertEqual(local.source_commitments, {})

            state = env.step([[], []])

        self.assertTrue(saw_non_stop_candidate)
        self.assertIsInstance(saw_rejection, bool)

    def test_epic2_masked_agent_runs_real_environment_for_50_steps_without_invalid_actions(self) -> None:
        from kaggle_environments import make

        env = make("orbit_wars", configuration={"seed": 7}, debug=True)
        state = env.reset()
        launched_turns = 0
        observed_masks = 0

        for _ in range(50):
            action = self._epic2_agent_action(state[0].observation, env.configuration, max_actions=2)
            if action:
                launched_turns += 1
            parsed, _, candidates, batch = self._epic2_batch_for_observation(state[0].observation, env.configuration)
            observed_masks += 1
            self.assertEqual(len(batch.mask), len(candidates))
            self.assertTrue(any(batch.mask))
            self.assertGreaterEqual(len(parsed.my_planets), 0)

            state = env.step([action, []])
            self.assertNotIn(state[0].status, {"ERROR", "INVALID"})
            self.assertNotIn(state[1].status, {"ERROR", "INVALID"})

        self.assertGreater(observed_masks, 40)
        self.assertGreater(launched_turns, 0)

    def test_epic2_agent_sequential_apply_prevents_same_turn_overcommit_in_real_observation(self) -> None:
        from kaggle_environments import make

        env = make("orbit_wars", configuration={"seed": 7}, debug=True)
        state = env.reset()
        parsed = parse_observation(state[0].observation, env.configuration)
        local = build_local_turn_state(parsed)
        source = parsed.my_planets[0]

        actions = []
        selected_candidates = []
        for _ in range(4):
            candidates = generate_candidates(local)
            batch = validate_and_mask(local, candidates)
            legal_launches = [
                candidate
                for candidate, legal in zip(batch.candidates, batch.mask)
                if legal and candidate.type not in {CandidateType.STOP, CandidateType.HOLD_SOURCE}
            ]
            if not legal_launches:
                break
            selected = legal_launches[0]
            actions.append(candidate_to_action(selected))
            selected_candidates.append(selected)
            self.assertTrue(local.apply(selected))

        self.assertGreater(len(actions), 0)
        committed_by_source = {}
        for candidate in selected_candidates:
            committed_by_source[candidate.source_id] = committed_by_source.get(candidate.source_id, 0) + candidate.ships

        for source_id, committed in committed_by_source.items():
            self.assertLessEqual(committed, next(planet.ships for planet in parsed.planets if planet.id == source_id))
            self.assertEqual(local.committed_ships(source_id), committed)

        state = env.step([actions, []])
        self.assertNotIn(state[0].status, {"ERROR", "INVALID"})
        first_turn_fleets = [fleet for fleet in state[0].observation.fleets if fleet[1] == 0]
        self.assertLessEqual(len(first_turn_fleets), len(actions))
        self.assertLessEqual(local.committed_ships(source.id), source.ships)


if __name__ == "__main__":
    unittest.main()
