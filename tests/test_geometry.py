from __future__ import annotations

import math
import unittest

from core.geometry import (
    estimate_eta,
    find_intercept_angle,
    fleet_speed_for_ships,
    is_sun_safe_route,
    path_crosses_sun,
    predict_position,
)
from core.types import Planet


class GeometrySafetyTests(unittest.TestCase):
    def test_direct_route_through_sun_is_unsafe(self) -> None:
        self.assertTrue(path_crosses_sun((0.0, 50.0), (100.0, 50.0)))
        self.assertFalse(is_sun_safe_route((0.0, 50.0), (100.0, 50.0)))

    def test_near_miss_outside_margin_is_safe(self) -> None:
        self.assertFalse(path_crosses_sun((0.0, 62.0), (100.0, 62.0)))
        self.assertTrue(is_sun_safe_route((0.0, 62.0), (100.0, 62.0)))

    def test_route_on_sun_margin_boundary_is_unsafe(self) -> None:
        self.assertTrue(path_crosses_sun((0.0, 61.5), (100.0, 61.5)))
        self.assertFalse(is_sun_safe_route((0.0, 61.5), (100.0, 61.5)))

    def test_zero_length_segment_away_from_sun_is_safe(self) -> None:
        self.assertFalse(path_crosses_sun((0.0, 0.0), (0.0, 0.0)))
        self.assertTrue(is_sun_safe_route((0.0, 0.0), (0.0, 0.0)))

    def test_zero_length_segment_inside_sun_danger_radius_is_unsafe(self) -> None:
        self.assertTrue(path_crosses_sun((50.0, 50.0), (50.0, 50.0)))
        self.assertFalse(is_sun_safe_route((50.0, 50.0), (50.0, 50.0)))

    def test_normal_route_far_from_sun_is_safe(self) -> None:
        self.assertFalse(path_crosses_sun((0.0, 0.0), (20.0, 0.0)))
        self.assertTrue(is_sun_safe_route((0.0, 0.0), (20.0, 0.0)))

    def test_custom_sun_parameters_are_used(self) -> None:
        self.assertTrue(
            path_crosses_sun(
                (0.0, 5.0),
                (10.0, 5.0),
                margin=0.0,
                sun_x=5.0,
                sun_y=5.0,
                sun_radius=1.0,
            )
        )
        self.assertFalse(
            is_sun_safe_route(
                (0.0, 5.0),
                (10.0, 5.0),
                margin=0.0,
                sun_x=5.0,
                sun_y=5.0,
                sun_radius=1.0,
            )
        )

    def test_fleet_speed_for_ships_is_deterministic(self) -> None:
        self.assertEqual(fleet_speed_for_ships(1), 1.0)
        self.assertAlmostEqual(fleet_speed_for_ships(1000), 6.0)
        self.assertAlmostEqual(
            fleet_speed_for_ships(10),
            1.0 + 5.0 * ((math.log(10) / math.log(1000.0)) ** 1.5),
        )

    def test_estimate_eta_is_deterministic(self) -> None:
        source = Planet(id=1, x=0.0, y=0.0)
        target = Planet(id=2, x=3.0, y=4.0)

        self.assertEqual(estimate_eta(source, target, speed=2.0), 3)
        self.assertEqual(estimate_eta(source, target, speed=0.0), 5)

    def test_find_intercept_angle_for_stationary_target(self) -> None:
        source = Planet(id=1, x=0.0, y=0.0, radius=1.0)
        target = Planet(id=2, x=10.0, y=0.0, radius=1.0)

        intercept = find_intercept_angle(source, target, speed=2.0, max_steps=10)

        self.assertIsNotNone(intercept)
        angle, step = intercept
        self.assertAlmostEqual(angle, 0.0)
        self.assertEqual(step, 4)

    def test_find_intercept_angle_for_orbiting_target(self) -> None:
        source = Planet(id=1, x=0.0, y=0.0, radius=1.0)
        target = Planet(
            id=2,
            x=10.0,
            y=0.0,
            radius=1.0,
            orbit_center_x=0.0,
            orbit_center_y=0.0,
            orbit_radius=10.0,
            orbit_angle=0.0,
            orbit_speed=0.05,
        )

        intercept = find_intercept_angle(source, target, speed=2.0, max_steps=10)

        self.assertIsNotNone(intercept)
        angle, step = intercept
        predicted = predict_position(target, step)
        travel = source.radius + 0.1 + 2.0 * step
        fleet_x = source.x + math.cos(angle) * travel
        fleet_y = source.y + math.sin(angle) * travel
        self.assertLessEqual(math.hypot(fleet_x - predicted[0], fleet_y - predicted[1]), target.radius)


if __name__ == "__main__":
    unittest.main()
