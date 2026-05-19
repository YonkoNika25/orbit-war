from __future__ import annotations

import math
from typing import Optional, Tuple

from .types import Planet


def distance(a: Planet, b: Planet) -> float:
    return math.hypot(b.x - a.x, b.y - a.y)


def angle_to(source: Tuple[float, float], target: Tuple[float, float]) -> float:
    return math.atan2(target[1] - source[1], target[0] - source[0])


def line_segment_min_distance(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    px: float,
    py: float,
) -> float:
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return math.hypot(x1 - px, y1 - py)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    return math.hypot(x1 + t * dx - px, y1 + t * dy - py)


def path_crosses_sun(
    source: Tuple[float, float],
    target: Tuple[float, float],
    margin: float = 1.5,
    sun_x: float = 50.0,
    sun_y: float = 50.0,
    sun_radius: float = 10.0,
) -> bool:
    return (
        line_segment_min_distance(
            source[0],
            source[1],
            target[0],
            target[1],
            sun_x,
            sun_y,
    )
    < sun_radius + margin
)


def predict_position(planet: Planet, future_step: int) -> Tuple[float, float]:
    """Predict planet position after a number of turns from the current snapshot."""

    if future_step <= 0:
        return planet.x, planet.y

    comet_path = planet.extras.get("comet_path") if planet.extras else None
    if comet_path:
        try:
            path = list(comet_path)
        except TypeError:
            path = []
        if path:
            index = int(planet.extras.get("comet_path_index", 0))
            target_index = min(len(path) - 1, max(0, index + future_step))
            point = path[target_index]
            return float(point[0]), float(point[1])

    if planet.orbit_speed == 0 or planet.orbit_radius <= 0:
        return planet.x, planet.y

    angle = planet.orbit_angle + planet.orbit_speed * future_step
    x = planet.orbit_center_x + math.cos(angle) * planet.orbit_radius
    y = planet.orbit_center_y + math.sin(angle) * planet.orbit_radius
    return x, y


def estimate_eta(source: Planet, target: Planet, speed: Optional[float] = None) -> int:
    fleet_speed = speed if speed and speed > 0 else 1.0
    return max(1, int(math.ceil(distance(source, target) / fleet_speed)))


def fleet_speed_for_ships(ships: int, max_speed: float = 6.0) -> float:
    """
    Orbit Wars fleet speed scales with fleet size.
    Formula from the env docs, clamped for safety.
    """

    if max_speed <= 1.0:
        return 1.0

    ships = max(1, int(ships))
    if ships <= 1:
        return 1.0

    ratio = math.log(ships) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (max_speed - 1.0) * (ratio ** 1.5)


def ray_circle_hit_distance(
    origin: Tuple[float, float],
    angle: float,
    center: Tuple[float, float],
    radius: float,
) -> Optional[float]:
    """
    Distance along a ray to the first intersection with a circle.

    Returns None when the ray misses the circle.
    """

    ox, oy = origin
    cx, cy = center
    dx = math.cos(angle)
    dy = math.sin(angle)

    vx = cx - ox
    vy = cy - oy
    projection = vx * dx + vy * dy
    if projection < 0:
        return None

    center_distance_sq = vx * vx + vy * vy
    perp_sq = center_distance_sq - projection * projection
    radius_sq = radius * radius
    if perp_sq > radius_sq:
        return None

    offset = math.sqrt(max(0.0, radius_sq - perp_sq))
    return max(0.0, projection - offset)
