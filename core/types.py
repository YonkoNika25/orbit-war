from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class Planet:
    id: int
    owner: int = -1
    ships: int = 0
    production: float = 0.0
    x: float = 0.0
    y: float = 0.0
    radius: float = 1.0
    orbit_center_x: float = 0.0
    orbit_center_y: float = 0.0
    orbit_radius: float = 0.0
    orbit_angle: float = 0.0
    orbit_speed: float = 0.0
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Fleet:
    id: int = -1
    owner: int = -1
    x: float = 0.0
    y: float = 0.0
    angle: float = 0.0
    from_planet_id: Optional[int] = None
    ships: int = 0
    eta: Optional[float] = None
    speed: float = 1.0
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> Optional[int]:
        return self.from_planet_id


@dataclass(slots=True)
class GameState:
    step: int
    player_id: int
    planets: List[Planet]
    fleets: List[Fleet]
    config: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def my_planets(self) -> List[Planet]:
        return [planet for planet in self.planets if planet.owner == self.player_id]

    @property
    def enemy_planets(self) -> List[Planet]:
        return [planet for planet in self.planets if planet.owner not in (-1, self.player_id)]

    @property
    def neutral_planets(self) -> List[Planet]:
        return [planet for planet in self.planets if planet.owner == -1]
