from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .types import Fleet, GameState, Planet


def parse_observation(obs: Any, config: Any = None) -> GameState:
    data = _to_mapping(obs)
    config_map = _to_mapping(config)
    config_map = {**config_map}

    for key in (
        "angular_velocity",
        "initial_planets",
        "comets",
        "comet_planet_ids",
        "next_fleet_id",
        "sunRadius",
        "boardSize",
    ):
        if key in data and key not in config_map:
            config_map[key] = data[key]

    step = _pick_int(data, ("step", "turn", "time"), default=0)
    player_id = _pick_int(
        data,
        ("player_id", "player", "current_player", "team_id", "agent_id"),
        default=0,
    )

    meta = _build_meta(data, config_map)

    planets_raw = _pick_any(data, ("planets", "planet_list", "nodes"), default=[])
    fleets_raw = _pick_any(data, ("fleets", "fleet_list", "ships_in_flight"), default=[])

    planets = [
        _coerce_planet(item, index, meta)
        for index, item in enumerate(_ensure_list(planets_raw))
    ]
    fleets = [_coerce_fleet(item) for item in _ensure_list(fleets_raw)]

    _apply_comet_metadata(planets, meta)
    _apply_initial_motion_metadata(planets, meta)

    return GameState(step=step, player_id=player_id, planets=planets, fleets=fleets, config=config_map)


def _to_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if isinstance(value, (list, tuple)):
        return _sequence_to_root_map(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _sequence_to_root_map(value: Sequence[Any]) -> Dict[str, Any]:
    if len(value) >= 4 and _looks_like_collection(value[2]) and _looks_like_collection(value[3]):
        return {
            "step": value[0],
            "player_id": value[1],
            "planets": value[2],
            "fleets": value[3],
        }
    if len(value) >= 2 and _looks_like_collection(value[0]) and _looks_like_collection(value[1]):
        return {
            "planets": value[0],
            "fleets": value[1],
        }
    return {}


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _looks_like_collection(value: Any) -> bool:
    return isinstance(value, (list, tuple, dict))


def _pick_any(data: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return default


def _pick_int(data: Dict[str, Any], keys: Sequence[str], default: int = 0) -> int:
    for key in keys:
        if key in data:
            try:
                return int(data[key])
            except (TypeError, ValueError):
                continue
    return default


def _coerce_planet(item: Any, fallback_id: int, meta: Dict[str, Any]) -> Planet:
    if isinstance(item, Planet):
        return item

    if isinstance(item, dict):
        payload = item
    elif isinstance(item, (list, tuple)):
        payload = _sequence_to_planet_map(item, fallback_id)
    else:
        payload = {}

    planet_id = _first_int(payload, ("id", "planet_id", "index"), fallback_id)
    owner = _first_int(payload, ("owner", "owner_id", "team"), -1)
    ships = _first_int(payload, ("ships", "ship_count", "units"), 0)
    production = _first_float(payload, ("production", "prod", "income"), 0.0)
    x = _first_float(payload, ("x", "pos_x", "cx"), 0.0)
    y = _first_float(payload, ("y", "pos_y", "cy"), 0.0)
    radius = _first_float(payload, ("radius", "r"), 1.0)
    orbit_center_x = _first_float(payload, ("orbit_center_x", "center_x", "cx0"), meta["center_x"])
    orbit_center_y = _first_float(payload, ("orbit_center_y", "center_y", "cy0"), meta["center_y"])
    orbit_angle = _first_float(payload, ("orbit_angle", "angle"), 0.0)
    orbit_speed = _first_float(payload, ("orbit_speed", "angular_speed", "speed_angle"), 0.0)
    orbit_radius = _first_float(payload, ("orbit_radius", "orbital_radius"), 0.0)

    # Do NOT infer orbit/movement here from angular_velocity alone.
    # In Orbit Wars, not every planet inside the sun-centered rotation radius is
    # necessarily a moving planet. The original bot first waits, then compares
    # current coordinates with initial_planets. Inferring movement for every
    # eligible planet makes the bot aim at fake future positions, which is the
    # most common reason fleets miss and fly out of the map.
    explicit_orbit = any(
        key in payload
        for key in (
            "orbit_radius",
            "orbital_radius",
            "orbit_speed",
            "angular_speed",
            "speed_angle",
            "orbit_center_x",
            "center_x",
            "cx0",
            "orbit_center_y",
            "center_y",
            "cy0",
        )
    )
    if orbit_radius <= 0.0 or not explicit_orbit:
        orbit_radius = 0.0
        if orbit_speed == 0.0:
            orbit_center_x = x
            orbit_center_y = y

    extras = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "id",
            "planet_id",
            "index",
            "owner",
            "owner_id",
            "team",
            "ships",
            "ship_count",
            "units",
            "production",
            "prod",
            "income",
            "x",
            "pos_x",
            "cx",
            "y",
            "pos_y",
            "cy",
            "radius",
            "r",
            "orbit_center_x",
            "center_x",
            "cx0",
            "orbit_center_y",
            "center_y",
            "cy0",
            "orbit_radius",
            "orbital_radius",
            "orbit_angle",
            "angle",
            "orbit_speed",
            "angular_speed",
            "speed_angle",
        }
    }

    return Planet(
        id=planet_id,
        owner=owner,
        ships=ships,
        production=production,
        x=x,
        y=y,
        radius=radius,
        orbit_center_x=orbit_center_x,
        orbit_center_y=orbit_center_y,
        orbit_radius=orbit_radius,
        orbit_angle=orbit_angle,
        orbit_speed=orbit_speed,
        extras=extras,
    )


def _sequence_to_planet_map(values: Sequence[Any], fallback_id: int) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}

    # Orbit Wars expected planet tuple/list format:
    # [id, owner, x, y, radius, ships, production]
    if len(values) > 0:
        payload["id"] = values[0]
    else:
        payload["id"] = fallback_id

    if len(values) > 1:
        payload["owner"] = values[1]
    if len(values) > 2:
        payload["x"] = values[2]
    if len(values) > 3:
        payload["y"] = values[3]
    if len(values) > 4:
        payload["radius"] = values[4]
    if len(values) > 5:
        payload["ships"] = values[5]
    if len(values) > 6:
        payload["production"] = values[6]

    return payload


def _coerce_fleet(item: Any) -> Fleet:
    if isinstance(item, Fleet):
        return item

    if isinstance(item, dict):
        payload = item
    elif isinstance(item, (list, tuple)):
        payload = _sequence_to_fleet_map(item)
    else:
        payload = {}

    return Fleet(
        id=_first_int(payload, ("id", "fleet_id", "index"), -1),
        owner=_first_int(payload, ("owner", "owner_id", "team"), -1),
        x=_first_float(payload, ("x", "pos_x"), 0.0),
        y=_first_float(payload, ("y", "pos_y"), 0.0),
        angle=_first_float(payload, ("angle", "heading"), 0.0),
        from_planet_id=_first_optional_int(payload, ("from_planet_id", "source", "source_id", "from_planet")),
        ships=_first_int(payload, ("ships", "ship_count", "units"), 0),
        eta=_first_optional_float(payload, ("eta", "turns_left", "remaining")),
        speed=_first_float(payload, ("speed", "velocity"), 1.0),
        extras={
            key: value
            for key, value in payload.items()
            if key
            not in {
                "id",
                "fleet_id",
                "index",
                "owner",
                "owner_id",
                "team",
                "x",
                "pos_x",
                "y",
                "pos_y",
                "angle",
                "heading",
                "from_planet_id",
                "source",
                "source_id",
                "from_planet",
                "ships",
                "ship_count",
                "units",
                "eta",
                "turns_left",
                "remaining",
                "speed",
                "velocity",
            }
        },
    )


def _sequence_to_fleet_map(values: Sequence[Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if len(values) > 0:
        payload["id"] = values[0]
    if len(values) > 1:
        payload["owner"] = values[1]
    if len(values) > 2:
        payload["x"] = values[2]
    if len(values) > 3:
        payload["y"] = values[3]
    if len(values) > 4:
        payload["angle"] = values[4]
    if len(values) > 5:
        payload["from_planet_id"] = values[5]
    if len(values) > 6:
        payload["ships"] = values[6]
    return payload


def _build_meta(data: Dict[str, Any], config_map: Dict[str, Any]) -> Dict[str, Any]:
    combined = {**config_map, **data}
    board_size = _first_float(combined, ("boardSize", "board_size", "size"), 100.0)
    sun_radius = _first_float(combined, ("sunRadius", "sun_radius"), 10.0)
    center_x = _first_float(combined, ("sunX", "sun_x", "center_x"), board_size / 2.0)
    center_y = _first_float(combined, ("sunY", "sun_y", "center_y"), board_size / 2.0)
    angular_velocity = _first_float(combined, ("angular_velocity", "angularVelocity"), 0.0)
    rotation_radius_limit = 50.0

    comet_map: Dict[int, Dict[str, Any]] = {}
    comets = _pick_any(data, ("comets",), default=[])
    for group in _ensure_list(comets):
        if not isinstance(group, dict):
            continue
        planet_ids = _ensure_list(group.get("planet_ids"))
        paths = _ensure_list(group.get("paths"))
        path_index = _first_int(group, ("path_index",), 0)
        for idx, pid in enumerate(planet_ids):
            try:
                pid_int = int(pid)
            except (TypeError, ValueError):
                continue
            path = paths[idx] if idx < len(paths) else []
            comet_map[pid_int] = {"path": path, "path_index": path_index}

    initial_by_id: Dict[int, Tuple[float, float]] = {}
    for index, item in enumerate(_ensure_list(_pick_any(combined, ("initial_planets",), default=[]))):
        if isinstance(item, dict):
            payload = item
        elif isinstance(item, (list, tuple)):
            payload = _sequence_to_planet_map(item, index)
        else:
            continue
        planet_id = _first_int(payload, ("id", "planet_id", "index"), index)
        initial_by_id[planet_id] = (
            _first_float(payload, ("x", "pos_x", "cx"), 0.0),
            _first_float(payload, ("y", "pos_y", "cy"), 0.0),
        )

    return {
        "board_size": board_size,
        "sun_radius": sun_radius,
        "center_x": center_x,
        "center_y": center_y,
        "angular_velocity": angular_velocity,
        "rotation_radius_limit": rotation_radius_limit,
        "comet_map": comet_map,
        "initial_by_id": initial_by_id,
    }


def _apply_comet_metadata(planets: List[Planet], meta: Dict[str, Any]) -> None:
    comet_map = meta.get("comet_map", {})
    if not comet_map:
        return
    for planet in planets:
        info = comet_map.get(planet.id)
        if not info:
            continue
        planet.extras["is_comet"] = True
        planet.extras["is_moving"] = True
        planet.extras["comet_path"] = info.get("path", [])
        planet.extras["comet_path_index"] = info.get("path_index", 0)


def _apply_initial_motion_metadata(planets: List[Planet], meta: Dict[str, Any]) -> None:
    initial_by_id = meta.get("initial_by_id", {})
    angular_velocity = float(meta.get("angular_velocity", 0.0) or 0.0)
    if not initial_by_id and angular_velocity == 0.0:
        return

    for planet in planets:
        if planet.extras.get("is_comet"):
            continue

        initial = initial_by_id.get(planet.id)
        moved_from_initial = False
        if initial is not None:
            moved_from_initial = math.hypot(planet.x - initial[0], planet.y - initial[1]) > 1e-6

        has_explicit_orbit = abs(planet.orbit_speed) > 0.0 and planet.orbit_radius > 0.0
        if moved_from_initial and angular_velocity != 0.0:
            dx = planet.x - meta["center_x"]
            dy = planet.y - meta["center_y"]
            planet.orbit_center_x = meta["center_x"]
            planet.orbit_center_y = meta["center_y"]
            planet.orbit_radius = math.hypot(dx, dy)
            planet.orbit_angle = math.atan2(dy, dx)
            planet.orbit_speed = angular_velocity
            planet.extras["is_moving"] = True
        elif has_explicit_orbit:
            planet.extras["is_moving"] = True
        else:
            planet.extras["is_moving"] = False


def _first_int(payload: Dict[str, Any], keys: Iterable[str], default: int) -> int:
    value = _first_value(payload, keys, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _first_optional_int(payload: Dict[str, Any], keys: Iterable[str]) -> Optional[int]:
    value = _first_value(payload, keys, None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_float(payload: Dict[str, Any], keys: Iterable[str], default: float) -> float:
    value = _first_value(payload, keys, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_optional_float(payload: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
    value = _first_value(payload, keys, None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_value(payload: Dict[str, Any], keys: Iterable[str], default: Any) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return default
