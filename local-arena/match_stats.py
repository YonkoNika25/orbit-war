from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional


SUN_X = 50.0
SUN_Y = 50.0
SUN_RADIUS = 10.0
DEFAULT_SHIP_SPEED = 6.0


def summarize_game_stats(result, game_result: Dict[str, Any], seat_names: List[str]) -> Dict[str, Any]:
    """Build per-game optimization stats from a Kaggle env result."""

    num_players = len(seat_names)
    player_stats = [_empty_player_stats(name) for name in seat_names]
    prev_fleets: Dict[int, Dict[str, Any]] = {}
    seen_fleets: set[int] = set()
    frame_count = 0

    for step_idx, step_data in enumerate(result):
        obs = _shared_observation(step_data)
        if not obs:
            continue

        frame_count += 1
        planets = _normalize_planets(obs.get("planets", []) or [])
        fleets = _normalize_fleets(obs.get("fleets", []) or [])
        _record_frame_player_state(player_stats, planets, fleets, num_players)

        current_fleets = {fleet["id"]: fleet for fleet in fleets}
        for fleet_id, fleet in current_fleets.items():
            if fleet_id not in seen_fleets and step_idx > 0:
                seen_fleets.add(fleet_id)
                owner = fleet["owner"]
                if 0 <= owner < num_players:
                    player_stats[owner]["launches"] += 1
                    player_stats[owner]["ships_launched"] += fleet["ships"]

        for fleet_id, previous in prev_fleets.items():
            if fleet_id in current_fleets:
                continue
            owner = previous["owner"]
            if not (0 <= owner < num_players):
                continue
            cause = _classify_fleet_disappearance(previous, planets)
            bucket = player_stats[owner]["fleet_disappearances"]
            bucket[cause]["count"] += 1
            bucket[cause]["ships"] += previous["ships"]

        prev_fleets = current_fleets

    _finalize_player_stats(player_stats, frame_count)
    return {
        "game_id": game_result.get("game_id"),
        "seat_names": seat_names,
        "winner": game_result.get("winner"),
        "winner_index": game_result.get("winner_index"),
        "rewards": game_result.get("rewards", []),
        "ship_counts": game_result.get("ship_counts", []),
        "steps": game_result.get("steps", 0),
        "elapsed": game_result.get("elapsed", 0.0),
        "statuses": game_result.get("statuses", []),
        "players": player_stats,
    }


def aggregate_game_stats(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    games = [game for game in results if not game.get("error")]
    summary: Dict[str, Any] = {
        "games": len(games),
        "wins": defaultdict(int),
        "avg_steps": 0.0,
        "avg_elapsed": 0.0,
        "players": defaultdict(_empty_aggregate_player_stats),
    }
    if not games:
        summary["wins"] = {}
        summary["players"] = {}
        return summary

    summary["avg_steps"] = sum(game.get("steps", 0) for game in games) / len(games)
    summary["avg_elapsed"] = sum(game.get("elapsed", 0.0) for game in games) / len(games)

    for game in games:
        winner = game.get("winner")
        if winner and winner != "TIE":
            summary["wins"][winner] += 1
        for player in game.get("stats", {}).get("players", []):
            aggregate = summary["players"][player["name"]]
            aggregate["games"] += 1
            for key in (
                "launches",
                "ships_launched",
                "final_planets",
                "final_total_ships",
                "max_planets",
                "max_total_ships",
            ):
                aggregate[key] += player.get(key, 0)
            for cause, values in player.get("fleet_disappearances", {}).items():
                aggregate["fleet_disappearances"][cause]["count"] += values.get("count", 0)
                aggregate["fleet_disappearances"][cause]["ships"] += values.get("ships", 0)

    for aggregate in summary["players"].values():
        games_count = max(1, aggregate["games"])
        for key in (
            "launches",
            "ships_launched",
            "final_planets",
            "final_total_ships",
            "max_planets",
            "max_total_ships",
        ):
            aggregate[f"avg_{key}"] = aggregate[key] / games_count

    summary["wins"] = dict(summary["wins"])
    summary["players"] = dict(summary["players"])
    return summary


def _empty_player_stats(name: str) -> Dict[str, Any]:
    return {
        "name": name,
        "launches": 0,
        "ships_launched": 0,
        "final_planets": 0,
        "final_planet_ships": 0,
        "final_fleet_ships": 0,
        "final_total_ships": 0,
        "max_planets": 0,
        "max_total_ships": 0,
        "avg_planets": 0.0,
        "avg_total_ships": 0.0,
        "_planet_sum": 0,
        "_ship_sum": 0,
        "fleet_disappearances": {
            "planet": {"count": 0, "ships": 0},
            "sun": {"count": 0, "ships": 0},
            "unknown": {"count": 0, "ships": 0},
        },
    }


def _empty_aggregate_player_stats() -> Dict[str, Any]:
    return {
        "games": 0,
        "launches": 0,
        "ships_launched": 0,
        "final_planets": 0,
        "final_total_ships": 0,
        "max_planets": 0,
        "max_total_ships": 0,
        "fleet_disappearances": {
            "planet": {"count": 0, "ships": 0},
            "sun": {"count": 0, "ships": 0},
            "unknown": {"count": 0, "ships": 0},
        },
    }


def _shared_observation(step_data) -> Optional[Dict[str, Any]]:
    for agent_data in step_data:
        obs = agent_data.get("observation")
        if obs is not None:
            return obs
    return None


def _normalize_planets(raw_planets) -> List[Dict[str, Any]]:
    planets = []
    for p in raw_planets:
        planets.append(
            {
                "id": int(p[0]),
                "owner": int(p[1]),
                "x": float(p[2]),
                "y": float(p[3]),
                "radius": float(p[4]),
                "ships": int(p[5]),
                "production": float(p[6]),
            }
        )
    return planets


def _normalize_fleets(raw_fleets) -> List[Dict[str, Any]]:
    fleets = []
    for f in raw_fleets:
        fleets.append(
            {
                "id": int(f[0]),
                "owner": int(f[1]),
                "x": float(f[2]),
                "y": float(f[3]),
                "angle": float(f[4]),
                "from_planet_id": int(f[5]),
                "ships": int(f[6]),
            }
        )
    return fleets


def _record_frame_player_state(
    player_stats: List[Dict[str, Any]],
    planets: List[Dict[str, Any]],
    fleets: List[Dict[str, Any]],
    num_players: int,
) -> None:
    for player_id in range(num_players):
        planet_count = 0
        planet_ships = 0
        fleet_ships = 0
        for planet in planets:
            if planet["owner"] == player_id:
                planet_count += 1
                planet_ships += planet["ships"]
        for fleet in fleets:
            if fleet["owner"] == player_id:
                fleet_ships += fleet["ships"]

        total_ships = planet_ships + fleet_ships
        stats = player_stats[player_id]
        stats["final_planets"] = planet_count
        stats["final_planet_ships"] = planet_ships
        stats["final_fleet_ships"] = fleet_ships
        stats["final_total_ships"] = total_ships
        stats["max_planets"] = max(stats["max_planets"], planet_count)
        stats["max_total_ships"] = max(stats["max_total_ships"], total_ships)
        stats["_planet_sum"] += planet_count
        stats["_ship_sum"] += total_ships


def _finalize_player_stats(player_stats: List[Dict[str, Any]], frame_count: int) -> None:
    frames = max(1, frame_count)
    for stats in player_stats:
        stats["avg_planets"] = stats["_planet_sum"] / frames
        stats["avg_total_ships"] = stats["_ship_sum"] / frames
        del stats["_planet_sum"]
        del stats["_ship_sum"]


def _classify_fleet_disappearance(fleet: Dict[str, Any], planets: List[Dict[str, Any]]) -> str:
    if _fleet_hits_sun_next_step(fleet):
        return "sun"
    if _fleet_hits_planet_next_step(fleet, planets):
        return "planet"
    return "unknown"


def _fleet_hits_sun_next_step(fleet: Dict[str, Any]) -> bool:
    end = _project_fleet(fleet)
    return _segment_point_distance(
        fleet["x"],
        fleet["y"],
        end[0],
        end[1],
        SUN_X,
        SUN_Y,
    ) <= SUN_RADIUS + 1.5


def _fleet_hits_planet_next_step(fleet: Dict[str, Any], planets: List[Dict[str, Any]]) -> bool:
    end = _project_fleet(fleet)
    for planet in planets:
        if _segment_point_distance(
            fleet["x"],
            fleet["y"],
            end[0],
            end[1],
            planet["x"],
            planet["y"],
        ) <= planet["radius"] + 1.5:
            return True
    return False


def _project_fleet(fleet: Dict[str, Any]) -> tuple[float, float]:
    speed = _fleet_speed_for_ships(fleet["ships"])
    return (
        fleet["x"] + math.cos(fleet["angle"]) * speed,
        fleet["y"] + math.sin(fleet["angle"]) * speed,
    )


def _fleet_speed_for_ships(ships: int, max_speed: float = DEFAULT_SHIP_SPEED) -> float:
    ships = max(1, int(ships))
    if max_speed <= 1.0 or ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (max_speed - 1.0) * (ratio ** 1.5)


def _segment_point_distance(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    px: float,
    py: float,
) -> float:
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return math.hypot(x1 - px, y1 - py)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_sq))
    return math.hypot(x1 + t * dx - px, y1 + t * dy - py)
