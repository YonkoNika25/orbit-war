from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, Mapping, Tuple

from core.types import Fleet, GameState, Planet


@dataclass(slots=True)
class ArrivalBucket:
    owner: int
    eta: int
    ships: int = 0
    fleet_ids: Tuple[int, ...] = ()

    def add(self, fleet: Fleet) -> None:
        self.ships += max(0, int(fleet.ships))
        self.fleet_ids = (*self.fleet_ids, int(fleet.id))


@dataclass(slots=True)
class TargetTimeline:
    target_id: int
    current_owner: int
    current_ships: int
    projected_owner: int
    projected_ships: int
    friendly_by_eta: Dict[int, int] = field(default_factory=dict)
    enemy_by_eta: Dict[int, int] = field(default_factory=dict)


@dataclass(slots=True)
class LocalTurnState:
    state: GameState
    planets: Tuple[Planet, ...]
    fleets: Tuple[Fleet, ...]
    planets_by_id: Dict[int, Planet]
    owned_planet_ids: Tuple[int, ...]
    enemy_planet_ids: Tuple[int, ...]
    neutral_planet_ids: Tuple[int, ...]
    source_commitments: Dict[int, int] = field(default_factory=dict)
    friendly_arrivals: Dict[int, Dict[int, ArrivalBucket]] = field(default_factory=dict)
    enemy_arrivals: Dict[int, Dict[int, ArrivalBucket]] = field(default_factory=dict)
    target_timelines: Dict[int, TargetTimeline] = field(default_factory=dict)
    rejections: Dict[int, str] = field(default_factory=dict)

    def committed_ships(self, source_id: int) -> int:
        return max(0, int(self.source_commitments.get(int(source_id), 0)))

    def available_ships(self, source_id: int) -> int:
        planet = self.planets_by_id.get(int(source_id))
        if planet is None or planet.owner != self.state.player_id:
            return 0
        return max(0, int(planet.ships) - self.committed_ships(planet.id))

    def record_rejection(self, candidate_index: int, reason: str) -> None:
        self.rejections[int(candidate_index)] = str(reason)

    def apply(self, candidate: Any, candidate_index: int | None = None) -> bool:
        applied = _coerce_applied_candidate(candidate)
        if applied is None:
            self._record_apply_rejection(candidate_index, "invalid_candidate")
            return False

        source_id, target_id, ships, eta = applied
        timeline = self.target_timelines.get(target_id)
        if timeline is None or source_id not in self.planets_by_id:
            self._record_apply_rejection(candidate_index, "invalid_candidate")
            return False
        if ships <= 0 or eta < 0:
            self._record_apply_rejection(candidate_index, "invalid_candidate")
            return False
        if ships > self.available_ships(source_id):
            self._record_apply_rejection(candidate_index, "insufficient_ships")
            return False
        if (
            timeline.current_owner != self.state.player_id
            and timeline.projected_owner == self.state.player_id
        ):
            self._record_apply_rejection(candidate_index, "already_winning_target")
            return False

        self.source_commitments[source_id] = self.committed_ships(source_id) + ships
        timeline.friendly_by_eta[eta] = timeline.friendly_by_eta.get(eta, 0) + ships
        _update_timeline_projection(timeline, self.state.player_id)
        return True

    def _record_apply_rejection(self, candidate_index: int | None, reason: str) -> None:
        if candidate_index is not None:
            self.record_rejection(candidate_index, reason)


def build_local_turn_state(state: GameState) -> LocalTurnState:
    planets = tuple(state.planets)
    fleets = tuple(state.fleets)
    planets_by_id = {int(planet.id): planet for planet in planets}
    owned_planet_ids = tuple(planet.id for planet in planets if planet.owner == state.player_id)
    enemy_planet_ids = tuple(
        planet.id for planet in planets if planet.owner not in (-1, state.player_id)
    )
    neutral_planet_ids = tuple(planet.id for planet in planets if planet.owner == -1)

    local = LocalTurnState(
        state=state,
        planets=planets,
        fleets=fleets,
        planets_by_id=planets_by_id,
        owned_planet_ids=owned_planet_ids,
        enemy_planet_ids=enemy_planet_ids,
        neutral_planet_ids=neutral_planet_ids,
        target_timelines={
            int(planet.id): TargetTimeline(
                target_id=int(planet.id),
                current_owner=int(planet.owner),
                current_ships=int(planet.ships),
                projected_owner=int(planet.owner),
                projected_ships=int(planet.ships),
            )
            for planet in planets
        },
    )

    for fleet in fleets:
        target_id = _explicit_target_id(fleet)
        eta = _valid_eta(fleet)
        if target_id is None or eta is None:
            continue
        arrivals = (
            local.friendly_arrivals
            if fleet.owner == state.player_id
            else local.enemy_arrivals
        )
        _add_arrival(arrivals, target_id, eta, fleet)

    _attach_arrivals_to_timelines(local)
    return local


def _explicit_target_id(fleet: Fleet) -> int | None:
    aliases = ("target_id", "to_planet_id", "destination", "target")
    for key in aliases:
        if key not in fleet.extras:
            continue
        try:
            return int(fleet.extras[key])
        except (TypeError, ValueError):
            continue
    return None


def _coerce_applied_candidate(candidate: Any) -> tuple[int, int, int, int] | None:
    payload = _candidate_mapping(candidate)
    if not payload:
        return None
    source_id = _first_int(payload, ("source_id", "source", "from_planet_id"))
    target_id = _first_int(payload, ("target_id", "target", "to_planet_id", "destination"))
    ships = _first_int(payload, ("ships", "ship_count", "units"))
    eta = _first_int(payload, ("eta", "turns_left", "remaining"))
    if source_id is None or target_id is None or ships is None or eta is None:
        return None
    return source_id, target_id, ships, eta


def _candidate_mapping(candidate: Any) -> Dict[str, Any]:
    if isinstance(candidate, Mapping):
        return dict(candidate)
    if is_dataclass(candidate) and not isinstance(candidate, type):
        return {field.name: getattr(candidate, field.name) for field in fields(candidate)}
    if hasattr(candidate, "__dict__"):
        return dict(vars(candidate))
    if hasattr(candidate, "__slots__"):
        return {
            name: getattr(candidate, name)
            for name in _iter_slot_names(candidate)
            if hasattr(candidate, name)
        }
    return {}


def _iter_slot_names(value: Any) -> Tuple[str, ...]:
    names: list[str] = []
    for cls in type(value).__mro__:
        slots = getattr(cls, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name not in {"__dict__", "__weakref__"}:
                names.append(name)
    return tuple(names)


def _first_int(payload: Mapping[str, Any], keys: Tuple[str, ...]) -> int | None:
    for key in keys:
        if key not in payload:
            continue
        try:
            value = float(payload[key])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        try:
            return int(value)
        except (OverflowError, TypeError, ValueError):
            continue
    return None


def _valid_eta(fleet: Fleet) -> int | None:
    if fleet.eta is None:
        return None
    try:
        eta = float(fleet.eta)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(eta) or eta < 0:
        return None
    return int(math.ceil(eta))


def _add_arrival(
    arrivals: Dict[int, Dict[int, ArrivalBucket]],
    target_id: int,
    eta: int,
    fleet: Fleet,
) -> None:
    target_arrivals = arrivals.setdefault(int(target_id), {})
    bucket = target_arrivals.get(int(eta))
    if bucket is None:
        bucket = ArrivalBucket(owner=int(fleet.owner), eta=int(eta))
        target_arrivals[int(eta)] = bucket
    bucket.add(fleet)


def _attach_arrivals_to_timelines(local: LocalTurnState) -> None:
    for target_id, arrivals_by_eta in local.friendly_arrivals.items():
        timeline = local.target_timelines.get(target_id)
        if timeline is None:
            continue
        timeline.friendly_by_eta = {
            eta: bucket.ships for eta, bucket in sorted(arrivals_by_eta.items())
        }
        _update_timeline_projection(timeline, local.state.player_id)

    for target_id, arrivals_by_eta in local.enemy_arrivals.items():
        timeline = local.target_timelines.get(target_id)
        if timeline is None:
            continue
        timeline.enemy_by_eta = {
            eta: bucket.ships for eta, bucket in sorted(arrivals_by_eta.items())
        }


def _update_timeline_projection(timeline: TargetTimeline, player_id: int) -> None:
    friendly_total = sum(timeline.friendly_by_eta.values())
    if timeline.current_owner == player_id:
        timeline.projected_owner = player_id
        timeline.projected_ships = max(0, timeline.current_ships) + friendly_total
        return

    surplus = friendly_total - max(0, timeline.current_ships)
    if surplus > 0:
        timeline.projected_owner = player_id
        timeline.projected_ships = surplus
    else:
        timeline.projected_owner = timeline.current_owner
        timeline.projected_ships = -surplus
