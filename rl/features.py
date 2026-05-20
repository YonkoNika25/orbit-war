from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Tuple

from core.geometry import distance as planet_distance, is_sun_safe_route
from rl.local_state import LocalTurnState
from rl.types import Candidate, CandidateBatch, CandidateType


FEATURE_SCHEMA_VERSION = "3.1.0"

_SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


class FeatureSchemaError(ValueError):
    """Raised when feature schema metadata or vector shape is invalid."""


@dataclass(frozen=True, slots=True)
class FeatureField:
    name: str


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    name: str
    version: str
    fields: Tuple[FeatureField, ...]

    def __init__(self, name: str, version: str, fields: Iterable[FeatureField]):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "fields", tuple(fields))
        self._validate()

    @property
    def keys(self) -> Tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    @property
    def length(self) -> int:
        return len(self.fields)

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "length": self.length,
            "keys": self.keys,
            "fingerprint": schema_fingerprint(self),
        }

    def validate_vector(self, values: Iterable[float]) -> Tuple[float, ...]:
        vector = tuple(values)
        if len(vector) != self.length:
            raise FeatureSchemaError(
                f"{self.name} feature vector length {len(vector)} does not match schema length {self.length}"
            )
        return vector

    def _validate(self) -> None:
        seen: set[str] = set()
        for key in self.keys:
            if not _SNAKE_CASE_RE.fullmatch(key):
                raise FeatureSchemaError(f"feature key must use snake_case: {key!r}")
            if key in seen:
                raise FeatureSchemaError(f"duplicate feature key: {key!r}")
            seen.add(key)


def schema_fingerprint(schema: FeatureSchema) -> str:
    payload = {
        "name": schema.name,
        "version": schema.version,
        "keys": schema.keys,
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


GLOBAL_FEATURE_SCHEMA = FeatureSchema(
    name="global",
    version=FEATURE_SCHEMA_VERSION,
    fields=(
        FeatureField("step_norm"),
        FeatureField("player_id"),
        FeatureField("owned_planets"),
        FeatureField("enemy_planets"),
        FeatureField("neutral_planets"),
        FeatureField("owned_production"),
        FeatureField("enemy_production"),
        FeatureField("neutral_production"),
        FeatureField("owned_ships_planets"),
        FeatureField("enemy_ships_planets"),
        FeatureField("neutral_ships_planets"),
        FeatureField("owned_ships_fleets"),
        FeatureField("enemy_ships_fleets"),
        FeatureField("score_advantage"),
        FeatureField("phase_early"),
        FeatureField("phase_mid"),
        FeatureField("phase_late"),
    ),
)


CANDIDATE_FEATURE_SCHEMA = FeatureSchema(
    name="candidate",
    version=FEATURE_SCHEMA_VERSION,
    fields=(
        FeatureField("type_stop"),
        FeatureField("type_attack"),
        FeatureField("type_expand_neutral"),
        FeatureField("type_reinforce"),
        FeatureField("type_defend"),
        FeatureField("type_harass"),
        FeatureField("type_hold_source"),
        FeatureField("source_ships_available"),
        FeatureField("ships_sent"),
        FeatureField("target_owner"),
        FeatureField("target_ships"),
        FeatureField("target_production"),
        FeatureField("eta"),
        FeatureField("distance"),
        FeatureField("projected_owner"),
        FeatureField("projected_ships"),
        FeatureField("overkill"),
        FeatureField("underkill"),
        FeatureField("reserve"),
        FeatureField("sun_safe"),
        FeatureField("friendly_arrivals_committed"),
    ),
)


def encode_global(local: LocalTurnState) -> Tuple[float, ...]:
    player_id = int(local.state.player_id)
    step_norm = _normalized_step(local.state.step, local.state.config)

    owned_planets = 0.0
    enemy_planets = 0.0
    neutral_planets = 0.0
    owned_production = 0.0
    enemy_production = 0.0
    neutral_production = 0.0
    owned_ships_planets = 0.0
    enemy_ships_planets = 0.0
    neutral_ships_planets = 0.0

    for planet in local.planets:
        ships = _non_negative_float(planet.ships)
        production = _non_negative_float(planet.production)
        if planet.owner == player_id:
            owned_planets += 1.0
            owned_production += production
            owned_ships_planets += ships
        elif planet.owner == -1:
            neutral_planets += 1.0
            neutral_production += production
            neutral_ships_planets += ships
        else:
            enemy_planets += 1.0
            enemy_production += production
            enemy_ships_planets += ships

    owned_ships_fleets = 0.0
    enemy_ships_fleets = 0.0
    for fleet in local.fleets:
        ships = _non_negative_float(fleet.ships)
        if fleet.owner == player_id:
            owned_ships_fleets += ships
        elif fleet.owner != -1:
            enemy_ships_fleets += ships

    owned_strength = owned_ships_planets + owned_ships_fleets + owned_production
    enemy_strength = enemy_ships_planets + enemy_ships_fleets + enemy_production
    score_advantage = _relative_advantage(owned_strength, enemy_strength)
    phase_early, phase_mid, phase_late = _phase_features(step_norm)

    return GLOBAL_FEATURE_SCHEMA.validate_vector(
        (
            step_norm,
            float(player_id),
            owned_planets,
            enemy_planets,
            neutral_planets,
            owned_production,
            enemy_production,
            neutral_production,
            owned_ships_planets,
            enemy_ships_planets,
            neutral_ships_planets,
            owned_ships_fleets,
            enemy_ships_fleets,
            score_advantage,
            phase_early,
            phase_mid,
            phase_late,
        )
    )


def encode_candidates(
    local: LocalTurnState,
    batch: CandidateBatch,
) -> Tuple[Tuple[float, ...], ...]:
    return tuple(_encode_candidate(local, candidate) for candidate in batch.candidates)


def _encode_candidate(
    local: LocalTurnState,
    candidate: Candidate,
) -> Tuple[float, ...]:
    source = local.planets_by_id.get(int(candidate.source_id)) if _has_int(candidate.source_id) else None
    target = local.planets_by_id.get(int(candidate.target_id)) if _has_int(candidate.target_id) else None
    timeline = local.target_timelines.get(target.id) if target is not None else None

    source_available = float(local.available_ships(source.id)) if source is not None else 0.0
    ships_sent = _non_negative_float(candidate.ships)
    target_owner = float(target.owner) if target is not None else 0.0
    target_ships = _non_negative_float(target.ships) if target is not None else 0.0
    target_production = _non_negative_float(target.production) if target is not None else 0.0
    eta = _non_negative_float(candidate.eta)
    route_distance = planet_distance(source, target) if source is not None and target is not None else 0.0
    projected_owner = _projected_owner(candidate, timeline)
    projected_ships = _projected_ships(candidate, timeline)
    friendly_arrivals = (
        _non_negative_float(sum(timeline.friendly_by_eta.values()))
        if timeline is not None
        else 0.0
    )
    sun_safe = _sun_safe_feature(source, target)

    return CANDIDATE_FEATURE_SCHEMA.validate_vector(
        (
            *_candidate_type_features(candidate),
            source_available,
            ships_sent,
            target_owner,
            target_ships,
            target_production,
            eta,
            route_distance,
            projected_owner,
            projected_ships,
            max(0.0, ships_sent - target_ships),
            max(0.0, target_ships - ships_sent),
            max(0.0, source_available - ships_sent),
            sun_safe,
            friendly_arrivals,
        )
    )


def _candidate_type_features(candidate: Candidate) -> tuple[float, ...]:
    candidate_type = candidate.type
    return (
        1.0 if candidate_type == CandidateType.STOP else 0.0,
        1.0 if candidate_type == CandidateType.ATTACK else 0.0,
        1.0 if candidate_type == CandidateType.EXPAND_NEUTRAL else 0.0,
        1.0 if candidate_type == CandidateType.REINFORCE else 0.0,
        1.0 if candidate_type == CandidateType.DEFEND else 0.0,
        1.0 if candidate_type == CandidateType.HARASS else 0.0,
        1.0 if candidate_type == CandidateType.HOLD_SOURCE else 0.0,
    )


def _projected_owner(candidate: Candidate, timeline: object) -> float:
    if candidate.estimated_owner is not None:
        return _finite_float(candidate.estimated_owner)
    if timeline is not None:
        return _finite_float(timeline.projected_owner)
    return 0.0


def _projected_ships(candidate: Candidate, timeline: object) -> float:
    if candidate.estimated_ships is not None:
        return _non_negative_float(candidate.estimated_ships)
    if timeline is not None:
        return _non_negative_float(timeline.projected_ships)
    return 0.0


def _sun_safe_feature(source: object, target: object) -> float:
    if source is None or target is None:
        return 1.0
    return 1.0 if is_sun_safe_route((source.x, source.y), (target.x, target.y)) else 0.0


def _normalized_step(step: Any, config: Mapping[str, Any]) -> float:
    episode_steps = _first_positive_float(
        config,
        ("episodeSteps", "episode_steps", "max_steps", "maxEpisodeSteps"),
        default=500.0,
    )
    try:
        raw_step = float(step)
    except (TypeError, ValueError):
        raw_step = 0.0
    if not math.isfinite(raw_step):
        raw_step = 0.0
    return min(1.0, max(0.0, raw_step / episode_steps))


def _first_positive_float(
    values: Mapping[str, Any],
    keys: Tuple[str, ...],
    default: float,
) -> float:
    for key in keys:
        try:
            value = float(values[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            return value
    return default


def _non_negative_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(0.0, number)


def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return number


def _has_int(value: Any) -> bool:
    try:
        int(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def _relative_advantage(owned_strength: float, enemy_strength: float) -> float:
    total = owned_strength + enemy_strength
    if total <= 0.0:
        return 0.0
    return (owned_strength - enemy_strength) / total


def _phase_features(step_norm: float) -> tuple[float, float, float]:
    if step_norm < (1.0 / 3.0):
        return 1.0, 0.0, 0.0
    if step_norm < (2.0 / 3.0):
        return 0.0, 1.0, 0.0
    return 0.0, 0.0, 1.0
