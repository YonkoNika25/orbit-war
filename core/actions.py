from __future__ import annotations

import math
from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Mapping


KaggleAction = List[float | int]


class ActionConversionError(ValueError):
    """Raised when an action cannot be converted without emitting invalid output."""


def safe_noop() -> KaggleAction:
    return []


def to_kaggle_action(source_id: Any, angle: Any, ships: Any) -> KaggleAction:
    source = _coerce_source_id(source_id)
    ship_count = _coerce_ships(ships)
    launch_angle = _coerce_angle(angle)
    return [source, launch_angle, ship_count]


def candidate_to_action(candidate: Any) -> KaggleAction:
    if _is_noop_candidate(candidate):
        return safe_noop()

    payload = _candidate_mapping(candidate)
    return to_kaggle_action(
        source_id=_first_value(payload, ("source_id", "source", "from_planet_id", "planet_id")),
        angle=_first_value(payload, ("angle", "launch_angle", "heading")),
        ships=_first_value(payload, ("ships", "ship_count", "units")),
    )


def safe_candidate_to_action(candidate: Any) -> KaggleAction:
    try:
        return candidate_to_action(candidate)
    except ActionConversionError:
        return safe_noop()


def _is_noop_candidate(candidate: Any) -> bool:
    if candidate is None:
        return True
    if candidate == []:
        return True

    payload = _candidate_mapping(candidate)
    marker = payload.get("kind", payload.get("type", payload.get("action_type")))
    if marker is None:
        return False
    return str(marker).upper() in {"STOP", "NOOP", "NO_OP", "HOLD", "HOLD_SOURCE"}


def _candidate_mapping(candidate: Any) -> Dict[str, Any]:
    if candidate is None:
        return {}
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
    raise ActionConversionError(f"Unsupported candidate type: {type(candidate).__name__}")


def _iter_slot_names(value: Any) -> List[str]:
    names: List[str] = []
    for cls in type(value).__mro__:
        slots = getattr(cls, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name not in {"__dict__", "__weakref__"}:
                names.append(name)
    return names


def _first_value(payload: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _coerce_source_id(value: Any) -> int:
    try:
        source_id = int(value)
    except (TypeError, ValueError):
        raise ActionConversionError("source_id must be an integer") from None
    if source_id < 0:
        raise ActionConversionError("source_id must be non-negative")
    return source_id


def _coerce_ships(value: Any) -> int:
    try:
        ships = int(value)
    except (TypeError, ValueError):
        raise ActionConversionError("ships must be an integer") from None
    if ships <= 0:
        raise ActionConversionError("ships must be positive")
    return ships


def _coerce_angle(value: Any) -> float:
    try:
        angle = float(value)
    except (TypeError, ValueError):
        raise ActionConversionError("angle must be numeric") from None
    if not math.isfinite(angle):
        raise ActionConversionError("angle must be finite")
    return angle
