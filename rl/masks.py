from __future__ import annotations

import math
from dataclasses import replace
from typing import Sequence

from core.actions import ActionConversionError, candidate_to_action
from core.geometry import is_sun_safe_route
from rl.local_state import LocalTurnState
from rl.types import Candidate, CandidateBatch, CandidateType


def validate_and_mask(local: LocalTurnState, candidates: Sequence[Candidate]) -> CandidateBatch:
    validated = tuple(_validated_candidate(local, candidate) for candidate in candidates)
    return CandidateBatch(validated, mask=tuple(candidate.legal for candidate in validated))


def _validated_candidate(local: LocalTurnState, candidate: Candidate) -> Candidate:
    reason = _rejection_reason(local, candidate)
    if reason is None:
        return replace(candidate, legal=True, reject_reason=None)
    return replace(candidate, legal=False, reject_reason=reason)


def _rejection_reason(local: LocalTurnState, candidate: Candidate) -> str | None:
    if candidate.type == CandidateType.STOP:
        return None

    if candidate.type == CandidateType.HOLD_SOURCE:
        return _hold_rejection_reason(local, candidate)

    if candidate.source_id is None:
        return "invalid_action"

    source_id = _int_or_none(candidate.source_id)
    if source_id is None:
        return "invalid_action"

    source = local.planets_by_id.get(source_id)
    if source is None or source.owner != local.state.player_id:
        return "invalid_source"

    ships = _int_or_none(candidate.ships)
    if ships is None or ships <= 0:
        return "invalid_ships"

    if ships > local.available_ships(source.id):
        return "insufficient_ships"

    if not _is_finite_number(candidate.angle):
        return "invalid_angle"

    if not _is_valid_eta(candidate.eta):
        return "invalid_eta"

    if candidate.target_id is None:
        return "invalid_target"

    target_id = _int_or_none(candidate.target_id)
    if target_id is None:
        return "invalid_target"

    target = local.planets_by_id.get(target_id)
    if target is None or target.id == source.id:
        return "invalid_target"

    action_reason = _action_format_reason(candidate)
    if action_reason is not None:
        return action_reason

    if not is_sun_safe_route((source.x, source.y), (target.x, target.y)):
        return "sun_collision"

    return None


def _hold_rejection_reason(local: LocalTurnState, candidate: Candidate) -> str | None:
    if candidate.source_id is None:
        return "invalid_source"
    source_id = _int_or_none(candidate.source_id)
    if source_id is None:
        return "invalid_source"
    source = local.planets_by_id.get(source_id)
    if source is None or source.owner != local.state.player_id:
        return "invalid_source"
    return None


def _action_format_reason(candidate: Candidate) -> str | None:
    try:
        candidate_to_action(candidate)
    except (ActionConversionError, TypeError, ValueError, OverflowError):
        return "invalid_action"
    return None


def _is_finite_number(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _is_valid_eta(value: object) -> bool:
    try:
        eta = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(eta) and eta >= 0


def _int_or_none(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None
