from __future__ import annotations

from typing import List, Tuple

from core.geometry import angle_to, estimate_eta
from core.types import Planet
from rl.local_state import LocalTurnState
from rl.types import Candidate, CandidateType


def generate_candidates(local: LocalTurnState) -> Tuple[Candidate, ...]:
    candidates: List[Candidate] = [Candidate.stop()]

    for source_id in local.owned_planet_ids:
        source = local.planets_by_id[source_id]
        available = local.available_ships(source_id)
        if available <= 0:
            continue

        candidates.append(
            Candidate(
                CandidateType.HOLD_SOURCE,
                source_id=source.id,
                target_id=None,
                ships=0,
                angle=0.0,
                eta=0,
                estimated_owner=source.owner,
                estimated_ships=available,
                score=float(available),
            )
        )

        for target_id in local.enemy_planet_ids:
            timeline = local.target_timelines[target_id]
            if timeline.projected_owner == local.state.player_id:
                continue
            target = local.planets_by_id[target_id]
            candidates.append(_attack_candidate(source, target, available, local.state.player_id))
            candidates.append(_harass_candidate(source, target, available, local.state.player_id))

        for target_id in local.neutral_planet_ids:
            target = local.planets_by_id[target_id]
            candidates.append(_expand_candidate(source, target, available, local.state.player_id))

        for target_id in local.owned_planet_ids:
            if target_id == source_id:
                continue
            target = local.planets_by_id[target_id]
            timeline = local.target_timelines[target_id]
            if timeline.enemy_by_eta:
                candidates.append(_defend_candidate(source, target, available, local.state.player_id))
            else:
                candidates.append(_reinforce_candidate(source, target, available, local.state.player_id))

    return tuple(candidates)


def _attack_candidate(source: Planet, target: Planet, available: int, player_id: int) -> Candidate:
    ships = max(1, min(available, int(target.ships) + 1))
    estimated_owner, estimated_ships = _estimate_capture(target, ships, player_id)
    return _launch_candidate(
        CandidateType.ATTACK,
        source,
        target,
        ships,
        estimated_owner=estimated_owner,
        estimated_ships=estimated_ships,
        score=float(ships - target.ships),
    )


def _expand_candidate(source: Planet, target: Planet, available: int, player_id: int) -> Candidate:
    ships = max(1, min(available, int(target.ships) + 1))
    estimated_owner, estimated_ships = _estimate_capture(target, ships, player_id)
    return _launch_candidate(
        CandidateType.EXPAND_NEUTRAL,
        source,
        target,
        ships,
        estimated_owner=estimated_owner,
        estimated_ships=estimated_ships,
        score=float(target.production),
    )


def _harass_candidate(source: Planet, target: Planet, available: int, player_id: int) -> Candidate:
    probe = max(1, int(target.ships) // 4 or 1)
    ships = max(1, min(available, probe))
    estimated_owner, estimated_ships = _estimate_capture(target, ships, player_id)
    return _launch_candidate(
        CandidateType.HARASS,
        source,
        target,
        ships,
        estimated_owner=estimated_owner,
        estimated_ships=estimated_ships,
        score=float(-ships),
    )


def _reinforce_candidate(source: Planet, target: Planet, available: int, player_id: int) -> Candidate:
    ships = max(1, available // 2)
    return _launch_candidate(
        CandidateType.REINFORCE,
        source,
        target,
        ships,
        estimated_owner=player_id,
        estimated_ships=int(target.ships) + ships,
        score=float(ships),
    )


def _defend_candidate(source: Planet, target: Planet, available: int, player_id: int) -> Candidate:
    ships = max(1, available // 2)
    return _launch_candidate(
        CandidateType.DEFEND,
        source,
        target,
        ships,
        estimated_owner=player_id,
        estimated_ships=int(target.ships) + ships,
        score=float(ships + target.production),
    )


def _launch_candidate(
    candidate_type: CandidateType,
    source: Planet,
    target: Planet,
    ships: int,
    *,
    estimated_owner: int,
    estimated_ships: int,
    score: float,
) -> Candidate:
    return Candidate(
        candidate_type,
        source_id=source.id,
        target_id=target.id,
        ships=ships,
        angle=angle_to((source.x, source.y), (target.x, target.y)),
        eta=estimate_eta(source, target),
        estimated_owner=estimated_owner,
        estimated_ships=estimated_ships,
        score=score,
    )


def _estimate_capture(target: Planet, ships: int, player_id: int) -> tuple[int, int]:
    surplus = ships - int(target.ships)
    if surplus > 0:
        return player_id, surplus
    return int(target.owner), -surplus
