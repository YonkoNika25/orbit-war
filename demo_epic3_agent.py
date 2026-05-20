from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List

from core.actions import safe_candidate_to_action, safe_noop
from core.telemetry import increment, record_step
from core.parser import parse_observation
from rl.candidates import generate_candidates
from rl.features import CANDIDATE_FEATURE_SCHEMA, GLOBAL_FEATURE_SCHEMA, encode_candidates, encode_global
from rl.local_state import build_local_turn_state
from rl.masks import validate_and_mask
from rl.policy import RandomMaskedPolicy
from rl.types import Candidate, CandidateType


_POLICY = RandomMaskedPolicy(rng=random.Random(7), stop_index=0)
_MAX_ACTIONS_PER_TURN = 2





def _candidate_payload(
    candidate: Candidate,
    vector: Iterable[float],
    index: int,
    local,
) -> Dict[str, Any]:
    source = local.planets_by_id.get(int(candidate.source_id)) if candidate.source_id is not None else None
    target = local.planets_by_id.get(int(candidate.target_id)) if candidate.target_id is not None else None
    features = _feature_map(CANDIDATE_FEATURE_SCHEMA.keys, vector)
    return {
        "index": int(index),
        "type": str(candidate.type),
        "legal": bool(candidate.legal),
        "reject_reason": candidate.reject_reason,
        "source_id": candidate.source_id,
        "target_id": candidate.target_id,
        "ships": int(candidate.ships),
        "eta": int(candidate.eta),
        "angle": float(candidate.angle),
        "score": candidate.score,
        "estimated_owner": candidate.estimated_owner,
        "estimated_ships": candidate.estimated_ships,
        "source": _planet_payload(source),
        "target": _planet_payload(target),
        "features": features,
        "label": _candidate_label(candidate),
    }


def _planet_payload(planet: Any) -> Dict[str, Any] | None:
    if planet is None:
        return None
    return {
        "id": int(planet.id),
        "owner": int(planet.owner),
        "x": float(planet.x),
        "y": float(planet.y),
        "ships": int(planet.ships),
        "production": int(planet.production),
        "radius": float(planet.radius),
        "orbit_center_x": float(planet.orbit_center_x),
        "orbit_center_y": float(planet.orbit_center_y),
        "orbit_radius": float(planet.orbit_radius),
        "orbit_angle": float(planet.orbit_angle),
        "orbit_speed": float(planet.orbit_speed),
        "is_moving": bool(planet.extras.get("is_moving")),
    }


def _feature_map(keys: Iterable[str], vector: Iterable[float]) -> Dict[str, float]:
    return {
        str(key): float(value)
        for key, value in zip(keys, vector)
    }


def _candidate_label(candidate: Candidate) -> str:
    if candidate.type == CandidateType.STOP:
        return "STOP"
    if candidate.type == CandidateType.HOLD_SOURCE:
        return f"HOLD {candidate.source_id}"
    return f"{candidate.type} {candidate.source_id}->{candidate.target_id} ships={candidate.ships} eta={candidate.eta}"


def agent(obs: Any, config: Any) -> List[List[float | int]]:
    parsed = parse_observation(obs, config)
    local = build_local_turn_state(parsed)
    global_vector = encode_global(local)
    my_fleets = [fleet for fleet in parsed.fleets if fleet.owner == parsed.player_id]
    enemy_fleets = [
        fleet for fleet in parsed.fleets if fleet.owner not in (-1, parsed.player_id)
    ]

    telemetry: Dict[str, Any] = {
        "step": int(parsed.step),
        "player_id": int(parsed.player_id),
        "summary": {
            "my_planets": len(parsed.my_planets),
            "enemy_planets": len(parsed.enemy_planets),
            "neutral_planets": len(parsed.neutral_planets),
            "my_fleets": len(my_fleets),
            "enemy_fleets": len(enemy_fleets),
            "remaining_overage_time": parsed.extras.get("remainingOverageTime"),
        },
        "global_features": _feature_map(GLOBAL_FEATURE_SCHEMA.keys, global_vector),
        "passes": [],
        "final_actions": [],
    }

    actions: List[List[float | int]] = []
    for pass_index in range(_MAX_ACTIONS_PER_TURN):
        candidates = generate_candidates(local)
        batch = validate_and_mask(local, candidates)
        candidate_vectors = encode_candidates(local, batch)
        selected_index = _POLICY.select_index(global_vector, candidate_vectors, batch.mask)
        selected = batch.selected(selected_index)
        action = safe_candidate_to_action(selected)
        selected_is_launch = bool(action) and selected.type not in {CandidateType.STOP, CandidateType.HOLD_SOURCE}

        telemetry["passes"].append(
            {
                "pass_index": pass_index,
                "candidate_count": len(batch.candidates),
                "legal_count": sum(1 for legal in batch.mask if legal),
                "source_commitments_before": dict(local.source_commitments),
                "selected_index": selected_index,
                "selected_candidate": _candidate_payload(selected, candidate_vectors[selected_index], selected_index, local),
                "selected_action": list(action),
                "applied": False,
                "candidates": [
                    _candidate_payload(candidate, vector, index, local)
                    for index, (candidate, vector) in enumerate(zip(batch.candidates, candidate_vectors))
                ],
            }
        )

        if not selected_is_launch:
            break

        actions.append(action)
        increment("launches")
        increment("ships_launched", int(selected.ships))
        applied = local.apply(selected, candidate_index=selected_index)
        telemetry["passes"][-1]["applied"] = bool(applied)
        telemetry["passes"][-1]["source_commitments_after"] = dict(local.source_commitments)
        if not applied:
            break

    telemetry["final_actions"] = [list(action) for action in actions]
    increment("turns")
    record_step(parsed.step, parsed.player_id, telemetry)
    return actions if actions else safe_noop()
