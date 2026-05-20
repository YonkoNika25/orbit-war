from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


_STATE: Dict[str, Any] = {
    "counters": {},
    "steps": [],
}


def reset() -> None:
    _STATE["counters"] = {}
    _STATE["steps"] = []


def increment(name: str, amount: int = 1) -> None:
    counters = _STATE["counters"]
    counters[name] = int(counters.get(name, 0)) + int(amount)


def record_step(step: int, player_id: int, payload: Dict[str, Any]) -> None:
    entry = {
        "step": int(step),
        "player_id": int(player_id),
        **deepcopy(payload),
    }

    steps = _STATE["steps"]
    for index, existing in enumerate(steps):
        if existing.get("step") == entry["step"] and existing.get("player_id") == entry["player_id"]:
            steps[index] = entry
            break
    else:
        steps.append(entry)


def snapshot() -> Dict[str, Any]:
    return deepcopy(_STATE)
