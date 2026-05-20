from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Sequence, Tuple


class CandidateType(str, Enum):
    STOP = "STOP"
    ATTACK = "ATTACK"
    EXPAND_NEUTRAL = "EXPAND_NEUTRAL"
    REINFORCE = "REINFORCE"
    DEFEND = "DEFEND"
    HARASS = "HARASS"
    HOLD_SOURCE = "HOLD_SOURCE"

    def __str__(self) -> str:
        return self.value


@dataclass(slots=True)
class Candidate:
    type: CandidateType
    source_id: Optional[int] = None
    target_id: Optional[int] = None
    ships: int = 0
    angle: float = 0.0
    eta: int = 0
    legal: bool = True
    reject_reason: Optional[str] = None
    estimated_owner: Optional[int] = None
    estimated_ships: Optional[int] = None
    score: Optional[float] = None

    @classmethod
    def stop(cls) -> Candidate:
        return cls(CandidateType.STOP)


@dataclass(slots=True)
class CandidateBatch:
    candidates: Tuple[Candidate, ...]
    mask: Tuple[bool, ...]

    def __init__(self, candidates: Sequence[Candidate], mask: Iterable[bool]):
        self.candidates = tuple(candidates)
        self.mask = tuple(bool(value) for value in mask)
        if len(self.candidates) != len(self.mask):
            raise ValueError("candidate mask length must match candidate count")

    def selected(self, index: int) -> Candidate:
        if index < 0 or index >= len(self.candidates):
            raise IndexError("selected candidate index out of range")
        return self.candidates[index]

    def legal_candidates(self) -> Tuple[Candidate, ...]:
        return tuple(
            candidate
            for candidate, is_legal in zip(self.candidates, self.mask)
            if is_legal
        )
