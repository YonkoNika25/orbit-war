from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Sequence


class PolicySelectionError(ValueError):
    """Raised when policy inputs cannot produce an aligned candidate index."""


@dataclass(slots=True)
class RandomMaskedPolicy:
    rng: random.Random | None = None
    stop_index: int = 0

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = random.Random()

    def select_index(
        self,
        global_features: Sequence[float],
        candidate_features: Sequence[Sequence[float]],
        mask: Iterable[bool],
    ) -> int:
        _ = global_features
        legal_mask = tuple(bool(value) for value in mask)
        if len(candidate_features) != len(legal_mask):
            raise PolicySelectionError("candidate feature rows must match mask length")

        legal_indexes = tuple(
            index for index, is_legal in enumerate(legal_mask) if is_legal
        )
        if legal_indexes:
            assert self.rng is not None
            return self.rng.choice(legal_indexes)

        if self.stop_index < 0 or self.stop_index >= len(legal_mask):
            raise PolicySelectionError("stop_index must be within mask range")
        return self.stop_index
