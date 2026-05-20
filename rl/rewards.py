from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from core.types import GameState, Planet
from rl.types import Candidate, CandidateType


REWARD_SCHEMA_VERSION = "4.2.0"


@dataclass(frozen=True, slots=True)
class RewardConfig:
    terminal_win: float = 1.0
    terminal_loss: float = -1.0
    terminal_draw: float = 0.0
    production_swing_weight: float = 0.02
    planet_capture_weight: float = 0.05
    planet_loss_weight: float = 0.05
    sun_waste_weight: float = 0.0
    overkill_weight: float = 0.01
    shaping_clip: float = 0.25


@dataclass(frozen=True, slots=True)
class RewardContext:
    previous_state: GameState | None
    current_state: GameState
    done: bool
    submitted_candidates: tuple[Candidate, ...] = ()
    trainer_info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    total: float
    terminal: float
    shaping: float
    production_swing: float
    planets_captured: float
    planets_lost: float
    sun_waste: float
    overkill_waste: float

    def metadata(self) -> dict[str, float]:
        return {
            "total": float(self.total),
            "terminal": float(self.terminal),
            "shaping": float(self.shaping),
            "production_swing": float(self.production_swing),
            "planets_captured": float(self.planets_captured),
            "planets_lost": float(self.planets_lost),
            "sun_waste": float(self.sun_waste),
            "overkill_waste": float(self.overkill_waste),
        }


def reward_config_metadata(config: RewardConfig) -> dict[str, float | str]:
    metadata = asdict(config)
    metadata["schema_version"] = REWARD_SCHEMA_VERSION
    return metadata


def calculate_reward(
    context: RewardContext,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    reward_config = config or RewardConfig()
    terminal = _terminal_reward(context.current_state, reward_config) if context.done else 0.0

    production_swing = 0.0
    planets_captured = 0.0
    planets_lost = 0.0
    sun_waste = 0.0
    overkill_waste = 0.0

    previous_state = context.previous_state
    if previous_state is not None:
        production_swing = (
            _production_advantage(context.current_state) - _production_advantage(previous_state)
        ) * reward_config.production_swing_weight
        capture_count, lost_count = _planet_ownership_swings(previous_state, context.current_state)
        planets_captured = capture_count * reward_config.planet_capture_weight
        planets_lost = -lost_count * reward_config.planet_loss_weight
        sun_waste = -_sun_waste_signal(context.trainer_info) * reward_config.sun_waste_weight
        overkill_waste = -_overkill_signal(previous_state, context.submitted_candidates) * reward_config.overkill_weight

    shaping_raw = production_swing + planets_captured + planets_lost + sun_waste + overkill_waste
    shaping = _clip_shaping(shaping_raw, reward_config.shaping_clip)
    total = terminal + shaping
    return RewardBreakdown(
        total=float(total),
        terminal=float(terminal),
        shaping=float(shaping),
        production_swing=float(production_swing),
        planets_captured=float(planets_captured),
        planets_lost=float(planets_lost),
        sun_waste=float(sun_waste),
        overkill_waste=float(overkill_waste),
    )


def _terminal_reward(state: GameState, config: RewardConfig) -> float:
    scores = _player_scores(state)
    player_id = int(state.player_id)
    my_score = scores.get(player_id, 0.0)
    if not scores:
        return config.terminal_draw

    max_score = max(scores.values())
    winners = [owner for owner, score in scores.items() if score == max_score]
    if my_score == max_score:
        if len(winners) == 1 and max_score > 0.0:
            return config.terminal_win
        return config.terminal_draw
    return config.terminal_loss


def _player_scores(state: GameState) -> dict[int, float]:
    scores: dict[int, float] = {}
    for planet in state.planets:
        if planet.owner == -1:
            continue
        scores[planet.owner] = scores.get(planet.owner, 0.0) + max(0.0, float(planet.ships))
    for fleet in state.fleets:
        if fleet.owner == -1:
            continue
        scores[fleet.owner] = scores.get(fleet.owner, 0.0) + max(0.0, float(fleet.ships))
    return scores


def _production_advantage(state: GameState) -> float:
    player_id = int(state.player_id)
    my_production = 0.0
    enemy_production = 0.0
    for planet in state.planets:
        production = max(0.0, float(planet.production))
        if planet.owner == player_id:
            my_production += production
        elif planet.owner != -1:
            enemy_production += production
    return my_production - enemy_production


def _planet_ownership_swings(
    previous_state: GameState,
    current_state: GameState,
) -> tuple[int, int]:
    player_id = int(current_state.player_id)
    previous_by_id = {int(planet.id): planet for planet in previous_state.planets}
    captured = 0
    lost = 0
    for planet in current_state.planets:
        previous = previous_by_id.get(int(planet.id))
        if previous is None:
            continue
        if previous.owner != player_id and planet.owner == player_id:
            captured += 1
        elif previous.owner == player_id and planet.owner != player_id:
            lost += 1
    return captured, lost


def _sun_waste_signal(trainer_info: Mapping[str, Any]) -> float:
    for key in ("sun_waste_ships", "sun_loss_ships", "sun_losses"):
        value = trainer_info.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric > 0.0:
            return numeric
    return 0.0


def _overkill_signal(
    previous_state: GameState,
    submitted_candidates: Sequence[Candidate],
) -> float:
    if not submitted_candidates:
        return 0.0

    planets_by_id = {int(planet.id): planet for planet in previous_state.planets}
    player_id = int(previous_state.player_id)
    total = 0.0
    for candidate in submitted_candidates:
        if candidate.type in {CandidateType.STOP, CandidateType.HOLD_SOURCE, CandidateType.REINFORCE, CandidateType.DEFEND}:
            continue
        if candidate.target_id is None:
            continue
        target = planets_by_id.get(int(candidate.target_id))
        if target is None or target.owner == player_id:
            continue
        minimum_useful = max(1, int(target.ships) + 1)
        if int(candidate.ships) > minimum_useful:
            total += float(int(candidate.ships) - minimum_useful)
    return total


def _clip_shaping(value: float, clip_limit: float) -> float:
    limit = max(0.0, float(clip_limit))
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value
