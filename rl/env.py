from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from kaggle_environments import make

from core.actions import safe_candidate_to_action, safe_noop
from core.parser import parse_observation
from rl.candidates import generate_candidates
from rl.features import (
    CANDIDATE_FEATURE_SCHEMA,
    GLOBAL_FEATURE_SCHEMA,
    encode_candidates,
    encode_global,
)
from rl.local_state import LocalTurnState, build_local_turn_state
from rl.masks import validate_and_mask
from rl.rewards import (
    RewardBreakdown,
    RewardConfig,
    RewardContext,
    calculate_reward,
    reward_config_metadata,
)
from rl.types import Candidate, CandidateBatch, CandidateType


RewardFn = Callable[[RewardContext], float]


@dataclass(slots=True)
class DecisionState:
    raw_observation: Any
    parsed: Any
    local: LocalTurnState
    batch: CandidateBatch
    global_features: tuple[float, ...]
    candidate_features: tuple[tuple[float, ...], ...]
    pending_actions: list[list[float | int]]
    pending_candidates: list[Candidate]
    pass_index: int


class OrbitWarsMaskableEnv(gym.Env[dict[str, np.ndarray], int]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        opponent: Any = "random",
        configuration: Mapping[str, Any] | None = None,
        max_candidates: int = 512,
        max_actions_per_turn: int = 2,
        debug: bool = False,
        reward_config: RewardConfig | Mapping[str, Any] | None = None,
        reward_fn: RewardFn | None = None,
    ) -> None:
        super().__init__()
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if max_actions_per_turn <= 0:
            raise ValueError("max_actions_per_turn must be positive")

        self._opponent = opponent
        self._configuration = dict(configuration or {})
        self.max_candidates = int(max_candidates)
        self.max_actions_per_turn = int(max_actions_per_turn)
        self._debug = bool(debug)
        self._reward_config = _coerce_reward_config(reward_config)
        self._reward_fn = reward_fn

        self.action_space = spaces.Discrete(self.max_candidates)
        self.observation_space = spaces.Dict(
            {
                "global_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(GLOBAL_FEATURE_SCHEMA.length,),
                    dtype=np.float32,
                ),
                "candidate_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.max_candidates, CANDIDATE_FEATURE_SCHEMA.length),
                    dtype=np.float32,
                ),
                "candidate_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.max_candidates,),
                    dtype=np.float32,
                ),
                "candidate_count": spaces.Box(
                    low=0.0,
                    high=float(self.max_candidates),
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )

        self._env = None
        self._trainer = None
        self._decision: DecisionState | None = None
        self._last_info: dict[str, Any] = {}

    @property
    def current_batch(self) -> CandidateBatch | None:
        return None if self._decision is None else self._decision.batch

    @property
    def current_pass_index(self) -> int:
        return 0 if self._decision is None else int(self._decision.pass_index)

    @property
    def pending_action_count(self) -> int:
        return 0 if self._decision is None else len(self._decision.pending_actions)

    @property
    def current_game_step(self) -> int:
        if self._decision is None:
            return 0
        return int(self._decision.parsed.step)

    def reward_config_metadata(self) -> dict[str, float | str]:
        return reward_config_metadata(self._reward_config)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        configuration = dict(self._configuration)
        if options:
            configuration.update(dict(options))
        if seed is not None:
            configuration["seed"] = int(seed)

        self._env = make("orbit_wars", configuration=configuration, debug=self._debug)
        self._trainer = self._env.train([None, self._opponent])
        raw_observation = self._trainer.reset()
        self._decision = self._build_decision_state(
            raw_observation=raw_observation,
            pending_actions=[],
            pending_candidates=[],
            pass_index=0,
        )
        observation = self._observation_from_decision(self._decision)
        info = self._base_info(turn_advanced=False, invalid_action_selected=False)
        self._last_info = info
        return observation, info

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._decision is None or self._trainer is None:
            raise RuntimeError("reset() must be called before step()")

        selected_index = self._coerce_action_index(action)
        invalid_action_selected = False
        batch = self._decision.batch
        if selected_index >= len(batch.candidates) or not batch.mask[selected_index]:
            selected_index = 0
            invalid_action_selected = True

        selected = batch.selected(selected_index)
        launch_action = safe_candidate_to_action(selected)
        selected_is_launch = bool(launch_action) and selected.type not in {
            CandidateType.STOP,
            CandidateType.HOLD_SOURCE,
        }

        next_pending_actions = [list(item) for item in self._decision.pending_actions]
        next_pending_candidates = list(self._decision.pending_candidates)
        turn_advanced = False
        reward = 0.0
        terminated = False
        truncated = False
        reward_breakdown: RewardBreakdown | None = None

        if selected_is_launch:
            next_pending_actions.append(list(launch_action))
            next_pending_candidates.append(selected)
            applied = self._decision.local.apply(selected, candidate_index=selected_index)
            if not applied:
                selected_index = 0
                selected = batch.selected(0)
                next_pending_actions.pop()
                next_pending_candidates.pop()
                invalid_action_selected = True
                selected_is_launch = False

        if selected_is_launch and (self._decision.pass_index + 1) < self.max_actions_per_turn:
            self._decision = self._rebuild_current_turn(
                pending_actions=next_pending_actions,
                pending_candidates=next_pending_candidates,
                pass_index=self._decision.pass_index + 1,
            )
            observation = self._observation_from_decision(self._decision)
            info = self._base_info(
                turn_advanced=False,
                invalid_action_selected=invalid_action_selected,
                selected_index=selected_index,
                selected_candidate=selected,
                reward_breakdown=reward_breakdown,
            )
            self._last_info = info
            return observation, reward, terminated, truncated, info

        trainer_action = next_pending_actions if next_pending_actions else safe_noop()
        previous_state = self._decision.parsed
        raw_observation, env_reward, done, trainer_info = self._trainer.step(trainer_action)
        turn_advanced = True
        self._decision = self._build_decision_state(
            raw_observation=raw_observation,
            pending_actions=[],
            pending_candidates=[],
            pass_index=0,
        )
        reward_context = RewardContext(
            previous_state=previous_state,
            current_state=self._decision.parsed,
            done=bool(done),
            submitted_candidates=tuple(next_pending_candidates),
            trainer_info=dict(trainer_info or {}),
        )
        reward_breakdown = calculate_reward(reward_context, self._reward_config)
        reward = (
            float(self._reward_fn(reward_context))
            if self._reward_fn is not None
            else float(reward_breakdown.total)
        )
        terminated, truncated = self._done_flags(bool(done), self._decision.parsed)
        observation = self._observation_from_decision(self._decision)
        info = self._base_info(
            turn_advanced=turn_advanced,
            invalid_action_selected=invalid_action_selected,
            selected_index=selected_index,
            selected_candidate=selected,
            trainer_info=dict(trainer_info or {}),
            actions_submitted=[list(item) for item in next_pending_actions],
            reward_breakdown=reward_breakdown,
        )
        info["env_reward"] = 0.0 if env_reward is None else float(env_reward)
        self._last_info = info
        return observation, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        if self._decision is None:
            return np.zeros(self.max_candidates, dtype=np.bool_)
        return self._padded_mask(self._decision.batch.mask)

    def close(self) -> None:
        self._env = None
        self._trainer = None
        self._decision = None
        self._last_info = {}

    def _build_decision_state(
        self,
        *,
        raw_observation: Any,
        pending_actions: list[list[float | int]],
        pending_candidates: list[Candidate],
        pass_index: int,
    ) -> DecisionState:
        parsed = parse_observation(raw_observation, self._env.configuration if self._env is not None else {})
        local = build_local_turn_state(parsed)
        batch, global_features, candidate_features = self._encoded_batch(local)
        return DecisionState(
            raw_observation=raw_observation,
            parsed=parsed,
            local=local,
            batch=batch,
            global_features=global_features,
            candidate_features=candidate_features,
            pending_actions=[list(item) for item in pending_actions],
            pending_candidates=list(pending_candidates),
            pass_index=int(pass_index),
        )

    def _rebuild_current_turn(
        self,
        *,
        pending_actions: list[list[float | int]],
        pending_candidates: list[Candidate],
        pass_index: int,
    ) -> DecisionState:
        if self._decision is None:
            raise RuntimeError("decision state missing")
        batch, global_features, candidate_features = self._encoded_batch(self._decision.local)
        return DecisionState(
            raw_observation=self._decision.raw_observation,
            parsed=self._decision.parsed,
            local=self._decision.local,
            batch=batch,
            global_features=global_features,
            candidate_features=candidate_features,
            pending_actions=[list(item) for item in pending_actions],
            pending_candidates=list(pending_candidates),
            pass_index=int(pass_index),
        )

    def _encoded_batch(
        self,
        local: LocalTurnState,
    ) -> tuple[CandidateBatch, tuple[float, ...], tuple[tuple[float, ...], ...]]:
        global_features = encode_global(local)
        batch = validate_and_mask(local, generate_candidates(local))
        if len(batch.candidates) > self.max_candidates:
            raise ValueError(
                f"candidate count {len(batch.candidates)} exceeds max_candidates={self.max_candidates}"
            )
        candidate_features = encode_candidates(local, batch)
        return batch, global_features, candidate_features

    def _observation_from_decision(self, decision: DecisionState) -> dict[str, np.ndarray]:
        candidate_rows = np.zeros(
            (self.max_candidates, CANDIDATE_FEATURE_SCHEMA.length),
            dtype=np.float32,
        )
        for index, row in enumerate(decision.candidate_features):
            candidate_rows[index] = np.asarray(row, dtype=np.float32)

        mask = self._padded_mask(decision.batch.mask)
        return {
            "global_features": np.asarray(decision.global_features, dtype=np.float32),
            "candidate_features": candidate_rows,
            "candidate_mask": mask.astype(np.float32),
            "candidate_count": np.asarray([len(decision.batch.candidates)], dtype=np.float32),
        }

    def _padded_mask(self, mask: Sequence[bool]) -> np.ndarray:
        padded = np.zeros(self.max_candidates, dtype=np.bool_)
        padded[: len(mask)] = np.asarray(mask, dtype=np.bool_)
        return padded

    def _done_flags(self, done: bool, parsed: Any) -> tuple[bool, bool]:
        if not done:
            return False, False
        episode_steps = int(parsed.config.get("episodeSteps", 500) or 500)
        truncated = int(parsed.step) >= max(0, episode_steps - 1)
        return (not truncated), truncated

    def _coerce_action_index(self, action: Any) -> int:
        try:
            return int(action)
        except (TypeError, ValueError, OverflowError):
            return self.max_candidates

    def _base_info(
        self,
        *,
        turn_advanced: bool,
        invalid_action_selected: bool,
        selected_index: int | None = None,
        selected_candidate: Candidate | None = None,
        trainer_info: Mapping[str, Any] | None = None,
        actions_submitted: Sequence[Sequence[float | int]] | None = None,
        reward_breakdown: RewardBreakdown | None = None,
    ) -> dict[str, Any]:
        if self._decision is None:
            return {}
        mask = self._decision.batch.mask
        info = {
            "step": int(self._decision.parsed.step),
            "pass_index": int(self._decision.pass_index),
            "candidate_count": len(self._decision.batch.candidates),
            "legal_count": sum(1 for value in mask if value),
            "turn_advanced": bool(turn_advanced),
            "invalid_action_selected": bool(invalid_action_selected),
            "pending_action_count": len(self._decision.pending_actions),
            "reward_config": self.reward_config_metadata(),
        }
        if selected_index is not None:
            info["selected_index"] = int(selected_index)
        if selected_candidate is not None:
            info["selected_type"] = str(selected_candidate.type)
        if trainer_info is not None:
            info["trainer_info"] = dict(trainer_info)
        if actions_submitted is not None:
            info["actions_submitted"] = [list(action) for action in actions_submitted]
        if reward_breakdown is not None:
            info["reward_breakdown"] = reward_breakdown.metadata()
        return info


def _coerce_reward_config(value: RewardConfig | Mapping[str, Any] | None) -> RewardConfig:
    if value is None:
        return RewardConfig()
    if isinstance(value, RewardConfig):
        return value
    return RewardConfig(**dict(value))
