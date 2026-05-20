from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import gymnasium as gym

from rl.checkpoint import build_checkpoint_id, checkpoint_artifact_dir, save_checkpoint_artifact
from rl.env import OrbitWarsMaskableEnv
from rl.features import CANDIDATE_FEATURE_SCHEMA, GLOBAL_FEATURE_SCHEMA
from rl.rewards import RewardConfig, reward_config_metadata

try:
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker

    TRAINING_STACK_AVAILABLE = True
    TRAINING_STACK_ERROR = None
except ModuleNotFoundError as exc:
    Monitor = None
    DummyVecEnv = None
    MaskablePPO = None
    ActionMasker = None
    TRAINING_STACK_AVAILABLE = False
    TRAINING_STACK_ERROR = exc


PROJECT_ROOT = Path(__file__).resolve().parent.parent
HEURISTIC_BASELINE_PATH = PROJECT_ROOT / "extracted_orbit-wars-1000_8af09943.py"


@dataclass(frozen=True, slots=True)
class TrainConfig:
    total_timesteps: int = 10_000
    opponents: tuple[str, ...] = ("random",)
    run_name: str | None = None
    output_root: str = "artifacts/train_runs"
    model_seed: int = 7
    env_seed: int | None = None
    max_candidates: int = 512
    max_actions_per_turn: int = 2
    policy: str = "MultiInputPolicy"
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    gamma: float = 0.99
    verbose: int = 1
    dry_run: bool = False
    env_debug: bool = False
    reward_config: RewardConfig = field(default_factory=RewardConfig)


class OpponentPoolEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        opponents: Sequence[str],
        env_kwargs: Mapping[str, Any],
        env_seed: int | None = None,
    ) -> None:
        if not opponents:
            raise ValueError("opponents must not be empty")
        self._opponent_keys = tuple(str(item) for item in opponents)
        self._env_kwargs = dict(env_kwargs)
        self._env_seed = env_seed
        self._episode_index = 0
        self._current_opponent_key = self._opponent_keys[0]
        self._child = OrbitWarsMaskableEnv(
            opponent=resolve_opponent_reference(self._current_opponent_key),
            **self._env_kwargs,
        )
        self.action_space = self._child.action_space
        self.observation_space = self._child.observation_space

    @property
    def current_opponent_key(self) -> str:
        return self._current_opponent_key

    def reset(self, *, seed: int | None = None, options: Mapping[str, Any] | None = None):
        self._child.close()
        self._current_opponent_key = self._opponent_keys[self._episode_index % len(self._opponent_keys)]
        self._episode_index += 1
        self._child = OrbitWarsMaskableEnv(
            opponent=resolve_opponent_reference(self._current_opponent_key),
            **self._env_kwargs,
        )
        effective_seed = seed
        if effective_seed is None and self._env_seed is not None:
            effective_seed = int(self._env_seed) + (self._episode_index - 1)
        observation, info = self._child.reset(seed=effective_seed, options=options)
        info = dict(info)
        info["opponent_key"] = self._current_opponent_key
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self._child.step(action)
        info = dict(info)
        info["opponent_key"] = self._current_opponent_key
        return observation, reward, terminated, truncated, info

    def action_masks(self):
        return self._child.action_masks()

    def close(self) -> None:
        self._child.close()


def resolve_opponent_reference(opponent: str) -> str:
    key = str(opponent).strip().lower()
    if key in {"random", "starter"}:
        return key
    if key == "heuristic":
        return str(HEURISTIC_BASELINE_PATH)
    candidate = Path(opponent)
    if candidate.exists():
        return str(candidate)
    raise ValueError(f"unknown opponent {opponent!r}")


def train_initial_policy(config: TrainConfig) -> dict[str, Any]:
    run_id = _build_run_id(config.run_name)
    run_dir = Path(config.output_root) / run_id
    checkpoint_id = build_checkpoint_id(run_id, "final")
    checkpoint_dir = checkpoint_artifact_dir(run_id, checkpoint_id)
    checkpoint_base = checkpoint_dir / "model"
    checkpoint_path = checkpoint_base.with_suffix(".zip")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metadata = build_run_metadata(
        config,
        run_id=run_id,
        run_dir=run_dir,
        checkpoint_id=checkpoint_id,
        checkpoint_path=checkpoint_path,
    )
    _write_json(run_dir / "run_config.json", metadata["run_config"])
    _write_json(run_dir / "metadata.json", metadata)

    if config.dry_run:
        return metadata

    if not TRAINING_STACK_AVAILABLE:
        raise RuntimeError(
            "Training dependencies are not installed. Install requirements-train.txt to run rl/train.py."
        ) from TRAINING_STACK_ERROR

    env = build_training_env(config)
    try:
        model = MaskablePPO(
            config.policy,
            env,
            seed=config.model_seed,
            verbose=config.verbose,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            gamma=config.gamma,
            device="auto",
        )
        model.learn(total_timesteps=config.total_timesteps, progress_bar=False)
        model.save(str(checkpoint_base))
    finally:
        env.close()

    metadata["completed_at"] = _timestamp()
    metadata["checkpoint_metadata"]["final"] = save_checkpoint_artifact(
        run_metadata=metadata,
        checkpoint_id=checkpoint_id,
        source_model_path=checkpoint_path,
    )
    _write_json(run_dir / "metadata.json", metadata)
    return metadata


def build_training_env(config: TrainConfig):
    if not TRAINING_STACK_AVAILABLE:
        raise RuntimeError(
            "Training dependencies are not installed. Install requirements-train.txt to run rl/train.py."
        ) from TRAINING_STACK_ERROR

    env_kwargs = {
        "configuration": {},
        "max_candidates": config.max_candidates,
        "max_actions_per_turn": config.max_actions_per_turn,
        "debug": config.env_debug,
        "reward_config": config.reward_config,
    }

    def factory():
        env = OpponentPoolEnv(
            opponents=config.opponents,
            env_kwargs=env_kwargs,
            env_seed=config.env_seed,
        )
        wrapped = Monitor(env)
        return ActionMasker(wrapped, _mask_fn)

    return DummyVecEnv([factory])


def build_run_metadata(
    config: TrainConfig,
    *,
    run_id: str,
    run_dir: Path,
    checkpoint_id: str,
    checkpoint_path: Path,
) -> dict[str, Any]:
    resolved_opponents = [resolve_opponent_reference(name) for name in config.opponents]
    return {
        "run_id": run_id,
        "created_at": _timestamp(),
        "run_dir": str(run_dir),
        "run_config": serialize_train_config(config),
        "opponent_set": list(config.opponents),
        "resolved_opponents": resolved_opponents,
        "seeds": {
            "model_seed": int(config.model_seed),
            "env_seed": config.env_seed,
        },
        "schema_versions": {
            "global_feature_schema": GLOBAL_FEATURE_SCHEMA.version,
            "candidate_feature_schema": CANDIDATE_FEATURE_SCHEMA.version,
        },
        "reward_config": reward_config_metadata(config.reward_config),
        "checkpoint_metadata": {
            "final": {
                "checkpoint_id": checkpoint_id,
                "artifact_dir": str(checkpoint_path.parent),
                "path": str(checkpoint_path),
                "written": False,
                "feature_schema_version": GLOBAL_FEATURE_SCHEMA.version,
                "candidate_schema_version": CANDIDATE_FEATURE_SCHEMA.version,
            }
        },
        "uses_imitation_learning": False,
    }


def serialize_train_config(config: TrainConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["reward_config"] = reward_config_metadata(config.reward_config)
    payload["opponents"] = list(config.opponents)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train initial MaskablePPO policy against fixed Orbit Wars baselines.")
    parser.add_argument("--timesteps", type=int, default=10_000, help="Total PPO timesteps.")
    parser.add_argument(
        "--opponents",
        default="random",
        help="Comma-separated opponent set: random,starter,heuristic or file paths.",
    )
    parser.add_argument("--run-name", default=None, help="Optional run name prefix.")
    parser.add_argument("--output-root", default="artifacts/train_runs", help="Directory for training run artifacts.")
    parser.add_argument("--model-seed", type=int, default=7, help="Model seed.")
    parser.add_argument("--env-seed", type=int, default=None, help="Optional base environment seed.")
    parser.add_argument("--max-candidates", type=int, default=512, help="Fixed candidate tensor width.")
    parser.add_argument("--max-actions-per-turn", type=int, default=2, help="Maximum sequential launch picks per turn.")
    parser.add_argument("--policy", default="MultiInputPolicy", help="SB3 policy class name.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="MaskablePPO learning rate.")
    parser.add_argument("--n-steps", type=int, default=256, help="Rollout horizon.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--verbose", type=int, default=1, help="SB3 verbosity.")
    parser.add_argument("--dry-run", action="store_true", help="Only write run metadata; do not start PPO training.")
    parser.add_argument("--env-debug", action="store_true", help="Enable Kaggle env debug logs inside training envs.")
    args = parser.parse_args(argv)
    opponents = tuple(item.strip() for item in str(args.opponents).split(",") if item.strip())
    return TrainConfig(
        total_timesteps=int(args.timesteps),
        opponents=opponents,
        run_name=args.run_name,
        output_root=args.output_root,
        model_seed=int(args.model_seed),
        env_seed=args.env_seed,
        max_candidates=int(args.max_candidates),
        max_actions_per_turn=int(args.max_actions_per_turn),
        policy=args.policy,
        learning_rate=float(args.learning_rate),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        gamma=float(args.gamma),
        verbose=int(args.verbose),
        dry_run=bool(args.dry_run),
        env_debug=bool(args.env_debug),
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(argv)
    metadata = train_initial_policy(config)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def _mask_fn(env):
    action_masks = getattr(env, "action_masks", None)
    if callable(action_masks):
        return action_masks()

    get_wrapper_attr = getattr(env, "get_wrapper_attr", None)
    if callable(get_wrapper_attr):
        try:
            action_masks = get_wrapper_attr("action_masks")
        except AttributeError:
            action_masks = None
        if callable(action_masks):
            return action_masks()

    unwrapped = getattr(env, "unwrapped", None)
    action_masks = getattr(unwrapped, "action_masks", None) if unwrapped is not None else None
    if callable(action_masks):
        return action_masks()

    raise AttributeError(f"{type(env).__name__!s} does not expose action_masks() for MaskablePPO")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_run_id(run_name: str | None) -> str:
    prefix = run_name.strip().replace(" ", "_") if run_name else "train"
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
