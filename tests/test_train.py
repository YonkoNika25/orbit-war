from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gymnasium as gym
from gymnasium import spaces

from rl.features import CANDIDATE_FEATURE_SCHEMA, GLOBAL_FEATURE_SCHEMA
from rl.rewards import RewardConfig
from rl.train import (
    HEURISTIC_BASELINE_PATH,
    TrainConfig,
    _mask_fn,
    build_run_metadata,
    parse_args,
    resolve_opponent_reference,
    serialize_train_config,
    train_initial_policy,
)


class TrainingEntryPointTests(unittest.TestCase):
    def test_resolve_baseline_opponents(self) -> None:
        self.assertEqual(resolve_opponent_reference("random"), "random")
        self.assertEqual(resolve_opponent_reference("starter"), "starter")
        self.assertEqual(resolve_opponent_reference("heuristic"), str(HEURISTIC_BASELINE_PATH))
        self.assertTrue(HEURISTIC_BASELINE_PATH.exists())

    def test_parse_args_supports_multiple_opponents(self) -> None:
        config = parse_args(
            [
                "--timesteps",
                "1234",
                "--opponents",
                "random,starter,heuristic",
                "--dry-run",
            ]
        )

        self.assertEqual(config.total_timesteps, 1234)
        self.assertEqual(config.opponents, ("random", "starter", "heuristic"))
        self.assertTrue(config.dry_run)

    def test_build_run_metadata_records_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainConfig(
                opponents=("random", "heuristic"),
                output_root=tmpdir,
                model_seed=11,
                env_seed=101,
                reward_config=RewardConfig(production_swing_weight=0.03),
            )
            run_dir = Path(tmpdir) / "example-run"
            metadata = build_run_metadata(
                config,
                run_id="example-run",
                run_dir=run_dir,
                checkpoint_id="example-run-final",
                checkpoint_path=run_dir / "checkpoints" / "final_model.zip",
            )

        self.assertEqual(metadata["opponent_set"], ["random", "heuristic"])
        self.assertEqual(metadata["seeds"]["model_seed"], 11)
        self.assertEqual(metadata["seeds"]["env_seed"], 101)
        self.assertEqual(metadata["schema_versions"]["global_feature_schema"], GLOBAL_FEATURE_SCHEMA.version)
        self.assertEqual(metadata["schema_versions"]["candidate_feature_schema"], CANDIDATE_FEATURE_SCHEMA.version)
        self.assertFalse(metadata["uses_imitation_learning"])
        self.assertIn("reward_config", metadata)
        self.assertEqual(
            metadata["checkpoint_metadata"]["final"]["candidate_schema_version"],
            CANDIDATE_FEATURE_SCHEMA.version,
        )

    def test_dry_run_writes_metadata_without_training_stack(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainConfig(
                opponents=("random", "starter", "heuristic"),
                output_root=tmpdir,
                run_name="smoke",
                dry_run=True,
                reward_config=RewardConfig(planet_capture_weight=0.07),
            )

            metadata = train_initial_policy(config)
            run_dir = Path(metadata["run_dir"])
            metadata_path = run_dir / "metadata.json"
            config_path = run_dir / "run_config.json"

            self.assertTrue(metadata_path.exists())
            self.assertTrue(config_path.exists())
            self.assertEqual(metadata["opponent_set"], ["random", "starter", "heuristic"])
            self.assertFalse(metadata["checkpoint_metadata"]["final"]["written"])

            stored_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            stored_config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(stored_metadata["run_id"], metadata["run_id"])
            self.assertEqual(stored_config["reward_config"]["planet_capture_weight"], 0.07)

    def test_missing_training_stack_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainConfig(output_root=tmpdir, dry_run=False)
            with mock.patch("rl.train.TRAINING_STACK_AVAILABLE", False), mock.patch(
                "rl.train.TRAINING_STACK_ERROR",
                ModuleNotFoundError("sb3 missing"),
            ):
                with self.assertRaisesRegex(RuntimeError, "requirements-train.txt"):
                    train_initial_policy(config)

    def test_serialize_train_config_includes_reward_config_metadata(self) -> None:
        config = TrainConfig(reward_config=RewardConfig(sun_waste_weight=0.02))

        payload = serialize_train_config(config)

        self.assertEqual(payload["reward_config"]["sun_waste_weight"], 0.02)
        self.assertIn("schema_version", payload["reward_config"])

    def test_mask_fn_resolves_action_masks_through_gym_wrapper(self) -> None:
        class MaskEnv(gym.Env):
            action_space = spaces.Discrete(2)
            observation_space = spaces.Discrete(1)

            def reset(self, *, seed=None, options=None):
                return 0, {}

            def step(self, action):
                return 0, 0.0, True, False, {}

            def action_masks(self):
                return [True, False]

        wrapped = gym.Wrapper(MaskEnv())

        self.assertEqual(_mask_fn(wrapped), [True, False])


if __name__ == "__main__":
    unittest.main()
