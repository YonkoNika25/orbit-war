from __future__ import annotations

import unittest

try:
    import numpy as np
    from rl.env import OrbitWarsMaskableEnv

    TRAINING_DEPS_AVAILABLE = True
except ModuleNotFoundError:
    np = None
    OrbitWarsMaskableEnv = None
    TRAINING_DEPS_AVAILABLE = False

from rl.types import CandidateType


@unittest.skipUnless(TRAINING_DEPS_AVAILABLE, "training dependencies are not installed")
class OrbitWarsMaskableEnvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = OrbitWarsMaskableEnv(
            configuration={"seed": 7},
            max_candidates=512,
            max_actions_per_turn=2,
            debug=True,
        )

    def tearDown(self) -> None:
        self.env.close()

    def _first_legal_launch_index(self) -> int:
        batch = self.env.current_batch
        self.assertIsNotNone(batch)
        for index, (candidate, legal) in enumerate(zip(batch.candidates, batch.mask)):
            if not legal:
                continue
            if candidate.type in {CandidateType.STOP, CandidateType.HOLD_SOURCE}:
                continue
            return index
        self.fail("expected at least one legal launch candidate")

    def test_reset_returns_fixed_shape_observation_and_masks(self) -> None:
        observation, info = self.env.reset()
        action_mask = self.env.action_masks()

        self.assertEqual(observation["global_features"].shape, (17,))
        self.assertEqual(observation["candidate_features"].shape, (512, 21))
        self.assertEqual(observation["candidate_mask"].shape, (512,))
        self.assertEqual(observation["candidate_count"].shape, (1,))
        self.assertEqual(action_mask.shape, (512,))
        self.assertEqual(action_mask.dtype, np.bool_)
        candidate_count = int(observation["candidate_count"][0])
        self.assertGreater(candidate_count, 0)
        np.testing.assert_array_equal(
            action_mask[:candidate_count],
            observation["candidate_mask"][:candidate_count].astype(bool),
        )
        self.assertFalse(action_mask[candidate_count:].any())
        self.assertEqual(info["candidate_count"], candidate_count)
        self.assertFalse(info["turn_advanced"])
        self.assertIn("reward_config", info)
        self.assertIn("schema_version", info["reward_config"])

    def test_launch_candidate_keeps_same_turn_until_stop(self) -> None:
        observation, _ = self.env.reset()
        initial_step = self.env.current_game_step
        launch_index = self._first_legal_launch_index()

        observation, reward, terminated, truncated, info = self.env.step(launch_index)

        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["turn_advanced"])
        self.assertEqual(self.env.current_game_step, initial_step)
        self.assertEqual(self.env.current_pass_index, 1)
        self.assertEqual(self.env.pending_action_count, 1)
        self.assertNotEqual(info["selected_type"], "STOP")
        self.assertEqual(int(observation["candidate_count"][0]), info["candidate_count"])

    def test_stop_finalizes_turn_and_advances_underlying_trainer(self) -> None:
        self.env.reset()
        initial_step = self.env.current_game_step
        launch_index = self._first_legal_launch_index()

        self.env.step(launch_index)
        observation, reward, terminated, truncated, info = self.env.step(0)

        self.assertIsInstance(reward, float)
        self.assertFalse(terminated and truncated)
        self.assertTrue(info["turn_advanced"])
        self.assertGreaterEqual(self.env.current_game_step, initial_step + 1)
        self.assertEqual(self.env.current_pass_index, 0)
        self.assertEqual(self.env.pending_action_count, 0)
        self.assertEqual(info["selected_type"], "STOP")
        self.assertIn("reward_breakdown", info)
        self.assertIn("env_reward", info)
        self.assertGreater(int(observation["candidate_count"][0]), 0)

    def test_max_actions_per_turn_finalizes_after_launch(self) -> None:
        env = OrbitWarsMaskableEnv(
            configuration={"seed": 7},
            max_candidates=512,
            max_actions_per_turn=1,
            debug=True,
        )
        try:
            env.reset()
            initial_step = env.current_game_step
            batch = env.current_batch
            self.assertIsNotNone(batch)
            launch_index = next(
                index
                for index, (candidate, legal) in enumerate(zip(batch.candidates, batch.mask))
                if legal and candidate.type not in {CandidateType.STOP, CandidateType.HOLD_SOURCE}
            )

            observation, reward, terminated, truncated, info = env.step(launch_index)

            self.assertIsInstance(reward, float)
            self.assertFalse(terminated and truncated)
            self.assertTrue(info["turn_advanced"])
            self.assertGreaterEqual(env.current_game_step, initial_step + 1)
            self.assertEqual(env.current_pass_index, 0)
            self.assertEqual(env.pending_action_count, 0)
            self.assertIn("reward_breakdown", info)
            self.assertGreater(int(observation["candidate_count"][0]), 0)
        finally:
            env.close()

    def test_invalid_selected_index_falls_back_safely(self) -> None:
        self.env.reset()

        observation, reward, terminated, truncated, info = self.env.step(9999)

        self.assertIsInstance(reward, float)
        self.assertFalse(terminated and truncated)
        self.assertTrue(info["invalid_action_selected"])
        self.assertEqual(info["selected_type"], "STOP")
        self.assertTrue(info["turn_advanced"])
        self.assertIn("reward_breakdown", info)
        self.assertGreater(int(observation["candidate_count"][0]), 0)


if __name__ == "__main__":
    unittest.main()
