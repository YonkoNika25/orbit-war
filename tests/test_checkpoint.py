from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rl.checkpoint import (
    CheckpointCompatibilityError,
    build_checkpoint_id,
    checkpoint_artifact_dir,
    load_checkpoint_artifact,
    load_checkpoint_metadata,
    save_checkpoint_artifact,
    validate_checkpoint_compatibility,
)
from rl.rewards import RewardConfig
from rl.train import TrainConfig, build_run_metadata


class CheckpointArtifactTests(unittest.TestCase):
    def _run_metadata(self, tmpdir: str) -> dict:
        config = TrainConfig(
            opponents=("random", "heuristic"),
            output_root=tmpdir,
            model_seed=11,
            env_seed=101,
            reward_config=RewardConfig(production_swing_weight=0.03),
        )
        run_id = "reviewed-run-20260521"
        checkpoint_id = build_checkpoint_id(run_id, "final")
        run_dir = Path(tmpdir) / "train_runs" / run_id
        checkpoint_path = checkpoint_artifact_dir(run_id, checkpoint_id, output_root=Path(tmpdir) / "checkpoints") / "model.zip"
        return build_run_metadata(
            config,
            run_id=run_id,
            run_dir=run_dir,
            checkpoint_id=checkpoint_id,
            checkpoint_path=checkpoint_path,
        )

    def test_build_checkpoint_id_is_path_safe(self) -> None:
        checkpoint_id = build_checkpoint_id("run:one", "final checkpoint")

        self.assertEqual(checkpoint_id, "run_one-final_checkpoint")
        self.assertNotIn(":", checkpoint_id)
        self.assertNotIn(" ", checkpoint_id)

    def test_save_checkpoint_artifact_writes_model_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_metadata = self._run_metadata(tmpdir)
            source_model_path = Path(tmpdir) / "model.zip"
            source_model_path.write_bytes(b"demo-model")

            metadata = save_checkpoint_artifact(
                run_metadata=run_metadata,
                checkpoint_id=run_metadata["checkpoint_metadata"]["final"]["checkpoint_id"],
                source_model_path=source_model_path,
                output_root=Path(tmpdir) / "checkpoints",
            )

            checkpoint_dir = checkpoint_artifact_dir(
                run_metadata["run_id"],
                metadata["checkpoint_id"],
                output_root=Path(tmpdir) / "checkpoints",
            )
            self.assertTrue((checkpoint_dir / "model.zip").exists())
            self.assertTrue((checkpoint_dir / "metadata.json").exists())
            self.assertEqual(metadata["run_id"], run_metadata["run_id"])
            self.assertEqual(metadata["feature_schema_version"], run_metadata["schema_versions"]["global_feature_schema"])
            self.assertEqual(
                metadata["candidate_schema_version"],
                run_metadata["schema_versions"]["candidate_feature_schema"],
            )
            self.assertEqual(metadata["training_config"], run_metadata["run_config"])
            self.assertEqual(metadata["opponent_set"], run_metadata["opponent_set"])
            self.assertNotIn("resolved_opponents", metadata)
            self.assertTrue(metadata["written"])

    def test_load_checkpoint_artifact_reads_directory_metadata_and_validates_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_metadata = self._run_metadata(tmpdir)
            source_model_path = Path(tmpdir) / "model.zip"
            source_model_path.write_bytes(b"demo-model")
            saved = save_checkpoint_artifact(
                run_metadata=run_metadata,
                checkpoint_id=run_metadata["checkpoint_metadata"]["final"]["checkpoint_id"],
                source_model_path=source_model_path,
                output_root=Path(tmpdir) / "checkpoints",
            )
            checkpoint_dir = checkpoint_artifact_dir(
                run_metadata["run_id"],
                saved["checkpoint_id"],
                output_root=Path(tmpdir) / "checkpoints",
            )

            loaded = load_checkpoint_artifact(checkpoint_dir)

            self.assertEqual(loaded["metadata"]["checkpoint_id"], saved["checkpoint_id"])
            self.assertEqual(Path(loaded["model_path"]).name, "model.zip")

    def test_validate_checkpoint_compatibility_rejects_mismatched_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_metadata = self._run_metadata(tmpdir)
            source_model_path = Path(tmpdir) / "model.zip"
            source_model_path.write_bytes(b"demo-model")
            saved = save_checkpoint_artifact(
                run_metadata=run_metadata,
                checkpoint_id=run_metadata["checkpoint_metadata"]["final"]["checkpoint_id"],
                source_model_path=source_model_path,
                output_root=Path(tmpdir) / "checkpoints",
            )
            saved["candidate_schema_version"] = "0.0.0"

            with self.assertRaises(CheckpointCompatibilityError):
                validate_checkpoint_compatibility(saved)

    def test_load_checkpoint_model_requires_training_stack(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_metadata = self._run_metadata(tmpdir)
            source_model_path = Path(tmpdir) / "model.zip"
            source_model_path.write_bytes(b"demo-model")
            saved = save_checkpoint_artifact(
                run_metadata=run_metadata,
                checkpoint_id=run_metadata["checkpoint_metadata"]["final"]["checkpoint_id"],
                source_model_path=source_model_path,
                output_root=Path(tmpdir) / "checkpoints",
            )
            checkpoint_dir = checkpoint_artifact_dir(
                run_metadata["run_id"],
                saved["checkpoint_id"],
                output_root=Path(tmpdir) / "checkpoints",
            )

            with mock.patch("rl.checkpoint.TRAINING_STACK_AVAILABLE", False), mock.patch(
                "rl.checkpoint.TRAINING_STACK_ERROR",
                ModuleNotFoundError("sb3 missing"),
            ):
                with self.assertRaisesRegex(RuntimeError, "requirements-train.txt"):
                    load_checkpoint_artifact(checkpoint_dir, load_model=True)

    def test_load_checkpoint_metadata_accepts_model_zip_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_metadata = self._run_metadata(tmpdir)
            source_model_path = Path(tmpdir) / "model.zip"
            source_model_path.write_bytes(b"demo-model")
            saved = save_checkpoint_artifact(
                run_metadata=run_metadata,
                checkpoint_id=run_metadata["checkpoint_metadata"]["final"]["checkpoint_id"],
                source_model_path=source_model_path,
                output_root=Path(tmpdir) / "checkpoints",
            )
            checkpoint_dir = checkpoint_artifact_dir(
                run_metadata["run_id"],
                saved["checkpoint_id"],
                output_root=Path(tmpdir) / "checkpoints",
            )

            metadata = load_checkpoint_metadata(checkpoint_dir / "model.zip")

            self.assertEqual(metadata["checkpoint_id"], saved["checkpoint_id"])
