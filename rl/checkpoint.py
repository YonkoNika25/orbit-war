from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from rl.features import CANDIDATE_FEATURE_SCHEMA, GLOBAL_FEATURE_SCHEMA

try:
    from sb3_contrib import MaskablePPO

    TRAINING_STACK_AVAILABLE = True
    TRAINING_STACK_ERROR = None
except ModuleNotFoundError as exc:
    MaskablePPO = None
    TRAINING_STACK_AVAILABLE = False
    TRAINING_STACK_ERROR = exc


CHECKPOINT_METADATA_VERSION = "4.4.0"
CHECKPOINT_OUTPUT_ROOT = Path("artifacts/checkpoints")
MODEL_FILENAME = "model.zip"
METADATA_FILENAME = "metadata.json"
_PATH_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


class CheckpointError(RuntimeError):
    """Raised when checkpoint artifacts are missing or malformed."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised when a checkpoint does not match the current feature schemas."""


def build_checkpoint_id(run_id: str, label: str = "final") -> str:
    run_segment = sanitize_checkpoint_segment(run_id)
    label_segment = sanitize_checkpoint_segment(label)
    return f"{run_segment}-{label_segment}"


def sanitize_checkpoint_segment(value: str) -> str:
    text = _PATH_SAFE_RE.sub("_", str(value).strip())
    text = text.strip("._-")
    return text or "checkpoint"


def checkpoint_artifact_dir(
    run_id: str,
    checkpoint_id: str,
    *,
    output_root: str | Path = CHECKPOINT_OUTPUT_ROOT,
) -> Path:
    return Path(output_root) / sanitize_checkpoint_segment(run_id) / sanitize_checkpoint_segment(checkpoint_id)


def build_checkpoint_metadata(
    *,
    run_metadata: Mapping[str, Any],
    checkpoint_id: str,
    checkpoint_dir: Path,
    model_path: Path,
    written: bool,
    size_bytes: int,
) -> dict[str, Any]:
    global_schema = GLOBAL_FEATURE_SCHEMA.metadata()
    candidate_schema = CANDIDATE_FEATURE_SCHEMA.metadata()
    return {
        "checkpoint_schema_version": CHECKPOINT_METADATA_VERSION,
        "checkpoint_id": checkpoint_id,
        "run_id": str(run_metadata["run_id"]),
        "created_at": _timestamp(),
        "model_path": model_path.name,
        "metadata_path": METADATA_FILENAME,
        "written": bool(written),
        "size_bytes": int(size_bytes),
        "feature_schema_version": global_schema["version"],
        "candidate_schema_version": candidate_schema["version"],
        "feature_schema_fingerprint": global_schema["fingerprint"],
        "candidate_schema_fingerprint": candidate_schema["fingerprint"],
        "training_config": run_metadata["run_config"],
        "opponent_set": list(run_metadata["opponent_set"]),
        "reward_config": run_metadata.get("reward_config"),
    }


def save_checkpoint_artifact(
    *,
    run_metadata: Mapping[str, Any],
    checkpoint_id: str,
    source_model_path: str | Path,
    output_root: str | Path = CHECKPOINT_OUTPUT_ROOT,
) -> dict[str, Any]:
    run_id = str(run_metadata["run_id"])
    checkpoint_dir = checkpoint_artifact_dir(run_id, checkpoint_id, output_root=output_root)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(source_model_path)
    if not source_path.exists():
        raise CheckpointError(f"checkpoint model file does not exist: {source_path}")

    target_model_path = checkpoint_dir / MODEL_FILENAME
    if source_path.resolve() != target_model_path.resolve():
        shutil.copy2(source_path, target_model_path)

    metadata = build_checkpoint_metadata(
        run_metadata=run_metadata,
        checkpoint_id=checkpoint_id,
        checkpoint_dir=checkpoint_dir,
        model_path=target_model_path,
        written=target_model_path.exists(),
        size_bytes=target_model_path.stat().st_size if target_model_path.exists() else 0,
    )
    _write_json(checkpoint_dir / METADATA_FILENAME, metadata)
    return metadata


def load_checkpoint_metadata(checkpoint_ref: str | Path) -> dict[str, Any]:
    metadata_path = resolve_checkpoint_metadata_path(checkpoint_ref)
    if not metadata_path.exists():
        raise CheckpointError(f"checkpoint metadata file does not exist: {metadata_path}")

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CheckpointError(f"checkpoint metadata must be a JSON object: {metadata_path}")
    return payload


def validate_checkpoint_compatibility(metadata: Mapping[str, Any]) -> None:
    expected_global = GLOBAL_FEATURE_SCHEMA.metadata()
    expected_candidate = CANDIDATE_FEATURE_SCHEMA.metadata()
    actual_global_version = str(metadata.get("feature_schema_version", ""))
    actual_candidate_version = str(metadata.get("candidate_schema_version", ""))
    actual_global_fingerprint = str(metadata.get("feature_schema_fingerprint", ""))
    actual_candidate_fingerprint = str(metadata.get("candidate_schema_fingerprint", ""))

    if actual_global_version != expected_global["version"] or (
        actual_global_fingerprint and actual_global_fingerprint != expected_global["fingerprint"]
    ):
        raise CheckpointCompatibilityError(
            "checkpoint global feature schema is incompatible with current runtime"
        )

    if actual_candidate_version != expected_candidate["version"] or (
        actual_candidate_fingerprint and actual_candidate_fingerprint != expected_candidate["fingerprint"]
    ):
        raise CheckpointCompatibilityError(
            "checkpoint candidate feature schema is incompatible with current runtime"
        )


def load_checkpoint_artifact(
    checkpoint_ref: str | Path,
    *,
    load_model: bool = False,
    device: str = "auto",
) -> dict[str, Any]:
    metadata_path = resolve_checkpoint_metadata_path(checkpoint_ref)
    metadata = load_checkpoint_metadata(metadata_path)
    validate_checkpoint_compatibility(metadata)

    checkpoint_dir = metadata_path.parent
    model_entry = str(metadata.get("model_path", MODEL_FILENAME))
    model_path = Path(model_entry)
    if not model_path.is_absolute():
        model_path = checkpoint_dir / model_path
    if not model_path.exists():
        raise CheckpointError(f"checkpoint model file does not exist: {model_path}")

    result: dict[str, Any] = {
        "metadata": metadata,
        "metadata_path": str(metadata_path),
        "model_path": str(model_path),
    }
    if not load_model:
        return result

    if not TRAINING_STACK_AVAILABLE:
        raise RuntimeError(
            "Training dependencies are not installed. Install requirements-train.txt to load MaskablePPO checkpoints."
        ) from TRAINING_STACK_ERROR

    result["model"] = MaskablePPO.load(str(model_path), device=device)
    return result


def resolve_checkpoint_metadata_path(checkpoint_ref: str | Path) -> Path:
    path = Path(checkpoint_ref)
    if path.is_dir():
        return path / METADATA_FILENAME
    if path.name == METADATA_FILENAME:
        return path
    if path.suffix.lower() == ".zip":
        return path.with_name(METADATA_FILENAME)
    raise CheckpointError(f"unsupported checkpoint reference: {checkpoint_ref}")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
