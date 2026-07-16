"""Capture the exact first policy observation as an atomic RGB bundle."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

FIRST_FRAME_FORMAT_VERSION = 1
METADATA_NAME = "metadata.json"
CONTACT_SHEET_NAME = "contact_sheet.png"


@dataclass(frozen=True)
class FirstFrameIdentity:
    recipe: str
    contract_hash: str
    task: str
    protocol: str
    profile: str
    scene_config: str
    seed: int
    layout_id: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        return [_json_value(item) for item in value]
    except TypeError:
        return str(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_camera_name(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_.-]+", value) is None or value in {".", ".."}:
        raise ValueError(f"camera observation key is not filesystem-safe: {value!r}")
    return value


def _artifact_matches(output: Path, artifact: object) -> bool:
    if not isinstance(artifact, dict):
        return False
    relative = artifact.get("path")
    expected_hash = artifact.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected_hash, str) or len(expected_hash) != 64:
        return False
    path = output / relative
    return path.is_file() and _sha256_file(path) == expected_hash


def completed_first_frame_matches(output_dir: str | Path, identity: FirstFrameIdentity) -> bool:
    """Return whether a complete RGB bundle matches this exact first-frame request."""
    output = Path(output_dir)
    try:
        metadata = json.loads((output / METADATA_NAME).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return False
    artifacts = metadata.get("artifacts")
    cameras = artifacts.get("cameras") if isinstance(artifacts, dict) else None
    return bool(
        metadata.get("complete")
        and metadata.get("format_version") == FIRST_FRAME_FORMAT_VERSION
        and metadata.get("identity") == identity.to_dict()
        and isinstance(cameras, dict)
        and cameras
        and _artifact_matches(output, artifacts.get("contact_sheet"))
        and all(_artifact_matches(output, artifact) for artifact in cameras.values())
    )


def _save_contact_sheet(output: Path, frames: Mapping[str, np.ndarray]) -> Path:
    label_height = 34
    labeled_images: list[Image.Image] = []
    for camera_name, frame in frames.items():
        image = Image.fromarray(frame)
        labeled = Image.new("RGB", (image.width, image.height + label_height), "#14212b")
        labeled.paste(image, (0, label_height))
        ImageDraw.Draw(labeled).text((10, 10), camera_name, fill="#f7fafb")
        labeled_images.append(labeled)
    width = sum(image.width for image in labeled_images)
    height = max(image.height for image in labeled_images)
    sheet = Image.new("RGB", (width, height), "#14212b")
    offset = 0
    for image in labeled_images:
        sheet.paste(image, (offset, 0))
        offset += image.width
    path = output / CONTACT_SHEET_NAME
    sheet.save(path)
    return path


def _identity(env, layout_id: int) -> FirstFrameIdentity:
    return FirstFrameIdentity(
        recipe=str(env.recipe_name),
        contract_hash=str(env.contract_hash),
        task=str(env.task_name),
        protocol=str(env.protocol_name),
        profile=str(env.config_name),
        scene_config=str(env.scene_config),
        seed=int(env.eval_seed),
        layout_id=int(layout_id),
    )


def capture_first_frame(env, output_dir: str | os.PathLike[str], layout_id: int) -> Path:
    """Capture ``env.get_obs()`` once and atomically persist all RGB cameras."""
    output = Path(output_dir).expanduser().resolve()
    identity = _identity(env, layout_id)
    if output.exists():
        if completed_first_frame_matches(output, identity):
            logger.info("[first-frame] reusing completed RGB bundle: %s", output)
            return output
        raise FileExistsError(f"first-frame directory already exists and does not match this run: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        observation = env.get_obs()
        vision = observation.get("vision") if isinstance(observation, dict) else None
        if not isinstance(vision, Mapping) or not vision:
            raise RuntimeError("first rollout observation contains no configured vision cameras")

        frames: dict[str, np.ndarray] = {}
        for camera_name, camera_data in vision.items():
            safe_name = _safe_camera_name(str(camera_name))
            color = camera_data.get("color") if isinstance(camera_data, Mapping) else None
            if color is None:
                raise RuntimeError(f"first rollout observation camera {camera_name!r} has no RGB color frame")
            frame = np.asarray(color)
            if frame.ndim != 3 or frame.shape[2] not in (3, 4):
                raise RuntimeError(f"camera {camera_name!r} returned invalid RGB shape {frame.shape}")
            frames[safe_name] = np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8)

        camera_artifacts: dict[str, dict[str, Any]] = {}
        rig_by_key = {camera.observation_key: camera for camera in env.camera_rig.cameras}
        for camera_name, frame in frames.items():
            path = temporary / f"{camera_name}.png"
            Image.fromarray(frame).save(path)
            spec = rig_by_key.get(camera_name)
            camera_artifacts[camera_name] = {
                "path": path.name,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
                "shape": list(frame.shape),
                "role": None if spec is None else spec.role,
            }
        contact_sheet = _save_contact_sheet(temporary, frames)
        metadata = {
            "format_version": FIRST_FRAME_FORMAT_VERSION,
            "complete": True,
            "created_at": datetime.now(UTC).isoformat(),
            "snapshot_boundary": "post_reset_first_rollout_observation_pre_action",
            "identity": identity.to_dict(),
            "instruction": _json_value(observation.get("instruction")),
            "camera_order": list(frames),
            "artifacts": {
                "contact_sheet": {
                    "path": contact_sheet.name,
                    "sha256": _sha256_file(contact_sheet),
                    "size_bytes": contact_sheet.stat().st_size,
                },
                "cameras": camera_artifacts,
            },
        }
        (temporary / METADATA_NAME).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output)
        logger.info("[first-frame] wrote %s RGB cameras and contact sheet to %s", len(frames), output)
        return output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
