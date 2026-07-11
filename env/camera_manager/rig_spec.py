"""Normalized camera-rig contracts with legacy configuration compatibility."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

VALID_MOUNT_KINDS = frozenset({"world", "robot_link", "scene_fixture"})
VALID_PROJECTION_MODELS = frozenset({"pinhole", "opencvFisheye"})


def hardware_camera_parent(hardware_path: str, camera_frame: str | None) -> str:
    """Resolve a named camera frame below an instantiated holder prim."""
    if not camera_frame:
        return hardware_path.rsplit("/", 1)[0]
    if camera_frame.startswith("/") or ".." in camera_frame.split("/"):
        raise ValueError("hardware camera frame must be a relative prim path")
    return hardware_path.rstrip("/") + "/" + camera_frame.strip("/")


@dataclass(frozen=True)
class CameraSpec:
    observation_key: str
    role: str
    camera_type: str
    mesh: str
    sensor: Mapping[str, Any]
    mount: Mapping[str, Any]
    projection: Mapping[str, Any]

    def __post_init__(self):
        mount_kind = self.mount.get("kind")
        if mount_kind not in VALID_MOUNT_KINDS:
            raise ValueError(f"{self.observation_key}: invalid mount kind {mount_kind!r}")
        if mount_kind != "world" and not self.mount.get("target"):
            raise ValueError(f"{self.observation_key}: {mount_kind} mount requires a target")
        position = self.mount.get("position", [0.0, 0.0, 0.0])
        orientation = self.mount.get("orientation", [1.0, 0.0, 0.0, 0.0])
        if len(position) != 3 or len(orientation) not in (3, 4):
            raise ValueError(f"{self.observation_key}: invalid mount pose")
        roll = float(self.mount.get("optical_roll_deg", 0.0))
        if roll not in (-180.0, -90.0, 0.0, 90.0, 180.0):
            raise ValueError(f"{self.observation_key}: optical roll must be a right-angle rotation")
        model = self.projection.get("model", "pinhole")
        if model not in VALID_PROJECTION_MODELS:
            raise ValueError(f"{self.observation_key}: invalid projection model {model!r}")
        stream = self.sensor.get("stream_resolution")
        if not isinstance(stream, (list, tuple)) or len(stream) != 2 or min(stream) <= 0:
            raise ValueError(f"{self.observation_key}: sensor stream_resolution must be [width, height]")
        if float(self.sensor.get("fps", 0)) <= 0:
            raise ValueError(f"{self.observation_key}: sensor fps must be positive")
        hardware = self.mount.get("hardware")
        if hardware is not None:
            if not isinstance(hardware, Mapping) or not hardware.get("asset"):
                raise ValueError(f"{self.observation_key}: mount hardware requires an asset")
            if len(hardware.get("position", [0, 0, 0])) != 3:
                raise ValueError(f"{self.observation_key}: invalid mount hardware position")
            if len(hardware.get("orientation", [0, 0, 0])) not in (3, 4):
                raise ValueError(f"{self.observation_key}: invalid mount hardware orientation")
            camera_frame = hardware.get("camera_frame")
            if camera_frame is not None and (
                not isinstance(camera_frame, str)
                or not camera_frame
                or camera_frame.startswith("/")
                or ".." in camera_frame.split("/")
            ):
                raise ValueError(f"{self.observation_key}: mount hardware camera_frame must be a relative prim path")
            if camera_frame is not None and (
                "position" in hardware or "orientation" in hardware
            ):
                raise ValueError(
                    f"{self.observation_key}: named camera frames derive hardware pose from the mount target"
                )

    def runtime_camera(self) -> dict[str, Any]:
        """Flatten the normalized contract for the existing Isaac camera implementation."""
        result = {
            "type": self.camera_type,
            "mesh": self.mesh,
            "role": self.role,
            "sensor_model": self.sensor.get("model"),
            "sensor_vendor": self.sensor.get("vendor"),
            "native_resolution": list(self.sensor.get("native_resolution", self.sensor["stream_resolution"])),
            "stream_resolution": list(self.sensor["stream_resolution"]),
            "sensor_fps": float(self.sensor["fps"]),
            "published_diagonal_fov_deg": float(self.sensor["diagonal_fov_deg"]),
            "mount_kind": self.mount["kind"],
            "mount_target": self.mount.get("target"),
            "pos": list(self.mount.get("position", [0.0, 0.0, 0.0])),
            "ori": list(self.mount.get("orientation", [1.0, 0.0, 0.0, 0.0])),
            "optical_roll_deg": float(self.mount.get("optical_roll_deg", 0.0)),
            "mount_basis": self.mount.get("basis"),
            "lens_distortion_model": self.projection.get("model", "pinhole"),
            "projection_backend": self.projection.get("backend", "native"),
        }
        for key in ("cx", "cy", "fx", "fy", "distortion_coefficients"):
            if key in self.projection:
                result[key] = deepcopy(self.projection[key])
        if self.mount.get("hardware"):
            hardware = self.mount["hardware"]
            result.update(
                mount_hardware_asset=hardware["asset"],
                mount_hardware_collision=bool(hardware.get("collision", True)),
                mount_hardware_camera_frame=hardware.get("camera_frame"),
            )
            if not hardware.get("camera_frame"):
                result.update(
                    mount_hardware_position=list(hardware.get("position", [0.0, 0.0, 0.0])),
                    mount_hardware_orientation=list(hardware.get("orientation", [0.0, 0.0, 0.0])),
                )
        return {key: value for key, value in result.items() if value is not None}


@dataclass(frozen=True)
class CameraRigSpec:
    profile_id: str
    cameras: tuple[CameraSpec, ...]
    default_frequency: float
    annotator: Mapping[str, Any]

    def __post_init__(self):
        keys = [camera.observation_key for camera in self.cameras]
        if len(keys) != len(set(keys)):
            raise ValueError(f"camera rig contains duplicate observation keys: {keys}")
        for camera in self.cameras:
            if abs(float(camera.sensor["fps"]) - float(self.default_frequency)) > 1e-9:
                raise ValueError(
                    f"{camera.observation_key}: sensor fps {camera.sensor['fps']} differs from rig frequency {self.default_frequency}"
                )

    def runtime_config(self) -> DictConfig:
        config: dict[str, Any] = {
            "default_frequency": self.default_frequency,
            "annotator": deepcopy(dict(self.annotator)),
        }
        for camera in self.cameras:
            config[camera.observation_key] = {"camera": camera.runtime_camera()}
        return OmegaConf.create(config)


def _legacy_camera_spec(name: str, section: Mapping[str, Any], frequency: float) -> CameraSpec:
    camera = deepcopy(dict(section["camera"]))
    camera_type = camera.get("type")
    from env_cfg.camera.template import CAMERA_TYPE_RESOLUTIONS

    stream = list(camera.get("stream_resolution", CAMERA_TYPE_RESOLUTIONS.get(camera_type, (640, 480))))
    mount_link = camera.get("mount_link")
    mount_kind = camera.get("mount_kind", "robot_link" if mount_link else "world")
    sensor = {
        "vendor": camera.get("sensor_vendor", "legacy"),
        "model": camera.get("sensor_model", camera_type),
        "native_resolution": list(camera.get("native_resolution", stream)),
        "stream_resolution": stream,
        "fps": float(camera.get("sensor_fps", frequency)),
        "diagonal_fov_deg": float(camera.get("published_diagonal_fov_deg", 0.0)),
    }
    mount = {
        "kind": mount_kind,
        "target": camera.get("mount_target", mount_link),
        "position": list(camera.get("pos", [0.0, 0.0, 0.0])),
        "orientation": list(camera.get("ori", [1.0, 0.0, 0.0, 0.0])),
        "optical_roll_deg": float(camera.get("optical_roll_deg", 0.0)),
        "basis": camera.get("mount_basis", "legacy"),
    }
    if camera.get("mount_hardware_asset"):
        hardware = {
            "asset": camera["mount_hardware_asset"],
            "collision": bool(camera.get("mount_hardware_collision", True)),
            "camera_frame": camera.get("mount_hardware_camera_frame"),
        }
        if not hardware["camera_frame"]:
            hardware.update(
                position=list(camera.get("mount_hardware_position", [0, 0, 0])),
                orientation=list(camera.get("mount_hardware_orientation", [0, 0, 0])),
            )
        mount["hardware"] = hardware
    projection = {
        "model": camera.get("lens_distortion_model", "pinhole"),
        "backend": camera.get("projection_backend", "native"),
    }
    for key in ("cx", "cy", "fx", "fy", "distortion_coefficients"):
        if key in camera:
            projection[key] = deepcopy(camera[key])
    role = camera.get("role") or {
        "cam_head": "base",
        "cam_left_wrist": "left_wrist",
        "cam_right_wrist": "right_wrist",
    }.get(name, name)
    return CameraSpec(name, role, camera_type, camera.get("mesh", "pinhole"), sensor, mount, projection)


def normalize_camera_rig(
    config: DictConfig | Mapping[str, Any], robot_cameras: Mapping[str, Mapping[str, Any]] | None = None
) -> CameraRigSpec:
    """Normalize the new layered schema or any existing flat camera config."""
    raw = OmegaConf.to_container(config, resolve=True) if isinstance(config, DictConfig) else deepcopy(dict(config))
    frequency = float(raw.get("default_frequency", 30.0))
    annotator = raw.get("annotator", {})
    layered = raw.get("camera_rig")
    if layered is not None:
        cameras = []
        for observation_key, value in layered.get("cameras", {}).items():
            cameras.append(
                CameraSpec(
                    observation_key=observation_key,
                    role=value["role"],
                    camera_type=value["type"],
                    mesh=value.get("mesh", "pinhole"),
                    sensor=deepcopy(value["sensor"]),
                    mount=deepcopy(value["mount"]),
                    projection=deepcopy(value["projection"]),
                )
            )
        return CameraRigSpec(layered["profile_id"], tuple(cameras), frequency, annotator)

    sections = {
        key: value
        for key, value in raw.items()
        if isinstance(value, Mapping) and isinstance(value.get("camera"), Mapping)
    }
    for key, value in (robot_cameras or {}).items():
        sections.setdefault(key, value)
    cameras = tuple(_legacy_camera_spec(key, value, frequency) for key, value in sections.items())
    return CameraRigSpec("legacy", cameras, frequency, annotator)
