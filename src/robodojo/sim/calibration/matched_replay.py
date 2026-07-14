"""Matched-state wrist-camera replay diagnostic.

This module is entered only through the developer environment variable
``ROBODOJO_MATCHED_REPLAY_DIR``; it does not add a launcher interface.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import urllib.request

import numpy as np
from PIL import Image, ImageDraw
import yaml

from robodojo.sim.environment.robot_manager.control_manager import MetaControl


def _control_from_right_first(state: list[float]) -> MetaControl:
    if len(state) != 16:
        raise ValueError("matched replay state must use the right-first 16-D contract")
    right_gripper_m = float(np.clip(-state[7] * 0.044 / 65.0, 0.0, 0.044))
    left_gripper_m = float(np.clip(-state[15] * 0.044 / 65.0, 0.0, 0.044))
    return MetaControl(
        {
            "right_arm_joint_state": {"position": np.deg2rad(state[:7]).tolist(), "velocity": [0.0] * 7},
            "right_ee_joint_state": {"position": [right_gripper_m], "velocity": [0.0]},
            "left_arm_joint_state": {"position": np.deg2rad(state[8:15]).tolist(), "velocity": [0.0] * 7},
            "left_ee_joint_state": {"position": [left_gripper_m], "velocity": [0.0]},
        }
    )


def _control_from_yam_left_first(state: list[float], contract: dict) -> MetaControl:
    """Adapt released LeRobot YAM state to simulator joint targets."""
    if contract.get("state_adapter") != "yam_left_first_14d" or len(state) != 14:
        raise ValueError("YAM matched replay state must use the left-first 14-D contract")
    signs = np.asarray(contract.get("simulator_arm_joint_signs_per_arm"), dtype=np.float64)
    if signs.shape != (6,) or not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError("YAM matched replay requires six simulator arm joint signs")
    state = np.asarray(state, dtype=np.float64)
    if not np.all(np.isfinite(state)):
        raise ValueError("YAM matched replay state must be finite")
    left_gripper = -0.0475 * float(np.clip(state[6], 0.0, 1.0))
    right_gripper = -0.0475 * float(np.clip(state[13], 0.0, 1.0))
    return MetaControl(
        {
            "left_arm_joint_state": {"position": (state[:6] * signs).tolist(), "velocity": [0.0] * 6},
            "left_ee_joint_state": {"position": [left_gripper], "velocity": [0.0]},
            "right_arm_joint_state": {"position": (state[7:13] * signs).tolist(), "velocity": [0.0] * 6},
            "right_ee_joint_state": {"position": [right_gripper], "velocity": [0.0]},
        }
    )


def _control_from_manifest(manifest: dict, state: list[float]) -> MetaControl:
    adapter = manifest.get("replay_contract", {}).get("state_adapter")
    if adapter == "yam_left_first_14d":
        return _control_from_yam_left_first(state, manifest["replay_contract"])
    if adapter in (None, "openarm_right_first_16d") and manifest.get("profile_id") == "openarm_lerobot":
        return _control_from_right_first(state)
    raise ValueError(f"unsupported matched replay state adapter: {adapter!r}")


def _fetch_reference(manifest: dict, side: str, observation: dict, output: Path) -> None:
    dataset = manifest["dataset"]
    url = f"{dataset['repository']}/resolve/{dataset['revision']}/{dataset['videos'][side]}"
    video = output.parent / f"source-{side}.mp4"
    if not video.exists():
        urllib.request.urlretrieve(url, video)
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(observation["video_time_s"]),
            "-i",
            video,
            "-frames:v",
            "1",
            output,
        ],
        check=True,
    )


def _overlay(image: Image.Image, landmarks: list[dict], color: str) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for landmark in landmarks:
        x, y = landmark["pixel"]
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=color, width=3)
        draw.text((x + 8, y - 8), landmark["name"], fill=color)
    return result


def _overlay_key(image: Image.Image, landmarks: list[dict], pixel_key: str, color: str) -> Image.Image:
    return _overlay(
        image,
        [{"name": landmark["name"], "pixel": landmark[pixel_key]} for landmark in landmarks],
        color,
    )


def _fetch_yam_reference(manifest: dict, camera_key: str, frame: dict, output: Path) -> None:
    dataset_name = frame["dataset"]
    dataset = manifest["datasets"][dataset_name]
    camera_name = camera_key.rsplit(".", 1)[-1]
    source = frame["camera_sources"][camera_name]
    video_path = source["video_path"]
    url = f"https://huggingface.co/datasets/{dataset_name}/resolve/{dataset['revision']}/{video_path}"
    video = output.parent / (
        f"source-{dataset_name.replace('/', '--')}-{camera_name}-{dataset['revision'][:12]}.mp4"
    )
    if not video.exists():
        urllib.request.urlretrieve(url, video)
    from robodojo.sim.calibration.wrist_camera import validate_frame

    checksum_stamp = video.with_suffix(".sha256")
    expected_video_sha256 = source["video_sha256"]
    if not checksum_stamp.is_file() or checksum_stamp.read_text().strip() != expected_video_sha256:
        validate_frame(video, expected_video_sha256)
        checksum_stamp.write_text(expected_video_sha256 + "\n")

    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(frame["timestamp_s"]),
            "-i",
            video,
            "-frames:v",
            "1",
            output,
        ],
        check=True,
    )
    validate_frame(output, source["image_sha256"])


def _run_yam_matched_replay(env, manifest: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    frames = manifest["selection_contract"]["frames"]
    for index, frame in enumerate(frames):
        state = frame["observation_state"]
        for _ in range(8):
            env.robot_manager.control_robot([_control_from_manifest(manifest, state)])
            for _ in range(int(env.obs_manager.collect_interval) * 4):
                env.sim_step(render=False)
        observation = env.get_obs_batch(env_idx_list=[0], last_frame=True)[0]
        for side, camera_key, observation_key in (
            ("left", "observation.images.left", "cam_left_wrist"),
            ("right", "observation.images.right", "cam_right_wrist"),
        ):
            annotation = frame["wrist_annotations"][side]
            reference_path = output_dir / f"{index:02d}-{side}-upstream.png"
            rendered_path = output_dir / f"{index:02d}-{side}-rendered.png"
            pair_path = output_dir / f"{index:02d}-{side}-pair.jpg"
            _fetch_yam_reference(manifest, camera_key, frame, reference_path)
            rendered = Image.fromarray(np.asarray(observation["vision"][observation_key]["color"], dtype=np.uint8))
            rendered.save(rendered_path)
            reference = _overlay_key(
                Image.open(reference_path).convert("RGB"),
                annotation["landmarks"],
                "pixel",
                "#00ff9d",
            )
            rendered_overlay = rendered
            pair = Image.new("RGB", (reference.width + rendered.width, max(reference.height, rendered.height)))
            pair.paste(reference, (0, 0))
            pair.paste(rendered_overlay, (reference.width, 0))
            pair.save(pair_path, quality=92)
            rows.append(
                {
                    "side": side,
                    "sample_id": frame["sample_id"],
                    "dataset": frame["dataset"],
                    "episode_index": frame["episode_index"],
                    "frame_index": frame["frame_index"],
                    "pair": pair_path.name,
                }
            )
    report = output_dir / "matched_replay.json"
    report.write_text(
        json.dumps({"state_order": manifest["dataset_contract"]["state_order"], "pairs": rows}, indent=2) + "\n"
    )
    return report


def run_matched_state_replay(env, manifest_path: Path, output_dir: Path) -> Path:
    manifest = yaml.safe_load(manifest_path.read_text())
    if manifest.get("profile_id") == "bimanual_yam":
        from robodojo.sim.calibration.wrist_camera import load_yam_matched_manifest

        manifest = load_yam_matched_manifest(manifest_path, require_complete=True)
        return _run_yam_matched_replay(env, manifest, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, left_observation in enumerate(manifest["cameras"]["left"]["observations"]):
        state = left_observation["state"]
        # MetaControl deliberately limits a physical gripper command to 20% of
        # travel. Re-issue the same dataset target while the arm controller
        # settles so the captured state is the requested state, not the first
        # interpolation waypoint.
        for _ in range(8):
            env.robot_manager.control_robot([_control_from_manifest(manifest, state)])
            for _ in range(int(env.obs_manager.collect_interval) * 4):
                env.sim_step(render=False)
        observation = env.get_obs_batch(env_idx_list=[0], last_frame=True)[0]
        for side, key in (("left", "cam_left_wrist"), ("right", "cam_right_wrist")):
            source_observation = manifest["cameras"][side]["observations"][index]
            reference_path = output_dir / f"{index:02d}-{side}-upstream.jpg"
            rendered_path = output_dir / f"{index:02d}-{side}-rendered.png"
            pair_path = output_dir / f"{index:02d}-{side}-pair.jpg"
            _fetch_reference(manifest, side, source_observation, reference_path)
            rendered = Image.fromarray(np.asarray(observation["vision"][key]["color"], dtype=np.uint8))
            rendered.save(rendered_path)
            reference = _overlay(Image.open(reference_path).convert("RGB"), source_observation["landmarks"], "#00ff9d")
            pair = Image.new("RGB", (2560, 720))
            pair.paste(reference, (0, 0))
            pair.paste(_overlay(rendered, source_observation["landmarks"], "#ff3b7f"), (1280, 0))
            pair.save(pair_path, quality=92)
            rows.append(
                {
                    "side": side,
                    "global_index": source_observation["global_index"],
                    "pair": pair_path.name,
                    "invalid_pixel_fraction": 0.0,
                }
            )
    report = output_dir / "matched_replay.json"
    report.write_text(json.dumps({"state_order": manifest["dataset"]["state_order"], "pairs": rows}, indent=2) + "\n")
    return report
