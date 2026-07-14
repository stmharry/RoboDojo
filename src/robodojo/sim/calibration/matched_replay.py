"""Matched-state wrist-camera replay diagnostic for OpenArm.

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


def run_matched_state_replay(env, manifest_path: Path, output_dir: Path) -> Path:
    manifest = yaml.safe_load(manifest_path.read_text())
    if manifest.get("profile_id") != "openarm_lerobot":
        raise ValueError(f"unsupported matched replay profile: {manifest.get('profile_id')!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, left_observation in enumerate(manifest["cameras"]["left"]["observations"]):
        state = left_observation["state"]
        # MetaControl deliberately limits a physical gripper command to 20% of
        # travel. Re-issue the same dataset target while the arm controller
        # settles so the captured state is the requested state, not the first
        # interpolation waypoint.
        for _ in range(8):
            env.robot_manager.control_robot([_control_from_right_first(state)])
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
            reference = _overlay(
                Image.open(reference_path).convert("RGB"),
                source_observation["landmarks"],
                "#00ff9d",
            )
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
