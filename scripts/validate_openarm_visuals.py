#!/usr/bin/env python3
"""Compare a rendered OpenARM rollout with pinned LeRobot reference frames.

The reference frames are a validation oracle only. Camera poses and robot
geometry come from the documented hardware setup and are never optimized by
this command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

try:
    from scripts.assets.openarm_camera_calibration import calibration_manifest
except ModuleNotFoundError:  # direct `python scripts/...` invocation
    from assets.openarm_camera_calibration import calibration_manifest

DATASET = "lerobot-data-collection/level2_final_quality3"
REVISION = "2e1b2e913cd367d74dc4481736954eed4a051ddc"
CAMERAS = {
    "base": {"sim": "cam_head", "resolution": (640, 480)},
    "left_wrist": {"sim": "cam_left_wrist", "resolution": (1280, 720)},
    "right_wrist": {"sim": "cam_right_wrist", "resolution": (1280, 720)},
}
FRAME_INDICES = (0, 10, 30)


def run(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return completed.stdout


def reference_url(camera: str) -> str:
    return (
        f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/videos/"
        f"observation.images.{camera}/chunk-000/file-000.mp4"
    )


def extract_frames(source: str, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"frame_{index:07d}.png" for index in FRAME_INDICES]
    if all(path.is_file() for path in paths):
        return paths
    select = "+".join(f"eq(n\\,{index})" for index in FRAME_INDICES)
    temporary_pattern = output_dir / "extracted_%02d.png"
    run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            source,
            "-vf",
            f"select='{select}'",
            "-vsync",
            "0",
            "-frames:v",
            str(len(FRAME_INDICES)),
            str(temporary_pattern),
        ]
    )
    extracted = sorted(output_dir.glob("extracted_*.png"))
    if len(extracted) != len(paths):
        raise RuntimeError(f"expected {len(paths)} frames from {source}, got {len(extracted)}")
    for src, dst in zip(extracted, paths, strict=True):
        src.replace(dst)
    return paths


def probe(video: Path) -> dict:
    payload = json.loads(
        run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_read_frames",
                "-of",
                "json",
                str(video),
            ]
        )
    )["streams"][0]
    numerator, denominator = (int(value) for value in payload["avg_frame_rate"].split("/"))
    return {
        "width": int(payload["width"]),
        "height": int(payload["height"]),
        "fps": numerator / denominator,
        "frames": int(payload["nb_read_frames"]),
    }


def image_metrics(path: Path) -> dict:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    center = gray[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
    border = max(8, min(height, width) // 20)
    return {
        "p01": float(np.percentile(gray, 1)),
        "p99": float(np.percentile(gray, 99)),
        "std": float(gray.std()),
        "dark_fraction": float(np.mean(gray < 45)),
        "bright_fraction": float(np.mean(gray > 205)),
        "center_dark_fraction": float(np.mean(center < 45)),
        "edge_dark_fraction": {
            "top": float(np.mean(gray[:border] < 70)),
            "bottom": float(np.mean(gray[-border:] < 70)),
            "left": float(np.mean(gray[:, :border] < 70)),
            "right": float(np.mean(gray[:, -border:] < 70)),
        },
        "bright_surface_first_row": float(np.argmax(np.mean(gray > 205, axis=1) > 0.5) / height),
        "bright_surface_last_row": float(
            np.max(np.flatnonzero(np.mean(gray > 205, axis=1) > 0.5), initial=0) / height
        ),
    }


def orientation_failures(camera: str, metrics: dict) -> list[str]:
    """Base RGB gates; wrist orientation is validated from instance masks."""
    failures = []
    if camera == "base":
        if not 0.01 <= metrics["dark_fraction"] < 0.35:
            failures.append("base: garment/robot silhouette coverage is outside the semantic envelope")
        if metrics["bright_fraction"] < 0.35:
            failures.append("base: complete bright working surface is not visible")
        if metrics["bright_surface_first_row"] > 0.3:
            failures.append("base: working surface begins too low; top robot region dominates")
        if not 0.65 < metrics["bright_surface_last_row"] < 0.99:
            failures.append("base: lower working-surface boundary is not visible")
    return failures


def load_hardware_mask(run_dir: Path, sim_name: str, frame: int = 0) -> tuple[np.ndarray, np.ndarray, dict]:
    path = run_dir / "validation_masks" / f"{sim_name}_frame_{frame:07d}.npz"
    if not path.is_file():
        raise RuntimeError(f"missing validation instance mask: {path}")
    payload = np.load(path)
    mask = np.asarray(payload["mask"]).squeeze()
    info = json.loads(str(payload["info"]))
    labels = info.get("idToLabels") or info.get("idToSemantics") or {}
    hardware_ids, holder_ids = [], []
    side = "left" if "left" in sim_name else "right"
    side_token = f"openarm_{side}_"
    for raw_id, label in labels.items():
        text = json.dumps(label, sort_keys=True).lower()
        instance_id = int(raw_id)
        is_side_hardware = side_token in text and any(
            token in text for token in ("cameraholder", "finger", "hand", "link7", "jaw")
        )
        if is_side_hardware:
            hardware_ids.append(instance_id)
        if side_token in text and "cameraholder" in text:
            holder_ids.append(instance_id)
    if not hardware_ids or not holder_ids:
        raise RuntimeError(f"instance metadata lacks wrist hardware labels in {path}: {labels}")
    return np.isin(mask, hardware_ids), np.isin(mask, holder_ids), info


def wrist_mask_failures(camera: str, hardware: np.ndarray, holder: np.ndarray) -> tuple[list[str], dict]:
    height, width = hardware.shape
    border = max(8, min(height, width) // 20)
    edge = {
        "top": float(hardware[:border].mean()),
        "bottom": float(hardware[-border:].mean()),
        "left": float(hardware[:, :border].mean()),
        "right": float(hardware[:, -border:].mean()),
    }
    coverage = float(hardware.mean())
    holder_points = np.argwhere(holder)
    holder_centroid = (
        [float(holder_points[:, 1].mean() / width), float(holder_points[:, 0].mean() / height)]
        if holder_points.size
        else [float("nan"), float("nan")]
    )
    failures = []
    if edge["bottom"] <= 0.05:
        failures.append(f"{camera}: holder/gripper instance mask does not enter from the bottom")
    if coverage >= 0.35:
        failures.append(f"{camera}: holder/gripper instance mask occupies at least 35% of the frame")
    if edge["left"] > 0.45 and edge["right"] > 0.45:
        failures.append(f"{camera}: holder/gripper instance mask spans both lateral edges")
    contact_region = hardware[: height // 2, width // 5 : 4 * width // 5]
    if float(contact_region.mean()) > 0.20:
        failures.append(f"{camera}: insufficient contact-region visibility above the jaw")
    return failures, {
        "coverage": coverage,
        "edge_fraction": edge,
        "contact_region_coverage": float(contact_region.mean()),
        "holder_centroid": holder_centroid,
    }


def cad_frame_failures() -> list[str]:
    """Reject the other valid-looking 90-degree wrist orientations structurally."""
    calibration = calibration_manifest()
    left = np.asarray(calibration["wrist"]["left_optical_frame_matrix"], dtype=float)[:3, :3]
    right = np.asarray(calibration["wrist"]["right_optical_frame_matrix"], dtype=float)[:3, :3]
    failures = []
    for side, rotation in (("left", left), ("right", right)):
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8) or np.linalg.det(rotation) < 0.999999:
            failures.append(f"{side}_wrist: CAD optical frame is not a right-handed rotation")
    if not np.allclose(left[:, 2], right[:, 2], atol=1e-8):
        failures.append("wrists: mirrored holders do not preserve the optical axis")
    if not np.allclose(left[:, :2], -right[:, :2], atol=1e-8):
        failures.append("wrists: asymmetric up/right landmarks do not form a physical mirror")
    root = Path(__file__).resolve().parents[1]
    profiles = []
    for name in ("openarm_cloth_folding", "openarm_cloth_folding_dyna"):
        profiles.append(yaml.safe_load((root / f"env_cfg/camera/{name}.yml").read_text())["camera_rig"])
    for key in ("cam_left_wrist", "cam_right_wrist"):
        if profiles[0]["cameras"][key] != profiles[1]["cameras"][key]:
            failures.append(f"wrists: {key} differs between policy-original and DYNA profiles")
    base = Rotation.from_euler("XYZ", [180.0, 0.0, 90.0], degrees=True)
    target = base * Rotation.from_euler("Z", 180.0, degrees=True)
    for camera, legacy_roll, expected_visual in (("left", -90.0, 90.0), ("right", 90.0, -90.0)):
        legacy = base * Rotation.from_euler("Z", legacy_roll, degrees=True)
        frame_delta = (legacy.inv() * target).as_rotvec(degrees=True)
        if not np.allclose(frame_delta, [0.0, 0.0, -expected_visual], atol=1e-8):
            failures.append(f"{camera}_wrist: rendered correction is not {expected_visual:+g} degrees")
    return failures


def histogram_distance(left: Path, right: Path) -> float:
    def histogram(path: Path):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        hist = cv2.calcHist([lab], [0, 1], None, [24, 24], [0, 256, 0, 256]).reshape(-1)
        return hist / max(float(hist.sum()), 1.0)

    a, b = histogram(left), histogram(right)
    midpoint = 0.5 * (a + b)

    def kl(p, q):
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / np.maximum(q[mask], 1e-12))))

    return 0.5 * kl(a, midpoint) + 0.5 * kl(b, midpoint)


def contact_sheet(reference: list[Path], rendered: list[Path], output: Path, label: str) -> None:
    rows = []
    for title, paths in (("official validation frames", reference), ("rendered rollout", rendered)):
        images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
        target_height = 240
        images = [
            cv2.resize(image, (round(image.shape[1] * target_height / image.shape[0]), target_height))
            for image in images
        ]
        row = cv2.hconcat(images)
        cv2.putText(row, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 180, 255), 2)
        rows.append(row)
    width = max(row.shape[1] for row in rows)
    rows = [cv2.copyMakeBorder(row, 0, 0, 0, width - row.shape[1], cv2.BORDER_CONSTANT) for row in rows]
    sheet = cv2.vconcat(rows)
    cv2.putText(sheet, label, (12, sheet.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 180, 255), 2)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet)


def find_video(run_dir: Path, sim_name: str) -> Path:
    matches = sorted(run_dir.glob(f"episode_0000000_{sim_name}_*.mp4"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {sim_name} video in {run_dir}, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/openarm_visual_reference"))
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--allow-partial", action="store_true", help="accept non-501-frame zero-action smoke videos")
    parser.add_argument("--profile-id", required=True, choices=("openarm_policy_original", "openarm_dyna"))
    parser.add_argument("--mask-run-dir", type=Path, help="zero-action run containing validation instance masks")
    args = parser.parse_args()
    report_dir = args.report_dir or args.run_dir / "visual_validation"
    cache_root = args.cache_dir / REVISION
    report: dict = {
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "role": "validation_only_not_geometry_or_camera_authority",
        "camera_profile": args.profile_id,
        "cameras": {},
        "observations": [],
        "passed": True,
        "failures": [],
    }
    report["failures"].extend(cad_frame_failures())

    for camera, config in CAMERAS.items():
        video = find_video(args.run_dir, config["sim"])
        video_info = probe(video)
        expected_width, expected_height = config["resolution"]
        if (video_info["width"], video_info["height"]) != (expected_width, expected_height):
            report["failures"].append(f"{camera}: wrong resolution {video_info}")
        if abs(video_info["fps"] - 30.0) > 1e-6:
            report["failures"].append(f"{camera}: fps is {video_info['fps']}, expected 30")
        if not args.allow_partial and video_info["frames"] != 501:
            report["failures"].append(f"{camera}: frame count is {video_info['frames']}, expected 501")

        reference = extract_frames(reference_url(camera), cache_root / camera)
        rendered = extract_frames(str(video), report_dir / "frames" / camera)
        reference_metrics = [image_metrics(path) for path in reference]
        rendered_metrics = [image_metrics(path) for path in rendered]
        distances = [histogram_distance(ref, sim) for ref, sim in zip(reference, rendered, strict=True)]

        for index, metrics in zip(FRAME_INDICES, rendered_metrics, strict=True):
            if metrics["std"] < 12.0:
                report["failures"].append(f"{camera} frame {index}: image is effectively blank")
            if metrics["dark_fraction"] < 0.01 or metrics["bright_fraction"] < 0.01:
                report["failures"].append(
                    f"{camera} frame {index}: lacks both dark garment/hardware and bright workspace"
                )
        report["failures"].extend(orientation_failures(camera, rendered_metrics[0]))
        # Distribution distance is diagnostic only. Making it a pass/fail
        # target would turn the episode into a camera-pose fitting oracle,
        # contrary to the protocol's authority hierarchy.
        if max(distances) > 0.82:
            report["observations"].append(
                f"{camera}: validation-only color/layout distance is {max(distances):.3f}"
            )

        contact_sheet(reference, rendered, report_dir / f"{camera}_comparison.png", camera)
        report["cameras"][camera] = {
            "video": str(video),
            "video_info": video_info,
            "reference_url": reference_url(camera),
            "reference_metrics": reference_metrics,
            "rendered_metrics": rendered_metrics,
            "histogram_js_distance": distances,
        }

    if args.mask_run_dir:
        wrist_masks = {}
        for camera in ("left_wrist", "right_wrist"):
            sim_name = CAMERAS[camera]["sim"]
            hardware, holder, _ = load_hardware_mask(args.mask_run_dir, sim_name)
            failures, metrics = wrist_mask_failures(camera, hardware, holder)
            report["failures"].extend(failures)
            wrist_masks[camera] = metrics
        left_x = wrist_masks["left_wrist"]["holder_centroid"][0]
        right_x = wrist_masks["right_wrist"]["holder_centroid"][0]
        if not np.isfinite(left_x + right_x) or abs((left_x + right_x) - 1.0) > 0.20:
            report["failures"].append("wrists: asymmetric holder landmarks are not physical mirrors")
        report["instance_mask_validation"] = {
            "source_run": str(args.mask_run_dir),
            "cameras": wrist_masks,
        }
    else:
        report["failures"].append("wrists: validation instance-mask run is required")

    report["passed"] = not report["failures"]
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "visual_validation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(report_path)
    if not report["passed"]:
        raise SystemExit("visual validation failed: " + "; ".join(report["failures"]))


if __name__ == "__main__":
    main()
