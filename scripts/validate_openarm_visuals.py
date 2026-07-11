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
    return {
        "p01": float(np.percentile(gray, 1)),
        "p99": float(np.percentile(gray, 99)),
        "std": float(gray.std()),
        "dark_fraction": float(np.mean(gray < 45)),
        "bright_fraction": float(np.mean(gray > 205)),
        "center_dark_fraction": float(np.mean(center < 45)),
    }


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
        images = [cv2.resize(image, (round(image.shape[1] * target_height / image.shape[0]), target_height)) for image in images]
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
    args = parser.parse_args()
    report_dir = args.report_dir or args.run_dir / "visual_validation"
    cache_root = args.cache_dir / REVISION
    report: dict = {
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "role": "validation_only_not_geometry_or_camera_authority",
        "cameras": {},
        "observations": [],
        "passed": True,
        "failures": [],
    }

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
                report["failures"].append(f"{camera} frame {index}: lacks both dark garment/hardware and bright workspace")
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

    report["passed"] = not report["failures"]
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "visual_validation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(report_path)
    if not report["passed"]:
        raise SystemExit("visual validation failed: " + "; ".join(report["failures"]))


if __name__ == "__main__":
    main()
