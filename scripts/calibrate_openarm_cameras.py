#!/usr/bin/env python3
"""Produce reproducible CAD/blog/matched-state OpenARM calibration evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.request

import cv2
import numpy as np
import pyarrow.parquet as pq

from scripts.assets.openarm_camera_calibration import (
    BLOG_SPACE_REVISION,
    HARDWARE_REVISION,
    calibration_manifest,
)
from scripts.validate_openarm_visuals import CAMERAS, FRAME_INDICES, extract_frames, find_video, reference_url

DATASET_REVISION = "2e1b2e913cd367d74dc4481736954eed4a051ddc"
DATASET = "lerobot-data-collection/level2_final_quality3"
BLOG_IMAGES = {
    "base": "cam_base.jpg",
    "left_wrist": "cam_left_wrist.jpg",
    "right_wrist": "cam_right_wrist.jpg",
}


def fetch(url: str, path: Path) -> Path:
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(path)
    return path


def blog_url(filename: str) -> str:
    return (
        "https://huggingface.co/spaces/lerobot/robot-folding/resolve/"
        f"{BLOG_SPACE_REVISION}/app/public/images/folding/{filename}"
    )


def parquet_url() -> str:
    return (
        f"https://huggingface.co/datasets/{DATASET}/resolve/{DATASET_REVISION}/"
        "data/chunk-000/file-000.parquet"
    )


def load_states(path: Path) -> list[dict]:
    table = pq.read_table(path, columns=["observation.state", "frame_index", "episode_index"])
    result = []
    for index in FRAME_INDICES:
        state = table["observation.state"][index].as_py()
        if len(state) != 16 or not np.isfinite(state).all():
            raise ValueError(f"dataset frame {index} has invalid OpenARM state")
        result.append({"frame": index, "episode": table["episode_index"][index].as_py(), "state": state})
    return result


def resized(path: Path, height: int = 240) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read {path}")
    return cv2.resize(image, (round(image.shape[1] * height / image.shape[0]), height))


def sheet(rows: list[tuple[str, list[Path]]], output: Path) -> None:
    rendered = []
    for label, paths in rows:
        images = [resized(path) for path in paths]
        row = cv2.hconcat(images)
        cv2.putText(row, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)
        rendered.append(row)
    width = max(row.shape[1] for row in rendered)
    rendered = [cv2.copyMakeBorder(row, 0, 0, 0, width - row.shape[1], cv2.BORDER_CONSTANT) for row in rendered]
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.vconcat(rendered))


def anchor_diagram(output: Path) -> None:
    manifest = calibration_manifest()
    canvas = np.full((940, 1600, 3), 248, dtype=np.uint8)
    cv2.putText(
        canvas,
        "OpenARM camera holders: CAD-derived frames",
        (35, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (25, 25, 25),
        2,
    )

    def matrix_lines(name: str, matrix) -> list[str]:
        rows = np.asarray(matrix, dtype=float)
        return [name] + ["  [" + "  ".join(f"{value: .6f}" for value in row) + "]" for row in rows]

    panels = (
        (
            "HEAD / head camera holder v4.stl",
            [
                "parent: upstream camera_stand terminal plate (setup-dependent)",
                f"mount origin CAD mm: {manifest['head']['mount_origin_mm']}",
                f"aperture origin CAD mm: {manifest['head']['optical_origin_mm']}",
                *matrix_lines("MountFrame -> OpticalFrame", manifest["head"]["optical_frame_matrix"]),
            ],
            45,
            105,
        ),
        (
            "LEFT WRIST / arducam_holder.step + .stl",
            [
                "parent: OpenARM left hand mounting plate (CAD/mechanical)",
                *matrix_lines("MountFrame -> OpticalFrame", manifest["wrist"]["left_optical_frame_matrix"]),
                "effective correction from legacy: +90 deg; runtime roll: 0 deg",
            ],
            45,
            500,
        ),
        (
            "RIGHT WRIST / physically mirrored holder",
            [
                "parent: OpenARM right hand mounting plate (CAD/mechanical)",
                *matrix_lines("MountFrame -> OpticalFrame", manifest["wrist"]["right_optical_frame_matrix"]),
                "effective correction from legacy: -90 deg; runtime roll: 0 deg",
            ],
            810,
            500,
        ),
    )
    for title, lines, x, y in panels:
        cv2.putText(canvas, title, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 80, 180), 2)
        for row, line in enumerate(lines, start=1):
            cv2.putText(canvas, line, (x + 15, y + 35 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (35, 35, 35), 1)
    # Explicit axis legend matches the authored USD camera convention.
    cv2.arrowedLine(canvas, (1350, 270), (1460, 270), (40, 40, 210), 4)
    cv2.arrowedLine(canvas, (1350, 270), (1350, 160), (40, 170, 40), 4)
    cv2.arrowedLine(canvas, (1350, 270), (1270, 345), (210, 70, 30), 4)
    cv2.putText(canvas, "+X right", (1465, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 210), 1)
    cv2.putText(canvas, "+Y up", (1355, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 170, 40), 1)
    cv2.putText(canvas, "-Z look", (1170, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 70, 30), 1)
    cv2.imwrite(str(output), canvas)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="zero-action or rollout directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/openarm_camera_calibration"))
    parser.add_argument("--previous-run-dir", type=Path, help="optional pre-fix DYNA run for before/after sheets")
    args = parser.parse_args()
    cache = args.cache_dir
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    blog_paths = {
        camera: fetch(blog_url(filename), cache / BLOG_SPACE_REVISION / "blog" / filename)
        for camera, filename in BLOG_IMAGES.items()
    }
    parquet = fetch(parquet_url(), cache / DATASET_REVISION / "file-000.parquet")
    states = load_states(parquet)
    (output / "matched_states.json").write_text(json.dumps(states, indent=2) + "\n")
    state_sequence = {str(item["frame"]): item["state"] for item in states}
    (output / "matched_state_environment.json").write_text(json.dumps(state_sequence) + "\n")
    anchor_diagram(output / "cad_anchor_diagram.png")

    cameras = {}
    for camera, config in CAMERAS.items():
        video = find_video(args.run_dir, config["sim"])
        sim_frames = extract_frames(str(video), output / "rendered" / camera)
        previous_frames = None
        if args.previous_run_dir:
            previous_video = find_video(args.previous_run_dir, config["sim"])
            previous_frames = extract_frames(str(previous_video), output / "previous" / camera)
        dataset_frames = extract_frames(reference_url(camera), cache / DATASET_REVISION / camera)
        sheet(
            [("official article envelope", [blog_paths[camera]]), ("CAD-derived render", [sim_frames[0]])],
            output / f"{camera}_blog_contact_sheet.png",
        )
        if previous_frames:
            sheet(
                [("previous DYNA frame", [previous_frames[0]]), ("CAD-frame DYNA frame", [sim_frames[0]])],
                output / f"{camera}_before_after_contact_sheet.png",
            )
        sheet(
            [("dataset states 0 / 10 / 30", dataset_frames), ("simulation captures 0 / 10 / 30", sim_frames)],
            output / f"{camera}_matched_state_contact_sheet.png",
        )
        cameras[camera] = {
            "blog_url": blog_url(BLOG_IMAGES[camera]),
            "dataset_video_url": reference_url(camera),
            "render_video": str(video),
            "previous_render_video": str(previous_video) if args.previous_run_dir else None,
        }

    report = {
        "authority": "CAD_and_pinned_article; imagery_is_validation_only",
        "article_revision": BLOG_SPACE_REVISION,
        "hardware_revision": HARDWARE_REVISION,
        "dataset_revision": DATASET_REVISION,
        "calibration": calibration_manifest(),
        "provenance": {
            "cad_derived": ["holder mesh frame", "mount holes", "aperture center", "optical forward/up/right"],
            "setup_dependent_nominal": ["upstream RoboDojo camera-stand world pose"],
            "not_fitted": ["focal length", "principal point", "robot geometry", "camera extrinsics"],
        },
        "matched_states": states,
        "cameras": cameras,
        "matched_state_capture": {
            "environment_variable": "ROBODOJO_OPENARM_CALIBRATION_STATES",
            "value_file": str(output / "matched_state_environment.json"),
            "note": (
                "The evaluator writes states 0, 10, and 30 immediately before those observations; "
                "no image fitting is performed."
            ),
        },
    }
    report_path = output / "camera_calibration.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(report_path)


if __name__ == "__main__":
    main()
