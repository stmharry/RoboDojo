#!/usr/bin/env python3
"""Render an asymmetric optical-roll harness for the three OpenARM streams."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


def asymmetric_target(width: int, height: int) -> np.ndarray:
    image = np.full((height, width, 3), 238, dtype=np.uint8)
    cv2.rectangle(image, (15, 15), (width // 3, height // 4), (20, 45, 210), -1)
    cv2.circle(image, (width - width // 7, height // 5), max(12, height // 14), (25, 185, 45), -1)
    triangle = np.array([[width // 8, height - 18], [width // 3, height - 18], [width // 8, height * 3 // 5]])
    cv2.fillPoly(image, [triangle], (210, 70, 25))
    cv2.putText(image, "TOP", (width // 2 - 45, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
    cv2.putText(image, "L", (18, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (10, 10, 10), 3)
    cv2.putText(image, "R", (width - 55, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (10, 10, 10), 3)
    return image


def roll_landscape(image: np.ndarray, degrees: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), borderValue=(0, 0, 0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="openarm_cloth_folding")
    parser.add_argument("--output-dir", type=Path, default=Path(".cache/openarm_orientation_harness"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "env_cfg/camera" / f"{args.profile}.yml").read_text())
    cameras = config["camera_rig"]["cameras"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for key, camera in cameras.items():
        width, height = camera["sensor"]["stream_resolution"]
        image = roll_landscape(asymmetric_target(width, height), camera["mount"]["optical_roll_deg"])
        path = args.output_dir / f"{key}.png"
        cv2.imwrite(str(path), image)
        rendered.append(cv2.resize(image, (640, 360)))
    cv2.imwrite(str(args.output_dir / "orientation_contact_sheet.png"), cv2.hconcat(rendered))
    print(args.output_dir / "orientation_contact_sheet.png")


if __name__ == "__main__":
    main()
