#!/usr/bin/env python3
"""Fit and verify LeRobot wrist calibration from pinned dataset landmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import urllib.request

from robodojo.sim.calibration.wrist_camera import (
    fit_manifest,
    fit_metrics,
    held_out_geometry_metrics,
    load_manifest,
    validate_frame,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs/reference/openarm_lerobot_wrist_calibration.yml"


def fetch_frames(manifest: dict, destination: Path) -> None:
    source = manifest["dataset"]
    for side, camera in manifest["cameras"].items():
        video_path = source["videos"][side]
        url = f"{source['repository']}/resolve/{source['revision']}/{video_path}"
        video = destination / f"{side}.mp4"
        urllib.request.urlretrieve(url, video)
        for observation in camera["observations"]:
            output = destination / observation["frame_file"]
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
            validate_frame(output, observation["sha256"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--fetch", action="store_true", help="fetch and checksum pinned source frames")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    if args.fetch:
        with tempfile.TemporaryDirectory(prefix="openarm-wrist-calibration-") as directory:
            fetch_frames(manifest, Path(directory))
    fits = fit_manifest(manifest)
    report = {
        "dataset_revision": manifest["dataset"]["revision"],
        "holder_audit": manifest["holder_audit"],
        "cameras": {
            side: {
                "position": fit.position.tolist(),
                "orientation": fit.orientation.tolist(),
                "fx_fy_cx_cy": fit.intrinsics.tolist(),
                "distortion": fit.distortion.tolist(),
                **fit_metrics(fit),
                **held_out_geometry_metrics(manifest["cameras"][side], fit),
            }
            for side, fit in fits.items()
        },
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
