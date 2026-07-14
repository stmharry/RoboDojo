#!/usr/bin/env python3
"""Validate or fit the pinned YAM matched-frame calibration contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from robodojo.sim.calibration.wrist_camera import (
    fit_yam_matched_manifest,
    load_yam_matched_manifest,
    yam_matched_manifest_status,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--camera-config", type=Path)
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    manifest = load_yam_matched_manifest(args.manifest)
    if not args.fit:
        print(json.dumps(yam_matched_manifest_status(manifest), indent=2))
        return
    if args.camera_config is None:
        raise SystemExit("--camera-config is required with --fit")
    camera_config = yaml.safe_load(args.camera_config.read_text(encoding="utf-8"))
    report = fit_yam_matched_manifest(manifest, camera_config).to_dict()
    rendered = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
