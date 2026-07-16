"""YAM asset-builder command facade."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from robodojo.workflows.asset_builders.yam.publication import build

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.source_root, args.output_root, args.manifest)
    logger.info("Built YAM asset:\n%s", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
