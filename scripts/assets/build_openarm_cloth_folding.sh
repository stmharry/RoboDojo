#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CACHE_DIR="${ROOT_DIR}/.cache/openarm_cloth_folding"
SOURCE_DIR="${CACHE_DIR}/openarm_isaac_lab"
HARDWARE_DIR="${CACHE_DIR}/hardware"
OUTPUT_DIR="${ROOT_DIR}/Assets/Robots/openarm"
OPENARM_REV="bad82e23716e6941c2de78ccb978f57c78b37734"
HARDWARE_REV="ffe34b93c070343042eb9412fbfeffce16139947"

mkdir -p "${CACHE_DIR}" "${HARDWARE_DIR}" "${OUTPUT_DIR}"
if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  git clone https://github.com/enactic/openarm_isaac_lab.git "${SOURCE_DIR}"
fi
git -C "${SOURCE_DIR}" fetch --depth 1 origin "${OPENARM_REV}"
git -C "${SOURCE_DIR}" checkout --detach "${OPENARM_REV}"

for name in \
  'J3-J4_Cover front extended.stl' \
  'J3-J4_Cover back extended.stl' \
  'jaw_normal.stl' \
  'head camera holder v4.stl' \
  'arducam_holder.stl' \
  'arducam_holder.step'; do
  encoded="${name// /%20}"
  curl -fL --retry 3 \
    "https://huggingface.co/datasets/lerobot/openarms-hardware-modifications/resolve/${HARDWARE_REV}/${encoded}" \
    -o "${HARDWARE_DIR}/${name}"
done

python3 - "${ROOT_DIR}/scripts/assets/openarm_sources.json" "${HARDWARE_DIR}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

spec = json.loads(Path(sys.argv[1]).read_text())["hardware_modifications"]
root = Path(sys.argv[2])
for name, expected in spec["sha256"].items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"checksum mismatch for {name}: {actual} != {expected}")
PY

rm -f "${OUTPUT_DIR}/manifest.json"
OMNI_KIT_ACCEPT_EULA=YES python "${ROOT_DIR}/scripts/assets/build_openarm_cloth_folding.py" \
  --source-root "${SOURCE_DIR}" \
  --hardware-root "${HARDWARE_DIR}" \
  --output-root "${OUTPUT_DIR}" \
  --config-template "${ROOT_DIR}/scripts/assets/openarm_robot_config.yml"
[[ -f "${OUTPUT_DIR}/manifest.json" ]] || {
  echo "OpenARM build failed before writing manifest.json" >&2
  exit 1
}
