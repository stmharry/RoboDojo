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

for name in 'J3-J4_Cover front extended.stl' 'J3-J4_Cover back extended.stl' 'jaw_normal.stl'; do
  encoded="${name// /%20}"
  curl -fL --retry 3 \
    "https://huggingface.co/datasets/lerobot/openarms-hardware-modifications/resolve/${HARDWARE_REV}/${encoded}" \
    -o "${HARDWARE_DIR}/${name}"
done

OMNI_KIT_ACCEPT_EULA=YES python "${ROOT_DIR}/scripts/assets/build_openarm_cloth_folding.py" \
  --source-root "${SOURCE_DIR}" \
  --hardware-root "${HARDWARE_DIR}" \
  --output-root "${OUTPUT_DIR}" \
  --config-template "${ROOT_DIR}/scripts/assets/openarm_robot_config.yml"
