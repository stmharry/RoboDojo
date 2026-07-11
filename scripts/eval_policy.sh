#!/usr/bin/env bash
# Private compatibility adapter for unchanged XPolicyLab launchers.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec uv run --project "${ROOT_DIR}" --extra sim --locked robodojo _adapter-client "$@"
