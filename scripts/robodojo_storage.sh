#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV_BIN:-$(command -v uv 2>/dev/null || true)}"
if [[ -z "${UV_BIN}" && -x "${HOME}/.local/bin/uv" ]]; then
  UV_BIN="${HOME}/.local/bin/uv"
fi
if [[ -z "${UV_BIN}" ]]; then
  echo "robodojo_storage.sh requires uv; install it or set UV_BIN" >&2
  exit 127
fi
exec "${UV_BIN}" run --project "${ROOT_DIR}" --frozen --no-sync \
  python "${ROOT_DIR}/scripts/internal/storage_cli.py" "$@"
