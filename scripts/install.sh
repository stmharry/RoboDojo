#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
UV_VERSION="${UV_VERSION:-0.11.21}"

info()  { echo -e "\e[1;32m>>> $*\e[0m"; }
warn()  { echo -e "\e[1;33m>>> $*\e[0m"; }
error() { echo -e "\e[1;31m[ERROR] $*\e[0m"; exit 1; }

setup_system() {
  local missing=()
  for pkg in cmake build-essential ffmpeg; do
    dpkg -s "${pkg}" &>/dev/null || missing+=("${pkg}")
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    warn "[1/4] System dependencies already installed, skipping."
    return
  fi

  info "[1/4] Installing system dependencies: ${missing[*]}"
  if [[ "$(id -u)" -eq 0 ]]; then
    apt-get update
    apt-get install -y "${missing[@]}"
  else
    sudo apt-get update
    sudo apt-get install -y "${missing[@]}"
  fi
}

setup_uv() {
  export PATH="${HOME}/.local/bin:${PATH}"
  if command -v uv &>/dev/null && [[ "$(uv --version)" == "uv ${UV_VERSION}" ]]; then
    warn "[2/4] uv ${UV_VERSION} already installed, skipping."
  else
    info "[2/4] Installing uv ${UV_VERSION}"
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
    command -v uv &>/dev/null || error "uv installation did not place uv on PATH"
  fi
  info "    Provisioning managed Python 3.11"
  uv python install 3.11
}

setup_submodules() {
  info "[3/4] Initializing pinned submodules"
  git -C "${ROOT_DIR}" submodule sync --recursive
  git -C "${ROOT_DIR}" submodule update --init --recursive --progress
  [[ -f "${ROOT_DIR}/third_party/IsaacLab/isaaclab.sh" ]] || error "IsaacLab submodule is unavailable"
  [[ -f "${ROOT_DIR}/third_party/curobo/pyproject.toml" ]] || error "CuRobo submodule is unavailable"
  [[ -f "${ROOT_DIR}/XPolicyLab/client_server/ws/model_client.py" ]] || error "XPolicyLab submodule is unavailable"
}

setup_sync() {
  export PATH="${HOME}/.local/bin:${PATH}"
  export OMNI_KIT_ACCEPT_EULA=YES
  info "[4/4] Syncing the locked RoboDojo environment"
  uv sync --directory "${ROOT_DIR}" --locked
  uv lock --directory "${ROOT_DIR}" --check
}

usage() {
  cat <<'EOF'
Usage: scripts/install.sh --install | --from STEP

  -i, --install   Install system dependencies, uv, submodules, and .venv
  --from STEP     Resume at: system | uv | submodules | sync
  -h, --help      Show this help

UV_VERSION may override the pinned uv bootstrap version for testing.
EOF
}

run_from() {
  local from="$1"
  local steps=(system uv submodules sync)
  local start=-1
  local i
  for i in "${!steps[@]}"; do
    if [[ "${steps[$i]}" == "${from}" ]]; then
      start="${i}"
      break
    fi
  done
  [[ "${start}" -ge 0 ]] || error "Unknown step '${from}'. Valid: ${steps[*]}"
  for i in "${!steps[@]}"; do
    if [[ "${i}" -ge "${start}" ]]; then
      "setup_${steps[$i]}"
    fi
  done
}

case "${1:-}" in
  -i|--install)
    run_from system
    ;;
  --from)
    [[ -n "${2:-}" ]] || error "--from requires a step"
    run_from "$2"
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage
    exit 2
    ;;
esac

info "RoboDojo environment ready. Run commands with: uv run --locked <command>"
