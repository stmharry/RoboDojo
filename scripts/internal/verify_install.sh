#!/usr/bin/env bash
# Preflight checks for RoboDojo eval. Does not launch Isaac Sim or a policy server.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

policy_dir=""
env_cfg="arx_x5"
task_name="stack_bowls"
ckpt_name=""
sim_env="RoboDojo"
policy_env=""
skip_isaac="false"
skip_conda="false"
skip_policy="false"
summary_path=""

usage() {
  cat <<'EOF'
Usage: bash scripts/internal/verify_install.sh [options]

Options:
  --policy-dir PATH     Optional XPolicyLab policy directory to validate
  --env-cfg NAME        env_cfg stem to validate (default: arx_x5)
  --task NAME           Task name to validate (default: stack_bowls)
  --ckpt NAME           Optional checkpoint directory under policy checkpoints
  --sim-env NAME        Simulator conda env for Isaac imports (default: RoboDojo)
  --policy-env NAME     Optional policy conda env to check
  --summary PATH        Write JSON summary to PATH
  --skip-isaac          Skip isaacsim/isaaclab import check
  --skip-conda          Skip conda env existence checks
  --skip-policy         Skip policy-dir/deploy/checkpoint checks
  -h, --help            Show this help
EOF
}

need_value() {
  if [[ $# -lt 2 || "$2" == --* ]]; then
    echo "[verify_install] Missing value for $1" >&2
    exit 2
  fi
}

abs_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "${ROOT_DIR}/${path}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-dir) need_value "$@"; policy_dir="$(abs_path "$2")"; shift 2 ;;
    --env-cfg) need_value "$@"; env_cfg="$2"; shift 2 ;;
    --task) need_value "$@"; task_name="$2"; shift 2 ;;
    --ckpt) need_value "$@"; ckpt_name="$2"; shift 2 ;;
    --sim-env) need_value "$@"; sim_env="$2"; shift 2 ;;
    --policy-env) need_value "$@"; policy_env="$2"; shift 2 ;;
    --summary) need_value "$@"; summary_path="$2"; shift 2 ;;
    --skip-isaac) skip_isaac="true"; shift ;;
    --skip-conda) skip_conda="true"; shift ;;
    --skip-policy) skip_policy="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[verify_install] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

RESULTS_FILE="$(mktemp)"
trap 'rm -f "${RESULTS_FILE}"' EXIT

record() {
  local status="$1"
  local name="$2"
  local message="$3"
  printf '[%s] %s - %s\n' "${status}" "${name}" "${message}"
  printf '%s\t%s\t%s\n' "${status}" "${name}" "${message}" >> "${RESULTS_FILE}"
}

check_path() {
  local kind="$1"
  local path="$2"
  local name="$3"
  if [[ "${kind}" == "dir" && -d "${path}" ]]; then
    record "PASS" "${name}" "${path}"
  elif [[ "${kind}" == "file" && -f "${path}" ]]; then
    record "PASS" "${name}" "${path}"
  else
    record "FAIL" "${name}" "missing ${kind}: ${path}"
  fi
}

echo "[verify_install] root=${ROOT_DIR}"
if [[ -n "${policy_dir}" && "${skip_policy}" != "true" ]]; then
  echo "[verify_install] policy_dir=${policy_dir}"
else
  echo "[verify_install] policy checks disabled"
fi

check_path dir "${ROOT_DIR}/env" "env package"
check_path file "${ROOT_DIR}/scripts/robodojo.sh" "robodojo launcher"
check_path file "${ROOT_DIR}/scripts/internal/task_inventory.py" "task inventory helper"
check_path file "${ROOT_DIR}/env_cfg/${env_cfg}.yml" "env_cfg"
check_path file "${ROOT_DIR}/task/RoboDojo/config/${task_name}.yml" "task config"
check_path dir "${ROOT_DIR}/Assets/Robots" "robot assets"
check_path dir "${ROOT_DIR}/Assets/Object/RoboDojo" "object assets"
check_path dir "${ROOT_DIR}/Assets/Eval_Layout/RoboDojo" "eval layouts"
check_path dir "${ROOT_DIR}/Assets/Material" "materials"

if [[ -n "${policy_dir}" && "${skip_policy}" != "true" ]]; then
  check_path file "${policy_dir}/eval.sh" "policy eval.sh"
  check_path file "${policy_dir}/deploy.yml" "policy deploy.yml"
  if [[ -n "${ckpt_name}" ]]; then
    check_path dir "${policy_dir}/checkpoints/${ckpt_name}" "policy checkpoint"
  else
    record "WARN" "policy checkpoint" "skipped; pass --ckpt to validate"
  fi
else
  record "WARN" "policy checks" "skipped; pass --policy-dir to validate a policy"
fi

if python3 "${ROOT_DIR}/scripts/internal/task_inventory.py" --format json --check >/tmp/robodojo_tasks_verify.json; then
  task_counts="$(python3 - <<'PY'
import json
with open('/tmp/robodojo_tasks_verify.json', encoding='utf-8') as f:
    payload = json.load(f)
print(payload['counts'])
PY
)"
  record "PASS" "task inventory" "${task_counts}"
else
  record "FAIL" "task inventory" "task inventory check failed"
fi
rm -f /tmp/robodojo_tasks_verify.json

if python3 - <<PY; then
from pathlib import Path
import sys
import yaml

root = Path("${ROOT_DIR}")
env_cfg = "${env_cfg}"
cfg_path = root / "env_cfg" / f"{env_cfg}.yml"
cfg = yaml.safe_load(cfg_path.read_text()) or {}
refs = cfg.get("config", {})
required = [
    root / "env_cfg" / "sim" / f"{refs.get('sim')}.yml",
    root / "env_cfg" / "scene" / f"{refs.get('scene')}.yml",
    root / "env_cfg" / "robot" / f"{refs.get('robot')}.yml",
    root / "env_cfg" / "camera" / f"{refs.get('camera')}.yml",
    root / "env_cfg" / "robot" / "_robot_info.json",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    print("\\n".join(missing), file=sys.stderr)
    raise SystemExit(1)
PY
  record "PASS" "env_cfg references" "all referenced sim/scene/robot/camera files exist"
else
  record "FAIL" "env_cfg references" "missing referenced env_cfg file"
fi

if [[ "${env_cfg}" == "openarm_cloth_folding" || "${env_cfg}" == "openarm_cloth_folding_dyna" ]]; then
  check_path file "${ROOT_DIR}/Assets/Robots/openarm/manifest.json" "OpenARM asset manifest"
  check_path file "${ROOT_DIR}/Assets/Robots/openarm/openarm_bimanual_cloth_folding.usd" "OpenARM functional-twin USD"
  check_path file "${ROOT_DIR}/Assets/Robots/openarm/robot_config.yml" "OpenARM robot config"
  if python3 - <<PY; then
import json
from pathlib import Path
import sys
import yaml

root = Path("${ROOT_DIR}")
env = yaml.safe_load((root / "env_cfg/openarm_cloth_folding.yml").read_text())
sim = yaml.safe_load((root / "env_cfg/sim/openarm_cloth_folding.yml").read_text())
camera_name = env["config"]["camera"]
camera = yaml.safe_load((root / "env_cfg/camera" / f"{camera_name}.yml").read_text())
scene = yaml.safe_load((root / "env_cfg/scene/openarm_cloth_folding.yml").read_text())
info = json.loads((root / "env_cfg/robot/_robot_info.json").read_text())["dual_openarm"]
manifest = json.loads((root / "Assets/Robots/openarm/manifest.json").read_text())
sources = json.loads((root / "scripts/assets/openarm_sources.json").read_text())
assert env["observation"]["collect_freq"] == 30
assert abs(1.0 / (sim["dt"] * 30) - 8.0) < 1e-9
assert sum(info["arm_dim"]) + sum(info["ee_dim"]) == 16
rig = camera["camera_rig"]
cameras = rig["cameras"]
assert list(cameras) == ["cam_head", "cam_left_wrist", "cam_right_wrist"]
base = cameras["cam_head"]
assert base["type"] == "openarm_base"
assert base["projection"]["model"] == "opencvFisheye"
assert base["mount"]["kind"] == "scene_fixture"
assert base["mount"]["target"] == "camera_stand"
assert base["mount"]["optical_roll_deg"] == 180.0
if camera_name == "openarm_cloth_folding":
    assert rig["profile_id"] == "openarm_policy_original"
    assert base["sensor"]["vendor"] == "Fafeicy"
    assert base["sensor"]["diagonal_fov_deg"] == 140.0
    assert base["projection"]["fx"] == 327.4045
else:
    assert rig["profile_id"] == "openarm_dyna"
    assert base["sensor"]["vendor"] == "Waveshare"
    assert base["sensor"]["diagonal_fov_deg"] == 145.0
    assert base["projection"]["fx"] == 316.1146
for key, roll in (("cam_left_wrist", -90.0), ("cam_right_wrist", 90.0)):
    wrist_camera = cameras[key]
    assert wrist_camera["sensor"]["vendor"] == "Arducam"
    assert wrist_camera["sensor"]["stream_resolution"] == [1280, 720]
    assert wrist_camera["mount"]["optical_roll_deg"] == roll
assert "remove_fixtures" not in scene.get("appearance_overrides", {})
assert manifest["upper_arm_extension_m"] == 0.05
assert len(manifest["joint_paths"]) == 2
assert len(manifest["jaw_paths"]) == 4
assert len(manifest["cover_paths"]) == 4
assert sources["openarm_isaac_lab"]["revision"] == "bad82e23716e6941c2de78ccb978f57c78b37734"
assert sources["hardware_modifications"]["revision"] == "ffe34b93c070343042eb9412fbfeffce16139947"

sys.path.insert(0, str(root))
from env_cfg.camera.template import OPENARM_BASE, OPENARM_WRIST
assert OPENARM_BASE["resolution"] == (640, 480)
assert OPENARM_WRIST["resolution"] == (1280, 720)
PY
    record "PASS" "OpenARM contract" "240/30 Hz, 16-D, three calibrated native camera resolutions, pinned asset manifest"
  else
    record "FAIL" "OpenARM contract" "timing, dimension, or camera contract invalid"
  fi
fi

if [[ -n "${policy_dir}" && "${skip_policy}" != "true" ]]; then
if python3 - <<PY; then
from pathlib import Path
import yaml

policy_dir = Path("${policy_dir}")
cfg = yaml.safe_load((policy_dir / "deploy.yml").read_text()) or {}
if cfg.get("policy_name") == "LeRobot_Pi05_OpenArm":
    expected = "695abe40dbf3aac04efda59c1501d748681fa0fb"
    assert cfg.get("checkpoint_revision") == expected
    assert cfg.get("action_dim") == 16
    assert cfg.get("chunk_size") == 30
    assert cfg.get("prompt") == "Fold the T-shirt properly."
    assert cfg.get("fps") == 30
    assert cfg.get("duration_s") == 2000
    assert cfg.get("interpolation_multiplier") == 3
    assert cfg.get("max_relative_target_deg") == 8.0
    rtc = cfg.get("rtc", {})
    assert rtc == {
        "enabled": True,
        "action_queue_size": 30,
        "execution_horizon": 20,
        "max_guidance_weight": 5.0,
        "prefix_attention_schedule": "LINEAR",
    }
    checkpoint = policy_dir / cfg["checkpoint_path"]
    assert (checkpoint / ".revision").read_text().strip() == expected
    for required in (
        "config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json",
        "policy_preprocessor_step_3_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ):
        assert (checkpoint / required).is_file(), required
processor = cfg.get("processor_path")
if processor:
    path = Path(processor).expanduser()
    if not path.is_absolute():
        path = policy_dir / path
    if not path.exists():
        raise SystemExit(f"missing processor_path: {path}")
PY
  record "PASS" "policy processor" "checkpoint revision and saved processors validated"
else
  record "FAIL" "policy processor" "deploy.yml processor_path is missing"
fi
else
  record "WARN" "policy processor" "skipped; pass --policy-dir to validate"
fi

if python3 - <<PY; then
from env.global_configs import BENCHMARK, ROOT_DIR
assert BENCHMARK == "RoboDojo"
assert ROOT_DIR
PY
  record "PASS" "python import" "env.global_configs imports"
else
  record "FAIL" "python import" "cannot import env.global_configs"
fi

if [[ "${skip_conda}" == "true" ]]; then
  record "WARN" "conda envs" "skipped by --skip-conda"
elif command -v conda >/dev/null 2>&1; then
  envs="$(conda env list | awk 'NF && $1 !~ /^#/ {print $1}')"
  if grep -qx "${sim_env}" <<< "${envs}"; then
    record "PASS" "sim conda env" "${sim_env}"
    if conda run -n "${sim_env}" python -c 'import torch; assert torch.cuda.is_available(); torch.ones(1, device="cuda")' >/dev/null; then
      record "PASS" "sim CUDA" "${sim_env} can execute a CUDA operation"
    else
      record "FAIL" "sim CUDA" "${sim_env} cannot execute a CUDA operation"
    fi
  else
    record "FAIL" "sim conda env" "${sim_env} not found"
  fi
  if [[ -z "${policy_env}" ]]; then
    record "WARN" "policy conda env" "skipped; pass --policy-env to validate"
  elif [[ "${policy_env}" == "uv" || "${policy_env}" == */* ]]; then
    record "WARN" "policy conda env" "policy env is path/uv; conda check skipped"
  elif grep -qx "${policy_env}" <<< "${envs}"; then
    record "PASS" "policy conda env" "${policy_env}"
    if conda run -n "${policy_env}" python -c 'import torch; assert torch.cuda.is_available(); torch.ones(1, device="cuda")' >/dev/null; then
      record "PASS" "policy CUDA" "${policy_env} can execute a CUDA operation"
    else
      record "FAIL" "policy CUDA" "${policy_env} cannot execute a CUDA operation"
    fi
  else
    record "FAIL" "policy conda env" "${policy_env} not found"
  fi
else
  record "FAIL" "conda" "conda command not found"
fi

if [[ "${skip_isaac}" == "true" ]]; then
  record "WARN" "Isaac imports" "skipped by --skip-isaac"
elif command -v conda >/dev/null 2>&1; then
  if conda run -n "${sim_env}" python - <<'PY'; then
import isaacsim  # noqa: F401
import isaaclab  # noqa: F401
PY
    record "PASS" "Isaac imports" "isaacsim and isaaclab import in ${sim_env}"
  else
    record "FAIL" "Isaac imports" "isaacsim/isaaclab import failed in ${sim_env}"
  fi
else
  record "FAIL" "Isaac imports" "conda unavailable; cannot check ${sim_env}"
fi

if [[ -n "${summary_path}" ]]; then
  mkdir -p "$(dirname "${summary_path}")"
  python3 - "${RESULTS_FILE}" "${summary_path}" <<'PY'
import json
from pathlib import Path
import sys

rows = []
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    status, name, message = line.split("\t", 2)
    rows.append({"status": status, "name": name, "message": message})
payload = {
    "counts": {
        "pass": sum(row["status"] == "PASS" for row in rows),
        "warn": sum(row["status"] == "WARN" for row in rows),
        "fail": sum(row["status"] == "FAIL" for row in rows),
    },
    "checks": rows,
}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
  echo "[verify_install] summary=${summary_path}"
fi

fail_count="$(awk -F '\t' '$1 == "FAIL" {count++} END {print count + 0}' "${RESULTS_FILE}")"
warn_count="$(awk -F '\t' '$1 == "WARN" {count++} END {print count + 0}' "${RESULTS_FILE}")"
pass_count="$(awk -F '\t' '$1 == "PASS" {count++} END {print count + 0}' "${RESULTS_FILE}")"
echo "[verify_install] pass=${pass_count} warn=${warn_count} fail=${fail_count}"

if [[ "${fail_count}" -gt 0 ]]; then
  exit 1
fi
