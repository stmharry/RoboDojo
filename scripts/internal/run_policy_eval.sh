#!/usr/bin/env bash
# RoboDojo eval orchestration: policy server plus uv-native simulator client.
set -euo pipefail

if [[ $# -lt 10 ]]; then
  echo "usage: bash scripts/internal/run_policy_eval.sh POLICY_DIR DATASET TASK CKPT ENV_CFG [EXPERT_NUM] ACTION_TYPE SEED POLICY_GPU ENV_GPU POLICY_ENV" >&2
  exit 2
fi

policy_dir="$(cd "$1" && pwd)"
shift

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
shift 4

if [[ $# -eq 5 ]]; then
  action_type=$1
  seed=$2
  policy_gpu_id=$3
  env_gpu_id=$4
  policy_env=$5
  shift 5
elif [[ $# -eq 6 ]]; then
  _expert_num=$1
  action_type=$2
  seed=$3
  policy_gpu_id=$4
  env_gpu_id=$5
  policy_env=$6
  shift 6
else
  echo "[run_policy_eval] unexpected trailing argument count: $#" >&2
  exit 2
fi

ROOT_DIR="$(cd "${policy_dir}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
SERVER_SCRIPT="${policy_dir}/setup_eval_policy_server.sh"
DEPLOY_FILE="${policy_dir}/deploy.yml"
CLIENT_SCRIPT="${ROOT_DIR}/scripts/eval_policy.sh"
policy_name="$(basename "${policy_dir}")"

for required in "${SERVER_SCRIPT}" "${DEPLOY_FILE}" "${CLIENT_SCRIPT}"; do
  [[ -f "${required}" ]] || { echo "[run_policy_eval] missing required file: ${required}" >&2; exit 1; }
done

read -r eval_batch protocol < <(python - "${DEPLOY_FILE}" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as stream:
    config = yaml.safe_load(stream) or {}
print(str(config.get("eval_batch", False)).lower(), config.get("protocol", "ws"))
PY
)

policy_server_port="$(bash "${UTILS_DIR}/get_free_port.sh")"
policy_server_ip="localhost"
ckpt_label="${ROBODOJO_CKPT_LABEL:-${ckpt_name}}"
additional_info="ckpt_name=${ckpt_label},action_type=${action_type}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    echo "[MAIN] kill server ${SERVER_PID}"
    kill "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[MAIN] start server, policy_server_port=${policy_server_port}"
(
  cd "${policy_dir}"
  bash setup_eval_policy_server.sh \
    "${bench_name}" \
    "${task_name}" \
    "${ckpt_name}" \
    "${env_cfg_type}" \
    "${action_type}" \
    "${seed}" \
    "${policy_gpu_id}" \
    "${policy_env}" \
    "${policy_server_port}"
) &
SERVER_PID=$!

bash "${UTILS_DIR}/wait_for_policy_server.sh" \
  "${policy_server_ip}" \
  "${policy_server_port}" \
  "${SERVER_PID}" \
  "Policy server" \
  600

echo "[MAIN] start uv-native client, server=${policy_server_ip}:${policy_server_port}"
bash "${CLIENT_SCRIPT}" \
  --root_dir "${ROOT_DIR}" \
  --task_name "${task_name}" \
  --env_cfg_type "${env_cfg_type}" \
  --device_id "${env_gpu_id}" \
  --policy_name "${policy_name}" \
  --host "${policy_server_ip}" \
  --port "${policy_server_port}" \
  --protocol "${protocol}" \
  --eval_batch "${eval_batch}" \
  --additional_info "${additional_info}" \
  --seed "${seed}"

echo "[MAIN] eval finished"
