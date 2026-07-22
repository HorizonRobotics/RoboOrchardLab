#!/usr/bin/env bash
set -euo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
policy_gpu_id=${7}
env_gpu_id=${8}
policy_conda_env=${9}
eval_env_conda_env=${10}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
SERVER_SCRIPT="${SCRIPT_DIR}/setup_eval_policy_server.sh"
CLIENT_SCRIPT="${SCRIPT_DIR}/setup_eval_env_client.sh"

policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip="localhost"
additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"

_kill_process_tree() {
    local pid=$1
    local sig=${2:-TERM}
    local child
    while read -r child; do
        [[ -n "${child}" ]] || continue
        _kill_process_tree "${child}" "${sig}"
    done < <(pgrep -P "${pid}" 2>/dev/null || true)
    kill "-${sig}" "${pid}" 2>/dev/null || true
}

cleanup() {
    trap '' EXIT INT TERM
    if [[ -n "${SERVER_PID:-}" ]]; then
        _kill_process_tree "${SERVER_PID}" TERM
        for _ in 1 2 3 4 5; do
            if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
                SERVER_PID=""
                return 0
            fi
            sleep 0.2
        done
        _kill_process_tree "${SERVER_PID}" KILL
        SERVER_PID=""
    fi
}
trap cleanup EXIT INT TERM

(
    cd "${SCRIPT_DIR}"
    exec bash "${SERVER_SCRIPT}" \
        "${bench_name}" "${task_name}" "${ckpt_name}" \
        "${env_cfg_type}" "${action_type}" "${seed}" \
        "${policy_gpu_id}" "${policy_conda_env}" \
        "${policy_server_port}"
) &
SERVER_PID=$!

bash "${UTILS_DIR}/wait_for_policy_server.sh" \
    "${policy_server_ip}" "${policy_server_port}" "${SERVER_PID}" \
    "Policy server" 1200

bash "${CLIENT_SCRIPT}" \
    "${bench_name}" "${task_name}" "${ckpt_name}" \
    "${env_cfg_type}" "${action_type}" "${seed}" "${env_gpu_id}" \
    "${eval_env_conda_env}" "${additional_info}" \
    "${policy_server_port}" "${policy_server_ip}"
