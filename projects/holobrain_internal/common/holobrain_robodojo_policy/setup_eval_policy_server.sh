#!/usr/bin/env bash
set -euo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
policy_gpu_id=${7}
policy_conda_env=${8}
policy_server_port=${9}
policy_server_host=${10:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"

if [[ "${policy_conda_env}" == */* ]]; then
    activate_script="${policy_conda_env%/}/bin/activate"
    if [[ ! -f "${activate_script}" ]]; then
        echo "[holobrain] virtualenv activate script not found: ${activate_script}" >&2
        exit 1
    fi
    source "${activate_script}"
else
    if ! command -v conda >/dev/null 2>&1; then
        conda_bin="${CONDA_EXE:-${HOME}/miniconda3/bin/conda}"
        if [[ ! -x "${conda_bin}" ]]; then
            echo "[holobrain] conda command not found" >&2
            exit 1
        fi
        export PATH="$(dirname "${conda_bin}"):${PATH}"
    fi
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${policy_conda_env}"
fi

python_path="${BENCH_ROOT}:${XPL_ROOT}:${PYTHONPATH:-}"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONPATH="${python_path}" \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    python "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}" \
            action_dim=14
