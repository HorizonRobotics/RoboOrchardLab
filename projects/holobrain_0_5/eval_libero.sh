#!/usr/bin/env bash
# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

# Evaluate every LIBERO task on every visible GPU.
#
#   MODEL_DIR=./outputs/holobrain_0_5_libero LIBERO_ROOT=./third_party/LIBERO \
#     bash projects/holobrain_0_5/eval_libero.sh
#
# Extra arguments are forwarded to libero_eval, so a narrower sweep is
#   ... eval_libero.sh --task-suites libero_goal --num-trials-per-task 2

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

: "${MODEL_DIR:?set MODEL_DIR to an export root or a checkpoints/checkpoint_<n> dir}"
: "${LIBERO_ROOT:?set LIBERO_ROOT to a LIBERO checkout}"

# LIBERO renders through EGL offscreen; its numba kernels are slower to compile
# than to interpret for this workload.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

exec python -m projects.holobrain_0_5.libero_eval \
  --benchmark libero \
  --libero-root "${LIBERO_ROOT}" \
  --model-dir "${MODEL_DIR}" \
  --processor "${PROCESSOR:-libero_processor}" \
  --processes-per-gpu "${PROCESSES_PER_GPU:-5}" \
  --output "${OUTPUT:-./outputs/libero_eval}" \
  "$@"
