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

# Evaluate every LIBERO-Plus variation on every visible GPU.
#
#   MODEL_DIR=./outputs/holobrain_0_5_libero \
#     LIBERO_PLUS_ROOT=./third_party/LIBERO-plus \
#     bash projects/holobrain_0_5/eval_libero_plus.sh
#
# The benchmark already varies each task, so it defaults to one trial each;
# extra arguments are forwarded to libero_eval.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

: "${MODEL_DIR:?set MODEL_DIR to an export root or a checkpoints/checkpoint_<n> dir}"
: "${LIBERO_PLUS_ROOT:?set LIBERO_PLUS_ROOT to a LIBERO-plus checkout}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

exec python -m projects.holobrain_0_5.libero_eval \
  --benchmark libero_plus \
  --libero-plus-root "${LIBERO_PLUS_ROOT}" \
  --model-dir "${MODEL_DIR}" \
  --processor "${PROCESSOR:-libero_processor}" \
  --processes-per-gpu "${PROCESSES_PER_GPU:-5}" \
  --output "${OUTPUT:-./outputs/libero_plus_eval}" \
  "$@"
