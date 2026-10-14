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

# Evaluate every RoboTwin task the checkout defines, on every visible GPU.
#
#   MODEL_DIR=./outputs/holobrain_0_5_robotwin ROBOTWIN_ROOT=./third_party/RoboTwin \
#     bash projects/holobrain_0_5/eval_robotwin.sh
#
# For the UR5-WSG body, override both the processor and the task config:
#   PROCESSOR=robotwin2_0_ur5_wsg_processor TASK_CONFIG=demo_clean_ur5-wsg ...
#
# Extra arguments are forwarded to robotwin_eval, e.g.
#   ... eval_robotwin.sh --tasks place_empty_cup --episodes 2

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

: "${MODEL_DIR:?set MODEL_DIR to an export root or a checkpoints/checkpoint_<n> dir}"
: "${ROBOTWIN_ROOT:?set ROBOTWIN_ROOT to a RoboTwin checkout}"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

exec python -m projects.holobrain_0_5.robotwin_eval \
  --robotwin-root "${ROBOTWIN_ROOT}" \
  --model-dir "${MODEL_DIR}" \
  --processor "${PROCESSOR:-robotwin2_0_processor}" \
  --task-config "${TASK_CONFIG:-demo_clean}" \
  --episodes "${EPISODES:-50}" \
  --processes-per-gpu "${PROCESSES_PER_GPU:-2}" \
  --output "${OUTPUT:-./outputs/robotwin_eval}" \
  "$@"
