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

# ruff: noqa: E501

# RoboDojo Memory-dimension tasks only -- the smoke/ablation set for the
# MemoryVLA port.
#
# Why these six: of RoboDojo's five dimensions, Memory is the one HoloBrain is
# worst at. RoboDojo-only post-training moved it from 0.00% to 0.67% between
# 20k and 100k steps, and only one of the six tasks scored at all
# (match_and_pick_from_conveyor, 4% at 100k). See
# projects/holobrain_internal/docs/robodojo_pipeline/07_results.md:197-206 and
# section 6.3. That is precisely the capability MemoryVLA targets, so it is
# the honest place to smoke-test the port.
#
# Selected via `--kwargs '{"dataset_specs":"configs/dataset_specs_memoryvla_robodojo_memory.py"}'`
# passed to train.py. Consumed by `dataset_factory.build_training_dataset` via
# `_load_module_from_ref`, which reads `training_datasets` from this module.
#
# Self-contained by design, matching dataset_specs_robodojo.py: does NOT
# import from dataset_specs.py, so unrelated dataset specs cannot be dragged
# in through side effects.

import copy
import os
from glob import glob

DATA_BASE = os.environ.get("HOLOBRAIN_DATA_BASE", "./data")

# Measured 2026-08-03: 600 episodes / 328,975 frames in total. Episodes are
# long -- median length runs from 276 (swap_T) to 1203
# (imitate_sorting_sequence) frames. That matters for the smoke run: with
# dataloader_type="stream" the bank only exercises its episode-clearing path
# once a run crosses an episode boundary, which at batch 16 takes ~18 steps on
# swap_T and ~76 on imitate_sorting_sequence.
MEMORY_TASKS = [
    "cover_blocks",
    "match_and_pick_from_conveyor",
    "swap_blocks",
    "swap_T",
    "press_by_number",
    "imitate_sorting_sequence",
]


def _memory_task_paths() -> list[str]:
    paths = []
    for task in MEMORY_TASKS:
        pattern = f"{DATA_BASE}/robodojo/lmdb/{task}"
        hits = sorted(glob(pattern))
        if not hits:
            raise FileNotFoundError(
                f"RoboDojo Memory task not found: {pattern}. Check that "
                "HOLOBRAIN_DATA_BASE points at the data root (the `data` "
                "symlink under projects/holobrain_internal/common/)."
            )
        paths.extend(hits)
    return paths


TRAINING_DATASETS = [
    dict(
        dataset_type="robodojo",
        dataset_name="robodojo",
        data_paths=_memory_task_paths,
        setting_type="arx_x5a",
    ),
]

# No held-out split: this set exists to exercise the memory path, not to
# measure generalisation.
VALIDATION_DATASETS = None

training_datasets = copy.deepcopy(TRAINING_DATASETS)
training_datasets.sort(key=lambda x: x["dataset_name"])
