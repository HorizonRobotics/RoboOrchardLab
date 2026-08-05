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

# One RoboDojo Memory task: match_and_pick_from_conveyor.
#
# Why this one out of the six. RoboDojo-only post-training at 100k left the
# Memory dimension at 0.67% overall, and every point of that came from this
# task alone (SR 4.0, see
# projects/holobrain_internal/docs/robodojo_pipeline/07_results.md:202). The
# other five scored 0.0 at both 20k and 100k. A single-task run is meant to
# give the memory the cleanest possible shot and to be readable afterwards;
# on a task that scores 0 either way, "0 -> 0" says nothing about whether
# the memory helped, hurt, or was never consulted. This one has a baseline
# number to move.
#
# Paired with the six-task set in dataset_specs_memoryvla_robodojo_memory.py:
# same data source, same setting_type, so the two differ only in breadth.
#
# Selected via `--kwargs '{"dataset_specs":"configs/dataset_specs_robodojo_conveyor.py"}'`
# passed to train.py. Consumed by `dataset_factory.build_training_dataset` via
# `_load_module_from_ref`, which reads `training_datasets` from this module.
#
# Self-contained by design, matching its two siblings: does NOT import from
# dataset_specs.py, so unrelated dataset specs cannot be dragged in through
# side effects.

import copy
import os
from glob import glob

DATA_BASE = os.environ.get("HOLOBRAIN_DATA_BASE", "./data")

TASK = "match_and_pick_from_conveyor"


def _task_paths() -> list[str]:
    pattern = f"{DATA_BASE}/robodojo/lmdb/{TASK}"
    hits = sorted(glob(pattern))
    if not hits:
        raise FileNotFoundError(
            f"RoboDojo task not found: {pattern}. Check that "
            "HOLOBRAIN_DATA_BASE points at the data root (the `data` symlink "
            "under projects/holobrain_internal/common/)."
        )
    return hits


TRAINING_DATASETS = [
    dict(
        dataset_type="robodojo",
        dataset_name="robodojo",
        data_paths=_task_paths,
        setting_type="arx_x5a",
    ),
]

# No held-out split, matching the six-task set: sim eval success rate is the
# metric that decides anything here.
VALIDATION_DATASETS = None

training_datasets = copy.deepcopy(TRAINING_DATASETS)
training_datasets.sort(key=lambda x: x["dataset_name"])
