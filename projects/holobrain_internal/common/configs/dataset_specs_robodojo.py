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

# Robodojo-only training specs for posttrain-from-v9.
#
# Selected via `--kwargs '{"dataset_specs":"configs/dataset_specs_robodojo.py"}'`
# passed to `train.py`. Consumed by `dataset_factory.build_training_dataset` via
# `_load_module_from_ref`, which reads `training_datasets` (and optionally
# `validation_datasets`) from this module.
#
# Self-contained by design: does NOT import from `dataset_specs.py`, so that
# unrelated dataset specs are not accidentally pulled in through side effects
# (e.g. lambdas closing over the shared DATA_BASE).

import copy
import os
from glob import glob

DATA_BASE = os.environ.get("HOLOBRAIN_DATA_BASE", "./data")


def _glob_sorted(
    *patterns: str, exclude_patterns: None | list[str] | str = None
) -> list[str]:
    # Copied from configs/dataset_specs.py:26-40 to keep this module standalone.
    data_paths = []
    for pattern in patterns:
        data_paths.extend(glob(pattern))
    data_paths = sorted(set(data_paths))
    if exclude_patterns is not None:
        if isinstance(exclude_patterns, str):
            exclude_patterns = [exclude_patterns]
        exclude_paths = []
        for exclude in exclude_patterns:
            exclude_paths.extend(glob(exclude))
        data_paths = [x for x in data_paths if x not in exclude_paths]
    return data_paths


TRAINING_DATASETS = [
    # ================ robodojo ==================
    dict(
        dataset_type="robodojo",
        dataset_name="robodojo",
        data_paths=lambda: _glob_sorted(
            f"{DATA_BASE}/robodojo/lmdb/*",
        ),
        setting_type="arx_x5a",
    ),
]

# No hold-out val split for now; sim eval success rate is the primary metric.
VALIDATION_DATASETS = None

training_datasets = copy.deepcopy(TRAINING_DATASETS)
training_datasets.sort(key=lambda x: x["dataset_name"])
