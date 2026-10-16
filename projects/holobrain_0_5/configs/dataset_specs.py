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


"""Dataset selections for RoboTwin, LIBERO and Panda Droid action LMDBs."""

import os

DATA_BASE = os.environ.get("HOLOBRAIN_DATA_BASE", "./data")
__all__ = ["training_datasets"]
training_datasets = (
    [
        dict(
            dataset_type="robotwin",
            dataset_name=name,
            setting_type=setting,
            data_paths=[f"{DATA_BASE}/{path}"],
        )
        for name, setting, path in [
            ("robotwin2_0", "aloha_v2", "robotwin2.0/aloha_agilex_demo_clean"),
            (
                "robotwin2_0_ur5_wsg",
                "ur5_wsg",
                "robotwin2.0/ur5_wsg_demo_clean",
            ),
        ]
    ]
    + [
        dict(
            dataset_type="libero",
            dataset_name=f"libero_{suite}",
            setting_type="libero",
            data_paths=f"{DATA_BASE}/libero/lmdb_{suite}_abs",
        )
        for suite in ["goal", "object", "spatial", "10"]
    ]
    + [
        dict(
            dataset_type="isaac",
            dataset_name=name,
            setting_type="panda_droid",
            # Unsegmented action LMDBs: no Arrow subtask truncation.
            data_paths=[
                f"{DATA_BASE}/horizon_isaac/lmdb/{path}" for path in paths
            ],
        )
        for name, paths in [
            (
                "panda_droid_rigid",
                [
                    "panda_droid-place_a2b-20261014",
                    "panda_droid-spatial_pick-20261014",
                    "panda_droid-pick_color-20261014",
                    "panda_droid-stack_cubes-20261014",
                ],
            ),
            (
                "panda_droid_articulated",
                [
                    "panda_droid-joint_direction-20261014",
                    "panda_droid-open_drawer-20261014",
                ],
            ),
        ]
    ]
)
