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


"""Processors exported for deployment, independent of what was trained.

One entry per distinct embodiment: a processor depends only on the body and
camera layout, not on where data lives. The four LIBERO training suites share
one ``libero`` setting, and Panda's rigid-object and articulated-object
selections share one ``panda_droid`` setting. The two RoboTwin bodies stay
separate.
"""

__all__ = ["deploy_datasets"]

deploy_datasets = [
    dict(
        dataset_type="robotwin",
        dataset_name="robotwin2_0",
        setting_type="aloha_v2",
    ),
    dict(
        dataset_type="robotwin",
        dataset_name="robotwin2_0_ur5_wsg",
        setting_type="ur5_wsg",
    ),
    dict(
        dataset_type="libero",
        dataset_name="libero",
        setting_type="libero",
    ),
    dict(
        dataset_type="isaac",
        dataset_name="panda_droid",
        setting_type="panda_droid",
    ),
]
