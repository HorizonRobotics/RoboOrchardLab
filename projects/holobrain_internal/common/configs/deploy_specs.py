# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
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

import copy

DEPLOY_DATASETS = [
    dict(
        dataset_type="agilex",
        dataset_name="horizon_beijing",
        setting_type="horizon_beijing",
    ),
    dict(
        dataset_type="agilex",
        dataset_name="horizon_beijing_piper_x",
        setting_type="horizon_beijing_piper_x",
    ),
    dict(
        dataset_type="agilex",
        dataset_name="horizon_shanghai",
        setting_type="horizon_shanghai",
    ),
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
        dataset_type="robotwin",
        dataset_name="robotwin2_0_arx_x5a",
        setting_type="arx_x5a",
    ),
    dict(
        dataset_type="robotwin",
        dataset_name="robotwin2_0_franka_panda",
        setting_type="franka_panda",
    ),
    dict(
        dataset_type="isaac",
        dataset_name="isaac_pick_place",
        setting_type="isaac_pick_place",
    ),
    dict(dataset_type="libero", dataset_name="libero"),
    dict(dataset_type="agibot_digit", dataset_name="agibot_digit_challenge"),
    # dict(
    #     dataset_type='agibot_geniesim',
    #     dataset_name='agibot_geniesim3_challenge',
    # ),
    dict(
        dataset_type="behavior",
        dataset_name="behavior1k",
    ),
    dict(
        dataset_type="interna1",
        dataset_name="interna1_arx_lift2",
        setting_type="interna1_arx_lift2",
    ),
    dict(
        dataset_type="interna1",
        dataset_name="interna1_agile_split_aloha",
        setting_type="interna1_agile_split_aloha",
    ),
    # dict(
    #     dataset_type='interna1',
    #     dataset_name='interna1_genieg1',
    #     setting_type='interna1_genieg1',
    # ),
    dict(
        dataset_type="table30v2",
        dataset_name="table30v2_ur5",
        setting_type="ur5",
    ),
    dict(
        dataset_type="table30v2",
        dataset_name="table30v2_arx5",
        setting_type="arx5",
    ),
    dict(
        dataset_type="table30v2",
        dataset_name="table30v2_aloha",
        setting_type="aloha",
    ),
    dict(
        dataset_type="table30v2",
        dataset_name="table30v2_dos_w1",
        setting_type="dos_w1",
    ),
]

deploy_datasets = copy.deepcopy(DEPLOY_DATASETS)
