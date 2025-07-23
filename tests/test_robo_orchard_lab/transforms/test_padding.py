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

import os

from robo_orchard_lab.dataset.robot.dataset import ROMultiRowDataset
from robo_orchard_lab.dataset.robot.row_sampler import (
    DeltaTimestampSamplerConfig,
)
from robo_orchard_lab.transforms.padding import PaddingConfig


class TestPadding:
    def test_padding_replace_none_header(
        self,
        ROBO_ORCHARD_TEST_WORKSPACE: str,
    ):
        path = os.path.join(
            ROBO_ORCHARD_TEST_WORKSPACE,
            "robo_orchard_workspace/datasets/robotwin/ro_dataset",
        )
        # RoboTwin dataset is 25FPS
        cfg = DeltaTimestampSamplerConfig(
            column_delta_ts={
                "joints": [0.0 + 1.0 / 25 * (i - 1) for i in range(26)],
            },
            tolerance=0.01,
        )
        dataset = ROMultiRowDataset(
            dataset_path=path,
            row_sampler=cfg,
        )
        row = dataset[0]

        assert row["joints"][0] is None

        padding_ts = PaddingConfig(
            input_columns=[
                "joints",
            ],
        )()
        row = padding_ts(row)
        assert row["joints"][0] is not None
        assert row["joints"][0].timestamps[0] == row["joints"][1].timestamps[0]
