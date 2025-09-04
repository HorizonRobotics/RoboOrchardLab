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
from typing import Literal

import pytest

from robo_orchard_lab.dataset.datatypes import (
    BatchCameraData,
    BatchCameraDataEncoded,
)
from robo_orchard_lab.dataset.robot.dataset import RODataset
from robo_orchard_lab.transforms.image import (
    ImageDecodeConfig,
)


class TestImageDecoder:
    @pytest.mark.parametrize("backend", ["pil", "cv2"])
    def test_decode(
        self, ROBO_ORCHARD_TEST_WORKSPACE: str, backend: Literal["pil", "cv2"]
    ):
        path = os.path.join(
            ROBO_ORCHARD_TEST_WORKSPACE,
            "robo_orchard_workspace/datasets/robotwin/ro_dataset",
        )
        dataset = RODataset(dataset_path=path, meta_index2meta=False)
        print(dataset[0]["head_camera"])

        decoder = ImageDecodeConfig(
            input_columns=["head_camera"], backend=backend
        )()

        ret = decoder(dataset[0])
        print(ret["head_camera"])

        assert isinstance(ret["head_camera"], BatchCameraData)
        assert isinstance(ret["right_camera"], BatchCameraDataEncoded)
        assert ret["head_camera"].sensor_data.shape[-1] == 3

        print(ret["head_camera"].sensor_data.shape)
