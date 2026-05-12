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

import torch

from robo_orchard_lab.models.layers.data_preprocessors import (
    BaseDataPreprocessor,
)


class TestBaseDataPreprocessor:
    """Tests for BaseDataPreprocessor."""

    def test_process_img_handles_sequence_and_none(self):
        preprocessor = BaseDataPreprocessor(
            mean=[1.0, 2.0, 3.0],
            std=[1.0, 2.0, 4.0],
            hwc_to_chw=False,
        )
        image = torch.tensor(
            [
                [
                    [[2.0, 4.0, 7.0]],
                ]
            ]
        )

        result = preprocessor.process_img([image, None])

        assert isinstance(result, list)
        assert result[1] is None
        assert torch.equal(
            result[0],
            torch.tensor(
                [
                    [
                        [[1.0, 1.0, 1.0]],
                    ]
                ]
            ),
        )
