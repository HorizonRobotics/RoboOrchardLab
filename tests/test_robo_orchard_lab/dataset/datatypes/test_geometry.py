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

from typing import Literal

import pytest
import torch
from datasets import Dataset
from datasets.features.features import (
    Features,
)

from robo_orchard_lab.dataset.datatypes.geometry import (
    BatchFrameTransform,
    BatchPose,
)
from test_robo_orchard_lab.dataset.datatypes._hf_datasets_compat import (
    get_generator_example,
)


class TestBatchPose:
    @pytest.mark.parametrize("dataset_from", ["dict", "generator"])
    def test_as_huggingface_dataset(
        self, dataset_from: Literal["dict", "generator"]
    ):
        data: list[BatchPose] = [
            BatchPose(
                xyz=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
                quat=torch.tensor(
                    [[4, 5, 6, 7]],
                    dtype=torch.float32,
                ),
                frame_id="camera",
            ),
            BatchPose(
                xyz=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
                quat=torch.tensor(
                    [[4, 5, 6, 7]],
                    dtype=torch.float32,
                ),
                frame_id="camera",
                timestamps=[
                    1,
                ],
            ),
        ]

        feature = data[0].dataset_feature()
        features = Features({"data": feature})
        if dataset_from == "generator":
            # Use a generator to create the dataset
            def dataset_iter():
                for item in data:
                    yield get_generator_example(features, {"data": item})

            dataset = Dataset.from_generator(
                dataset_iter,
                features=features,
                streaming=True,
            )
        elif dataset_from == "dict":
            dataset = Dataset.from_dict(
                {"data": [item for item in data]},
                features=features,
            )
        else:
            raise ValueError(f"Unknown dataset_from value: {dataset_from}")

        for original, recovered in zip(data, dataset, strict=True):
            recovered = recovered["data"]
            assert isinstance(recovered, BatchPose)

            # Check the properties of the recovered object
            assert torch.equal(original.xyz, recovered.xyz)
            assert torch.equal(original.quat, recovered.quat)
            assert original.frame_id == recovered.frame_id
            assert original.timestamps == recovered.timestamps


class TestBatchFrameTransform:
    @pytest.mark.parametrize("dataset_from", ["dict", "generator"])
    def test_as_huggingface_dataset(
        self, dataset_from: Literal["dict", "generator"]
    ):
        data: list[BatchFrameTransform] = [
            BatchFrameTransform(
                xyz=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
                quat=torch.tensor(
                    [[4, 5, 6, 7]],
                    dtype=torch.float32,
                ),
                parent_frame_id="world",
                child_frame_id="camera",
            ),
            BatchFrameTransform(
                xyz=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
                quat=torch.tensor(
                    [[4, 5, 6, 7]],
                    dtype=torch.float32,
                ),
                parent_frame_id="world",
                child_frame_id="camera",
                timestamps=[
                    1,
                ],
            ),
        ]

        feature = data[0].dataset_feature()
        features = Features({"data": feature})
        if dataset_from == "generator":
            # Use a generator to create the dataset
            def dataset_iter():
                for item in data:
                    yield get_generator_example(features, {"data": item})

            dataset = Dataset.from_generator(
                dataset_iter,
                features=features,
                streaming=True,
            )
        elif dataset_from == "dict":
            dataset = Dataset.from_dict(
                {"data": [item for item in data]},
                features=features,
            )
        else:
            raise ValueError(f"Unknown dataset_from value: {dataset_from}")

        for original, recovered in zip(data, dataset, strict=True):
            recovered = recovered["data"]
            assert isinstance(recovered, BatchFrameTransform)

            # Check the properties of the recovered object
            assert torch.equal(original.xyz, recovered.xyz)
            assert torch.equal(original.quat, recovered.quat)
            assert original.parent_frame_id == recovered.parent_frame_id
            assert original.child_frame_id == recovered.child_frame_id
            assert original.timestamps == recovered.timestamps
