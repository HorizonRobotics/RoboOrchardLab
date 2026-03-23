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

from robo_orchard_lab.dataset.datatypes.joint_state import (
    BatchJointsState,
)
from test_robo_orchard_lab.dataset.datatypes._hf_datasets_compat import (
    get_generator_example,
)


class TestBatchJointsState:
    @pytest.mark.parametrize("dataset_from", ["dict", "generator"])
    def test_as_huggingface_dataset(
        self, dataset_from: Literal["dict", "generator"]
    ):
        data: list[BatchJointsState] = [
            BatchJointsState(
                position=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
                velocity=torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float32),
                effort=torch.tensor([[7.0, 8.0, 9.0]], dtype=torch.float32),
                names=["joint1", "joint2", "joint3"],
                timestamps=[1],
            ),
            BatchJointsState(
                position=torch.tensor([[10.0, 11.0]], dtype=torch.float32),
                velocity=torch.tensor([[12.0, 13.0]], dtype=torch.float32),
                effort=torch.tensor([[14.0, 15.0]], dtype=torch.float32),
                names=["joint4", "joint5"],
                # missing timestamps
            ),
            BatchJointsState(
                position=torch.tensor([[10.0, 11.0]], dtype=torch.float32),
                # missing velocity
                velocity=None,
                effort=torch.tensor([[14.0, 15.0]], dtype=torch.float32),
                names=["joint4", "joint5"],
            ),
            BatchJointsState(
                position=torch.tensor([[10.0, 11.0]], dtype=torch.float32),
                velocity=None,
                effort=torch.tensor([[14.0, 15.0]], dtype=torch.float32),
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

        for i, item in enumerate(dataset):
            dataset_item = item["data"]
            assert isinstance(dataset_item, BatchJointsState)
            origin_item = data[i]
            if origin_item.position is not None:
                assert torch.equal(
                    dataset_item.position,  # type: ignore
                    origin_item.position,
                ), f"Position mismatch at index {i}"
            if origin_item.velocity is not None:
                assert torch.equal(
                    dataset_item.velocity,  # type: ignore
                    origin_item.velocity,
                ), f"Velocity mismatch at index {i}"
            if origin_item.effort is not None:
                assert torch.equal(
                    dataset_item.effort,  # type: ignore
                    origin_item.effort,
                ), f"Effort mismatch at index {i}"
            if origin_item.names is not None:
                assert dataset_item.names == origin_item.names, (
                    f"Names mismatch at index {i}: "
                    f"{dataset_item.names} != {origin_item.names}"
                )
            if origin_item.timestamps is not None:
                assert dataset_item.timestamps == origin_item.timestamps, (
                    f"Timestamps mismatch at index {i}: "
                    f"{dataset_item.timestamps} != {origin_item.timestamps}"
                )


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
