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

from robo_orchard_lab.dataset.datatypes import (
    BatchFrameTransform,
    BatchFrameTransformGraph,
)


class TestBatchJointsState:
    @pytest.mark.parametrize("dataset_from", ["dict", "generator"])
    def test_as_huggingface_dataset(
        self, dataset_from: Literal["dict", "generator"]
    ):
        data: list[BatchFrameTransformGraph] = [
            BatchFrameTransformGraph(
                tf_list=[
                    BatchFrameTransform(
                        xyz=torch.rand(size=(3, 3)),
                        quat=torch.rand(size=(3, 4)),
                        parent_frame_id="0",
                        child_frame_id="1",
                    ),
                ],
            ),
            BatchFrameTransformGraph(
                tf_list=[
                    BatchFrameTransform(
                        xyz=torch.rand(size=(3, 3)),
                        quat=torch.rand(size=(3, 4)),
                        parent_frame_id="0",
                        child_frame_id="1",
                    ),
                ],
                bidirectional=False,
            ),
            BatchFrameTransformGraph(
                tf_list=[
                    BatchFrameTransform(
                        xyz=torch.rand(size=(3, 3)),
                        quat=torch.rand(size=(3, 4)),
                        parent_frame_id="0",
                        child_frame_id="1",
                        timestamps=[1, 2, 3],
                    ),
                ],
            ),
            BatchFrameTransformGraph(
                tf_list=[
                    BatchFrameTransform(
                        xyz=torch.rand(size=(3, 3)),
                        quat=torch.rand(size=(3, 4)),
                        parent_frame_id="0",
                        child_frame_id="1",
                        timestamps=[1, 2, 3],
                    ),
                ],
                static_tf=[True],
            ),
        ]

        feature = data[0].dataset_feature()
        features = Features({"data": feature})
        if dataset_from == "generator":
            # Use a generator to create the dataset
            def dataset_iter():
                for item in data:
                    yield features.encode_example({"data": item})

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
            assert isinstance(dataset_item, BatchFrameTransformGraph)
            origin_item = data[i]
            assert dataset_item == origin_item
