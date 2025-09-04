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

from robo_orchard_lab.dataset.datatypes.camera import (
    BatchCameraData,
    BatchCameraDataEncoded,
    Distortion,
    ImageMode,
)


class TestBatchCameraData:
    @pytest.mark.parametrize("dataset_from", ["dict", "generator"])
    def test_as_huggingface_dataset(
        self, dataset_from: Literal["dict", "generator"]
    ):
        data: list[BatchCameraData] = [
            BatchCameraData(
                sensor_data=torch.rand(
                    size=(2, 12, 11, 3), dtype=torch.float32
                ),
                pix_fmt=ImageMode.BGR,
            ),
            BatchCameraData(
                sensor_data=torch.rand(
                    size=(2, 12, 11, 3), dtype=torch.float32
                ),
                pix_fmt=ImageMode.BGR,
                # with intrinsic matrices
                intrinsic_matrices=torch.tensor(
                    [
                        [
                            [1000.0, 0.0, 320.0],
                            [0.0, 1000.0, 240.0],
                            [0.0, 0.0, 1.0],
                        ]
                    ],
                    dtype=torch.float32,
                ).repeat(2, 1, 1),
            ),
            BatchCameraData(
                sensor_data=torch.rand(
                    size=(2, 12, 11, 3), dtype=torch.float32
                ),
                pix_fmt=ImageMode.BGR,
                # with distortion
                distortion=Distortion(
                    model="plumb_bob",
                    coefficients=torch.tensor(
                        [0.1, 0.01, 0.001, 0.0001], dtype=torch.float32
                    ),
                ),
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

        for original, recovered in zip(data, dataset, strict=True):
            recovered = recovered["data"]
            for origin_k in original.__dict__.keys():
                src_value = getattr(original, origin_k)
                dst_value = getattr(recovered, origin_k)
                if isinstance(src_value, torch.Tensor):
                    assert torch.equal(src_value, dst_value)
                elif isinstance(src_value, Distortion):
                    assert src_value.model == dst_value.model
                    if src_value.coefficients is not None:
                        assert torch.equal(
                            src_value.coefficients, dst_value.coefficients
                        )
                else:
                    assert src_value == dst_value, (
                        f"Mismatch in {origin_k}: {src_value} != {dst_value}"
                    )


class TestBatchCameraDataEncoded:
    @pytest.mark.parametrize("dataset_from", ["dict", "generator"])
    def test_as_huggingface_dataset(
        self, dataset_from: Literal["dict", "generator"]
    ):
        data: list[BatchCameraDataEncoded] = [
            BatchCameraDataEncoded(
                sensor_data=[b"image_data_1", b"image_data_2"],
                format="jpeg",
            ),
            BatchCameraDataEncoded(
                sensor_data=[b"image_data_1", b"image_data_2"],
                format="jpeg",
                # with intrinsic matrices
                intrinsic_matrices=torch.tensor(
                    [
                        [
                            [1000.0, 0.0, 320.0],
                            [0.0, 1000.0, 240.0],
                            [0.0, 0.0, 1.0],
                        ]
                    ],
                    dtype=torch.float32,
                ).repeat(2, 1, 1),
            ),
            BatchCameraDataEncoded(
                sensor_data=[b"image_data_1", b"image_data_2"],
                format="jpeg",
                # with distortion
                distortion=Distortion(
                    model="plumb_bob",
                    coefficients=torch.tensor(
                        [0.1, 0.01, 0.001, 0.0001], dtype=torch.float32
                    ),
                ),
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

        for original, recovered in zip(data, dataset, strict=True):
            recovered = recovered["data"]
            for origin_k in original.__dict__.keys():
                src_value = getattr(original, origin_k)
                dst_value = getattr(recovered, origin_k)
                if isinstance(src_value, torch.Tensor):
                    assert torch.equal(src_value, dst_value)
                elif isinstance(src_value, Distortion):
                    assert src_value.model == dst_value.model
                    if src_value.coefficients is not None:
                        assert torch.equal(
                            src_value.coefficients, dst_value.coefficients
                        )
                else:
                    assert src_value == dst_value, (
                        f"Mismatch in {origin_k}: {src_value} != {dst_value}"
                    )
