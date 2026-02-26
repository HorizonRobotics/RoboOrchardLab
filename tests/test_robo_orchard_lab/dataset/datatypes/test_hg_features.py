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
import numpy as np
import pyarrow as pa
import pytest
import torch
from datasets import Dataset
from datasets.arrow_writer import TypedSequence
from datasets.features.features import (
    Features,
    decode_nested_example,
    encode_nested_example,
)

from robo_orchard_lab.dataset.datatypes.hg_features import PickleFeature
from robo_orchard_lab.dataset.datatypes.hg_features.tensor import (
    AnyTensorFeature,
    TypedTensorFeature,
)


class TestTypedTensorFeature:
    @pytest.mark.parametrize(
        "feature",
        [
            TypedTensorFeature(dtype="float32", as_torch_tensor=True),
            TypedTensorFeature(dtype="float32", as_torch_tensor=False),
        ],
    )
    def test_typed_sequence_encode_decode(self, feature: AnyTensorFeature):
        data = [
            np.array([1, 2, 3], dtype=np.float32),
            torch.tensor([4, 5, 6], dtype=torch.float32),
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
        ]

        typed_seq = TypedSequence(
            data=[encode_nested_example(feature, item) for item in data],
            type=feature,  # type: ignore
        )
        pa_arr = pa.array(typed_seq)
        print(pa_arr)
        recovered_data = [
            decode_nested_example(feature, item.as_py()) for item in pa_arr
        ]
        for original, recovered in zip(data, recovered_data, strict=True):
            if feature.as_torch_tensor:
                assert isinstance(recovered, torch.Tensor)
            else:
                assert isinstance(recovered, np.ndarray)

            assert original.shape == recovered.shape
            # convert original and recovered to numpy
            if isinstance(original, torch.Tensor):
                original = original.numpy()
            if isinstance(recovered, torch.Tensor):
                recovered = recovered.numpy()

            assert np.array_equal(original, recovered)

    def test_datasets(self):
        feature = TypedTensorFeature(dtype="float32")
        data = [
            np.array([1, 2, 3], dtype=np.float32),
            torch.tensor([4, 5, 6], dtype=torch.float32),
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
        ]
        features = Features({"data": feature})

        def generate_data():
            for item in data:
                yield features.encode_example({"data": item})

        d = Dataset.from_generator(
            generate_data, features=features, streaming=True
        )
        for original, recovered in zip(data, d, strict=True):
            recovered = recovered["data"]
            if feature.as_torch_tensor:
                assert isinstance(recovered, torch.Tensor)
            else:
                assert isinstance(recovered, np.ndarray)

            assert original.shape == recovered.shape
            # convert original and recovered to numpy
            if isinstance(original, torch.Tensor):
                original = original.numpy()
            if isinstance(recovered, torch.Tensor):
                recovered = recovered.numpy()

            assert np.array_equal(original, recovered)


class TestAnyTensorFeature:
    def test_datasets(self):
        feature = AnyTensorFeature(as_torch_tensor=True)
        data = [
            np.array([1, 2, 3], dtype=np.int8),
            torch.tensor([4, 5, 6]),
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
            np.array([[[1], [2]], [[3], [4]]], dtype=np.float32),
        ]
        features = Features({"data": feature})

        def generate_data():
            for item in data:
                yield features.encode_example({"data": item})

        d = Dataset.from_generator(
            generate_data, features=features, streaming=True
        )

        for original, recovered in zip(data, d, strict=True):
            recovered = recovered["data"]
            if feature.as_torch_tensor:
                assert isinstance(recovered, torch.Tensor)
            else:
                assert isinstance(recovered, np.ndarray)

            assert original.shape == recovered.shape
            # convert original and recovered to numpy
            if isinstance(original, torch.Tensor):
                original = original.numpy()
            if isinstance(recovered, torch.Tensor):
                recovered = recovered.numpy()

            assert np.array_equal(original, recovered)

    @pytest.mark.parametrize(
        "feature",
        [
            AnyTensorFeature(as_torch_tensor=True),
            AnyTensorFeature(as_torch_tensor=False),
        ],
    )
    def test_typed_sequence_encode_decode(self, feature: AnyTensorFeature):
        data = [
            np.array([1, 2, 3], dtype=np.int8),
            torch.tensor([4, 5, 6]),
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
            np.array([[[1], [2]], [[3], [4]]], dtype=np.float32),
        ]
        typed_seq = TypedSequence(
            data=[encode_nested_example(feature, item) for item in data],
            type=feature,  # type: ignore
        )
        pa_arr = pa.array(typed_seq)
        recovered_data = [
            decode_nested_example(feature, item.as_py()) for item in pa_arr
        ]
        for original, recovered in zip(data, recovered_data, strict=True):
            if feature.as_torch_tensor:
                assert isinstance(recovered, torch.Tensor)
            else:
                assert isinstance(recovered, np.ndarray)

            assert original.shape == recovered.shape
            # convert original and recovered to numpy
            if isinstance(original, torch.Tensor):
                original = original.numpy()
            if isinstance(recovered, torch.Tensor):
                recovered = recovered.numpy()

            assert np.array_equal(original, recovered)


class TestPickleFeature:
    def test_pickle_feature_encode_decode_torch_tensor(self):
        feature = PickleFeature(class_type=torch.Tensor)
        data = [
            torch.tensor([1, 2, 3], dtype=torch.int8),
            torch.tensor([4, 5, 6], dtype=torch.float32),
            torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        ]
        typed_seq = TypedSequence(
            data=[encode_nested_example(feature, item) for item in data],
            type=feature,  # type: ignore
        )
        pa_arr = pa.array(typed_seq)
        recovered_data = [
            decode_nested_example(feature, item.as_py()) for item in pa_arr
        ]
        for original, recovered in zip(data, recovered_data, strict=True):
            assert isinstance(recovered, torch.Tensor)

            assert torch.equal(original, recovered)

    def test_datasets(self):
        feature = PickleFeature(class_type=torch.Tensor)
        data = [
            torch.tensor([1, 2, 3], dtype=torch.int8),
            torch.tensor([4, 5, 6], dtype=torch.float32),
            torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        ]
        features = Features({"data": feature})

        def generate_data():
            for item in data:
                yield features.encode_example({"data": item})

        d = Dataset.from_generator(
            generate_data, features=features, streaming=True
        )

        for original, recovered in zip(data, d, strict=True):
            recovered = recovered["data"]
            assert isinstance(recovered, torch.Tensor)

            assert torch.equal(original, recovered)


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
