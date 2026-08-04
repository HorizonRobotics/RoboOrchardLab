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
import base64
import pickle
import subprocess
import sys

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
from robo_orchard_lab.dataset.datatypes.hg_features.tensor_pickle import (
    _rebuild_tensor_numpy_v1,
)
from test_robo_orchard_lab.dataset.datatypes._hf_datasets_compat import (
    get_generator_example,
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
                yield get_generator_example(features, {"data": item})

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
                yield get_generator_example(features, {"data": item})

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
    @pytest.mark.parametrize("tensor_encoding", ["torch_legacy", "numpy_v1"])
    def test_feature_metadata_round_trip(self, tensor_encoding: str):
        feature = PickleFeature(
            class_type=dict,
            tensor_encoding=tensor_encoding,  # type: ignore[arg-type]
        )

        recovered = Features.from_dict(Features({"data": feature}).to_dict())[
            "data"
        ]

        assert isinstance(recovered, PickleFeature)
        assert recovered.tensor_encoding == tensor_encoding

    def test_missing_historical_metadata_is_readable_and_prefers_numpy_v1(
        self,
    ):
        metadata = {
            "data": {
                "_type": "PickleFeature",
                "class_type": "torch:Tensor",
            }
        }
        feature = Features.from_dict(metadata)["data"]
        expected = torch.arange(3)
        legacy_payload = pickle.dumps(expected)

        assert isinstance(feature, PickleFeature)
        assert feature.tensor_encoding == "numpy_v1"
        assert torch.equal(feature.decode_example(legacy_payload), expected)

        new_payload = feature.encode_example(expected)
        assert b"_rebuild_tensor_numpy_v1" in new_payload
        assert torch.equal(feature.decode_example(new_payload), expected)

    def test_tensor_encoding_field_preserves_existing_positional_order(self):
        feature = PickleFeature(dict, False, "large_binary", "zstd")

        assert feature.decode is False
        assert feature.binary_type == "large_binary"
        assert feature.compression == "zstd"
        assert feature.tensor_encoding == "numpy_v1"

    def test_invalid_tensor_encoding_fails_at_construction(self):
        with pytest.raises(ValueError, match="tensor_encoding"):
            PickleFeature(
                class_type=dict,
                tensor_encoding="invalid",  # type: ignore[arg-type]
            )

    def test_torch_legacy_is_explicit_native_pickle_opt_in(self):
        value = {"tensor": torch.arange(4), "label": "legacy"}
        feature = PickleFeature(
            class_type=dict,
            tensor_encoding="torch_legacy",
        )

        assert feature.encode_example(value) == pickle.dumps(value)

    @pytest.mark.parametrize("tensor_encoding", ["torch_legacy", "numpy_v1"])
    @pytest.mark.parametrize("compression", [None, "zstd"])
    def test_tensor_writer_policies_support_compression(
        self,
        tensor_encoding: str,
        compression: str | None,
    ):
        value = {"tensor": torch.arange(6).reshape(2, 3)}
        feature = PickleFeature(
            class_type=dict,
            tensor_encoding=tensor_encoding,  # type: ignore[arg-type]
            compression=compression,  # type: ignore[arg-type]
        )

        recovered = feature.decode_example(feature.encode_example(value))

        assert torch.equal(recovered["tensor"], value["tensor"])

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        ],
    )
    def test_numpy_v1_preserves_supported_tensor_values(
        self,
        dtype: torch.dtype,
    ):
        value = torch.tensor([[0, 1], [2, 3]], dtype=dtype)
        feature = PickleFeature(class_type=torch.Tensor)

        recovered = feature.decode_example(feature.encode_example(value))

        assert recovered.dtype == dtype
        assert recovered.shape == value.shape
        assert torch.equal(recovered, value)
        assert recovered.device.type == "cpu"
        assert recovered.is_contiguous()

    def test_numpy_v1_normalizes_tensor_runtime_state(self):
        base = torch.arange(12, dtype=torch.float32)
        tracked = torch.ones(2, requires_grad=True)
        tracked.grad = torch.full_like(tracked, 2)
        value = {
            "noncontiguous": base.reshape(3, 4).T,
            "left": base[:4],
            "right": base[2:6],
            "tracked": tracked,
        }
        feature = PickleFeature(class_type=dict)

        recovered = feature.decode_example(feature.encode_example(value))

        for tensor in recovered.values():
            assert tensor.device.type == "cpu"
            assert tensor.is_contiguous()
        assert torch.equal(recovered["noncontiguous"], value["noncontiguous"])
        assert not recovered["tracked"].requires_grad
        assert recovered["tracked"].grad is None

        right_before = recovered["right"].clone()
        recovered["left"][2] = -1
        assert torch.equal(recovered["right"], right_before)

    @pytest.mark.parametrize(
        ("value", "error_match"),
        [
            (torch.nn.Parameter(torch.ones(2)), "subclasses"),
            (
                torch.sparse_coo_tensor(
                    torch.tensor([[0], [1]]),
                    torch.tensor([1.0]),
                    (2, 2),
                ),
                "strided",
            ),
            (
                torch.quantize_per_tensor(
                    torch.arange(4, dtype=torch.float32),
                    scale=0.1,
                    zero_point=10,
                    dtype=torch.quint8,
                ),
                "quantized",
            ),
            (torch.empty(2, device="meta"), "meta"),
            (torch.ones(2, dtype=torch.bfloat16), "losslessly"),
        ],
        ids=["subclass", "sparse", "quantized", "meta", "bfloat16"],
    )
    def test_numpy_v1_rejects_unsupported_tensor_semantics(
        self,
        value: torch.Tensor,
        error_match: str,
    ):
        feature = PickleFeature(class_type=dict)

        with pytest.raises(TypeError, match=error_match):
            feature.encode_example({"tensor": value})

    @pytest.mark.parametrize("reader_policy", ["torch_legacy", "numpy_v1"])
    def test_decoder_is_payload_driven_for_mixed_cells(
        self,
        reader_policy: str,
    ):
        value = {"tensor": torch.arange(4, dtype=torch.float32)}
        legacy_writer = PickleFeature(
            class_type=dict,
            tensor_encoding="torch_legacy",
        )
        numpy_writer = PickleFeature(class_type=dict)
        reader = PickleFeature(
            class_type=dict,
            tensor_encoding=reader_policy,  # type: ignore[arg-type]
        )
        payloads = [
            legacy_writer.encode_example(value),
            numpy_writer.encode_example(value),
            legacy_writer.encode_example(value),
        ]

        binary_column = pa.array(payloads, type=pa.binary())
        recovered = [
            reader.decode_example(item.as_py()) for item in binary_column
        ]

        assert all(
            torch.equal(item["tensor"], value["tensor"]) for item in recovered
        )

    def test_numpy_v1_rebuild_fqn_loads_in_fresh_interpreter(self):
        feature = PickleFeature(class_type=dict)
        payload = feature.encode_example({"tensor": torch.arange(3)})
        module_name = (
            "robo_orchard_lab.dataset.datatypes.hg_features.tensor_pickle"
        )
        helper_name = "_rebuild_tensor_numpy_v1"

        assert _rebuild_tensor_numpy_v1.__module__ == module_name
        assert _rebuild_tensor_numpy_v1.__qualname__ == helper_name
        assert module_name.encode() in payload
        assert helper_name.encode() in payload

        encoded_payload = base64.b64encode(payload).decode("ascii")
        script = """
import base64
import pickle
import sys

value = pickle.loads(base64.b64decode(sys.argv[1]))
assert value["tensor"].tolist() == [0, 1, 2]
"""
        subprocess.run(
            [sys.executable, "-c", script, encoded_payload],
            check=True,
        )

    def test_numpy_v1_loads_are_writable_and_mutation_isolated(self):
        feature = PickleFeature(class_type=dict)
        payload = feature.encode_example(
            {"tensor": torch.arange(4, dtype=torch.float32)}
        )

        first = feature.decode_example(payload)
        second = feature.decode_example(payload)
        first["tensor"][0] = -1

        assert second["tensor"][0].item() == 0

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
                yield get_generator_example(features, {"data": item})

        d = Dataset.from_generator(
            generate_data, features=features, streaming=True
        )

        for original, recovered in zip(data, d, strict=True):
            recovered = recovered["data"]
            assert isinstance(recovered, torch.Tensor)

            assert torch.equal(original, recovered)


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
