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
import pytest
import torch

from robo_orchard_lab.utils.torch import to_device

# --- Test Setup using Pytest Fixtures ---


@pytest.fixture(
    params=[
        pytest.param("cpu", id="cpu"),
        pytest.param(
            "cuda",
            id="cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA device not available",
            ),
        ),
    ]
)
def device(request: pytest.FixtureRequest) -> torch.device:
    """Provides a torch.device object for CPU and CUDA (if available)."""
    return torch.device(request.param)


# --- Unit Tests ---


def test_to_device_single_tensor(device: torch.device):
    """Tests moving a single torch.Tensor."""
    tensor = torch.randn(2, 2)
    tensor_on_device = to_device(tensor, device)
    assert tensor_on_device.device.type == device.type


def test_to_device_list_of_tensors(device: torch.device):
    """Tests moving a list of tensors."""
    tensors = [torch.randn(2) for _ in range(3)]
    tensors_on_device = to_device(tensors, device)

    assert isinstance(tensors_on_device, list)
    for tensor in tensors_on_device:
        assert isinstance(tensor, torch.Tensor)
        assert tensor.device.type == device.type


def test_to_device_tuple_of_tensors(device: torch.device):
    """Tests moving a tuple of tensors."""
    tensors = tuple(torch.randn(2) for _ in range(3))
    tensors_on_device = to_device(tensors, device)

    assert isinstance(tensors_on_device, tuple)
    for tensor in tensors_on_device:
        assert isinstance(tensor, torch.Tensor)
        assert tensor.device.type == device.type


def test_to_device_dict_of_tensors(device: torch.device):
    """Tests moving a dictionary of tensors."""
    tensor_dict = {"a": torch.randn(2), "b": torch.randn(3)}
    dict_on_device = to_device(tensor_dict, device)

    assert isinstance(dict_on_device, dict)
    for _, tensor in dict_on_device.items():
        assert isinstance(tensor, torch.Tensor)
        assert tensor.device.type == device.type


def test_to_device_nested_structure(device: torch.device):
    """Tests a complex nested structure with mixed data types."""
    original_data = {
        "id": "sample_001",
        "is_valid": True,
        "image_tensor": torch.randn(3, 32, 32),
        "history": [
            {"action": torch.tensor([0.1, 0.2]), "reward": 1.0},
            {"action": torch.tensor([-0.2, 0.3]), "reward": 1.5},
        ],
        "metadata": None,
        "coords": np.array([1, 2, 3]),  # Should not be moved
    }

    data_on_device = to_device(original_data, device)

    # Assert tensors are moved
    assert data_on_device["image_tensor"].device.type == device.type
    assert data_on_device["history"][0]["action"].device.type == device.type
    assert data_on_device["history"][1]["action"].device.type == device.type

    # Assert non-tensor data is unchanged
    assert data_on_device["id"] == "sample_001"
    assert data_on_device["is_valid"] is True
    assert data_on_device["metadata"] is None
    assert isinstance(data_on_device["history"][0]["reward"], float)
    assert np.array_equal(data_on_device["coords"], original_data["coords"])


def test_to_device_data_with_no_tensors(device: torch.device):
    """Tests that data structures without any tensors are returned unmodified."""  # noqa: E501
    data = {"a": 1, "b": "hello", "c": [1, 2, 3]}
    # deepcopy would be better for a perfect test, but equality check is sufficient  # noqa: E501
    original_data = data.copy()

    processed_data = to_device(data, device)

    assert processed_data == original_data


def test_to_device_empty_structures(device: torch.device):
    """Tests that empty lists, tuples, and dicts are handled correctly."""
    assert to_device([], device) == []
    assert to_device((), device) == ()
    assert to_device({}, device) == {}
