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

from typing import Any

import torch


def to_device(data: Any, device: torch.device) -> Any:
    """Recursively moves tensors in a nested data structure to a specified device.

    This function traverses common nested data structures such as dictionaries,
    lists, and tuples. For each `torch.Tensor` it finds, it applies the
    `.to(device)` method. Other data types within the structure are
    returned unmodified.

    Args:
        data (Any): The data structure to move. It can be a `torch.Tensor`,
            a dictionary, a list, a tuple, or other basic types.
        device (torch.device): The target device to which tensors should be
            moved (e.g., `torch.device('cuda:0')` or `torch.device('cpu')`).

    Returns:
        Any: A new data structure of the same type and shape as `data`, but
             with all `torch.Tensor` instances located on the target device.
    """  # noqa: E501
    # Base case: If data is a tensor, move it directly to the device.
    if isinstance(data, torch.Tensor):
        return data.to(device)

    # Recursive step for dictionaries: apply to_device to each value.
    elif isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}

    # Recursive step for lists and tuples: apply to_device to each item.
    # This preserves the original container type (list or tuple).
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(item, device) for item in data)

    # If data is not a tensor or a supported container, return it as is.
    else:
        return data
