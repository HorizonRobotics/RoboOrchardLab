# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
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

"""Persistence helpers for NumPy-backed tensor pickle payloads.

``_rebuild_tensor_numpy_v1`` is a persisted ABI entry point. Existing
payloads import its fully qualified name, so it must not be moved or renamed.
"""

from __future__ import annotations
import io
import pickle
from typing import Any

import numpy as np
import torch

__all__: list[str] = []


class _NumpyTensorPickler(pickle.Pickler):
    """Lower exact built-in tensors while rejecting tensor subclasses."""

    def reducer_override(self, obj: Any):
        if not isinstance(obj, torch.Tensor):
            return NotImplemented
        if type(obj) is not torch.Tensor:
            raise TypeError(
                "numpy_v1 does not support torch.Tensor subclasses: "
                f"{type(obj).__module__}.{type(obj).__qualname__}"
            )
        return _rebuild_tensor_numpy_v1, (_tensor_to_numpy_v1(obj),)


def _dumps_numpy_v1(obj: Any) -> bytes:
    """Serialize an object while lowering exact tensors to NumPy arrays.

    Non-tensor objects retain their native pickle behavior. Tensor subclasses
    and tensor representations that cannot round-trip through NumPy fail at
    this boundary instead of silently falling back to PyTorch pickle.

    Args:
        obj (Any): Trusted Python object to serialize.

    Returns:
        bytes: An in-band pickle protocol 5 payload.
    """

    buffer = io.BytesIO()
    _NumpyTensorPickler(buffer, protocol=5).dump(obj)
    return buffer.getvalue()


# Persisted ABI: pickle payloads import this exact fully qualified name.
def _rebuild_tensor_numpy_v1(array: np.ndarray) -> torch.Tensor:
    """Rebuild a tensor stored by the persisted ``numpy_v1`` contract.

    The fully qualified name of this helper is embedded in persisted pickle
    payloads and must remain importable. It returns a writable CPU tensor view
    over the independently unpickled, C-contiguous NumPy array.

    Args:
        array (np.ndarray): Array restored from the pickle payload.

    Returns:
        torch.Tensor: CPU, dense, contiguous tensor sharing the array storage.

    Raises:
        TypeError: If a payload does not contain a supported NumPy array.
        ValueError: If the restored array is not writable or C-contiguous.
    """

    if not isinstance(array, np.ndarray):
        raise TypeError(
            "numpy_v1 tensor payload must contain a NumPy ndarray, "
            f"but got {type(array)!r}"
        )
    if not array.flags.c_contiguous:
        raise ValueError("numpy_v1 tensor array must be C-contiguous")
    if not array.flags.writeable:
        raise ValueError("numpy_v1 tensor array must be writable")
    try:
        return torch.from_numpy(array)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(
            "numpy_v1 cannot rebuild a tensor from NumPy dtype "
            f"{array.dtype!s} with shape {array.shape}"
        ) from exc


def _tensor_to_numpy_v1(tensor: torch.Tensor) -> np.ndarray:
    """Normalize one exact tensor and verify lossless NumPy round-trip."""

    if tensor.is_quantized:
        raise TypeError(
            "numpy_v1 does not support quantized torch.Tensor values"
        )
    if tensor.is_meta:
        raise TypeError("numpy_v1 does not support meta torch.Tensor values")
    if tensor.layout is not torch.strided:
        raise TypeError(
            "numpy_v1 only supports strided torch.Tensor values, "
            f"but got layout {tensor.layout}"
        )

    normalized = (
        tensor.detach().cpu().resolve_conj().resolve_neg().contiguous()
    )
    try:
        array = normalized.numpy()
        rebuilt = torch.from_numpy(array)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(
            "numpy_v1 cannot losslessly encode torch.Tensor dtype "
            f"{tensor.dtype} with layout {tensor.layout}"
        ) from exc

    if rebuilt.dtype != normalized.dtype or rebuilt.shape != normalized.shape:
        raise TypeError(
            "numpy_v1 NumPy round-trip changed torch.Tensor metadata: "
            f"dtype {normalized.dtype} -> {rebuilt.dtype}, "
            f"shape {tuple(normalized.shape)} -> {tuple(rebuilt.shape)}"
        )
    if not array.flags.c_contiguous:
        raise TypeError("numpy_v1 normalized tensor array is not C-contiguous")
    return array
