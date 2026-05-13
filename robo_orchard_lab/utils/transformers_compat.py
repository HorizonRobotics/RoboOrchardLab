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
from __future__ import annotations
from collections.abc import Mapping
from typing import Any, Literal

import torch
import transformers

try:
    from transformers.modeling_utils import (
        get_parameter_device as _hf_get_parameter_device,
        get_parameter_dtype as _hf_get_parameter_dtype,
    )
except ImportError:
    _hf_get_parameter_device = None
    _hf_get_parameter_dtype = None


def get_module_device(module: torch.nn.Module) -> torch.device:
    """Return the runtime device while preserving the TF4 helper path."""

    if _hf_get_parameter_device is not None:
        return _hf_get_parameter_device(module)

    for tensor in module.parameters(recurse=True):
        return tensor.device
    for tensor in module.buffers(recurse=True):
        return tensor.device
    return torch.device("cpu")


def get_module_dtype(module: torch.nn.Module) -> torch.dtype:
    """Return the runtime dtype while preserving the TF4 helper path."""

    if _hf_get_parameter_dtype is not None:
        return _hf_get_parameter_dtype(module)

    fallback = None
    for tensor in module.parameters(recurse=True):
        fallback = tensor.dtype
        if tensor.is_floating_point():
            return tensor.dtype
    for tensor in module.buffers(recurse=True):
        fallback = tensor.dtype
        if tensor.is_floating_point():
            return tensor.dtype
    return fallback or torch.float32


def batch_encode_tokenizer(tokenizer: Any, *args: Any, **kwargs: Any) -> Any:
    """Batch encode text while preserving the legacy tokenizer method path."""

    batch_encode_plus = getattr(tokenizer, "batch_encode_plus", None)
    if callable(batch_encode_plus):
        return batch_encode_plus(*args, **kwargs)
    return tokenizer(*args, **kwargs)


def prepare_hf_attention_mask(
    model: torch.nn.Module,
    attention_mask: torch.Tensor,
    input_shape: tuple[int, ...],
) -> torch.Tensor:
    """Prepare attention masks while preserving legacy Transformers paths."""

    if attention_mask.dim() == 3 and callable(
        getattr(model, "_create_attention_masks", None)
    ):
        return model.get_extended_attention_mask(
            attention_mask,
            input_shape,
            dtype=get_module_dtype(model),
        )
    return attention_mask


def _numeric_prefix(token: str) -> int | None:
    numeric_prefix = ""
    for char in token:
        if not char.isdigit():
            break
        numeric_prefix += char
    if not numeric_prefix:
        return None
    return int(numeric_prefix)


def runtime_hf_dtype_kwarg_name() -> Literal["dtype", "torch_dtype"]:
    """Return the dtype kwarg name expected by the installed Transformers."""

    version_tokens = transformers.__version__.split(".")
    parsed_tokens = [
        parsed
        for token in version_tokens[:2]
        if (parsed := _numeric_prefix(token)) is not None
    ]
    if len(parsed_tokens) < 2:
        return "torch_dtype"
    return "dtype" if tuple(parsed_tokens[:2]) >= (4, 56) else "torch_dtype"


def normalize_hf_dtype_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize ``dtype`` and ``torch_dtype`` aliases for Transformers."""

    normalized_kwargs = dict(kwargs)
    dtype = normalized_kwargs.pop("dtype", None)
    torch_dtype = normalized_kwargs.pop("torch_dtype", None)

    if dtype is None and torch_dtype is None:
        return normalized_kwargs

    if dtype is not None and torch_dtype is not None and dtype != torch_dtype:
        raise ValueError(
            "`dtype` and `torch_dtype` must match when both are provided."
        )

    normalized_kwargs[runtime_hf_dtype_kwarg_name()] = (
        dtype if dtype is not None else torch_dtype
    )
    return normalized_kwargs
