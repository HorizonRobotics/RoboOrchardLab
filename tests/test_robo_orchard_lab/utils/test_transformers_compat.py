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
import pytest
import torch
from torch import nn

import robo_orchard_lab.utils.transformers_compat as transformers_compat


class ParameterModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, dtype=torch.float16))


class BufferOnlyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("ids", torch.ones(2, dtype=torch.int64))
        self.register_buffer("scale", torch.ones(2, dtype=torch.float64))


class IntBufferOnlyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("ids", torch.ones(2, dtype=torch.int64))


class LegacyTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def batch_encode_plus(self, *args, **kwargs):
        self.calls.append(("batch_encode_plus", args, kwargs))
        return "legacy"

    def __call__(self, *args, **kwargs):
        self.calls.append(("call", args, kwargs))
        return "modern"


class CallableTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append(("call", args, kwargs))
        return "modern"


class ModernAttentionMaskModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dtype=torch.float16))
        self.calls = []

    def _create_attention_masks(self):
        raise AssertionError("capability probe should not call this method")

    def get_extended_attention_mask(
        self,
        attention_mask,
        input_shape,
        dtype=None,
    ):
        self.calls.append((attention_mask, input_shape, dtype))
        return attention_mask[:, None, :, :].to(dtype=dtype)


def test_get_module_device_delegates_to_hf_helper_when_available(monkeypatch):
    monkeypatch.setattr(
        transformers_compat,
        "_hf_get_parameter_device",
        lambda module: torch.device("meta"),
    )

    assert transformers_compat.get_module_device(ParameterModule()) == (
        torch.device("meta")
    )


def test_get_module_dtype_delegates_to_hf_helper_when_available(monkeypatch):
    monkeypatch.setattr(
        transformers_compat,
        "_hf_get_parameter_dtype",
        lambda module: torch.bfloat16,
    )

    assert transformers_compat.get_module_dtype(ParameterModule()) == (
        torch.bfloat16
    )


def test_get_module_device_uses_first_parameter_device(monkeypatch):
    monkeypatch.setattr(transformers_compat, "_hf_get_parameter_device", None)
    module = ParameterModule()

    assert transformers_compat.get_module_device(module) == (
        module.weight.device
    )


def test_get_module_device_falls_back_to_buffer_device(monkeypatch):
    monkeypatch.setattr(transformers_compat, "_hf_get_parameter_device", None)
    module = BufferOnlyModule()

    assert transformers_compat.get_module_device(module) == module.ids.device


def test_get_module_device_falls_back_to_cpu_for_empty_module(monkeypatch):
    monkeypatch.setattr(transformers_compat, "_hf_get_parameter_device", None)

    assert transformers_compat.get_module_device(nn.Module()) == torch.device(
        "cpu"
    )


def test_get_module_dtype_prefers_floating_parameter_dtype(monkeypatch):
    monkeypatch.setattr(transformers_compat, "_hf_get_parameter_dtype", None)
    module = ParameterModule()

    assert transformers_compat.get_module_dtype(module) == torch.float16


def test_get_module_dtype_prefers_floating_buffer_over_integer_buffer(
    monkeypatch,
):
    monkeypatch.setattr(transformers_compat, "_hf_get_parameter_dtype", None)
    module = BufferOnlyModule()

    assert transformers_compat.get_module_dtype(module) == torch.float64


def test_get_module_dtype_returns_integer_fallback_when_no_floating_tensor(
    monkeypatch,
):
    monkeypatch.setattr(transformers_compat, "_hf_get_parameter_dtype", None)
    module = IntBufferOnlyModule()

    assert transformers_compat.get_module_dtype(module) == torch.int64


def test_get_module_dtype_falls_back_to_float32_for_empty_module(monkeypatch):
    monkeypatch.setattr(transformers_compat, "_hf_get_parameter_dtype", None)

    assert transformers_compat.get_module_dtype(nn.Module()) == torch.float32


def test_batch_encode_tokenizer_prefers_legacy_batch_encode_plus():
    tokenizer = LegacyTokenizer()

    output = transformers_compat.batch_encode_tokenizer(
        tokenizer,
        ["hello"],
        padding="max_length",
    )

    assert output == "legacy"
    assert tokenizer.calls == [
        ("batch_encode_plus", (["hello"],), {"padding": "max_length"})
    ]


def test_batch_encode_tokenizer_falls_back_to_callable_tokenizer():
    tokenizer = CallableTokenizer()

    output = transformers_compat.batch_encode_tokenizer(
        tokenizer,
        ["hello"],
        padding="max_length",
    )

    assert output == "modern"
    assert tokenizer.calls == [
        ("call", (["hello"],), {"padding": "max_length"})
    ]


def test_prepare_hf_attention_mask_keeps_legacy_3d_mask():
    mask = torch.eye(3, dtype=torch.bool).unsqueeze(0)

    prepared = transformers_compat.prepare_hf_attention_mask(
        nn.Module(),
        mask,
        (1, 3),
    )

    assert prepared is mask


def test_prepare_hf_attention_mask_extends_modern_3d_mask():
    model = ModernAttentionMaskModel()
    mask = torch.eye(3, dtype=torch.bool).unsqueeze(0)

    prepared = transformers_compat.prepare_hf_attention_mask(
        model,
        mask,
        (1, 3),
    )

    assert prepared.shape == (1, 1, 3, 3)
    assert prepared.dtype == torch.float16
    assert model.calls == [(mask, (1, 3), torch.float16)]


def test_prepare_hf_attention_mask_keeps_2d_mask_for_modern_model():
    model = ModernAttentionMaskModel()
    mask = torch.ones((1, 3), dtype=torch.bool)

    prepared = transformers_compat.prepare_hf_attention_mask(
        model,
        mask,
        (1, 3),
    )

    assert prepared is mask
    assert model.calls == []


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("4.49.0", "torch_dtype"),
        ("4.55.2", "torch_dtype"),
        ("4.56.0", "dtype"),
        ("5.8.0", "dtype"),
    ],
)
def test_runtime_hf_dtype_kwarg_name(version, expected, monkeypatch):
    monkeypatch.setattr(
        transformers_compat.transformers, "__version__", version
    )

    assert transformers_compat.runtime_hf_dtype_kwarg_name() == expected


def test_normalize_hf_dtype_kwargs_uses_runtime_alias(monkeypatch):
    monkeypatch.setattr(
        transformers_compat,
        "runtime_hf_dtype_kwarg_name",
        lambda: "torch_dtype",
    )

    normalized = transformers_compat.normalize_hf_dtype_kwargs(
        {"dtype": torch.float16, "device_map": "auto"}
    )

    assert normalized == {
        "torch_dtype": torch.float16,
        "device_map": "auto",
    }


def test_normalize_hf_dtype_kwargs_rejects_conflicting_aliases():
    with pytest.raises(ValueError, match="must match"):
        transformers_compat.normalize_hf_dtype_kwargs(
            {"dtype": torch.float16, "torch_dtype": torch.float32}
        )
