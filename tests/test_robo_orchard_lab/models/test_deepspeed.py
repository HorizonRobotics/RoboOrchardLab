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
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import (
    save_file as safetensors_save_file,
    save_model as safetensors_save_model,
)

from robo_orchard_lab.models._deepspeed import (
    _has_zero3_partitioned_parameters,
    _load_model_weights_into_zero3_model,
    _resolve_safetensors_weight_files,
)


class _MixedTiedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(3, 2)
        self.decoder = torch.nn.Linear(3, 2)
        self.decoder.weight = self.encoder.weight
        self.register_buffer("scale", torch.ones(2))


class _ShardedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Linear(3, 2)
        self.second = torch.nn.Linear(2, 1)


class _MetadataModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.source = torch.nn.Parameter(torch.ones(2))
        self.alias = torch.nn.Parameter(torch.full((2,), 9.0))


class _PersistentBufferAliasModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.register_buffer("persisted", torch.full((2,), 3.0))
        self.register_buffer("ephemeral", self.persisted, persistent=False)


class _FakeGatheredParameters:
    calls: list[tuple[list[torch.nn.Parameter], int | None]] = []

    def __init__(
        self,
        parameters: list[torch.nn.Parameter],
        modifier_rank: int | None = None,
    ) -> None:
        self.parameters = list(parameters)
        self.modifier_rank = modifier_rank

    def __enter__(self) -> None:
        for parameter in self.parameters:
            ds_shape = getattr(parameter, "ds_shape", None)
            if ds_shape is not None and parameter.shape != torch.Size(
                ds_shape
            ):
                parameter.data = torch.empty(
                    tuple(ds_shape),
                    dtype=parameter.dtype,
                    device=parameter.device,
                )
        self.calls.append((self.parameters, self.modifier_rank))

    def __exit__(self, *args: object) -> None:
        return None


class _FakeDeepSpeedComm:
    rank = 0
    groups: list[object] = []

    @classmethod
    def get_rank(cls, group: object = None) -> int:
        cls.groups.append(group)
        return cls.rank


@pytest.fixture
def fake_deepspeed(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeGatheredParameters.calls.clear()
    _FakeDeepSpeedComm.rank = 0
    _FakeDeepSpeedComm.groups.clear()
    module = SimpleNamespace(
        zero=SimpleNamespace(GatheredParameters=_FakeGatheredParameters),
        comm=SimpleNamespace(get_rank=_FakeDeepSpeedComm.get_rank),
    )
    monkeypatch.setitem(sys.modules, "deepspeed", module)


def _mark_zero3(parameter: torch.nn.Parameter, ds_id: int) -> None:
    parameter.ds_id = ds_id  # pyright: ignore[reportAttributeAccessIssue]


def test_deepspeed_helpers_import_without_deepspeed() -> None:
    code = """
import builtins
import importlib.util

original_import = builtins.__import__
original_find_spec = importlib.util.find_spec

def guarded_import(name, *args, **kwargs):
    level = kwargs.get("level", args[3] if len(args) > 3 else 0)
    if level == 0 and (name == "deepspeed" or name.startswith("deepspeed.")):
        raise AssertionError("DeepSpeed was imported eagerly")
    return original_import(name, *args, **kwargs)

def find_spec_without_deepspeed(name, *args, **kwargs):
    if name == "deepspeed" or name.startswith("deepspeed."):
        return None
    return original_find_spec(name, *args, **kwargs)

builtins.__import__ = guarded_import
importlib.util.find_spec = find_spec_without_deepspeed
import robo_orchard_lab.models._deepspeed
"""

    subprocess.run([sys.executable, "-c", code], check=True)


def test_detects_zero3_parameters_without_importing_deepspeed() -> None:
    model = _MixedTiedModel()

    assert not _has_zero3_partitioned_parameters(model)

    _mark_zero3(model.encoder.weight, ds_id=0)

    assert _has_zero3_partitioned_parameters(model)


def test_loads_single_file_with_tied_and_regular_state(
    tmp_path, fake_deepspeed: None
) -> None:
    torch.manual_seed(7)
    source = _MixedTiedModel()
    source.scale.fill_(3)
    weights_path = tmp_path / "model.safetensors"
    safetensors_save_model(source, weights_path)

    target = _MixedTiedModel()
    for parameter in target.parameters():
        parameter.data.zero_()
    target.scale.zero_()
    _mark_zero3(target.encoder.weight, ds_id=0)

    missing, unexpected = _load_model_weights_into_zero3_model(
        target, weights_path
    )

    assert missing == []
    assert unexpected == []
    assert target.encoder.weight is target.decoder.weight
    for name, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], value)
    assert _FakeGatheredParameters.calls
    assert all(
        modifier_rank == 0
        for _, modifier_rank in _FakeGatheredParameters.calls
    )


class _SharedBufferChild(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.full((2,), 4.0))


class _SharedBufferModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        child = _SharedBufferChild()
        self.left = child
        self.right = child


class _PublicStateDictModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kept = torch.nn.Linear(2, 2)
        self.omitted = torch.nn.Linear(2, 2)

    def state_dict(
        self,
        *args: object,
        destination: dict[str, torch.Tensor] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, torch.Tensor]:
        state_dict = super().state_dict(
            *args,
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
        )
        return {
            key: value
            for key, value in state_dict.items()
            if not key.startswith(f"{prefix}omitted.")
        }


class _RenamedPublicStateDictModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))

    def state_dict(
        self,
        *args: object,
        destination: dict[str, torch.Tensor] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, torch.Tensor]:
        del args, destination
        del prefix, keep_vars
        return {"renamed": self.weight}


class _OmittedTiedAliasModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.source = torch.nn.Parameter(torch.ones(2))
        self.alias = self.source

    def state_dict(
        self,
        *args: object,
        destination: dict[str, torch.Tensor] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, torch.Tensor]:
        state_dict = super().state_dict(
            *args,
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
        )
        state_dict.pop(f"{prefix}source")
        return state_dict


def test_loads_safetensors_shard_index(tmp_path, fake_deepspeed: None) -> None:
    torch.manual_seed(11)
    source = _ShardedModel()
    state_dict = source.state_dict()
    first_shard = "model-00001-of-00002.safetensors"
    second_shard = "model-00002-of-00002.safetensors"
    safetensors_save_file(
        {
            "first.weight": state_dict["first.weight"],
            "first.bias": state_dict["first.bias"],
        },
        tmp_path / first_shard,
    )
    safetensors_save_file(
        {
            "second.weight": state_dict["second.weight"],
            "second.bias": state_dict["second.bias"],
        },
        tmp_path / second_shard,
    )
    index = {
        "metadata": {},
        "weight_map": {
            "first.weight": first_shard,
            "first.bias": first_shard,
            "second.weight": second_shard,
            "second.bias": second_shard,
        },
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    target = _ShardedModel()
    for parameter in target.parameters():
        parameter.data.zero_()
    _mark_zero3(target.first.weight, ds_id=0)

    missing, unexpected = _load_model_weights_into_zero3_model(
        target, tmp_path
    )

    assert missing == []
    assert unexpected == []
    for name, value in state_dict.items():
        torch.testing.assert_close(target.state_dict()[name], value)


def test_rejects_unreferenced_safetensors_beside_index(tmp_path) -> None:
    shard_name = "model-00001-of-00001.safetensors"
    safetensors_save_file({"weight": torch.ones(2)}, tmp_path / shard_name)
    safetensors_save_file(
        {"stale": torch.ones(2)}, tmp_path / "model.safetensors"
    )
    index = {
        "metadata": {},
        "weight_map": {"weight": shard_name},
    }
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index))

    with pytest.raises(ValueError, match="unreferenced weight files"):
        _resolve_safetensors_weight_files(tmp_path)
    with pytest.raises(ValueError, match="unreferenced weight files"):
        _resolve_safetensors_weight_files(index_path)


def test_rejects_sibling_layout_for_explicit_single_file(tmp_path) -> None:
    selected_path = tmp_path / "model.safetensors"
    safetensors_save_file({"weight": torch.ones(2)}, selected_path)
    safetensors_save_file(
        {"stale": torch.ones(2)}, tmp_path / "model-00001-of-00001.safetensors"
    )

    with pytest.raises(ValueError, match="mixed or ambiguous layout"):
        _resolve_safetensors_weight_files(selected_path)


def test_ignores_informational_safetensors_metadata_for_aliases(
    tmp_path, fake_deepspeed: None
) -> None:
    source = _MetadataModel()
    weights_path = tmp_path / "model.safetensors"
    safetensors_save_file(
        {"source": source.source},
        weights_path,
        metadata={"alias": "source"},
    )

    target = _MetadataModel()
    _mark_zero3(target.source, ds_id=0)

    missing, unexpected = _load_model_weights_into_zero3_model(
        target, weights_path, strict=False
    )
    assert missing == ["alias"]
    assert unexpected == []
    torch.testing.assert_close(target.alias, torch.full((2,), 9.0))

    strict_target = _MetadataModel()
    _mark_zero3(strict_target.source, ds_id=0)
    with pytest.raises(RuntimeError, match="Missing key.*alias"):
        _load_model_weights_into_zero3_model(strict_target, weights_path)


def test_ignores_non_persistent_buffer_aliases(
    tmp_path, fake_deepspeed: None
) -> None:
    source = _PersistentBufferAliasModel()
    weights_path = tmp_path / "model.safetensors"
    safetensors_save_file(source.state_dict(), weights_path)

    target = _PersistentBufferAliasModel()
    target.weight.data.zero_()
    target.persisted.zero_()
    _mark_zero3(target.weight, ds_id=0)

    missing, unexpected = _load_model_weights_into_zero3_model(
        target, weights_path
    )

    assert missing == []
    assert unexpected == []
    torch.testing.assert_close(target.persisted, source.persisted)
    assert target.ephemeral is target.persisted


def test_restores_persistent_buffer_aliases_through_shared_submodule(
    tmp_path, fake_deepspeed: None
) -> None:
    source = _SharedBufferModel()
    weights_path = tmp_path / "model.safetensors"
    safetensors_save_model(source, weights_path)

    target = _SharedBufferModel()
    target.weight.data.zero_()
    target.left.scale.zero_()
    _mark_zero3(target.weight, ds_id=0)

    missing, unexpected = _load_model_weights_into_zero3_model(
        target, weights_path
    )

    assert missing == []
    assert unexpected == []
    torch.testing.assert_close(target.weight, source.weight)
    torch.testing.assert_close(target.left.scale, source.left.scale)
    assert target.left is target.right


def test_honors_public_state_dict_omissions(
    tmp_path, fake_deepspeed: None
) -> None:
    source = _PublicStateDictModel()
    weights_path = tmp_path / "model.safetensors"
    safetensors_save_file(source.state_dict(), weights_path)

    target = _PublicStateDictModel()
    target.kept.weight.data.zero_()
    target.kept.bias.data.zero_()
    target.omitted.weight.data.fill_(9.0)
    target.omitted.bias.data.fill_(9.0)
    _mark_zero3(target.kept.weight, ds_id=0)

    missing, unexpected = _load_model_weights_into_zero3_model(
        target, weights_path
    )

    assert missing == []
    assert unexpected == []
    torch.testing.assert_close(target.kept.weight, source.kept.weight)
    torch.testing.assert_close(target.kept.bias, source.kept.bias)
    torch.testing.assert_close(
        target.omitted.weight, torch.full_like(target.omitted.weight, 9.0)
    )


def test_rejects_public_state_dict_keys_without_registered_owner(
    tmp_path, fake_deepspeed: None
) -> None:
    model = _RenamedPublicStateDictModel()
    _mark_zero3(model.weight, ds_id=0)
    weights_path = tmp_path / "model.safetensors"
    safetensors_save_file({"renamed": torch.ones(2)}, weights_path)

    with pytest.raises(ValueError, match="map directly"):
        _load_model_weights_into_zero3_model(model, weights_path)


def test_rejects_broadcastable_shape_mismatch(
    tmp_path, fake_deepspeed: None
) -> None:
    model = _ShardedModel()
    _mark_zero3(model.first.weight, ds_id=0)
    state_dict = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    state_dict["second.weight"] = torch.ones(1, 1)
    weights_path = tmp_path / "model.safetensors"
    safetensors_save_file(state_dict, weights_path)

    with pytest.raises(RuntimeError, match="size mismatch.*second.weight"):
        _load_model_weights_into_zero3_model(model, weights_path)
    assert _FakeGatheredParameters.calls == []


def test_uses_zero_rank_of_parameter_process_group(
    tmp_path, fake_deepspeed: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _ShardedModel()
    weights_path = tmp_path / "model.safetensors"
    safetensors_save_file(source.state_dict(), weights_path)

    target = _ShardedModel()
    target.first.weight.data.zero_()
    _mark_zero3(target.first.weight, ds_id=0)
    process_group = object()
    target.first.weight.ds_process_group = process_group  # type: ignore[attr-defined]
    _FakeDeepSpeedComm.rank = 0
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 2)

    _load_model_weights_into_zero3_model(target, weights_path)

    torch.testing.assert_close(target.first.weight, source.first.weight)
    assert _FakeDeepSpeedComm.groups == [process_group]


def test_rejects_ordinary_meta_parameter(
    tmp_path, fake_deepspeed: None
) -> None:
    source = _ShardedModel()
    weights_path = tmp_path / "model.safetensors"
    safetensors_save_file(source.state_dict(), weights_path)

    target = _ShardedModel()
    target.second.weight = torch.nn.Parameter(
        torch.empty_like(target.second.weight, device="meta")
    )
    _mark_zero3(target.first.weight, ds_id=0)

    with pytest.raises(ValueError, match="meta-device"):
        _load_model_weights_into_zero3_model(target, weights_path)
    assert target.second.weight.is_meta
    assert _FakeGatheredParameters.calls == []


def test_does_not_restore_omitted_tied_alias_keys(
    tmp_path, fake_deepspeed: None
) -> None:
    source = _OmittedTiedAliasModel()
    weights_path = tmp_path / "model.safetensors"
    safetensors_save_file(source.state_dict(), weights_path)

    target = _OmittedTiedAliasModel()
    target.source.data.zero_()
    _mark_zero3(target.source, ds_id=0)

    missing, unexpected = _load_model_weights_into_zero3_model(
        target, weights_path
    )

    assert missing == []
    assert unexpected == []
    torch.testing.assert_close(target.source, source.source)
    assert target.alias is target.source


def test_loads_public_tied_alias_when_parameter_storage_is_partitioned(
    tmp_path, fake_deepspeed: None
) -> None:
    source = _OmittedTiedAliasModel()
    weights_path = tmp_path / "model.safetensors"
    safetensors_save_file(source.state_dict(), weights_path)

    target = _OmittedTiedAliasModel()
    target.source.data.zero_()
    target.source.ds_shape = tuple(  # type: ignore[attr-defined]
        target.source.shape
    )
    target.source.data = torch.empty(0)
    _mark_zero3(target.source, ds_id=0)

    missing, unexpected = _load_model_weights_into_zero3_model(
        target, weights_path
    )

    assert missing == []
    assert unexpected == []
    torch.testing.assert_close(target.source, source.source)
    assert target.alias is target.source


def test_does_not_use_unexpected_alias_as_public_alias_source(
    tmp_path, fake_deepspeed: None
) -> None:
    source = _OmittedTiedAliasModel()
    weights_path = tmp_path / "model.safetensors"
    safetensors_save_file({"source": source.source}, weights_path)

    target = _OmittedTiedAliasModel()
    target.source.data.zero_()
    _mark_zero3(target.source, ds_id=0)

    missing, unexpected = _load_model_weights_into_zero3_model(
        target, weights_path, strict=False
    )

    assert missing == ["alias"]
    assert unexpected == ["source"]
    torch.testing.assert_close(target.source, torch.zeros_like(target.source))


def test_reports_unexpected_keys_in_strict_and_non_strict_modes(
    tmp_path, fake_deepspeed: None
) -> None:
    model = _ShardedModel()
    _mark_zero3(model.first.weight, ds_id=0)
    state_dict = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    state_dict["unexpected"] = torch.ones(1)
    weights_path = tmp_path / "model.safetensors"
    safetensors_save_file(state_dict, weights_path)

    missing, unexpected = _load_model_weights_into_zero3_model(
        model, weights_path, strict=False
    )

    assert missing == []
    assert unexpected == ["unexpected"]

    strict_model = _ShardedModel()
    _mark_zero3(strict_model.first.weight, ds_id=0)
    with pytest.raises(RuntimeError, match="Unexpected key.*unexpected"):
        _load_model_weights_into_zero3_model(strict_model, weights_path)


def test_rejects_registered_load_state_hooks(
    tmp_path, fake_deepspeed: None
) -> None:
    source = _ShardedModel()
    weights_path = tmp_path / "model.safetensors"
    safetensors_save_file(source.state_dict(), weights_path)

    target = _ShardedModel()
    _mark_zero3(target.first.weight, ds_id=0)
    target.register_load_state_dict_post_hook(lambda *_args: None)

    with pytest.raises(ValueError, match="does not support registered"):
        _load_model_weights_into_zero3_model(target, weights_path)


def test_strict_false_reports_missing_keys(
    tmp_path, fake_deepspeed: None
) -> None:
    model = _ShardedModel()
    _mark_zero3(model.first.weight, ds_id=0)
    weights_path = tmp_path / "model.safetensors"
    safetensors_save_file(
        {"first.weight": torch.ones_like(model.first.weight)}, weights_path
    )

    missing, unexpected = _load_model_weights_into_zero3_model(
        model, weights_path, strict=False
    )

    assert missing == ["first.bias", "second.bias", "second.weight"]
    assert unexpected == []


def test_strict_true_rejects_missing_keys(
    tmp_path, fake_deepspeed: None
) -> None:
    model = _ShardedModel()
    _mark_zero3(model.first.weight, ds_id=0)
    weights_path = tmp_path / "model.safetensors"
    safetensors_save_file(
        {"first.weight": torch.ones_like(model.first.weight)}, weights_path
    )

    with pytest.raises(RuntimeError, match="Missing key.*second.weight"):
        _load_model_weights_into_zero3_model(model, weights_path)
