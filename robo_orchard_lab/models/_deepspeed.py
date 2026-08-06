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

"""Layerwise safetensors loading for DeepSpeed ZeRO-3 models."""

from __future__ import annotations
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file as safetensors_load_file

__all__ = [
    "_has_zero3_partitioned_parameters",
    "_load_model_weights_into_zero3_model",
]


def _has_zero3_partitioned_parameters(model: torch.nn.Module) -> bool:
    """Return whether DeepSpeed ZeRO-3 manages any model parameter."""

    return any(hasattr(parameter, "ds_id") for parameter in model.parameters())


def _load_model_weights_into_zero3_model(
    model: torch.nn.Module,
    model_weights_path: str | os.PathLike[str],
    *,
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    """Load an ordinary safetensors artifact into a ZeRO-3 model.

    This follows the narrow layerwise contract used by Transformers' ZeRO-3
    loader: the model's public ``state_dict()`` key set is the checkpoint
    contract, and every rank visits modules in the same order. Partitioned
    parameters are gathered with ``modifier_rank=0`` before the copy on the
    rank-zero member of each parameter's ZeRO data-parallel process group;
    ordinary parameters and buffers are copied directly on every rank.

    The loader does not invoke a model's custom ``load_state_dict`` method or
    emulate load-state hooks. A model that registers load pre/post hooks must
    use the ordinary loading path. Tied aliases are restored only when the
    target exposes the names as the same persistent ``Parameter`` or buffer
    object. Safetensors metadata, repository alias sidecars, arbitrary key
    remapping, publication, staging, cleanup, and offloaded-writer provenance
    are outside this loading contract.

    Args:
        model: Model already initialized under DeepSpeed ZeRO-3.
        model_weights_path: Safetensors model weights file, index file, or
            directory.
        strict: Whether missing or unexpected keys against the model's public
            ``state_dict()`` contract are errors.

    Returns:
        Sorted missing and unexpected public state-dict keys. Logical tensor
        shape and copy errors always raise before distributed loading starts.

    Raises:
        ValueError: If the model has no ZeRO-3 parameters, registers load
            hooks, or the weights path is ambiguous or malformed.
        RuntimeError: If DeepSpeed is unavailable or weights cannot be loaded.
    """

    if not _has_zero3_partitioned_parameters(model):
        raise ValueError(
            "The model has no DeepSpeed ZeRO-3 managed parameters."
        )
    if any(
        module._load_state_dict_pre_hooks or module._load_state_dict_post_hooks
        for module in model.modules()
    ):
        raise ValueError(
            "DeepSpeed ZeRO-3 loading does not support registered "
            "load_state_dict pre/post hooks."
        )

    try:
        import deepspeed  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError(
            "Loading weights into a ZeRO-3 initialized model requires the "
            "optional DeepSpeed dependency."
        ) from exc

    public_state = model.state_dict(keep_vars=True)
    public_state_keys = set(public_state)
    state_owners: dict[str, torch.Tensor] = {
        name: parameter
        for name, parameter in model.named_parameters(remove_duplicate=False)
    }
    for module_prefix, module in model.named_modules(remove_duplicate=False):
        for name, buffer in module._buffers.items():
            if buffer is None or name in module._non_persistent_buffers_set:
                continue
            qualified_name = (
                f"{module_prefix}.{name}" if module_prefix else name
            )
            state_owners[qualified_name] = buffer
    supported_state_keys = set(state_owners)
    unsupported_public_keys = public_state_keys - supported_state_keys
    if unsupported_public_keys:
        raise ValueError(
            "DeepSpeed ZeRO-3 loading requires every public state_dict key "
            "to map directly to a persistent parameter or buffer; "
            "unsupported key(s): " + ", ".join(sorted(unsupported_public_keys))
        )
    meta_public_keys = sorted(
        key
        for key in public_state_keys
        if not hasattr(state_owners[key], "ds_id")
        and getattr(state_owners[key], "is_meta", False)
    )
    if meta_public_keys:
        raise ValueError(
            "DeepSpeed ZeRO-3 loading does not support ordinary meta-device "
            "parameters or buffers; use the meta-device dispatch path for: "
            + ", ".join(meta_public_keys)
        )

    state_dict = _load_safetensors_state_dict(model_weights_path)
    _restore_tied_weight_aliases(
        model, state_dict, allowed_keys=public_state_keys
    )
    public_state_dict = {
        key: value
        for key, value in state_dict.items()
        if key in public_state_keys
    }

    shape_errors = []
    for key in sorted(public_state_keys & state_dict.keys()):
        expected = state_owners[key]
        actual = state_dict[key]
        if not isinstance(actual, torch.Tensor):
            continue
        expected_shape = getattr(expected, "ds_shape", None)
        if expected_shape is None or not hasattr(expected, "ds_id"):
            expected_shape = expected.shape
        if torch.Size(expected_shape) != actual.shape:
            shape_errors.append(
                f"size mismatch for {key}: copying a param with shape "
                f"{tuple(actual.shape)} from checkpoint, the shape in "
                f"current model is {tuple(expected_shape)}"
            )
    if shape_errors:
        raise RuntimeError(
            f"Error(s) in loading state_dict for "
            f"{model.__class__.__name__}:\n\t" + "\n\t".join(shape_errors)
        )

    missing_keys = public_state_keys - state_dict.keys()
    unexpected_keys = set(state_dict) - public_state_keys
    error_messages: list[str] = []

    def is_modifier_rank_zero(
        managed_parameters: list[torch.nn.Parameter],
    ) -> bool:
        process_group = getattr(
            managed_parameters[0], "ds_process_group", None
        )
        if process_group is not None and hasattr(deepspeed, "comm"):
            return deepspeed.comm.get_rank(process_group) == 0
        return not torch.distributed.is_initialized() or (
            torch.distributed.get_rank() == 0
        )

    def load_module(module: torch.nn.Module, prefix: str) -> None:
        local_metadata = {"assign_to_params_buffers": False}
        load_args = (
            public_state_dict,
            prefix,
            local_metadata,
            True,
            [],
            [],
            error_messages,
        )
        named_parameters = dict(
            module.named_parameters(
                prefix=prefix[:-1], recurse=False, remove_duplicate=False
            )
        )
        managed_parameters = [
            parameter
            for key, parameter in named_parameters.items()
            if (
                key in public_state_keys
                and key in state_dict
                and hasattr(parameter, "ds_id")
            )
        ]
        if managed_parameters:
            with deepspeed.zero.GatheredParameters(
                managed_parameters, modifier_rank=0
            ):
                if is_modifier_rank_zero(managed_parameters):
                    module._load_from_state_dict(*load_args)
            for key in named_parameters:
                if (
                    key in public_state_keys
                    and key in state_dict
                    and hasattr(named_parameters[key], "ds_id")
                ):
                    missing_keys.discard(key)

        for key, parameter in named_parameters.items():
            if (
                key in public_state_keys
                and key in state_dict
                and not hasattr(parameter, "ds_id")
            ):
                with torch.no_grad():
                    parameter.copy_(state_dict[key])
                missing_keys.discard(key)

        named_buffers = dict(
            module.named_buffers(
                prefix=prefix[:-1], recurse=False, remove_duplicate=False
            )
        )
        for key, buffer in named_buffers.items():
            if (
                key in public_state_keys
                and key in state_dict
                and buffer is not None
            ):
                with torch.no_grad():
                    buffer.copy_(state_dict[key])
                missing_keys.discard(key)

        for child_name, child in module._modules.items():
            if child is not None:
                load_module(child, prefix + child_name + ".")

    load_module(model, "")

    missing_keys = sorted(missing_keys)
    unexpected_keys = sorted(unexpected_keys)
    if strict and missing_keys:
        error_messages.append(
            "Missing key(s) in state_dict: "
            + ", ".join(f'"{key}"' for key in missing_keys)
        )
    if strict and unexpected_keys:
        error_messages.append(
            "Unexpected key(s) in state_dict: "
            + ", ".join(f'"{key}"' for key in unexpected_keys)
        )
    if error_messages:
        details = "\n\t".join(error_messages)
        raise RuntimeError(
            f"Error(s) in loading state_dict for "
            f"{model.__class__.__name__}:\n\t{details}"
        )

    return missing_keys, unexpected_keys


def _load_safetensors_state_dict(
    model_weights_path: str | os.PathLike[str],
) -> dict[str, torch.Tensor]:
    """Load selected safetensors files into a CPU state dictionary."""

    weight_files = _resolve_safetensors_weight_files(model_weights_path)
    state_dict: dict[str, torch.Tensor] = {}
    for weight_file in weight_files:
        shard = safetensors_load_file(str(weight_file), device="cpu")
        duplicate_keys = state_dict.keys() & shard.keys()
        if duplicate_keys:
            duplicates = ", ".join(sorted(duplicate_keys))
            raise ValueError(
                f"Duplicate state-dict keys across safetensors files: "
                f"{duplicates}"
            )
        state_dict.update(shard)
    return state_dict


def _resolve_safetensors_weight_files(
    model_weights_path: str | os.PathLike[str],
) -> list[Path]:
    """Resolve one unambiguous safetensors layout."""

    path = Path(model_weights_path)
    if not path.exists():
        raise FileNotFoundError(f"Model weights path does not exist: {path}")

    if path.is_file():
        if path.name.endswith(".safetensors.index.json"):
            return _resolve_indexed_weight_files(path)
        if path.suffix != ".safetensors":
            raise ValueError(
                "ZeRO-3 model weight loading supports safetensors files "
                f"only, got: {path}"
            )
        sibling_weight_files = sorted(path.parent.glob("*.safetensors"))
        sibling_index_files = sorted(
            path.parent.glob("*.safetensors.index.json")
        )
        other_weight_files = [
            weight_file
            for weight_file in sibling_weight_files
            if weight_file.resolve() != path.resolve()
        ]
        if other_weight_files or sibling_index_files:
            siblings = other_weight_files + sibling_index_files
            raise ValueError(
                "Safetensors directory contains a mixed or ambiguous layout "
                f"beside {path.name}: "
                + ", ".join(sibling.name for sibling in siblings)
            )
        return [path]

    index_files = sorted(path.glob("*.safetensors.index.json"))
    if len(index_files) > 1:
        raise ValueError(
            f"Multiple safetensors index files found under {path}: "
            + ", ".join(index.name for index in index_files)
        )
    if index_files:
        return _resolve_indexed_weight_files(index_files[0])

    weight_files = sorted(path.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(
            f"No safetensors model weights found under {path}"
        )
    if len(weight_files) > 1:
        raise ValueError(
            f"Multiple safetensors files found under {path} without an "
            "index: "
            + ", ".join(weight_file.name for weight_file in weight_files)
        )
    return weight_files


def _resolve_indexed_weight_files(index_path: Path) -> list[Path]:
    """Resolve an index and reject sibling safetensors it does not name."""

    indexed_weight_files = _weight_files_from_index(index_path)
    indexed_weight_paths = {
        weight_file.resolve() for weight_file in indexed_weight_files
    }
    sibling_index_files = sorted(
        sibling
        for sibling in index_path.parent.glob("*.safetensors.index.json")
        if sibling.resolve() != index_path.resolve()
    )
    if sibling_index_files:
        raise ValueError(
            "Multiple safetensors index files found beside "
            f"{index_path.name}: "
            + ", ".join(index.name for index in sibling_index_files)
        )
    unreferenced_weight_files = sorted(
        weight_file
        for weight_file in index_path.parent.glob("*.safetensors")
        if weight_file.resolve() not in indexed_weight_paths
    )
    if unreferenced_weight_files:
        raise ValueError(
            "Safetensors directory contains unreferenced weight files "
            f"beside {index_path.name}: "
            + ", ".join(
                weight_file.name for weight_file in unreferenced_weight_files
            )
        )
    return indexed_weight_files


def _weight_files_from_index(index_path: Path) -> list[Path]:
    """Validate an index and return its unique in-directory shard paths."""

    try:
        with index_path.open("r", encoding="utf-8") as file:
            index = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Safetensors index is unreadable: {index_path}"
        ) from exc
    if not isinstance(index, dict):
        raise ValueError(
            f"Safetensors index must contain a JSON object: {index_path}"
        )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(
            f"Safetensors index has no non-empty weight_map: {index_path}"
        )
    if not all(
        isinstance(key, str) and isinstance(filename, str)
        for key, filename in weight_map.items()
    ):
        raise ValueError(
            f"Safetensors index weight_map is malformed: {index_path}"
        )
    if not all(
        filename.endswith(".safetensors") for filename in weight_map.values()
    ):
        raise ValueError(
            "Safetensors index references a non-safetensors shard: "
            f"{index_path}"
        )

    root = index_path.parent.resolve()
    weight_files: list[Path] = []
    for filename in dict.fromkeys(weight_map.values()):
        weight_file = (index_path.parent / filename).resolve()
        if not weight_file.is_relative_to(root):
            raise ValueError(
                f"Safetensors shard escapes its index directory: {filename}"
            )
        if not weight_file.is_file():
            raise FileNotFoundError(
                f"Safetensors shard referenced by {index_path} does not "
                f"exist: {weight_file}"
            )
        weight_files.append(weight_file)
    return weight_files


def _restore_tied_weight_aliases(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    allowed_keys: set[str] | None = None,
) -> None:
    """Fill omitted keys for aliases represented by the same model object.

    This is target-model-only compatibility. Shared storage between distinct
    objects, safetensors metadata, and repository alias sidecars are not
    treated as provenance. Non-persistent buffers are excluded because
    PyTorch intentionally omits them from ``state_dict()``.
    """

    aliases_by_tensor: defaultdict[int, list[str]] = defaultdict(list)
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases_by_tensor[id(parameter)].append(name)
    for module_prefix, module in model.named_modules(remove_duplicate=False):
        for name, buffer in module._buffers.items():
            if buffer is None or name in module._non_persistent_buffers_set:
                continue
            qualified_name = (
                f"{module_prefix}.{name}" if module_prefix else name
            )
            aliases_by_tensor[id(buffer)].append(qualified_name)

    for aliases in aliases_by_tensor.values():
        loaded_alias = next(
            (
                name
                for name in aliases
                if name in state_dict
                and (allowed_keys is None or name in allowed_keys)
            ),
            None,
        )
        if loaded_alias is None:
            continue
        for alias in aliases:
            if allowed_keys is None or alias in allowed_keys:
                state_dict.setdefault(alias, state_dict[loaded_alias])
