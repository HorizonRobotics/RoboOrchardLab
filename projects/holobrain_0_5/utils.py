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


"""Local configuration and strict action-checkpoint loading."""

import importlib.util
import json
from pathlib import Path

import torch
from safetensors.torch import load_model

from robo_orchard_lab.utils.huggingface import resolve_hf_compatible_path

__all__ = [
    "load_config",
    "parse_kwargs",
    "load_checkpoint",
    "resolve_processor_dir",
]


def resolve_processor_dir(model_dir, processor_name):
    """Find the directory holding a processor JSON.

    Training keeps per-run artifacts at the workspace root but per-step weights
    in ``checkpoints/checkpoint_<n>``, so a checkpoint directory is tried first
    and the workspace two levels up second.

    Args:
        model_dir (str | Path): Directory holding the model weights.
        processor_name (str): Processor file name, including ``.json``.

    Returns:
        Path: The directory containing ``processor_name``.
    """
    model_dir = Path(model_dir)
    candidates = [model_dir, *model_dir.parents[1:2]]
    for candidate in candidates:
        if (candidate / processor_name).is_file():
            return candidate
    tried = "\n".join(f"  - {c / processor_name}" for c in candidates)
    raise FileNotFoundError(
        f"No {processor_name} for model dir {model_dir}. Tried:\n{tried}"
    )


def load_config(path):
    """Load a Python model configuration from an explicit local path."""
    spec = importlib.util.spec_from_file_location("holobrain_0_5_config", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Invalid config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_kwargs(raw):
    """Read overrides from a JSON object or a local JSON file."""
    if raw is None:
        return {}
    value = json.loads(
        Path(raw).read_text() if not raw.lstrip().startswith("{") else raw
    )
    if not isinstance(value, dict):
        raise ValueError("Config overrides must be a JSON object")
    return value


def load_checkpoint(model, checkpoint=None):
    """Load an action checkpoint strictly; incompatible weights fail.

    An ``hf://`` URI is downloaded first; any other value is a local path.
    """
    if checkpoint is None:
        return
    resolved = resolve_hf_compatible_path(checkpoint)
    path = Path(resolved)
    if not path.is_file():
        location = (
            checkpoint
            if resolved == checkpoint
            else f"{checkpoint} (resolved to {resolved})"
        )
        raise FileNotFoundError(f"Expected a checkpoint file: {location}")
    if path.suffix == ".safetensors":
        load_model(model, str(path), strict=True)
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("state_dict", state), strict=True)
