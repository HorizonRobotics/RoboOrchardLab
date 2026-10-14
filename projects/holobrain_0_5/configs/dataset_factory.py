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


"""Build only the action datasets selected in a HoloBrain 0.5 config."""

import importlib

import torch

from robo_orchard_lab.dataset.collates import collate_batch_dict
from robo_orchard_lab.dataset.dataset_wrapper import ConcatDatasetWithFlag

__all__ = [
    "build_training_dataset",
    "build_validation_dataset",
    "build_processors",
    "collate_action_batch",
]


def _specs(config, mode):
    module = importlib.import_module(config["dataset_specs"])
    available = {x["dataset_name"]: x for x in module.training_datasets}
    selected = config.get(f"{mode}_datasets", [])
    if not selected and mode == "training":
        raise ValueError("Select at least one training dataset")
    for name in selected:
        if name not in available:
            raise ValueError(
                f"Unknown dataset {name!r}; choices: {list(available)}"
            )
        spec = available[name].copy()
        paths = config.get("data_paths", {}).get(name)
        if paths is not None:
            spec["data_paths"] = paths
        yield spec


def _builder(data_type):
    if data_type not in ("robotwin", "libero"):
        raise ValueError(f"Unsupported dataset type: {data_type}")
    return importlib.import_module(
        f"projects.holobrain_0_5.configs.data_configs.config_{data_type}_dataset"
    )


def _build(config, mode, lazy_init):
    datasets = []
    for spec in _specs(config, mode):
        module = _builder(spec.pop("dataset_type"))
        datasets.append(
            module.build_datasets(
                config, mode=mode, lazy_init=lazy_init, **spec
            )
        )
    return ConcatDatasetWithFlag(datasets) if datasets else None


def build_training_dataset(config, lazy_init=False):
    """Build selected LMDB datasets, grouped by embodiment for batching."""
    return _build(config, "training", lazy_init)


def build_validation_dataset(config, lazy_init=False):
    """Build optional validation datasets; return None when unconfigured."""
    return _build(config, "validation", lazy_init)


def build_processors(config):
    """Build every deployment processor named by ``deploy_specs``."""
    module = importlib.import_module(config["deploy_specs"])
    processors = {}
    for spec in module.deploy_datasets:
        spec = spec.copy()
        builder = _builder(spec.pop("dataset_type"))
        processors[spec["dataset_name"]] = builder.build_processors(
            config, spec["dataset_name"], spec["setting_type"]
        )
    return processors


def collate_action_batch(batch):
    """Collate action samples, including scalar tensor metadata."""
    scalar_keys = [
        key
        for key, value in batch[0].items()
        if isinstance(value, torch.Tensor) and value.ndim == 0
    ]
    output = collate_batch_dict(
        [
            {
                key: value
                for key, value in sample.items()
                if key not in scalar_keys
            }
            for sample in batch
        ]
    )
    for key in scalar_keys:
        output[key] = torch.stack([sample[key] for sample in batch])
    return output
