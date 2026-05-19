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

import importlib
import importlib.util
import logging
import threading
from pathlib import Path
from types import ModuleType
from typing import Callable

logger = logging.getLogger(__name__)

_REGISTRY_LOCK = threading.Lock()
REGISTERED = False

TRAIN_DATASET_BUILD_FUNCS: dict[str, Callable] = {}
VALIDATION_DATASET_BUILD_FUNCS: dict[str, Callable] = {}
PROCESSOR_BUILD_FUNCS: dict[str, Callable] = {}


def _register_func(registry: dict[str, Callable], data_type: str):
    if not data_type:
        raise ValueError("`data_type` must be provided when registering.")

    def decorator(func: Callable):
        registry[data_type] = func
        return func

    return decorator


def train_dataset_register(data_type: str):
    return _register_func(TRAIN_DATASET_BUILD_FUNCS, data_type=data_type)


def validation_dataset_register(data_type: str):
    return _register_func(VALIDATION_DATASET_BUILD_FUNCS, data_type=data_type)


def processor_register(data_type: str):
    return _register_func(PROCESSOR_BUILD_FUNCS, data_type=data_type)


def apply_dataset_register():
    global REGISTERED
    if REGISTERED:
        return
    with _REGISTRY_LOCK:
        if REGISTERED:
            return
        importlib.import_module("data_configs")
        REGISTERED = True


def _load_module_from_ref(module_ref: str) -> ModuleType:
    module_path = Path(module_ref)
    if module_ref.endswith(".py") or module_path.exists():
        if not module_path.is_absolute():
            module_path = module_path.resolve()
        spec = importlib.util.spec_from_file_location(
            module_path.stem, module_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load dataset specs: {module_ref}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(module_ref)


def _load_dataset_specs_from_config(
    config: dict,
    config_key: str,
    attr_name: str,
    required: bool = False,
):
    module_ref = config.get(config_key)
    if module_ref is None:
        if required:
            raise KeyError(f"`{config_key}` must be provided.")
        return None
    if not isinstance(module_ref, str):
        raise TypeError(
            f"`{config_key}` must be a module reference string or None."
        )

    spec_module = _load_module_from_ref(module_ref)
    if not hasattr(spec_module, attr_name):
        raise AttributeError(f"{config_key} module must define `{attr_name}`.")
    return getattr(spec_module, attr_name)


def _load_dataset_specs_from_module_ref(
    module_ref: str,
    attr_name: str,
    required: bool = False,
):
    spec_module = _load_module_from_ref(module_ref)
    if not hasattr(spec_module, attr_name):
        if required:
            raise AttributeError(
                f"dataset specs module must define `{attr_name}`."
            )
        return None
    return getattr(spec_module, attr_name)


def _resolve_data_paths(data_paths):
    return data_paths() if callable(data_paths) else data_paths


def _finalize_dataset_sample_weights(
    config: dict,
    dataset_names: list[str],
    dataset_sample_weights: dict | None,
):
    if dataset_sample_weights is None or len(dataset_sample_weights) == 0:
        return
    missing = [
        name for name in dataset_names if name not in dataset_sample_weights
    ]
    if missing:
        logger.warning(
            "dataset_sample_weights is missing keys for the following "
            "datasets: %s. Available keys: %s",
            missing,
            list(dataset_sample_weights.keys()),
        )
        raise KeyError(f"dataset_sample_weights missing keys: {missing}")
    config["dataset_sample_weights"] = [
        dataset_sample_weights[name] for name in dataset_names
    ]


def _build_typed_datasets(
    config: dict,
    dataset_specs: list[dict],
    registry: dict[str, Callable],
    mode: str,
    lazy_init: bool = False,
):
    datasets = {}
    dataset_names = []
    for dataset_spec in dataset_specs:
        if not isinstance(dataset_spec, dict):
            raise TypeError("Dataset specs must be dict instances.")

        dataset_spec = dataset_spec.copy()
        dataset_type = dataset_spec.pop("dataset_type")
        dataset_name = dataset_spec["dataset_name"]
        if (
            (mode == "training")
            and ("training_datasets" in config)
            and (dataset_name not in config["training_datasets"])
        ):
            continue
        if (
            (mode == "validation")
            and ("validation_datasets" in config)
            and (dataset_name not in config["validation_datasets"])
        ):
            continue

        if "data_paths" in dataset_spec:
            dataset_spec["data_paths"] = _resolve_data_paths(
                dataset_spec["data_paths"]
            )
        build_func = registry.get(dataset_type)
        if build_func is None:
            raise KeyError(
                f"Dataset type `{dataset_type}` has not been registered."
            )
        datasets[dataset_name] = build_func(
            config,
            mode=mode,
            lazy_init=lazy_init,
            **dataset_spec,
        )
        if not hasattr(datasets[dataset_name], "dataset_name"):
            datasets[dataset_name].dataset_name = dataset_name
        dataset_names.append(dataset_name)
    return datasets, dataset_names


def build_training_dataset(config, lazy_init=False):
    from robo_orchard_lab.dataset.dataset_wrapper import ConcatDatasetWithFlag

    apply_dataset_register()
    training_datasets = _load_dataset_specs_from_module_ref(
        config["dataset_specs"],
        attr_name="training_datasets",
        required=True,
    )

    dataset_sample_weights = config.get("dataset_sample_weights", {})
    normalized_specs = []
    for dataset_spec in training_datasets:
        dataset_spec = dataset_spec.copy()
        dataset_name = dataset_spec["dataset_name"]
        sample_weight = dataset_spec.pop("sample_weight", None)
        if sample_weight is not None:
            dataset_sample_weights[dataset_name] = sample_weight
        normalized_specs.append(dataset_spec)

    datasets, dataset_names = _build_typed_datasets(
        config,
        normalized_specs,
        TRAIN_DATASET_BUILD_FUNCS,
        mode="training",
        lazy_init=lazy_init,
    )
    _finalize_dataset_sample_weights(
        config,
        dataset_names,
        dataset_sample_weights,
    )
    return ConcatDatasetWithFlag(
        datasets=[datasets[name] for name in dataset_names]
    )


def build_validation_dataset(config, lazy_init=False):
    from robo_orchard_lab.dataset.dataset_wrapper import ConcatDatasetWithFlag

    apply_dataset_register()
    validation_datasets = _load_dataset_specs_from_module_ref(
        config["dataset_specs"],
        attr_name="validation_datasets",
    )
    if not validation_datasets:
        return None

    datasets, dataset_names = _build_typed_datasets(
        config,
        validation_datasets,
        VALIDATION_DATASET_BUILD_FUNCS,
        mode="validation",
        lazy_init=lazy_init,
    )
    return ConcatDatasetWithFlag(
        datasets=[datasets[name] for name in dataset_names]
    )


def build_processors(config):
    apply_dataset_register()
    deploy_datasets = _load_dataset_specs_from_config(
        config,
        config_key="deploy_specs",
        attr_name="deploy_datasets",
    )
    if not deploy_datasets:
        return {}

    processors = {}
    for dataset_spec in deploy_datasets:
        dataset_spec = dataset_spec.copy()
        dataset_type = dataset_spec.pop("dataset_type")
        dataset_name = dataset_spec["dataset_name"]
        build_func = PROCESSOR_BUILD_FUNCS.get(dataset_type)
        if build_func is None:
            raise KeyError(
                f"Dataset type `{dataset_type}` has not been registered."
            )
        processors[dataset_name] = build_func(config, **dataset_spec)
    return processors
