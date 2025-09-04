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

import json
import logging
import os
from typing import Any, TypeVar

import torch
from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassInitFromConfigMixin,
    ClassType_co,  # noqa: F401
    load_config_class,
)
from safetensors.torch import load_file, save_file

from robo_orchard_lab.utils import set_env
from robo_orchard_lab.utils.huggingface import download_repo
from robo_orchard_lab.utils.path import (
    DirectoryNotEmptyError,
    in_cwd,
    is_empty_directory,
)

__all__ = [
    "TorchModuleCfg",
    "ModelMixin",
    "TorchModelMixin",
    "ClassType_co",
    "TorchModuleCfgType_co",
]

logger = logging.getLogger(__name__)

TorchNNModuleTypeCo = TypeVar(
    "TorchNNModuleTypeCo", bound=torch.nn.Module, covariant=True
)


class TorchModuleCfg(ClassConfig[TorchNNModuleTypeCo]):
    """Configuration class for PyTorch nn.Module.

    This class extends `ClassConfig` to specifically handle configurations
    for PyTorch modules, ensuring type safety for the module class.
    """

    pass


TorchModuleCfgType_co = TypeVar(
    "TorchModuleCfgType_co", bound=TorchModuleCfg, covariant=True
)


def _set_nested_attr(obj: Any, names: list[str], value: Any):
    """Sets a nested attribute on an object.

    Args:
        obj: The target object.
        names: A list of attribute names representing the path.
        value: The value to set on the final attribute.

    Example:
        >>> _set_nested_attr(model, "encoder.layer.weight".split("."), tensor)
        equivalent to `model.encoder.layer.weight = tensor`.

    """
    for name in names[:-1]:
        obj = getattr(obj, name)
    setattr(obj, names[-1], value)


class TorchModelMixin(torch.nn.Module, ClassInitFromConfigMixin):
    """A mixin class for PyTorch `nn.Module` providing model saving and loading utilities.

    This mixin standardizes how models are configured, saved, and loaded,
    integrating with a configuration system and supporting `safetensors`
    for model weights.
    """  # noqa: E501

    def __init__(self, cfg: TorchModuleCfg):
        """Initializes the ModelMixin.

        Args:
            cfg (TorchModuleCfg): The configuration object for the model.
        """
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def accelerator_save_state_pre_hook(
        models: list[torch.nn.Module],
        weights: list[dict[str, torch.Tensor]],
        output_dir: str,
    ):
        """Pre-hook for Hugging Face Accelerate to save model configurations.

        This static method is designed to be registered with Accelerate's
        `save_state` mechanism. It iterates through the provided models and,
        if a model is an instance of `ModelMixin`, saves its configuration
        to a JSON file in the `output_dir`.

        The configuration for the first model (index 0) is saved as
        `model.config.json`, and subsequent models are saved as
        `model_idx.config.json`.

        Note:
            This hook only saves the configuration. The model weights (`state_dict`)
            are expected to be saved by the Accelerate framework itself.

        Args:
            models: A list of `torch.nn.Module` instances to process.
            weights: A list of model state dictionaries. This argument is part
                of the Accelerate hook signature but is not directly used by
                this method as Accelerate handles weight saving.
            output_dir: The directory where the configuration files will be saved.
        """  # noqa: E501
        for idx, model_i in enumerate(models):
            if isinstance(model_i, ModelMixin):
                if idx == 0:
                    filename = "model.config.json"
                else:
                    filename = f"model_{idx}.config.json"
                with open(os.path.join(output_dir, filename), "w") as f:
                    f.write(model_i.cfg.model_dump_json(indent=4))

    def save_model(
        self,
        directory: str,
        model_prefix: str = "model",
        allow_shared_tensor: bool = False,
        required_empty: bool = True,
    ):
        """Saves the model's config and weights to a directory.

        This method saves the model's configuration to `{model_prefix}.config.json`
        and its weights to `{model_prefix}.safetensors`.

        If `allow_shared_tensor` is True, it handles models with tied weights by:

        1.  Saving only a single copy of each shared tensor to the `.safetensors` file.

        2.  Creating a `{model_prefix}.shared_keys.json` file that maps the
        duplicate parameter names to their original counterparts. This allows for
        perfect model restoration with `strict=True` loading.

        Args:
            directory: The path to the directory for saving the model.
            model_prefix: The file prefix for the config and weights files.
            allow_shared_tensor: If True, enables the logic to handle
                tied weights by saving a de-duplicated state dict and a
                key-sharing map.
            required_empty (bool): If True, raises an error if the target
                directory is not empty. Defaults to True.

        Raises:
            DirectoryNotEmptyError: If the target directory already exists and
                is not empty.
        """  # noqa: E501

        os.makedirs(directory, exist_ok=True)
        if required_empty and not is_empty_directory(directory):
            raise DirectoryNotEmptyError(f"{directory} is not empty!")

        config_path = os.path.join(directory, f"{model_prefix}.config.json")
        weights_path = os.path.join(directory, f"{model_prefix}.safetensors")

        with open(config_path, "w") as f:
            f.write(self.cfg.model_dump_json(indent=4))

        model_state_dict = self.state_dict()

        if allow_shared_tensor:
            data_ptr_to_key = dict()
            shared_keys_map = dict()
            unique_model_state_dict = dict()

            for key, tensor in model_state_dict.items():
                data_ptr = tensor.data_ptr()
                if data_ptr not in data_ptr_to_key:
                    data_ptr_to_key[data_ptr] = key
                    unique_model_state_dict[key] = tensor
                else:
                    original_key = data_ptr_to_key[data_ptr]
                    shared_keys_map[key] = original_key

            model_state_dict = unique_model_state_dict

            shared_keys_map_path = os.path.join(
                directory, f"{model_prefix}.shared_keys.json"
            )
            with open(shared_keys_map_path, "w") as fp:
                json.dump(shared_keys_map, fp)

        save_file(model_state_dict, weights_path)

    @staticmethod
    def load_model(
        directory: str,
        load_weight: bool = True,
        strict: bool = True,
        device: str = "cpu",
        model_prefix: str = "model",
    ) -> "ModelMixin":
        """Loads a model from a local directory or the Hugging Face Hub.

        This method supports loading from a local path or a Hugging Face Hub
        repository. For Hub models, a URI format is used:
        `hf://[<token>@]<repo_id>`

        .. code-block:: text

            Public model: `hf://google/gemma-7b`

            Private model: `hf://hf_YourToken@username/private-repo`

        .. warning::

            Embedding tokens directly in the URL can be a security risk.
            The URL may be logged in shell history or server logs. It is often safer
            to rely on the environment variable **HF_TOKEN** or the
            local token cache from `huggingface-cli login`.

        This method first loads the model's configuration from a JSON file
        (e.g., "model.config.json") found in the given directory. It then
        instantiates the model using this configuration. The instantiation
        occurs within a context where the **ORCHARD_LAB_CHECKPOINT_DIRECTORY**
        environment variable is temporarily set to the `directory` path.

        If `load_weight` is True, the method proceeds to load the model's
        weights (state dictionary) from a ".safetensors" file (e.g.,
        "model.safetensors") located in the same directory.

        Args:
            directory: The path to the directory containing the model's
                configuration file and, if applicable, the state dictionary file.
            load_weight: If True (default), the model's state dictionary is
                loaded from the ".safetensors" file. If False, the model is
                initialized from the configuration but weights are not loaded.
            strict: A boolean indicating whether to strictly enforce that the keys
                in `state_dict` match the keys returned by this module's
                `state_dict()` function. Passed to `model.load_state_dict()`.
                Defaults to True.
            device: The device (e.g., "cpu", "cuda:0") onto which the model's
                state dictionary should be loaded. Passed to `load_file` for
                loading the safetensors file. Defaults to "cpu".
            model_prefix: The prefix for the configuration and state dictionary
                files. For example, if "model", files will be sought as
                "model.config.json" and "model.safetensors". Defaults to "model".

        Returns:
            torch.nn.Module: An instance of the model (typed as "ModelMixin" or a subclass),
                initialized from the configuration and optionally with weights loaded.

        Raises:
            FileNotFoundError: If the specified `directory` does not exist,
                or if the configuration file (`{model_prefix}.config.json`)
                or the state dictionary file (`{model_prefix}.safetensors}`,
                when `load_weight` is True) is not found in the directory.
            ValueError: If the Hugging Face Hub URI is invalid.
        """  # noqa: E501

        if directory.startswith("hf://"):
            directory = download_repo(directory, repo_type="model")

        directory = os.path.abspath(directory)

        if not os.path.exists(directory):
            raise FileNotFoundError(f"checkpoint {directory} does not exists!")

        config_file = os.path.join(directory, f"{model_prefix}.config.json")

        with open(config_file, "r") as f:
            cfg: TorchModuleCfg = load_config_class(f.read())  # type: ignore

        with (
            in_cwd(directory),
            set_env(ORCHARD_LAB_CHECKPOINT_DIRECTORY=directory),
        ):
            model = cfg()

        if not load_weight:
            return model

        ckpt_path = os.path.join(directory, f"{model_prefix}.safetensors")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"{ckpt_path} does not exists!")

        state_dict = load_file(ckpt_path, device=device)

        shared_keys_map_path = os.path.join(
            directory, f"{model_prefix}.shared_keys.json"
        )

        if os.path.exists(shared_keys_map_path):
            with open(shared_keys_map_path, "r") as fp:
                shared_keys_map = json.load(fp)

            # Step 1: Reconstruct the state_dict in memory.
            # This adds the duplicate keys back into the state_dict, ensuring
            # it perfectly matches the model's expected keys for `strict=True` loading.  # noqa: E501
            for duplicate_key, original_key in shared_keys_map.items():
                if original_key in state_dict:
                    state_dict[duplicate_key] = state_dict[original_key]

            model.load_state_dict(state_dict, strict=strict)

            # Step 2: Re-establish the actual memory sharing on the model object.  # noqa: E501
            # This ensures `tensor_a is tensor_b` holds true after loading.
            for duplicate_key, original_key in shared_keys_map.items():
                # Find the original tensor object on the model
                original_tensor = model
                for part in original_key.split("."):
                    original_tensor = getattr(original_tensor, part)

                # Point the duplicate parameter to the original tensor object
                _set_nested_attr(
                    model, duplicate_key.split("."), original_tensor
                )

        else:
            model.load_state_dict(state_dict, strict=strict)

        return model


ModelMixin = TorchModelMixin
