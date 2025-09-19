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
from __future__ import annotations
import logging
import os
from typing import Any, Literal, TypeVar, overload

import torch
from accelerate import Accelerator
from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassInitFromConfigMixin,
    ClassType_co,  # noqa: F401
    load_config_class,
)
from safetensors.torch import (
    load_model as safetensors_load_model,
    save_model as safetensors_save_model,
)
from typing_extensions import deprecated

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

        self._accelerate_model_id: int = -1

    @property
    def accelerate_model_id(self) -> int:
        """The model index of the prepared model in the 'accelerator' instance.

        This should be set when the model is prepared with
        `accelerator.prepare()`.

        If not available, it returns -1.

        """
        return self._accelerate_model_id

    @accelerate_model_id.setter
    def accelerate_model_id(self, value: int):
        self._accelerate_model_id = value

    def accelerator_register_all_hooks(
        self,
        accelerator: Accelerator,
    ) -> list[torch.utils.hooks.RemovableHandle]:
        """Register all necessary hooks to the given Hugging Face Accelerator.

        Args:
            accelerator (Accelerator): The Hugging Face Accelerator instance.
        """

        if accelerator.is_main_process is False:
            logger.info("Not the main process, skip registering hooks.")
            return []

        model_id = self.accelerate_model_id
        if model_id < 0:
            raise ValueError(
                "Model's accelerate_model_id is not set. "
                "Ensure accelerate_model_id is set when preparing the model "
                "with `accelerator.prepare()`."
            )

        for hook in accelerator._save_model_state_pre_hook.keys():
            if hook == self.accelerator_save_state_pre_hook:
                logger.warning(
                    f"accelerator_save_state_pre_hook of {self}"
                    " is already registered. Skip registering again."
                )
                return []
        return [
            accelerator.register_save_state_pre_hook(
                self.accelerator_save_state_pre_hook
            )
        ]

    def accelerator_save_state_pre_hook(
        self,
        models: list[torch.nn.Module],
        weights: list[dict[str, torch.Tensor]],
        output_dir: str,
    ):
        """Pre-hook for Hugging Face Accelerate to save model configurations.

        This static method is designed to be registered with Accelerate's
        `save_state` mechanism. It iterates through the provided models and,
        if a model is an instance of `ModelMixin`, saves its configuration
        to a JSON file in the `output_dir`.

        The configuration is saved as `model_{id}.config.json` where `{id}`
        corresponds to the model's `accelerate_model_id`.

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

        model_id = self.accelerate_model_id
        filename = f"model_{model_id}.config.json"
        logger.info(f"Saving model config to {filename}.")
        with open(os.path.join(output_dir, filename), "w") as f:
            f.write(self.cfg.model_dump_json(indent=4))

    @overload
    def save_model(
        self,
        directory: str,
        model_prefix: str = "model",
        allow_shared_tensor: None = None,
        required_empty: bool = True,
        accelerator: Accelerator | None = None,
    ): ...

    @overload
    @deprecated(
        "The `allow_shared_tensor` argument is deprecated and will be "
        "removed in future versions. The shared tensor handling is "
        "always enabled."
    )
    def save_model(
        self,
        directory: str,
        model_prefix: str = "model",
        allow_shared_tensor: bool = False,
        required_empty: bool = True,
        accelerator: Accelerator | None = None,
    ): ...

    def save_model(
        self,
        directory: str,
        model_prefix: str = "model",
        allow_shared_tensor: bool | None = None,
        required_empty: bool = True,
        accelerator: Accelerator | None = None,
    ):
        """Saves the model's config and weights to a directory.

        This method saves the model's configuration to `{model_prefix}.config.json`
        and its weights to `{model_prefix}.safetensors`.

        If an `accelerator` instance is provided, it uses the accelerator's
        `save_model` method to save the model, which is useful for distributed
        training scenarios. In this case, `model_prefix` is ignored.

        Args:
            directory: The path to the directory for saving the model.
            model_prefix: The file prefix for the config and weights files.
                If None and `accelerator` is not provided, defaults to "model".
                Ignored if `accelerator` is provided.
            allow_shared_tensor: If True, enables the logic to handle
                tied weights by saving a de-duplicated state dict and a
                key-sharing map. This field is deprecated and will be
                removed in future versions. The shared tensor handling
                is always enabled.
            required_empty (bool): If True, raises an error if the target
                directory is not empty. Defaults to True.
            accerator: An optional `Accelerator` instance from Hugging Face
                Accelerate. If provided, the model will be saved using the
                accelerator's `save_model` method.

        Raises:
            DirectoryNotEmptyError: If the target directory already exists and
                is not empty.
        """  # noqa: E501

        os.makedirs(directory, exist_ok=True)
        if required_empty and not is_empty_directory(directory):
            raise DirectoryNotEmptyError(f"{directory} is not empty!")

        if allow_shared_tensor is not None:
            logger.warning(
                "The `allow_shared_tensor` argument is deprecated and will be "
                "removed in future versions. The shared tensor handling is "
                "always enabled."
            )

        if accelerator is not None:
            accelerator.save_model(self, save_directory=directory)
        else:
            assert model_prefix is not None
            weights_path = os.path.join(
                directory, f"{model_prefix}.safetensors"
            )
            safetensors_save_model(
                self, weights_path, metadata={"format": "pt"}
            )

        config_path = os.path.join(directory, f"{model_prefix}.config.json")

        with open(config_path, "w") as f:
            f.write(self.cfg.model_dump_json(indent=4))

    @staticmethod
    def load_model(
        directory: str,
        load_weight: bool = True,
        strict: bool = True,
        device: str = "cpu",
        model_prefix: str = "model",
        load_impl: Literal["native", "accelerate"] = "accelerate",
    ) -> TorchModelMixin:
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
            model: TorchModelMixin = cfg()

        if not load_weight:
            return model

        def load_impl_native():
            ckpt_path = os.path.join(directory, f"{model_prefix}.safetensors")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"{ckpt_path} does not exists!")

            missing, unexpected = safetensors_load_model(
                model, filename=ckpt_path, device=device, strict=strict
            )
            if len(missing) > 0:
                logger.warning(
                    f"Some weights are missing when loading state_dict from "
                    f"{ckpt_path}: {missing}"
                )
            if len(unexpected) > 0:
                logger.warning(
                    f"Some unexpected weights are found when "
                    f"loading state_dict from {ckpt_path}: {unexpected}"
                )

            return model

        if load_impl == "native":
            return load_impl_native()
        elif load_impl == "accelerate":
            from accelerate import load_checkpoint_in_model

            ckpt_path = os.path.join(directory, f"{model_prefix}.safetensors")
            if os.path.exists(ckpt_path):
                # if the safetensors file exists, `load_checkpoint_in_model`
                # is compatible with the native implementation.
                load_checkpoint_in_model(model, ckpt_path, strict=strict)
            else:
                load_checkpoint_in_model(model, directory, strict=strict)
            model = model.to(device=device)
            return model

        else:
            raise ValueError(
                f"Invalid load_impl: {load_impl}, "
                f"expected 'native' or 'accelerate'."
            )


ModelMixin = TorchModelMixin
