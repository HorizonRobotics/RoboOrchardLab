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
import abc
import logging
import os
from typing import Generic, Literal

import torch
from robo_orchard_core.utils.config import ClassConfig, ClassType_co, load_from
from typing_extensions import TypeVar

from robo_orchard_lab.models.mixin import TorchModelMixin, TorchModuleCfg
from robo_orchard_lab.utils.huggingface import download_repo
from robo_orchard_lab.utils.path import (
    DirectoryNotEmptyError,
    in_cwd,
    is_empty_directory,
)

__all__ = [
    "ClassType_co",
    "InferencePipelineMixin",
    "InferencePipelineMixinCfg",
]

logger = logging.getLogger(__name__)

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


class InferencePipelineMixin(
    Generic[InputType, OutputType],
    metaclass=abc.ABCMeta,
):
    """An abstract base class for end-to-end inference pipelines.

    This generic mixin provides a common framework for orchestrating the
    inference process. It is responsible for holding the model and its
    configuration, but delegates the specific inference logic to its
    subclasses. It standardizes methods for saving and loading the entire
    pipeline state (model and configuration).

    Subclasses should be specialized with `InputType` and `OutputType` and must
    implement the `__call__` method to define the core task-specific logic.

    Template Args:
        InputType: The type of the input data to the pipeline.
        OutputType: The type of the output data from the pipeline.

    """

    InitFromConfig: bool = True

    model: TorchModelMixin
    """The underlying model used in the pipeline"""
    cfg: InferencePipelineMixinCfg
    """The configuration for the pipeline"""

    def __init__(self, cfg: InferencePipelineMixinCfg):
        """Initializes the inference pipeline with the given configuration.

        This constructor sets up the model based on the provided configuration.
        It ensures that the model is properly instantiated and ready for use.

        Args:
            cfg (InferencePipelineMixinCfg): The configuration for the
                inference pipeline, including the model configuration.
        """
        if cfg.model is None:
            raise ValueError("The model configuration is missing.")
        self._configure(cfg=cfg, model=cfg.model())

    @classmethod
    def from_shared(
        cls, cfg: InferencePipelineMixinCfg, model: TorchModelMixin
    ):
        """Creates an inference pipeline from a given model instance.

        This factory method allows for the creation of an inference pipeline
        using an already instantiated model. It is useful when the model has
        been created or loaded separately and needs to be integrated into a
        pipeline.

        Note:
            If the provided configuration contains a model configuration that
            differs from that of the given model instance, a warning is logged
            and the model instance's configuration is used. If so, the model
            configuration in `cfg` is also asigned to that of the model
            instance to ensure consistency.

        Args:
            cfg (InferencePipelineMixinCfg): The configuration for the
                inference pipeline.
            model (TorchModelMixin): An instance of the model to be used in
                the pipeline.

        Returns:
            InferencePipelineMixin: An instance of the inference pipeline
                initialized with the provided model.
        """
        pipeline = cls.__new__(cls)
        if (cfg.model is not None) and cfg.model != model.cfg:
            logger.warning(
                "The provided model configuration in the pipeline "
                "differs from the configuration of the given model "
                "instance. The latter will be used."
            )
            cfg.model = model.cfg
        pipeline._configure(cfg=cfg, model=model)
        return pipeline

    def _configure(
        self, cfg: InferencePipelineMixinCfg, model: TorchModelMixin
    ):
        """Configures the pipeline with the given configuration and model.

        This method sets the internal configuration and model of the pipeline.
        It is useful for scenarios where the pipeline needs to be reconfigured
        after initialization.

        Args:
            cfg (InferencePipelineMixinCfg): The new configuration for the
                inference pipeline.
            model (TorchModelMixin): The new model to be used in the pipeline.
        """
        self.cfg = cfg
        self.model = model

    def to(self, device: torch.device):
        """Moves the underlying model to the specified device.

        Args:
            device (torch.device): The target device to move the model to.
        """
        self.model.to(device)

    @property
    def device(self) -> torch.device:
        """The device where the model's parameters are located."""
        return next(self.model.parameters()).device

    @abc.abstractmethod
    def __call__(self, data: InputType) -> OutputType:
        """Executes the end-to-end inference for a single data point.

        This method defines the core logic of the inference pipeline and must be
        implemented by subclasses.

        Args:
            data (InputType): The raw input data for the pipeline, matching the
                `InputType` generic specification.

        Returns:
            OutputType: The final, processed result, matching the `OutputType`
                generic specification.
        """  # noqa: E501
        pass

    def save(
        self,
        directory: str,
        inference_prefix: str = "inference",
        model_prefix: str = "model",
        required_empty: bool = True,
    ):
        """Saves the full pipeline (model and config) to a directory.

        This method saves the model's weights and configuration by calling its
        `save_model` method, and also saves the pipeline's own configuration
        file.

        Args:
            directory (str): The target directory to save the pipeline to.
            inference_prefix (str): The prefix for the pipeline's config file.
                Defaults to "inference".
            model_prefix (str): The prefix for the model files, passed to the
                model's save method. Defaults to "model".
            required_empty (bool): If True, raises an error if the target
                directory is not empty. Defaults to True.
        """
        os.makedirs(directory, exist_ok=True)
        if required_empty and not is_empty_directory(directory):
            raise DirectoryNotEmptyError(f"{directory} is not empty!")

        self.model.save_model(
            directory=directory,
            model_prefix=model_prefix,
            required_empty=False,
        )
        with open(
            os.path.join(directory, f"{inference_prefix}.config.json"), "w"
        ) as fh:
            cfg_copy = self.cfg.model_copy()
            cfg_copy.model = None  # Avoid redundancy of model config
            fh.write(cfg_copy.model_dump_json(indent=4))

    @staticmethod
    def load(
        directory: str,
        inference_prefix: str = "inference",
        load_weights: bool = True,
        strict: bool = True,
        device: str | None = "cpu",
        device_map: str | dict[str, int | str | torch.device] | None = None,
        model_prefix: str = "model",
        load_impl: Literal["native", "accelerate"] = "accelerate",
    ):
        """Loads a pipeline from a directory or a Hugging Face Hub repository.

        This factory method dynamically instantiates the correct pipeline class
        based on the `class_type` specified in the saved configuration file.
        It first loads the model and then uses the configuration to create the
        appropriate pipeline instance.

        Args:
            directory (str): The local directory or Hugging Face Hub repo ID
                (prefixed with "hf://") to load from.
            inference_prefix (str, optional): The prefix of the pipeline's
                config file. Defaults to "inference".
            load_weights (bool, optional): Whether to load the model weights.
                If False, the model will be initialized randomly. Defaults to
                True.
            strict (bool, optional): Whether to strictly enforce that the keys in the
                model's state_dict match. Defaults to True.
            device (str | None, optional): The device to map the model to. If
                None, the model will be loaded to the same device as it was
                saved from. Defaults to "cpu".
            device_map (str | dict[str, int | str | torch.device] | None, optional):
                The device map to use when loading the model. This is only used
                if `device` is None. Defaults to None.
            model_prefix (str, optional): The prefix of the model files. Defaults to "model".

        Returns:
            An initialized instance of the specific pipeline subclass defined
            in the configuration.
        """  # noqa: E501
        if directory.startswith("hf://"):
            directory = download_repo(directory, repo_type="model")

        with in_cwd(directory):
            cfg = load_from(
                f"{inference_prefix}.config.json",
                ensure_type=InferencePipelineMixinCfg,
            )
            cfg.update_model_cfg(f"{model_prefix}.config.json")
            if cfg.model is None:
                raise ValueError("The model configuration is missing.")
            pipeline: InferencePipelineMixin = cfg.class_type(cfg=cfg)  # type: ignore
            if load_weights:
                pipeline.model.load_weights(
                    directory=directory,
                    model_prefix=model_prefix,
                    strict=strict,
                    device=device,
                    device_map=device_map,
                    load_impl=load_impl,
                )

        return pipeline

    def reset(self) -> None:
        """Reset the internal state of the pipeline, if applicable.

        This method can be overridden by subclasses that maintain internal
        state (e.g., RNNs or stateful processors). The default implementation
        does nothing.
        """
        pass


InferencePipelineMixinType_co = TypeVar(
    "InferencePipelineMixinType_co",
    bound=InferencePipelineMixin,
    covariant=True,
)


class InferencePipelineMixinCfg(ClassConfig[InferencePipelineMixinType_co]):
    """Configuration class for an inference pipeline.

    This class uses Pydantic for data validation and stores the configuration
    for the pipeline, including the specific pipeline class to instantiate and
    its associated components.
    """

    model: TorchModuleCfg | None = None
    """The configuration for the model used in the pipeline."""

    def update_model_cfg(self, model_cfg_path: str):
        """Update model configuration from a file."""
        if self.model is not None and os.path.exists(model_cfg_path):
            logger.warning(
                f"Both the pipeline config and {model_cfg_path} "
                "contain a model configuration. The latter will be used."
            )
            self.model = load_from(model_cfg_path, ensure_type=TorchModuleCfg)
        if self.model is None:
            self.model = load_from(model_cfg_path, ensure_type=TorchModuleCfg)
