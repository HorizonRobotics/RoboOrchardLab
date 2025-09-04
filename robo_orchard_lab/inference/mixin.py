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

import abc
import os
from typing import Generic, TypeVar, get_args

import torch
from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassInitFromConfigMixin,
    ClassType_co,  # noqa: F401
    load_config_class,
)

from robo_orchard_lab.models.mixin import ModelMixin
from robo_orchard_lab.utils import set_env
from robo_orchard_lab.utils.huggingface import download_repo
from robo_orchard_lab.utils.path import (
    DirectoryNotEmptyError,
    in_cwd,
    is_empty_directory,
)

__all__ = ["InferencePipelineMixin", "InferencePipelineMixinCfg"]


InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


class InferencePipelineMixin(
    ClassInitFromConfigMixin,
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
    """

    input_type: type[InputType]
    output_type: type[OutputType]

    @classmethod
    def __init_subclass__(cls) -> None:
        """A class-level hook to set generic type attributes.

        This method is automatically called when this class is subclassed. It
        inspects the generic type arguments (`InputType`, `OutputType`) provided
        by the subclass and sets them as class attributes for runtime
        inspection.

        Raises:
            ValueError: If the subclass does not correctly specify the two
                required generic type arguments.
        """  # noqa: E501
        type_args = get_args(cls.__orig_bases__[0])  # type: ignore
        if type_args == () or len(type_args) != 2:
            raise ValueError(
                "Subclasses of InferencePipelineMixin should have two type arguments: "  # noqa: E501
                f"InputType and OutputType. Gotten: {type_args}. "
                """Example:
class MyInferencePipeline(InferencePipelineMixin[Source, Target]):
    pass
                """
            )

        cls.input_type = type_args[0]
        cls.output_type = type_args[1]

    def __init__(self, model: ModelMixin, cfg: "InferencePipelineMixinCfg"):
        """Initializes the inference pipeline.

        Args:
            model (ModelMixin): An initialized model instance that adheres to
                the `ModelMixin` interface.
            cfg (InferencePipelineCfg): The configuration object for this
                pipeline.
        """
        self.model = model
        self.cfg = cfg

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

    def accelerator_save_state_pre_hook(
        self,
        models: list[torch.nn.Module],
        weights: list[dict[str, torch.Tensor]],
        output_dir: str,
    ):
        """Saves the pipeline config as a pre-hook for `accelerate.save_state`.

        This method is designed to be used with Hugging Face Accelerate's
        `save_state` function. It ensures that the pipeline's configuration
        is saved alongside the model state in the same directory.

        Args:
            models (List[torch.nn.Module]): A list of models being saved
                (unused in this implementation).
            weights (List[Dict[str, torch.Tensor]]): A list of state dicts
                (unused in this implementation).
            output_dir (str): The directory where the state is being saved.
        """
        with open(
            os.path.join(output_dir, "inference.config.json"), "w"
        ) as fh:
            fh.write(self.cfg.model_dump_json(indent=4))

    def save(
        self,
        directory: str,
        inference_prefix: str = "inference",
        model_prefix: str = "model",
        allow_shared_tensor: bool = False,
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
            allow_shared_tensor (bool): Whether to allow shared tensors when
                saving. Defaults to False.
            required_empty (bool): If True, raises an error if the target
                directory is not empty. Defaults to True.
        """
        os.makedirs(directory, exist_ok=True)
        if required_empty and not is_empty_directory(directory):
            raise DirectoryNotEmptyError(f"{directory} is not empty!")

        self.model.save_model(
            directory=directory,
            model_prefix=model_prefix,
            allow_shared_tensor=allow_shared_tensor,
            required_empty=False,
        )
        with open(
            os.path.join(directory, f"{inference_prefix}.config.json"), "w"
        ) as fh:
            fh.write(self.cfg.model_dump_json(indent=4))

    @staticmethod
    def load(
        directory: str,
        inference_prefix: str = "inference",
        strict: bool = True,
        device: str = "cpu",
        model_prefix: str = "model",
    ):
        """Loads a pipeline from a directory or a Hugging Face Hub repository.

        This factory method dynamically instantiates the correct pipeline class
        based on the `class_type` specified in the saved configuration file.
        It first loads the model and then uses the configuration to create the
        appropriate pipeline instance.

        Args:
            directory (str): The local directory or Hugging Face Hub repo ID
                (prefixed with "hf://") to load from.
            inference_prefix (str): The prefix of the pipeline's config file.
                Defaults to "inference".
            strict (bool): Whether to strictly enforce that the keys in the
                model's state_dict match. Defaults to True.
            device (str): The device to load the model onto (e.g., "cpu", "cuda").
                Defaults to "cpu".
            model_prefix (str): The prefix of the model files. Defaults to "model".

        Returns:
            An initialized instance of the specific pipeline subclass defined
            in the configuration.
        """  # noqa: E501
        if directory.startswith("hf://"):
            directory = download_repo(directory, repo_type="model")

        model = ModelMixin.load_model(
            directory=directory,
            load_weight=True,
            strict=strict,
            device=device,
            model_prefix=model_prefix,
        )

        with open(
            os.path.join(directory, f"{inference_prefix}.config.json"), "r"
        ) as fh:
            cfg = load_config_class(fh.read())

        with (
            in_cwd(directory),
            set_env(ORCHARD_LAB_CHECKPOINT_DIRECTORY=directory),
        ):
            pipeline = cfg.class_type(model=model, cfg=cfg)  # type: ignore

        return pipeline


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

    pass
