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
from typing import TypeVar

from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassInitFromConfigMixin,
    ClassType_co,  # noqa: F401
)

__all__ = [
    "ProcessorMixin",
    "ProcessorMixinType_co",
    "ProcessorMixinCfg",
    "ProcessorMixinCfgType_co",
]


class ProcessorMixin(ClassInitFromConfigMixin, metaclass=abc.ABCMeta):
    """An abstract base class for data processing modules.

    This class defines a standard interface for data processors, which are
    responsible for encapsulating the pre-processing and post-processing logic
    required for a model. Subclasses must implement the `pre_process` method.

    The primary role of a processor is to convert raw, user-provided data into
    a format suitable for model consumption (pre-processing) and to convert the
    model's raw output into a user-friendly, understandable format
    (post-processing).
    """

    def __init__(self, cfg: "ProcessorMixinCfg"):
        self.cfg = cfg

    @abc.abstractmethod
    def pre_process(self, data):
        """Transforms raw data into a model-ready format.

        Subclasses must implement this method to handle the conversion of
        user-provided input into the specific format expected by the model.

        Args:
            data: The raw input data.

        Returns:
            The processed data, ready to be collated into a batch.
        """
        pass

    def post_process(self, batch, model_outputs):
        """Transforms model output into a user-friendly format.

        This method can be overridden by subclasses to convert the raw tensor
        outputs from the model into a more structured or interpretable format,
        such as dictionaries with meaningful keys, text, or image objects.

        By default, it performs an identity mapping, returning the model
        outputs as is.

        Args:
            batch: The transformed batch.
            model_outputs: The raw output from the model's forward pass.

        Returns:
            The post-processed, user-friendly output.
        """
        return model_outputs


ProcessorMixinType_co = TypeVar(
    "ProcessorMixinType_co",
    bound=ProcessorMixin,
    covariant=True,
)


class ProcessorMixinCfg(ClassConfig[ProcessorMixinType_co]):
    """Base configuration class for a `ProcessorMixin`.

    This Pydantic-based class is used to configure and instantiate a
    corresponding `ProcessorMixin` subclass. It leverages `ClassConfig` to
    dynamically create the processor instance specified by the `class_type`
    field.
    """

    pass


ProcessorMixinCfgType_co = TypeVar(
    "ProcessorMixinCfgType_co",
    bound=ProcessorMixinCfg,
    covariant=True,
)
