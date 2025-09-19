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

from typing import Any, TypeVar

import pydantic
import torch
from robo_orchard_core.utils.config import CallableType

from robo_orchard_lab.dataset.collates import collate_batch_dict
from robo_orchard_lab.inference.mixin import (
    InferencePipelineMixin,
    InferencePipelineMixinCfg,
    InputType,
    OutputType,
)
from robo_orchard_lab.inference.processor import (
    IdentityProcessorCfg,
    ProcessorMixinCfgType_co,
)
from robo_orchard_lab.models.mixin import ModelMixin
from robo_orchard_lab.utils.torch import switch_model_mode, to_device

__all__ = ["InferencePipeline", "InferencePipelineCfg"]


class InferencePipeline(InferencePipelineMixin[InputType, OutputType]):
    """A high-level, concrete implementation of an end-to-end inference pipeline.

    This class provides a standard, out-of-the-box workflow for inference that
    is suitable for many common use cases. It orchestrates a multi-step process
    by integrating a `ProcessorMixin` for data handling.

    The defined workflow in the `__call__` method is:

    1. Pre-process the raw input data using the configured processor.

    2. Collate the processed data into a mini-batch.

    3. Move the batch to the model's device.

    4. Perform model inference (forward pass).

    5. Post-process the model's output using the processor.

    """  # noqa: E501

    cfg: "InferencePipelineCfg"  # for type hint

    def __init__(self, model: ModelMixin, cfg: "InferencePipelineCfg"):
        """Initializes the concrete inference pipeline.

        In addition to the base class initialization, this also instantiates
        the data processor from the provided configuration.
        """
        super().__init__(model, cfg)
        self.processor = self.cfg.processor()

    @torch.inference_mode()
    def __call__(self, data: InputType) -> OutputType:
        """Executes the standard end-to-end inference workflow.

        This method orchestrates the full pipeline: pre-processing, collation,
        device transfer, model forwarding, and post-processing.

        Args:
            data (InputType): The raw input data for the pipeline.

        Returns:
            OutputType: The final, post-processed result.
        """
        with switch_model_mode(self.model, "eval"):
            data = self.processor.pre_process(data)
            batch = self.cfg.collate_fn([data])
            batch = self.cfg.to_device_fn(batch, self.device)

            model_outputs = self.model(batch)

            outputs = self.processor.post_process(batch, model_outputs)

            return outputs


InferencePipelineMixinType_co = TypeVar(
    "InferencePipelineMixinType_co",
    bound=InferencePipelineMixin,
    covariant=True,
)


class InferencePipelineCfg(InferencePipelineMixinCfg):
    """Configuration for the concrete `InferencePipeline`.

    This class extends the base pipeline configuration with additional, specific
    settings for data handling, including the processor, collate function, and
    device transfer function.
    """  # noqa: E501

    processor: ProcessorMixinCfgType_co = pydantic.Field(  # type: ignore
        default_factory=lambda: IdentityProcessorCfg()
    )
    """The configuration for the data processor. Defaults to an identity processor that does nothing."""  # noqa: E501

    collate_fn: CallableType[[list[Any]], Any] = collate_batch_dict
    """The function used to collate a list of single data items into a single batch for the model."""  # noqa: E501

    to_device_fn: CallableType[[Any, torch.device], Any] = to_device
    """The function used to move a batch of data to the correct torch device."""  # noqa: E501
