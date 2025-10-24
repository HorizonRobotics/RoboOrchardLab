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
import warnings
from typing import (
    Any,
    Generator,
    Generic,
    Iterable,
    TypeAlias,
    TypeVar,
    overload,
)

import torch
from robo_orchard_core.utils.config import CallableType, ConfigInstanceOf
from torch.utils.data import Dataset

from robo_orchard_lab.dataset.collates import collate_batch_dict
from robo_orchard_lab.inference.mixin import (
    InferencePipelineMixin,
    InferencePipelineMixinCfg,
    InputType,
    OutputType,
)
from robo_orchard_lab.inference.processor import (
    ProcessorMixin,
    ProcessorMixinCfg,
)
from robo_orchard_lab.models.mixin import TorchModelMixin

__all__ = ["InferencePipeline", "InferencePipelineCfg"]


DatasetType: TypeAlias = Dataset | list | tuple | Generator


class InferencePipeline(
    InferencePipelineMixin[InputType, OutputType],
    Generic[InputType, OutputType],
):
    """A high-level, concrete implementation of an end-to-end inference pipeline.

    Like `Pipeline` in huggingface transformers, this class provides a user-friendly
    interface for performing inference with a model, handling all necessary steps
    such as pre-processing, batching, model forwarding, and post-processing.

    The defined workflow in the `__call__` method is:

    1. Pre-process the raw input data using the configured processor.

    2. Collate the processed data into a mini-batch.

    3. Perform model inference (forward pass).

    4. Post-process the model's output using the processor.

    """  # noqa: E501

    cfg: InferencePipelineCfg
    processor: ProcessorMixin | None

    def __init__(self, cfg: InferencePipelineCfg):
        """Initializes the concrete inference pipeline.

        In addition to the base class initialization, this also instantiates
        the data processor from the provided configuration.
        """
        super().__init__(cfg)
        self.processor = self.cfg.processor() if self.cfg.processor else None

    def _configure(
        self, cfg: InferencePipelineMixinCfg, model: TorchModelMixin
    ):
        super()._configure(cfg, model)
        self.processor = self.cfg.processor() if self.cfg.processor else None

    @overload
    def __call__(self, data: InputType) -> OutputType: ...

    @overload
    def __call__(self, data: DatasetType) -> Iterable[OutputType]: ...

    @torch.inference_mode()
    def __call__(
        self, data: InputType | DatasetType
    ) -> OutputType | Iterable[OutputType]:
        """Executes the standard end-to-end inference workflow.

        This method orchestrates the full pipeline: pre-processing, collation,
        device transfer, model forwarding, and post-processing.

        Args:
            data (InputType | Iterable[InputType]): The raw input data for the
                pipeline. It can be a single sample or an iterable of samples,
                such as generator, dataset, or list. If an iterable is
                provided, the data will be processed in mini-batches of size
                `self.cfg.batch_size` and the method will yield results
                one batch at a time.

        Returns:
            OutputType | Iterable[OutputType]: The results after processing.
                If input is iterable, an iterable of results is returned,
                yielding one batch at a time.

        """
        if not isinstance(data, (Dataset, list, tuple, Generator)):
            return self._inference_single(data)
        else:
            return self._inference_batch_gen(data)

    def _inference_batch_gen(self, data: DatasetType) -> Iterable[OutputType]:
        batch = []
        for sample in data:
            if len(batch) != self.cfg.batch_size:
                batch.append(sample)
            if len(batch) == self.cfg.batch_size:
                yield self._inference_batch(batch)
                batch = []
        if len(batch) > 0:
            yield self._inference_batch(batch)

    def _inference_batch(self, batch: Iterable[InputType]) -> OutputType:
        """Executes the  inference workflow for a batch of data.

        This method orchestrates the full pipeline: pre-processing, collation,
        device transfer, model forwarding, and post-processing.

        Args:
            data (Iterable[InputType]): A batch of raw input data for the
                pipeline.

        Returns:
            Iterable[OutputType]: An iterable of final, post-processed results.
        """
        if self.cfg.collate_fn is None:
            warnings.warn(
                "No collate function is specified in the pipeline config for "
                "batch inference. Using default collate function "
                "`collate_batch_dict`, which assumes each data sample is "
                "a dictionary."
            )
            collate_fn = collate_batch_dict
        else:
            collate_fn = self.cfg.collate_fn

        if self.processor is not None:
            batch_data = [self.processor.pre_process(data) for data in batch]
        else:
            batch_data = list(batch)

        batch = collate_fn(batch_data)  # type: ignore

        model_outputs = self.model(batch)
        if self.processor is not None:
            outputs = self.processor.post_process(model_outputs, batch)
        else:
            outputs = model_outputs
        return outputs

    def _inference_single(self, data: InputType) -> OutputType:
        """Executes the standard end-to-end inference workflow.

        This method orchestrates the full pipeline: pre-processing, collation,
        device transfer, model forwarding, and post-processing.

        Args:
            data (InputType): The raw input data for the pipeline.

        Returns:
            OutputType: The final, post-processed result.
        """
        # with switch_model_mode(self.model, "eval"):
        # user should make sure that the model is in eval mode before calling
        # the pipeline. Calling eval() for each inference may be costly.

        if self.processor is not None:
            data = self.processor.pre_process(data)
        if self.cfg.collate_fn is not None:
            batch = self.cfg.collate_fn([data])
        else:
            batch = data  # type: ignore
        # the model should handle device placement internally,
        # as it may be distributed across multiple devices
        model_outputs = self.model(batch)
        if self.processor is not None:
            outputs = self.processor.post_process(model_outputs, batch)
        else:
            outputs = model_outputs

        return outputs


InferencePipelineType_co = TypeVar(
    "InferencePipelineType_co",
    bound=InferencePipeline,
    covariant=True,
)


class InferencePipelineCfg(
    InferencePipelineMixinCfg[InferencePipelineType_co]
):
    """Configuration for the concrete `InferencePipeline`.

    This class extends the base pipeline configuration with additional, specific
    settings for data handling, including the processor, collate function, and
    device transfer function.
    """  # noqa: E501

    processor: ConfigInstanceOf[ProcessorMixinCfg] | None = None
    """The configuration for the data processor. """

    collate_fn: CallableType[[list[Any]], Any] | None = None
    """The function used to collate single data items into a single batch.

    This method is required when the input data is an iterable (e.g., dataset,
    list, generator).
    """  # noqa: E501

    batch_size: int = 1
    """The number of samples to process in each mini-batch during inference."""
