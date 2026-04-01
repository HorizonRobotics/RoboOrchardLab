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
from robo_orchard_core.utils.config import (
    CallableType,
    ClassType_co,
    ConfigInstanceOf,
)
from torch.utils.data import Dataset

from robo_orchard_lab.dataset.collates import (
    CollatorConfig,
    collate_batch_dict,
)
from robo_orchard_lab.models.mixin import TorchModelMixin
from robo_orchard_lab.pipeline.inference.mixin import (
    InferencePipelineMixin,
    InferencePipelineMixinCfg,
    InputType,
    OutputType,
)
from robo_orchard_lab.processing.io_processor import (
    ModelIOProcessor,
    ModelIOProcessorCfg,
)
from robo_orchard_lab.utils.torch import to_device

__all__ = ["InferencePipeline", "InferencePipelineCfg"]


DatasetType: TypeAlias = Dataset | list | tuple | Generator


class InferencePipeline(
    InferencePipelineMixin[InputType, OutputType],
    Generic[InputType, OutputType],
):
    """A concrete end-to-end inference pipeline.

    Like ``Pipeline`` in Hugging Face Transformers, this class provides a
    user-friendly interface for performing inference with a model while
    handling the surrounding runtime steps such as pre-processing, batching,
    model forwarding, and post-processing.

    The defined workflow in :meth:`__call__` is:

    1. Pre-process the raw input data using the configured processor.
    2. Collate the processed data into a mini-batch when batching is needed.
    3. Perform model inference.
    4. Post-process the model's output using the processor.

    In this workflow, ``processor.pre_process`` is invoked once per raw
    sample before collation. ``processor.post_process`` runs after model
    forward, so it usually receives batched model outputs together with the
    collated model input, even when the effective batch size is 1.
    """

    cfg: InferencePipelineCfg
    processor: ModelIOProcessor | None
    collate_fn: CallableType[[list[Any]], Any] | None

    def __init__(
        self,
        cfg: InferencePipelineMixinCfg,
        model: TorchModelMixin | None = None,
    ):
        """Initialize the inference pipeline.

        Args:
            cfg (InferencePipelineMixinCfg): Configuration for the pipeline.
                Concrete configs may additionally define an optional I/O
                processor, collate function, and batch size.
            model (TorchModelMixin | None, optional): A pre-built model
                instance to bind to the pipeline. If None, the model is
                instantiated from ``cfg.model_cfg`` by the mixin. Default is
                None.
        """
        super().__init__(cfg=cfg, model=model)

    def _setup(self, cfg: InferencePipelineMixinCfg, model: TorchModelMixin):
        """Configure the pipeline runtime helpers.

        Besides storing ``cfg`` and ``model`` through the mixin, this method
        instantiates the optional model I/O processor and resolves the
        effective collate function so the pipeline can be reused after loading
        or state restoration.

        Args:
            cfg (InferencePipelineMixinCfg): The pipeline configuration.
            model (TorchModelMixin): The model bound to the pipeline.
        """
        super()._setup(cfg, model)
        self.processor = self.cfg.processor() if self.cfg.processor else None
        if isinstance(self.cfg.collate_fn, CollatorConfig):
            self.collate_fn = self.cfg.collate_fn()
        else:
            self.collate_fn = self.cfg.collate_fn

    def _get_ignore_save_attributes(self) -> list[str]:
        """Return runtime-only attributes that should not be serialized.

        The processor and collate function can be reconstructed from
        ``self.cfg``, so saved state only needs to persist the model and the
        configuration metadata.

        Returns:
            list[str]: Attribute names that should be excluded from saved
                state.
        """
        return super()._get_ignore_save_attributes() + [
            "processor",
            "collate_fn",
        ]

    @overload
    def __call__(self, data: InputType) -> OutputType: ...

    @overload
    def __call__(self, data: DatasetType) -> Iterable[OutputType]: ...

    @torch.inference_mode()
    def __call__(
        self, data: InputType | DatasetType
    ) -> OutputType | Iterable[OutputType]:
        """Executes the standard end-to-end inference workflow.

        This method orchestrates the full pipeline: pre-processing,
        collation, model forwarding, and post-processing.

        Args:
            data (InputType | DatasetType): The raw input data for the
                pipeline. It can be a single sample or a dataset-like
                iterable of samples, such as a generator, dataset, list, or
                tuple. When an iterable is provided, the data is processed in
                mini-batches of size ``self.cfg.batch_size`` and the method
                yields one result per batch.

        Returns:
            OutputType | Iterable[OutputType]: The post-processed inference
                result. A single input returns one output, while a dataset-like
                input returns an iterator that yields one output per batch.
        """
        if not isinstance(data, (Dataset, list, tuple, Generator)):
            return self._inference_single(data)
        else:
            return self._inference_batch_gen(data)

    def _inference_batch_gen(self, data: DatasetType) -> Iterable[OutputType]:
        """Yield batched inference results from a dataset-like input.

        Args:
            data (DatasetType): Dataset-like input containing raw samples.

        Returns:
            Iterable[OutputType]: An iterator that yields the post-processed
                result for each mini-batch assembled from ``data``.
        """
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
        """Executes the standard inference workflow for a batch of data.

        This method orchestrates the full batch pipeline: pre-processing,
        collation, model forwarding, and post-processing. If no collate
        function is configured, it falls back to
        :func:`robo_orchard_lab.dataset.collates.collate_batch_dict`, which
        assumes each sample is a dictionary.

        The processor contract in this path is asymmetric by design:
        ``pre_process`` runs once per sample before collation, while
        ``post_process`` runs once on the batched model outputs and the
        collated batch.

        Args:
            batch (Iterable[InputType]): A batch of raw input samples for the
                pipeline.

        Returns:
            OutputType: The final, post-processed batch result.
        """
        if self.collate_fn is None:
            warnings.warn(
                "No collate function is specified in the pipeline config for "
                "batch inference. Using default collate function "
                "`collate_batch_dict`, which assumes each data sample is "
                "a dictionary."
            )
            collate_fn = collate_batch_dict
        else:
            collate_fn = self.collate_fn

        if self.processor is not None:
            batch_data = [self.processor.pre_process(data) for data in batch]
        else:
            batch_data = list(batch)

        batch = collate_fn(batch_data)  # type: ignore
        model_outputs = self._model_forward(batch)
        if self.processor is not None:
            outputs = self.processor.post_process(model_outputs, batch)
        else:
            outputs = model_outputs
        return outputs

    def _inference_single(self, data: InputType) -> OutputType:
        """Executes the standard end-to-end inference workflow for one sample.

        This method applies optional pre-processing, optional single-sample
        collation, model forwarding, and optional post-processing for a scalar
        input.

        When a collate function is configured, the single pre-processed sample
        is still wrapped into a size-1 batch before model forward. As a
        result, ``processor.post_process`` usually still receives batched
        inputs and outputs in this path.

        Args:
            data (InputType): The raw input data for the pipeline.

        Returns:
            OutputType: The final, post-processed result.
        """
        if self.processor is not None:
            data = self.processor.pre_process(data)
        if self.collate_fn is not None:
            batch = self.collate_fn([data])
        else:
            batch = data  # type: ignore
        model_outputs = self._model_forward(batch)
        if self.processor is not None:
            outputs = self.processor.post_process(model_outputs, batch)
        else:
            outputs = model_outputs

        return outputs

    def _model_forward(self, data: Any) -> Any:
        """Perform the model's forward pass.

        For simple models, this method directly calls the model with the
        prepared batch. More specialized pipelines, such as ones wrapping
        auto-regressive generation models, can override this method to
        implement task-specific forward or generation logic.

        Args:
            data (Any): Input data for the model. It is already batched when a
                collate function is configured or when dataset-like inference
                is used.

        Returns:
            Any: Raw model outputs before optional post-processing.
        """
        data = to_device(data, self.model.device)
        return self.model(data)


InferencePipelineType_co = TypeVar(
    "InferencePipelineType_co",
    bound=InferencePipeline,
    covariant=True,
)


class InferencePipelineCfg(
    InferencePipelineMixinCfg[InferencePipelineType_co]
):
    """Configuration for the concrete :class:`InferencePipeline`.

    This class extends :class:`InferencePipelineMixinCfg` with additional,
    runtime-specific settings for data handling, including the processor,
    collate function, and mini-batch size.
    """

    class_type: ClassType_co[InferencePipelineType_co] = InferencePipeline  # type: ignore # noqa: E501

    processor: ConfigInstanceOf[ModelIOProcessorCfg] | None = None
    """The configuration for the model I/O processor."""

    collate_fn: CallableType[[list[Any]], Any] | CollatorConfig | None = None
    """The function used to collate single data items into a batch.

    This setting is required when the input data is a dataset-like iterable,
    such as a dataset, list, tuple, or generator.
    """

    batch_size: int = 1
    """The number of samples to process in each mini-batch during inference."""
