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

import logging
from abc import abstractmethod
from typing import (
    Any,
    Callable,
    Optional,
    Sequence,
    Tuple,
    overload,
)

import torch
from accelerate import Accelerator
from typing_extensions import deprecated

from robo_orchard_lab.pipeline.batch_processor.mixin import BatchProcessorMixin
from robo_orchard_lab.pipeline.hooks.mixin import (
    PipelineHookArgs,
    PipelineHooks,
)

__all__ = [
    "LossNotProvidedError",
    "SimpleBatchProcessor",
    "BatchProcessorFromCallable",
]


class DeprecatedError(Exception):
    pass


logger = logging.getLogger(__name__)


forward_fn_type = Callable[[Callable, Any], Tuple[Any, Optional[torch.Tensor]]]


class LossNotProvidedError(Exception):
    pass


class SimpleBatchProcessor(BatchProcessorMixin):
    """A processor for handling batches in a training or inference pipeline."""

    @overload
    def __init__(self, need_backward: bool = True): ...

    @overload
    @deprecated(
        "transforms is not supported anymore. "
        "If you want to transform the input batch, "
        "please implement it in the forward function or "
        "in the data pipeline."
    )
    def __init__(
        self,
        need_backward: bool = True,
        transforms: Optional[Callable | Sequence[Callable]] = None,
    ): ...

    def __init__(
        self,
        need_backward: bool = True,
        transforms: Optional[Callable | Sequence[Callable]] = None,
    ) -> None:
        """Initializes the batch processor.

        Args:
            need_backward (bool): Whether backward computation is needed.
                If True, the loss should be provided during the forward pass.

        """

        if transforms is not None:
            raise DeprecatedError(
                "transforms is not supported anymore. "
                "If you want to transform the input batch, "
                "please implement it in the forward function or "
                "in the data pipeline."
            )

        self.need_backward = need_backward

        self._is_prepared = False
        self.accelerator: Optional[Accelerator] = None

    @staticmethod
    def from_callable(
        forward_fn: forward_fn_type,
        need_backward: bool = True,
        transforms: Optional[Callable | Sequence[Callable]] = None,
    ):
        """Creates a SimpleBatchProcessor instance from a callable.

        Args:
            forward_fn (Callable): The forward function to be used for
                processing batches.
            need_backward (bool): Whether backward computation is needed.
                If True, the loss should be provided during the forward pass.
            transforms (Optional[Callable | Sequence[Callable]]): A callable
                or a sequence of callables for transforming the batch.

        Returns:
            SimpleBatchProcessor: An instance of SimpleBatchProcessor.
        """
        return BatchProcessorFromCallable(
            forward_fn=forward_fn,
            need_backward=need_backward,
            transforms=transforms,
        )

    def _initialize(self, accelerator: Accelerator) -> None:
        if self._is_prepared:
            return

        for key, obj in vars(self).items():
            if isinstance(obj, torch.nn.Module):
                new_obj = accelerator.prepare(obj)
                setattr(self, key, new_obj)

        self._is_prepared = True

    @abstractmethod
    def forward(
        self,
        model: Callable,
        batch: Any,
    ) -> Tuple[Any, Optional[torch.Tensor]]:
        """Defines the forward pass logic for the model.

        This method handles the execution of the forward pass, processing
        the input batch and computing the outputs of the model. It also
        optionally computes a loss tensor if required for training.

        Args:
            model (Callable): The model to be used for inference or training.
                It should be a callable object (e.g., a PyTorch `nn.Module`
                or a function).
            batch (Any): The input batch data. This can be a tuple, dictionary,
                or other structure, depending on the data pipeline's format.

        Returns:
            Tuple[Any, Optional[torch.Tensor]]:
                - The first element is the model's outputs. It can be any type
                  that the model produces, such as a tensor, a list of tensors,
                  or a dictionary.
                - The second element is an optional loss tensor. This
                  is used during training when backward computation is required.
                  If loss is not applicable (e.g., during inference), this value
                  can be `None`.

        Notes:
            - In most cases, the `accelerator` will already ensure that both
              the model and the batch data are moved to the appropriate device
              before the forward pass.
            - However, if additional operations or modifications are performed
              on the batch data or model within this method, it is the
              responsibility of the implementation to confirm they remain on
              the correct device.
            - The returned loss tensor should already be reduced (e.g., mean or
              sum over batch elements) to facilitate the backward pass.
            - This method does not handle backpropagation; it focuses solely
              on the forward pass.
            - The transformation of the input batch, if needed, should already
              be handled prior to this method via the `self.transform` pipeline.
        """  # noqa: E501
        pass

    def __call__(
        self,
        pipeline_hooks: PipelineHooks,
        on_batch_hook_args: PipelineHookArgs,
        model: Callable,
    ) -> None:
        self._initialize(accelerator=on_batch_hook_args.accelerator)
        batch = on_batch_hook_args.batch
        with pipeline_hooks.begin(
            "on_model_forward",
            arg=on_batch_hook_args.copy_with_updates(batch=batch),
        ) as on_forward_hook_args:
            self.accelerator = on_forward_hook_args.accelerator
            outputs, loss = self.forward(model=model, batch=batch)
            on_forward_hook_args.model_outputs = outputs
            reduce_loss: torch.Tensor | None = loss
            if self.accelerator.num_processes > 1 and loss is not None:
                reduce_loss = self.accelerator.reduce(loss, reduction="mean")  # type: ignore

            on_forward_hook_args.reduce_loss = reduce_loss

        if self.need_backward:
            if loss is None:
                raise LossNotProvidedError()

            with pipeline_hooks.begin(
                "on_model_backward",
                arg=on_batch_hook_args.copy_with_updates(
                    batch=batch,
                    model_outputs=outputs,
                    reduce_loss=reduce_loss,
                ),
            ) as on_backward_hook_args:
                on_backward_hook_args.accelerator.backward(loss)

            # all reduce to get the average loss

        on_batch_hook_args.model_outputs = outputs
        on_batch_hook_args.reduce_loss = reduce_loss


class BatchProcessorFromCallable(
    SimpleBatchProcessor,
):
    """A processor for handling batches in a training or inference pipeline.

    This class is designed to be used as a callable object, allowing it to
    be easily integrated into various training or inference pipelines.
    It provides a flexible interface for processing batches of data and
    performing model inference or training.
    """

    @overload
    def __init__(self, forward_fn: forward_fn_type): ...

    @overload
    @deprecated(
        "transforms is not supported anymore. "
        "If you want to transform the input batch, "
        "please implement it in the forward function or "
        "in the data pipeline."
    )
    def __init__(
        self,
        forward_fn: forward_fn_type,
        need_backward: bool = True,
        transforms: Optional[Callable | Sequence[Callable]] = None,
    ): ...

    def __init__(
        self,
        forward_fn: forward_fn_type,
        need_backward: bool = True,
        transforms: Optional[Callable | Sequence[Callable]] = None,
    ) -> None:
        if transforms is not None:
            super().__init__(
                need_backward=need_backward, transforms=transforms
            )
        else:
            super().__init__(need_backward=need_backward)

        self._forward_fn = forward_fn

    def forward(
        self,
        model: Callable,
        batch: Any,
    ) -> Tuple[Any, Optional[torch.Tensor]]:
        return self._forward_fn(model, batch)
