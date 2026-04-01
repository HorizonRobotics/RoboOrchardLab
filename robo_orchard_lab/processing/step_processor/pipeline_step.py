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
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Optional,
    Tuple,
)

import torch
from accelerate import Accelerator

from robo_orchard_lab.pipeline.hooks.mixin import (
    ModelOutput,
    ModelOutputHasLossKeys,
    PipelineHookArgs,
    PipelineHooks,
)
from robo_orchard_lab.processing.io_processor import ModelIOProcessor

__all__ = [
    "BatchStepProcessorMixin",
    "DeprecatedError",
    "LossNotProvidedError",
    "SimpleStepProcessor",
    "StepProcessorFromCallable",
]


class DeprecatedError(Exception):
    """Legacy compatibility error retained from historical constructors.

    Step processor constructors no longer accept the old ``transforms``
    argument, but the symbol is kept to avoid breaking deprecated import
    surfaces that may still reference it.
    """

    pass


logger = logging.getLogger(__name__)


forward_fn_type = Callable[[Callable, Any], Tuple[Any, Optional[torch.Tensor]]]


class LossNotProvidedError(Exception):
    """Raised when backward is requested but no loss is returned."""

    pass


class BatchStepProcessorMixin(ABC):
    """Abstract interface for executing one batch in a runtime loop.

    A step processor coordinates the execution of a single batch inside a
    training, evaluation, or inference loop. Unlike an I/O processor, which
    only transforms model inputs and outputs, a step processor owns the model
    forward call, optional loss computation, optional backward propagation, and
    the hook-visible bookkeeping around that batch.
    """

    @abstractmethod
    def __call__(
        self,
        pipeline_hooks: PipelineHooks,
        on_batch_hook_args: PipelineHookArgs,
        model: Callable,
    ) -> None:
        """Execute the batch processing pipeline.

        The processed outputs and reduced loss, if any, are stored in
        ``on_batch_hook_args`` for downstream hooks and loop logic.

        Args:
            pipeline_hooks (PipelineHooks): Hook container used to wrap the
                batch lifecycle.
            on_batch_hook_args (PipelineHookArgs): Workspace for the current
                batch. Before the call it should at least contain
                ``accelerator`` and ``batch``. After the call it contains
                ``model_outputs`` and ``reduce_loss``.
            model (Callable): The model function or callable.
        """
        pass


class SimpleStepProcessor(BatchStepProcessorMixin):
    """A default processor for handling batches in a runtime step.

    This class wraps the common step lifecycle around :meth:`forward`:

    1. Optionally pre-process the raw batch with ``io_processor``.
    2. Execute the model forward pass via :meth:`forward`.
    3. Optionally post-process the raw model outputs.
    4. Publish outputs and reduced loss into hook arguments.
    5. Run backward when ``need_backward`` is enabled.

    Subclasses typically only need to implement :meth:`forward`.
    """

    def __init__(
        self,
        need_backward: bool = True,
        *,
        io_processor: ModelIOProcessor | None = None,
        apply_post_process: bool = False,
    ) -> None:
        """Initializes the step processor.

        Args:
            need_backward (bool, optional): Whether backward computation is
                needed. When True, :meth:`forward` must return a loss tensor.
                Default is True.
            io_processor (ModelIOProcessor | None, optional): Optional
                model I/O processor used to pre-process batches before the
                forward pass and optionally post-process model outputs.
                Default is None.
            apply_post_process (bool, optional): Whether to call
                ``io_processor.post_process`` after the forward pass. This is
                usually enabled for evaluation or deployment, and disabled for
                training so hooks can inspect raw model outputs. Default is
                False.
        """
        self.need_backward = need_backward
        self.io_processor = io_processor
        self.apply_post_process = apply_post_process
        self._is_prepared = False
        self.accelerator: Optional[Accelerator] = None

    @staticmethod
    def from_callable(
        forward_fn: forward_fn_type,
        need_backward: bool = True,
        *,
        io_processor: ModelIOProcessor | None = None,
        apply_post_process: bool = False,
    ) -> "StepProcessorFromCallable":
        """Create a :class:`SimpleStepProcessor` from a plain callable.

        This factory is useful for tests or lightweight integration points
        where defining a dedicated subclass would add unnecessary boilerplate.

        Args:
            forward_fn (forward_fn_type): Callable implementing the forward
                step. It must accept ``(model, batch)`` and return
                ``(outputs, loss)``.
            need_backward (bool, optional): Whether backward computation is
                needed. Default is True.
            io_processor (ModelIOProcessor | None, optional): Optional model
                I/O processor used to transform batches and outputs. Default is
                None.
            apply_post_process (bool, optional): Whether to call
                ``io_processor.post_process`` after forward execution. Default
                is False.

        Returns:
            StepProcessorFromCallable: A callable-backed step processor.
        """
        return StepProcessorFromCallable(
            forward_fn=forward_fn,
            need_backward=need_backward,
            io_processor=io_processor,
            apply_post_process=apply_post_process,
        )

    def _initialize(self, accelerator: Accelerator) -> None:
        """Prepare owned modules with the current accelerator once.

        Any ``torch.nn.Module`` stored on the processor instance is passed
        through ``accelerator.prepare`` the first time the processor executes a
        batch. This keeps helper modules aligned with the same distributed
        runtime setup as the surrounding loop.

        Args:
            accelerator (Accelerator): Accelerator used to prepare owned
                modules.
        """
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
    ) -> Tuple[ModelOutput | ModelOutputHasLossKeys, Optional[torch.Tensor]]:
        """Define the forward pass logic for the model.

        This method handles the execution of the forward pass, processes the
        prepared batch, and computes the outputs of the model. It also
        optionally computes a loss tensor when the step requires backward
        propagation.

        Args:
            model (Callable): The model to be used for inference or training.
                It should be a callable object such as a PyTorch module or a
                plain function.
            batch (Any): The batch data for the model. This may be a tuple,
                dictionary, tensor, or another structure depending on the data
                pipeline and optional ``io_processor``.

        Returns:
            tuple: A pair of the model output and an optional reduced loss
                tensor. The first element is usually a dict or a custom
                ``ModelOutput`` value and may provide ``loss_keys`` when
                multiple loss terms are present. The second element may be None
                for forward-only steps such as pure inference.

        Notes:
            - In most cases, the accelerator already ensures that the model
              and batch data are on the correct device before the forward
              pass.
            - If additional operations or modules are introduced here, it is
              the implementation's responsibility to keep them on the correct
              device.
            - The returned loss tensor should already be reduced, for example
              by taking a mean over the batch, so it can be used directly for
              backward propagation.
            - This method does not handle backpropagation; it focuses solely
              on the forward computation.
            - Any input transformation should already be handled by the data
              pipeline or ``io_processor`` before this method runs.
        """
        pass

    def __call__(
        self,
        pipeline_hooks: PipelineHooks,
        on_batch_hook_args: PipelineHookArgs,
        model: Callable,
    ) -> None:
        """Execute one full batch step.

        This method wires the processor into the hook lifecycle. It optionally
        pre-processes the raw batch, runs the ``on_model_forward`` hook scope
        around :meth:`forward`, reduces the loss across processes when needed,
        optionally runs the ``on_model_backward`` hook scope, and writes the
        final outputs back into ``on_batch_hook_args``.

        Args:
            pipeline_hooks (PipelineHooks): Hook container used to wrap forward
                and backward execution.
            on_batch_hook_args (PipelineHookArgs): Workspace for the current
                batch. Before the call it should at least provide
                ``accelerator`` and ``batch``. After the call it stores
                ``model_outputs`` and ``reduce_loss``.
            model (Callable): The model function or callable.

        Raises:
            LossNotProvidedError: If ``need_backward`` is True but
                :meth:`forward` returns ``loss=None``.
        """
        self._initialize(accelerator=on_batch_hook_args.accelerator)
        batch = on_batch_hook_args.batch
        if self.io_processor is not None:
            model_input = self.io_processor.pre_process(batch)
        else:
            model_input = batch

        with pipeline_hooks.begin(
            "on_model_forward",
            arg=on_batch_hook_args.copy_with_updates(batch=model_input),
        ) as on_forward_hook_args:
            accelerator = on_forward_hook_args.accelerator
            self.accelerator = accelerator
            raw_outputs, loss = self.forward(model=model, batch=model_input)
            if self.io_processor is not None and self.apply_post_process:
                outputs = self.io_processor.post_process(
                    raw_outputs, model_input
                )
            else:
                outputs = raw_outputs

            on_forward_hook_args.model_outputs = outputs
            reduce_loss: torch.Tensor | None = loss
            if accelerator.num_processes > 1 and loss is not None:
                reduce_loss = accelerator.reduce(loss, reduction="mean")  # type: ignore

            on_forward_hook_args.reduce_loss = reduce_loss

        if self.need_backward:
            if loss is None:
                raise LossNotProvidedError()

            with pipeline_hooks.begin(
                "on_model_backward",
                arg=on_batch_hook_args.copy_with_updates(
                    batch=model_input,
                    model_outputs=outputs,
                    reduce_loss=reduce_loss,
                ),
            ) as on_backward_hook_args:
                on_backward_hook_args.accelerator.backward(loss)

        on_batch_hook_args.model_outputs = outputs
        on_batch_hook_args.reduce_loss = reduce_loss


class StepProcessorFromCallable(SimpleStepProcessor):
    """A :class:`SimpleStepProcessor` built from a forward callable.

    This adapter makes it easy to plug a plain callable into trainer or test
    code without defining a dedicated processor subclass.
    """

    def __init__(
        self,
        forward_fn: forward_fn_type,
        need_backward: bool = True,
        *,
        io_processor: ModelIOProcessor | None = None,
        apply_post_process: bool = False,
    ) -> None:
        """Initialize the callable-backed step processor.

        Args:
            forward_fn (forward_fn_type): Callable implementing the forward
                step logic.
            need_backward (bool, optional): Whether backward propagation is
                required. Default is True.
            io_processor (ModelIOProcessor | None, optional): Optional model
                I/O processor used to transform batches and outputs. Default is
                None.
            apply_post_process (bool, optional): Whether to call
                ``io_processor.post_process`` after forward execution. Default
                is False.
        """
        super().__init__(
            need_backward=need_backward,
            io_processor=io_processor,
            apply_post_process=apply_post_process,
        )

        self._forward_fn = forward_fn

    def forward(
        self,
        model: Callable,
        batch: Any,
    ) -> Tuple[Any, Optional[torch.Tensor]]:
        """Delegate the forward step to the wrapped callable.

        Args:
            model (Callable): The model function or callable.
            batch (Any): Prepared batch data to pass to ``forward_fn``.

        Returns:
            Tuple[Any, Optional[torch.Tensor]]: The output tuple returned by
                ``forward_fn``.
        """
        return self._forward_fn(model, batch)
