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
from typing import Callable

import torch

from robo_orchard_lab.pipeline.hooks.mixin import (
    HookContext,
    PipelineHookArgs,
    PipelineHooks,
    PipelineHooksConfig,
)
from robo_orchard_lab.utils.torch import switch_model_mode

__all__ = ["ValidationHook", "ValidationHookConfig"]


class ValidationHook(PipelineHooks):
    """Run a caller-provided evaluation callback at training boundaries.

    Step-frequency evaluation is tied to committed optimizer steps through
    ``on_optimizer_step``. Epoch-frequency evaluation is tied to ``on_epoch``
    and does not reuse the step schedule.

    Args:
        cfg (ValidationHookConfig): The configuration for the ValidationHook.
            Please refer to ValidationHookConfig for details.
    """

    def __init__(
        self,
        cfg: ValidationHookConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.step_eval_freq = cfg.step_eval_freq
        self.epoch_eval_freq = cfg.epoch_eval_freq

        if cfg.epoch_eval_freq is not None:
            self.register_hook(
                "on_epoch",
                HookContext.from_callable(
                    after=self._on_epoch_end, before=None
                ),
            )
        if cfg.step_eval_freq is not None:
            self.register_hook(
                "on_optimizer_step",
                HookContext.from_callable(
                    after=self._on_step_end, before=None
                ),
            )

        if cfg.eval_at_begin:
            self.register_hook(
                "on_loop",
                HookContext.from_callable(
                    before=self._on_loop_begin, after=None
                ),
            )

    def evaluate(self, hook_args: PipelineHookArgs) -> None:
        """Switch the model to eval mode and call the configured callback.

        Args:
            hook_args (PipelineHookArgs): Hook context carrying the model to
                evaluate.

        Raises:
            ValueError: If the hook context does not carry a model.
        """
        if hook_args.model is None:
            raise ValueError("Model is not set in the hook arguments.")
        with switch_model_mode(hook_args.model, target_mode="eval"):
            self.cfg.eval_callback(hook_args.model)

    def need_step_eval(
        self,
        hook_args: PipelineHookArgs,
    ) -> bool:
        """Checks if step-frequency evaluation is needed.

        This predicate is used only from ``on_optimizer_step`` after-hooks, so
        step-frequency validation cannot be triggered from epoch-end events.
        """
        return (
            self.step_eval_freq is not None
            and hook_args.global_step_id % self.step_eval_freq == 0
        )

    def need_epoch_eval(
        self,
        hook_args: PipelineHookArgs,
    ) -> bool:
        """Checks if epoch-frequency evaluation is needed.

        This predicate is used only from ``on_epoch`` after-hooks and ignores
        ``step_eval_freq``.
        """
        return (
            self.epoch_eval_freq is not None
            and (hook_args.epoch_id + 1) % self.epoch_eval_freq == 0
        )

    def _on_loop_begin(
        self,
        hook_args: PipelineHookArgs,
    ) -> None:
        """Run optional evaluation before the first training step."""
        if self.cfg.eval_at_begin:
            self.evaluate(hook_args)

    def _on_step_end(
        self,
        hook_args: PipelineHookArgs,
    ) -> None:
        """Run step-frequency eval after committed optimizer steps only."""
        if hook_args.exception is not None:
            return
        if not hook_args.is_optimizer_step_committed:
            return

        if self.need_step_eval(hook_args):
            self.evaluate(hook_args)

    def _on_epoch_end(
        self,
        hook_args: PipelineHookArgs,
    ) -> None:
        """Run epoch-frequency eval after matching epoch scopes only."""
        if hook_args.exception is not None:
            return

        if self.need_epoch_eval(hook_args):
            self.evaluate(hook_args)


class ValidationHookConfig(PipelineHooksConfig[ValidationHook]):
    """Configuration class for ValidationHook."""

    class_type: type[ValidationHook] = ValidationHook

    eval_callback: Callable[[torch.nn.Module], None]
    """A callback function to be called for evaluation. This function should
    take model as input and should not return any values. A common use case
    is to pass a closure that performs the evaluation.
    """
    step_eval_freq: int | None = None
    """The frequency of evaluation in committed optimizer-step units.

    If specified, the evaluation will be performed every `step_eval_freq`
    committed optimizer steps.
    """
    epoch_eval_freq: int | None = None
    """The frequency of evaluation in terms of epochs. If specified, the
    evaluation will be performed every `epoch_eval_freq` epochs."""

    eval_at_begin: bool = False
    """If True, evaluation will be performed at the beginning of training. """

    def __post_init__(self):
        if self.step_eval_freq is None and self.epoch_eval_freq is None:
            raise ValueError(
                "Either `step_eval_freq` or `epoch_eval_freq` "
                "must be specified."
            )
        if self.step_eval_freq is not None and self.step_eval_freq < 1:
            raise ValueError(
                "step_eval_freq = {} < 1 is not allowed".format(
                    self.step_eval_freq
                )
            )
        if self.epoch_eval_freq is not None and self.epoch_eval_freq < 1:
            raise ValueError(
                "epoch_eval_freq = {} < 1 is not allowed".format(
                    self.epoch_eval_freq
                )
            )
