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
from typing import Literal

import torch
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.utils import DistributedType

from robo_orchard_lab.pipeline.hooks.mixin import (
    HookContext,
    PipelineHookArgs,
    PipelineHooks,
    PipelineHooksConfig,
)

__all__ = ["GradientClippingHook", "GradientClippingHookConfig"]


def _clip_optimizer_gradients(
    *,
    accelerator: Accelerator,
    optimizer: AcceleratedOptimizer,
    clip_mode: Literal["norm", "value"],
    clip_value: float | None,
    max_norm: float | None,
    norm_type: float,
) -> None:
    """Apply gradient clipping to trainable parameters with gradients."""

    params: list[torch.Tensor] = []
    for param_group in optimizer.param_groups:
        params.extend(param_group["params"])

    params = [
        param
        for param in params
        if param.requires_grad and param.grad is not None
    ]
    if len(params) == 0:
        return

    if clip_mode == "value":
        accelerator.clip_grad_value_(params, clip_value)
    elif clip_mode == "norm":
        accelerator.clip_grad_norm_(params, max_norm, norm_type)


class GradientClippingHook(PipelineHooks):
    """Compatibility hook for gradient clipping during training.

    This hook is responsible for clipping model gradients to prevent
    exploding gradients. It performs clipping after each ``on_step`` scope
    when used directly as a pipeline hook.

    Note:
        ``HookBasedTrainer`` consumes ``GradientClippingHookConfig`` as a
        trainer-owned optimizer-finalization setting through
        ``HookBasedTrainer(..., grad_clip=GradientClippingHookConfig(...))``.
        Passing this already constructed hook directly is a compatibility
        hook surface and does not participate in trainer-owned DeepSpeed
        gradient clipping.

        Precomposed ``PipelineHooks`` can hide this hook from trainer
        initialization-time validation. On non-DeepSpeed backends, mixing
        hook-form clipping with trainer-owned ``grad_clip=...`` can therefore
        clip the same gradients twice. DeepSpeed requires the trainer-owned
        ``grad_clip=...`` path because clipping must be lowered into the
        DeepSpeed config before ``accelerator.prepare(...)``.

    Args:
        cfg (GradientClippingHookConfig): The configuration for the
            GradientClippingHook.
    """

    def __init__(
        self,
        cfg: GradientClippingHookConfig,
    ):
        super().__init__()
        warnings.warn(
            "GradientClippingHook is kept for compatibility, but "
            "HookBasedTrainer(..., grad_clip=GradientClippingHookConfig(...)) "
            "is the recommended gradient clipping API. Hook-form clipping can "
            "double clip if it is combined with trainer-owned grad_clip, and "
            "DeepSpeed requires the trainer-owned grad_clip path.",
            FutureWarning,
            stacklevel=2,
        )
        self.clip_mode = cfg.clip_mode
        self.clip_value = cfg.clip_value
        self.max_norm = cfg.max_norm
        self.norm_type = cfg.norm_type

        self.register_hook(
            "on_step",
            HookContext.from_callable(
                after=self._gradient_clipping,
                before=None,
            ),
        )

    def _gradient_clipping(
        self,
        hook_args: PipelineHookArgs,
    ) -> None:
        """Performs gradient clipping.

        Args:
            hook_args (PipelineHookArgs): The workspace for the gradient
                clipping hook. It should contain the following attributes:
                  - accelerator: The Accelerator instance.
                - optimizer: The optimizer instance.

        """
        if hook_args.exception is not None:
            return

        if hook_args.accelerator.distributed_type == DistributedType.DEEPSPEED:
            raise RuntimeError(
                "GradientClippingHook cannot run when DeepSpeed is active. "
                "Pass HookBasedTrainer(..., "
                "grad_clip=GradientClippingHookConfig(...)) so gradient "
                "clipping is configured before accelerator.prepare(...)."
            )

        if hook_args.optimizer is None:
            raise ValueError("Optimizer is not set in the hook arguments.")

        optimizer = hook_args.optimizer
        accelerator = hook_args.accelerator

        if hook_args.is_optimizer_step_boundary:
            _clip_optimizer_gradients(
                accelerator=accelerator,
                optimizer=optimizer,
                clip_mode=self.clip_mode,
                clip_value=self.clip_value,
                max_norm=self.max_norm,
                norm_type=self.norm_type,
            )


class GradientClippingHookConfig(PipelineHooksConfig[GradientClippingHook]):
    """Configuration for trainer-owned gradient clipping.

    Prefer passing this config to ``HookBasedTrainer(..., grad_clip=...)``.
    That path lets ordinary backends clip immediately before
    ``optimizer.step`` and lets DeepSpeed receive equivalent clipping config
    before ``accelerator.prepare(...)``. Instantiating this config as a
    ``GradientClippingHook`` remains available for compatibility, but hook-form
    clipping can double clip if mixed with trainer-owned ``grad_clip=...``.
    """

    class_type: type[GradientClippingHook] = GradientClippingHook

    clip_mode: Literal["norm", "value"]
    """The mode of gradient clipping.
        - "norm": Clips gradients by norm.
        - "value": Clips gradients by value.
    """
    clip_value: float | None = None
    """ The maximum norm to clip the gradients to. This parameter
    is only used when `clip_mode` is "norm"."""
    max_norm: float | None = None
    """The maximum norm to clip the gradients to. This parameter is only
    used when `clip_mode` is "norm"."""
    norm_type: float = 2.0
    """The type of norm to use for clipping. Default is 2.0 (L2 norm)."""

    def __post_init__(self) -> None:
        """Post-initialization method to validate the configuration."""
        if self.clip_mode == "value":
            if self.clip_value is None:
                raise ValueError(
                    "clip_value must be specified when clip_mode is 'value'."
                )
            if self.clip_value < 0:
                raise ValueError(
                    "clip_value must be non-negative when clip_mode "
                    "is 'value'."
                )
        elif self.clip_mode == "norm":
            if self.max_norm is None:
                raise ValueError(
                    "max_norm must be specified when clip_mode is 'norm'."
                )
            if self.max_norm < 0:
                raise ValueError(
                    "max_norm must be non-negative when clip_mode is 'norm'."
                )
