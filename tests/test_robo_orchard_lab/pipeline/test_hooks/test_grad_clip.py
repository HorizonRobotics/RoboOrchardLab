# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
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
from unittest.mock import MagicMock

import pytest
import torch
from accelerate.utils import DistributedType

from robo_orchard_lab.pipeline.hooks.grad_clip import (
    GradientClippingHookConfig,
)
from robo_orchard_lab.pipeline.hooks.mixin import PipelineHookArgs


def test_gradient_clipping_uses_optimizer_step_boundary():
    """Gradient clipping runs only on optimizer-step boundaries."""
    param = torch.nn.Parameter(torch.tensor([1.0]))
    param.grad = torch.tensor([2.0])
    optimizer = MagicMock()
    optimizer.param_groups = [{"params": [param]}]
    accelerator = MagicMock()
    with pytest.warns(FutureWarning, match="grad_clip"):
        hook = GradientClippingHookConfig(
            clip_mode="norm",
            max_norm=1.0,
        )()

    hook._gradient_clipping(
        PipelineHookArgs(
            accelerator=accelerator,
            optimizer=optimizer,
            is_optimizer_step_boundary=False,
        )
    )

    accelerator.clip_grad_norm_.assert_not_called()

    hook._gradient_clipping(
        PipelineHookArgs(
            accelerator=accelerator,
            optimizer=optimizer,
            is_optimizer_step_boundary=True,
        )
    )

    accelerator.clip_grad_norm_.assert_called_once()


def test_gradient_clipping_hook_warns_about_trainer_owned_config():
    """Legacy hook construction should point users to trainer grad_clip."""

    with pytest.warns(
        FutureWarning,
        match=r"HookBasedTrainer\(\.\.\., grad_clip=",
    ):
        GradientClippingHookConfig(
            clip_mode="norm",
            max_norm=1.0,
        )()


def test_gradient_clipping_hook_rejects_deepspeed_runtime():
    """DeepSpeed must use the trainer-owned grad_clip config path."""

    param = torch.nn.Parameter(torch.tensor([1.0]))
    param.grad = torch.tensor([2.0])
    optimizer = MagicMock()
    optimizer.param_groups = [{"params": [param]}]
    accelerator = MagicMock()
    accelerator.distributed_type = DistributedType.DEEPSPEED
    with pytest.warns(FutureWarning, match="grad_clip"):
        hook = GradientClippingHookConfig(
            clip_mode="norm",
            max_norm=1.0,
        )()

    with pytest.raises(RuntimeError, match="DeepSpeed"):
        hook._gradient_clipping(
            PipelineHookArgs(
                accelerator=accelerator,
                optimizer=optimizer,
                is_optimizer_step_boundary=True,
            )
        )

    accelerator.clip_grad_norm_.assert_not_called()


def test_gradient_clipping_hook_preserves_existing_step_exception():
    """The DeepSpeed guard should not hide a prior hook-scope exception."""

    optimizer = MagicMock()
    accelerator = MagicMock()
    accelerator.distributed_type = DistributedType.DEEPSPEED
    with pytest.warns(FutureWarning, match="grad_clip"):
        hook = GradientClippingHookConfig(
            clip_mode="norm",
            max_norm=1.0,
        )()

    hook._gradient_clipping(
        PipelineHookArgs(
            accelerator=accelerator,
            optimizer=optimizer,
            is_optimizer_step_boundary=True,
            exception=RuntimeError("original failure"),
        )
    )

    accelerator.clip_grad_norm_.assert_not_called()
