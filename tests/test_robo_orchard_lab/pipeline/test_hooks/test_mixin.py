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

from dataclasses import replace
from unittest.mock import MagicMock

import pytest
import torch

from robo_orchard_lab.pipeline.hooks.mixin import (
    HookContext,
    PipelineHookArgs,
    PipelineHooks,
)


def test_pipeline_hooks_records_body_exception_before_after_hook():
    """After hooks should see the original body exception."""

    hooks = PipelineHooks()
    after_seen: dict[str, BaseException | None] = {}

    def after_hook(args: PipelineHookArgs) -> None:
        after_seen["exception"] = args.exception

    hooks.register_hook("on_step", HookContext.from_callable(after=after_hook))
    args = PipelineHookArgs(accelerator=MagicMock())

    with pytest.raises(RuntimeError, match="body failed") as exc_info:
        with hooks.begin("on_step", args):
            raise RuntimeError("body failed")

    assert args.exception is exc_info.value
    assert after_seen["exception"] is exc_info.value


def test_pipeline_hooks_supports_optimizer_step_context_channel():
    """The optimizer-step channel is a scoped context hook."""

    hooks = PipelineHooks()
    events: list[tuple[str, bool]] = []

    def before_hook(args: PipelineHookArgs) -> None:
        events.append(("before", args.is_optimizer_step_committed))

    def after_hook(args: PipelineHookArgs) -> None:
        events.append(("after", args.is_optimizer_step_committed))

    hooks.register_hook(
        "on_optimizer_step",
        HookContext.from_callable(before=before_hook, after=after_hook),
    )
    args = PipelineHookArgs(accelerator=MagicMock())

    with hooks.begin("on_optimizer_step", args):
        args.is_optimizer_step_committed = True

    assert events == [("before", False), ("after", True)]


def test_pipeline_hook_args_retires_backward_loss_access_after_construction():
    """Legacy construction works while retired loss reads fail closed."""

    args = PipelineHookArgs(
        accelerator=MagicMock(),
        reduced_backward_loss=torch.tensor(3.0),
    )

    copied = args.copy_with_updates(global_step_id=2)

    assert copied.global_step_id == 2
    with pytest.raises(RuntimeError, match="model_outputs"):
        _ = args.reduced_backward_loss
    with pytest.raises(RuntimeError, match="model_outputs"):
        _ = args.reduce_loss


def test_pipeline_hook_args_supports_standard_dataclass_replace():
    """Retired constructor compatibility must not break dataclass copying."""

    args = PipelineHookArgs(
        accelerator=MagicMock(),
        global_step_id=1,
        reduced_backward_loss=torch.tensor(3.0),
    )

    copied = replace(args, global_step_id=2)

    assert copied.global_step_id == 2
    assert copied.accelerator is args.accelerator
