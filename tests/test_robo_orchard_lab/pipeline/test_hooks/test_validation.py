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

from unittest.mock import MagicMock

import torch

from robo_orchard_lab.pipeline.hooks.mixin import PipelineHookArgs
from robo_orchard_lab.pipeline.hooks.validation import ValidationHookConfig


def test_validation_runs_after_committed_optimizer_step():
    """Step validation should run only after committed optimizer steps."""

    callback = MagicMock()
    hook = ValidationHookConfig(
        eval_callback=callback,
        step_eval_freq=2,
    )()
    model = torch.nn.Linear(1, 1)
    args = PipelineHookArgs(
        accelerator=MagicMock(),
        model=model,
        global_step_id=2,
        step_id=2,
        is_optimizer_step_committed=True,
    )

    with hook.begin("on_optimizer_step", args):
        pass

    callback.assert_called_once_with(model)


def test_validation_skips_uncommitted_optimizer_boundary():
    """Skipped optimizer boundaries should not run step validation."""

    callback = MagicMock()
    hook = ValidationHookConfig(
        eval_callback=callback,
        step_eval_freq=1,
    )()
    args = PipelineHookArgs(
        accelerator=MagicMock(),
        model=torch.nn.Linear(1, 1),
        global_step_id=0,
        step_id=0,
        is_optimizer_step_committed=False,
    )

    with hook.begin("on_optimizer_step", args):
        pass

    callback.assert_not_called()


def test_validation_epoch_end_ignores_step_frequency():
    """Epoch validation should not be triggered by step_eval_freq."""

    callback = MagicMock()
    hook = ValidationHookConfig(
        eval_callback=callback,
        step_eval_freq=2,
        epoch_eval_freq=999,
    )()
    args = PipelineHookArgs(
        accelerator=MagicMock(),
        model=torch.nn.Linear(1, 1),
        epoch_id=0,
        global_step_id=2,
        step_id=2,
        is_optimizer_step_committed=False,
    )

    with hook.begin("on_epoch", args):
        pass

    callback.assert_not_called()
