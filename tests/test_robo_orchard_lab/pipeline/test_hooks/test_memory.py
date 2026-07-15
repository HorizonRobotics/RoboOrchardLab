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

from robo_orchard_lab.pipeline.hooks.memory import ClearCacheHookConfig
from robo_orchard_lab.pipeline.hooks.mixin import PipelineHookArgs


def test_clear_cache_runs_after_committed_optimizer_step(mocker):
    """Step cache clearing should run only after committed optimizer steps."""

    clear_device_cache = mocker.patch(
        "robo_orchard_lab.pipeline.hooks.memory.clear_device_cache"
    )
    hook = ClearCacheHookConfig(
        empty_cache_at="step",
        empty_cache_freq=2,
        garbage_collection=True,
    )()
    args = PipelineHookArgs(
        accelerator=MagicMock(),
        global_step_id=2,
        step_id=2,
        is_optimizer_step_committed=True,
    )

    with hook.begin("on_optimizer_step", args):
        pass

    clear_device_cache.assert_called_once_with(garbage_collection=True)


def test_clear_cache_skips_uncommitted_optimizer_boundary(mocker):
    """Skipped optimizer boundaries should not clear step cache."""

    clear_device_cache = mocker.patch(
        "robo_orchard_lab.pipeline.hooks.memory.clear_device_cache"
    )
    hook = ClearCacheHookConfig(empty_cache_at="step", empty_cache_freq=1)()
    args = PipelineHookArgs(
        accelerator=MagicMock(),
        global_step_id=0,
        step_id=0,
        is_optimizer_step_committed=False,
    )

    with hook.begin("on_optimizer_step", args):
        pass

    clear_device_cache.assert_not_called()
