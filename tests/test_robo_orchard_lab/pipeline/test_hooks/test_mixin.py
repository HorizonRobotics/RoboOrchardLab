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

from unittest.mock import MagicMock

import pytest

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
