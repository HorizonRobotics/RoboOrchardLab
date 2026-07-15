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

from robo_orchard_lab.pipeline.hooks.mixin import (
    PipelineHooks,
    PipelineHooksConfig,
)

__all__ = ["OptimizerHook", "OptimizerHookConfig"]


class OptimizerHook(PipelineHooks):
    """Deprecated no-op compatibility hook.

    ``HookBasedTrainer`` owns optimizer and scheduler stepping internally.
    This hook remains only to keep old import/config paths loadable.

    """

    def __init__(self, cfg: OptimizerHookConfig | None):
        super().__init__()
        warnings.warn(
            "OptimizerHook is deprecated and is now a no-op. "
            "HookBasedTrainer owns optimizer and scheduler stepping "
            "internally.",
            DeprecationWarning,
            stacklevel=2,
        )


class OptimizerHookConfig(PipelineHooksConfig[OptimizerHook]):
    class_type: type[OptimizerHook] = OptimizerHook
