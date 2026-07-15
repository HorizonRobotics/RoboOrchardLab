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

import pytest

from robo_orchard_lab.pipeline.hooks.optimizer import OptimizerHook


def test_training_hooks_compat_import_exports_optimizer_hook_config():
    """The legacy training.hooks package remains import-compatible."""
    from robo_orchard_lab.pipeline.training.hooks import OptimizerHookConfig

    assert OptimizerHookConfig.__name__ == "OptimizerHookConfig"


def test_optimizer_hook_is_deprecated_noop():
    """Deprecated OptimizerHook no longer registers optimizer side effects."""
    with pytest.warns(DeprecationWarning, match="OptimizerHook is deprecated"):
        hook = OptimizerHook(None)

    assert len(hook.hooks["on_step"]) == 0
