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

"""Inference pipeline components and utilities.

The inference pipeline focuses on the data processing and model
inference parts, and can be used in various scenarios. The inference
pipeline is a more generic concept than policy, which is
used to interact with the environment.
"""

from .basic import InferencePipeline, InferencePipelineCfg
from .mixin import (
    ClassType_co,
    InferencePipelineMixin,
    InferencePipelineMixinCfg,
)
