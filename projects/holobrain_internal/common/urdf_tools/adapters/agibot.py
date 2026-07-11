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

"""AgiBot packer adapter."""

from __future__ import annotations
from typing import Any, Mapping, Optional

from projects.holobrain_internal.common.urdf_tools.adapters import (
    register_adapter,
)
from projects.holobrain_internal.common.urdf_tools.adapters.base import (
    DatasetAdapter,
)


@register_adapter("agibot")
class AgibotAdapter(DatasetAdapter):
    """Adapter for profile entries with ``kinematics_config.urdf``."""

    packer_module = "config_agibot_dataset"

    def resolve_urdf_path(self, entry: Mapping[str, Any]) -> Optional[str]:
        kinematics = entry.get("kinematics_config")
        if isinstance(kinematics, Mapping):
            value = kinematics.get("urdf")
            if isinstance(value, str) and value:
                return value
        return None
