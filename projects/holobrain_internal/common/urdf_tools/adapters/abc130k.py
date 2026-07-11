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

"""ABC130k packer adapter."""

from __future__ import annotations
from typing import Any, Mapping, Optional

from projects.holobrain_internal.common.urdf_tools.adapters import (
    register_adapter,
)
from projects.holobrain_internal.common.urdf_tools.adapters.base import (
    DatasetAdapter,
)


@register_adapter("abc130k")
class Abc130kAdapter(DatasetAdapter):
    """Adapter for flat ABC130k entries with ``entry["urdf"]``.

    ABC130k's ``dataset_config[setting_type]`` records the URDF path flat at
    the top level (same shape as AgileX), so the resolution mirrors the
    AgileX adapter rather than the nested ``kinematics_config.urdf`` shape
    used by AgiBot.
    """

    packer_module = "config_abc130k_dataset"

    def resolve_urdf_path(self, entry: Mapping[str, Any]) -> Optional[str]:
        value = entry.get("urdf")
        if isinstance(value, str) and value:
            return value
        return None
