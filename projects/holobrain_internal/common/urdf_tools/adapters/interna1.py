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

"""InternA1 packer adapter.

The adapter exposes a single ``resolve_urdf_path`` hook that reads the URDF
path from a flat InternA1-schema config entry (``entry["urdf"]``). All other
alignment intent (arm chains, EE frame rotations, camera references) lives in
the per-embodiment ``alignment.yaml`` manifests, so this adapter carries no
discovery logic.
"""

from __future__ import annotations
from typing import Any, Mapping, Optional

from projects.holobrain_internal.common.urdf_tools.adapters import (
    register_adapter,
)
from projects.holobrain_internal.common.urdf_tools.adapters.base import (
    DatasetAdapter,
)


@register_adapter("interna1")
class InternA1Adapter(DatasetAdapter):
    """Adapter for the ``config_interna1_dataset`` packer."""

    packer_module = "config_interna1_dataset"

    def resolve_urdf_path(self, entry: Mapping[str, Any]) -> Optional[str]:
        """Read ``entry["urdf"]`` from a flat InternA1-schema entry."""

        value = entry.get("urdf")
        if isinstance(value, str) and value:
            return value
        return None
