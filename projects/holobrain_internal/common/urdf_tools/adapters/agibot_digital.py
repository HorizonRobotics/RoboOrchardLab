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

"""AgiBot-Digital packer adapter.

The digit (G1) and geniesim (G2) configs both expose their per-embodiment
kinematics dict flat at the top level (``entry["urdf"]``), unlike the older
``agibot`` config whose entries nest the URDF path under
``kinematics_config``. This adapter mirrors ``abc130k`` / ``interna1`` so the
alignment CLI can route ``agibot_digital/{g1,g2}_omnipicker`` cases without
special-casing the flat shape.
"""

from __future__ import annotations
from typing import Any, Mapping, Optional

from projects.holobrain_internal.common.urdf_tools.adapters import (
    register_adapter,
)
from projects.holobrain_internal.common.urdf_tools.adapters.base import (
    DatasetAdapter,
)


@register_adapter("agibot_digital")
class AgibotDigitalAdapter(DatasetAdapter):
    """Adapter for flat AgiBot-Digital entries with ``entry["urdf"]``.

    The digit / geniesim configs (``config_agibot_digit_dataset`` /
    ``config_agibot_geniesim_dataset``) each export a single module-level
    kinematics dict wrapped by a ``get_kinematics_config()`` getter, which
    returns ``{"agibot_digit": g1_kinematics_config}`` or
    ``{"agibot_geniesim": g2_kinematics_config}`` respectively. The URDF
    path lives at the top level of each value dict, so resolution reads
    ``entry["urdf"]`` directly (identical to InternA1 / ABC130k shapes).
    """

    packer_module = "config_agibot_digital_dataset"

    def resolve_urdf_path(self, entry: Mapping[str, Any]) -> Optional[str]:
        value = entry.get("urdf")
        if isinstance(value, str) and value:
            return value
        return None
