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

"""Dataset-adapter contract for URDF path resolution.

Adapters translate one dataset's config-entry schema into the ``origin_urdf``
path the alignment pipeline needs. Only the ``resolve_urdf_path`` hook remains
runtime-critical — the yaml-first alignment flow authors every other field
by hand in the per-embodiment ``alignment.yaml`` manifests, so the old
discover-driven schema (camera reference discovery, joint semantics,
extrinsic rotation autoprobing) is gone.

One adapter per packer family; the concrete class is registered by string
via :func:`register_adapter`.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional


class DatasetAdapter(ABC):
    """Abstract adapter contract."""

    packer_module: str = ""

    @abstractmethod
    def resolve_urdf_path(self, entry: Mapping[str, Any]) -> Optional[str]:
        """Return the URDF path recorded on a packer-config entry.

        Different packer families store the URDF path at different keys —
        InternA1 keeps it flat under ``entry["urdf"]``, RoboTwin nests it
        under ``entry["kinematics_config"]["urdf"]``. The alignment tools
        route every packer-schema-shape decision through this method so
        the rest of the pipeline (case loader, wiring check, tests) does
        not need to know about the difference.

        Returns:
            The URDF path string (typically ``./urdf/...`` relative to
            ``projects/holobrain_internal/common``) or ``None`` when the
            entry does not carry one.
        """
