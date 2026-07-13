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

"""Every alignment case wires into its packer config on disk.

Stage 4 of the alignment pipeline is a config-wiring check: the packer
config for every embodiment must actually reference the aligned URDF
(either through the companion ``*_urdf_v2`` key or via base-cutover on
the original key). If the wiring drifts, training runs silently pick up
the wrong URDF. The synthetic branch tests were removed —
``verify_case_config_wiring`` is fully exercised by the real cases
below, so those unit tests were tautological.
"""

from __future__ import annotations

import pytest

from projects.holobrain_internal.common.urdf_tools.cases import (
    UrdfAlignmentCase,
)
from projects.holobrain_internal.common.urdf_tools.config_wiring import (
    verify_case_config_wiring,
)
from test_robo_orchard_lab.projects.holobrain_internal.common.urdf_tools.conftest import (  # noqa: E501
    alignment_cases,
    repo_root,
    require_resolved,
)


@pytest.mark.parametrize(
    "case",
    alignment_cases(),
    ids=lambda case: case.name,
)
def test_every_alignment_case_wires_into_its_packer_config(
    case: UrdfAlignmentCase,
):
    """Each case's packer config points at its aligned URDF."""

    require_resolved(case)
    report = verify_case_config_wiring(case, repo_root=repo_root())
    assert report.ok, (
        f"config wiring failed for {case.name}:\n{report.format()}"
    )
