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

"""Shared plumbing for the URDF alignment test package.

Every test file in this directory pulls its cases through the helpers
here so:

- there is exactly one place to change how manifests are discovered, and
- the "skip if the bucket isn't mounted" rule (:func:`require_resolved`)
  fires uniformly for parametrized *and* focused tests. The previous
  autouse fixture only inspected ``callspec.params['case']`` and never
  fired for tests that grabbed cases via :func:`alignment_case`, which
  is why non-parametrized embodiment tests raised ``FileNotFoundError``
  when the aligned URDF was missing.
"""

from __future__ import annotations
from pathlib import Path

import pytest

from projects.holobrain_internal.common.urdf_tools.cases import (
    UrdfAlignmentCase,
    default_alignment_cases,
)


def repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"Cannot find repo root from {current}")


def alignment_cases() -> list[UrdfAlignmentCase]:
    """Every alignment case discovered under the default manifest root."""

    return default_alignment_cases(repo_root())


def alignment_case(adapter: str, config_key: str) -> UrdfAlignmentCase:
    """Return the one case matching ``(adapter, config_key)`` or raise."""

    for case in alignment_cases():
        if case.adapter == adapter and case.config_key == config_key:
            return case
    raise AssertionError(
        f"No alignment case found for adapter={adapter!r}, "
        f"config_key={config_key!r}"
    )


def require_resolved(case: UrdfAlignmentCase) -> None:
    """Skip when the aligned or origin URDF for ``case`` is not on disk.

    Call as the first line of every test that touches ``case.aligned_urdf``
    or ``case.origin_urdf``. Emits a clean skip whether the test is
    parametrized over cases or grabs one focused case directly.
    """

    if not case.resolved:
        pytest.skip(
            f"aligned URDF not on disk for {case.name} at {case.aligned_urdf}"
        )
    if not case.origin_urdf.exists():
        pytest.skip(
            f"origin URDF not on disk for {case.name} at {case.origin_urdf}"
        )


def wrist_axis_alignment_cases() -> list[UrdfAlignmentCase]:
    """Cases whose gripper_end extends the last wrist joint's +Z axis.

    The set is explicit rather than derived: non-wrist-axis embodiments
    (Panda, R1 Pro, dual-arm G1) intentionally rotate the ee frame and
    the +Z-of-last-wrist claim doesn't hold there.
    """

    targets = {
        ("agilex", "horizon_piper_x_435"),
        ("robotwin", "ur5_wsg"),
        ("rh20t", "ur5"),
        ("rh20t", "ur5_v2"),
        ("rh20t", "kuka"),
        ("rh20t", "kuka_v2"),
    }
    return [
        case
        for case in alignment_cases()
        if (case.adapter, case.config_key) in targets
    ]
