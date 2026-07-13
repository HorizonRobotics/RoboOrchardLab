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
import sys
from pathlib import Path

import pytest

# The urdf_tools tests import product code with
# ``from projects.holobrain_internal.…``. ``projects/`` is a plain
# directory at the repo root — it isn't installed as a package (the
# top-level ``pyproject.toml`` only ships ``robo_orchard_lab``) — so the
# import only resolves when the repo root is on ``sys.path``. Under the
# ``make test_ut`` target pytest is launched from ``build/test``, so
# neither cwd nor the pytest rootdir put the repo root on ``sys.path``
# by default; prepend it explicitly here.
#
# The sibling ``__init__.py`` files under ``tests/…/projects/…`` exist
# for the other half of the puzzle: without them, pytest's rootdir walk
# would register ``tests/…/projects/`` as an implicit namespace package
# named ``projects`` at collection time, and that cached module would
# shadow the real package regardless of ``sys.path``, producing
# ``ModuleNotFoundError: No module named 'projects.holobrain_internal'``
# at conftest-import time (which is what CI previously hit). With inits
# in place, this test tree is collected under
# ``test_robo_orchard_lab.projects.…`` and does not compete with the
# real top-level ``projects``.
_REPO_ROOT = Path(__file__).resolve().parents[6]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from projects.holobrain_internal.common.urdf_tools.cases import (  # noqa: E402
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
