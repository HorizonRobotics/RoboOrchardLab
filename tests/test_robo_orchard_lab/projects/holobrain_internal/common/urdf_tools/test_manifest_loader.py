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

"""``alignment.yaml`` → ``UrdfAlignmentCase`` loader contract.

The pipeline's data model is populated from per-embodiment yaml
manifests. This file pins the load-bearing shape: a single
``alignment.yaml`` under a tmp manifest root resolves into one case with
the schema fields the rest of the pipeline consumes (config wiring,
motion links, joint samples, gripper_forward default). If the loader
ever silently drops a field the pipeline needs, this test fails before
the runtime code sees a half-populated case.
"""

from __future__ import annotations
from pathlib import Path

import pytest

from projects.holobrain_internal.common.urdf_tools.cases import (
    AlignmentSpec,
    ArmSpec,
    ConfigWiring,
    EeFrameSpec,
    UrdfAlignmentCase,
    load_alignment_manifest,
)


def _write_tiny_urdf(path: Path) -> None:
    path.write_text(
        """
<robot name="tiny_arm">
  <link name="base" />
  <link name="link1" />
  <joint name="joint1" type="revolute">
    <parent link="base" />
    <child link="link1" />
    <origin xyz="1 0 0" rpy="0 0 0" />
    <axis xyz="0 0 1" />
  </joint>
</robot>
""".strip(),
        encoding="utf-8",
    )


def test_manifest_loads_into_urdf_alignment_case_with_schema_fields(
    tmp_path: Path,
):
    """The yaml manifest fully populates ``UrdfAlignmentCase``.

    Origin URDF path is *not* declared in the manifest anymore: the
    loader reads it from the packer config module referenced by the
    case's ``config`` block, resolved relative to
    ``projects/holobrain_internal/common``. The aligned URDF is
    discovered next to the manifest. All derived accessors
    (``joints``, ``motion_links``, ``joint_samples``, default
    ``gripper_forward``) resolve without extra manifest fields.
    """

    common_root = tmp_path / "projects/holobrain_internal/common"
    origin_urdf = common_root / "urdf/toy/toy.urdf"
    align_root = common_root / "urdf_align/toy_ds/toy_emb"
    aligned_urdf = align_root / "toy.urdf"
    origin_urdf.parent.mkdir(parents=True)
    align_root.mkdir(parents=True)
    _write_tiny_urdf(origin_urdf)
    aligned_urdf.write_text(
        """
<robot name="tiny_arm">
  <link name="base" />
  <link name="link1" />
  <joint name="joint1" type="revolute">
    <parent link="base" />
    <child link="link1" />
    <origin xyz="1 0 0" rpy="0 0 0" />
    <axis xyz="0 0 1" />
  </joint>
  <link name="link1_gripper_end" />
  <joint name="link1_gripper_end_joint" type="fixed">
    <parent link="link1" />
    <child link="link1_gripper_end" />
    <origin xyz="0 0 0.2" rpy="0 0 0" />
  </joint>
</robot>
""".strip(),
        encoding="utf-8",
    )
    config_root = tmp_path / "configs"
    config_root.mkdir()
    (config_root / "config_toy_dataset.py").write_text(
        "def get_config():\n"
        '    return {"toy": {"urdf": "./urdf/toy/toy.urdf"}}\n',
        encoding="utf-8",
    )
    manifest_path = align_root / "alignment.yaml"
    manifest_path.write_text(
        """
adapter: interna1
config:
  module: config_toy_dataset
  getter: get_config
  key: toy
  aligned_key: toy_urdf_v2
arms:
  - arm_link_keys:
      - base
      - link1
    ee:
      parent: link1
      rotate_z_deg: 90
camera_references:
  - link1
""".strip(),
        encoding="utf-8",
    )

    cases = load_alignment_manifest(
        manifest_path,
        tmp_path,
        config_root=config_root,
    )

    assert len(cases) == 1
    only = cases[0]
    assert only == UrdfAlignmentCase(
        name="toy",
        adapter="interna1",
        origin_urdf=origin_urdf,
        aligned_urdf=aligned_urdf,
        config=ConfigWiring(
            module="config_toy_dataset",
            getter="get_config",
            keys=("toy",),
            aligned_keys=("toy_urdf_v2",),
        ),
        alignment=AlignmentSpec(
            arms=(
                ArmSpec(
                    arm_link_keys=("base", "link1"),
                    ee=EeFrameSpec(
                        parent="link1",
                        rotate_z_deg=90,
                    ),
                ),
            ),
            camera_references=("link1",),
        ),
        manifest_path=manifest_path,
    )
    # Derived accessors the rest of the pipeline reads through.
    assert only.config_module == "config_toy_dataset"
    assert only.aligned_config_key == "toy_urdf_v2"
    assert only.alignment.camera_references == ("link1",)
    # Default gripper_forward is the canonical 0.20 m tip offset unless
    # a manifest overrides it.
    assert only.alignment.arms[0].ee.gripper_forward == pytest.approx(0.20)
    # `joints` comes from the origin URDF's actuated joint list; the
    # pipeline never renames actuated joints.
    assert only.joints == ("joint1",)
    assert {link.origin_name for link in only.motion_links} == {
        "base",
        "link1",
    }
    assert only.joint_samples[0] == {}
    assert only.joint_samples[1]["joint1"] == pytest.approx(0.08)
