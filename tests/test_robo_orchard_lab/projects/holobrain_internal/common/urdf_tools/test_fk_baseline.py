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

"""FK baseline determinism + asset-tree copy contract.

``compute_fk_baseline`` produces a JSON that every downstream consistency
check (motion, camera) diffs against. That JSON must be deterministic —
same URDF + same joint positions → same bytes — otherwise the checks
report spurious drift. ``copy_urdf_asset_tree`` is the sibling that
mirrors relative meshes alongside a copied URDF; if it drops a mesh, the
aligned URDF renders empty.
"""

from __future__ import annotations
import json
from pathlib import Path

from projects.holobrain_internal.common.urdf_tools.fk_baseline import (
    FkBaselineSpec,
    compute_fk_baseline,
)
from projects.holobrain_internal.common.urdf_tools.sync_assets import (
    copy_urdf_asset_tree,
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


def test_compute_fk_baseline_writes_deterministic_link_pose_json(
    tmp_path: Path,
):
    """The emitted baseline JSON matches the returned dict byte-for-byte."""

    urdf_path = tmp_path / "tiny.urdf"
    _write_tiny_urdf(urdf_path)
    output_path = tmp_path / "baseline.json"

    baseline = compute_fk_baseline(
        FkBaselineSpec(
            name="tiny",
            urdf=urdf_path,
            link_keys=["link1"],
            joint_positions=[[0.0], [1.5707963267948966]],
            output_path=output_path,
        )
    )

    assert baseline["name"] == "tiny"
    assert baseline["link_keys"] == ["link1"]
    assert baseline["samples"][0]["links"]["link1"]["position"] == [
        1.0,
        0.0,
        0.0,
    ]
    sample_quaternion = baseline["samples"][1]["links"]["link1"][
        "quaternion_wxyz"
    ]
    assert sample_quaternion[0] == 0.707107
    assert json.loads(output_path.read_text(encoding="utf-8")) == baseline


def test_copy_urdf_asset_tree_copies_relative_meshes(tmp_path: Path):
    """Copying a URDF also copies its relative mesh + material files."""

    source_dir = tmp_path / "legacy" / "robot"
    mesh_dir = source_dir / "meshes"
    mesh_dir.mkdir(parents=True)
    source_urdf = source_dir / "robot.urdf"
    mesh_file = mesh_dir / "part.obj"
    material_file = mesh_dir / "part.mtl"
    mesh_file.write_text("mesh", encoding="utf-8")
    material_file.write_text("material", encoding="utf-8")
    source_urdf.write_text(
        """
<robot name="robot">
  <link name="base">
    <visual><geometry><mesh filename="meshes/part.obj" /></geometry></visual>
  </link>
</robot>
""".strip(),
        encoding="utf-8",
    )
    target_urdf = tmp_path / "urdf_align" / "dataset" / "emb" / "robot.urdf"

    copied = copy_urdf_asset_tree(source_urdf, target_urdf)

    assert copied.urdf == target_urdf
    assert target_urdf.exists()
    assert copied.copied_files == (
        target_urdf,
        target_urdf.parent / "meshes" / "part.obj",
        target_urdf.parent / "meshes" / "part.mtl",
    )
    assert (target_urdf.parent / "meshes" / "part.obj").read_text(
        encoding="utf-8"
    ) == "mesh"
    assert (target_urdf.parent / "meshes" / "part.mtl").read_text(
        encoding="utf-8"
    ) == "material"
