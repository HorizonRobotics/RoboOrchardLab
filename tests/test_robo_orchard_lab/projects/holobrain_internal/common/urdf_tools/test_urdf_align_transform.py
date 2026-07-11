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

"""``insert_ee_children`` re-expresses moved geometry in the child frame.

When the manifest declares ``rotate_z_deg != 0``, ``insert_ee_children``
moves visual/collision/inertial elements from the parent link into a new
``*_ee`` child link that is rotated by that same amount. To keep the
world pose of every mesh unchanged, each moved ``<origin>`` must be
pre-multiplied by ``Rz(-rotate_z_deg)``. Without that compensation the
rendered mesh rotates with the joint — this is the exact symptom that
prompted the transform fix.
"""

from __future__ import annotations
from xml.etree import ElementTree

import numpy as np

from projects.holobrain_internal.common.urdf_tools.cases import EeFrameSpec
from projects.holobrain_internal.common.urdf_tools.transform.ee_frames import (  # noqa: E501
    insert_ee_children,
)


def test_insert_ee_children_preserves_moved_geometry_world_pose():
    """Origins are re-expressed by S⁻¹ so world pose stays fixed."""

    urdf_src = """
<robot name="rotate_ee_probe">
  <link name="wrist">
    <inertial>
      <origin xyz="0.1 0 0" rpy="0 0 0" />
      <mass value="1.0" />
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1" />
    </inertial>
    <visual>
      <origin xyz="0.2 0 0" rpy="0 0 0" />
      <geometry><box size="0.1 0.05 0.02" /></geometry>
    </visual>
    <collision>
      <origin xyz="0.2 0 0" rpy="0 0 0" />
      <geometry><box size="0.1 0.05 0.02" /></geometry>
    </collision>
  </link>
</robot>
""".strip()
    root = ElementTree.fromstring(urdf_src)
    insert_ee_children(
        root,
        (EeFrameSpec(parent="wrist", rotate_z_deg=180),),
    )

    links = {link.attrib["name"]: link for link in root.findall("link")}
    ee_link = links["wrist_ee"]
    # wrist_ee is Rz(180) of wrist, so a point at (+0.2, 0, 0) in wrist
    # must be expressed as (-0.2, 0, 0) in wrist_ee to remain at
    # (+0.2, 0, 0) in world after the joint applies its Rz(180).
    for tag in ("inertial", "visual", "collision"):
        element = ee_link.find(tag)
        assert element is not None
        origin = element.find("origin")
        assert origin is not None
        xyz = np.asarray(
            [float(v) for v in origin.attrib["xyz"].split()],
            dtype=float,
        )
        expected_offset = -0.2 if tag != "inertial" else -0.1
        assert np.allclose(xyz, np.array([expected_offset, 0.0, 0.0]))
