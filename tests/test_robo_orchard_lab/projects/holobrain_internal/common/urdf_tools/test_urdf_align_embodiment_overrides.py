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

"""Embodiment-specific invariants not covered by the generic contract.

The generic per-case contract in ``test_urdf_align_contract.py`` covers
motion/camera consistency, axis normalization, byte-idempotency, and
``gripper_end`` FK. Two embodiments carry bespoke invariants that are
not expressible generically and would silently regress without a
dedicated test:

- **Behavior R1 Pro** — the last arm joint rotates about X (stage-1
  compensated to Z) and the gripper extends along link7 +X, so the
  default ``xyz=(0, 0, gripper_forward)`` origin doesn't apply. The
  manifest overrides the attach link to the real ``*_gripper_link``
  with a hand-tuned look-ahead grasp point. This test pins the
  load-bearing parts of that override (documented in the
  ``r1-gripper-end-geometry`` memory).
- **G1 (interna1)** — camera mounts must sit *between* the last
  actuated wrist joint and the gripper base, otherwise the ee joint
  ends up owning both children and the camera transform is lost.
"""

from __future__ import annotations
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from conftest import alignment_case, repo_root, require_resolved

from projects.holobrain_internal.common.urdf_tools.cases import (
    UrdfAlignmentCase,
)

# ---------------------------------------------------------------------------
# Behavior R1 Pro — gripper_end override under the real gripper link.
# ---------------------------------------------------------------------------


def _load_obj_vertices(path: Path) -> np.ndarray:
    verts = []
    with open(path) as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                verts.append(
                    [float(parts[1]), float(parts[2]), float(parts[3])]
                )
    return np.array(verts, dtype=np.float64)


def _r1_pro_case() -> UrdfAlignmentCase:
    return alignment_case(adapter="behavior", config_key="behavior")


def test_behavior_r1_pro_gripper_end_override_geometry():
    """R1 Pro attaches gripper_end under ``*_gripper_link`` on the +Z ax.

    Contract pins (see ``r1-gripper-end-geometry`` memory):

    1. Override present and attached under the real ``*_gripper_link``.
    2. Override's Rx(180 deg) is preserved so gripper_end +Z equals the
       gripper approach direction. The in-plane yaw must not disturb +Z
       since runtime records the full quaternion.
    3. gripper_end xy stays on the gripper approach axis — verified
       against the finger-mesh tip centroid, which lies on that axis.
    4. Emitted URDF joint origin equals the manifest origin exactly.

    The absolute ``xyz`` z (-0.08) is NOT asserted against the mesh —
    it is an intentional grasp look-ahead distance past the finger tip.
    """

    case = _r1_pro_case()
    require_resolved(case)
    root = ElementTree.parse(case.aligned_urdf).getroot()
    align_dir = case.aligned_urdf.parent
    from projects.holobrain_internal.common.urdf_tools.transform._linalg import (  # noqa: E501
        OriginTransform,
    )

    def joint_origin_matrix(parent: str, child: str) -> np.ndarray:
        for joint in root.iter("joint"):
            p = joint.find("parent")
            c = joint.find("child")
            if (
                p is not None
                and c is not None
                and p.get("link") == parent
                and c.get("link") == child
            ):
                origin = joint.find("origin")
                rpy = (
                    [float(v) for v in origin.get("rpy", "0 0 0").split()]
                    if origin is not None
                    else [0.0, 0.0, 0.0]
                )
                xyz = (
                    [float(v) for v in origin.get("xyz", "0 0 0").split()]
                    if origin is not None
                    else [0.0, 0.0, 0.0]
                )
                return OriginTransform.from_rpy_xyz(rpy, xyz).as_matrix()
        raise KeyError(f"No joint {parent} -> {child} in {case.name}")

    neg_z = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    xy_tol = 2e-3  # 2 mm — finger-tip slab centroid xy sits on axis.

    for side, gripper_end in zip(
        ("left", "right"),
        case.alignment.gripper_ends,
        strict=True,
    ):
        # (1) Override present, attached to the real gripper link.
        assert gripper_end.has_override, (
            f"{case.name}: {side} arm must declare a gripper_end override"
        )
        gripper_link = f"{side}_gripper_link"
        assert gripper_end.attach_link == gripper_link

        origin_matrix = OriginTransform.from_rpy_xyz(
            gripper_end.rpy, gripper_end.xyz
        ).as_matrix()

        # (2) Rx(pi) preserved: gripper_end +Z == approach direction.
        approach_axis = origin_matrix[:3, :3] @ np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(
            approach_axis,
            neg_z,
            atol=1e-9,
            err_msg=f"{side} approach axis",
        )

        # (3) xy stays on the finger tip axis; only z extends past it.
        finger_verts = []
        for finger in (1, 2):
            mesh_path = (
                align_dir
                / "meshes"
                / f"{side}_gripper_finger_link{finger}.obj"
            )
            assert mesh_path.exists(), f"missing finger mesh {mesh_path}"
            joint = joint_origin_matrix(
                gripper_link, f"{side}_gripper_finger_link{finger}"
            )
            verts = _load_obj_vertices(mesh_path)
            finger_verts.append(
                (joint[:3, :3] @ verts.T).T + joint[:3, 3][None, :]
            )
        all_verts = np.vstack(finger_verts)
        z_min = all_verts[:, 2].min()
        slab = all_verts[all_verts[:, 2] < z_min + 1e-3]
        finger_tip_xy = slab.mean(axis=0)[:2]

        gripper_end_pos = origin_matrix[:3, 3]
        np.testing.assert_allclose(
            gripper_end_pos[:2],
            finger_tip_xy,
            atol=xy_tol,
            err_msg=f"{side} gripper_end xy off the approach axis",
        )
        assert gripper_end_pos[2] < z_min - 1e-3, (
            f"{side} gripper_end z {gripper_end_pos[2]} should be a "
            f"look-ahead point past the finger tip (z_min={z_min})"
        )

        # (4) URDF joint origin equals the manifest origin exactly.
        emitted = joint_origin_matrix(gripper_link, gripper_end.child)
        np.testing.assert_allclose(
            emitted,
            origin_matrix,
            atol=1e-9,
            rtol=0.0,
            err_msg=f"{side} emitted joint origin != manifest origin",
        )


# ---------------------------------------------------------------------------
# G1 (interna1) — camera mount must sit before the gripper base.
# ---------------------------------------------------------------------------


def _child_link(joint: ElementTree.Element) -> str:
    child = joint.find("child")
    assert child is not None
    return child.attrib["link"]


def test_g1_aligned_urdf_keeps_camera_mounts_before_gripper_base():
    """G1 wrist joint's child is the camera mount, not the gripper base."""

    aligned_urdf = repo_root() / (
        "projects/holobrain_internal/common/urdf_align/interna1/"
        "g1_120s/g1_120s.urdf"
    )
    if not aligned_urdf.exists():
        import pytest

        pytest.skip(f"aligned URDF not on disk: {aligned_urdf}")
    root = ElementTree.parse(aligned_urdf).getroot()
    joints_by_name = {
        joint.attrib["name"]: joint for joint in root.iter("joint")
    }

    assert _child_link(joints_by_name["idx61_ee_l_joint"]) == (
        "arm_l_camera_mount"
    )
    assert _child_link(joints_by_name["idx91_ee_r_joint"]) == (
        "arm_r_camera_mount"
    )
    assert "idx61_ee_l_camera_mount_joint" not in joints_by_name
    assert "idx91_ee_r_camera_mount_joint" not in joints_by_name
