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

"""Per-case contract every aligned URDF must satisfy.

The alignment pipeline runs per embodiment; this file parametrizes the
six invariants that must hold for *every* case so that adding an
embodiment is a manifest change, not a bespoke test. The invariants are:

- **motion consistency** — same joint values place motion links at the
  same world pose in the origin and aligned URDFs.
- **camera consistency** — camera mount frames map cleanly (position +
  orientation) from origin to aligned via the compat link if any.
- **actuated joint axes are normalized** — every actuated joint's axis
  is ``+Z`` so control vectors are portable.
- **camera compat links only added when needed** — no spurious compat
  link when the aligned reference already matches the origin frame.
- **``apply_alignment`` is byte-idempotent** — running the pipeline
  again yields the same bytes; running it over an already-aligned URDF
  is a no-op (guards the per-stage idempotency checks).
- **``gripper_end`` child per arm** — the fixed tip link exists,
  attaches to the correct parent, and its FK equals ``attach @
  Origin(rpy, xyz)`` from the manifest override.

The wrist-axis extension check is scoped to embodiments where the
manifest doesn't explicitly rotate the EE frame — see
``wrist_axis_alignment_cases`` in ``conftest.py``.
"""

from __future__ import annotations
from pathlib import Path
from xml.etree import ElementTree

import pytest
import torch
from conftest import (
    alignment_cases,
    require_resolved,
    wrist_axis_alignment_cases,
)

from projects.holobrain_internal.common.urdf_tools.cases import (
    UrdfAlignmentCase,
)
from projects.holobrain_internal.common.urdf_tools.fk_baseline import (
    CameraMountConsistencySpec,
    LinkNamePair,
    NamedFkConsistencySpec,
    assert_camera_mount_consistent,
    assert_named_motion_consistent,
)
from projects.holobrain_internal.common.urdf_tools.transform import (
    _format_aligned_urdf,
    apply_alignment,
)

# ---------------------------------------------------------------------------
# Helpers used only by this file.
# ---------------------------------------------------------------------------


def _normalized_axis(axis: ElementTree.Element) -> str:
    values = [float(value) for value in axis.attrib.get("xyz", "").split()]
    return " ".join(f"{value:g}" for value in values)


def _nearest_actuated_child_link(
    root: ElementTree.Element,
    link: str,
) -> str:
    joints_by_child = {
        joint.find("child").attrib["link"]: joint
        for joint in root.iter("joint")
        if joint.find("child") is not None
    }
    current = link
    while current in joints_by_child:
        joint = joints_by_child[current]
        if joint.attrib.get("type") in {
            "revolute",
            "continuous",
            "prismatic",
        }:
            return current
        parent = joint.find("parent")
        assert parent is not None
        current = parent.attrib["link"]
    raise AssertionError(f"No actuated ancestor joint found for link {link!r}")


def _link_full_pose_matches(
    *,
    case: UrdfAlignmentCase,
    origin_link: str,
    aligned_link: str,
) -> bool:
    try:
        assert_camera_mount_consistent(
            CameraMountConsistencySpec(
                name=f"{case.name}:{origin_link}->{aligned_link}",
                origin_urdf=case.origin_urdf,
                aligned_urdf=case.aligned_urdf,
                joint_names=case.joints,
                mounts=(
                    LinkNamePair(
                        semantic_name="camera_reference",
                        origin_name=origin_link,
                        aligned_name=aligned_link,
                    ),
                ),
                joint_samples=case.joint_samples,
                atol=2e-5,
            )
        )
    except (AssertionError, KeyError):
        return False
    return True


# ---------------------------------------------------------------------------
# Per-case invariants.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    alignment_cases(),
    ids=lambda case: case.name,
)
def test_aligned_urdf_preserves_motion_positions(case: UrdfAlignmentCase):
    """Axis normalization keeps motion-link world positions unchanged."""

    require_resolved(case)
    assert_named_motion_consistent(
        NamedFkConsistencySpec(
            name=case.name,
            origin_urdf=case.origin_urdf,
            aligned_urdf=case.aligned_urdf,
            joint_names=case.joints,
            links=case.motion_links,
            joint_samples=case.joint_samples,
        )
    )


@pytest.mark.parametrize(
    "case",
    alignment_cases(),
    ids=lambda case: case.name,
)
def test_aligned_camera_reference_compat_links_match_origin_frames(
    case: UrdfAlignmentCase,
):
    """Camera compat links preserve the origin reference frame's pose."""

    require_resolved(case)
    assert_camera_mount_consistent(
        CameraMountConsistencySpec(
            name=case.name,
            origin_urdf=case.origin_urdf,
            aligned_urdf=case.aligned_urdf,
            joint_names=case.joints,
            mounts=case.camera_mount_links,
            joint_samples=case.joint_samples,
            atol=2e-5,
        )
    )


@pytest.mark.parametrize(
    "case",
    alignment_cases(),
    ids=lambda case: case.name,
)
def test_aligned_urdf_actuated_joint_axes_are_normalized(
    case: UrdfAlignmentCase,
):
    """Every actuated joint's axis is ``+Z`` in the aligned URDF."""

    require_resolved(case)
    root = ElementTree.parse(case.aligned_urdf).getroot()
    invalid_axes = []
    for joint in root.iter("joint"):
        joint_type = joint.attrib.get("type")
        if joint_type not in {"revolute", "continuous", "prismatic"}:
            continue
        axis = joint.find("axis")
        axis_xyz = "0 0 1" if axis is None else _normalized_axis(axis)
        if axis_xyz != "0 0 1":
            invalid_axes.append((joint.attrib.get("name"), axis_xyz))

    assert invalid_axes == []


@pytest.mark.parametrize(
    "case",
    alignment_cases(),
    ids=lambda case: case.name,
)
def test_camera_compat_link_added_only_when_reference_frame_changes(
    case: UrdfAlignmentCase,
):
    """No compat link when the aligned mount already matches origin."""

    require_resolved(case)
    unchanged_references = []
    for mount in case.camera_mount_links:
        if _link_full_pose_matches(
            case=case,
            origin_link=mount.origin_name,
            aligned_link=mount.origin_name,
        ):
            unchanged_references.append(mount.semantic_name)

    assert unchanged_references == []


@pytest.mark.parametrize(
    "case",
    alignment_cases(),
    ids=lambda case: case.name,
)
def test_apply_alignment_output_is_byte_idempotent(
    case: UrdfAlignmentCase,
    tmp_path: Path,
):
    """Running the pipeline twice yields identical bytes.

    Two flavors of idempotency here:

    - Rerunning the whole pipeline against the same origin URDF returns
      the same bytes.
    - Parsing an already-formatted aligned URDF and running only the
      final format pass yields the same bytes — the canonical layout is
      a fixed point of the serializer.
    """

    require_resolved(case)
    first = apply_alignment(case.origin_urdf, case)
    second = apply_alignment(case.origin_urdf, case)
    assert second.aligned_urdf_bytes == first.aligned_urdf_bytes

    round_trip_path = tmp_path / case.aligned_urdf.name
    round_trip_path.write_bytes(first.aligned_urdf_bytes)
    reformatted = _format_aligned_urdf(
        ElementTree.parse(round_trip_path).getroot()
    )
    assert reformatted == first.aligned_urdf_bytes


@pytest.mark.parametrize(
    "case",
    alignment_cases(),
    ids=lambda case: case.name,
)
def test_apply_alignment_on_aligned_urdf_is_a_no_op(
    case: UrdfAlignmentCase,
    tmp_path: Path,
):
    """Re-running the pipeline on an aligned URDF changes nothing.

    Exercises the per-stage detection guards in
    ``insert_ee_children`` and ``insert_gripper_end_children`` — without
    them, a re-run would move geometry twice and duplicate fixed joints.
    """

    require_resolved(case)
    first = apply_alignment(case.origin_urdf, case)

    aligned_input = tmp_path / f"aligned-{case.aligned_urdf.name}"
    aligned_input.write_bytes(first.aligned_urdf_bytes)
    second = apply_alignment(aligned_input, case)

    assert second.aligned_urdf_bytes == first.aligned_urdf_bytes
    assert second.report.inserted_ee_joints == ()
    assert second.report.inserted_gripper_end_joints == ()
    assert second.report.inserted_camera_compat_links == ()


@pytest.mark.parametrize(
    "case",
    alignment_cases(),
    ids=lambda case: case.name,
)
def test_aligned_urdf_exposes_gripper_end_child_per_arm(
    case: UrdfAlignmentCase,
):
    """Every arm has a fixed ``*_gripper_end`` child at the manifest origin.

    Pins:
      1. the child link is in the URDF,
      2. the fixed joint attaches to the declared parent,
      3. FK in zero-config places the tip at
         ``attach @ Origin(rpy, xyz)`` — the contract for both the
         default (``xyz=(0, 0, gripper_forward)``, identity rpy on the
         EE frame) and override paths (e.g. R1 Pro).
    """

    require_resolved(case)
    import pytorch_kinematics as pk

    from projects.holobrain_internal.common.urdf_tools.transform._linalg import (  # noqa: E501
        OriginTransform,
    )

    root = ElementTree.parse(case.aligned_urdf).getroot()
    joints_by_child = {
        joint.find("child").attrib["link"]: joint
        for joint in root.iter("joint")
        if joint.find("child") is not None
    }
    chain = pk.build_chain_from_urdf(case.aligned_urdf.read_bytes())
    zero_positions = torch.zeros(
        (1, len(chain.get_joint_parameter_names())),
        dtype=torch.float32,
    )
    link_poses = chain.forward_kinematics(zero_positions)

    for gripper_end in case.alignment.gripper_ends:
        assert gripper_end.child in joints_by_child, (
            f"{case.name}: gripper_end child link "
            f"{gripper_end.child!r} not in aligned URDF"
        )
        joint = joints_by_child[gripper_end.child]
        assert joint.attrib.get("type") == "fixed"
        assert joint.find("parent").attrib["link"] == gripper_end.attach_link

        assert gripper_end.child in link_poses
        assert gripper_end.attach_link in link_poses
        attach = link_poses[gripper_end.attach_link].get_matrix()[0]
        tip = link_poses[gripper_end.child].get_matrix()[0]
        origin_matrix = torch.tensor(
            OriginTransform.from_rpy_xyz(
                gripper_end.rpy, gripper_end.xyz
            ).as_matrix(),
            dtype=attach.dtype,
        )
        expected_tip = attach @ origin_matrix
        assert torch.allclose(tip, expected_tip, atol=1e-5, rtol=0.0), (
            f"{case.name}: {gripper_end.child} FK pose does not equal "
            f"attach @ Origin(rpy={gripper_end.rpy}, xyz={gripper_end.xyz})"
        )


@pytest.mark.parametrize(
    "case",
    wrist_axis_alignment_cases(),
    ids=lambda case: case.name,
)
def test_gripper_end_extends_last_wrist_axis(case: UrdfAlignmentCase):
    """For non-rotated EE cases, gripper_end is along +Z of last wrist."""

    require_resolved(case)
    import pytorch_kinematics as pk

    root = ElementTree.parse(case.aligned_urdf).getroot()
    chain = pk.build_chain_from_urdf(case.aligned_urdf.read_bytes())
    zero_positions = torch.zeros(
        (1, len(chain.get_joint_parameter_names())),
        dtype=torch.float32,
    )
    link_poses = chain.forward_kinematics(zero_positions)
    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

    for arm, gripper_end in zip(
        case.alignment.arms,
        case.alignment.gripper_ends,
        strict=True,
    ):
        last_actuated_link = _nearest_actuated_child_link(
            root,
            arm.ee.parent,
        )
        assert last_actuated_link in link_poses
        assert gripper_end.child in link_poses

        wrist_pose = link_poses[last_actuated_link].get_matrix()[0]
        tip_pose = link_poses[gripper_end.child].get_matrix()[0]
        wrist_to_tip_world = tip_pose[:3, 3] - wrist_pose[:3, 3]
        wrist_to_tip_local = wrist_pose[:3, :3].T @ wrist_to_tip_world
        direction = wrist_to_tip_local / torch.linalg.vector_norm(
            wrist_to_tip_local
        )

        assert torch.dot(direction, z_axis) > 0.99, (
            f"{case.name}: {gripper_end.child} is not forward along "
            f"{last_actuated_link}'s +Z wrist axis; local direction is "
            f"{direction.tolist()}"
        )
