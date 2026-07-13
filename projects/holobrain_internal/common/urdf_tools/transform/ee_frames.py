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

"""Insert semantic `*_ee` fixed children for right/down/front EE frames.

For each entry declared in the case's ``alignment.ee_frames`` block:

* If ``rotate_z_deg == 0``, no ``*_ee`` child is inserted; the parent link is
  the semantic EE frame directly.
* Otherwise a new ``<link name="<child>">`` is created and a fixed joint named
  ``<parent>_ee_joint`` (unique per case) rotates around the parent's Z axis
  by ``rotate_z_deg``.
* The ``<inertial>``, ``<visual>``, and ``<collision>`` children of the parent
  link move onto the new EE child. Because the EE link frame is the parent
  frame post-multiplied by ``S = Rz(rotate_z_deg)``, each moved element's
  ``<origin>`` is re-expressed by ``S⁻¹`` (left-multiplied) so the mesh /
  collision / inertial world-frame pose is invariant. Inertia tensor
  components are left as-is: the inertial ``<origin>`` rotation absorbs the
  frame change, so the tensor stays expressed in the same physical frame
  (mirrors ``ALIGNMENT_NOTES.md § Axis Compensation Rule`` steps 3 and 5).

Subsequent gripper joints continue to attach to the origin parent link (they
already carry the calibration/mechanical origin), so the transformer does
not rewire them here.
"""

from __future__ import annotations
import xml.etree.ElementTree as ET
from typing import Sequence

import numpy as np

from projects.holobrain_internal.common.urdf_tools.cases import EeFrameSpec
from projects.holobrain_internal.common.urdf_tools.transform._linalg import (
    OriginTransform,
    format_vec,
    rotation_z,
)


def insert_ee_children(
    root: ET.Element,
    ee_frames: Sequence[EeFrameSpec],
) -> list[str]:
    """Insert semantic EE fixed children as declared in `ee_frames`.

    Args:
        root: ``<robot>`` element modified in place.
        ee_frames: Manifest-derived EE frame specs.

    Returns:
        list[str]: Names of newly inserted ``<joint>`` elements, in order.
    """

    links_by_name = {
        link.get("name", ""): link for link in root.findall("link")
    }
    joints_by_name = {
        joint.get("name", ""): joint for joint in root.findall("joint")
    }
    inserted_joints: list[str] = []
    for spec in ee_frames:
        if spec.rotate_z_deg == 0:
            continue
        parent_link = links_by_name.get(spec.parent)
        if parent_link is None:
            raise ValueError(
                "ee_frames entry references missing parent link "
                f"'{spec.parent}'"
            )
        theta = float(np.deg2rad(spec.rotate_z_deg))
        s_matrix = rotation_z(theta)
        joint_name = _make_joint_name(spec.parent)
        if _ee_joint_already_applied(
            joints_by_name.get(joint_name),
            expected_parent=spec.parent,
            expected_child=spec.child,
            expected_s=s_matrix,
        ):
            # Idempotency guard: re-running the pipeline on an already-aligned
            # URDF finds the *_ee child + rotate joint already in place, so
            # skip this arm to avoid duplicating the geometry move.
            continue
        ee_link = _find_or_create_link(root, links_by_name, spec.child)
        _move_link_geometry(parent_link, ee_link, s_inv=s_matrix.T)
        _insert_ee_joint(
            root,
            joint_name=joint_name,
            parent=spec.parent,
            child=spec.child,
            s_matrix=s_matrix,
        )
        inserted_joints.append(joint_name)
    return inserted_joints


# ---------------------------------------------------------------------------
# Building blocks.
# ---------------------------------------------------------------------------


def _find_or_create_link(
    root: ET.Element,
    links_by_name: dict[str, ET.Element],
    link_name: str,
) -> ET.Element:
    """Return the URDF ``<link>`` named `link_name`, creating it if missing."""

    if link_name in links_by_name:
        return links_by_name[link_name]
    element = ET.SubElement(root, "link")
    element.set("name", link_name)
    links_by_name[link_name] = element
    return element


def _move_link_geometry(
    parent_link: ET.Element,
    ee_link: ET.Element,
    s_inv: np.ndarray,
) -> None:
    """Move `<inertial>`, `<visual>`, `<collision>` from parent to ee link.

    The moved element's ``<origin>`` is re-expressed by ``S⁻¹``
    (left-multiplied) so its world-frame pose is invariant across the
    frame change from the parent link into the rotated ``*_ee`` child
    link. Inertia tensor
    components stay untouched (the inertial ``<origin>`` rotation absorbs the
    frame change).
    """

    for tag in ("inertial", "visual", "collision"):
        for element in list(parent_link.findall(tag)):
            parent_link.remove(element)
            _rewrite_element_origin(element, s_inv)
            ee_link.append(element)


def _rewrite_element_origin(element: ET.Element, s_inv: np.ndarray) -> None:
    """Left-multiply the `<origin>` of `element` by ``S⁻¹`` in place."""

    origin = element.find("origin")
    if origin is None:
        origin = ET.SubElement(element, "origin")
        origin.set("xyz", "0 0 0")
        origin.set("rpy", "0 0 0")
    raw_rpy = origin.get("rpy")
    raw_xyz = origin.get("xyz")
    rpy = [float(v) for v in raw_rpy.split()] if raw_rpy else None
    xyz = [float(v) for v in raw_xyz.split()] if raw_xyz else None
    transform = OriginTransform.from_rpy_xyz(rpy, xyz)
    new_rot = s_inv @ transform.rot
    new_xyz = s_inv @ transform.xyz
    updated = OriginTransform(xyz=new_xyz, rot=new_rot)
    new_rpy, new_xyz_arr = updated.to_rpy_xyz()
    origin.set("rpy", format_vec(new_rpy))
    origin.set("xyz", format_vec(new_xyz_arr))


def _insert_ee_joint(
    root: ET.Element,
    *,
    joint_name: str,
    parent: str,
    child: str,
    s_matrix: np.ndarray,
) -> None:
    joint = ET.SubElement(root, "joint")
    joint.set("name", joint_name)
    joint.set("type", "fixed")
    origin = ET.SubElement(joint, "origin")
    transform = OriginTransform(xyz=np.zeros(3), rot=s_matrix)
    rpy, xyz = transform.to_rpy_xyz()
    origin.set("rpy", format_vec(rpy))
    origin.set("xyz", format_vec(xyz))
    parent_el = ET.SubElement(joint, "parent")
    parent_el.set("link", parent)
    child_el = ET.SubElement(joint, "child")
    child_el.set("link", child)


def _make_joint_name(parent_link: str) -> str:
    """Derive a stable joint name from the parent link name.

    ``"left/link6"`` becomes ``"left/joint6_ee"``; a link that does not follow
    the ``"prefix/linkN"`` pattern falls back to ``"<parent>_ee_joint"``.
    """

    if "/link" in parent_link:
        head, _, tail = parent_link.partition("/link")
        return f"{head}/joint{tail}_ee"
    return f"{parent_link}_ee_joint"


def _ee_joint_already_applied(
    joint: ET.Element | None,
    *,
    expected_parent: str,
    expected_child: str,
    expected_s: np.ndarray,
) -> bool:
    """Return True when `joint` matches the EE fixed joint we would emit.

    Used as an idempotency guard: on a re-run over an already-aligned URDF,
    the ``<parent>_ee_joint`` fixed joint plus its ``<parent>_ee`` child link
    are already present. We recognize that state by matching the joint's
    ``parent``/``child`` links and its ``<origin>`` rotation against the
    expected :math:`S = R_z(\\theta)` (within a numerical tolerance). Only then
    do we skip re-emitting the joint and moving the parent's geometry a second
    time.
    """

    if joint is None or joint.get("type") != "fixed":
        return False
    parent_el = joint.find("parent")
    child_el = joint.find("child")
    if parent_el is None or child_el is None:
        return False
    if parent_el.get("link") != expected_parent:
        return False
    if child_el.get("link") != expected_child:
        return False
    origin = joint.find("origin")
    if origin is None:
        return False
    raw_rpy = origin.get("rpy")
    raw_xyz = origin.get("xyz")
    rpy = [float(v) for v in raw_rpy.split()] if raw_rpy else None
    xyz = [float(v) for v in raw_xyz.split()] if raw_xyz else None
    transform = OriginTransform.from_rpy_xyz(rpy, xyz)
    if not np.allclose(transform.xyz, np.zeros(3), atol=1e-9):
        return False
    return bool(np.allclose(transform.rot, expected_s, atol=1e-9))
