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

"""Insert a `<parent>_gripper_end` fixed child per arm.

For every arm declared in the manifest the alignment pipeline emits one
``<link name="{ee.parent}_gripper_end">`` and a fixed joint attaching it to
the arm's gripper-end attach link:

* When the arm's EE rotates (``rotate_z_deg != 0``) and no override is set,
  the attach link is the freshly-inserted ``<parent>_ee`` child from stage 2.
* When the arm's last link is already convention-compliant
  (``rotate_z_deg == 0``) and no override is set, the attach link is
  ``<parent>`` directly.
* When the arm declares a ``gripper_end`` override, the attach link is the
  override's ``attach_link`` (a real link, e.g. the gripper link), and the full
  ``xyz`` / ``rpy`` origin comes from the override.

The fixed joint's origin is ``xyz="0 0 {gripper_forward}"`` (default
``0.20`` m) with identity rotation on the default path, so the emitted link
sits on the +Z axis of the EE frame — the convention for robots whose last
arm joint rotates about the forward axis. The override path instead places the
link at a full origin relative to a real link, which is required for robots
whose gripper pokes perpendicular to the actuated axis (behavior R1 Pro).
Either way the child link name is uniform, so dataset configs get a stable
``finger_keys`` handle that lives at the semantic gripper-tip position of
every embodiment, and the runtime finger-mean stays a plain componentwise mean
over a single-row tensor.

The joint name follows the same convention as
:func:`transform.ee_frames._make_joint_name`: an ``"prefix/linkN"`` parent
turns into ``"prefix/jointN_gripper_end"``; otherwise the joint name is
``"<attach_link>_gripper_end_joint"``.
"""

from __future__ import annotations
import xml.etree.ElementTree as ET
from typing import Sequence

from projects.holobrain_internal.common.urdf_tools.cases import (
    GripperEndSpec,
)
from projects.holobrain_internal.common.urdf_tools.transform._linalg import (
    format_vec,
)


def insert_gripper_end_children(
    root: ET.Element,
    gripper_ends: Sequence[GripperEndSpec],
) -> list[str]:
    """Insert ``<parent>_gripper_end`` fixed children as declared.

    Args:
        root: ``<robot>`` element modified in place. When an arm's rotate
            stage 2 already inserted the ``<parent>_ee`` child, that link
            must exist on ``root`` before this stage runs.
        gripper_ends: Manifest-derived attach-link + emitted-child + origin
            per arm. Typically obtained via
            :pyattr:`AlignmentSpec.gripper_ends`.

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
    for spec in gripper_ends:
        if spec.attach_link not in links_by_name:
            raise ValueError(
                "gripper_end entry references missing attach link "
                f"'{spec.attach_link}' (for an override it must be a real "
                "link already present in the URDF; on the default path the "
                "*_ee child must have been inserted in stage 2)"
            )
        joint_name = _make_joint_name(spec.attach_link)
        if _gripper_end_joint_already_applied(
            joints_by_name.get(joint_name),
            expected_parent=spec.attach_link,
            expected_child=spec.child,
            expected_xyz=spec.xyz,
            expected_rpy=spec.rpy,
        ):
            # Idempotency guard: re-running the pipeline finds the
            # <parent>_gripper_end fixed joint already in place with the
            # expected origin, so do not re-insert it.
            continue
        _find_or_create_link(root, links_by_name, spec.child)
        _insert_gripper_end_joint(
            root,
            joint_name=joint_name,
            parent=spec.attach_link,
            child=spec.child,
            xyz=spec.xyz,
            rpy=spec.rpy,
        )
        inserted_joints.append(joint_name)
    return inserted_joints


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _find_or_create_link(
    root: ET.Element,
    links_by_name: dict[str, ET.Element],
    link_name: str,
) -> ET.Element:
    """Return the URDF ``<link>`` named ``link_name``, creating it if missing.

    Mirrors ``transform.ee_frames._find_or_create_link``. Duplicated instead of
    imported because the stage 2 helper is intentionally private and the two
    modules have no other reason to share code.
    """

    if link_name in links_by_name:
        return links_by_name[link_name]
    element = ET.SubElement(root, "link")
    element.set("name", link_name)
    links_by_name[link_name] = element
    return element


def _insert_gripper_end_joint(
    root: ET.Element,
    *,
    joint_name: str,
    parent: str,
    child: str,
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float],
) -> None:
    """Append a fixed joint at the given ``xyz`` / ``rpy`` origin.

    On the default path the caller passes ``xyz=(0, 0, forward)`` and
    ``rpy=(0, 0, 0)``. Origin bytes are rendered via :func:`format_vec` (the
    same serializer the axis / EE stages use) so emitted URDF bytes stay
    canonical.
    """

    joint = ET.SubElement(root, "joint")
    joint.set("name", joint_name)
    joint.set("type", "fixed")
    origin = ET.SubElement(joint, "origin")
    origin.set("rpy", format_vec(rpy))
    origin.set("xyz", format_vec(xyz))
    parent_el = ET.SubElement(joint, "parent")
    parent_el.set("link", parent)
    child_el = ET.SubElement(joint, "child")
    child_el.set("link", child)


def _make_joint_name(attach_link: str) -> str:
    """Derive a stable joint name from the attach link name.

    Mirrors the ``prefix/linkN`` convention used by
    :func:`transform.ee_frames._make_joint_name`:

    * ``"left/link6_ee"``  → ``"left/joint6_ee_gripper_end"``
    * ``"left/link6"``     → ``"left/joint6_gripper_end"``
    * ``"fl_link6_ee"``    → ``"fl_link6_ee_gripper_end_joint"`` (fallback)
    """

    if "/link" in attach_link:
        head, _, tail = attach_link.partition("/link")
        return f"{head}/joint{tail}_gripper_end"
    return f"{attach_link}_gripper_end_joint"


def _gripper_end_joint_already_applied(
    joint: ET.Element | None,
    *,
    expected_parent: str,
    expected_child: str,
    expected_xyz: tuple[float, float, float],
    expected_rpy: tuple[float, float, float],
) -> bool:
    """Return True when `joint` matches the fixed gripper-end joint we'd emit.

    Idempotency guard for stage 3: a re-run over an already-aligned URDF finds
    the ``<parent>_gripper_end`` fixed joint already in place. We treat it as
    already applied only when the parent link, child link, and origin (xyz +
    rpy, within 1e-9) all match the spec — otherwise the stage falls through
    and raises the existing "attach link missing" error / emits a fresh joint,
    matching legacy behavior when a hand-edited joint sits at the target name.
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
    raw_xyz = origin.get("xyz")
    raw_rpy = origin.get("rpy")
    try:
        xyz = [float(v) for v in raw_xyz.split()] if raw_xyz else [0.0] * 3
        rpy = [float(v) for v in raw_rpy.split()] if raw_rpy else [0.0] * 3
    except ValueError:
        return False
    if len(xyz) != 3 or len(rpy) != 3:
        return False
    tol = 1e-9
    for got, want in zip(xyz, expected_xyz):
        if abs(got - want) > tol:
            return False
    for got, want in zip(rpy, expected_rpy):
        if abs(got - want) > tol:
            return False
    return True


__all__ = ["insert_gripper_end_children"]
