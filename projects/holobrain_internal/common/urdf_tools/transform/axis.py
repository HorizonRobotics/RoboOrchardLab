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

"""Actuated-joint axis normalization for URDF alignment.

For every actuated (`revolute`, `continuous`, `prismatic`) joint whose axis
after unit-normalization does not equal `ẑ`, we rewrite the joint frame so the
axis becomes `0 0 1` while preserving control semantics. See
``ALIGNMENT_NOTES.md § Axis Compensation Rule`` for the derivation.

The rewrite:

    1. Compute ``S`` that maps ``ẑ`` to the old axis via minimum rotation.
    2. Post-multiply the joint's ``<origin>`` rotation by ``S``.
    3. Re-express visual / collision / inertial ``<origin>`` blocks of the
       immediate child link by ``S⁻¹`` (translation and rpy).
    4. Re-express every outgoing joint's ``<origin>`` from that child link by
       ``S⁻¹``.
    5. Rewrite the joint's ``<axis>`` to ``0 0 1``.

Inertia tensor values are left untouched: because the ``<inertial>``
``<origin>`` is also rotated by ``S⁻¹``, the CoM's rotation matrix
relative to the world frame is unchanged, and the tensor stays expressed
in the same physical frame.
"""

from __future__ import annotations
import logging
import xml.etree.ElementTree as ET
from typing import Iterable

import numpy as np

from projects.holobrain_internal.common.urdf_tools.transform._linalg import (
    OriginTransform,
    format_vec,
    is_unit,
    rotation_from_ez_to,
    unit,
)

_LOGGER = logging.getLogger(__name__)

_ACTUATED_JOINT_TYPES = frozenset({"revolute", "continuous", "prismatic"})


def normalize_actuated_axes(root: ET.Element) -> list[str]:
    """Rewrite the URDF `<robot>` element so actuated axes are `0 0 1`.

    Args:
        root: The XML ``<robot>`` element. Modified in place.

    Returns:
        list[str]: Ordered names of the joints whose axes were rewritten.
    """

    joints_by_name = {
        joint.get("name", ""): joint for joint in root.findall("joint")
    }
    links_by_name = {
        link.get("name", ""): link for link in root.findall("link")
    }

    outgoing_by_parent: dict[str, list[ET.Element]] = {}
    for joint in root.findall("joint"):
        parent_link = _child_get(joint, "parent", "link")
        if parent_link:
            outgoing_by_parent.setdefault(parent_link, []).append(joint)

    rewritten: list[str] = []
    for name, joint in joints_by_name.items():
        joint_type = joint.get("type", "")
        if joint_type not in _ACTUATED_JOINT_TYPES:
            continue
        axis = _read_axis(joint)
        if axis is None:
            continue
        if _is_close_to_positive_z(axis):
            _write_axis(joint, np.array([0.0, 0.0, 1.0]))
            continue
        s_matrix = rotation_from_ez_to(axis)
        _post_multiply_joint_origin(joint, s_matrix)
        _write_axis(joint, np.array([0.0, 0.0, 1.0]))
        child_link_name = _child_get(joint, "child", "link")
        if not child_link_name:
            raise ValueError(
                f"joint '{name}' has no <child link=..>; cannot normalize axis"
            )
        child_link = links_by_name.get(child_link_name)
        if child_link is None:
            raise ValueError(
                f"joint '{name}' references missing child link "
                f"'{child_link_name}'"
            )
        s_inv = s_matrix.T
        _rewrite_link_origins(child_link, s_inv)
        for outgoing in outgoing_by_parent.get(child_link_name, ()):
            _rewrite_element_origin(outgoing, s_inv)
        rewritten.append(name)

    return rewritten


# ---------------------------------------------------------------------------
# Element-level helpers.
# ---------------------------------------------------------------------------


def _read_axis(joint: ET.Element) -> np.ndarray | None:
    axis_el = joint.find("axis")
    if axis_el is None:
        return None
    raw = axis_el.get("xyz")
    if not raw:
        return None
    try:
        values = [float(v) for v in raw.split()]
    except ValueError:
        return None
    if len(values) != 3:
        return None
    vec = np.asarray(values, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        # `axis xyz="0 0 0"` is a URDF quirk seen on non-actuated placeholder
        # joints; leave those alone.
        return None
    if not is_unit(vec):
        _LOGGER.warning(
            "joint '%s' axis is not unit length (%.6f); normalizing",
            joint.get("name", "<unnamed>"),
            norm,
        )
    return unit(vec)


def _write_axis(joint: ET.Element, axis: Iterable[float]) -> None:
    axis_el = joint.find("axis")
    if axis_el is None:
        axis_el = ET.SubElement(joint, "axis")
    axis_el.set("xyz", format_vec(axis))


def _is_close_to_positive_z(axis: np.ndarray, atol: float = 1e-6) -> bool:
    return bool(np.allclose(unit(axis), [0.0, 0.0, 1.0], atol=atol))


def _post_multiply_joint_origin(
    joint: ET.Element,
    s_matrix: np.ndarray,
) -> None:
    """Set the joint origin's rotation to `R_old @ S`; leave xyz unchanged."""

    origin = _ensure_origin(joint)
    transform = _read_origin(origin)
    new_rot = transform.rot @ s_matrix
    _write_origin(
        origin,
        OriginTransform(xyz=transform.xyz, rot=new_rot),
    )


def _rewrite_link_origins(link: ET.Element, s_inv: np.ndarray) -> None:
    """Re-express visual/collision/inertial `<origin>` by `S⁻¹` in-place."""

    for tag in ("inertial", "visual", "collision"):
        for element in link.findall(tag):
            _rewrite_element_origin(element, s_inv)


def _rewrite_element_origin(element: ET.Element, s_inv: np.ndarray) -> None:
    """Multiply the `<origin>` of `element` by `s_inv` on the left."""

    origin = element.find("origin")
    if origin is None:
        origin = ET.SubElement(element, "origin")
        origin.set("xyz", "0 0 0")
        origin.set("rpy", "0 0 0")
    transform = _read_origin(origin)
    new_rot = s_inv @ transform.rot
    new_xyz = s_inv @ transform.xyz
    _write_origin(origin, OriginTransform(xyz=new_xyz, rot=new_rot))


def _ensure_origin(joint: ET.Element) -> ET.Element:
    origin = joint.find("origin")
    if origin is None:
        origin = ET.SubElement(joint, "origin")
        origin.set("xyz", "0 0 0")
        origin.set("rpy", "0 0 0")
    return origin


def _read_origin(origin: ET.Element) -> OriginTransform:
    raw_rpy = origin.get("rpy")
    raw_xyz = origin.get("xyz")
    rpy = _parse_triple(raw_rpy) if raw_rpy else None
    xyz = _parse_triple(raw_xyz) if raw_xyz else None
    return OriginTransform.from_rpy_xyz(rpy, xyz)


def _write_origin(origin: ET.Element, transform: OriginTransform) -> None:
    rpy, xyz = transform.to_rpy_xyz()
    origin.set("rpy", format_vec(rpy))
    origin.set("xyz", format_vec(xyz))


def _parse_triple(value: str) -> list[float]:
    return [float(v) for v in value.split()]


def _child_get(element: ET.Element, tag: str, attr: str) -> str:
    child = element.find(tag)
    if child is None:
        return ""
    return child.get(attr, "") or ""
