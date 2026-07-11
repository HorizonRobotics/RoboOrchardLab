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

"""Insert `*_camera_mount_compat` aliases when reference frames moved.

Compare the FK pose of each declared camera reference link in the origin URDF
against the transformer's in-progress URDF. If the reference link stayed at the
same rotation and translation (up to floating-point tolerance), no compat
alias is needed. If it moved, insert a fixed child link
``<origin_link>_camera_mount_compat`` whose transform restores the origin
pose.

Insertion order matters — the compat joint must appear before any
gripper-subtree joint of the same parent so downstream loaders (which iterate
children in document order for extrinsic lookup) find the calibration frame
first. This mirrors the assertion in
``test_g1_aligned_urdf_keeps_camera_mounts_before_gripper_base`` and is
enforced here by inserting the compat joint immediately after any existing
child joint whose parent link is the reference link, before the first
gripper-flavored joint.
"""

from __future__ import annotations
import xml.etree.ElementTree as ET
from typing import Sequence

import numpy as np
import pytorch_kinematics as pk
import torch

from projects.holobrain_internal.common.urdf_tools.transform._linalg import (
    OriginTransform,
    format_vec,
)

_TOLERANCE = 1e-6


def insert_camera_compat_if_moved(
    aligned_root: ET.Element,
    origin_urdf_bytes: bytes,
    aligned_urdf_bytes: bytes,
    camera_references: Sequence[str],
) -> list[str]:
    """Insert `*_camera_mount_compat` links where reference frames moved.

    Args:
        aligned_root: ``<robot>`` element under construction, modified in
            place.
        origin_urdf_bytes: Raw origin URDF bytes.
        aligned_urdf_bytes: Serialized in-progress aligned URDF bytes. Kept
            separate from `aligned_root` so we do not force the caller to
            re-parse the tree.
        camera_references: Manifest-declared reference link names.

    Returns:
        list[str]: Names of newly inserted compat links, in order.
    """

    if not camera_references:
        return []

    origin_chain = pk.build_chain_from_urdf(origin_urdf_bytes)
    aligned_chain = pk.build_chain_from_urdf(aligned_urdf_bytes)

    # Zero-pose FK is sufficient here: the compat alias is a fixed transform
    # relative to the parent link, and axis normalization / EE insertion are
    # frame-only rewrites.
    origin_frames = origin_chain.forward_kinematics(_zero_pose(origin_chain))
    aligned_frames = aligned_chain.forward_kinematics(
        _zero_pose(aligned_chain)
    )

    inserted: list[str] = []
    for link_name in camera_references:
        origin_pose = _pose_matrix(origin_frames, link_name)
        aligned_pose = _pose_matrix(aligned_frames, link_name)
        if origin_pose is None:
            raise ValueError(
                f"camera reference link '{link_name}' missing from origin URDF"
            )
        if aligned_pose is None:
            raise ValueError(
                f"camera reference link '{link_name}' missing from aligned "
                "URDF"
            )
        if np.allclose(origin_pose, aligned_pose, atol=_TOLERANCE):
            continue
        compat_link_name = f"{link_name}_camera_mount_compat"
        compat_joint_name = _compat_joint_name(link_name)
        # Transform that recovers the origin pose expressed in the aligned
        # reference-link frame: T = aligned⁻¹ @ origin.
        compensator = np.linalg.inv(aligned_pose) @ origin_pose
        _insert_compat_link(
            aligned_root,
            parent_link_name=link_name,
            compat_link_name=compat_link_name,
            compat_joint_name=compat_joint_name,
            compensator=compensator,
        )
        inserted.append(compat_link_name)
    return inserted


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _zero_pose(chain) -> dict:
    joint_names = chain.get_joint_parameter_names()
    return {name: torch.tensor(0.0) for name in joint_names}


def _pose_matrix(frames, link_name: str) -> np.ndarray | None:
    if link_name not in frames:
        return None
    return frames[link_name].get_matrix()[0].cpu().numpy()


def _compat_joint_name(reference_link: str) -> str:
    return f"{reference_link}_camera_mount_compat_joint"


def _insert_compat_link(
    root: ET.Element,
    *,
    parent_link_name: str,
    compat_link_name: str,
    compat_joint_name: str,
    compensator: np.ndarray,
) -> None:
    """Insert a compat link + fixed joint before gripper children.

    The compat link is appended at the end of ``<robot>``. The compat joint is
    inserted immediately before the first joint whose parent link is
    `parent_link_name` and whose child link name looks like a gripper subtree
    (``gripper_``/``Left_``/``Right_``/``finger_``). If no such joint exists,
    the compat joint is appended at the end.
    """

    compat_link = ET.SubElement(root, "link")
    compat_link.set("name", compat_link_name)

    joint = ET.Element("joint")
    joint.set("name", compat_joint_name)
    joint.set("type", "fixed")
    origin = ET.SubElement(joint, "origin")
    transform = OriginTransform.from_matrix(compensator)
    rpy, xyz = transform.to_rpy_xyz()
    origin.set("rpy", format_vec(rpy))
    origin.set("xyz", format_vec(xyz))
    parent_el = ET.SubElement(joint, "parent")
    parent_el.set("link", parent_link_name)
    child_el = ET.SubElement(joint, "child")
    child_el.set("link", compat_link_name)

    children = list(root)
    insert_at = len(children)
    for idx, element in enumerate(children):
        if element.tag != "joint":
            continue
        joint_parent = element.find("parent")
        joint_child = element.find("child")
        if joint_parent is None or joint_child is None:
            continue
        if joint_parent.get("link") != parent_link_name:
            continue
        child_name = joint_child.get("link", "")
        if _looks_like_gripper_link(child_name):
            insert_at = idx
            break
    root.insert(insert_at, joint)


_GRIPPER_PREFIXES = (
    "gripper_",
    "left_gripper",
    "right_gripper",
    "finger_",
)

_GRIPPER_MARKERS = (
    "gripper",
    "finger",
    "left_00_link",
    "left_01_link",
    "left_support",
    "left_pad",
    "right_00_link",
    "right_01_link",
    "right_support",
    "right_pad",
)


def _looks_like_gripper_link(name: str) -> bool:
    if not name:
        return False
    lowered = name.lower()
    if lowered.startswith(_GRIPPER_PREFIXES):
        return True
    return any(marker in lowered for marker in _GRIPPER_MARKERS)
