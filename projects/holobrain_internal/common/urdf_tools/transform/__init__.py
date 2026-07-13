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

"""Compose the URDF alignment transformer stages.

The pipeline runs in a fixed order:

    1. ``normalize_actuated_axes`` — rewrite non-`ẑ` actuated joint axes with
       S-compensation on the child link and downstream child joints.
    2. ``insert_ee_children`` — insert semantic ``*_ee`` fixed children,
       moving link geometry as documented in ``ALIGNMENT_NOTES.md``.
    3. ``insert_gripper_end_children`` — insert a ``<parent>_gripper_end``
       fixed child per arm on the EE frame (identity rotation, +Z offset),
       giving dataset configs a uniform ``finger_keys`` handle.
    4. ``insert_camera_compat_if_moved`` — compare origin vs aligned FK for
       each declared camera reference link; insert
       ``*_camera_mount_compat`` aliases only where reference frames moved.
       Runs last so the FK detection sees the fully-built aligned tree.
    5. Final pretty-print pass via :func:`_format_aligned_urdf` so every
       aligned URDF ships with the same 2-space indent regardless of origin
       whitespace. Without this the diff of a re-align run grows to hundreds
       of lines every time upstream whitespace shifts, drowning out the real
       semantic changes.

The result is returned as a serialized bytes buffer so the caller can write
it or diff it without re-parsing.
"""

from __future__ import annotations
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from projects.holobrain_internal.common.urdf_tools.cases import (
    UrdfAlignmentCase,
)
from projects.holobrain_internal.common.urdf_tools.transform.axis import (
    normalize_actuated_axes,
)
from projects.holobrain_internal.common.urdf_tools.transform.camera import (
    insert_camera_compat_if_moved,
)
from projects.holobrain_internal.common.urdf_tools.transform.ee_frames import (
    insert_ee_children,
)
from projects.holobrain_internal.common.urdf_tools.transform.gripper_end import (  # noqa: E501
    insert_gripper_end_children,
)


@dataclass(frozen=True)
class TransformReport:
    """What each stage produced. Useful for debug logs and the CLI."""

    normalized_joints: tuple[str, ...]
    inserted_ee_joints: tuple[str, ...]
    inserted_gripper_end_joints: tuple[str, ...]
    inserted_camera_compat_links: tuple[str, ...]


@dataclass(frozen=True)
class TransformResult:
    """Serialized aligned URDF plus a stage-by-stage report."""

    aligned_urdf_bytes: bytes
    report: TransformReport


def apply_alignment(
    origin_urdf_path: Path,
    case: UrdfAlignmentCase,
) -> TransformResult:
    """Apply the alignment pipeline to `origin_urdf_path` and return bytes."""

    origin_bytes = origin_urdf_path.read_bytes()
    tree = ET.parse(origin_urdf_path)
    root = tree.getroot()

    normalized = normalize_actuated_axes(root)
    inserted_ee = insert_ee_children(root, case.alignment.ee_frames)
    inserted_gripper_end = insert_gripper_end_children(
        root, case.alignment.gripper_ends
    )
    intermediate_bytes = ET.tostring(root, encoding="utf-8")
    inserted_compat = insert_camera_compat_if_moved(
        root,
        origin_urdf_bytes=origin_bytes,
        aligned_urdf_bytes=intermediate_bytes,
        camera_references=case.alignment.camera_references,
    )
    aligned_bytes = _format_aligned_urdf(root)
    return TransformResult(
        aligned_urdf_bytes=aligned_bytes,
        report=TransformReport(
            normalized_joints=tuple(normalized),
            inserted_ee_joints=tuple(inserted_ee),
            inserted_gripper_end_joints=tuple(inserted_gripper_end),
            inserted_camera_compat_links=tuple(inserted_compat),
        ),
    )


def _format_aligned_urdf(root: ET.Element) -> bytes:
    """Serialize `root` with a stable 2-space indent and trailing newline.

    Upstream URDFs land in the pipeline with wildly different whitespace
    (tabs, 4-space, mixed). ``ET.tostring`` alone preserves that noise, so
    every re-align run produces churny diffs even when the transformer's
    semantic edits are tiny. ``ET.indent`` normalizes the tree in place
    before serialization so the on-disk aligned URDF has one canonical
    layout: 2-space indent, one element per line, trailing newline.

    Idempotence is important — running this pass on its own output yields
    identical bytes, which is what makes a re-align a no-op when nothing
    upstream changed. The dedicated test in the pytest suite pins that
    invariant.
    """

    # Strip any pre-existing inter-element whitespace text so `ET.indent`
    # starts from a clean slate; otherwise mixed origin whitespace can leak
    # into text nodes and defeat re-indentation.
    for element in root.iter():
        if element.text is not None and not element.text.strip():
            element.text = None
        if element.tail is not None and not element.tail.strip():
            element.tail = None
    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="utf-8")
    if not body.endswith(b"\n"):
        body += b"\n"
    return body


__all__ = [
    "TransformReport",
    "TransformResult",
    "apply_alignment",
]
