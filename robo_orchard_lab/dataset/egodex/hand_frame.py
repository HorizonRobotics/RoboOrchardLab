# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
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

"""Derive anatomically-consistent hand frames from EgoDex joint transforms.

The EgoDex tracker exposes per-joint 4x4 transforms for each hand. Downstream
training and evaluation want a compact "gripper" state per hand: an origin
between fingertips, a right-handed orientation, and a scalar openness. This
module implements that derivation and also returns the intermediate anatomical
"guides" used by the debug viewer (``visualize_egodex_hand_frame.py``).

Coordinate convention (right-handed, per hand):
    * ``z`` — thumb knuckle -> thumb tip.
    * ``y`` — index-knuckle-side -> little-finger-side, projected onto z's
      normal plane so it is orthogonal to z.
    * ``x`` — ``y x z`` completes the right-handed basis (points "right" when
      y is "down" and z is "front").

Rotation matrices are stored with columns ``[x, y, z]`` in the input frame,
matching the PyTorch3D convention consumed by
:func:`pytorch3d.transforms.matrix_to_quaternion`.
"""

from dataclasses import dataclass

import torch

# Keep normalization finite before callers reject vectors shorter than this
# threshold. It is small enough not to distort valid EgoDex measurements,
# whose relevant joint spacings are approximately 5e-3 m and above.
_EPS = 1e-6

# Non-thumb fingers listed in the anatomical order used to average fingertip
# positions and to pick the y-axis endpoints (index -> little).
_FINGERS = ("IndexFinger", "MiddleFinger", "RingFinger", "LittleFinger")


@dataclass(frozen=True)
class HandFrame:
    """Compact per-hand gripper state consumed by the training pipeline.

    All tensors share the input transforms' batch shape ``[...]``, dtype, and
    device.
    """

    #: Minimum thumb-tip to non-thumb fingertip distance, shaped ``[..., 1]``.
    #: Meters.
    openness: torch.Tensor
    #: Gripper origin, shaped ``[..., 3]``. Midpoint between the thumb tip and
    #: the centroid of the four non-thumb fingertips.
    origin: torch.Tensor
    #: Right-handed rotation matrix shaped ``[..., 3, 3]``. Columns are the
    #: local ``x, y, z`` axes expressed in the input coordinate frame.
    rotation: torch.Tensor


@dataclass(frozen=True)
class HandFrameDiagnostics:
    """Raw anatomical endpoints used to build a :class:`HandFrame`.

    The viewer overlays these on the 3D scene so the derived axes can be
    visually cross-checked against the original joint chain. All tensors share
    the input transforms' batch shape ``[...]`` and are 3D positions in the
    input frame.
    """

    #: Thumb knuckle position (z-axis tail).
    thumb_start: torch.Tensor
    #: Thumb tip position (z-axis head).
    thumb_end: torch.Tensor
    #: Index-finger knuckle position; anchors the raw y-axis ray before
    #: projection onto z's normal plane.
    y_guide_start: torch.Tensor
    #: Little-finger knuckle position.
    y_guide_end: torch.Tensor


def _normalize(vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit-normalize along the last axis and also return the raw norm.

    The norm is returned separately (not just clamped inside the division) so
    callers can detect degenerate inputs — a valid direction has norm >= _EPS,
    while a collapsed vector yields a finite placeholder direction backed by a
    near-zero norm.

    Args:
        vector (torch.Tensor): Any tensor with a trailing size-3 axis.

    Returns:
        tuple: ``(unit_vector, norm)`` where ``unit_vector`` shares
            ``vector``'s shape and ``norm`` drops the trailing axis.
    """
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    return vector / norm.clamp_min(_EPS), norm[..., 0]


def _derive_thumb_direction(
    thumb: torch.Tensor, side: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the unit thumb direction (z-axis) and its raw endpoints.

    Args:
        thumb (torch.Tensor): Thumb joint positions shaped
            ``[..., 4, 3]`` ordered ``knuckle -> intermediate_base ->
            intermediate_tip -> tip``.
        side (str): ``"left"`` or ``"right"``, used only for error messages.

    Returns:
        tuple: ``(direction, thumb_start, thumb_end)`` where ``direction`` is
            the unit vector from ``thumb_start`` (knuckle) to ``thumb_end``
            (tip).

    Raises:
        ValueError: If any batch entry has a collapsed knuckle-to-tip span.
    """
    thumb_start = thumb[..., 0, :]
    thumb_end = thumb[..., 3, :]
    direction, norm = _normalize(thumb_end - thumb_start)
    if torch.any(norm < _EPS):
        raise ValueError(
            f"Cannot derive {side} thumb direction from tracked joints."
        )
    return direction, thumb_start, thumb_end


def _derive_one_hand(
    transforms: torch.Tensor,
    name_to_index: dict[str, int],
    side: str,
) -> tuple[HandFrame, HandFrameDiagnostics]:
    """Compute the frame and diagnostics for one hand.

    Args:
        transforms (torch.Tensor): Full joint transforms shaped
            ``[..., J, 4, 4]``.
        name_to_index (dict[str, int]): Joint name -> index into ``J``.
        side (str): ``"left"`` or ``"right"``; drives the joint-name prefix.

    Returns:
        tuple: ``(HandFrame, HandFrameDiagnostics)`` for the requested hand.
    """
    # Resolve all joint indices up front so subsequent slices are simple
    # advanced-indexing calls and there is no per-frame string lookup.
    thumb_indices = [
        name_to_index[f"{side}ThumbKnuckle"],
        name_to_index[f"{side}ThumbIntermediateBase"],
        name_to_index[f"{side}ThumbIntermediateTip"],
        name_to_index[f"{side}ThumbTip"],
    ]
    finger_tip_indices = [
        name_to_index[f"{side}{finger}Tip"] for finger in _FINGERS
    ]
    knuckle_indices = [
        name_to_index[f"{side}{finger}Knuckle"] for finger in _FINGERS
    ]
    # Only translation columns of the 4x4 transforms are used; each tensor
    # below has shape [..., N, 3] with N being the count of selected joints.
    thumb = transforms[..., thumb_indices, :3, 3]
    thumb_tip = thumb[..., 3, :]
    finger_tips = transforms[..., finger_tip_indices, :3, 3]

    # Gripper origin: the midpoint between the thumb tip and
    # the centroid of the four non-thumb fingertips.
    origin = (thumb_tip + finger_tips.mean(dim=-2)) / 2
    # Openness: distance to the *closest* opposing fingertip. Using min rather
    # than mean matches how a two-finger gripper "closes" on whatever contact
    # point is nearest, reaching 0 when the thumb tip coincides with one of the
    # non-thumb fingertips.
    openness = (
        torch.linalg.vector_norm(finger_tips - thumb_tip[..., None, :], dim=-1)
        .min(dim=-1, keepdim=True)
        .values
    )

    # z-axis: knuckle -> tip along the thumb.
    z_axis, thumb_start, thumb_end = _derive_thumb_direction(thumb, side)

    # Positive y follows the anatomical index-side -> little-finger-side ray.
    knuckles = transforms[..., knuckle_indices, :3, 3]
    y_guide_start = knuckles[..., 0, :]  # index knuckle
    y_guide_end = knuckles[..., 3, :]  # little-finger knuckle

    # Remove its component parallel to z so y lies in z's normal plane:
    #     projected_y = raw_y - dot(raw_y, z) * z
    primary_y = y_guide_end - y_guide_start
    projected_primary = primary_y - (
        (primary_y * z_axis).sum(dim=-1, keepdim=True) * z_axis
    )
    y_axis, y_norm = _normalize(projected_primary)
    if torch.any(y_norm < _EPS):
        raise ValueError(f"Cannot derive {side} y-axis from tracked joints.")

    # For canonical y=down and z=front, y cross z points right.
    x_axis, x_norm = _normalize(torch.linalg.cross(y_axis, z_axis, dim=-1))
    if torch.any(x_norm < _EPS):
        raise ValueError(f"Cannot derive {side} x-axis from tracked joints.")

    # Rotation-matrix columns are the local x/y/z axes expressed in the input
    # coordinate frame, matching the convention consumed by PyTorch3D.
    rotation = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    frame = HandFrame(openness=openness, origin=origin, rotation=rotation)

    diagnostics = HandFrameDiagnostics(
        thumb_start=thumb_start,
        thumb_end=thumb_end,
        y_guide_start=y_guide_start,
        y_guide_end=y_guide_end,
    )
    return frame, diagnostics


def derive_hand_frames(
    transforms: torch.Tensor, joint_names: list[str]
) -> tuple[dict[str, HandFrame], dict[str, HandFrameDiagnostics]]:
    """Derive right-handed hand frames and their anatomical diagnostics.

    The z-axis follows the thumb from knuckle to tip. The y-axis points from
    the index-knuckle side toward the little-finger side after projection onto
    the plane perpendicular to z. The x-axis is ``y x z`` (right-handed).

    The tensors returned inside :class:`HandFrame` and
    :class:`HandFrameDiagnostics` share ``transforms``'s batch shape, dtype,
    and device — the leading ``[..., J, 4, 4]`` shape reduces to ``[..., N]``
    with the joint axis removed.

    Args:
        transforms (torch.Tensor): Joint transforms shaped ``[..., J, 4, 4]``.
        joint_names (list[str]): Names of length ``J`` corresponding to the
            joint dimension. Must contain the full set of thumb, knuckle, and
            fingertip joints for both ``left`` and ``right`` sides
            (e.g. ``"leftThumbKnuckle"``, ``"rightIndexFingerTip"``, ...).

    Returns:
        tuple: ``(frames, diagnostics)`` — two dicts each keyed by ``"left"``
            and ``"right"``. ``frames`` holds :class:`HandFrame` (production
            gripper state); ``diagnostics`` holds :class:`HandFrameDiagnostics`
            (the raw endpoints used to build the axes).

    Raises:
        TypeError: If ``transforms`` is not a :class:`torch.Tensor`.
        ValueError: If ``transforms`` does not end in ``[..., 4, 4]``, or if
            any hand's z/y/x axis collapses (e.g. thumb knuckle == thumb tip).
    """
    if not isinstance(transforms, torch.Tensor):
        raise TypeError("transforms must be a torch.Tensor.")
    if transforms.shape[-2:] != (4, 4):
        raise ValueError("transforms must have shape [..., joints, 4, 4].")

    # Build the lookup table once per call and share it across both hands,
    # openness, origin, and orientation extraction.
    name_to_index = {name: index for index, name in enumerate(joint_names)}
    frames = {}
    diagnostics = {}
    for side in ("left", "right"):
        frame, side_diagnostics = _derive_one_hand(
            transforms, name_to_index, side
        )
        frames[side] = frame
        diagnostics[side] = side_diagnostics
    return frames, diagnostics
