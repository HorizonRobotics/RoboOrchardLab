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

"""Unit tests for the EgoDex hand-frame derivation helpers.

Covers :func:`robo_orchard_lab.dataset.egodex.hand_frame.derive_hand_frames`
and its use through :class:`HandTF2Gripper.tf2gripper`:

- Canonical joint layouts produce the identity rotation and the expected
  ``[x, y, z]`` column ordering.
- Origin / openness follow the tip-midpoint and minimum-distance formulas
  documented on :class:`HandFrame`.
- Degenerate inputs (collapsed thumb, y-axis parallel to z, etc.) surface
  descriptive :class:`ValueError`\\ s naming the offending side.
- The end-to-end gripper state produced by :class:`HandTF2Gripper` matches
  the per-hand :class:`HandFrame` it was assembled from.
"""

import pytest
import torch
from pytorch3d.transforms import quaternion_to_matrix

from robo_orchard_lab.dataset.egodex.hand_frame import derive_hand_frames
from robo_orchard_lab.dataset.egodex.transforms import HandTF2Gripper

FINGER_PARTS = (
    "LittleFinger",
    "RingFinger",
    "MiddleFinger",
    "IndexFinger",
)


def _joint_names() -> list[str]:
    names = []
    for side in ("left", "right"):
        names.extend(
            [
                f"{side}Hand",
                f"{side}ThumbKnuckle",
                f"{side}ThumbIntermediateBase",
                f"{side}ThumbIntermediateTip",
                f"{side}ThumbTip",
            ]
        )
        for finger in FINGER_PARTS:
            names.extend(
                [
                    f"{side}{finger}Metacarpal",
                    f"{side}{finger}Knuckle",
                    f"{side}{finger}IntermediateBase",
                    f"{side}{finger}IntermediateTip",
                    f"{side}{finger}Tip",
                ]
            )
    return names


def _canonical_transforms() -> tuple[torch.Tensor, list[str]]:
    joint_names = _joint_names()
    transforms = torch.eye(4, dtype=torch.float64).repeat(
        2, len(joint_names), 1, 1
    )

    def set_position(side: str, joint: str, xyz: tuple[float, float, float]):
        transforms[:, joint_names.index(f"{side}{joint}"), :3, 3] = (
            torch.tensor(xyz, dtype=transforms.dtype)
        )

    for side in ("left", "right"):
        hand = (0.0, 0.02, -0.03)
        knuckles = {
            "IndexFinger": (0.0, -0.04, 0.0),
            "MiddleFinger": (0.0, -0.015, 0.005),
            "RingFinger": (0.0, 0.01, 0.004),
            "LittleFinger": (0.0, 0.04, 0.0),
        }
        set_position(side, "Hand", hand)
        set_position(side, "ThumbKnuckle", (0.0, -0.03, 0.0))
        set_position(side, "ThumbIntermediateBase", (0.0, -0.03, 0.025))
        set_position(side, "ThumbIntermediateTip", (0.0, -0.03, 0.055))
        set_position(side, "ThumbTip", (0.0, -0.03, 0.08))

        for finger, knuckle in knuckles.items():
            set_position(side, f"{finger}Metacarpal", hand)
            set_position(side, f"{finger}Knuckle", knuckle)
            set_position(
                side,
                f"{finger}IntermediateBase",
                (knuckle[0], knuckle[1], knuckle[2] + 0.025),
            )
            set_position(
                side,
                f"{finger}IntermediateTip",
                (knuckle[0], knuckle[1], knuckle[2] + 0.05),
            )
            set_position(
                side,
                f"{finger}Tip",
                (knuckle[0], knuckle[1], knuckle[2] + 0.075),
            )
    return transforms, joint_names


@pytest.mark.parametrize("side", ["left", "right"])
def test_canonical_frame_is_right_down_front(side: str):
    transforms, joint_names = _canonical_transforms()

    frames, _ = derive_hand_frames(transforms, joint_names)
    frame = frames[side]

    expected_rotation = torch.eye(3, dtype=transforms.dtype).repeat(2, 1, 1)
    torch.testing.assert_close(
        frame.rotation, expected_rotation, atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        frame.rotation.transpose(-1, -2) @ frame.rotation,
        expected_rotation,
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        torch.linalg.det(frame.rotation),
        torch.ones(2, dtype=transforms.dtype),
        atol=1e-6,
        rtol=0,
    )


def test_production_hand_frame_has_narrow_state_surface():
    transforms, joint_names = _canonical_transforms()

    frames, _ = derive_hand_frames(transforms, joint_names)
    frame = frames["left"]

    assert set(frame.__dataclass_fields__) == {
        "openness",
        "origin",
        "rotation",
    }


def test_origin_preserves_existing_tip_midpoint_formula():
    transforms, joint_names = _canonical_transforms()

    frames, _ = derive_hand_frames(transforms, joint_names)

    for side, frame in frames.items():
        thumb_tip = transforms[:, joint_names.index(f"{side}ThumbTip"), :3, 3]
        finger_tips = torch.stack(
            [
                transforms[:, joint_names.index(f"{side}{finger}Tip"), :3, 3]
                for finger in FINGER_PARTS
            ],
            dim=-2,
        )
        expected = (thumb_tip + finger_tips.mean(dim=-2)) / 2
        torch.testing.assert_close(frame.origin, expected)


def test_helper_preserves_batch_dtype_and_device():
    transforms, joint_names = _canonical_transforms()
    transforms = transforms.to(dtype=torch.float32)

    frames, _ = derive_hand_frames(transforms, joint_names)

    for frame in frames.values():
        assert frame.origin.shape == (2, 3)
        assert frame.rotation.shape == (2, 3, 3)
        assert frame.origin.dtype == transforms.dtype
        assert frame.origin.device == transforms.device


def test_collapsed_thumb_knuckle_to_tip_raises_descriptive_error():
    transforms, joint_names = _canonical_transforms()
    knuckle = transforms[:, joint_names.index("leftThumbKnuckle"), :3, 3]
    transforms[:, joint_names.index("leftThumbTip"), :3, 3] = knuckle

    with pytest.raises(ValueError, match="left thumb direction"):
        derive_hand_frames(transforms, joint_names)


def test_y_direction_is_projected_index_to_little():
    transforms, joint_names = _canonical_transforms()
    expected = torch.tensor([1.0, 2.0, 0.0], dtype=transforms.dtype)
    expected = expected / torch.linalg.vector_norm(expected)
    for side in ("left", "right"):
        transforms[
            :, joint_names.index(f"{side}LittleFingerKnuckle"), :3, 3
        ] = torch.tensor([0.04, 0.04, 0.04], dtype=transforms.dtype)

    frames, diagnostics = derive_hand_frames(transforms, joint_names)

    for side, frame in frames.items():
        torch.testing.assert_close(
            frame.rotation[..., :, 1],
            expected.expand(2, 3),
            atol=1e-6,
            rtol=0,
        )
        assert set(diagnostics[side].__dataclass_fields__) == {
            "thumb_start",
            "thumb_end",
            "y_guide_start",
            "y_guide_end",
        }


def test_parallel_index_to_little_raises_without_secondary_fallback():
    transforms, joint_names = _canonical_transforms()
    transforms[:, joint_names.index("leftIndexFingerKnuckle"), :3, 3] = (
        torch.tensor([0.0, 0.0, 0.0], dtype=transforms.dtype)
    )
    transforms[:, joint_names.index("leftLittleFingerKnuckle"), :3, 3] = (
        torch.tensor([0.0, 0.0, 0.08], dtype=transforms.dtype)
    )
    transforms[:, joint_names.index("leftHand"), :3, 3] = torch.tensor(
        [-0.04, 0.0, 0.08], dtype=transforms.dtype
    )

    with pytest.raises(ValueError, match="left y-axis"):
        derive_hand_frames(transforms, joint_names)


def test_unrecoverable_y_direction_raises_descriptive_error():
    transforms, joint_names = _canonical_transforms()
    for side in ("left", "right"):
        transforms[
            :, joint_names.index(f"{side}IndexFingerKnuckle"), :3, 3
        ] = torch.tensor([0.0, 0.0, 0.0], dtype=transforms.dtype)
        transforms[
            :, joint_names.index(f"{side}LittleFingerKnuckle"), :3, 3
        ] = torch.tensor([0.0, 0.0, 0.08], dtype=transforms.dtype)
        transforms[:, joint_names.index(f"{side}Hand"), :3, 3] = torch.tensor(
            [0.0, 0.0, -0.08], dtype=transforms.dtype
        )

    with pytest.raises(ValueError, match="left y-axis"):
        derive_hand_frames(transforms, joint_names)


def test_unrecoverable_thumb_direction_raises_descriptive_error():
    transforms, joint_names = _canonical_transforms()
    for joint in (
        "ThumbKnuckle",
        "ThumbIntermediateBase",
        "ThumbIntermediateTip",
        "ThumbTip",
    ):
        transforms[:, joint_names.index(f"left{joint}"), :3, 3] = 0

    with pytest.raises(ValueError, match="left thumb direction"):
        derive_hand_frames(transforms, joint_names)


def test_hand_tf_to_gripper_uses_anatomical_frames():
    transforms, joint_names = _canonical_transforms()
    for side in ("left", "right"):
        transforms[
            :, joint_names.index(f"{side}LittleFingerKnuckle"), :3, 3
        ] = torch.tensor([0.04, 0.04, 0.04], dtype=transforms.dtype)
    expected_frames, _ = derive_hand_frames(transforms, joint_names)

    state = HandTF2Gripper().tf2gripper(transforms, joint_names)

    assert state.shape == (2, 2, 8)
    for hand_index, side in enumerate(("left", "right")):
        frame = expected_frames[side]
        torch.testing.assert_close(
            state[:, hand_index, 0], frame.openness[..., 0]
        )
        torch.testing.assert_close(state[:, hand_index, 1:4], frame.origin)
        torch.testing.assert_close(
            quaternion_to_matrix(state[:, hand_index, 4:]),
            frame.rotation,
            atol=1e-6,
            rtol=0,
        )

        thumb_tip = transforms[:, joint_names.index(f"{side}ThumbTip"), :3, 3]
        finger_tips = torch.stack(
            [
                transforms[:, joint_names.index(f"{side}{finger}Tip"), :3, 3]
                for finger in FINGER_PARTS
            ],
            dim=-2,
        )
        expected_openness = (
            torch.linalg.vector_norm(
                finger_tips - thumb_tip[..., None, :], dim=-1
            )
            .min(dim=-1)
            .values
        )
        torch.testing.assert_close(state[:, hand_index, 0], expected_openness)
