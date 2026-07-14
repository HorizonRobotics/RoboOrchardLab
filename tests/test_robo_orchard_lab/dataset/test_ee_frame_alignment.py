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

"""Round-trip regression tests for ee_frame_alignment.

The training-time TransformRobotState sandwiches an SE(3) alignment on the
local action/state frame (right-multiply) between the base->ego pose
(left-multiply). The deploy-time inverse must recover the original robot state
so downstream simulators see the pre-alignment action convention. These tests
exercise the LIBERO and RoboCasa variants end-to-end.
"""

import numpy as np
import pytest
import torch
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

from robo_orchard_lab.dataset.libero.transforms import (
    TransformRobotState as LiberoTransformRobotState,
)
from robo_orchard_lab.dataset.robocasa.transforms import (
    TransformRobotState as RoboCasaTransformRobotState,
)

# LIBERO alignment: (left, up, forward) -> (right, down, forward) is R_z(180),
# stored as a full 4x4 SE(3) with zero translation.
_R_ALIGN_LIBERO = torch.tensor(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=torch.float64,
)

# RoboCasa alignment: (up, right, forward) -> (right, down, forward) is
# R_z(90) about the local z axis, stored as a full 4x4 SE(3).
_R_ALIGN_ROBOCASA = torch.tensor(
    [
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=torch.float64,
)


def _random_embodiedment_mat(seed: int) -> torch.Tensor:
    """Build a random SE(3) transform for a base->ego style embodiment."""
    rng = np.random.default_rng(seed)
    # Random rotation via QR of a Gaussian matrix, then translation.
    a = rng.standard_normal((3, 3))
    q, r = np.linalg.qr(a)
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = q
    mat[:3, 3] = rng.standard_normal(3)
    return torch.tensor(mat, dtype=torch.float64)


def _random_robot_state(seed: int, shape: tuple[int, ...]) -> torch.Tensor:
    """Random 8D state batch: [gripper, xyz, quat_wxyz], quat normalized."""
    rng = np.random.default_rng(seed)
    gripper = rng.uniform(0.0, 1.0, size=shape + (1,))
    pos = rng.standard_normal(shape + (3,))
    quat = rng.standard_normal(shape + (4,))
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    state = np.concatenate([gripper, pos, quat], axis=-1)
    return torch.tensor(state, dtype=torch.float64)


def _apply_transform_libero_deploy(
    robot_state: torch.Tensor,
    transform: torch.Tensor,
    transform_right: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reimplements the LIBERO deploy `apply_transform`.

    Kept in-test so the regression test does not depend on importing project
    code (which requires holobrain_utils). Behaviorally identical to the
    deploy helper — a pure 4x4 SE(3) sandwich.
    """
    device = robot_state.device
    dtype = robot_state.dtype
    original_shape = robot_state.shape
    state_flat = robot_state.reshape(-1, 8)
    joint_val = state_flat[:, :1]
    pos = state_flat[:, 1:4]
    quat = state_flat[:, 4:]
    r_mats = quaternion_to_matrix(quat)
    t_mats = torch.eye(4, device=device, dtype=dtype).repeat(
        state_flat.shape[0], 1, 1
    )
    t_mats[:, :3, :3] = r_mats
    t_mats[:, :3, 3] = pos
    t_new = transform.to(device, dtype) @ t_mats
    if transform_right is not None:
        t_new = t_new @ torch.as_tensor(
            transform_right, device=device, dtype=dtype
        )
    pos_new = t_new[:, :3, 3]
    quat_new = matrix_to_quaternion(t_new[:, :3, :3])
    res = torch.cat([joint_val, pos_new, quat_new], dim=-1)
    return res.reshape(original_shape)


def _apply_inverse_embodiment_robocasa_deploy(
    robot_state: torch.Tensor,
    embodiedment_mat: torch.Tensor,
    ee_frame_alignment: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reimplements the RoboCasa deploy `_apply_inverse_embodiment`."""
    transform = torch.linalg.inv(embodiedment_mat)
    state_flat = robot_state.reshape(-1, 8)
    joint_val = state_flat[:, :1]
    pos = state_flat[:, 1:4]
    quat = state_flat[:, 4:]
    r_mats = quaternion_to_matrix(quat)
    t_mats = torch.eye(
        4, device=robot_state.device, dtype=robot_state.dtype
    ).repeat(state_flat.shape[0], 1, 1)
    t_mats[:, :3, :3] = r_mats
    t_mats[:, :3, 3] = pos
    t_new = transform.to(robot_state.device, robot_state.dtype) @ t_mats
    if ee_frame_alignment is not None:
        align_mat = torch.as_tensor(
            ee_frame_alignment,
            device=robot_state.device,
            dtype=robot_state.dtype,
        )
        t_new = t_new @ torch.linalg.inv(align_mat)
    pos_new = t_new[:, :3, 3]
    quat_new = matrix_to_quaternion(t_new[:, :3, :3])
    ret = torch.cat([joint_val, pos_new, quat_new], dim=-1)
    return ret.reshape(robot_state.shape)


def _assert_states_close(a: torch.Tensor, b: torch.Tensor, atol: float):
    torch.testing.assert_close(a[..., :1], b[..., :1], atol=atol, rtol=atol)
    torch.testing.assert_close(a[..., 1:4], b[..., 1:4], atol=atol, rtol=atol)
    # Quaternions can flip sign and still represent the same rotation.
    rot_a = quaternion_to_matrix(a[..., 4:].reshape(-1, 4))
    rot_b = quaternion_to_matrix(b[..., 4:].reshape(-1, 4))
    torch.testing.assert_close(rot_a, rot_b, atol=atol, rtol=atol)


@pytest.mark.parametrize(
    "r_align",
    [_R_ALIGN_LIBERO, _R_ALIGN_ROBOCASA],
    ids=["libero_180", "robocasa_90"],
)
def test_libero_transform_round_trip(r_align: torch.Tensor):
    """train_forward then deploy_inverse must recover the raw robot state."""
    original = _random_robot_state(seed=0, shape=(4, 2))
    embodiedment_mat = _random_embodiedment_mat(seed=1)
    data = {
        "embodiedment_mat": embodiedment_mat,
        "ee_frame_alignment": r_align,
        "hist_robot_state": original.clone(),
        "pred_robot_state": original.clone(),
    }
    transformed = LiberoTransformRobotState()(data)
    inv_embodiedment = torch.linalg.inv(embodiedment_mat)
    inv_align = torch.linalg.inv(r_align)
    recovered_hist = _apply_transform_libero_deploy(
        transformed["hist_robot_state"],
        inv_embodiedment,
        transform_right=inv_align,
    )
    recovered_pred = _apply_transform_libero_deploy(
        transformed["pred_robot_state"],
        inv_embodiedment,
        transform_right=inv_align,
    )
    _assert_states_close(recovered_hist, original, atol=1e-10)
    _assert_states_close(recovered_pred, original, atol=1e-10)


@pytest.mark.parametrize(
    "r_align",
    [_R_ALIGN_LIBERO, _R_ALIGN_ROBOCASA],
    ids=["libero_180", "robocasa_90"],
)
def test_robocasa_transform_round_trip(r_align: torch.Tensor):
    """Same round-trip guarantee for the RoboCasa variant."""
    original = _random_robot_state(seed=2, shape=(3, 1))
    embodiedment_mat = _random_embodiedment_mat(seed=3)
    data = {
        "embodiedment_mat": embodiedment_mat,
        "ee_frame_alignment": r_align,
        "hist_robot_state": original.clone(),
        "pred_robot_state": original.clone(),
    }
    transformed = RoboCasaTransformRobotState()(data)
    recovered_hist = _apply_inverse_embodiment_robocasa_deploy(
        transformed["hist_robot_state"],
        embodiedment_mat,
        ee_frame_alignment=r_align,
    )
    recovered_pred = _apply_inverse_embodiment_robocasa_deploy(
        transformed["pred_robot_state"],
        embodiedment_mat,
        ee_frame_alignment=r_align,
    )
    _assert_states_close(recovered_hist, original, atol=1e-10)
    _assert_states_close(recovered_pred, original, atol=1e-10)


def test_alignment_leaves_positions_untouched_libero():
    """Right-mult SO(3) on the local frame must not shift world positions."""
    original = _random_robot_state(seed=4, shape=(5, 2))
    embodiedment_mat = _random_embodiedment_mat(seed=5)
    data_with = {
        "embodiedment_mat": embodiedment_mat,
        "ee_frame_alignment": _R_ALIGN_LIBERO,
        "hist_robot_state": original.clone(),
    }
    data_without = {
        "embodiedment_mat": embodiedment_mat,
        "hist_robot_state": original.clone(),
    }
    with_align = LiberoTransformRobotState()(data_with)
    without_align = LiberoTransformRobotState()(data_without)
    torch.testing.assert_close(
        with_align["hist_robot_state"][..., 1:4],
        without_align["hist_robot_state"][..., 1:4],
        atol=1e-12,
        rtol=1e-12,
    )
    # Rotations must differ.
    rot_with = quaternion_to_matrix(
        with_align["hist_robot_state"][..., 4:].reshape(-1, 4)
    )
    rot_without = quaternion_to_matrix(
        without_align["hist_robot_state"][..., 4:].reshape(-1, 4)
    )
    assert not torch.allclose(rot_with, rot_without, atol=1e-6)


def test_alignment_leaves_positions_untouched_robocasa():
    original = _random_robot_state(seed=6, shape=(4, 1))
    embodiedment_mat = _random_embodiedment_mat(seed=7)
    data_with = {
        "embodiedment_mat": embodiedment_mat,
        "ee_frame_alignment": _R_ALIGN_ROBOCASA,
        "hist_robot_state": original.clone(),
    }
    data_without = {
        "embodiedment_mat": embodiedment_mat,
        "hist_robot_state": original.clone(),
    }
    with_align = RoboCasaTransformRobotState()(data_with)
    without_align = RoboCasaTransformRobotState()(data_without)
    torch.testing.assert_close(
        with_align["hist_robot_state"][..., 1:4],
        without_align["hist_robot_state"][..., 1:4],
        atol=1e-12,
        rtol=1e-12,
    )
    rot_with = quaternion_to_matrix(
        with_align["hist_robot_state"][..., 4:].reshape(-1, 4)
    )
    rot_without = quaternion_to_matrix(
        without_align["hist_robot_state"][..., 4:].reshape(-1, 4)
    )
    assert not torch.allclose(rot_with, rot_without, atol=1e-6)


def test_absent_ee_frame_alignment_is_noop():
    """When ee_frame_alignment is absent, behavior equals the old code."""
    original = _random_robot_state(seed=8, shape=(3, 2))
    embodiedment_mat = _random_embodiedment_mat(seed=9)
    for cls in (LiberoTransformRobotState, RoboCasaTransformRobotState):
        data = {
            "embodiedment_mat": embodiedment_mat,
            "hist_robot_state": original.clone(),
        }
        out = cls()(data)
        # Old code path: pos_new = R @ pos + t; rot_new = R @ rot.
        rot = embodiedment_mat[:3, :3].to(original)
        pos_expected = (rot @ original[..., 1:4].reshape(-1, 3).T).T.reshape(
            original.shape[:-1] + (3,)
        ) + embodiedment_mat[:3, 3].to(original)
        torch.testing.assert_close(
            out["hist_robot_state"][..., 1:4],
            pos_expected,
            atol=1e-10,
            rtol=1e-10,
        )
