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

from __future__ import annotations
import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

ACTION_POS_SCALE = 0.05
ACTION_ROT_SCALE = 0.5
GRIPPER_WIDTH = 0.08

DEFAULT_CAMERA_CONFIGS: dict[str, dict[str, Any]] = {
    "robot0_agentview_center": {
        "pos": [-0.6, 0.0, 1.15],
        "quat": [
            0.636945903301239,
            0.3325185477733612,
            -0.3199238181114197,
            -0.6175596117973328,
        ],
        "parent_body": "mobilebase0_support",
    },
    "robot0_agentview_left": {
        "pos": [-0.5, 0.35, 1.05],
        "quat": [0.55623853, 0.29935253, -0.37678665, -0.6775092],
        "camera_attribs": {"fovy": "60"},
        "parent_body": "mobilebase0_support",
    },
    "robot0_agentview_right": {
        "pos": [-0.5, -0.35, 1.05],
        "quat": [
            0.6775091886520386,
            0.3767866790294647,
            -0.2993525564670563,
            -0.55623859167099,
        ],
        "camera_attribs": {"fovy": "60"},
        "parent_body": "mobilebase0_support",
    },
    "robot0_frontview": {
        "pos": [-0.5, 0.0, 0.95],
        "quat": [
            0.6088936924934387,
            0.3814677894115448,
            -0.3673907518386841,
            -0.5905545353889465,
        ],
        "camera_attribs": {"fovy": "60"},
        "parent_body": "mobilebase0_support",
    },
    "robot0_eye_in_hand": {
        "pos": [0.05, 0.0, 0.0],
        "quat": [0.0, 0.707107, 0.707107, 0.0],
        "camera_attribs": {"fovy": "75"},
        "parent_body": "robot0_right_hand",
    },
}

# RoboCasa's LeRobot state stores the grip-site position together with the
# right_hand body orientation. Shift the state origin back to the right_hand
# body before applying the wrist camera's local MJCF pose.
DEFAULT_EEF_TO_HAND = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -0.097],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

STATE_SLICES = {
    "base_position": slice(0, 3),
    "base_rotation_xyzw": slice(3, 7),
    "eef_position_relative": slice(7, 10),
    "eef_rotation_relative_xyzw": slice(10, 14),
    "gripper_qpos": slice(14, 16),
}
ACTION_SLICES = {
    "base_motion": slice(0, 4),
    "control_mode": slice(4, 5),
    "eef_position": slice(5, 8),
    "eef_rotation": slice(8, 11),
    "gripper_close": slice(11, 12),
}

BASE_TRANSLATION_THRESHOLD_M = 0.02
BASE_ROTATION_THRESHOLD_RAD = math.radians(2.0)


def osc_action_to_ee_pose(
    ee_state: np.ndarray,
    osc_action: np.ndarray,
) -> np.ndarray:
    ee_state = np.asarray(ee_state, dtype=np.float64)
    osc_action = np.asarray(osc_action, dtype=np.float64)
    action_pos = ee_state[..., :3] + osc_action[..., :3] * ACTION_POS_SCALE
    current_rot = Rotation.from_quat(ee_state[..., 3:7], scalar_first=True)
    delta_rot = Rotation.from_rotvec(osc_action[..., 3:6] * ACTION_ROT_SCALE)
    action_rot = current_rot * delta_rot
    action_quat = action_rot.as_quat(scalar_first=True)
    return np.concatenate([action_pos, action_quat], axis=-1)


def ee_pose_to_osc_action(
    ee_state: np.ndarray,
    ee_pose: np.ndarray,
) -> np.ndarray:
    ee_state = np.asarray(ee_state, dtype=np.float64)
    ee_pose = np.asarray(ee_pose, dtype=np.float64)

    target_pos = ee_pose[..., :3]
    target_rot = Rotation.from_quat(ee_pose[..., 3:7], scalar_first=True)

    current_rot = Rotation.from_quat(ee_state[..., 3:7], scalar_first=True)
    delta_pos = (target_pos - ee_state[..., :3]) / ACTION_POS_SCALE
    delta_rot = current_rot.inv() * target_rot
    delta_rotvec = delta_rot.as_rotvec() / ACTION_ROT_SCALE
    return np.concatenate([delta_pos, delta_rotvec], axis=-1)


def get_gripper_openness(gripper_state: np.ndarray) -> np.ndarray:
    return (gripper_state[:, :1] - gripper_state[:, 1:2]) / GRIPPER_WIDTH


def quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if np.linalg.norm(q) < 1e-12:
        return np.eye(3, dtype=np.float64)
    return (
        Rotation.from_quat(q, scalar_first=False)
        .as_matrix()
        .astype(np.float64)
    )


def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if np.linalg.norm(q) < 1e-12:
        return np.eye(3, dtype=np.float64)
    return (
        Rotation.from_quat(q, scalar_first=True).as_matrix().astype(np.float64)
    )


def xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat)
    return quat[..., [3, 0, 1, 2]]


def make_pose(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rot
    pose[:3, 3] = pos
    return pose


def pose_inv(pose: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float64)
    rot = pose[:3, :3]
    trans = pose[:3, 3]
    inv[:3, :3] = rot.T
    inv[:3, 3] = -rot.T @ trans
    return inv


def camera_axis_correction() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def camera_local_pose(cam_cfg: dict[str, Any]) -> np.ndarray:
    local_pos = np.asarray(cam_cfg["pos"], dtype=np.float64)
    local_rot = quat_wxyz_to_mat(np.asarray(cam_cfg["quat"], dtype=np.float64))
    return make_pose(local_pos, local_rot)


def intrinsic_from_fovy(fovy: float, height: int, width: int) -> np.ndarray:
    f = 0.5 * height / math.tan(float(fovy) * math.pi / 360.0)
    return np.array(
        [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def get_camera_fovy(cam_cfg: dict[str, Any] | None, camera_name: str) -> float:
    if cam_cfg is not None:
        attribs = cam_cfg.get("camera_attribs") or {}
        if "fovy" in attribs:
            return float(attribs["fovy"])
    default_cfg = DEFAULT_CAMERA_CONFIGS.get(camera_name, {})
    return float((default_cfg.get("camera_attribs") or {}).get("fovy", 60.0))


def state_base_pose(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = state[STATE_SLICES["base_position"]].astype(np.float64, copy=True)
    rot = quat_xyzw_to_mat(state[STATE_SLICES["base_rotation_xyzw"]])
    return pos, rot


def state_eef_pose_world(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base_pos, base_rot = state_base_pose(state)
    eef_rel = state[STATE_SLICES["eef_position_relative"]]
    eef_rel_rot = quat_xyzw_to_mat(
        state[STATE_SLICES["eef_rotation_relative_xyzw"]]
    )
    return base_pos + base_rot @ eef_rel, base_rot @ eef_rel_rot


def infer_if_mobile(
    base_position: np.ndarray, base_rotation_xyzw: np.ndarray
) -> bool:
    if len(base_position) <= 1:
        return False

    base_xy = base_position[:, :2].astype(np.float64, copy=False)
    step_translation = np.linalg.norm(np.diff(base_xy, axis=0), axis=1).sum()
    max_translation = np.linalg.norm(base_xy - base_xy[0], axis=1).max()

    quats = base_rotation_xyzw.astype(np.float64, copy=True)
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-12
    quats[valid] /= norms[valid]
    dots = np.abs(quats @ quats[0])
    angles = 2.0 * np.arccos(np.clip(dots, -1.0, 1.0))
    max_rotation = float(angles.max()) if len(angles) else 0.0

    return bool(
        step_translation >= BASE_TRANSLATION_THRESHOLD_M
        or max_translation >= BASE_TRANSLATION_THRESHOLD_M
        or max_rotation >= BASE_ROTATION_THRESHOLD_RAD
    )


def state_camera_world_pose(
    state: np.ndarray,
    camera_name: str,
    cam_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    local_pose = camera_local_pose(cam_cfg)
    parent_body = cam_cfg.get("parent_body")
    if parent_body == "robot0_right_hand":
        parent_pos, parent_rot = state_eef_pose_world(state)
        parent_pose = make_pose(parent_pos, parent_rot) @ DEFAULT_EEF_TO_HAND
    elif parent_body:
        parent_pos, parent_rot = state_base_pose(state)
        parent_pose = make_pose(parent_pos, parent_rot)
    else:
        parent_pose = np.eye(4, dtype=np.float64)
    del camera_name
    camera_pose = parent_pose @ local_pose
    return camera_pose[:3, 3], camera_pose[:3, :3]


def t_base2cam_from_world_pose(
    camera_pos: np.ndarray,
    camera_rot: np.ndarray,
    base_pos: np.ndarray,
    base_rot: np.ndarray,
) -> np.ndarray:
    t_base_to_world = make_pose(base_pos, base_rot)
    t_cam_to_world = (
        make_pose(camera_pos, camera_rot) @ camera_axis_correction()
    )
    t_base_to_cam = pose_inv(t_cam_to_world) @ t_base_to_world
    return t_base_to_cam


def camera_calibration_from_config(
    cam_cfg: dict[str, Any],
) -> np.ndarray:
    local_pose = camera_local_pose(cam_cfg)
    parent_body = cam_cfg.get("parent_body")
    if parent_body == "robot0_right_hand":
        parent_to_camera = DEFAULT_EEF_TO_HAND @ local_pose
    else:
        parent_to_camera = local_pose
    return pose_inv(parent_to_camera @ camera_axis_correction())
