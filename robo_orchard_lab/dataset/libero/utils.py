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

import numpy as np
from scipy.spatial.transform import Rotation as R  # noqa: N817


def pose_inv(pose):
    """Computes the inverse of a homogeneous matrix.

    Args:
        pose (np.array): 4x4 matrix for the pose to inverse

    Returns:
        np.array: 4x4 matrix for the inverse pose
    """

    pose_inv = np.zeros((4, 4))
    pose_inv[:3, :3] = pose[:3, :3].T
    pose_inv[:3, 3] = -pose_inv[:3, :3].dot(pose[:3, 3])
    pose_inv[3, 3] = 1.0
    return pose_inv


def transform_ee_rotations(ee_states, delta_matrix):
    """Combines EE rotations (axis-angle) with an incremental rotation matrix.

    Args:
        ee_states (np.array): Array of shape (N, 6) or (6,), where [3:6]
                              is the axis-angle vector.
        delta_matrix (np.array): A 3x3 rotation matrix.

    Returns:
        np.array: Quaternions of shape (N, 4) in [w, x, y, z] order.
    """
    r_delta = R.from_matrix(delta_matrix)
    ee_states = np.atleast_2d(ee_states)
    ee_rotvecs = ee_states[:, 3:6]
    r_ee_original = R.from_rotvec(ee_rotvecs)
    r_ee_transformed = r_ee_original * r_delta
    q_xyzw = r_ee_transformed.as_quat()
    q_wxyz = q_xyzw[:, [3, 0, 1, 2]]

    return q_wxyz
