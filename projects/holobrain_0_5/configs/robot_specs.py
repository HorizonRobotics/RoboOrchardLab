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


import copy
import os

__all__ = ["get_robot_spec"]

URDF_BASE = os.environ.get("HOLOBRAIN_URDF_BASE", "./urdf")

_ROBOT_SPECS = {
    "robotwin_aloha": dict(
        urdf=f"{URDF_BASE}/robotwin/aloha/arx5_description_isaac.pruned.urdf",
        kinematics=dict(
            arm_joint_id=[list(range(6)), list(range(8, 14))],
            arm_link_keys=[
                [
                    "fl_link1",
                    "fl_link2",
                    "fl_link3",
                    "fl_link4",
                    "fl_link5",
                    "fl_link6",
                ],
                [
                    "fr_link1",
                    "fr_link2",
                    "fr_link3",
                    "fr_link4",
                    "fr_link5",
                    "fr_link6",
                ],
            ],
            finger_keys=[["fl_link6_tcp"], ["fr_link6_tcp"]],
            arm_connection_joint_indices=[6, 6],
        ),
        num_joint=14,
        arm_length=0.921,
        symmetric_groups=[[0, 1]],
        gripper_indices=[6, 13],
        joint_mask=([True] * 6 + [False]) * 2,
    ),
    "robotwin_ur5_wsg": dict(
        urdf=f"{URDF_BASE}/robotwin/ur5_wsg/robotwin2_dual_arm_ur5_wsg.pruned.urdf",
        kinematics=dict(
            arm_joint_id=[list(range(6)), list(range(8, 14))],
            arm_link_keys=[
                [
                    "left_shoulder_link",
                    "left_upper_arm_link",
                    "left_forearm_link",
                    "left_wrist_1_link",
                    "left_wrist_2_link",
                    "left_wrist_3_link",
                ],
                [
                    "right_shoulder_link",
                    "right_upper_arm_link",
                    "right_forearm_link",
                    "right_wrist_1_link",
                    "right_wrist_2_link",
                    "right_wrist_3_link",
                ],
            ],
            finger_keys=[
                ["left_wrist_3_link_tcp"],
                ["right_wrist_3_link_tcp"],
            ],
            arm_connection_joint_indices=[6, 6],
        ),
        num_joint=14,
        arm_length=1.393,
        symmetric_groups=[[0, 1]],
        gripper_indices=[6, 13],
        joint_mask=([True] * 6 + [False]) * 2,
    ),
}


def get_robot_spec(robot_type: str) -> dict:
    """Return the spec registered for ``robot_type``.

    Args:
        robot_type (str): Name of a registered robot.

    Returns:
        dict: ``urdf``, ``kinematics``, ``num_joint``, ``gripper_indices``
            and ``joint_mask``. It is a fresh deep copy, so callers may
            override entries on it without affecting later calls.

    Raises:
        KeyError: If no robot is registered under that name.
    """
    if robot_type not in _ROBOT_SPECS:
        raise KeyError(
            f"No robot is registered as {robot_type!r}. Every robot a data "
            "config names needs an entry in _ROBOT_SPECS."
        )
    return copy.deepcopy(_ROBOT_SPECS[robot_type])
