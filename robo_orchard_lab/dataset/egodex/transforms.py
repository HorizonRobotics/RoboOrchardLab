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
import torch
from pytorch3d.transforms import matrix_to_quaternion


class SimpleStateSampling:
    def __init__(
        self,
        hist_steps,
        pred_steps,
    ):
        self.hist_steps = hist_steps
        self.pred_steps = pred_steps

    def __call__(self, data):
        state = data.pop("joint_transforms")  # T, J, 4, 4
        step_index = data["step_index"]
        hist_steps = self.hist_steps
        pred_steps = self.pred_steps

        pred_state = state[step_index + 1 : step_index + 1 + pred_steps]
        pred_mask = np.zeros(pred_steps)
        pred_mask[: pred_state.shape[0]] = 1

        if pred_state.shape[0] != pred_steps:
            padding = np.tile(
                state[-1:], (pred_steps - pred_state.shape[0], 1, 1, 1)
            )
            pred_state = np.concatenate([pred_state, padding], axis=0)

        hist_state = state[
            max(0, step_index + 1 - hist_steps) : step_index + 1
        ]
        if hist_state.shape[0] != hist_steps:
            padding = np.tile(
                state[:1], (hist_steps - hist_state.shape[0], 1, 1, 1)
            )
            hist_state = np.concatenate([padding, hist_state], axis=0)

        data.update(
            hist_joint_transforms=hist_state,
            pred_joint_transforms=pred_state,
            pred_mask=pred_mask,
        )
        return data


class HandTF2Gripper:
    LEFT_FINGER_TIPS = [
        "leftIndexFingerTip",
        "leftLittleFingerTip",
        "leftMiddleFingerTip",
        "leftRingFingerTip",
    ]
    LEFT_THUMB_TIP = "leftThumbTip"
    RIGHT_FINGER_TIPS = [
        "rightIndexFingerTip",
        "rightLittleFingerTip",
        "rightMiddleFingerTip",
        "rightRingFingerTip",
    ]
    RIGHT_THUMB_TIP = "rightThumbTip"
    LEFT_CONVENTION = np.array(
        [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    RIGHT_CONVENTION = np.array(
        [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )

    def __call__(self, data):
        joint_names = data["joint_names"]
        if "hist_joint_transforms" in data:
            data["hist_robot_state"] = self.tf2gripper(
                data["hist_joint_transforms"], joint_names
            )
        if "pred_joint_transforms" in data:
            data["pred_robot_state"] = self.tf2gripper(
                data["pred_joint_transforms"], joint_names
            )
        if "joint_transforms" in data:
            data["robot_state"] = self.tf2gripper(
                data["joint_transforms"], joint_names
            )
        return data

    def tf2gripper(self, tf, joint_names):
        if not isinstance(tf, torch.Tensor):
            tf = torch.tensor(tf)
        left_tip_id = [joint_names.index(x) for x in self.LEFT_FINGER_TIPS]
        left_thumb_id = joint_names.index(self.LEFT_THUMB_TIP)
        right_tip_id = [joint_names.index(x) for x in self.RIGHT_FINGER_TIPS]
        right_thumb_id = joint_names.index(self.RIGHT_THUMB_TIP)

        left_tip_tf = tf[..., left_tip_id, :, :]
        left_thumb_tf = tf[..., left_thumb_id, :, :]
        left_openness = (
            torch.linalg.norm(
                left_tip_tf[..., :, :3, 3] - left_thumb_tf[..., None, :3, 3],
                dim=-1,
            )
            .min(-1, keepdims=True)
            .values
        )
        left_tf = (left_tip_tf.mean(axis=-3) + left_thumb_tf) / 2
        left_tf = left_tf @ self.LEFT_CONVENTION
        left_xyz = left_tf[..., :3, 3]

        left_quat = matrix_to_quaternion(left_tf[..., :3, :3])
        left_state = torch.cat([left_openness, left_xyz, left_quat], dim=-1)

        right_tip_tf = tf[..., right_tip_id, :, :]
        right_thumb_tf = tf[..., right_thumb_id, :, :]
        right_openness = (
            torch.linalg.norm(
                right_tip_tf[..., :, :3, 3] - right_thumb_tf[..., None, :3, 3],
                dim=-1,
            )
            .min(-1, keepdims=True)
            .values
        )
        right_tf = (right_tip_tf.mean(axis=-3) + right_thumb_tf) / 2
        right_tf = right_tf @ self.RIGHT_CONVENTION
        right_xyz = right_tf[..., :3, 3]
        right_quat = matrix_to_quaternion(right_tf[..., :3, :3])
        right_state = torch.cat(
            [right_openness, right_xyz, right_quat], dim=-1
        )

        state = torch.stack([left_state, right_state], dim=-2)
        return state


class UpSampleRobotState:
    def __init__(self, pred_steps, hist_steps=None):
        self.pred_steps = pred_steps
        self.hist_steps = hist_steps

    def __call__(self, data):
        robot_state = torch.cat(
            [data["hist_robot_state"][-1:], data["pred_robot_state"]]
        )  # steps x num_joint x 8
        state_dim = robot_state.shape[-1]
        pred_mask = torch.cat([data["pred_mask"][:1], data["pred_mask"]])[
            :, None
        ]
        robot_state = torch.cat(
            [robot_state.flatten(-2), pred_mask.to(robot_state)], dim=-1
        )
        robot_state = robot_state.T[None]  # 1 x [num_joint*8] x steps

        robot_state = torch.nn.functional.interpolate(
            robot_state, self.pred_steps + 1, mode="linear", align_corners=True
        )
        data["pred_robot_state"] = (
            robot_state[0].T[1:, :-1].unflatten(-1, (-1, state_dim))
        )
        data["pred_mask"] = robot_state[0].T[1:, -1].to(dtype=torch.bool)
        if (
            self.hist_steps is not None
            and data["hist_robot_state"].shape[0] != self.hist_steps
        ):
            data["hist_robot_state"] = torch.nn.functional.interpolate(
                data["hist_robot_state"].flatten(-2).T[None],
                self.hist_steps,
                mode="linear",
                align_corners=True,
            )[0].T.unflatten(-1, (-1, state_dim))
        return data
