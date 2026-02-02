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

import copy

import numpy as np
import torch
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

__all__ = [
    "AddJointScaleShift",
    "SimpleStateSampling",
    "RobotStateJitter",
    "TransformRobotState",
]


class AddJointScaleShift:
    def __init__(self, scale_shift):
        if isinstance(scale_shift, (list, tuple)):
            scale_shift = torch.Tensor(scale_shift)
        elif isinstance(scale_shift, np.ndarray):
            scale_shift = torch.from_numpy(scale_shift)
        self.scale_shift = scale_shift

    def __call__(self, data):
        data["joint_scale_shift"] = copy.deepcopy(self.scale_shift)
        return data


class SimpleStateSampling:
    def __init__(self, hist_steps, pred_steps):
        self.hist_steps = hist_steps
        self.pred_steps = pred_steps

    def __call__(self, data):
        state = data["joint_state"]
        step_index = data["step_index"]
        hist_steps = self.hist_steps

        hist_state = state[
            max(0, step_index + 1 - hist_steps) : step_index + 1
        ]
        if hist_state.shape[0] != hist_steps:
            padding = np.tile(
                state[:1], (hist_steps - hist_state.shape[0], 1, 1)
            )
            hist_state = np.concatenate([padding, hist_state], axis=0)

        data.update(
            hist_robot_state=hist_state,
        )

        if "action" not in data:
            return data

        action = data["action"]
        step_index = data["step_index"]
        pred_steps = self.pred_steps

        pred_state = action[step_index : step_index + pred_steps]
        pred_mask = np.ones(pred_steps, dtype=bool)
        if pred_state.shape[0] != pred_steps:
            padding = np.tile(
                action[-1:], (pred_steps - pred_state.shape[0], 1, 1)
            )
            pred_mask[-padding.shape[0]:] = False
            pred_state = np.concatenate([pred_state, padding], axis=0)

        data.update(
            pred_robot_state=pred_state,
            pred_mask=pred_mask,
        )
        return data


class RobotStateJitter:
    def __init__(self, noise_range, add_to_pred=False):
        self.range = np.array(noise_range)
        self.add_to_pred = add_to_pred

    def __call__(self, data):
        assert "hist_robot_state" in data
        num_steps, num_joints, dim = data["hist_robot_state"].shape
        if self.add_to_pred:
            num_steps = 1
        noise = np.random.uniform(
            self.range[..., 0],
            self.range[..., 1],
            size=[num_steps, num_joints, dim],
        )
        noise = torch.from_numpy(noise).to(data["hist_robot_state"])
        data["hist_robot_state"] = data["hist_robot_state"] + noise
        if self.add_to_pred:
            data["pred_robot_state"] = data["pred_robot_state"] + noise
        return data


class TransformRobotState:
    def __call__(self, data):
        embodiedment_mat = data["embodiedment_mat"]
        data["hist_robot_state"] = self._apply_transform(
            data["hist_robot_state"], embodiedment_mat
        )
        if "pred_robot_state" in data:
            data["pred_robot_state"] = self._apply_transform(
                data["pred_robot_state"], embodiedment_mat
            )
        return data

    def _apply_transform(self, robot_state, transform):
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
        pos_new = t_new[:, :3, 3]
        quat_new = matrix_to_quaternion(t_new[:, :3, :3])
        res = torch.cat([joint_val, pos_new, quat_new], dim=-1)
        return res.reshape(original_shape)
