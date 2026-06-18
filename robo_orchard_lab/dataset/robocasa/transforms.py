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

import numpy as np
import torch
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix


class SimpleStateSampling:
    def __init__(
        self,
        hist_steps,
        pred_steps,
        use_master_openness=True,
        use_master_state=True,
    ):
        self.hist_steps = hist_steps
        self.pred_steps = pred_steps
        self.use_master_openness = use_master_openness
        self.use_master_state = use_master_state

    def __call__(self, data):
        if "robot_state" not in data and "hist_robot_state" in data:
            return data
        if "robot_state" not in data and "joint_state" in data:
            state = data["joint_state"]
            if state.ndim == 2:
                state = state[:, None]
            data["robot_state"] = state

        state = data["robot_state"]  # T x num_joint x 8
        step_index = data["step_index"]

        hist_state = state[
            max(0, step_index + 1 - self.hist_steps) : step_index + 1
        ]
        if hist_state.shape[0] != self.hist_steps:
            padding = np.tile(
                state[:1],
                (self.hist_steps - hist_state.shape[0], 1, 1),
            )
            hist_state = np.concatenate([padding, hist_state], axis=0)

        action_state = data.get("master_robot_state")
        if action_state is None:
            pred_source = state
        else:
            pred_source = state.copy()
            if self.use_master_state:
                pred_source[..., 1:] = action_state[..., 1:]
            if self.use_master_openness:
                pred_source[..., :1] = action_state[..., :1]

        pred_state = pred_source[step_index : step_index + self.pred_steps]
        pred_mask = np.ones(pred_state.shape[0], dtype=bool)
        if pred_state.shape[0] != self.pred_steps:
            padding = np.tile(
                pred_source[-1:],
                (self.pred_steps - pred_state.shape[0], 1, 1),
            )
            pred_state = np.concatenate([pred_state, padding], axis=0)
            pred_mask = np.concatenate(
                [
                    pred_mask,
                    np.zeros(padding.shape[0], dtype=bool),
                ],
                axis=0,
            )

        data["hist_robot_state"] = hist_state
        data["pred_robot_state"] = pred_state
        data["pred_mask"] = pred_mask
        data.pop("robot_state", None)
        data.pop("master_robot_state", None)
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

    def _apply_transform(
        self, robot_state: torch.Tensor, transform: torch.Tensor
    ) -> torch.Tensor:
        original_shape = robot_state.shape
        state_flat = robot_state.reshape(-1, 8)
        joint_val = state_flat[:, :1]
        pos = state_flat[:, 1:4]
        quat = state_flat[:, 4:]

        transform = transform.to(
            device=robot_state.device, dtype=robot_state.dtype
        )
        transform_rot = transform[..., :3, :3]
        transform_pos = transform[..., :3, 3]

        pos_new = (
            torch.matmul(transform_rot, pos.unsqueeze(-1)).squeeze(-1)
            + transform_pos
        )
        quat_new = matrix_to_quaternion(
            transform_rot @ quaternion_to_matrix(quat)
        )
        res = torch.cat([joint_val, pos_new, quat_new], dim=-1)
        return res.reshape(original_shape)
