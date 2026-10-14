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


import logging

import numpy as np
import torch
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

from projects.holobrain_0_5.utils import resolve_processor_dir
from robo_orchard_lab.dataset.libero.utils import pose_inv
from robo_orchard_lab.models.holobrain_0_5.processor import (
    HoloBrainProcessor,
    MultiArmManipulationInput,
)
from robo_orchard_lab.models.mixin import ModelMixin

logger = logging.getLogger(__name__)

__all__ = [
    "HoloBrainPolicy",
    "get_model",
    "reset_model",
    "predict_action",
    "apply_transform",
]

_STATE_DIM = 8

_JOINT_DIM = 1

_POS_END = 4

_QUAT_START = 4


def apply_transform(robot_state, transform, transform_right=None):
    """Undo ego and end-effector alignment transforms on pose actions.

    States have shape [..., 8]: gripper, xyz, scalar-first quaternion.
    The result uses inv(transform) @ pose @ inv(transform_right); the
    right transform is optional. The input is not modified.
    """
    robot_state = robot_state.to(torch.float64)
    transform = transform.to(robot_state)
    device = robot_state.device
    dtype = robot_state.dtype
    original_shape = robot_state.shape
    state_flat = robot_state.reshape(-1, _STATE_DIM)
    joint_val = state_flat[:, :_JOINT_DIM]
    pos = state_flat[:, _JOINT_DIM:_POS_END]
    quat = state_flat[:, _QUAT_START:]
    r_mats = quaternion_to_matrix(quat)
    t_mats = torch.eye(4, device=device, dtype=dtype).repeat(
        state_flat.shape[0], 1, 1
    )
    t_mats[:, :3, :3] = r_mats
    t_mats[:, :3, 3] = pos
    t_new = torch.linalg.solve(transform, t_mats)
    if transform_right is not None:
        t_new = torch.linalg.solve(
            transform_right.to(robot_state), t_new, left=False
        )
    pos_new = t_new[:, :3, 3]
    quat_new = matrix_to_quaternion(t_new[:, :3, :3])
    res = torch.cat([joint_val, pos_new, quat_new], dim=-1)
    return res.reshape(original_shape)


class HoloBrainPolicy:
    """Local HoloBrain 0.5 libero observation and action adapter."""

    def __init__(self, model_dir, processor, device="cuda"):
        """Load a local exported model and its matching dataset processor."""
        name = f"{processor}.json"
        processor_dir = resolve_processor_dir(model_dir, name)
        logger.info("Loading %s from %s", name, processor_dir)
        self.processor = HoloBrainProcessor.load(str(processor_dir), name)
        self.model = ModelMixin.load_model(model_dir, load_impl="native")
        self.model.eval().requires_grad_(False).to(device)

    def data_preprocess(self, obs, instruction):
        images = {}
        depths = {}
        t_world2cam = {}
        intrinsic = {}
        for cam_name, camera_data in obs["observation"].items():
            images[cam_name] = [camera_data["rgb"]]
            depths[cam_name] = [camera_data["depth"]]
            t_world2cam[cam_name] = pose_inv(camera_data["extrinsic_cv"])
            tmp_ = np.eye(4)
            tmp_[:3, :3] = camera_data["intrinsic_cv"]
            intrinsic[cam_name] = tmp_
        robot_state = obs["robot_state"][None]
        data = MultiArmManipulationInput(
            image=images,
            depth=depths,
            intrinsic=intrinsic,
            t_world2cam=t_world2cam,
            history_joint_state=robot_state,
            instruction=instruction,
        )
        data = self.processor.pre_process(data)
        return data

    @torch.inference_mode()
    def get_action(self, observation, instruction):
        data = self.data_preprocess(observation, instruction)
        model_outs = self.model(data)
        actions = self.processor.post_process(model_outs, data).pose.squeeze(1)
        actions = apply_transform(
            actions, data["embodiedment_mat"], data.get("ee_frame_alignment")
        )
        valid_action_step = 8
        actions = actions[:valid_action_step].cpu().numpy()
        actions[..., :1] = 1 - 2 * actions[..., :1]
        return actions


def get_model(usr_args):
    """RoboTwin/LIBERO callback; arguments come from the local eval config."""
    return HoloBrainPolicy(
        usr_args["model_config"],
        usr_args["model_processor"],
        usr_args.get("device", "cuda"),
    )


def reset_model(model):
    """Episode callback; this policy has no cross-call action buffer."""
    pass


def predict_action(model, observation, task_description):
    """Return eight absolute end-effector actions for the LIBERO rollout."""
    return model.get_action(observation, task_description)
