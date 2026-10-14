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

from projects.holobrain_0_5.utils import resolve_processor_dir
from robo_orchard_lab.models.holobrain_0_5.processor import (
    HoloBrainProcessor,
    MultiArmManipulationInput,
)
from robo_orchard_lab.models.mixin import ModelMixin

__all__ = ["HoloBrainPolicy", "get_model", "reset_model", "eval"]

logger = logging.getLogger(__name__)


class HoloBrainPolicy:
    """Local HoloBrain 0.5 robotwin observation and action adapter."""

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
            depths[cam_name] = [camera_data["depth"] / 1000]
            _tmp = np.eye(4)
            _tmp[:3] = camera_data["extrinsic_cv"]
            t_world2cam[cam_name] = _tmp
            _tmp = np.eye(4)
            _tmp[:3, :3] = camera_data["intrinsic_cv"]
            intrinsic[cam_name] = _tmp
        joint_state = []
        joint_action = obs["joint_action"]
        joint_action = (
            joint_action["left_arm"]
            + [joint_action["left_gripper"]]
            + joint_action["right_arm"]
            + [joint_action["right_gripper"]]
        )
        joint_state.append(joint_action)
        data = MultiArmManipulationInput(
            image=images,
            depth=depths,
            intrinsic=intrinsic,
            t_world2cam=t_world2cam,
            history_joint_state=joint_state,
            instruction=instruction,
        )
        data = self.processor.pre_process(data)
        return data

    @torch.inference_mode()
    def get_action(self, observation, instruction):
        data = self.data_preprocess(observation, instruction)
        model_outs = self.model(data)
        actions = self.processor.post_process(model_outs, data).action
        valid_action_step = 32
        actions = actions[:valid_action_step].cpu().numpy()
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


def eval(task_env, model, observation):
    """Execute up to 32 predicted joint actions in the RoboTwin environment."""
    for action in model.get_action(observation, task_env.get_instruction()):
        task_env.take_action(action)
        if (
            task_env.eval_success
            or task_env.take_action_cnt >= task_env.step_lim
        ):
            break
