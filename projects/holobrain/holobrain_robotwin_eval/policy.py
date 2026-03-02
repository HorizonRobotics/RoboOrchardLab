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

import logging
import os

import numpy as np
import torch

from robo_orchard_lab.models.holobrain.pipeline import (
    HoloBrainInferencePipeline,
)
from robo_orchard_lab.models.holobrain.processor import (
    MultiArmManipulationInput,
)

logger = logging.getLogger(__file__)


class HoloBrainRoboTwinPolicy:
    def __init__(
        self,
        config,
        vlm_ckpt_dir=None,
        urdf_dir=None,
        model_prefix="model",
        valid_action_step=32,
    ):
        logger.info(f"model config: {config}")

        target_vlm_ckpt_dir = os.path.join(config, "ckpt")
        target_urdf_dir = os.path.join(config, "urdf")
        if vlm_ckpt_dir is not None and not os.path.exists(
            target_vlm_ckpt_dir
        ):
            os.symlink(vlm_ckpt_dir, target_vlm_ckpt_dir)
        if urdf_dir is not None and not os.path.exists(target_urdf_dir):
            os.symlink(urdf_dir, target_urdf_dir)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = HoloBrainInferencePipeline.load_pipeline(
            directory=config,
            device=device,
            load_weights=True,
            load_impl="native",
            model_prefix=model_prefix,
        )
        self.pipeline.model.eval()
        self.valid_action_step = valid_action_step

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

        return MultiArmManipulationInput(
            image=images,
            depth=depths,
            intrinsic=intrinsic,
            t_world2cam=t_world2cam,
            history_joint_state=joint_state,
            instruction=instruction,
        )

    def get_action(self, observation, instruction):
        data = self.data_preprocess(observation, instruction)
        output = self.pipeline(data)
        actions = output.action[: self.valid_action_step].cpu().numpy()
        return actions


def get_model(usr_args):  # from your deploy_policy.yml
    policy = HoloBrainRoboTwinPolicy(
        usr_args["model_config"],
        usr_args.get("vlm_ckpt_dir"),
        usr_args.get("urdf_dir"),
        usr_args.get("model_prefix", "model"),
        usr_args.get("valid_action_step", 32),
    )
    return policy


def eval(task_env, model, observation):
    instruction = task_env.get_instruction()
    actions = model.get_action(observation, instruction)

    for action in actions:  # Execute each step of the action
        observation = task_env.get_obs()
        task_env.take_action(action)


def reset_model(model):
    pass
