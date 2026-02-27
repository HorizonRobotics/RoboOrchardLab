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


import importlib
import logging
import os

import numpy as np
import torch
from robo_orchard_core.utils.config import load_config_class
from scipy.spatial.transform import Rotation

from robo_orchard_lab.dataset.behavior import utils
from robo_orchard_lab.models.holobrain.processor import (
    MultiArmManipulationInput,
)
from robo_orchard_lab.models.mixin import ModelMixin
from robo_orchard_lab.utils import seed_everything
from robo_orchard_lab.utils.path import in_cwd

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

logger = logging.getLogger(__file__)


def load_config(config_file):
    assert config_file.endswith(".py")
    module_name = os.path.split(config_file)[-1][:-3]
    spec = importlib.util.spec_from_file_location(module_name, config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


class HoloBrainPolicy:
    def __init__(
        self,
        model_path,
        processor=None,
        model_prefix="model",
    ):
        seed_everything(2025)

        if processor is None:
            processor = "processor"

        if model_path.startswith("http"):
            model_path = "./holobrain_eval_model"
        logger.info(f"model path: {model_path}, processor: {processor}")

        processor_cfg = load_config_class(
            open(os.path.join(model_path, f"{processor}.json")).read()
        )
        with in_cwd(model_path):
            self.processor = processor_cfg()

        self.model = ModelMixin.load_model(
            model_path, model_prefix=model_prefix, strict=False
        )
        self.model.eval()
        self.model.requires_grad_(False)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.intrinsic = {
            "left_wrist": np.array(
                [
                    [388.6, 0.0, 240.0, 0.0],
                    [0.0, 388.6, 240.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            "right_wrist": np.array(
                [
                    [388.6, 0.0, 240.0, 0.0],
                    [0.0, 388.6, 240.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            "head": np.array(
                [
                    [306.0, 0.0, 360.0, 0.0],
                    [0.0, 306.0, 360.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        }

    def data_preprocess(self, obs):
        intrinsic = self.intrinsic
        extrinsic = {}
        for idx, cam in enumerate(utils.ROBOT_CAMERA_NAMES["R1Pro"]):
            cam2base = obs["robot_r1::cam_rel_poses"][7 * idx : 7 * idx + 7]
            pos, quat = cam2base[:3], cam2base[3:]

            # Add camera coordinate system adjustment:
            # 180 degree rotation around X-axis
            rot = Rotation.from_quat(quat).as_matrix()  # (T, 3, 3)
            rot_add = Rotation.from_euler("xyz", [np.pi, 0, 0]).as_matrix()
            rot_matrix = rot @ rot_add

            extr = np.eye(4, dtype=float)  # (4, 4)
            extr[:3, :3] = rot_matrix
            extr[:3, 3] = pos
            extrinsic[cam] = extr

        proprio = obs["robot_r1::proprio"]
        joint_state = np.hstack(
            [
                proprio[utils.PROPRIO_QPOS_INDICES["R1Pro"]["torso"]],
                proprio[utils.PROPRIO_QPOS_INDICES["R1Pro"]["left_arm"]],
                proprio[
                    utils.PROPRIO_QPOS_INDICES["R1Pro"]["left_gripper"]
                ].sum(axis=-1, keepdims=True),
                proprio[utils.PROPRIO_QPOS_INDICES["R1Pro"]["right_arm"]],
                proprio[
                    utils.PROPRIO_QPOS_INDICES["R1Pro"]["right_gripper"]
                ].sum(axis=-1, keepdims=True),
            ]
        )
        joint_state = joint_state.reshape(1, -1)

        # origin image is rgba ==> bgr
        rgb_left_wrist = obs[
            "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"
        ][..., :3][..., ::-1].astype(np.float32)
        rgb_right_wrist = obs[
            "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"
        ][..., :3][..., ::-1].astype(np.float32)
        rgb_head = obs["robot_r1::robot_r1:zed_link:Camera:0::rgb"][..., :3][
            ..., ::-1
        ].astype(np.float32)

        rgb_left_wrist = np.expand_dims(rgb_left_wrist, axis=0)
        rgb_right_wrist = np.expand_dims(rgb_right_wrist, axis=0)
        rgb_head = np.expand_dims(rgb_head, axis=0)

        # depth
        depth_left_wrist = obs[
            "robot_r1::robot_r1:left_realsense_link:Camera:0::depth_linear"
        ].astype(np.float32)
        depth_right_wrist = obs[
            "robot_r1::robot_r1:right_realsense_link:Camera:0::depth_linear"
        ].astype(np.float32)
        depth_head = obs[
            "robot_r1::robot_r1:zed_link:Camera:0::depth_linear"
        ].astype(np.float32)

        # depth maybe nan
        depth_left_wrist = np.nan_to_num(
            depth_left_wrist, nan=10, posinf=10, neginf=0
        )
        depth_right_wrist = np.nan_to_num(
            depth_right_wrist, nan=10, posinf=10, neginf=0
        )
        depth_head = np.nan_to_num(depth_head, nan=10, posinf=10, neginf=0)

        # depth ~ [0, 1]
        depth_left_wrist /= 10.0
        depth_right_wrist /= 10.0
        depth_head /= 10.0

        depth_left_wrist = np.expand_dims(depth_left_wrist, axis=0)
        depth_right_wrist = np.expand_dims(depth_right_wrist, axis=0)
        depth_head = np.expand_dims(depth_head, axis=0)

        instruction = obs["instruction"]

        data = MultiArmManipulationInput(
            image={
                "left_wrist": rgb_left_wrist,
                "right_wrist": rgb_right_wrist,
                "head": rgb_head,
            },
            depth={
                "left_wrist": depth_left_wrist,
                "right_wrist": depth_right_wrist,
                "head": depth_head,
            },
            intrinsic=intrinsic,
            t_world2cam=extrinsic,
            history_joint_state=joint_state,
            instruction=instruction,
            # remaining_actions=obs.get("remaining_actions", None),
            # remaining_trajs=obs.get("remaining_trajs", None),
            # delay_horizon=obs.get("delay_horizon", None),
        )
        data = self.processor.pre_process(data)

        return data

    def forward(self, obs):
        data = self.data_preprocess(obs)
        model_outs = self.model(data)
        output = self.processor.post_process(data, model_outs)

        action = output.action.cpu().numpy()

        mobile_traj = output.mobile_traj
        if mobile_traj is not None:
            mobile_traj = mobile_traj.cpu().numpy()

        return action, mobile_traj

    def reset(self) -> None:
        pass


def get_model(usr_args):  # from your deploy_policy.yml
    policy = HoloBrainPolicy(
        usr_args["model_config"],
        usr_args["model_processor"],
        usr_args["vlm_ckpt_dir"],
        usr_args["urdf_dir"],
        usr_args["model_prefix"],
    )
    return policy


def eval(task_env, model, obs):
    # TODO: excute action in env
    # actions = model.get_action(obs)
    pass


def reset_model(model):
    pass


if __name__ == "__main__":
    import yaml

    with open("deploy_policy.yml", "r") as f:
        deploy_config = yaml.safe_load(f)
    print(deploy_config)

    policy = HoloBrainPolicy(
        deploy_config["model_path"],
        deploy_config["model_processor"],
        deploy_config["vlm_ckpt_dir"],
        deploy_config["urdf_dir"],
        deploy_config["model_prefix"],
    )
