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
import importlib
import os

import numpy as np
import torch

from robo_orchard_lab.utils.build import build

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def load_config(config_file):
    assert config_file.endswith(".py")
    module_name = os.path.split(config_file)[-1][:-3]
    spec = importlib.util.spec_from_file_location(module_name, config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


class SEMPolicy:
    def __init__(self, ckpt_path, config_file="config_sem_robotwin.py"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        ckpt_path = os.path.join(os.path.dirname(__file__), ckpt_path)
        self.ckpt_path = ckpt_path
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), config_file)
        self.cfg = load_config(config_path)

        # Build transforms (only validation needed for deployment)
        _, self.transforms = self._build_transforms()
        if self.transforms is not None:
            self.transforms = [build(x) for x in self.transforms]
        # Build model
        self.model = self._build_model()

        self._load_checkpoint()
        self.model.eval()
        self.model.to(self.device)
        self.take_action_cnt = 0

    def _build_transforms(self):
        """Builds data transforms using the function from the config file."""
        if hasattr(self.cfg, "build_transforms"):
            print("Building transforms...")
            train_transforms, val_transforms = self.cfg.build_transforms(
                self.cfg.config
            )
            return (
                train_transforms,
                val_transforms,
            )  # Keep both for potential inverse transforms later
        else:
            print(
                "Warning: Config file does not contain a ",
                "build_transforms' function.",
            )
            return None, None

    def _build_model(self):
        """Builds the model using the function from the config file."""
        if hasattr(self.cfg, "build_model"):
            print("Building model...")
            model = self.cfg.build_model(self.cfg.config)
            return model
        else:
            raise AttributeError(
                "Config file must contain a 'build_model' function."
            )

    def _load_checkpoint(self):
        """Loads the model checkpoint."""
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint file not found: {self.ckpt_path}"
            )
        print(f"Loading checkpoint from: {self.ckpt_path}")
        try:
            from safetensors.torch import load_model as load_safetensors

            load_safetensors(
                self.model, self.ckpt_path, device=str(self.device)
            )
        except ImportError:
            print("safetensors not found, using torch.load.")
            state_dict = torch.load(self.ckpt_path, map_location=self.device)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            missing_keys, unexpected_keys = self.model.load_state_dict(
                state_dict, strict=False
            )
            print(
                f"num of missing_keys: {len(missing_keys)},"
                f"num of unexpected_keys: {len(unexpected_keys)}"
            )
            print(
                f"missing_keys:\n {missing_keys}\n"
                f"unexpected_keys:\n {unexpected_keys}"
            )
        print("Checkpoint loaded successfully.")

    def get_action(self, data):
        actions = self.model(data)[0]["pred_actions"][0]
        valid_action_step = 32
        actions = actions[:valid_action_step, :, 0].cpu().numpy()
        return actions


def encode_obs(observation, data_transforms, instruction):
    images = []
    depths = []
    t_world2cam = []
    intrinsics = []
    joint_state = []
    for _, camera_data in observation["observation"].items():
        images.append(camera_data["rgb"])
        depths.append(camera_data["depth"] / 1000)

        _ext = np.eye(4, dtype=np.float32)
        _ext[:3] = camera_data["extrinsic_cv"]
        t_world2cam.append(_ext)

        _int = np.eye(4, dtype=np.float32)
        _int[:3, :3] = camera_data["intrinsic_cv"]
        intrinsics.append(_int)

    joint_action = observation["joint_action"]
    joint_action = (
        joint_action["left_arm"]
        + [joint_action["left_gripper"]]
        + joint_action["right_arm"]
        + [joint_action["right_gripper"]]
    )

    joint_state.append(joint_action)

    t_world2cam = np.stack(t_world2cam, dtype=np.float32)
    default_t_base2world = np.array(
        [
            [0, -1, 0, 0],
            [1, 0, 0, -0.65],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    data = dict(
        imgs=np.stack(images),
        depths=np.stack(depths),
        joint_state=np.stack(copy.deepcopy(joint_state), dtype=np.float32),
        intrinsic=np.stack(intrinsics, dtype=np.float32),
        T_world2cam=t_world2cam,
        T_base2world=default_t_base2world,
        text=instruction,
        step_index=len(joint_state) - 1,
    )

    # apply data transform
    for transform in data_transforms:
        if transform is None:
            continue
        data = transform(data)
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v[None]
        else:
            data[k] = [v]
    return data


def get_model(usr_args):  # from your deploy_policy.yml
    policy = SEMPolicy(usr_args["ckpt_path"], usr_args["config_file"])
    return policy


def eval(task_env, model, observation):
    instruction = task_env.get_instruction()
    data = encode_obs(
        observation, model.transforms, instruction
    )  # Post-Process Observation
    actions = model.get_action(data)

    for action in actions:  # Execute each step of the action
        observation = task_env.get_obs()
        task_env.take_action(action)


def reset_model(model):
    pass
