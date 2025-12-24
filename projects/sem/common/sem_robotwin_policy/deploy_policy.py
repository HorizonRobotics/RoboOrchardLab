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
import tempfile

import numpy as np
import requests
import torch
from filelock import FileLock, Timeout
from robo_orchard_core.utils.config import load_config_class

from robo_orchard_lab.models.mixin import ModelMixin
from robo_orchard_lab.models.sem_modules.processor import (
    MultiArmManipulationInput,
)
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


def download_file(url, file_name, timeout=180):
    if os.path.exists(file_name):
        print(f"File existed: {file_name}")
        return

    lock_path = file_name + ".lock"
    lock = FileLock(lock_path, timeout=timeout)

    try:
        with lock:
            if os.path.exists(file_name):
                print(f"File existed: {file_name}")
                return

            temp_dir = os.path.dirname(file_name) or "."
            with tempfile.NamedTemporaryFile(
                delete=False, dir=temp_dir
            ) as tmp_file:
                tmp_path = tmp_file.name
                try:
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            tmp_file.write(chunk)
                    tmp_file.flush()
                    os.fsync(tmp_file.fileno())

                    os.rename(tmp_path, file_name)
                    print(f"Download success: {file_name}")
                except Exception as e:
                    print(f"Download fail: {file_name}, error: {e}")
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    raise
    except Timeout as e:
        print(f"Download timeout: {file_name}")
        raise e


def download_job_ckpt_processor(
    ckpt_url, processor_name, output_dir="./model", model_prefix="model"
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    while ckpt_url.endswith("/"):
        ckpt_url = ckpt_url[:-1]
    model_url = f"{ckpt_url}/model.safetensors"
    model_config_url = f"{ckpt_url}/{model_prefix}.config.json"
    processor_url = "/".join(
        ckpt_url.split("/")[:-2] + [f"{processor_name}.json"]
    )
    print(
        f"model_ckpt: {model_url}\n"
        f"model_config: {model_config_url}\n"
        f"procssor: {processor_url}"
    )
    for url in [model_url, model_config_url, processor_url]:
        file_name = os.path.join(output_dir, url.split("/")[-1])
        if url.endswith("config.json"):
            file_name = file_name.replace(
                f"{model_prefix}.config.json", "model.config.json"
            )
        download_file(url, file_name)


class SEMPolicy:
    def __init__(
        self,
        config,
        processor=None,
        vlm_ckpt_dir=None,
        urdf_dir=None,
        model_prefix="model",
    ):
        if processor is None:
            processor = "processor"
        if config.startswith("http"):
            download_job_ckpt_processor(
                ckpt_url=config,
                processor_name=processor,
                output_dir="./sem_eval_model",
                model_prefix=model_prefix,
            )
            config = "./sem_eval_model"
        logger.info(f"model config: {config}, processor: {processor}")

        target_vlm_ckpt_dir = os.path.join(config, "ckpt")
        target_urdf_dir = os.path.join(config, "urdf")
        if vlm_ckpt_dir is not None and not os.path.exists(
            target_vlm_ckpt_dir
        ):
            os.symlink(vlm_ckpt_dir, target_vlm_ckpt_dir)
        if urdf_dir is not None and not os.path.exists(target_urdf_dir):
            os.symlink(urdf_dir, target_urdf_dir)

        processor_cfg = load_config_class(
            open(os.path.join(config, f"{processor}.json")).read()
        )
        with in_cwd(config):
            self.processor = processor_cfg()

        self.model = ModelMixin.load_model(config, load_impl="native")
        self.model.eval()
        self.model.requires_grad_()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.take_action_cnt = 0

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

    def get_action(self, observation, instruction):
        data = self.data_preprocess(observation, instruction)
        model_outs = self.model(data)
        actions = self.processor.post_process(data, model_outs).action
        valid_action_step = 32
        actions = actions[:valid_action_step].cpu().numpy()
        return actions


def get_model(usr_args):  # from your deploy_policy.yml
    policy = SEMPolicy(
        usr_args["model_config"],
        usr_args["model_processor"],
        usr_args["vlm_ckpt_dir"],
        usr_args["urdf_dir"],
        usr_args["model_prefix"],
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
