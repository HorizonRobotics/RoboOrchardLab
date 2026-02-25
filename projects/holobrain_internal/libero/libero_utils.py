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

"""Utils for evaluating policies in LIBERO simulation environments."""

import os
import random
import time

import imageio
import numpy as np
import torch
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


def get_libero_env(task, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""  # noqa: E501
    task_description = task.language
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )  # noqa: E501
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_depths": True,
    }  # noqa: E501
    env = OffScreenRenderEnv(**env_args)
    env.seed(
        0
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state # noqa: E501
    return env, task_description


def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""  # noqa: E501
    return [0, 0, 0, 0, 0, 0, -1]


def depthimg2meters(env, depth):
    extent = env.sim.model.stat.extent
    near = env.sim.model.vis.map.znear * extent
    far = env.sim.model.vis.map.zfar * extent
    image = near / (1 - depth * (1 - near / far))
    return image


def get_libero_agentview_image(env, obs):
    """Extracts image from observations and preprocesses it."""
    img = obs["agentview_image"]
    depth = depthimg2meters(env, obs["agentview_depth"])
    img = img[::-1]
    depth = depth[::-1]
    return img, depth


def get_libero_wrist_image(env, obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["robot0_eye_in_hand_image"]
    depth = depthimg2meters(env, obs["robot0_eye_in_hand_depth"])
    img = img[::-1]
    depth = depth[::-1]
    return img, depth


def save_rollout_video(
    output_dir, rollout_images, idx, success, task_description, log_file=None
):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"{output_dir}/rollouts"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = (
        task_description.lower()
        .replace(" ", "_")
        .replace("\n", "_")
        .replace(".", "_")[:50]
    )  # noqa: E501
    mp4_path = f"{rollout_dir}/episode={idx}--success={success}--task={processed_task_description}.mp4"  # noqa: E501
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
