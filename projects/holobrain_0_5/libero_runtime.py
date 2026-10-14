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


"""Configure a local standard LIBERO checkout and its RGB-D renderer."""

import os
import sys
import tempfile
from pathlib import Path

import imageio
import yaml

__all__ = [
    "prepare_libero_runtime",
    "get_libero_env",
    "depthimg2meters",
    "get_libero_dummy_action",
    "get_libero_agentview_image",
    "get_libero_wrist_image",
    "save_rollout_video",
]


def prepare_libero_runtime(root):
    """Use the specified checkout without changing the user's LIBERO config."""
    root = Path(root).expanduser().resolve(strict=True)
    data = root / "libero/libero"
    if not (data / "assets").is_dir():
        raise FileNotFoundError(f"LIBERO assets missing under {data}")
    sys.path.insert(0, str(root))
    config_dir = Path(tempfile.mkdtemp(prefix="holobrain_libero_"))
    config = dict(
        assets=str(data / "assets"),
        bddl_files=str(data / "bddl_files"),
        benchmark_root=str(data),
        datasets=str(root / "libero/datasets"),
        init_states=str(data / "init_files"),
    )
    (config_dir / "config.yaml").write_text(yaml.safe_dump(config))
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    return root


def get_libero_env(task, resolution=256):
    """Create an RGB-D offscreen environment; the caller must close it."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=str(
            Path(get_libero_path("bddl_files"))
            / task.problem_folder
            / task.bddl_file
        ),
        camera_heights=resolution,
        camera_widths=resolution,
        camera_depths=True,
    )
    env.seed(0)
    return env, task.language


def get_libero_dummy_action():
    """Return a neutral delta action while objects settle."""
    return [0, 0, 0, 0, 0, 0, -1]


def depthimg2meters(env, depth):
    """Convert MuJoCo normalized depth to camera distance in meters."""
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
    return (img, depth)


def get_libero_wrist_image(env, obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["robot0_eye_in_hand_image"]
    depth = depthimg2meters(env, obs["robot0_eye_in_hand_depth"])
    img = img[::-1]
    depth = depth[::-1]
    return (img, depth)


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
    )
    mp4_path = (
        f"{rollout_dir}/episode={idx}--success={success}"
        f"--task={processed_task_description}.mp4"
    )
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path
