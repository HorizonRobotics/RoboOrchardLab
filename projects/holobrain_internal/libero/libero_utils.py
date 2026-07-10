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

import importlib
import os
import random
import re
import sys
import tempfile
import time
from pathlib import Path

import imageio
import numpy as np
import torch

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

SUPPORTED_BENCHMARKS = {"libero", "libero_plus"}


def resolve_benchmark_root(
    benchmark_name: str,
    benchmark_root: str | None = None,
) -> Path | None:
    """Resolve the local checkout root for LIBERO or LIBERO-Plus."""
    if benchmark_name not in SUPPORTED_BENCHMARKS:
        raise ValueError(
            f"Unsupported benchmark `{benchmark_name}`. "
            f"Available benchmarks: {sorted(SUPPORTED_BENCHMARKS)}"
        )
    if benchmark_root:
        return Path(benchmark_root).expanduser().resolve()

    env_name = (
        "LIBERO_PLUS_ROOT"
        if benchmark_name == "libero_plus"
        else "LIBERO_ROOT"
    )
    env_root = os.environ.get(env_name)
    if env_root:
        return Path(env_root).expanduser().resolve()

    default_dir = (
        "LIBERO-plus" if benchmark_name == "libero_plus" else "LIBERO"
    )
    for parent in [Path.cwd(), *Path.cwd().parents]:
        candidate = parent / default_dir
        if candidate.exists():
            return candidate.resolve()
    return None


def make_libero_config_path(benchmark_name: str, benchmark_root: Path) -> Path:
    """Create a non-interactive LIBERO config for a local checkout."""
    safe_root = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(benchmark_root))
    config_dir = (
        Path("/tmp")
        / "robo_orchard_libero_configs"
        / (f"{benchmark_name}_{safe_root}")
    )
    config_dir.mkdir(parents=True, exist_ok=True)
    benchmark_data_root = benchmark_root / "libero" / "libero"
    config_text = "\n".join(
        [
            f"assets: {benchmark_data_root / 'assets'}",
            f"bddl_files: {benchmark_data_root / 'bddl_files'}",
            f"benchmark_root: {benchmark_data_root}",
            f"datasets: {benchmark_root / 'libero' / 'datasets'}",
            f"init_states: {benchmark_data_root / 'init_files'}",
            "",
        ]
    )
    config_path = config_dir / "config.yaml"
    if config_path.exists():
        try:
            if config_path.read_text(encoding="utf-8") == config_text:
                return config_dir
        except OSError:
            pass

    fd, tmp_name = tempfile.mkstemp(
        dir=config_dir,
        prefix="config.yaml.",
        suffix=".tmp",
        text=True,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(config_text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, config_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return config_dir


def prepare_libero_runtime(
    benchmark_name: str = "libero",
    benchmark_root: str | None = None,
    *,
    force_config: bool = False,
) -> Path | None:
    """Prepare sys.path and LIBERO_CONFIG_PATH for a local LIBERO checkout."""
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/robo_orchard_matplotlib")
    root = resolve_benchmark_root(benchmark_name, benchmark_root)
    if root is None:
        return None
    if not root.exists():
        raise FileNotFoundError(
            f"{benchmark_name} root does not exist: {root}"
        )

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    if force_config or os.environ.get("LIBERO_CONFIG_PATH") is None:
        os.environ["LIBERO_CONFIG_PATH"] = str(
            make_libero_config_path(benchmark_name, root)
        )
    return root


def build_subprocess_env(
    benchmark_name: str,
    benchmark_root: str | None,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build env vars for an eval subprocess using one LIBERO checkout."""
    env = dict(os.environ if base_env is None else base_env)
    root = resolve_benchmark_root(benchmark_name, benchmark_root)
    if root is not None:
        existing_pythonpath = env.get("PYTHONPATH")
        pythonpath_parts = [str(root)]
        if existing_pythonpath:
            pythonpath_parts.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
        env["LIBERO_CONFIG_PATH"] = str(
            make_libero_config_path(benchmark_name, root)
        )
    env.setdefault("NUMBA_DISABLE_JIT", "1")
    env.setdefault("MPLCONFIGDIR", "/tmp/robo_orchard_matplotlib")
    return env


def get_benchmark_module(
    benchmark_name: str = "libero",
    benchmark_module: str | None = None,
    benchmark_root: str | None = None,
) -> object:
    """Return the benchmark module for LIBERO or LIBERO-Plus eval."""
    prepare_libero_runtime(
        benchmark_name,
        benchmark_root,
        force_config=benchmark_root is not None,
    )
    if benchmark_module is not None:
        return importlib.import_module(benchmark_module)
    return importlib.import_module("libero.libero.benchmark")


def get_libero_env(
    task,
    resolution=256,
    benchmark_name: str = "libero",
    benchmark_root: str | None = None,
):
    """Initializes and returns the LIBERO environment, along with the task description."""  # noqa: E501
    prepare_libero_runtime(
        benchmark_name,
        benchmark_root,
        force_config=benchmark_root is not None,
    )
    libero_pkg = importlib.import_module("libero.libero")
    env_module = importlib.import_module("libero.libero.envs")

    task_description = task.language
    task_bddl_file = os.path.join(
        libero_pkg.get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )  # noqa: E501
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_depths": True,
    }  # noqa: E501
    env = env_module.OffScreenRenderEnv(**env_args)
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
