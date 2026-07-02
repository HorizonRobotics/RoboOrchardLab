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

from __future__ import annotations
import argparse
import contextlib
import importlib
import json
import logging
import multiprocessing
import os
import pkgutil
import subprocess
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from holobrain_robocasa_policy import (
    ROBOCASA_CAMERAS,
    HoloBrainRoboCasaPolicy,
    HoloBrainRoboCasaPolicyCfg,
)

from robo_orchard_lab.utils import log_basic_config

logger = logging.getLogger(__file__)
DEFAULT_VIDEO_CAMERAS = ("robot0_agentview_center", *ROBOCASA_CAMERAS)
LOG_FORMAT = "%(asctime)s %(levelname)s %(filename)s:%(lineno)d | %(message)s"


def prepare_robocasa_runtime(assets_root: str) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

    import robocasa.models

    if assets_root is not None:
        robocasa.models.assets_root = assets_root


def register_robocasa_gym_envs() -> None:
    import robocasa.environments.kitchen.atomic as atomic_tasks
    import robocasa.environments.kitchen.composite as composite_tasks

    for package in (atomic_tasks, composite_tasks):
        for module in pkgutil.walk_packages(
            package.__path__,
            package.__name__ + ".",
        ):
            importlib.import_module(module.name)

    import robocasa.wrappers.gym_wrapper  # noqa: F401


def resolve_task_names(
    *,
    task_set: str,
    task_names: str | None,
) -> list[str]:
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY

    if task_names:
        return [x.strip() for x in task_names.split(",") if x.strip()]
    if task_set not in TASK_SET_REGISTRY:
        raise ValueError(
            f"Unknown RoboCasa task_set `{task_set}`. "
            f"Available: {sorted(TASK_SET_REGISTRY)}"
        )
    return list(TASK_SET_REGISTRY[task_set])


def make_env(
    task_name: str,
    args: argparse.Namespace,
    seed: int,
) -> Any:
    import gymnasium as gym

    return gym.make(
        f"robocasa/{task_name}",
        split=args.split,
        seed=seed,
        camera_widths=args.camera_width,
        camera_heights=args.camera_height,
        enable_render=True,
    )


def configure_robocasa_env_absolute_action(env: Any) -> None:
    """Configure RoboCasa PandaOmron OSC for base-frame absolute poses."""
    inner_env = getattr(env.unwrapped, "env", env.unwrapped)
    composite_controller = inner_env.robots[0].composite_controller
    right_arm_controller = composite_controller.get_controller("right")
    if right_arm_controller.name_suffix != "POSE":
        raise ValueError(
            "RoboCasa absolute action eval expects the right arm controller "
            f"to control pose, got suffix {right_arm_controller.name_suffix}."
        )
    right_arm_controller.input_type = "absolute"
    right_arm_controller.input_ref_frame = "base"
    right_arm_controller.input_min = np.full(
        right_arm_controller.control_dim,
        -np.inf,
        dtype=np.float64,
    )
    right_arm_controller.input_max = np.full(
        right_arm_controller.control_dim,
        np.inf,
        dtype=np.float64,
    )


def save_video_frame(
    writer: Any,
    env: Any,
    camera_name: str,
) -> None:
    frame = env.sim.render(
        height=512,
        width=768,
        camera_name=camera_name,
    )[::-1]
    writer.append_data(frame)


def parse_video_cameras(video_camera: str | None) -> tuple[str, ...]:
    if video_camera is None:
        return DEFAULT_VIDEO_CAMERAS
    video_camera = video_camera.strip()
    if not video_camera:
        return DEFAULT_VIDEO_CAMERAS
    if video_camera.lower() == "default":
        return DEFAULT_VIDEO_CAMERAS

    cameras = []
    seen = set()
    for camera in video_camera.split(","):
        camera = camera.strip()
        if camera and camera not in seen:
            cameras.append(camera)
            seen.add(camera)
    if not cameras:
        raise ValueError("--video_camera must contain at least one camera.")
    return tuple(cameras)


def video_path_for_camera(
    save_video_path: Path,
    camera_name: str,
    *,
    multi_camera: bool,
) -> Path:
    if not multi_camera:
        return save_video_path
    safe_camera_name = camera_name.replace("/", "_").replace(".", "_")
    return save_video_path.with_name(
        f"{save_video_path.stem}_{safe_camera_name}{save_video_path.suffix}"
    )


@contextlib.contextmanager
def task_log_file(log_file: Path) -> Any:
    """Route logs and plain stdout/stderr emitted during one task to a file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    if root_logger.getEffectiveLevel() > logging.INFO:
        root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    try:
        with (
            contextlib.redirect_stdout(file_handler.stream),
            contextlib.redirect_stderr(file_handler.stream),
        ):
            yield
    finally:
        root_logger.removeHandler(file_handler)
        root_logger.setLevel(previous_level)
        file_handler.close()


def run_trial(
    *,
    env: Any,
    policy: HoloBrainRoboCasaPolicy,
    horizon: int,
    seed: int,
    save_video_path: Path | None = None,
    video_cameras: tuple[str, ...] = DEFAULT_VIDEO_CAMERAS,
) -> dict[str, Any]:
    writers = {}
    if save_video_path is not None:
        import imageio

        save_video_path.parent.mkdir(parents=True, exist_ok=True)
        multi_camera = len(video_cameras) > 1
        writers = {
            camera_name: imageio.get_writer(
                str(
                    video_path_for_camera(
                        save_video_path,
                        camera_name,
                        multi_camera=multi_camera,
                    )
                ),
                fps=20,
            )
            for camera_name in video_cameras
        }

    try:
        obs, info = env.reset(seed=seed)
        configure_robocasa_env_absolute_action(env)
        action_queue: deque[dict[str, np.ndarray]] = deque()
        success = bool(info.get("success", False))
        steps_executed = 0

        for _ in range(horizon):
            configure_robocasa_env_absolute_action(env)
            if not action_queue:
                action_queue.extend(policy.get_action_dicts(obs, env=env))
            action = action_queue.popleft()
            obs, reward, terminated, truncated, info = env.step(action)
            steps_executed += 1

            for camera_name, writer in writers.items():
                save_video_frame(writer, env, camera_name)

            if steps_executed % 10 == 0:
                logger.info(f"Eval: {steps_executed} / {horizon}")

            success = bool(info.get("success", False) or reward > 0)
            if success or terminated or truncated:
                break

        return {"success": success, "steps": steps_executed}
    finally:
        for writer in writers.values():
            writer.close()


def evaluate_task(
    *,
    task_name: str,
    policy: HoloBrainRoboCasaPolicy,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    task_dir = output_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    with task_log_file(task_dir / "log.txt"):
        return _evaluate_task(
            task_name=task_name,
            policy=policy,
            args=args,
            task_dir=task_dir,
        )


def _evaluate_task(
    *,
    task_name: str,
    policy: HoloBrainRoboCasaPolicy,
    args: argparse.Namespace,
    task_dir: Path,
) -> dict[str, Any]:
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    horizon = args.horizon or get_task_horizon(task_name)
    results = []
    logger.info(
        "Start RoboCasa task=%s split=%s trials=%d horizon=%d",
        task_name,
        args.split,
        args.num_trials_per_task,
        horizon,
    )

    env = make_env(task_name, args, seed=args.seed)
    try:
        video_cameras = parse_video_cameras(args.video_camera)
        for trial_id in range(args.num_trials_per_task):
            seed = args.seed + trial_id
            video_path = (
                task_dir / f"trial_{trial_id:03d}.mp4"
                if args.save_video
                else None
            )
            trial_result = run_trial(
                env=env,
                policy=policy,
                horizon=horizon,
                seed=seed,
                save_video_path=video_path,
                video_cameras=video_cameras,
            )
            trial_result["seed"] = seed
            results.append(trial_result)
            logger.info(
                "Finished task=%s trial=%d success=%s steps=%d",
                task_name,
                trial_id,
                trial_result["success"],
                trial_result["steps"],
            )
    finally:
        env.close()

    success_rate = (
        sum(float(x["success"]) for x in results) / len(results)
        if results
        else 0.0
    )
    summary = {
        "task_name": task_name,
        "split": args.split,
        "horizon": horizon,
        "num_trials": len(results),
        "success_rate": success_rate,
        "trials": results,
    }
    with open(task_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    logger.info("Task %s success_rate=%.4f", task_name, success_rate)
    return summary


def build_policy(args: argparse.Namespace) -> HoloBrainRoboCasaPolicy:
    cfg = HoloBrainRoboCasaPolicyCfg(
        model_dir=args.model_config,
        model_processor=args.model_processor,
        model_prefix=args.model_prefix,
        load_impl=args.load_impl,
        vlm_ckpt_dir=args.vlm_ckpt_dir,
        urdf_dir=args.urdf_dir,
        valid_action_step=args.valid_action_step,
        use_env_calibration=not args.disable_env_calibration,
    )
    return HoloBrainRoboCasaPolicy(cfg=cfg)


def get_available_gpu_ids() -> list[str]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        return [
            gpu_id.strip()
            for gpu_id in visible_devices.split(",")
            if gpu_id.strip()
        ]

    try:
        ret = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        ret = None
    if ret is not None and ret.returncode == 0:
        gpu_ids = [
            line.strip() for line in ret.stdout.splitlines() if line.strip()
        ]
        if gpu_ids:
            return gpu_ids

    try:
        import torch
    except ImportError:
        logger.warning("PyTorch is unavailable; running RoboCasa eval on CPU.")
        return []

    num_gpus = torch.cuda.device_count()
    return [str(gpu_id) for gpu_id in range(num_gpus)]


def allocate_tasks_to_gpus(
    task_names: list[str],
    gpu_ids: list[str],
) -> list[tuple[str | None, list[str]]]:
    if not task_names:
        return []
    if not gpu_ids:
        return [(None, list(task_names))]

    num_workers = min(len(gpu_ids), len(task_names))
    task_groups = [(gpu_ids[i], []) for i in range(num_workers)]
    for index, task_name in enumerate(task_names):
        task_groups[index % num_workers][1].append(task_name)
    return task_groups


def evaluate_task_group(
    *,
    gpu_id: str | None,
    task_names: list[str],
    args: argparse.Namespace,
    output_dir: Path,
    shared_results: Any | None = None,
) -> list[dict[str, Any]]:
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    prepare_robocasa_runtime(args.assets_root)
    register_robocasa_gym_envs()
    logger.info("Worker gpu=%s task_names=%s", gpu_id, task_names)
    policy = build_policy(args)

    task_results = []
    for task_name in task_names:
        task_result = evaluate_task(
            task_name=task_name,
            policy=policy,
            args=args,
            output_dir=output_dir,
        )
        task_results.append(task_result)
        if shared_results is not None:
            shared_results[task_name] = task_result
    return task_results


def run_evaluation(
    *,
    task_names: list[str],
    gpu_ids: list[str],
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict[str, Any]]:
    task_groups = allocate_tasks_to_gpus(task_names, gpu_ids)
    logger.info(
        "Found %d GPU(s). Distributing %d task(s) across %d worker(s).",
        len(gpu_ids),
        len(task_names),
        len(task_groups),
    )
    for gpu_id, gpu_task_names in task_groups:
        logger.info("gpu=%s task_names=%s", gpu_id, gpu_task_names)

    if len(task_groups) <= 1:
        if not task_groups:
            return []
        gpu_id, gpu_task_names = task_groups[0]
        return evaluate_task_group(
            gpu_id=gpu_id,
            task_names=gpu_task_names,
            args=args,
            output_dir=output_dir,
        )

    ctx = multiprocessing.get_context("spawn")
    manager = ctx.Manager()
    shared_results = manager.dict()
    processes = []
    for gpu_id, gpu_task_names in task_groups:
        process = ctx.Process(
            target=evaluate_task_group,
            kwargs={
                "gpu_id": gpu_id,
                "task_names": gpu_task_names,
                "args": args,
                "output_dir": output_dir,
                "shared_results": shared_results,
            },
        )
        process.start()
        processes.append(process)

    failed_workers = []
    for process in processes:
        process.join()
        if process.exitcode != 0:
            failed_workers.append(process.exitcode)
    if failed_workers:
        raise RuntimeError(
            f"RoboCasa eval worker failed with exit code(s): {failed_workers}"
        )

    result_by_task = dict(shared_results)
    missing_tasks = [
        task_name
        for task_name in task_names
        if task_name not in result_by_task
    ]
    if missing_tasks:
        raise RuntimeError(
            f"RoboCasa eval finished without results for: {missing_tasks}"
        )
    return [result_by_task[task_name] for task_name in task_names]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument(
        "--model_processor",
        type=str,
        default="robocasa_processor",
    )
    parser.add_argument("--model_prefix", type=str, default="model")
    parser.add_argument("--load_impl", type=str, default="native")
    parser.add_argument("--vlm_ckpt_dir", type=str, default=None)
    parser.add_argument("--urdf_dir", type=str, default=None)
    parser.add_argument("--task_set", type=str, default="target50")
    parser.add_argument("--task_names", type=str, default=None)
    parser.add_argument("--split", type=str, default="target")
    parser.add_argument("--num_trials_per_task", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--valid_action_step", type=int, default=8)
    parser.add_argument("--camera_height", type=int, default=256)
    parser.add_argument("--camera_width", type=int, default=256)
    parser.add_argument("--assets_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument(
        "--video_camera",
        type=str,
        default="default",
        help=(
            "Comma-separated camera names to save. Use `default` for "
            f"{','.join(DEFAULT_VIDEO_CAMERAS)}."
        ),
    )
    parser.add_argument("--disable_env_calibration", action="store_true")
    return parser.parse_args()


def main() -> None:
    log_basic_config(
        format=LOG_FORMAT,
        level=logging.INFO,
    )
    args = parse_args()
    logger.info("\n%s", json.dumps(vars(args), indent=4))

    prepare_robocasa_runtime(args.assets_root)
    register_robocasa_gym_envs()
    task_names = resolve_task_names(
        task_set=args.task_set,
        task_names=args.task_names,
    )
    output_dir = Path(
        args.output_dir
        or os.environ.get("ROBOCASA_EVAL_OUTPUT_DIR", "eval_result/robocasa")
    )
    logger.info(f"output_dir: {output_dir}, task_names: {task_names}")
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    gpu_ids = get_available_gpu_ids()
    task_results = run_evaluation(
        task_names=task_names,
        gpu_ids=gpu_ids,
        args=args,
        output_dir=output_dir,
    )
    mean_success_rate = (
        sum(x["success_rate"] for x in task_results) / len(task_results)
        if task_results
        else 0.0
    )
    final_output = {
        "overall_summary": {
            "task_set": args.task_set,
            "split": args.split,
            "num_tasks": len(task_results),
            "average_success_rate": mean_success_rate,
            "elapsed": str(datetime.now() - start_time),
        },
        "tasks_detail": {
            result["task_name"]: result for result in task_results
        },
    }
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)
    logger.info("RoboCasa evaluation finished:")
    logger.info(json.dumps(final_output["overall_summary"], indent=4))


if __name__ == "__main__":
    main()
