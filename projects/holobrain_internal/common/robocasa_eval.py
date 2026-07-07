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
import importlib.machinery
import json
import logging
import multiprocessing
import os
import pkgutil
import subprocess
import sys
import traceback
import types
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

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
ROBOCASA_MODEL_ASSET_DIRS = (
    "fixtures",
    "textures",
    "objects",
    "generative_textures",
)


def _find_package_dir(package_name: str) -> Path:
    package_parts = package_name.split(".")
    for search_path in sys.path:
        base_path = Path(search_path or os.getcwd())
        package_dir = base_path.joinpath(*package_parts)
        if (package_dir / "__init__.py").is_file():
            return package_dir
    raise ModuleNotFoundError(f"Cannot find package `{package_name}`.")


def _make_package_module(
    module_name: str,
    package_dir: Path,
) -> types.ModuleType:
    module = types.ModuleType(module_name)
    module.__file__ = str(package_dir / "__init__.py")
    module.__path__ = [str(package_dir)]
    module.__package__ = module_name
    module.__spec__ = importlib.machinery.ModuleSpec(
        module_name,
        loader=None,
        is_package=True,
    )
    return module


def _bootstrap_robocasa_assets_root(assets_root: str) -> None:
    """Set assets root before RoboCasa eager imports build registries."""
    registry_module = sys.modules.get(
        "robocasa.models.objects.kitchen_object_utils"
    )
    if registry_module is not None:
        registry_assets_root = str(
            Path(registry_module.BASE_ASSET_ZOO_PATH).parent
        )
        if Path(registry_assets_root) != Path(assets_root):
            raise RuntimeError(
                "RoboCasa object registry was imported before "
                f"--assets_root was applied. Registry assets root is "
                f"`{registry_assets_root}`, requested `{assets_root}`."
            )

    robocasa_module = sys.modules.get("robocasa")
    if robocasa_module is None:
        robocasa_dir = _find_package_dir("robocasa")
        robocasa_module = _make_package_module("robocasa", robocasa_dir)
        sys.modules["robocasa"] = robocasa_module
    else:
        robocasa_dir = Path(robocasa_module.__path__[0])

    models_module = sys.modules.get("robocasa.models")
    if models_module is None:
        models_dir = robocasa_dir / "models"
        models_module = _make_package_module("robocasa.models", models_dir)
        sys.modules["robocasa.models"] = models_module

    models_module.assets_root = assets_root
    robocasa_module.models = models_module


def prepare_robocasa_runtime(assets_root: str | None) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

    if assets_root is not None:
        _bootstrap_robocasa_assets_root(assets_root)
    else:
        import robocasa.models  # noqa: F401


def rewrite_robocasa_model_asset_paths(
    xml_str: str,
    assets_root: str,
) -> str:
    """Redirect RoboCasa kitchen XML assets to the configured assets root."""
    root = ET.fromstring(xml_str)
    asset = root.find("asset")
    if asset is None:
        return xml_str

    for elem in asset:
        normalized_path = (elem.get("file") or "").replace("\\", "/")
        _, marker, asset_rel_path = normalized_path.partition(
            "/models/assets/"
        )
        if (
            elem.tag in {"mesh", "texture"}
            and marker
            and asset_rel_path.startswith(ROBOCASA_MODEL_ASSET_DIRS)
        ):
            elem.set("file", str(Path(assets_root) / asset_rel_path))

    return ET.tostring(root, encoding="unicode")


def patch_robocasa_xml_asset_paths() -> None:
    """Make RoboCasa texture paths honor robocasa.models.assets_root."""
    import robocasa.environments.kitchen.kitchen as kitchen_module
    import robocasa.models

    kitchen_cls = kitchen_module.Kitchen
    if getattr(kitchen_cls, "_robo_orchard_asset_path_patch", False):
        return

    original_edit_model_xml = kitchen_cls.edit_model_xml

    def edit_model_xml(self: Any, xml_str: str) -> str:
        xml_str = original_edit_model_xml(self, xml_str)
        return rewrite_robocasa_model_asset_paths(
            xml_str,
            robocasa.models.assets_root,
        )

    kitchen_cls.edit_model_xml = edit_model_xml
    kitchen_cls._robo_orchard_asset_path_patch = True


def register_robocasa_gym_envs() -> None:
    import robocasa.environments.kitchen.atomic as atomic_tasks
    import robocasa.environments.kitchen.composite as composite_tasks

    for package in (atomic_tasks, composite_tasks):
        for module in pkgutil.walk_packages(
            package.__path__,
            package.__name__ + ".",
        ):
            importlib.import_module(module.name)

    patch_robocasa_xml_asset_paths()

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


def parse_video_cameras(video_camera: str | None) -> tuple[str, ...]:
    video_camera = (video_camera or "").strip()
    if not video_camera or video_camera.lower() == "default":
        return DEFAULT_VIDEO_CAMERAS

    cameras = tuple(
        dict.fromkeys(
            camera.strip()
            for camera in video_camera.split(",")
            if camera.strip()
        )
    )
    if not cameras:
        raise ValueError("--video_camera must contain at least one camera.")
    return cameras


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
        for camera_name in video_cameras:
            video_path = save_video_path
            if multi_camera:
                safe_camera_name = camera_name.replace("/", "_")
                safe_camera_name = safe_camera_name.replace(".", "_")
                video_path = save_video_path.with_name(
                    f"{save_video_path.stem}_{safe_camera_name}"
                    f"{save_video_path.suffix}"
                )
            writers[camera_name] = imageio.get_writer(str(video_path), fps=20)

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
                writer.append_data(
                    env.sim.render(
                        height=512,
                        width=768,
                        camera_name=camera_name,
                    )[::-1]
                )

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
        try:
            return _evaluate_task(
                task_name=task_name,
                policy=policy,
                args=args,
                task_dir=task_dir,
            )
        except Exception as exc:
            logger.exception("RoboCasa task %s failed.", task_name)
            return write_empty_task_result(
                task_name,
                args,
                task_dir,
                "failed",
                type(exc).__name__,
                str(exc),
                traceback.format_exc(),
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
            num_finished_trials = len(results)
            num_success_trials = sum(int(x["success"]) for x in results)
            current_success_rate = num_success_trials / num_finished_trials
            logger.info(
                "Finished task=%s trial=%d success=%s steps=%d "
                "current_success_rate=%.4f successes=%d/%d",
                task_name,
                trial_id,
                trial_result["success"],
                trial_result["steps"],
                current_success_rate,
                num_success_trials,
                num_finished_trials,
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
        "status": "completed",
        "split": args.split,
        "horizon": horizon,
        "num_trials": len(results),
        "success_rate": success_rate,
        "trials": results,
    }
    write_task_result(task_dir, summary)
    logger.info("Task %s success_rate=%.4f", task_name, success_rate)
    return summary


def write_missing_task_result(
    *,
    task_name: str,
    args: argparse.Namespace,
    task_dir: Path,
    reason: str,
) -> dict[str, Any]:
    return write_empty_task_result(
        task_name,
        args,
        task_dir,
        "missing",
        "MissingResult",
        reason,
    )


def write_empty_task_result(
    task_name: str,
    args: argparse.Namespace,
    task_dir: Path,
    status: str,
    error_type: str,
    error: str,
    traceback_text: str = "",
) -> dict[str, Any]:
    return write_task_result(
        task_dir,
        {
            "task_name": task_name,
            "status": status,
            "split": args.split,
            "horizon": args.horizon,
            "num_trials": 0,
            "success_rate": 0.0,
            "trials": [],
            "error_type": error_type,
            "error": error,
            "traceback": traceback_text,
        },
    )


def write_task_result(
    task_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    with open(task_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    return summary


def _record_task_result(
    task_results: list[dict[str, Any]],
    task_result: dict[str, Any],
    shared_results: Any | None,
    log_task_completion: bool,
    total_tasks: int,
) -> None:
    task_results.append(task_result)
    if shared_results is not None:
        shared_results[task_result["task_name"]] = task_result
    if log_task_completion:
        _log_main_task_result(
            task_result,
            completed_tasks=len(task_results),
            total_tasks=total_tasks,
        )


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


def _log_main_task_result(
    task_result: dict[str, Any],
    *,
    completed_tasks: int,
    total_tasks: int,
) -> None:
    status = task_result.get("status", "completed")
    trials = task_result.get("trials", [])
    num_trials = int(task_result.get("num_trials", len(trials)))
    num_success_trials = sum(int(x["success"]) for x in trials)
    log_fn = logger.warning if status in {"failed", "missing"} else logger.info
    log_fn(
        "Main process received task=%s status=%s success_rate=%.4f "
        "successes=%d/%d completed_tasks=%d/%d error=%s",
        task_result["task_name"],
        status,
        task_result["success_rate"],
        num_success_trials,
        num_trials,
        completed_tasks,
        total_tasks,
        task_result.get("error"),
    )


def _log_new_main_task_results(
    *,
    shared_results: Any,
    logged_task_names: set[str],
    task_names: list[str],
) -> None:
    for task_name in task_names:
        if task_name in logged_task_names or task_name not in shared_results:
            continue
        logged_task_names.add(task_name)
        _log_main_task_result(
            dict(shared_results[task_name]),
            completed_tasks=len(logged_task_names),
            total_tasks=len(task_names),
        )


def evaluate_task_group(
    *,
    gpu_id: str | None,
    task_names: list[str],
    args: argparse.Namespace,
    output_dir: Path,
    shared_results: Any | None = None,
    log_task_completion: bool = False,
    total_tasks: int | None = None,
) -> list[dict[str, Any]]:
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    try:
        prepare_robocasa_runtime(args.assets_root)
        register_robocasa_gym_envs()
        logger.info("Worker gpu=%s task_names=%s", gpu_id, task_names)
        policy = build_policy(args)
    except Exception as exc:
        task_results = []
        for task_name in task_names:
            task_dir = output_dir / task_name
            task_dir.mkdir(parents=True, exist_ok=True)
            with task_log_file(task_dir / "log.txt"):
                logger.exception(
                    "RoboCasa worker failed before evaluating task %s.",
                    task_name,
                )
                task_result = write_empty_task_result(
                    task_name,
                    args,
                    task_dir,
                    "failed",
                    type(exc).__name__,
                    str(exc),
                    traceback.format_exc(),
                )
            _record_task_result(
                task_results,
                task_result,
                shared_results,
                log_task_completion,
                total_tasks or len(task_names),
            )
        return task_results

    task_results = []
    for task_name in task_names:
        task_result = evaluate_task(
            task_name=task_name,
            policy=policy,
            args=args,
            output_dir=output_dir,
        )
        _record_task_result(
            task_results,
            task_result,
            shared_results,
            log_task_completion,
            total_tasks or len(task_names),
        )
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
            log_task_completion=True,
            total_tasks=len(task_names),
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

    logged_task_names = set()
    while any(process.is_alive() for process in processes):
        _log_new_main_task_results(
            shared_results=shared_results,
            logged_task_names=logged_task_names,
            task_names=task_names,
        )
        for process in processes:
            process.join(timeout=0.2)

    _log_new_main_task_results(
        shared_results=shared_results,
        logged_task_names=logged_task_names,
        task_names=task_names,
    )
    failed_workers = [
        process.exitcode for process in processes if process.exitcode != 0
    ]
    if failed_workers:
        logger.warning(
            "RoboCasa eval worker failed with exit code(s): %s. "
            "Returning available task results.",
            failed_workers,
        )

    result_by_task = dict(shared_results)
    missing_tasks = [x for x in task_names if x not in result_by_task]
    if missing_tasks:
        logger.warning(
            "RoboCasa eval finished without results for task(s): %s",
            missing_tasks,
        )
        reason = "Worker exited without reporting a result for this task."
        for task_name in missing_tasks:
            task_dir = output_dir / task_name
            task_dir.mkdir(parents=True, exist_ok=True)
            with open(task_dir / "log.txt", "a", encoding="utf-8") as f:
                f.write(f"{reason}\n")
            result_by_task[task_name] = write_missing_task_result(
                task_name=task_name,
                args=args,
                task_dir=task_dir,
                reason=reason,
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
    parser.add_argument("--num_trials_per_task", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--valid_action_step", type=int, default=32)
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
    failed_task_names = [
        result["task_name"]
        for result in task_results
        if result.get("status") == "failed"
    ]
    missing_task_names = [
        result["task_name"]
        for result in task_results
        if result.get("status") == "missing"
    ]
    completed_task_results = [
        result
        for result in task_results
        if result.get("status", "completed") == "completed"
    ]
    task_success_rates = {
        result["task_name"]: result["success_rate"] for result in task_results
    }
    mean_success_rate = (
        sum(x["success_rate"] for x in completed_task_results)
        / len(completed_task_results)
        if completed_task_results
        else 0.0
    )
    final_output = {
        "overall_summary": {
            "task_set": args.task_set,
            "split": args.split,
            "num_requested_tasks": len(task_names),
            "num_tasks": len(task_results),
            "num_completed_tasks": len(completed_task_results),
            "average_success_rate": mean_success_rate,
            "num_failed_tasks": len(failed_task_names),
            "num_missing_tasks": len(missing_task_names),
            "failed_task_names": failed_task_names,
            "missing_task_names": missing_task_names,
            "elapsed": str(datetime.now() - start_time),
            **task_success_rates,
        },
        "tasks_detail": task_results,
    }
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)
    logger.info("RoboCasa evaluation finished:")
    logger.info(json.dumps(final_output["overall_summary"], indent=4))


if __name__ == "__main__":
    main()
