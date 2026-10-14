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


"""Local LIBERO and LIBERO-Plus closed-loop evaluation with HoloBrain 0.5.

Tasks are fanned out across GPUs: the launcher splits ``(suite, task)`` pairs
between workers and each worker runs a subprocess that loads the policy once
and sweeps its share. A MuJoCo segfault cannot be caught in Python and would
kill the whole run, so a worker relaunches with whatever has not yet written a
``results.json``.
"""

import argparse
import json
import logging
import multiprocessing
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import tqdm

from projects.holobrain_0_5.libero_runtime import prepare_libero_runtime
from robo_orchard_lab.utils import log_basic_config

__all__ = [
    "run_task",
    "build_task_list",
    "allocate_tasks_to_workers",
    "resolve_trials_per_task",
    "summarize_results",
    "main",
]
logger = logging.getLogger(__name__)

LIBERO_SUITES = [
    "libero_10",
    "libero_goal",
    "libero_object",
    "libero_spatial",
]
#: Trials per task, by benchmark. LIBERO-Plus already varies each task, so a
#: single trial per variation matches the internal sweep.
DEFAULT_TRIALS = {"libero": 50, "libero_plus": 1}
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
#: Relaunches without progress before a task is declared unevaluable.
MAX_STALLED_ATTEMPTS = 3
#: Perturbation axes LIBERO-Plus groups its tasks by, in report order.
LIBERO_PLUS_CATEGORY_ORDER = [
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
]


def resolve_trials_per_task(benchmark, requested):
    """Return the trial count, defaulting per benchmark when unset."""
    if requested is not None:
        return requested
    return DEFAULT_TRIALS[benchmark]


def build_task_list(suites, task_ids, task_counts):
    """Expand the selection into ``(suite, task_id)`` pairs.

    Args:
        suites (list[str]): Suite names to evaluate.
        task_ids (list[int] | None): Specific ids, or None for every task.
        task_counts (dict[str, int]): Task count per suite.

    Returns:
        list[tuple[str, int]]: Ordered evaluation targets.
    """
    tasks = []
    for suite in suites:
        if suite not in task_counts:
            raise ValueError(
                f"Unknown suite {suite!r}; choices: {sorted(task_counts)}"
            )
        count = task_counts[suite]
        selected = range(count) if task_ids is None else task_ids
        for task_id in selected:
            if not 0 <= task_id < count:
                raise ValueError(
                    f"Task id {task_id} is out of range for {suite}: "
                    f"[0, {count})"
                )
            tasks.append((suite, task_id))
    return tasks


def allocate_tasks_to_workers(tasks, num_gpus, processes_per_gpu=1):
    """Deal tasks round-robin to one worker slot per GPU process."""
    if num_gpus < 1:
        raise ValueError("num_gpus must be >= 1")
    if processes_per_gpu < 1:
        raise ValueError("processes_per_gpu must be >= 1")
    worker_gpu_ids = [
        gpu_id for gpu_id in range(num_gpus) for _ in range(processes_per_gpu)
    ][: len(tasks)]
    allocated = [(gpu_id, []) for gpu_id in worker_gpu_ids]
    for index, task in enumerate(tasks):
        allocated[index % len(allocated)][1].append(task)
    return allocated


def summarize_results(results, suites, detail=True):
    """Fold per-task success rates into per-suite and overall averages."""
    nested = {}
    for (suite, task_id), rate in results.items():
        nested.setdefault(suite, {})[f"task_{task_id}"] = rate
    summary = {}
    suite_means = []
    for suite in suites:
        if suite not in nested:
            continue
        rates = list(nested[suite].values())
        mean = sum(rates) / len(rates)
        summary[suite] = {
            "average_success_rate": mean,
            "num_tasks": len(rates),
        }
        suite_means.append(mean)
    summary["average_success_rate_across_suites"] = (
        sum(suite_means) / len(suite_means) if suite_means else 0.0
    )
    output = {"overall_summary": summary}
    if detail:
        output["suites_detail"] = nested
    return output


def load_plus_categories(benchmark_root):
    """Map ``(suite, task name)`` to its LIBERO-Plus perturbation category."""
    path = (
        Path(benchmark_root)
        / "libero/libero/benchmark/task_classification.json"
    )
    if not path.is_file():
        logger.warning("No LIBERO-Plus task classification at %s", path)
        return {}
    classification = json.loads(path.read_text(encoding="utf-8"))
    return {
        (suite, item["name"]): item["category"]
        for suite, items in classification.items()
        for item in items
    }


def summarize_plus_categories(results, task_names, benchmark_root):
    """Average success by perturbation axis, the point of LIBERO-Plus.

    Args:
        results (dict): ``(suite, task_id)`` to success rate.
        task_names (dict): ``(suite, task_id)`` to the task's name.
        benchmark_root (str): LIBERO-Plus checkout holding the classification.
    """
    category_by_task = load_plus_categories(benchmark_root)
    rates_by_category = {}
    for key, rate in results.items():
        name = task_names.get(key)
        category = category_by_task.get((key[0], name), "Unclassified")
        rates_by_category.setdefault(category, []).append(rate)

    ordered = [
        *LIBERO_PLUS_CATEGORY_ORDER,
        *sorted(
            category
            for category in rates_by_category
            if category not in LIBERO_PLUS_CATEGORY_ORDER
        ),
    ]
    detail = {}
    means = []
    all_rates = []
    for category in ordered:
        rates = rates_by_category.get(category)
        if not rates:
            continue
        mean = sum(rates) / len(rates)
        detail[category] = {
            "average_success_rate": mean,
            "num_tasks": len(rates),
        }
        means.append(mean)
        all_rates.extend(rates)
    return {
        "categories_detail": detail,
        "average_success_rate_across_categories": (
            sum(means) / len(means) if means else 0.0
        ),
        "average_success_rate_all_tasks": (
            sum(all_rates) / len(all_rates) if all_rates else 0.0
        ),
        "num_tasks": len(all_rates),
    }


def run_task(
    args,
    task_id,
    suite_name,
    task_suite,
    model,
    eval_func,
    reset_func,
    output_dir,
):
    """Evaluate one LIBERO task and close its environment on every exit."""
    import robosuite.utils.transform_utils as T  # noqa: N812
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
    )

    from projects.holobrain_0_5.libero_runtime import (
        get_libero_agentview_image,
        get_libero_dummy_action,
        get_libero_env,
        get_libero_wrist_image,
        save_rollout_video,
    )
    from robo_orchard_lab.dataset.libero.utils import transform_ee_rotations

    task_suite_name = suite_name
    num_trials_per_task = args.num_trials_per_task
    num_steps_wait = args.num_steps_wait
    save_video = args.save_video
    task = task_suite.get_task(task_id)
    with torch.serialization.safe_globals(
        [
            np.core.multiarray._reconstruct,
            np.ndarray,
            np.dtype,
            np.dtypes.Float64DType,
        ]
    ):
        initial_states = task_suite.get_task_init_states(task_id)
    (env, task_description) = get_libero_env(
        task,
        resolution=256,
    )
    try:
        (task_episodes, task_successes) = (0, 0)
        for episode_idx in tqdm.tqdm(range(num_trials_per_task)):
            logger.info(f"Task: {task_description}")
            for robot in env.env.robots:
                robot.controller.use_delta = True
            env.reset()
            reset_func(model)
            obs = env.set_init_state(initial_states[episode_idx])
            t = 0
            replay_images = []
            max_steps = MAX_STEPS.get(task_suite_name, 300)
            logger.info(f"Starting episode {task_episodes + 1}...")
            action_counter = 0
            while t < max_steps + num_steps_wait:
                if t < num_steps_wait:
                    (obs, reward, done, info) = env.step(
                        get_libero_dummy_action()
                    )
                    t += 1
                    continue
                (side_img, side_img_depth) = get_libero_agentview_image(
                    env, obs
                )
                side_extrinsic = get_camera_extrinsic_matrix(
                    env.sim, "agentview"
                )
                side_intrinsic = get_camera_intrinsic_matrix(
                    env.sim, "agentview", camera_height=256, camera_width=256
                )
                (wrist_img, wrist_img_depth) = get_libero_wrist_image(env, obs)
                wrist_extrinsic = get_camera_extrinsic_matrix(
                    env.sim, "robot0_eye_in_hand"
                )
                wrist_intrinsic = get_camera_intrinsic_matrix(
                    env.sim,
                    "robot0_eye_in_hand",
                    camera_height=256,
                    camera_width=256,
                )
                replay_images.append(side_img)
                current_width = (
                    obs["robot0_gripper_qpos"][0]
                    - obs["robot0_gripper_qpos"][1]
                )
                max_width = 0.08
                gripper_state = current_width / max_width
                r_diff = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
                ee_state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        T.quat2axisangle(obs["robot0_eef_quat"]),
                    )
                )[None].astype(np.float32)
                ee_quat = transform_ee_rotations(ee_state, r_diff)
                ee_state = np.concatenate([ee_state[:, :3], ee_quat], axis=1)
                robot_state = np.concatenate(
                    [gripper_state[None, None], ee_state], axis=1
                )
                observation = {
                    "observation": {
                        "agentview": {
                            "rgb": side_img,
                            "depth": side_img_depth,
                            "extrinsic_cv": side_extrinsic,
                            "intrinsic_cv": side_intrinsic,
                        },
                        "eye_in_hand": {
                            "rgb": wrist_img,
                            "depth": wrist_img_depth,
                            "extrinsic_cv": wrist_extrinsic,
                            "intrinsic_cv": wrist_intrinsic,
                        },
                    },
                    "robot_state": robot_state.astype(np.float32),
                }
                if action_counter == 0:
                    action = eval_func(model, observation, task_description)
                    action_counter = action.shape[0]
                for robot in env.env.robots:
                    robot.controller.use_delta = False
                step_action = action[-action_counter]
                step_action_quat = T.quat2axisangle(
                    step_action[4:8][[1, 2, 3, 0]]
                )
                step_arm_action = np.concatenate(
                    (step_action[1:4], step_action_quat)
                )
                step_action = np.concatenate(
                    (step_arm_action, step_action[:1])
                )
                (obs, _, done, _) = env.step(step_action.tolist())
                action_counter -= 1
                if done:
                    task_successes += 1
                    break
                t += 1
            task_episodes += 1
            if save_video:
                save_rollout_video(
                    output_dir,
                    replay_images,
                    task_episodes,
                    success=done,
                    task_description=task_description,
                    log_file=None,
                )
            logger.info(
                "Success: %s, success rate: %s/%s",
                done,
                task_successes,
                task_episodes,
            )
        logger.info("\n" + "=" * 25 + "TASK RESULTS" + "=" * 25)
        logger.info(
            "%s_task_%s success rate: %s/%s",
            task_suite_name,
            task_id,
            task_successes,
            task_episodes,
        )
        return task_successes, task_episodes
    finally:
        env.close()


def _benchmark_root(args):
    """Return the checkout backing the selected benchmark."""
    root = (
        args.libero_plus_root
        if args.benchmark == "libero_plus"
        else args.libero_root
    )
    if root is None:
        flag = (
            "--libero-plus-root"
            if args.benchmark == "libero_plus"
            else "--libero-root"
        )
        raise SystemExit(
            f"{flag} is required with --benchmark {args.benchmark}"
        )
    return root


def _result_path(args, suite, task_id):
    return Path(args.output) / suite / f"task_{task_id}" / "results.json"


def _read_success_rate(args, suite, task_id):
    """Return a finished task's success rate, or None if it has not run."""
    try:
        return json.loads(_result_path(args, suite, task_id).read_text())[
            "success_rate"
        ]
    except (OSError, ValueError, KeyError):
        return None


def _worker_command(args, suite, task_ids):
    command = [
        sys.executable,
        "-m",
        "projects.holobrain_0_5.libero_eval",
        "--worker",
        "--benchmark",
        args.benchmark,
        "--model-dir",
        args.model_dir,
        "--processor",
        args.processor,
        "--task-suites",
        suite,
        "--task-ids",
        *[str(task_id) for task_id in task_ids],
        "--num-trials-per-task",
        str(args.num_trials_per_task),
        "--num-steps-wait",
        str(args.num_steps_wait),
        "--output",
        args.output,
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    if args.libero_root is not None:
        command += ["--libero-root", args.libero_root]
    if args.libero_plus_root is not None:
        command += ["--libero-plus-root", args.libero_plus_root]
    if args.save_video:
        command.append("--save-video")
    return command


def _eval_suite(gpu_id, worker_id, suite, task_ids, args, results):
    """Sweep one suite in a subprocess, relaunching past fatal crashes.

    The subprocess loads the policy once and loops. A MuJoCo segfault takes
    the whole process down uncatchably, so relaunch with whatever has not
    produced a ``results.json`` yet and give up on a task that never
    progresses.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_dir = Path(args.output) / suite
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"worker_{worker_id}.log"

    pending = [
        task_id
        for task_id in task_ids
        if _read_success_rate(args, suite, task_id) is None
    ]
    stalled = 0
    while pending:
        logger.info(
            "Worker %s (gpu %s) evaluating %s: %s task(s) left, log: %s",
            worker_id,
            gpu_id,
            suite,
            len(pending),
            log_file,
        )
        with log_file.open("a", encoding="utf-8") as stream:
            completed = subprocess.run(
                _worker_command(args, suite, pending),
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        remaining = [
            task_id
            for task_id in pending
            if _read_success_rate(args, suite, task_id) is None
        ]
        made_progress = len(remaining) < len(pending)
        pending = remaining
        if completed.returncode == 0:
            break
        if made_progress:
            stalled = 0
            continue
        stalled += 1
        if stalled >= MAX_STALLED_ATTEMPTS:
            logger.error(
                "Skipping %s task_%s after %s crashes with no progress",
                suite,
                pending[0],
                stalled,
            )
            pending = pending[1:]
            stalled = 0

    for task_id in task_ids:
        rate = _read_success_rate(args, suite, task_id)
        if rate is None:
            logger.error("Failed to evaluate %s task_%s", suite, task_id)
        else:
            results[(suite, task_id)] = rate
            logger.info(
                "Finished %s task_%s, success rate: %s", suite, task_id, rate
            )


def eval_tasks(gpu_id, tasks, args, results=None, worker_id=0):
    """Run one worker's share, grouped so each suite starts one subprocess."""
    if results is None:
        results = {}
    by_suite = {}
    for suite, task_id in tasks:
        by_suite.setdefault(suite, []).append(task_id)
    for suite, task_ids in by_suite.items():
        _eval_suite(gpu_id, worker_id, suite, task_ids, args, results)
    return results


def _run_worker(args):
    """Evaluate this worker's tasks in-process and write per-task results."""
    prepare_libero_runtime(_benchmark_root(args))
    from accelerate.utils import set_seed
    from libero.libero import benchmark

    from projects.holobrain_0_5.policy.libero_policy import (
        HoloBrainPolicy,
        predict_action,
        reset_model,
    )

    model = HoloBrainPolicy(args.model_dir, args.processor, args.device)
    benchmark_dict = benchmark.get_benchmark_dict()
    for suite_name in args.task_suites:
        suite = benchmark_dict[suite_name]()
        for task_id in args.task_ids:
            if not 0 <= task_id < suite.n_tasks:
                raise ValueError(f"Invalid task id: {task_id}")
            set_seed(args.seed)
            output = Path(args.output) / suite_name / f"task_{task_id}"
            output.mkdir(parents=True, exist_ok=True)
            successes, episodes = run_task(
                args,
                task_id,
                suite_name,
                suite,
                model,
                predict_action,
                reset_model,
                str(output),
            )
            (output / "results.json").write_text(
                json.dumps(
                    dict(
                        successes=successes,
                        episodes=episodes,
                        success_rate=successes / episodes,
                    ),
                    indent=2,
                )
            )


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--libero-root",
        default=None,
        help="LIBERO checkout, required with --benchmark libero",
    )
    parser.add_argument(
        "--libero-plus-root",
        default=None,
        help="LIBERO-Plus checkout, required with --benchmark libero_plus",
    )
    parser.add_argument(
        "--benchmark", choices=sorted(DEFAULT_TRIALS), default="libero"
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--processor", default="libero_processor")
    parser.add_argument(
        "--task-suites", nargs="+", default=list(LIBERO_SUITES)
    )
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        default=None,
        help="Defaults to every task in each selected suite",
    )
    parser.add_argument(
        "--num-trials-per-task",
        type=int,
        default=None,
        help="Defaults to 50 for libero and 1 for libero_plus",
    )
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--processes-per-gpu", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected tasks and exit without evaluating",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Internal: evaluate the given tasks in this process",
    )
    args = parser.parse_args()
    args.num_trials_per_task = resolve_trials_per_task(
        args.benchmark, args.num_trials_per_task
    )
    if args.num_trials_per_task < 1:
        parser.error("--num-trials-per-task must be positive")
    return args


def main():
    """Fan selected tasks across GPUs and summarize their success rates."""
    args = _parse_args()
    log_basic_config(
        format="%rank %(asctime)s %(levelname)s %(filename)s:%(lineno)d | %(message)s",  # noqa: E501
        level=logging.INFO,
    )
    if args.worker:
        _run_worker(args)
        return

    prepare_libero_runtime(_benchmark_root(args))
    from libero.libero import benchmark

    logger.info("\n%s", json.dumps(vars(args), indent=4, default=str))
    benchmark_dict = benchmark.get_benchmark_dict()
    suites = {
        suite: benchmark_dict[suite]()
        for suite in args.task_suites
        if suite in benchmark_dict
    }
    counts = {suite: instance.n_tasks for suite, instance in suites.items()}
    tasks = build_task_list(args.task_suites, args.task_ids, counts)
    if args.dry_run:
        logger.info(
            "%s task(s), %s trial(s) each: %s",
            len(tasks),
            args.num_trials_per_task,
            tasks,
        )
        return

    num_gpus = args.num_gpus or torch.cuda.device_count()
    if num_gpus < 1:
        raise RuntimeError("No GPUs available for evaluation")
    allocated = allocate_tasks_to_workers(
        tasks, num_gpus, args.processes_per_gpu
    )
    logger.info(
        "Found %s GPU(s). Distributing %s tasks with %s process(es) per GPU",
        num_gpus,
        len(tasks),
        args.processes_per_gpu,
    )
    for worker_id, (gpu_id, gpu_tasks) in enumerate(allocated):
        logger.info(
            "worker %s (gpu %s): %s task(s)", worker_id, gpu_id, len(gpu_tasks)
        )

    started = datetime.now()
    if len(allocated) > 1:
        manager = multiprocessing.Manager()
        shared = manager.dict()
        workers = []
        for worker_id, (gpu_id, gpu_tasks) in enumerate(allocated):
            worker = multiprocessing.Process(
                target=eval_tasks,
                args=(gpu_id, gpu_tasks, args, shared, worker_id),
            )
            worker.start()
            workers.append(worker)
        for worker in workers:
            worker.join()
        results = dict(shared)
    else:
        results = eval_tasks(allocated[0][0], allocated[0][1], args)

    is_plus = args.benchmark == "libero_plus"
    # LIBERO-Plus has thousands of task variations per suite; the per-task
    # detail would drown the log, and the category breakdown is the point.
    summary = summarize_results(results, args.task_suites, detail=not is_plus)
    if is_plus:
        task_names = {
            (suite, task_id): suites[suite].get_task(task_id).name
            for suite, task_id in results
            if suite in suites
        }
        summary["category_summary"] = summarize_plus_categories(
            results, task_names, _benchmark_root(args)
        )
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    logger.info("=" * 50)
    logger.info("Evaluation finished in %s", datetime.now() - started)
    logger.info("Evaluated %s of %s task(s)", len(results), len(tasks))
    logger.info("Detailed JSON results:")
    logger.info("\n%s", json.dumps(summary, indent=4))
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
