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


"""Run the upstream RoboTwin evaluator against a local HoloBrain 0.5 export."""

import argparse
import json
import logging
import multiprocessing
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

from robo_orchard_lab.utils import log_basic_config

__all__ = [
    "evaluate",
    "eval_tasks",
    "allocate_tasks_to_workers",
    "list_robotwin_tasks",
    "select_tasks",
    "summarize_results",
    "main",
]

logger = logging.getLogger(__name__)

#: Tasks the expert planner practically cannot solve on a given embodiment. It
#: rejects seed after seed, so they burn the whole seed budget and either never
#: finish or report a rate measured over a handful of episodes. Skip them
#: outright rather than pay hours for a number that gets discarded anyway.
EXCLUDED_TASKS_BY_TASK_CONFIG = {
    "demo_clean_ur5-wsg": (
        "put_bottles_dustbin",
        "handover_block",
        "put_object_cabinet",
    ),
}


def list_robotwin_tasks(robotwin_root):
    """Return every task the RoboTwin checkout defines an environment for."""
    env_dir = Path(robotwin_root).expanduser().resolve(strict=True) / "envs"
    return sorted(
        path.stem
        for path in env_dir.glob("*.py")
        if not path.name.startswith("_")
    )


def select_tasks(available, requested, task_config):
    """Narrow ``requested`` to tasks this checkout and embodiment can run.

    ``requested`` of ``None`` means every available task. Unknown names are
    warned about rather than raised on, so one stale name in a long sweep does
    not cost the whole run.
    """
    excluded = set(EXCLUDED_TASKS_BY_TASK_CONFIG.get(task_config, ()))
    available = set(available)
    selected = []
    for task in sorted(available) if requested is None else requested:
        if task not in available:
            logger.warning("Task[%s] is not defined by this checkout", task)
            continue
        if task in excluded:
            logger.warning(
                "Skip task[%s] under task_config[%s]: the expert planner "
                "cannot reliably solve it on this embodiment",
                task,
                task_config,
            )
            continue
        selected.append(task)
    return selected


def summarize_results(results, episodes):
    """Fold per-task counts into the shape the RoboTwin reports use.

    ``success_rate`` is a percentage per task and ``mean`` averages those
    percentages, matching how RoboTwin sweeps are reported.
    """
    rates = {
        task: 100.0 * counts["successes"] / counts["episodes"]
        for task, counts in results.items()
    }
    tested = {task: counts["episodes"] for task, counts in results.items()}
    return {
        "success_rate": dict(sorted(rates.items())),
        "num_tested": dict(sorted(tested.items())),
        "test_num_per_task": episodes,
        "num_tasks_in_mean": len(rates),
        "mean": sum(rates.values()) / len(rates) if rates else 0.0,
    }


def allocate_tasks_to_workers(tasks, num_gpus, processes_per_gpu=1):
    """Round-robin tasks over ``num_gpus * processes_per_gpu`` workers.

    Slots are ordered GPU-major so a short task list spreads across every GPU
    before any GPU takes a second process.
    """
    if num_gpus < 1:
        raise ValueError("num_gpus must be >= 1")
    if processes_per_gpu < 1:
        raise ValueError("processes_per_gpu must be >= 1")
    if not tasks:
        return []
    worker_gpu_ids = [
        gpu_id for _ in range(processes_per_gpu) for gpu_id in range(num_gpus)
    ][: len(tasks)]
    allocated = [(gpu_id, []) for gpu_id in worker_gpu_ids]
    for index, task in enumerate(tasks):
        allocated[index % len(allocated)][1].append(task)
    return allocated


def eval_tasks(
    gpu_id, tasks, root, config_path, env, output, episodes, results
):
    """Evaluate one worker's tasks on a single GPU.

    A task that crashes or reports too few episodes is logged and skipped so a
    long sweep is not lost to one bad environment; it simply stays out of the
    results.
    """
    env = dict(env, CUDA_VISIBLE_DEVICES=str(gpu_id))
    for task in tasks:
        task_dir = output / task
        task_dir.mkdir(parents=True, exist_ok=True)
        log = task_dir / "log.txt"
        command = [
            sys.executable,
            str(root / "script/eval_policy.py"),
            "--config",
            str(config_path),
            "--overrides",
            "--task_name",
            task,
        ]
        logger.info(
            "Start to eval task[%s] on gpu %s, log file: %s", task, gpu_id, log
        )
        with log.open("w") as stream:
            completed = subprocess.run(
                command,
                cwd=root,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text())
        counts = re.findall(r"Success rate:\s*(\d+)\s*/\s*(\d+)", text)
        if completed.returncode != 0 or not counts:
            logger.error(
                "Failed to eval task[%s] (exit %s); inspect %s",
                task,
                completed.returncode,
                log,
            )
            continue
        successes, tested = map(int, counts[-1])
        if tested != episodes:
            logger.error(
                "task[%s] scored %s of %s episodes; inspect %s",
                task,
                tested,
                episodes,
                log,
            )
            continue
        record = dict(
            successes=successes,
            episodes=tested,
            success_rate=successes / tested,
        )
        results[task] = record
        (task_dir / "results.json").write_text(json.dumps(record, indent=2))
        logger.info(
            "Finished eval task[%s]. success rate: %s/%s",
            task,
            successes,
            tested,
        )
    return results


def evaluate(
    robotwin_root,
    model_dir,
    processor,
    tasks,
    output,
    task_config="demo_clean",
    episodes=2,
    seed=0,
    num_gpus=None,
    processes_per_gpu=1,
):
    """Fan tasks across GPUs, each worker driving the upstream RoboTwin CLI.

    The upstream checkout owns environment construction and seed selection.
    """
    root = Path(robotwin_root).expanduser().resolve(strict=True)
    model_dir = Path(model_dir).expanduser().resolve(strict=True)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if episodes < 1:
        raise ValueError("episodes must be positive")
    config_path = output / "policy.yaml"
    config_path.write_text(
        yaml.safe_dump(
            dict(
                policy_name="projects.holobrain_0_5.policy.robotwin_policy",
                model_config=str(model_dir),
                model_processor=processor,
                task_config=task_config,
                ckpt_setting="holobrain_0.5",
                seed=seed,
                instruction_type="unseen",
                test_num=episodes,
            )
        )
    )
    env = os.environ.copy()
    repo = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [repo, env.get("PYTHONPATH")])
    )

    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        raise RuntimeError("No GPUs available for evaluation")
    allocated = allocate_tasks_to_workers(tasks, num_gpus, processes_per_gpu)
    logger.info(
        "Found %s GPU(s). Distributing %s tasks with %s process(es) per GPU",
        num_gpus,
        len(tasks),
        processes_per_gpu,
    )
    for worker_id, (gpu_id, worker_tasks) in enumerate(allocated):
        logger.info("worker %s (gpu %s): %s", worker_id, gpu_id, worker_tasks)

    args = (root, config_path, env, output, episodes)
    if len(allocated) > 1:
        manager = multiprocessing.Manager()
        shared = manager.dict()
        workers = []
        for gpu_id, worker_tasks in allocated:
            worker = multiprocessing.Process(
                target=eval_tasks, args=(gpu_id, worker_tasks, *args, shared)
            )
            worker.start()
            workers.append(worker)
        for worker in workers:
            worker.join()
        return dict(shared)
    if not allocated:
        return {}
    gpu_id, worker_tasks = allocated[0]
    return eval_tasks(gpu_id, worker_tasks, *args, {})


def main():
    """Evaluate explicit local tasks using the upstream RoboTwin CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--processor", default="robotwin2_0_processor")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Tasks to evaluate; defaults to every task the checkout defines.",
    )
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--processes-per-gpu", type=int, default=1)
    args = parser.parse_args()
    log_basic_config(
        format="%rank %(asctime)s %(levelname)s %(filename)s:%(lineno)d | %(message)s",  # noqa: E501
        level=logging.INFO,
    )
    logger.info("\n%s", json.dumps(vars(args), indent=4, default=str))
    tasks = select_tasks(
        list_robotwin_tasks(args.robotwin_root), args.tasks, args.task_config
    )
    if not tasks:
        raise SystemExit("No RoboTwin tasks left to evaluate")
    started = datetime.now()
    results = evaluate(
        args.robotwin_root,
        args.model_dir,
        args.processor,
        tasks,
        args.output,
        args.task_config,
        args.episodes,
        args.seed,
        args.num_gpus,
        args.processes_per_gpu,
    )
    summary = summarize_results(results, args.episodes)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("=" * 50)
    logger.info("Evaluation finished in %s", datetime.now() - started)
    logger.info("Scored %s of %s task(s)", len(results), len(tasks))
    logger.info("Detailed JSON results:")
    logger.info("\n%s", json.dumps(summary, indent=4))
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
