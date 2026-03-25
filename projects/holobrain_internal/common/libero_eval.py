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

import argparse
import json
import logging
import multiprocessing
import os
import subprocess
from datetime import datetime

import torch
from libero.libero import benchmark

from robo_orchard_lab.utils import log_basic_config


def eval_tasks(gpu_id, tasks, args, results=None):
    if_cluster = os.environ.get("CLUSTER") is not None
    if results is None:
        results = {}
    for suite_name, task_id in tasks:
        if not if_cluster:
            log_dir = f"eval_result/{suite_name}/task_{task_id}"
        else:
            log_dir = f"/job_data/{suite_name}/task_{task_id}"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "log.txt")
        logger.info(
            f"Start to eval {suite_name} task_{task_id}, log file: {log_file}"
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        command = [
            "python3",
            "eval_policy.py",
            "--config",
            "holobrain_libero_policy/deploy_policy.yml",
            "--task_id",
            str(task_id),
            "--output_dir",
            log_dir,
            "--task_suite",
            suite_name,
            "--num_trials_per_task",
            str(args.num_trials_per_task),
            "--num_steps_wait",
            str(args.num_steps_wait),
            "--save_video",
            str(args.save_video),
            "--overrides",
            "--model_config",
            args.model_config,
            "--model_prefix",
            args.model_prefix,
            "--vlm_ckpt_dir",
            args.vlm_ckpt_dir,
            "--urdf_dir",
            args.urdf_dir,
            "--model_processor",
            args.model_processor,
        ]
        with open(log_file, "w", encoding="utf-8") as f:
            ret = subprocess.run(
                command, env=env, stdout=f, stderr=subprocess.STDOUT
            )
        if ret.returncode == 0:
            json_result_file = os.path.join(log_dir, "results.json")
            with open(json_result_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            results[(suite_name, f"task_{task_id}")] = data["success_rate"]
            logger.info(
                f"Finished eval {suite_name} task_{task_id}."
                f" success rate: {data['success_rate']}"
            )
        else:
            logger.info(
                f"Fail to eval {suite_name} task_{task_id}"
                f"with returncode {ret.returncode}"
            )
    return results


if __name__ == "__main__":
    logger = logging.getLogger(__file__)
    log_basic_config(
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d | %(message)s",  # noqa: E501
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--model_prefix", type=str, default=None)
    parser.add_argument("--vlm_ckpt_dir", type=str, default=None)
    parser.add_argument("--urdf_dir", type=str, default=None)
    parser.add_argument(
        "--model_processor", type=str, default="libero_processor"
    )
    parser.add_argument(
        "--task_suite",
        type=str,
        default="libero_10,libero_goal,libero_object,libero_spatial",
    )
    parser.add_argument(
        "--task_id", type=int, default=-1, help="Specific task id to run"
    )
    parser.add_argument("--num_trials_per_task", type=int, default=1)
    parser.add_argument("--num_steps_wait", type=int, default=100)
    parser.add_argument("--save_video", type=bool, default=True)

    args = parser.parse_args()

    logger.info("\n" + json.dumps(vars(args), indent=4))
    benchmark_dict = benchmark.get_benchmark_dict()
    target_suites = [
        x for x in args.task_suite.lower().split(",") if x in benchmark_dict
    ]

    all_tasks = []
    for s_name in target_suites:
        task_suite_instance = benchmark_dict[s_name]()
        num_tasks = task_suite_instance.n_tasks
        for i in range(num_tasks):
            all_tasks.append((s_name, i))

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs found! Please check CUDA availability.")

    logger.info(f"Found {num_gpus} GPUs. Distributing {num_tasks} tasks...")

    tasks_allocated = [[] for _ in range(min(num_gpus, len(all_tasks)))]
    for index, task in enumerate(all_tasks):
        tasks_allocated[index % num_gpus].append(task)

    if len(all_tasks) > 1:
        manager = multiprocessing.Manager()
        results = manager.dict()
        processes = []
        start_time = datetime.now()

        for gpu_id, gpu_tasks in enumerate(tasks_allocated):
            logger.info(f"{gpu_id}: {gpu_tasks}")
            p = multiprocessing.Process(
                target=eval_tasks,
                args=(gpu_id, gpu_tasks, args, results),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        results = dict(results)
    else:
        results = eval_tasks(0, all_tasks, args)

    nested_results = {}
    for (s_name, t_key), rate in results.items():
        if s_name not in nested_results:
            nested_results[s_name] = {}
        nested_results[s_name][t_key] = rate

    final_output = {"overall_summary": {}, "suites_detail": nested_results}

    logger.info("=" * 50)
    logger.info(f"Evaluation Finished in {datetime.now() - start_time}")

    mean_average_rate = []
    for s_name in target_suites:
        if s_name in nested_results:
            rates = list(nested_results[s_name].values())
            mean_rate = sum(rates) / len(rates)

            final_output["overall_summary"][s_name] = {
                "average_success_rate": mean_rate,
                "num_tasks": len(rates),
            }
            mean_average_rate.append(mean_rate)
    final_output["overall_summary"]["average_success_rate_across_suites"] = (
        sum(mean_average_rate) / len(mean_average_rate)
        if mean_average_rate
        else 0.0
    )

    logger.info("Detailed JSON Results:")
    logger.info(json.dumps(final_output, indent=4))
    logger.info("=" * 50)
