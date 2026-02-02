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
import sys
from datetime import datetime

import torch
from libero.libero import benchmark

from robo_orchard_lab.utils import log_basic_config

bash_command_template = (
    "export CUDA_VISIBLE_DEVICES={gpu_id} && "
    "python3 eval_policy.py --config sem_libero_policy/deploy_policy.yml"
    "  --task_id {task_id} "
    "  --output_dir {output_dir} "
    "  --overrides "
    "  --model_config {model_config} "
    "  --model_prefix {model_prefix} "
    "  --vlm_ckpt_dir {vlm_ckpt_dir} "
    "  --urdf_dir {urdf_dir} "
    "  --model_processor {model_processor} "
    "  --task_suite {task_suite} "
    "  --num_trials_per_task {num_trials_per_task} "
    "  --num_steps_wait {num_steps_wait} "
    "  --save_video {save_video} "
)


def eval_tasks(gpu_id, grouped_batches, args, results=None):
    if_cluster = os.environ.get("CLUSTER") is not None
    if results is None:
        results = {}
    for suite_name, task_id in grouped_batches:
        if not if_cluster:
            log_dir = f"eval_result/gpu_{gpu_id}/{suite_name}"
        else:
            log_dir = f"/job_data/gpu_{gpu_id}/{suite_name}"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "log.txt")
        logger.info(f"Start to eval task[{task_id}], log file: {log_file}")
        task_id_str = json.dumps(task_id)
        command = bash_command_template.format(
            gpu_id=gpu_id,
            task_id=f"'{task_id_str}'",
            model_config=args.model_config,
            model_prefix=args.model_prefix,
            vlm_ckpt_dir=args.vlm_ckpt_dir,
            urdf_dir=args.urdf_dir,
            model_processor=args.model_processor,
            task_suite=suite_name,
            num_trials_per_task=args.num_trials_per_task,
            num_steps_wait=args.num_steps_wait,
            save_video=args.save_video,
            output_dir=log_dir,
        )
        with open(log_file, "w", encoding="utf-8") as f:
            ret = subprocess.run(
                command, shell=True, stdout=f, stderr=subprocess.STDOUT
            )
        json_result_file = os.path.join(log_dir, "results.json")
        if ret.returncode == 0:
            if os.path.exists(json_result_file):
                try:
                    with open(json_result_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if suite_name in data:
                            for t_key, rate in data[suite_name].items():
                                results[(suite_name, t_key)] = rate
                except Exception as e:
                    logger.error(
                        f"Failed to load JSON {json_result_file}: {e}"
                    )
        else:
            logger.info(f"Batch failed with return code {ret.returncode}")
    return results


if __name__ == "__main__":
    logger = logging.getLogger(__file__)
    log_basic_config(
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d | %(message)s",  # noqa: E501
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--model_prefix", type=str, default="model_0")

    parser.add_argument(
        "--vlm_ckpt_dir",
        type=str,
        default="/horizon-bucket/robot_lab/users/xuewu.lin/ckpt",
    )
    parser.add_argument(
        "--urdf_dir",
        type=str,
        default="/horizon-bucket/robot_lab/users/xuewu.lin/urdf",
    )
    parser.add_argument(
        "--model_processor", type=str, default="libero_processor"
    )
    parser.add_argument("--task_suite", type=str, default="-1")
    parser.add_argument(
        "--task_id", type=int, default=-1, help="Specific task id to run"
    )
    parser.add_argument("--num_trials_per_task", type=int, default=1)
    parser.add_argument("--num_steps_wait", type=int, default=100)
    parser.add_argument("--save_video", type=bool, default=True)

    args = parser.parse_args()

    logger.info("\n" + json.dumps(vars(args), indent=4))
    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite == "-1":
        target_suites = [
            "libero_10",
            "libero_goal",
            "libero_object",
            "libero_spatial",
        ]
        logger.info(
            f"Task Suite set to -1. Will run ALL suites: {target_suites}"
        )
    else:
        if args.task_suite not in benchmark_dict:
            logger.error(
                f"Task suite '{args.task_suite}' not found in benchmark dict."
            )
            sys.exit(1)
        target_suites = [args.task_suite]

    all_jobs = []
    for s_name in target_suites:
        task_suite_instance = benchmark_dict[s_name]()
        num_tasks = task_suite_instance.n_tasks
        for i in range(num_tasks):
            all_jobs.append((s_name, i))

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        logger.error("No GPUs found! Please check CUDA availability.")
        sys.exit(1)

    logger.info(f"Found {num_gpus} GPUs. Distributing {num_tasks} tasks...")

    tasks_allocated = [[] for _ in range(num_gpus)]
    total_jobs = len(all_jobs)
    chunk_size = total_jobs // num_gpus
    remainder = total_jobs % num_gpus
    start_idx = 0
    for i in range(num_gpus):
        current_count = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_count
        tasks_allocated[i] = all_jobs[start_idx:end_idx]
        start_idx = end_idx

    def merge_tasks_by_suite(gpu_job_list):
        from collections import defaultdict

        merged_dict = defaultdict(list)
        for s_name, t_id in gpu_job_list:
            merged_dict[s_name].append(t_id)
        return list(merged_dict.items())

    if len(all_jobs) > 1:
        manager = multiprocessing.Manager()
        results = manager.dict()
        processes = []
        start_time = datetime.now()
        for gpu_id in range(num_gpus):
            gpu_tasks = tasks_allocated[gpu_id]
            grouped_batches = merge_tasks_by_suite(gpu_tasks)
            if not grouped_batches:
                continue
            p = multiprocessing.Process(
                target=eval_tasks,
                args=(gpu_id, grouped_batches, args, results),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        results = dict(results)
    else:
        results = eval_tasks(0, all_jobs, args)

    nested_results = {}
    for (s_name, t_key), rate in results.items():
        if s_name not in nested_results:
            nested_results[s_name] = {}
        nested_results[s_name][t_key] = rate

    final_output = {"overall_summary": {}, "suites_detail": nested_results}

    logger.info("=" * 50)
    logger.info(f"Evaluation Finished in {datetime.now() - start_time}")
    for s_name in target_suites:
        if s_name in nested_results:
            rates = list(nested_results[s_name].values())
            mean_rate = sum(rates) / len(rates)

            final_output["overall_summary"][s_name] = {
                "mean_success_rate": mean_rate,
                "num_tasks": len(rates),
            }

    logger.info("Detailed JSON Results:")
    logger.info(json.dumps(final_output, indent=4))
    logger.info("=" * 50)
