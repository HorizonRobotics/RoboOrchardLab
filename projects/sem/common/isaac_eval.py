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
import glob
import json
import logging
import os
import pickle
import subprocess

from robo_orchard_lab.utils import log_basic_config

os.environ["NV_ASSET_ROOT_DIR"] = (
    "/horizon-bucket/robot_lab/assets/NVIDIA/Assets/Isaac/4.1"  # noqa: E501
)

logger = logging.getLogger(__file__)
log_basic_config(
    format="%rank %(asctime)s %(levelname)s %(filename)s:%(lineno)d | %(message)s",  # noqa: E501
    level=logging.INFO,
)

bash_command_template = (
    "python3 examples/manipulation-app/pick_place/config/gen_dualarm_piper_{task_name}.py && "  # noqa: E501
    "python3 examples/manipulation-app/pick_place/scripts/eval_policy_sem.py "  # noqa: E501
    "  --model_config {model_config} "
    "  --model_processor {model_processor} "
    "  --model_prefix {model_prefix}"
    "  --vlm_ckpt_dir {vlm_ckpt_dir}"
    "  --urdf_dir {urdf_dir}"
    "  --task_name {task_name}"
    "  --seed {seed}"
    "  --rollouts {test_num}"
    "  --output_dir {output_dir}"
    "  --maximum_step 1000"  # TODO, make it configurable
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str)
    parser.add_argument(
        "--model_processor", type=str, default="isaac_pick_place_processor"
    )
    parser.add_argument("--model_prefix", type=str, default="model_0")
    parser.add_argument("--task_names", type=str, default=None)
    parser.add_argument("--vlm_ckpt_dir", type=str, default=None)
    parser.add_argument("--urdf_dir", type=str, default=None)
    parser.add_argument("--test_num", type=int, default=100)
    parser.add_argument("--seed", type=int, default=100000)
    args = parser.parse_args()
    logger.info("\n" + json.dumps(vars(args), indent=4))

    task_names = args.task_names.strip().split(",")
    logger.info(f"Evaluating tasks: {task_names}")

    if_cluster = os.environ.get("CLUSTER") is not None
    if not if_cluster:
        log_root = "eval_result/"
    else:
        log_root = "/job_data/"

    processes = []
    task_to_proc = {}
    for gpu_id, task_name in enumerate(task_names):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logger.info(f"{gpu_id}: {task_name}")

        log_dir = os.path.join(log_root, task_name)
        os.makedirs(log_dir, exist_ok=True)

        command = [
            "bash",
            "-c",
            (
                bash_command_template.format(
                    gpu_id=gpu_id,
                    task_name=task_name,
                    vlm_ckpt_dir=args.vlm_ckpt_dir,
                    urdf_dir=args.urdf_dir,
                    model_config=args.model_config,
                    model_processor=args.model_processor,
                    model_prefix=args.model_prefix,
                    test_num=args.test_num,
                    seed=args.seed,
                    output_dir=log_dir,
                )
            ),
        ]
        p = subprocess.Popen(
            command,
            env=env,
        )
        processes.append(p)
        task_to_proc[p] = task_name
    for p in processes:
        ret = p.wait()
        if ret != 0:
            raise RuntimeError(f"Process for task {task_to_proc[p]} failed")

    results = {}
    for task_name in task_names:
        pattern = os.path.join(
            log_root, task_name, f"eval_{task_name}_*/task_res.pkl"
        )
        task_res_paths = glob.glob(pattern)
        task_res_paths.sort()
        task_res_path = task_res_paths[-1]  # using the latest result
        logger.info("Loading results from {}".format(task_res_path))
        with open(task_res_path, "rb") as f:
            task_res = pickle.load(f)
        results[task_name] = task_res

    results = dict(sorted(results.items()))
    task_means = []
    task_progresses = []
    for task_name, task_data in results.items():
        successes = list(task_data["task_success"].values())
        task_mean = sum(successes) / len(successes)
        task_means.append(task_mean)
        processes = list(task_data["task_progress"].values())
        task_progresses.append(sum(processes) / len(processes))
        logger.info(
            f"Task {task_name}: Success Rate = {task_mean:.4f}, "
            f"Avg Progress = {sum(processes) / len(processes):.4f}"
        )
    mean_success_rate = sum(task_means) / len(task_means)
    mean_progress = sum(task_progresses) / len(task_progresses)
    results["num_tasks"] = len(results)
    results["mean_success"] = mean_success_rate
    results["mean_progress"] = mean_progress
    results["test_num_per_task"] = args.test_num
    logger.info(json.dumps(results, indent=4))
