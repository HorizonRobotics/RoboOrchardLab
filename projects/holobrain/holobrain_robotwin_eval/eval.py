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
import re
import subprocess

import torch

from robo_orchard_lab.utils import log_basic_config

bash_command_template = (
    "export CUDA_VISIBLE_DEVICES={gpu_id} \n"
    "cp -r projects/holobrain/holobrain_robotwin_eval {robotwin_dir} \n"
    "cd {robotwin_dir} \n"
    "python3 script/eval_policy.py --config holobrain_robotwin_eval/eval_config.yml "  # noqa: E501
    "  --overrides "
    "  --task_config {task_config} "
    "  --task_name {task_name} "
    "  --vlm_ckpt_dir {vlm_ckpt_dir} "
    "  --urdf_dir {urdf_dir} "
    "  --model_config {model_config} "
    "  --model_processor {model_processor} "
    "  --model_prefix {model_prefix} "
    "  --test_num {test_num}"
)


def eval_tasks(robotwin_dir, gpu_id, task_names, args, results=None):
    if_cluster = os.environ.get("CLUSTER") is not None
    if results is None:
        results = {}
    for task_name in task_names:
        if not if_cluster:
            log_dir = f"eval_result/{task_name}/{args.task_config}"
        else:
            log_dir = f"/job_data/{task_name}/{args.task_config}"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "log.txt")
        logger.info(f"Start to eval task[{task_name}], log file: {log_file}")

        robotwin_dir = os.path.abspath(robotwin_dir)
        model_config = os.path.abspath(args.model_config)
        vlm_ckpt_dir = args.vlm_ckpt_dir
        urdf_dir = args.urdf_dir
        if vlm_ckpt_dir is not None:
            vlm_ckpt_dir = os.path.abspath(vlm_ckpt_dir)
        if urdf_dir is not None:
            urdf_dir = os.path.abspath(urdf_dir)
        command = bash_command_template.format(
            robotwin_dir=robotwin_dir,
            gpu_id=gpu_id,
            task_name=task_name,
            task_config=args.task_config,
            vlm_ckpt_dir=vlm_ckpt_dir,
            urdf_dir=urdf_dir,
            model_config=model_config,
            model_processor=args.model_processor,
            model_prefix=args.model_prefix,
            test_num=args.test_num,
        )
        with open(log_file, "w", encoding="utf-8") as f:
            ret = subprocess.run(
                command, shell=True, stdout=f, stderr=subprocess.STDOUT
            )
        if ret.returncode == 0:
            with open(log_file, "r", encoding="utf-8") as f:
                output = f.read().strip().split("\n")
            for out in output[::-1]:
                if "Success rate" in out:
                    results[task_name] = float(
                        re.findall(r"\d+\.?\d+%", out)[0][:-1]
                    )
                    logger.info(f"{task_name}: {out}")
                    break
        else:
            logger.info(
                f"Fail to eval task[{task_name}], "
                f"with returncode {ret.returncode}."
                f"Refer to {log_file} for more infomation."
            )
    return results


if __name__ == "__main__":
    logger = logging.getLogger(__file__)
    log_basic_config(
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d | %(message)s",  # noqa: E501
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_config", type=str)
    parser.add_argument("--model_config", type=str)
    parser.add_argument(
        "--model_processor", type=str, default="robotwin2_0_processor"
    )
    parser.add_argument("--model_prefix", type=str, default="model")
    parser.add_argument("--task_names", type=str, default=None)
    parser.add_argument("--vlm_ckpt_dir", type=str, default=None)
    parser.add_argument("--urdf_dir", type=str, default=None)
    parser.add_argument("--test_num", type=int, default=100)
    parser.add_argument("--robotwin_dir", type=str, default="./robotwin")
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    logger.info("\n" + json.dumps(vars(args), indent=4))

    robotwin_dir = args.robotwin_dir
    if not os.path.isdir(robotwin_dir):
        raise FileNotFoundError("The robotwin_dir must be correctly specified")
    task_names = args.task_names
    all_task_names = [
        x[:-3]
        for x in os.listdir(os.path.join(robotwin_dir, "envs"))
        if x.endswith(".py") and not x.startswith("_")
    ]
    if task_names is None:
        task_names = all_task_names
    else:
        task_names = task_names.strip().split(",")

    num_gpus = min(torch.cuda.device_count(), args.num_workers)
    task_names_allocated = [[] for _ in range(min(num_gpus, len(task_names)))]
    for index, task_name in enumerate(task_names):
        if task_name not in all_task_names:
            logger.warning(f"Task {task_name} is invalid !!")
            continue
        task_names_allocated[index % num_gpus].append(task_name)

    if len(task_names_allocated) > 1:
        processes = []
        manager = multiprocessing.Manager()
        results = manager.dict()
        for gpu_id, task_names_per_gpu in enumerate(task_names_allocated):
            logger.info(f"{gpu_id}: {task_names_per_gpu}")
            p = multiprocessing.Process(
                target=eval_tasks,
                args=(robotwin_dir, gpu_id, task_names_per_gpu, args, results),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        results = dict(results)
    else:
        results = eval_tasks(robotwin_dir, 0, task_names_allocated[0], args)
    results = dict(sorted(results.items()))
    mean_success_rate = sum(list(results.values())) / len(results)
    results["num_tasks"] = len(results)
    results["mean"] = mean_success_rate
    results["test_num_per_task"] = args.test_num
    logger.info(json.dumps(results, indent=4))
