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

import yaml

from robo_orchard_lab.utils import log_basic_config

os.environ["NV_ASSET_ROOT_DIR"] = (
    "/horizon-bucket/robot_lab/assets/NVIDIA/Assets/Isaac/4.1"  # noqa: E501
)

logger = logging.getLogger(__file__)
log_basic_config(
    format="%rank %(asctime)s %(levelname)s %(filename)s:%(lineno)d | %(message)s",  # noqa: E501
    level=logging.INFO,
)


def load_multi_task_config(config_path):
    """Load and parse the multi-task YAML config file.

    Args:
        config_path: Path to the multi-task config YAML file

    Returns:
        Parsed config dictionary
    """
    if not os.path.exists(config_path):
        logger.warning(f"Multi-task config file not found: {config_path}")
        return {}

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded multi-task config from: {config_path}")
        return config if config else {}
    except Exception as e:
        logger.error(f"Failed to load multi-task config: {e}")
        return {}


def get_task_config(multi_task_config, task_name):
    """Get complete config for a specific task by.

    Args:
        multi_task_config: The full multi-task configuration dictionary
        task_name: Name of the task to get config for

    Returns:
        Complete task configuration with defaults merged
    """

    def deep_merge_dict(base, override):
        result = base.copy()
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = deep_merge_dict(result[key], value)
            else:
                result[key] = value
        return result

    default_config = multi_task_config.get("_default", {})
    task_specific_config = multi_task_config.get(task_name, {})

    # Merge default with task-specific config
    merged_config = deep_merge_dict(default_config, task_specific_config)

    logger.info(
        f"Task '{task_name}' config: {json.dumps(merged_config, indent=2)}"
    )
    return merged_config


bash_command_template = (
    "python3 examples/manipulation-app/pick_place/config/gen_dualarm_piper_{task_name}.py "  # noqa: E501
    "  --task_config {task_config} && "
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
    "  --task_config {task_config} "
    "  --maximum_step 1000"  # TODO, make it configurable
)


def log_task_table(all_task_data):
    def center(val, width):
        s = f"{val:.2f}"
        left_pad = (width - len(s)) // 2
        right_pad = width - len(s) - left_pad
        return " " * left_pad + s + " " * right_pad

    rows = [
        (
            task_name,
            sum(d["task_success"].values()) / len(d["task_success"]),
            sum(d["task_progress"].values()) / len(d["task_progress"]),
        )
        for task_name, d in all_task_data.items()
    ]
    rows.append(
        (
            "mean",
            sum(r[1] for r in rows) / len(rows),
            sum(r[2] for r in rows) / len(rows),
        )
    )

    task_w = max(len("task"), *(len(r[0]) for r in rows))
    succ_w = max(len("success rate"), 4)
    prog_w = max(len("progress score"), 4)

    header = (
        f"| {'task':<{task_w}} | "
        f"{'success rate':^{succ_w}} | "
        f"{'progress score':^{prog_w}} |"
    )
    sep = f"|{'-' * (task_w + 2)}|{'-' * (succ_w + 2)}|{'-' * (prog_w + 2)}|"
    lines = [header, sep]

    lines += [
        f"| {t:<{task_w}} | {center(s, succ_w)} | {center(p, prog_w)} |"
        for t, s, p in rows
    ]

    logger.info("\n" + "\n".join(lines))


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
    parser.add_argument(
        "--multi_task_config",
        type=str,
        default="isaac_task_config/multi_task_setting.yaml",
        help="Path to multi-task environment settings YAML file",
    )
    args = parser.parse_args()
    logger.info("\n" + json.dumps(vars(args), indent=4))

    task_names = args.task_names.strip().split(",")
    logger.info(f"Evaluating tasks: {task_names}")

    # Load multi-task configuration
    multi_task_config = load_multi_task_config(args.multi_task_config)

    if_cluster = os.environ.get("CLUSTER") is not None
    if not if_cluster:
        log_root = "eval_result/"
        temp_config_dir = "eval_result/task_configs/"
    else:
        log_root = "/job_data/"
        temp_config_dir = "/job_data/task_configs/"

    # Create temp config directory
    os.makedirs(temp_config_dir, exist_ok=True)
    logger.info(f"Temp task config directory: {temp_config_dir}")

    processes = []
    task_to_proc = {}
    for gpu_id, task_name in enumerate(task_names):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logger.info(f"{gpu_id}: {task_name}")

        log_dir = os.path.join(log_root, task_name)
        os.makedirs(log_dir, exist_ok=True)

        # Get merged task config and write to temp file
        task_config = get_task_config(multi_task_config, task_name)
        task_config_path = os.path.join(
            temp_config_dir, f"{task_name}_env_setting.yaml"
        )
        with open(task_config_path, "w") as f:
            yaml.dump(task_config, f, default_flow_style=False)
        logger.info(
            f"Generated temp config for {task_name}: {task_config_path}"
        )

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
                    task_config=task_config_path,
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
    logger.info(json.dumps(results, indent=4))

    log_task_table(results)
