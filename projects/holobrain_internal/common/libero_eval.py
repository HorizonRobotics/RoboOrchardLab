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
from pathlib import Path

import torch

logger = logging.getLogger(__file__)

_LIBERO_UTILS_DIR = Path(__file__).resolve().parents[1] / "libero"
if _LIBERO_UTILS_DIR.exists():
    sys.path.insert(0, str(_LIBERO_UTILS_DIR))

LIBERO_PLUS_CATEGORY_ORDER = [
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
]


def _append_override(command, key, value):
    if value is None:
        return
    command.extend([key, str(value)])


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def _default_eval_policy_path():
    source_tree_path = Path(__file__).resolve().parents[1] / (
        "libero/eval_policy.py"
    )
    if source_tree_path.exists():
        return source_tree_path
    return Path(__file__).resolve().parent / "eval_policy.py"


def _default_policy_config_path():
    source_tree_path = Path(__file__).resolve().parent / (
        "holobrain_libero_policy/deploy_policy.yml"
    )
    if source_tree_path.exists():
        return source_tree_path
    return Path("holobrain_libero_policy/deploy_policy.yml")


def _prepend_pythonpath(env, *paths):
    existing = env.get("PYTHONPATH")
    pythonpath_parts = []
    for path in paths:
        if path is None:
            continue
        path = Path(path)
        if path.exists():
            pythonpath_parts.append(str(path.resolve()))
    if existing:
        pythonpath_parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)


def _source_repo_root():
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / "robo_orchard_lab").exists():
            return parent
    return None


def _selected_benchmark_root(args):
    if args.benchmark == "libero_plus":
        return getattr(args, "libero_plus_root", None)
    return getattr(args, "libero_root", None)


def _get_benchmark_dict(
    benchmark_name,
    benchmark_module=None,
    benchmark_root=None,
):
    from libero_utils import get_benchmark_module

    return get_benchmark_module(
        benchmark_name,
        benchmark_module,
        benchmark_root=benchmark_root,
    ).get_benchmark_dict()


def resolve_target_suites(task_suite_arg, benchmark_dict):
    requested = [x.strip().lower() for x in task_suite_arg.split(",")]
    if requested == ["-1"] or requested == ["all"]:
        return list(benchmark_dict)
    return [suite for suite in requested if suite in benchmark_dict]


def build_task_list(task_suite_arg, task_id, benchmark_dict):
    all_tasks = []
    for suite_name in resolve_target_suites(task_suite_arg, benchmark_dict):
        task_suite_instance = benchmark_dict[suite_name]()
        num_tasks = task_suite_instance.n_tasks
        if task_id >= 0:
            if task_id >= num_tasks:
                raise ValueError(
                    f"task_id {task_id} is out of range for {suite_name}: "
                    f"[0, {num_tasks})"
                )
            all_tasks.append((suite_name, task_id))
            continue
        for i in range(num_tasks):
            all_tasks.append((suite_name, i))
    return all_tasks


def allocate_tasks_to_workers(tasks, num_gpus, processes_per_gpu=1):
    if num_gpus < 1:
        raise ValueError("num_gpus must be >= 1.")
    if processes_per_gpu < 1:
        raise ValueError("processes_per_gpu must be >= 1.")
    worker_gpu_ids = [
        gpu_id for gpu_id in range(num_gpus) for _ in range(processes_per_gpu)
    ]
    worker_gpu_ids = worker_gpu_ids[: len(tasks)]
    allocated = [(gpu_id, []) for gpu_id in worker_gpu_ids]
    for index, task in enumerate(tasks):
        allocated[index % len(allocated)][1].append(task)
    return allocated


def _parse_task_key(task_key):
    prefix, task_id = task_key.split("_", maxsplit=1)
    if prefix != "task":
        raise ValueError(f"Unexpected task result key: {task_key}")
    return int(task_id)


def _load_libero_plus_classification(benchmark_root):
    if benchmark_root is None:
        return {}
    classification_path = Path(benchmark_root) / (
        "libero/libero/benchmark/task_classification.json"
    )
    if not classification_path.exists():
        logger.warning(
            "LIBERO-Plus task classification file does not exist: %s",
            classification_path,
        )
        return {}
    with open(classification_path, "r", encoding="utf-8") as f:
        classification = json.load(f)
    return {
        (suite_name, item["name"]): item["category"]
        for suite_name, items in classification.items()
        for item in items
    }


def summarize_libero_plus_categories(
    results,
    benchmark_dict,
    benchmark_root,
):
    category_by_task = _load_libero_plus_classification(benchmark_root)
    rates_by_category = {}
    suite_instances = {}
    for (suite_name, task_key), rate in results.items():
        task_id = _parse_task_key(task_key)
        if suite_name not in suite_instances:
            suite_instances[suite_name] = benchmark_dict[suite_name]()
        task_name = suite_instances[suite_name].get_task(task_id).name
        category = category_by_task.get(
            (suite_name, task_name),
            "Unclassified",
        )
        rates_by_category.setdefault(category, []).append(rate)

    category_order = [
        *LIBERO_PLUS_CATEGORY_ORDER,
        *sorted(
            category
            for category in rates_by_category
            if category not in LIBERO_PLUS_CATEGORY_ORDER
        ),
    ]
    category_summary = {}
    category_means = []
    all_rates = []
    for category in category_order:
        rates = rates_by_category.get(category)
        if not rates:
            continue
        mean_rate = sum(rates) / len(rates)
        category_summary[category] = {
            "average_success_rate": mean_rate,
            "num_tasks": len(rates),
        }
        category_means.append(mean_rate)
        all_rates.extend(rates)

    return {
        "categories_detail": category_summary,
        "average_success_rate_across_categories": (
            sum(category_means) / len(category_means)
            if category_means
            else 0.0
        ),
        "average_success_rate_all_tasks": (
            sum(all_rates) / len(all_rates) if all_rates else 0.0
        ),
        "num_tasks": len(all_rates),
    }


def eval_tasks(gpu_id, tasks, args, results=None):
    from libero_utils import build_subprocess_env

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
        benchmark_root = _selected_benchmark_root(args)
        env = build_subprocess_env(
            args.benchmark,
            benchmark_root,
            os.environ.copy(),
        )
        _prepend_pythonpath(
            env,
            _source_repo_root(),
            Path(__file__).resolve().parent,
            _default_eval_policy_path().parent,
        )
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        eval_policy_path = (
            getattr(args, "eval_policy_path", None)
            or _default_eval_policy_path()
        )
        policy_config_path = (
            getattr(args, "policy_config_path", None)
            or _default_policy_config_path()
        )
        command = [
            "python3",
            str(eval_policy_path),
            "--config",
            str(policy_config_path),
            "--benchmark",
            args.benchmark,
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
        ]
        if args.benchmark_module is not None:
            command.extend(["--benchmark_module", args.benchmark_module])
        if benchmark_root is not None:
            command.extend(["--benchmark_root", benchmark_root])
        command.append("--overrides")
        _append_override(command, "--model_config", args.model_config)
        _append_override(command, "--model_prefix", args.model_prefix)
        _append_override(command, "--vlm_ckpt_dir", args.vlm_ckpt_dir)
        _append_override(command, "--urdf_dir", args.urdf_dir)
        _append_override(command, "--model_processor", args.model_processor)
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
    source_repo_root = _source_repo_root()
    if source_repo_root is not None:
        sys.path.insert(0, str(source_repo_root))
    from robo_orchard_lab.utils import log_basic_config

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
        "--benchmark",
        type=str,
        default="libero",
        choices=["libero", "libero_plus"],
        help="Benchmark family to evaluate.",
    )
    parser.add_argument(
        "--benchmark_module",
        type=str,
        default=None,
        help=(
            "Optional module override that provides get_benchmark_dict(). "
            "Useful when a LIBERO-Plus checkout uses a custom import path."
        ),
    )
    parser.add_argument(
        "--libero_root",
        type=str,
        default=None,
        help="Local LIBERO checkout root. Defaults to ./LIBERO if present.",
    )
    parser.add_argument(
        "--libero_plus_root",
        type=str,
        default=None,
        help=(
            "Local LIBERO-Plus checkout root. Defaults to ./LIBERO-plus "
            "if present."
        ),
    )
    parser.add_argument(
        "--eval_policy_path",
        type=str,
        default=None,
        help="Path to eval_policy.py. Defaults to this repository's script.",
    )
    parser.add_argument(
        "--policy_config_path",
        type=str,
        default=None,
        help=(
            "Path to holobrain_libero_policy/deploy_policy.yml. Defaults "
            "to this repository's config."
        ),
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
    parser.add_argument("--processes_per_gpu", type=int, default=1)
    parser.add_argument("--save_video", type=parse_bool, default=True)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="List selected tasks and exit before requiring GPUs or models.",
    )

    args = parser.parse_args()

    logger.info("\n" + json.dumps(vars(args), indent=4))
    benchmark_root = _selected_benchmark_root(args)
    benchmark_dict = _get_benchmark_dict(
        args.benchmark,
        args.benchmark_module,
        benchmark_root,
    )
    target_suites = resolve_target_suites(args.task_suite, benchmark_dict)
    all_tasks = build_task_list(args.task_suite, args.task_id, benchmark_dict)
    if not all_tasks:
        raise ValueError(
            f"No tasks selected by task_suite={args.task_suite!r}. "
            f"Available suites: {sorted(benchmark_dict)}"
        )
    if args.dry_run:
        logger.info("Dry run selected tasks: %s", all_tasks)
        raise SystemExit(0)

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs found! Please check CUDA availability.")

    logger.info(
        f"Found {num_gpus} GPUs. Distributing {len(all_tasks)} tasks "
        f"with {args.processes_per_gpu} process(es) per GPU..."
    )

    tasks_allocated = allocate_tasks_to_workers(
        all_tasks,
        num_gpus,
        args.processes_per_gpu,
    )

    start_time = datetime.now()
    if len(all_tasks) > 1:
        manager = multiprocessing.Manager()
        results = manager.dict()
        processes = []

        for gpu_id, gpu_tasks in tasks_allocated:
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

    final_output = {"overall_summary": {}}
    if args.benchmark != "libero_plus":
        final_output["suites_detail"] = nested_results

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
    if args.benchmark == "libero_plus":
        final_output["category_summary"] = summarize_libero_plus_categories(
            results,
            benchmark_dict,
            benchmark_root,
        )

    logger.info("Detailed JSON Results:")
    logger.info(json.dumps(final_output, indent=4))
    logger.info("=" * 50)
