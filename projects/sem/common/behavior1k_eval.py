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
import tempfile
import uuid
from collections import defaultdict

import requests
import torch
from filelock import FileLock, Timeout
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES

from robo_orchard_lab.utils import log_basic_config

valid_task_names = list(TASK_NAMES_TO_INDICES)


# behavior1k will download omni.kit when running,
# so add a proxy here
bash_command_template = (
    "export CUDA_VISIBLE_DEVICES={gpu_id} \n"
    "export OMNIGIBSON_APPDATA_PATH=/tmp/appdata_{run_id} \n"
    "xvfb-run --server-num={xvfb_num} python3 OmniGibson/omnigibson/learning/eval_b1k.py "  # noqa: E501
    "  policy=local "
    "  env_wrapper._target_=omnigibson.learning.wrappers.rgbd_obs_wrapper.RGBDObservationWrapper "  # noqa: E501
    "  task.name={task_name} "
    "  log_path={log_path} "
    "  +vlm_ckpt_dir={vlm_ckpt_dir} "
    "  +urdf_dir={urdf_dir} "
    "  +model_path={model_dir} "
    "  +model_processor={model_processor} "
    "  +model_prefix={model_prefix} "
    "  '+instances_to_run={instances_to_run}'"
)


def cal_q_score(metric_dir):
    logger = logging.getLogger(__name__)

    task_scores = defaultdict(list)

    def extract_key(fname: str) -> int:
        # assumes "..._<int>_<something>.json"
        return int(fname.split("_")[-2])

    if not os.path.exists(metric_dir):
        logger.error(f"metrics dir not found: {metric_dir}")
        return

    for fname in sorted(os.listdir(metric_dir), key=extract_key):
        if not fname.endswith(".json"):
            continue

        task = fname.rsplit("_", 2)[0]

        path = os.path.join(metric_dir, fname)
        with open(path, "r") as f:
            data = json.load(f)

        q = data.get("q_score", {}).get("final", 0)
        task_scores[task].append(float(q))
        logger.info(f"{fname}: {q:.4f}")

    task_mean_q = {}
    for task, qs in task_scores.items():
        task_mean_q[task] = sum(qs) / len(qs)

    num_tasks = len(task_mean_q)
    if num_tasks != 50:
        logger.warning(f"[WARN] expected 50 tasks, but got {num_tasks}")

    overall_q = sum(task_mean_q.values()) / num_tasks if num_tasks > 0 else 0.0

    logger.info("===== Task-level Q =====")
    for task in sorted(task_mean_q):
        logger.info(
            f"{task}: mean_q={task_mean_q[task]:.6f} "
            f"(n={len(task_scores[task])})"
        )

    logger.info("\n===== Overall Score =====")
    logger.info(
        f"Overall Q (averaged over {num_tasks} tasks): {overall_q:.6f}"
    )


def download_file(url, file_name, timeout=180):
    logger = logging.getLogger(__name__)

    if os.path.exists(file_name):
        logger.info(f"File existed: {file_name}")
        return

    lock_path = file_name + ".lock"
    lock = FileLock(lock_path, timeout=timeout)

    try:
        with lock:
            if os.path.exists(file_name):
                logger.info(f"File existed: {file_name}")
                return

            temp_dir = os.path.dirname(file_name) or "."
            with tempfile.NamedTemporaryFile(
                delete=False, dir=temp_dir
            ) as tmp_file:
                tmp_path = tmp_file.name
                try:
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            tmp_file.write(chunk)
                    tmp_file.flush()
                    os.fsync(tmp_file.fileno())

                    os.rename(tmp_path, file_name)
                    logger.info(f"Download success: {file_name}")
                except Exception as e:
                    logger.error(f"Download fail: {file_name}, error: {e}")
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    raise
    except Timeout as e:
        logger.error(f"Download timeout: {file_name}")
        raise e


def download_job_ckpt_processor(
    ckpt_url,
    processor_name,
    output_dir="./model",
    model_prefix="model",
    vlm_ckpt_dir=None,
    urdf_dir=None,
):
    logger = logging.getLogger(__name__)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    while ckpt_url.endswith("/"):
        ckpt_url = ckpt_url[:-1]
    model_url = f"{ckpt_url}/model.safetensors"
    model_config_url = f"{ckpt_url}/{model_prefix}.config.json"
    processor_url = "/".join(
        ckpt_url.split("/")[:-2] + [f"{processor_name}.json"]
    )
    logger.info(
        f"model_ckpt: {model_url}\n"
        f"model_config: {model_config_url}\n"
        f"procssor: {processor_url}"
    )
    for url in [model_url, model_config_url, processor_url]:
        file_name = os.path.join(output_dir, url.split("/")[-1])
        # if url.endswith("config.json"):
        #     file_name = file_name.replace(
        #         f"{model_prefix}.config.json", "model.config.json"
        #     )
        download_file(url, file_name)

    target_vlm_ckpt_dir = os.path.join(output_dir, "ckpt")
    if vlm_ckpt_dir is not None and not os.path.exists(target_vlm_ckpt_dir):
        os.symlink(vlm_ckpt_dir, target_vlm_ckpt_dir)

    target_urdf_dir = os.path.join(output_dir, "urdf")
    if urdf_dir is not None and not os.path.exists(target_urdf_dir):
        os.symlink(urdf_dir, target_urdf_dir)


def worker_loop(gpu_id: int, worker_local_id: int, job_queue, args, results):
    logger = logging.getLogger(__file__)

    # Make xvfb server-num stable and unique per (gpu_id, worker_local_id),
    # and still safe if the worker runs multiple jobs.
    # Each job increments an offset to avoid reuse.
    base = 1000 + gpu_id * 100 + worker_local_id * 10
    job_count = 0

    while True:
        job = job_queue.get()
        if job is None:  # sentinel
            job_queue.task_done()
            logger.info(f"[worker] gpu={gpu_id} wid={worker_local_id} exit")
            return

        task_name, inst_list = job
        inst_str = str(inst_list)

        xvfb_num = base + (job_count % 10)  # small rotation per worker
        job_count += 1

        command = bash_command_template.format(
            gpu_id=gpu_id,
            xvfb_num=xvfb_num,
            run_id=uuid.uuid4().hex[:8],
            task_name=task_name,
            log_path=args.log_path,
            vlm_ckpt_dir=args.vlm_ckpt_dir,
            urdf_dir=args.urdf_dir,
            model_dir=args.model_path,
            model_processor=args.model_processor,
            model_prefix=args.model_prefix,
            instances_to_run=inst_str,
        )

        logger.info(
            f"[worker] gpu={gpu_id} "
            f"wid={worker_local_id} "
            f"run: {task_name} {inst_list}"
        )
        logger.info(command)

        ok = True
        try:
            inst_tag = f"{inst_list[0]}_{inst_list[-1]}"
            log_file = os.path.join(
                args.job_log_dir,
                f"job_{task_name}_{inst_tag}.log",
            )

            with open(log_file, "w") as f:
                ret = subprocess.run(
                    command,
                    shell=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

            if ret.returncode != 0:
                ok = False
                logger.error(
                    f"[worker] gpu={gpu_id} wid={worker_local_id} "
                    f"{task_name} {inst_list} failed: {ret.returncode}"
                )
        except Exception as e:
            ok = False
            logger.error(
                f"[worker] gpu={gpu_id} wid={worker_local_id} "
                f"{task_name} {inst_list} crashed: {e}"
            )

        results[(task_name, tuple(inst_list))] = bool(ok)
        job_queue.task_done()


if __name__ == "__main__":
    logger = logging.getLogger(__file__)
    log_basic_config(
        format="%(filename)s:%(lineno)d | %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
        force=True,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--model_processor", type=str, default="behavior1k_processor"
    )
    parser.add_argument("--model_prefix", type=str, default="model_0")
    parser.add_argument("--vlm_ckpt_dir", type=str, default=None)
    parser.add_argument("--urdf_dir", type=str, default=None)

    # task list
    parser.add_argument("--task_names", type=str, required=True)

    # job slicing
    parser.add_argument("--instance_per_job", type=int, default=10)

    parser.add_argument("--max_jobs_per_gpu", type=int, default=1)

    args = parser.parse_args()

    # derive log_path
    parts = args.model_path.strip("/").split("/")
    suffix = [parts[-4], parts[-1]]
    args.log_path = os.path.join("/job_data/", *suffix)

    # job-level log dir
    args.job_log_dir = os.path.join("/job_data", "job_logs")
    os.makedirs(args.job_log_dir, exist_ok=True)

    logger.info("\n" + json.dumps(vars(args), indent=4))

    # download model
    download_job_ckpt_processor(
        ckpt_url=args.model_path,
        processor_name=args.model_processor,
        model_prefix=args.model_prefix,
        output_dir="./sem_eval_model",
        vlm_ckpt_dir=args.vlm_ckpt_dir,
        urdf_dir=args.urdf_dir,
    )

    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No CUDA devices available"

    # validate tasks
    task_names = [t.strip() for t in args.task_names.split(",") if t.strip()]
    for task in task_names:
        if task not in valid_task_names:
            raise ValueError(f"Found invalid task: {task}")

    # default 10 episode per task for eval
    total_instances = 10
    all_instances = list(range(total_instances))

    chunk_size = int(args.instance_per_job)
    assert 1 <= chunk_size <= total_instances, (
        f"instance_per_job must be in [1, {total_instances}]"
    )

    # build jobs in (task, instance_list)
    jobs = []
    for task in task_names:
        for i in range(0, total_instances, chunk_size):
            jobs.append((task, all_instances[i : i + chunk_size]))

    logger.info("====== Job Plan ======")
    for j, (t, inst) in enumerate(jobs):
        # this line is only informational now; real scheduling is dynamic
        logger.info(f"job={j:03d} task={t} inst={inst}")
    logger.info("======================")

    max_jobs_per_gpu = int(args.max_jobs_per_gpu)
    assert max_jobs_per_gpu >= 1, "--max_jobs_per_gpu must be >= 1"

    manager = multiprocessing.Manager()
    results = manager.dict()

    job_queue = multiprocessing.JoinableQueue(maxsize=0)

    # enqueue jobs
    for job in jobs:
        job_queue.put(job)

    # start workers: num_gpus * max_jobs_per_gpu
    workers = []
    for gpu_id in range(num_gpus):
        for wid in range(max_jobs_per_gpu):
            p = multiprocessing.Process(
                target=worker_loop,
                args=(gpu_id, wid, job_queue, args, results),
            )
            p.daemon = False
            p.start()
            workers.append(p)

    # after all jobs enqueued, push sentinels (one per worker)
    for _ in range(len(workers)):
        job_queue.put(None)

    # wait all queue tasks done
    job_queue.join()

    # join all workers
    for p in workers:
        p.join()
        if p.exitcode != 0:
            logger.error(f"Worker crashed with exitcode={p.exitcode}")

    # summarize results
    results = dict(results)
    success = sum(bool(v) for v in results.values())
    logger.info(
        f"Success: {success}/{len(jobs)} (results_recorded={len(results)})"
    )

    # compute q_score
    cal_q_score(os.path.join(args.log_path, "metrics"))
