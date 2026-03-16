# Project RoboOrchard
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
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

import logging
import os
import time
from datetime import datetime

import fsspec
import tqdm
import yaml
from robo_orchard_core.utils.task_executor import (
    OrderedTaskExecutor,
    TaskQueueFulledError,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

os.environ["AIDI_FS_TYPE"] = "S3"
os.environ["AIDI_ENDPOINT"] = "http://aidi.hobot.cc"


def get_fsspec_filesystem(path):
    target_splits = path.split("://")

    if len(target_splits) == 1:
        target_splits.insert(0, "file")
        target_splits[-1] = os.path.abspath(target_splits[-1])

    fs: fsspec.AbstractFileSystem = fsspec.filesystem(
        protocol=target_splits[0]
    )
    return fs


def copy_file(src, target, retry_num=3) -> dict:
    def try_to_copy():
        with fsspec.open(src, "rb") as src_file:
            with fsspec.open(target, "wb") as target_file:
                # source file is too large, so we read and write in chunks
                # use tqdm to show progress
                with tqdm.tqdm(
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:

                    def update_progress(bytes_read):
                        pbar.update(bytes_read)

                    while True:
                        chunk = src_file.read(1024 * 1024 * 64)  # 64MB
                        if not chunk:
                            break
                        target_file.write(chunk)
                        update_progress(len(chunk))

    retry_sleep = 5
    success = False
    for i in range(retry_num):
        try:
            try_to_copy()
            success = True
            break
        except Exception as e:
            logger.error(f"Error copying {src} to {target}: {e}")
            if i < retry_num - 1:
                logger.info(f"Retrying in {retry_sleep} seconds...")
                time.sleep(retry_sleep)
            else:
                logger.error(
                    f"Failed to copy {src} to {target} after {i + 1} retries."
                )

    return success


def search_valid_data(date_prefix, user_names, task_names, data_root):
    if isinstance(date_prefix, str):
        date_prefix = [x.strip() for x in date_prefix.split(",") if x.strip()]

    valid_data_episodes = []
    for data_time in os.listdir(data_root):
        if not any([data_time.startswith(x) for x in date_prefix]):
            continue

        for user_name in user_names:
            for task_name in task_names:
                data_path = os.path.join(
                    data_root, data_time, "data", user_name, task_name
                )
                if not os.path.exists(data_path):
                    continue
                valid_data_episodes.extend(
                    [os.path.join(data_path, x) for x in os.listdir(data_path)]
                )
    return valid_data_episodes


if __name__ == "__main__":
    import argparse

    try:
        from robo_orchard_lab.utils import log_basic_config

        log_basic_config(
            format="%(asctime)s %(levelname)s-%(lineno)d: %(message)s",
            level=logging.INFO,
        )
    except ImportError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--user_names", type=str)
    parser.add_argument("--task_names", type=str)
    parser.add_argument("--date_prefix", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--token", type=str, default=None)
    args = parser.parse_args()

    date_prefix = args.date_prefix
    if date_prefix is None:
        date_prefix = [datetime.now().strftime("%Y_%m_%d")]
    else:
        date_prefix = [x.strip() for x in date_prefix.split(",") if x.strip()]

    user_names = args.user_names.split(",")
    task_names = args.task_names.split(",")
    valid_data_episodes = search_valid_data(
        date_prefix, user_names, task_names, args.input_path
    )
    num_total_episode = len(valid_data_episodes)
    logger.info(f"Number of valid episodes: {num_total_episode}")
    for user_name in user_names:
        for task_name in task_names:
            logger.info(
                f"Number of valid episodes [{user_name} - {task_name}]: "
                f"{len([x for x in valid_data_episodes if task_name in x and user_name in x])}"  # noqa: E501
            )

    output_path = args.output_path
    if output_path.startswith("/horizon-bucket"):
        output_path = "dmpv2:/" + output_path[len("/horizon-bucket") :]

    token = args.token
    if token is None:
        token = yaml.safe_load(
            open(
                os.path.join(os.path.expanduser("~"), "/.aidi/config.yaml"),
                "r",
            )
        )["token"]
    os.environ["AIDI_TOKEN"] = token

    num_workers = args.num_workers
    executor = OrderedTaskExecutor(
        fn=copy_file,
        num_workers=num_workers,
        max_queue_size=num_workers * 4,
    )
    for i, episode in enumerate(valid_data_episodes):
        logger.info(f"Start copy [{i + 1} / {num_total_episode}]: {episode}")
        for file in os.listdir(episode):
            src_file = os.path.join(episode, file)
            target_file = os.path.join(
                output_path, *src_file.split(os.sep)[-4:]
            )
            try:
                executor.put(
                    src=src_file,
                    target=target_file,
                )
            except TaskQueueFulledError:
                result: dict = executor.get(block=True)
                executor.put(
                    src=src_file,
                    target=target_file,
                )

    while executor.buf_size > 0:
        result: dict = executor.get(block=True)
