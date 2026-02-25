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
import os

from utils import load_config

from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseLmdbManipulationDataset,
)
from robo_orchard_lab.utils import log_basic_config

logger = logging.getLogger(__file__)


def main(args):
    os.makedirs(args.workspace, exist_ok=True)
    config = load_config(args.config)
    _config = config.config
    if not args.vis_validation:
        build_dataset = config.build_training_dataset
        if args.dataset_names:
            _config["training_datasets"] = args.dataset_names
    else:
        build_dataset = config.build_validation_dataset
        if args.dataset_names:
            _config["validation_datasets"] = args.dataset_names

    concat_dataset = build_dataset(_config)
    for dataset in concat_dataset.datasets:
        if isinstance(dataset, BaseLmdbManipulationDataset):
            if not hasattr(dataset, "visualize"):
                logger.warning(f"{dataset} do not has visualize function")
                continue
            if not args.manual:
                num_vis = 0
                for episode_idx in range(
                    0, dataset.num_episode, args.episode_interval
                ):
                    dataset.visualize(
                        episode_idx, output_path=args.workspace, **args.kwargs
                    )
                    num_vis += 1
                    if (
                        args.max_episode is not None
                        and num_vis > args.max_episode
                    ):
                        break
            else:
                while True:
                    user_input = input(
                        "input episode_idx ('q'->quit, 'c'->next dataset): "
                    )
                    if user_input.lower() == "q":
                        return
                    elif user_input.lower() == "c":
                        break
                    dataset.visualize(
                        int(user_input),
                        output_path=args.workspace,
                        **args.kwargs,
                    )
        else:
            logger.warning(
                f"Visualization of {type(dataset)} is not supported."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--dataset_names", type=str, nargs="+")
    parser.add_argument("--workspace", type=str, default="./workspace")
    parser.add_argument("--vis_validation", action="store_true")
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--episode_interval", type=int, default=1)
    parser.add_argument("--max_episode", type=int, default=None)
    parser.add_argument("--kwargs", type=json.loads, default="{}")
    args = parser.parse_args()

    log_basic_config(
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d | %(message)s",  # noqa: E501
        level=logging.INFO,
    )
    main(args)
