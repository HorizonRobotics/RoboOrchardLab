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
import os

from pydantic import Field
from robo_orchard_core.utils.cli import SettingConfig, pydantic_from_argparse

from robo_orchard_lab.dataset.robot.dataset import RODataset
from robo_orchard_lab.dataset.robot.merge import merge_datasets


class Config(SettingConfig):
    dataset_paths: list[str] = Field(
        description="The dataset paths to merge, separated by comma. "
        "For example, ['/path/to/dataset1','/path/to/dataset2']",
    )
    target_path: str = Field(
        description="The target path to save the merged dataset. "
        "For example, /path/to/merged_dataset",
    )
    force_overwrite: bool = Field(
        description="Whether to overwrite the target path if it exists.",
        default=False,
    )
    max_shard_size: str | int = Field(
        default="8000MB",
        description="The maximum size of each shard. "
        "This can be a string (e.g., '8000MB') or an integer "
        "(e.g., 8000 * 1024 * 1024 for 8000MB).",
    )
    batch_size: int | None = Field(
        default=None,
        description="The batch size to use when saving the dataset. "
        "Large batch sizes can improve performance but require more memory. "
        "If None, the default batch size from Hugging Face Datasets will "
        "be used.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=Config.__doc__)
    try:
        args: Config = pydantic_from_argparse(Config, parser)
    except SystemExit as e:
        # Handle the case where the script is run with --help
        if e.code == 2:
            parser.print_help()
        exit(0)
    if args.force_overwrite is False and os.path.exists(args.target_path):
        raise FileExistsError(
            f"Target path {args.target_path} already exists. "
            "Use --force_overwrite to overwrite it.",
        )

    datasets = [RODataset(path) for path in args.dataset_paths]

    merge_datasets(
        datasets=datasets,
        target_path=args.target_path,
        max_shard_size=args.max_shard_size,
        batch_size=args.batch_size,
    )
