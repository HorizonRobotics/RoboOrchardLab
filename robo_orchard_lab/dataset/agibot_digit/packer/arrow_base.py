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

import logging
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

import datasets as hg_datasets
from tqdm import tqdm

from robo_orchard_lab.dataset.robot.packaging import (
    DatasetPackaging,
    EpisodePackaging,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ROPackConfig:
    dataset_name: str

    # path
    input_dir: str
    output_dir: str
    cached_meta_path: str

    # robot
    robot_name: str
    urdf_path: str

    # sharding
    force_overwrite: bool = False
    max_shard_size: str = "8GB"
    writer_batch_size: int = 2000

    # distributed
    num_jobs: int = 1
    job_idx: int = 0


class BaseRODataPacker:

    def __init__(self, cfg: ROPackConfig):
        self.cfg = cfg
        logger.info(f"Initialized packer with config: {self.cfg}")

    @abstractmethod
    def get_dataset_features(self) -> hg_datasets.Features:
        pass

    @abstractmethod
    def collect_all_metas(self) -> list[dict]:
        pass

    @abstractmethod
    def build_episode(self, meta: dict) -> EpisodePackaging:
        pass

    def build_episodes(self, metas: list[dict]) -> list[EpisodePackaging]:
        episodes = []
        for meta in metas:
            try:
                episode = self.build_episode(meta)
            except Exception as e:
                logger.error(f"Failed to build episode for meta {meta}: {e}")
                continue
            episodes.append(episode)
        return episodes

    def get_shard_slice(self, total: int) -> slice:
        if self.cfg.num_jobs <= 0:
            raise ValueError("--num_jobs must be > 0")
        if not (0 <= self.cfg.job_idx < self.cfg.num_jobs):
            raise ValueError(f"--job_idx must be in [0, {self.cfg.num_jobs})")

        chunk_size = (total + self.cfg.num_jobs - 1) // self.cfg.num_jobs
        start = self.cfg.job_idx * chunk_size
        end = min(start + chunk_size, total)

        return slice(start, end)

    def pack(self):
        output_path = Path(self.cfg.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        job_name = f"part-{self.cfg.job_idx:05d}-of-{self.cfg.num_jobs:05d}"
        output_path = output_path / job_name

        logger.info("Start collect all episode metas")
        metas = self.collect_all_metas()
        logger.info(f"Found {len(metas)} episodes in {self.cfg.dataset_name}")

        selected_slice = self.get_shard_slice(len(metas))
        selected_metas = metas[selected_slice]

        if len(selected_metas) == 0:
            logger.warning(
                f"Job {self.cfg.job_idx}/{self.cfg.num_jobs} has no episodes to pack."
            )
            return
        else:
            logger.info(
                f"Job {self.cfg.job_idx}/{self.cfg.num_jobs} selected {len(selected_metas)} episodes"
                f"{selected_slice.start}:{selected_slice.stop} / total {len(metas)}"
            )

        selected_episodes = self.build_episodes(selected_metas)
        selected_episodes = tqdm(selected_episodes, desc="Packaging episodes")

        packaging = DatasetPackaging(features=self.get_dataset_features())
        packaging.packaging(
            episodes=selected_episodes,
            dataset_path=str(output_path),
            max_shard_size=self.cfg.max_shard_size,
            force_overwrite=self.cfg.force_overwrite,
            writer_batch_size=self.cfg.writer_batch_size,
        )

        logger.info(f"Success! RODataset save to: {output_path}")
        logger.info(
            f"Packed {len(selected_metas)} episodes by job {self.cfg.job_idx} / {self.cfg.num_jobs}"
        )
