# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
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

import numpy as np
from datasets import Dataset as HFDataset
from robo_orchard_core.utils.config import ClassType

from robo_orchard_lab.dataset.robot.row_sampler import (
    MultiRowSampler,
    MultiRowSamplerConfig,
)

__all__ = [
    "EpisodeChunkSampler",
    "EpisodeChunkSamplerConfig",
]


class EpisodeChunkSampler(MultiRowSampler):
    def __init__(self, cfg: "EpisodeChunkSamplerConfig") -> None:
        self.cfg = cfg
        self.hist_steps = cfg.hist_steps
        self.pred_steps = cfg.pred_steps
        self.chunk_size = self.hist_steps + self.pred_steps

    @property
    def column_rows_keys(self) -> dict[str, list[str]]:
        ret = {}
        for column in self.cfg.target_columns:
            ret[column] = [f"chunk_row_{i}" for i in range(self.chunk_size)]
        return ret

    def sample_row_idx(
        self,
        index_dataset: HFDataset,
        index: int,
    ) -> dict[str, list[int | None]]:
        cur_row = index_dataset[index]
        cur_episode_idx = cur_row["episode_index"]
        dataset_len = len(index_dataset)

        raw_start_idx = index - self.hist_steps + 1
        raw_end_idx = index + self.pred_steps

        start_idx = max(raw_start_idx, 0)
        while start_idx < dataset_len:
            start_row = index_dataset[start_idx]
            if start_row["episode_index"] == cur_episode_idx:
                break
            start_idx += 1

        end_idx = min(raw_end_idx, dataset_len - 1)
        while end_idx >= 0:
            end_row = index_dataset[end_idx]
            if end_row["episode_index"] == cur_episode_idx:
                break
            end_idx -= 1

        raw_indices = np.arange(raw_start_idx, raw_end_idx + 1)
        padded_indices = np.clip(raw_indices, start_idx, end_idx)

        chunk_indices: list[int] = padded_indices.tolist()

        ret = {}
        for column in self.cfg.target_columns:
            ret[column] = chunk_indices

        return ret


class EpisodeChunkSamplerConfig(MultiRowSamplerConfig[EpisodeChunkSampler]):
    """Configuration for the EpisodeChunkSampler."""

    class_type: ClassType[EpisodeChunkSampler] = EpisodeChunkSampler

    target_columns: list[str]

    hist_steps: int = 1
    pred_steps: int = 64
