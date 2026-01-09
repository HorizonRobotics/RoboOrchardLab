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

import numpy as np
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.sampler import Sampler

from robo_orchard_lab.distributed.utils import get_dist_info

logger = logging.getLogger(__file__)


class ConcatDatasetWithFlag(ConcatDataset):
    def __init__(self, datasets, mode="concat_flag"):
        super().__init__(datasets)
        self.generate_flag()

    def generate_flag(self):
        flags = []
        for i, dataset in enumerate(self.datasets):
            flag = getattr(dataset, "flag", None)
            if flag is None:
                flag = i
            else:
                assert isinstance(flag, int) and flag >= len(self.datasets), (
                    "Please use a larger integer as the flag "
                    "(currently set to {x}) to avoid conflicts with the "
                    "default value, which may cause unexpected issues."
                )
            flags.append(np.full(len(dataset), flag))
        self.flags = np.concatenate(flags)


class DistributedBatchFlagSampler(Sampler[list[int]]):
    def __init__(
        self,
        data_source,
        batch_size,
        drop_last=False,
        seed=0,
        dataset_sample_weights=None,
    ):
        dist_info = get_dist_info()
        self.rank = dist_info.rank

        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        flags = np.copy(self.data_source.flags)
        assert len(flags) == len(self.data_source)
        self.groups_length = {}
        uniuqe_flags = np.unique(flags)
        for f in uniuqe_flags:
            self.groups_length[f] = (f == flags).sum()
        self.flags = flags
        self.seed = seed + self.rank
        logger.info(
            f"dataset length: {len(self.data_source)}, "
            f"number of batches: {self.__len__()}, "
            f"dataset flags: {uniuqe_flags}"
        )
        self._epoch = 0
        if dataset_sample_weights is not None:
            assert len(dataset_sample_weights) == len(
                self.data_source.datasets
            )
            sum_weights = sum(dataset_sample_weights)
            dataset_sample_weights = [
                x / sum_weights for x in dataset_sample_weights
            ]
            log_info = "\ndataset sample weights: "
            for i, dataset in enumerate(self.data_source.datasets):
                log_info += (
                    f"\n|---{getattr(dataset, 'dataset_name', 'unnamed')}"
                    f": {dataset_sample_weights[i]}"
                )
            logger.info(log_info)
        self.dataset_sample_weights = dataset_sample_weights
        self.reset()

    def reset(self):
        self._epoch += 1
        self.num_batches = 0
        self.batches = {}

    def set_epoch(self, epoch):
        self._epoch = epoch

    def _indices_generator(self):
        if self.dataset_sample_weights is None:
            n = len(self.data_source)
            generator = np.random.default_rng(seed=self.seed + self._epoch)
            yield from generator.permutation(n)
        else:
            lengths = [len(dataset) for dataset in self.data_source.datasets]
            num_dataset = len(lengths)
            prefix_length = np.cumsum([0] + lengths)
            dataset_indices_queues = []
            for i, length in enumerate(lengths):
                tmp = np.random.default_rng(self.seed + i).permutation(length)
                tmp = [x + prefix_length[i] for x in tmp]
                dataset_indices_queues.append(tmp)
            queue_indices = np.zeros(num_dataset, np.int32)
            dataset_epoch = np.zeros(num_dataset, np.int32)
            while True:
                dataset_idx = np.random.choice(
                    num_dataset,
                    p=self.dataset_sample_weights,
                )
                queue = dataset_indices_queues[dataset_idx]
                queue_idx = queue_indices[dataset_idx]
                idx = queue[queue_idx]
                queue_indices[dataset_idx] += 1
                if queue_indices[dataset_idx] >= lengths[dataset_idx]:
                    dataset_epoch[dataset_idx] += 1
                    tmp = np.random.default_rng(
                        self.seed + dataset_epoch[dataset_idx] + dataset_idx
                    ).permutation(lengths[dataset_idx])
                    tmp = [x + prefix_length[dataset_idx] for x in tmp]
                    dataset_indices_queues[dataset_idx] = tmp
                    queue_indices[dataset_idx] = 0
                yield idx

    def __iter__(self):
        generator = self._indices_generator()
        while True:
            try:
                i = next(generator).item()
            except StopIteration:
                if self.drop_last:
                    self.reset()
                    return
                flags = list(self.batches.keys())
                for flag in flags:
                    batch = self.batches.pop(flag)
                    if len(batch) == 0:
                        continue
                    while len(batch) < self.batch_size:
                        batch.extend(batch)
                    self.num_batches += 1
                    yield batch[: self.batch_size]
                self.reset()
                return

            flag = self.flags[i]
            if flag not in self.batches:
                self.batches[flag] = []
            self.batches[flag].append(i)

            if len(self.batches[flag]) == self.batch_size:
                output = self.batches[flag]
                self.batches[flag] = []
                self.num_batches += 1
                yield output

    def __len__(self):
        ret = 0
        for group_len in self.groups_length.values():
            if self.drop_last:
                ret += group_len // self.batch_size
            else:
                ret += (group_len + self.batch_size - 1) // self.batch_size
        return ret
