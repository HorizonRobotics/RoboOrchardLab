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
            flags.append(np.full(len(dataset), i))
        self.flags = np.concatenate(flags)


class DistributedBatchFlagSampler(Sampler[list[int]]):
    def __init__(self, data_source, batch_size, drop_last=False, seed=0):
        dist_info = get_dist_info()
        self.rank = dist_info.rank

        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        flags = np.copy(self.data_source.flags)
        assert len(flags) == len(self.data_source)
        self.groups_length = {}
        for i in range(len(flags)):
            if i == 0:
                last = 0
                continue
            elif flags[i] != flags[i - 1]:
                self.groups_length[flags[i - 1]] = i - last
                last = i
        self.groups_length[flags[-1]] = len(flags) - last
        self.flags = flags
        self.seed = seed + self.rank
        logger.info(
            f"dataset length: {len(self.data_source)}, "
            f"number of batches: {self.__len__()}"
        )
        self._epoch = 0
        self.reset()

    def reset(self):
        self._epoch += 1
        self.num_batches = 0
        self.batches = {}

    def set_epoch(self, epoch):
        self._epoch = epoch

    def _indices_queue_generator(self):
        n = len(self.data_source)
        generator = np.random.default_rng(seed=self.seed + self._epoch)
        yield from generator.permutation(n)

    def __iter__(self):
        generator = self._indices_queue_generator()
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
