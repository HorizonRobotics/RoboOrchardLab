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

"""Episode-ordered batch sampler for MemoryVLA's ``stream`` memory.

The host's default ``DistributedBatchFlagSampler`` draws a full random
permutation (dataset_wrapper.py:133). That is fine for a single-frame model
and useless for a memory bank: with shuffled frames the bank is keyed by an
episode it sees once and then never again, so it retrieves nothing and the
method silently degenerates to an identity-ish transform.

This sampler instead shuffles *episodes* and walks each one forward in time,
emitting ``batch_size`` consecutive frames of a single episode per batch --
exactly the input ``dataloader_type="stream"`` assumes.

It wraps rather than modifies the host sampler: the host's own sampler is
untouched, and this one is selected from config.
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

import numpy as np
from torch.utils.data import Sampler

__all__ = ["MemoryVLAEpisodeStreamBatchSampler"]

logger = logging.getLogger(__name__)


def _episode_spans(dataset) -> list[tuple[int, int]]:
    """Contiguous ``[start, end)`` global index spans, one per episode.

    Uses ``_get_indices``, which every LMDB manipulation dataset here exposes
    and which maps a global index to ``(lmdb_index, episode_index,
    step_index)`` without touching image data
    (robodojo_lmdb_dataset.py:152). Frames of one episode are contiguous and
    step-ordered in the global index -- verified on swap_T, where indices
    0,1,2 map to steps 0,1,2 of episode 0.
    """
    subsets = getattr(dataset, "datasets", None)
    if subsets is None:
        subsets, offsets = [dataset], [0]
    else:
        offsets, acc = [], 0
        for d in subsets:
            offsets.append(acc)
            acc += len(d)

    spans: list[tuple[int, int]] = []
    for sub, offset in zip(subsets, offsets):
        get = getattr(sub, "_get_indices", None)
        if get is None:
            raise TypeError(
                f"{type(sub).__name__} has no `_get_indices`, so episode "
                "boundaries cannot be derived. MemoryVLA's stream memory "
                "needs episode-ordered batches; use a dataset that exposes "
                "it, or switch the bank to dataloader_type='group'."
            )
        n = len(sub)
        prev_key = None
        start = 0
        for i in range(n):
            lmdb_i, ep_i, _step = get(i)
            key = (lmdb_i, ep_i)
            if prev_key is None:
                prev_key = key
            elif key != prev_key:
                spans.append((offset + start, offset + i))
                start = i
                prev_key = key
        if n:
            spans.append((offset + start, offset + n))
    return spans


class MemoryVLAEpisodeStreamBatchSampler(Sampler[list[int]]):
    """Yield batches of consecutive frames drawn from a single episode.

    Args:
        data_source: the training dataset.
        batch_size: frames per batch. All of them come from one episode.
        drop_last: drop an episode's trailing partial batch. Default True,
            matching the host dataloader, and it also keeps every batch a
            uniform size for the bank.
        seed: base seed for the episode shuffle.
        num_replicas / rank: distributed sharding, by episode rather than by
            frame, so an episode never straddles two ranks.
        allow_partial_episode_batches: keep the trailing partial batch. Off by
            default; turning it on makes batch sizes ragged.
    """

    def __init__(
        self,
        data_source,
        batch_size: int,
        drop_last: bool = True,
        seed: int = 0,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        allow_partial_episode_batches: bool = False,
    ):
        super().__init__(data_source)
        if num_replicas is None or rank is None:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
                rank = dist.get_rank()
            else:
                num_replicas, rank = 1, 0

        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last and not allow_partial_episode_batches
        self.seed = seed
        self.num_replicas = num_replicas
        self.rank = rank
        self._epoch = 0

        spans = _episode_spans(data_source)
        # shard by episode; an episode must not straddle ranks
        self.spans = spans[rank::num_replicas]
        self._num_batches = sum(
            self._batches_in(end - start) for start, end in self.spans
        )
        logger.info(
            "MemoryVLAEpisodeStreamBatchSampler: %d episodes total, %d on "
            "rank %d/%d, %d batches of %d",
            len(spans), len(self.spans), rank, num_replicas,
            self._num_batches, batch_size,
        )
        if not self.spans:
            raise ValueError(
                "no episodes on this rank -- the dataset yielded no episode "
                "spans, so stream memory would never accumulate."
            )

    def _batches_in(self, n: int) -> int:
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def reset(self) -> None:
        self._epoch += 1

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        order = rng.permutation(len(self.spans))
        for j in order:
            start, end = self.spans[j]
            # forward in time within the episode -- never shuffled
            for b in range(start, end, self.batch_size):
                batch = list(range(b, min(b + self.batch_size, end)))
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch
        self._epoch += 1
