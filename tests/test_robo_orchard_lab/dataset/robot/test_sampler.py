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

import torch

from robo_orchard_lab.dataset.sampler import (
    IndiceTableSampler,
    ShardedIndiceSampler,
)


class TestIndiceTableSampler:
    def test_len(self):
        sampler = IndiceTableSampler(list(range(10)), shuffle=False)
        assert len(sampler) == 10

        sampler = IndiceTableSampler(list(range(5)), shuffle=False)
        assert len(sampler) == 5

        sampler = IndiceTableSampler([], shuffle=False)
        assert len(sampler) == 0

    def test_iter_no_shuffle(self):
        sampler = IndiceTableSampler(list(range(10)), shuffle=False)
        indices = list(sampler)
        assert indices == list(range(10))

        sampler = IndiceTableSampler(list(range(5)), shuffle=False)
        indices = list(sampler)
        assert indices == list(range(5))

        sampler = IndiceTableSampler([], shuffle=False)
        indices = list(sampler)
        assert indices == []

    def test_iter_with_shuffle(self):
        sampler = IndiceTableSampler(
            list(range(10)),
            shuffle=True,
            generator=torch.Generator().manual_seed(42),
        )
        indices = list(sampler)
        assert sorted(indices) == list(range(10))
        assert indices != list(range(10))


class TestShardedIndiceSampler:
    def test_len(self):
        sampler = ShardedIndiceSampler(
            list(range(10)),
            num_shards=3,
            index=0,
            shuffle=False,
        )
        assert len(sampler) == 4  # 0, 3, 6, 9

        sampler = ShardedIndiceSampler(
            list(range(10)),
            num_shards=3,
            index=1,
            shuffle=False,
        )
        assert len(sampler) == 3  # 1, 4, 7

        sampler = ShardedIndiceSampler(
            list(range(10)),
            num_shards=3,
            index=2,
            shuffle=False,
        )
        assert len(sampler) == 3  # 2, 5, 8

        sampler = ShardedIndiceSampler(
            list(range(5)),
            num_shards=2,
            index=0,
            shuffle=False,
        )
        assert len(sampler) == 3  # 0, 2, 4

        sampler = ShardedIndiceSampler(
            list(range(5)),
            num_shards=2,
            index=1,
            shuffle=False,
        )
        assert len(sampler) == 2  # 1, 3

        sampler = ShardedIndiceSampler(
            [],
            num_shards=2,
            index=0,
            shuffle=False,
        )
        assert len(sampler) == 0

    def test_iter_no_shuffle_contiguous(self):
        sampler = ShardedIndiceSampler(
            list(range(10)),
            num_shards=3,
            index=0,
            shuffle=False,
        )
        indices = list(sampler)
        assert indices == [0, 1, 2, 3]

        sampler = ShardedIndiceSampler(
            list(range(10)),
            num_shards=3,
            index=1,
            shuffle=False,
        )
        indices = list(sampler)
        assert indices == [4, 5, 6]

        sampler = ShardedIndiceSampler(
            list(range(10)),
            num_shards=3,
            index=2,
            shuffle=False,
        )
        indices = list(sampler)
        assert indices == [7, 8, 9]

    def test_iter_no_shuffle_not_contiguous(self):
        sampler = ShardedIndiceSampler(
            list(range(1, 11)),
            num_shards=3,
            index=0,
            contiguous=False,
            shuffle=False,
        )
        indices = list(sampler)
        assert indices == [1, 4, 7, 10]

        sampler = ShardedIndiceSampler(
            list(range(1, 11)),
            num_shards=3,
            index=1,
            shuffle=False,
            contiguous=False,
        )
        indices = list(sampler)
        assert indices == [2, 5, 8]

        sampler = ShardedIndiceSampler(
            list(range(1, 11)),
            num_shards=3,
            index=2,
            shuffle=False,
            contiguous=False,
        )
        indices = list(sampler)
        assert indices == [3, 6, 9]

        sampler = ShardedIndiceSampler(
            11,
            num_shards=3,
            index=0,
            contiguous=False,
            shuffle=False,
        )
        indices = list(sampler)
        assert indices == [0, 3, 6, 9]
