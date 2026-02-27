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
import os
import random
import string

import numpy as np
import pytest
import torch

from robo_orchard_lab.dataset.sampler import (
    IndiceTable,
    IndiceTableSampler,
)


class TestIndiceTable:
    @pytest.mark.parametrize(
        "data",
        [
            list(range(10)),
            np.arange(10),
            torch.arange(10),
            10,  # This will create a table with indices from 0 to 9
        ],
    )
    def test_table_init(
        self, data: list[int] | np.ndarray | torch.Tensor | int
    ):
        table = IndiceTable(data)
        assert len(table) == 10
        assert table[0] == 0
        assert table[5] == 5
        assert table[9] == 9

    def test_table_iterable(self):
        table = IndiceTable(list(range(10)))
        indices = list(table)
        assert indices == list(range(10))

    def test_table_save(self, tmp_local_folder):
        table = IndiceTable(list(range(10)))
        filepath = os.path.join(
            tmp_local_folder,
            "test_table_save"
            + "".join(random.choices(string.ascii_lowercase, k=8)),
        )

        table.save_to_file(filepath)

        new_table = IndiceTable(filepath)
        assert len(new_table) == 10
        for i in range(10):
            assert new_table[i] == i

        assert table.indice_source == "memory"

        table.save_to_file(filepath, reload=True)
        assert table.indice_source != "memory"

    def test_shuffle(self):
        table = IndiceTable(list(range(20)))
        old_list = table.to_pylist()
        new_table = table.shuffle()
        new_list = new_table.to_pylist()
        assert sorted(new_list) == old_list
        assert new_list != old_list
        # print(table.table.to_pylist())


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

        sampler.shuffle_indices()
        indices = list(sampler)
        assert indices != list(range(10))

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
        full_sampler = IndiceTableSampler(list(range(10)), shuffle=False)

        sampler = full_sampler.shard(
            num_shards=3,
            shard_id=0,
        )
        assert len(sampler) == 4  # 0, 3, 6, 9

        sampler = full_sampler.shard(
            num_shards=3,
            shard_id=1,
        )
        assert len(sampler) == 3  # 1, 4, 7

        sampler = full_sampler.shard(
            num_shards=3,
            shard_id=2,
        )
        assert len(sampler) == 3  # 2, 5, 8

        full_sampler = IndiceTableSampler(list(range(5)), shuffle=False)
        sampler = full_sampler.shard(
            num_shards=2,
            shard_id=0,
        )
        assert len(sampler) == 3  # 0, 2, 4

        sampler = full_sampler.shard(
            num_shards=2,
            shard_id=1,
        )
        assert len(sampler) == 2  # 1, 3
        full_sampler = IndiceTableSampler([], shuffle=False)
        sampler = full_sampler.shard(
            num_shards=2,
            shard_id=0,
        )
        assert len(sampler) == 0

    def test_iter_no_shuffle_contiguous(self):
        full_sampler = IndiceTableSampler(
            list(range(10)),
            shuffle=False,
        )
        sampler = full_sampler.shard(
            num_shards=3,
            shard_id=0,
        )
        indices = list(sampler)
        assert indices == [0, 1, 2, 3]

        sampler = full_sampler.shard(
            num_shards=3,
            shard_id=1,
        )
        indices = list(sampler)
        assert indices == [4, 5, 6]

        sampler = full_sampler.shard(
            num_shards=3,
            shard_id=2,
        )
        indices = list(sampler)
        assert indices == [7, 8, 9]

    def test_iter_no_shuffle_not_contiguous(self):
        full_sampler = IndiceTableSampler(
            list(range(1, 11)),
            shuffle=False,
        )
        sampler = full_sampler.shard(
            num_shards=3,
            shard_id=0,
            contiguous=False,
        )
        indices = list(sampler)
        assert indices == [1, 4, 7, 10]

        sampler = full_sampler.shard(
            num_shards=3,
            shard_id=1,
            contiguous=False,
        )
        indices = list(sampler)
        assert indices == [2, 5, 8]

        sampler = full_sampler.shard(
            num_shards=3,
            shard_id=2,
            contiguous=False,
        )
        indices = list(sampler)
        assert indices == [3, 6, 9]
        full_sampler = IndiceTableSampler(
            11,
            shuffle=False,
        )
        sampler = full_sampler.shard(
            num_shards=3,
            shard_id=0,
            contiguous=False,
        )
        indices = list(sampler)
        assert indices == [0, 3, 6, 9]

    def test_shard_even_drop_last(self):
        full_sampler = IndiceTableSampler(
            list(range(10)),
            shuffle=False,
        )
        sampler = full_sampler.shard(
            num_shards=3,
            shard_id=0,
            contiguous=False,
            shard_strategy="drop_last",
        )
        indices = list(sampler)
        assert indices == [0, 3, 6]

        sampler = full_sampler.shard(
            num_shards=3,
            shard_id=1,
            contiguous=False,
            shard_strategy="drop_last",
        )
        indices = list(sampler)
        assert indices == [1, 4, 7]

        sampler = full_sampler.shard(
            num_shards=3,
            shard_id=2,
            contiguous=False,
            shard_strategy="drop_last",
        )
        indices = list(sampler)
        assert indices == [2, 5, 8]

    def test_shard_even_drop_last_contiguous(self):
        full_sampler = IndiceTableSampler(
            list(range(10)),
            shuffle=False,
        )
        sampler = full_sampler.shard(
            num_shards=3,
            shard_id=0,
            contiguous=True,
            shard_strategy="drop_last",
        )
        indices = list(sampler)
        assert indices == [0, 1, 2]

        sampler = full_sampler.shard(
            num_shards=3,
            shard_id=1,
            contiguous=True,
            shard_strategy="drop_last",
        )
        indices = list(sampler)
        assert indices == [3, 4, 5]

        sampler = full_sampler.shard(
            num_shards=3,
            shard_id=2,
            contiguous=True,
            shard_strategy="drop_last",
        )
        indices = list(sampler)
        assert indices == [6, 7, 8]

    def test_shard_even_pad_last(self):
        full_sampler = IndiceTableSampler(
            list(range(10)),
            shuffle=False,
        )
        sampler = full_sampler.shard(
            num_shards=3,
            shard_id=0,
            contiguous=False,
            shard_strategy="pad_last",
        )
        indices = list(sampler)
        assert indices == [0, 3, 6, 9]

        sampler = full_sampler.shard(
            num_shards=3,
            shard_id=1,
            contiguous=False,
            shard_strategy="pad_last",
        )
        indices = list(sampler)
        assert indices == [1, 4, 7, 0]

        sampler = full_sampler.shard(
            num_shards=3,
            shard_id=2,
            contiguous=False,
            shard_strategy="pad_last",
        )
        indices = list(sampler)
        assert indices == [2, 5, 8, 1]

    def test_shard_even_pad_last_contiguous(self):
        full_sampler = IndiceTableSampler(
            list(range(10)),
            shuffle=False,
        )
        sampler = full_sampler.shard(
            num_shards=3,
            shard_id=0,
            contiguous=True,
            shard_strategy="pad_last",
        )
        indices = list(sampler)
        assert indices == [0, 1, 2, 3]

        sampler = full_sampler.shard(
            num_shards=3,
            shard_id=1,
            contiguous=True,
            shard_strategy="pad_last",
        )
        indices = list(sampler)
        assert indices == [4, 5, 6, 7]

        sampler = full_sampler.shard(
            num_shards=3,
            shard_id=2,
            contiguous=True,
            shard_strategy="pad_last",
        )
        indices = list(sampler)
        assert indices == [8, 9, 0, 1]
