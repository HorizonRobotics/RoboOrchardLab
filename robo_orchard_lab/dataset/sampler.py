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

from typing import Iterator, Protocol, runtime_checkable

import numpy as np
import pyarrow as pa
import torch
from datasets.arrow_dataset import InMemoryTable, MemoryMappedTable, Table
from torch.utils.data.sampler import Sampler

__all__ = [
    "IndiceTableSampler",
    "ShardedIndiceSampler",
]


@runtime_checkable
class Sized(Protocol):
    """A protocol for object that have a length."""

    def __len__(self) -> int: ...


class IndiceTableSampler(Sampler[int]):
    """Sampler that samples elements from a given list of indices.

    Args:
        indices (Table | list[int] | str | torch.Tensor | np.ndarray): The
            indices to sample from. It can be a pyarrow Table with one column
            of type uint64, a list of integers, a numpy array of integers,
            a torch tensor of integers, or a string representing the path to
            a pyarrow Table file.
        shuffle (bool, optional): If True, then the indices will be shuffled
            before being returned. Default: False.
        generator (torch.Generator, optional): Generator used in sampling.
            Default: None.

    """

    def __init__(
        self,
        indices: Table | list[int] | str | torch.Tensor | np.ndarray,
        shuffle: bool = False,
        generator: torch.Generator | None = None,
    ) -> None:
        self.generator = generator
        self.shuffle = shuffle
        if isinstance(indices, Table):
            self.table = indices
        elif isinstance(indices, (list, np.ndarray, torch.Tensor)):
            self.table = self._list2memtable(indices)
        elif isinstance(indices, str):
            self.table = self._table_from_file(indices)
        else:
            raise TypeError(
                f"indices must be of type Table, list[int], or str, "
                f"but got {type(indices)}"
            )
        if self.table.num_columns != 1:
            raise ValueError(
                f"indices table must have exactly one column, "
                f"but got {self.table.num_columns}"
            )
        if self.table.column(0).type != pa.uint64():
            raise ValueError(
                f"indices table column must be of type uint64, "
                f"but got {self.table.column(0).type}"
            )

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            if self.generator is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
                generator = torch.Generator()
                generator.manual_seed(seed)
            else:
                generator = self.generator

            indices = torch.randperm(len(self), generator=generator)
            for i in indices:
                yield self.table.column(0)[i.item()].as_py()
        else:
            for i in range(len(self)):
                yield self.table.column(0)[i].as_py()

    def _list2memtable(
        self, indices: list[int] | np.ndarray | torch.Tensor
    ) -> InMemoryTable:
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()
        if isinstance(indices, np.ndarray):
            indice_arr = pa.array(indices)
        else:
            indice_arr = pa.array(indices, type=pa.uint64())
        return InMemoryTable.from_arrays([indice_arr], names=["indices"])

    def _table_from_file(self, filepath: str) -> MemoryMappedTable:
        return MemoryMappedTable.from_file(filepath)

    def __len__(self) -> int:
        return len(self.table)


class ShardedIndiceSampler(IndiceTableSampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a :class:`DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    Note:
        * Dataset is assumed to be of constant size.
        * If you are using `accelerate` for distributed training, you
          should not use this sampler, as `accelerate` will automatically
          handle the data sharding!


    Args:
        indices (Table | list[int] | str | torch.Tensor | np.ndarray | int):
            The indices to sample from. It can be a pyarrow Table with one
            column of type uint64, a list of integers, a numpy array of
            integers, a torch tensor of integers, or a string representing
            the path to a pyarrow Table file. If an integer is provided,
            it is treated as the length of the dataset, and the indices
            will be generated as a range from 0 to length-1.
        num_shards (int): Number of shards to divide the dataset into.
        shard_id (int): Index of the current shard.
        contiguous (bool, optional): If True, then the dataset will be split
            into contiguous chunks. If False, then the dataset will be split
            into non-contiguous chunks. Default: True.
        shuffle (bool, optional): If True, then the indices will be shuffled
            before being returned. Default: False.
        generator (torch.Generator, optional): Generator used in sampling.
            Default: None.

    """

    def __init__(
        self,
        indices: Table | list[int] | str | torch.Tensor | np.ndarray | int,
        num_shards: int,
        shard_id: int,
        contiguous: bool = True,
        shuffle: bool = False,
        generator: torch.Generator | None = None,
    ) -> None:
        if isinstance(indices, str):
            indices = MemoryMappedTable.from_file(indices)
        if isinstance(indices, (Table, list, torch.Tensor, np.ndarray)):
            dataset_len = len(indices)
        else:
            dataset_len = indices
        if not 0 <= shard_id < num_shards:
            raise ValueError("shard_id should be in [0, num_shards-1]")
        if contiguous:
            div = dataset_len // num_shards
            mod = dataset_len % num_shards
            start = div * shard_id + min(shard_id, mod)
            end = start + div + (1 if shard_id < mod else 0)
            sliced_indices = (
                indices[start:end]
                if not isinstance(indices, int)
                else np.arange(start, end, dtype=np.uint64)
            )
        else:
            sliced_indices = (
                [
                    indices[i]
                    for i in np.arange(
                        shard_id, len(indices), num_shards, dtype=np.uint64
                    )
                ]
                if not isinstance(indices, int)
                else np.arange(
                    shard_id, dataset_len, num_shards, dtype=np.uint64
                )
            )
        super().__init__(
            sliced_indices,  # type: ignore
            shuffle=shuffle,
            generator=generator,
        )
