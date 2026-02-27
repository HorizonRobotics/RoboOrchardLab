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

from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import Callable, Generic, Iterable, Iterator

import numpy as np
import torch
from pydantic import Field
from robo_orchard_core.utils.config import ClassType, Config
from torch.utils.data import (
    Dataset as TorchDataset,
    IterableDataset as TorchIterableDataset,
)

from robo_orchard_lab.dataset.robot._hf_dataset import (
    add_hf_iterable_cls as _add_hf_iterable_cls,
)
from robo_orchard_lab.dataset.robot.dataset import (
    DatasetType,
    Features,
    RODataset,
    get_row_num_from_dataset_info,
)
from robo_orchard_lab.dataset.sampler import (
    IndiceTable,
    IndiceTableSampler,
    ShardStrategy,
    Sized,
)

__all__ = [
    "IterableDatasetMixin",
    "DatasetWithIndices",
    "IterableWithLenDataset",
    "ShardConfig",
    "BatchLoaderConfig",
    "DatasetItem",
    "RODatasetItem",
    "DictIterableDataset",
]


class IterableDatasetMixin(metaclass=ABCMeta):
    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abstractmethod
    def shard(self, num_shards: int, shard_id: int):
        """Shard the dataset into multiple shards.

        Args:
            num_shards (int): The total number of shards to create.
            shard_id (int): The ID of the shard to return. Must be in the
                range [0, num_shards - 1].
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def total_iterator_length(self) -> int:
        """Get the total number of data samples in the iterator."""
        raise NotImplementedError

    @property
    @abstractmethod
    def total_dataset_length(self) -> int:
        """Get the total length of the underlying dataset."""
        raise NotImplementedError

    @abstractmethod
    def get_total_batch_num(
        self, num_workers: int, batch_size: int = 1, drop_last: bool = False
    ) -> int:
        """Calculate the total number of batches for the dataset.

        Pytorch `DataLoader` with multiple workers will shard the dataset into
        `num_workers` shards, and the default method to calculate the total
        number of batches does not consider the sharding, which will cause
        inaccurate total batch number when using multiple workers. This method
        provides a way to calculate the actual batch number.

        Note:
            The parameters should be the same as the parameters used in the
            DataLoader, otherwise the calculated batch number may
            be inaccurate.

        Args:
            num_workers (int): The number of workers to use for loading
                the data.
            batch_size (int, optional): The batch size to use for loading
                the data. Defaults to 1.
            drop_last (bool, optional): Whether to drop the last incomplete
                batch. Defaults to False.

        """
        raise NotImplementedError


class DatasetWithIndices(TorchDataset, Generic[DatasetType]):
    """A dataset wrapper that allows indexing with an IndiceTable.

    Args:
        dataset (DatasetType): The underlying dataset to wrap.
        indices (IndiceTable | None): An optional IndiceTable to specify which
            indices of the dataset to use. If None, all indices will be used.

    """

    dataset: DatasetType
    indices: IndiceTable

    def __init__(
        self, dataset: DatasetType, indices: IndiceTable | None = None
    ):
        self.dataset = dataset
        if indices is None:
            if isinstance(dataset, Sized):
                indices = IndiceTable(len(dataset))
            else:
                raise ValueError(
                    "Dataset does not have a length, indices must be provided."
                )
        self.indices = indices

    def shard(
        self,
        num_shards: int,
        shard_id: int,
        contiguous: bool = True,
        shard_strategy: ShardStrategy | None = None,
    ):
        """Shard the dataset into multiple shards.

        Args:
            num_shards (int): The total number of shards to create.
            shard_id (int): The ID of the shard to return. Must be in the
                range [0, num_shards - 1].
            contiguous (bool, optional): Whether to create contiguous shards.
                If True, each shard will contain contiguous indices. If False,
                the indices will be distributed in a round-robin fashion.
                Defaults to True.
            shard_strategy (ShardStrategy | None, optional): The strategy to
                use for sharding the dataset. If None, the default strategy
                will be used, which is to drop the last incomplete shard if
                the total number of indices is not divisible by the number of
                shards. Defaults to None.
        """
        return DatasetWithIndices(
            dataset=self.dataset,
            indices=self.indices.shard(
                num_shards=num_shards,
                shard_id=shard_id,
                contiguous=contiguous,
                shard_strategy=shard_strategy,
            ),
        )

    def shuffle(
        self,
        generator: torch.Generator | np.random.Generator | None = None,
    ):
        """Shuffle the dataset indices.

        Args:
            generator (torch.Generator | np.random.Generator | None): An
                optional generator to use for shuffling. If None, a new
                generator will be created with a random seed.

        """
        return DatasetWithIndices(
            dataset=self.dataset,
            indices=self.indices.shuffle(generator),
        )

    def take(
        self, key: int | slice | range | Iterator[int]
    ) -> DatasetWithIndices:
        """Return a new DatasetWithIndices with the rows specified by key."""
        return DatasetWithIndices(
            dataset=self.dataset,
            indices=self.indices.take(key),
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({repr(self.dataset)}, "
            f"indices={repr(self.indices)})"
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int):
        actual_index = self.indices[index]
        return self.dataset[actual_index]

    def __getitems__(self, index: list[int]) -> list:
        if hasattr(self.dataset, "__getitems__"):
            actual_indices = [self.indices[i] for i in index]
            return self.dataset.__getitems__(actual_indices)  # type: ignore

        else:
            return [self.dataset[self.indices[i]] for i in index]

    def to_iterable_dataset(
        self,
        shuffle: bool = False,
        shard_kwargs: ShardConfig | None = None,
        generator: torch.Generator | np.random.Generator | None = None,
        batch_loader_kwargs: BatchLoaderConfig | dict | None = None,
    ) -> IterableWithLenDataset[DatasetType]:
        return IterableWithLenDataset(
            dataset=self.dataset,
            indices=self.indices,
            shuffle=shuffle,
            shard_kwargs=shard_kwargs,
            generator=generator,
            batch_loader_kwargs=batch_loader_kwargs,
        )


class RODatasetWithIndices(DatasetWithIndices[RODataset]):
    @property
    def features(self) -> Features:
        return self.dataset.features

    def _get_info_dict(self):
        data_info = self.dataset._get_info_dict()
        data_info.num_rows = len(self)
        return data_info


class ShardConfig(Config):
    contiguous: bool = True
    shard_strategy: ShardStrategy = None


class BatchLoaderConfig(Config):
    batch_size: int = 1
    collate_fn: Callable | None = None
    drop_last: bool = False
    prefetch_factor: int | None = None


class IterableWithLenDataset(
    TorchIterableDataset, IterableDatasetMixin, Generic[DatasetType]
):
    """A Iterable dataset wrapper that allows indexing with an IndiceTable.

    This class is designed to be compatible with PyTorch's DataLoader with
    multiple workers. When used with multiple workers, each worker will only
    iterate over its own shard of the data.

    Note:
        The purpose of this class is to provide a way to wrap an indexable
        dataset as an iterable dataset. This is useful when we partition
        a very large dataset into multiple subsets/chunks that can be indexed,
        but we want to load them in an iterable way to save resources.
        The input dataset should be indexable with an IndiceTable, and the
        indices should be compatible with the sharding strategy used in
        the DataLoader.

    Args:
        dataset (DatasetType): The underlying dataset to wrap.
        indices (IndiceTable | None): An optional IndiceTable to specify which
            indices of the dataset to use. If None, all indices will be used.
        shuffle (bool, optional): Whether to shuffle the indices. Defaults to
            False.
        shard_kwargs (ShardConfig | None, optional): Configuration for
            sharding the dataset. Sharding will be applied when using multiple
            processors in `accelerate`. Defaults to None, which means the
            default sharding strategy will be used (contiguous shards).
        generator (torch.Generator | np.random.Generator | None, optional): An
            optional generator to use for shuffling. If None, a new generator
            will be created with a random seed. Defaults to None.
        batch_loader_kwargs (BatchLoaderConfig | dict | None, optional): An
            optional configuration for using a batch loader. If provided, the
            dataset will be wrapped with a DataLoader to return batches
            of data. Defaults to None, which means no batch loader will
            be used.


    """

    dataset: DatasetType
    indice_sampler: IndiceTableSampler
    shard_kwargs: ShardConfig
    batch_loader_kwargs: BatchLoaderConfig | None

    def __init__(
        self,
        dataset: DatasetType,
        indices: IndiceTable | None = None,
        shuffle: bool = False,
        shard_kwargs: ShardConfig | None = None,
        generator: torch.Generator | np.random.Generator | None = None,
        batch_loader_kwargs: BatchLoaderConfig | dict | None = None,
    ):
        self.dataset = dataset
        if indices is None:
            if isinstance(dataset, Sized):
                indices = IndiceTable(len(dataset))
            else:
                raise ValueError(
                    "Dataset does not have a length, indices must be provided."
                )
        self.indice_sampler = IndiceTableSampler(
            indices=indices, shuffle=shuffle, generator=generator
        )

        # add to base classes but not inherit to avoid unnecessary methods.
        # prefer modifying class bases, but allow instance-level fallback
        _add_hf_iterable_cls(self.__class__, instance=self)

        self.shard_kwargs = (
            shard_kwargs if shard_kwargs is not None else ShardConfig()
        )
        if isinstance(batch_loader_kwargs, dict):
            batch_loader_kwargs = BatchLoaderConfig(**batch_loader_kwargs)
        self.batch_loader_kwargs = batch_loader_kwargs

    def shuffle_indices(self):
        """Shuffle the dataset indices."""
        self.indice_sampler.shuffle_indices()

    def shard(self, num_shards: int, shard_id: int):
        """Shard the dataset into multiple shards.

        Args:
            num_shards (int): The total number of shards to create.
            shard_id (int): The ID of the shard to return. Must be in the
                range [0, num_shards - 1].
        """
        shard_sampler = self.indice_sampler.shard(
            num_shards=num_shards,
            shard_id=shard_id,
            contiguous=self.shard_kwargs.contiguous,
        )
        return IterableWithLenDataset(
            dataset=self.dataset,
            indices=shard_sampler.table,
            shard_kwargs=self.shard_kwargs,
            shuffle=shard_sampler.shuffle,
            generator=shard_sampler.generator,
            batch_loader_kwargs=self.batch_loader_kwargs,
        )

    def take(
        self, key: int | slice | range | Iterator[int]
    ) -> IterableWithLenDataset[DatasetType]:
        """Return a new IterableWithLenDataset with the rows specified by key."""  # noqa: E501
        return IterableWithLenDataset(
            dataset=self.dataset,
            indices=self.indice_sampler.table.take(key),
            shard_kwargs=self.shard_kwargs,
            shuffle=self.indice_sampler.shuffle,
            generator=self.indice_sampler.generator,
            batch_loader_kwargs=self.batch_loader_kwargs,
        )

    def iter(self):
        """Iterate over the dataset and yield data samples.

        This method does not handle sharding for multiple workers.
        The sharding will be handled in the `__iter__` method, which will call
        this method to get the data samples for the current shard.

        """
        if self.batch_loader_kwargs is None:
            for item in self.indice_sampler:
                yield self.dataset[item]
        else:
            # create a DataLoader with 0 worker to load batches of data,
            # and the sharding will be handled by the DataLoader's worker
            # initialization function.
            batch_loader = torch.utils.data.DataLoader(
                dataset=IterableWithLenDataset(
                    dataset=self.dataset,
                    indices=self.indice_sampler.table,
                    shard_kwargs=self.shard_kwargs,
                    shuffle=self.indice_sampler.shuffle,
                    generator=self.indice_sampler.generator,
                    batch_loader_kwargs=None,
                ),
                num_workers=0,
                **self.batch_loader_kwargs.to_dict(),
            )
            for batch in batch_loader:
                yield batch

    def __iter__(self):
        if self.batch_loader_kwargs is None and self._is_torch_multi_worker():
            worker_info = torch.utils.data.get_worker_info()
            assert worker_info is not None
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            # do not call shard() here to avoid recursive sharding.
            sharded_indices = self.indice_sampler.shard(
                num_shards=num_workers,
                shard_id=worker_id,
                contiguous=self.shard_kwargs.contiguous,
            )
            for idx in sharded_indices:
                yield self.dataset[idx]

        else:
            yield from self.iter()

    @property
    def total_iterator_length(self) -> int:
        """Get the total number of data samples in the iterator."""
        return len(self.indice_sampler)

    @property
    def total_dataset_length(self) -> int:
        """Get the total length of the underlying dataset."""
        if isinstance(self.dataset, Sized):
            return len(self.dataset)
        else:
            raise ValueError(
                "Underlying dataset does not have a length, cannot get "
                "total dataset length."
            )

    def get_total_batch_num(
        self, num_workers: int, batch_size: int = 1, drop_last: bool = False
    ) -> int:
        """Calculate the total number of batches for the dataset.

        Pytorch `DataLoader` with multiple workers will shard the dataset into
        `num_workers` shards, and the default method to calculate the total
        number of batches does not consider the sharding, which will cause
        inaccurate total batch number when using multiple workers. This method
        provides a way to calculate the actual batch number.

        Note:
            The parameters should be the same as the parameters used in the
            DataLoader, otherwise the calculated batch number may
            be inaccurate.

        Args:
            num_workers (int): The number of workers to use for loading
                the data.
            batch_size (int, optional): The batch size to use for loading
                the data. Defaults to 1.
            drop_last (bool, optional): Whether to drop the last incomplete
                batch. Defaults to False.

        """
        return _get_total_batch_num(
            rows=self.total_iterator_length,
            num_workers=num_workers,
            batch_size=batch_size,
            drop_last=drop_last,
        )

    def _is_torch_multi_worker(self) -> bool:
        import torch.utils.data

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            return True
        return False

    @property
    def n_shards(self) -> int:
        """Get the number of shards for the current dataset.

        Currently this property returns the total number of data samples in the
        iterator.

        Note:
            In most cases, we do not need to know the number of shards, but
            this is reserved to be compatible with `prepare` method in
            accelerate, which needs to know the number of shards to prepare
            the dataset.
        """
        return self.total_iterator_length

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({repr(self.dataset)}, "
            f"indices={repr(self.indice_sampler)}, "
        )


class DatasetItem(Config, Generic[DatasetType], metaclass=ABCMeta):
    """A configuration for creating a dataset.

    User should inherit this class and implement the `_create_dataset` method
    to create a dataset from the configuration, and implement the
    `get_dataset_row_num` method to return the number of rows in the dataset
    before sharding.

    This class also include the sharding information, and the `create_dataset`
    method will apply the sharding to the created dataset. This is useful when
    we want to create a sharded dataset directly from the configuration.
    """

    class_type: ClassType[DatasetType]

    shard_id: int = Field(
        default=0, description="The ID of the shard to return.", ge=0
    )
    num_shards: int = Field(
        default=1, description="The total number of shards to create.", ge=1
    )

    def __post_init__(self):
        if self.shard_id >= self.num_shards:
            raise ValueError(
                f"shard_id must be in the range [0, num_shards - 1], but got "
                f"shard_id={self.shard_id} and num_shards={self.num_shards}."
            )

    @abstractmethod
    def get_dataset_row_num(self) -> int:
        """Get the number of rows in the dataset.

        This method should provide a lightweight way to get the
        number of rows in the dataset. This is important for efficiently
        calculating the total number of batches when using
        multiple workers in a DataLoader.
        """
        raise NotImplementedError(
            "get_dataset_row_num must be implemented by subclasses "
            "of DatasetItem."
        )

    def get_sharded_row_num(self, shard_config: ShardConfig) -> int:
        """Get the number of rows in the sharded dataset.

        This method calculates the number of rows in the dataset after sharding
        based on the sharding configuration.
        """
        total_rows = self.get_dataset_row_num()
        if self.num_shards <= 1:
            return total_rows

        if shard_config.shard_strategy is None:
            # Default sharding strategy: drop the last incomplete shard
            rows_per_shard = total_rows // self.num_shards
            residual = total_rows % self.num_shards
            return rows_per_shard + (1 if self.shard_id < residual else 0)
        elif shard_config.shard_strategy == "drop_last":
            rows_per_shard = total_rows // self.num_shards
            return rows_per_shard
        elif shard_config.shard_strategy == "pad_last":
            rows_per_shard = (
                total_rows + self.num_shards - 1
            ) // self.num_shards
            return rows_per_shard
        else:
            raise ValueError(
                f"Invalid shard strategy: {shard_config.shard_strategy}"
            )

    @abstractmethod
    def _create_dataset(self) -> DatasetType:
        """Create a dataset from the dataset item configuration."""
        raise NotImplementedError(
            "_create_dataset must be implemented by subclasses of DatasetItem."
        )

    def create_dataset(
        self, shard_config: ShardConfig
    ) -> DatasetWithIndices[DatasetType]:
        """Create a DatasetWithIndices from the dataset item configuration.

        This method applies the sharding configuration to the dataset by
        creating a DatasetWithIndices with the appropriate shard of indices.

        """
        ret = DatasetWithIndices(dataset=self._create_dataset())
        if self.is_sharded:
            return ret.shard(
                num_shards=self.num_shards,
                shard_id=self.shard_id,
                **shard_config.to_dict(),
            )
        return ret

    @property
    def is_sharded(self) -> bool:
        return self.num_shards > 1

    def shard(
        self, shard_id: int, num_shards: int
    ) -> DatasetItem[DatasetType]:
        """Shard the dataset item by returning a new DatasetItem.

        The new DatasetItem will have the same configuration as the original
        one, but with the updated shard_id and num_shards. The new sharding
        information will be calculated by:
        - new_num_shards: self.num_shards * num_shards
        - new_shard_id: self.shard_id * num_shards + shard_id

        Note that the sharding information is always calculated based on the
        original dataset.

        """
        if shard_id >= num_shards:
            raise ValueError(
                f"shard_id must be in the range [0, num_shards - 1], but got "
                f"shard_id={shard_id} and num_shards={num_shards}."
            )
        if shard_id < 0:
            raise ValueError(
                f"shard_id must be non-negative, but got shard_id={shard_id}."
            )
        if num_shards < 1:
            raise ValueError(
                "num_shards must be at least 1, "
                f"but got num_shards={num_shards}."
            )

        return self.replace(
            num_shards=self.num_shards * num_shards,
            shard_id=self.shard_id * num_shards + shard_id,
        )


class RODatasetItem(DatasetItem[RODataset]):
    """A DatasetItem for RODataset."""

    class_type: ClassType[RODataset] = RODataset
    dataset_path: str
    storage_options: dict | None = None
    meta_index2meta: bool = False

    transform: Callable | None = None

    def get_dataset_row_num(self) -> int:
        """Get the number of rows in the dataset."""
        rows = get_row_num_from_dataset_info(
            dataset_path=self.dataset_path,
        )
        if rows is not None:
            return rows

        dataset = RODataset(
            dataset_path=self.dataset_path,
            storage_options=self.storage_options,
            meta_index2meta=self.meta_index2meta,
        )
        return len(dataset)

    def _create_dataset(self) -> RODataset:
        """Create a dataset from the dataset item configuration."""
        dataset = RODataset(
            dataset_path=self.dataset_path,
            storage_options=self.storage_options,
            meta_index2meta=self.meta_index2meta,
        )
        dataset.set_transform(self.transform)
        return dataset


class DictIterableDataset(TorchIterableDataset, IterableDatasetMixin):
    dataset_items: list[DatasetItem]

    def __init__(
        self,
        datasets: Iterable[DatasetItem],
        shuffle: bool = False,
        shard_kwargs: ShardConfig | None = None,
        generator: torch.Generator | np.random.Generator | None = None,
        batch_loader_kwargs: BatchLoaderConfig | dict | None = None,
        max_dataset_concurrency: int = 4,
    ):
        # try to make this instance compatible with HF Iterable at class-level
        # or instance-level if class-level MRO change fails
        _add_hf_iterable_cls(self.__class__, instance=self)
        self.dataset_items = list(datasets)

        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)

        self._shard_kwargs = (
            shard_kwargs if shard_kwargs is not None else ShardConfig()
        )
        self._generator = generator
        self._shuffle = shuffle
        if isinstance(batch_loader_kwargs, dict):
            batch_loader_kwargs = BatchLoaderConfig(**batch_loader_kwargs)
        self._batch_loader_kwargs = batch_loader_kwargs
        self._max_dataset_concurrency = max_dataset_concurrency
        self._total_dataset_length: list[int] | None = None
        self._total_indices_length: list[int] | None = None

    def shard(self, shard_id: int, num_shards: int) -> DictIterableDataset:
        """Shard the dataset by sharding each dataset item."""
        sharded_items = [
            item.shard(shard_id=shard_id, num_shards=num_shards)
            for item in self.dataset_items
        ]
        return DictIterableDataset(
            datasets=sharded_items,
            shuffle=self._shuffle,
            generator=self._generator,
            batch_loader_kwargs=self._batch_loader_kwargs,
            max_dataset_concurrency=self._max_dataset_concurrency,
            shard_kwargs=self._shard_kwargs,
        )

    @property
    def total_dataset_length(self) -> int:
        if self._total_dataset_length is None:
            self._total_dataset_length = [
                item.get_dataset_row_num() for item in self.dataset_items
            ]
        return sum(self._total_dataset_length)

    @property
    def total_iterator_length(self) -> int:
        if self._total_indices_length is None:
            self._total_indices_length = [
                item.get_sharded_row_num(shard_config=self._shard_kwargs)
                for item in self.dataset_items
            ]
        return sum(self._total_indices_length)

    def get_total_batch_num(
        self, num_workers: int, batch_size: int, drop_last: bool
    ) -> int:
        total_batch_num = 0
        _ = self.total_iterator_length
        assert self._total_indices_length is not None

        if self._batch_loader_kwargs is not None:
            for indices_length in self._total_indices_length:
                total_batch_num += _get_total_batch_num(
                    rows=indices_length,
                    num_workers=num_workers,
                    batch_size=batch_size,
                    drop_last=drop_last,
                )
            return total_batch_num

        if num_workers <= 1:
            return _get_total_batch_num(
                rows=self.total_iterator_length,
                num_workers=1,
                batch_size=batch_size,
                drop_last=drop_last,
            )

        ret = 0
        for workder_id in range(num_workers):
            # get the total number of rows for the worker by
            # summing up the rows for each dataset item.
            total_worker_rows = 0
            for indices_length in self._total_indices_length:
                worker_rows = indices_length // num_workers
                if workder_id < indices_length % num_workers:
                    worker_rows += 1
                total_worker_rows += worker_rows
            ret += _get_total_batch_num(
                rows=total_worker_rows,
                num_workers=1,
                batch_size=batch_size,
                drop_last=drop_last,
            )
        return ret

    @property
    def n_shards(self) -> int:
        from accelerate.state import AcceleratorState

        state = AcceleratorState()
        return state.num_processes

    def _prepare_dataset_for_iter(
        self,
        cur_dataset_iters: list[tuple[int, Iterator]],
        remaining_dataset_indices: list[int],
    ) -> np.ndarray:
        """Prepare the dataset for iteration and return the sampling weights.

        Args:
            cur_dataset_iters (list[tuple[int, Iterator]]): The current
                dataset iterators.
            remaining_dataset_indices (list[int]): The remaining dataset
                indices to be processed.

        Returns:
            np.ndarray: The sampling weights for each dataset iterator.
        """
        while (
            len(cur_dataset_iters) < self._max_dataset_concurrency
            and len(remaining_dataset_indices) > 0
        ):
            idx = remaining_dataset_indices.pop(0)
            iter_dataset = (
                self.dataset_items[idx]
                .create_dataset(shard_config=self._shard_kwargs)
                .to_iterable_dataset(
                    shuffle=self._shuffle,
                    shard_kwargs=self._shard_kwargs,
                    generator=self._generator,
                    batch_loader_kwargs=self._batch_loader_kwargs,
                )
            )
            cur_dataset_iters.append((idx, iter(iter_dataset)))
        assert self._total_indices_length is not None
        weights = []
        for idx, _ in cur_dataset_iters:
            weights.append(self._total_indices_length[idx])
        weights = np.array(weights, dtype=np.float32)
        weights = weights / weights.sum()
        return weights

    def __iter__(self):
        cur_dataset_iters: list[tuple[int, Iterator]] = []
        dataset_indices = list(
            IndiceTableSampler(
                len(self.dataset_items),
                shuffle=self._shuffle,
                generator=self._generator,
            )
        )
        # Access total_iterator_length to trigger the calculation of total
        # iterator length
        _ = self.total_iterator_length
        assert self._total_indices_length is not None
        weights = self._prepare_dataset_for_iter(
            cur_dataset_iters=cur_dataset_iters,
            remaining_dataset_indices=dataset_indices,
        )

        while len(cur_dataset_iters) > 0:
            # calulate the sampling weight for each dataset iterator based
            # on the indices length of the corresponding dataset.
            if self._shuffle:
                if isinstance(self._generator, np.random.Generator):
                    selected_idx = self._generator.choice(
                        len(cur_dataset_iters), p=weights, replace=False
                    )
                elif isinstance(self._generator, torch.Generator):
                    selected_idx = int(
                        torch.multinomial(
                            torch.tensor(weights), 1, generator=self._generator
                        ).item()
                    )
                else:
                    raise ValueError(
                        "Generator must be either a torch.Generator or a "
                        "numpy.random.Generator."
                    )
            else:
                selected_idx = 0
            idx, iter_dataset = cur_dataset_iters[selected_idx]
            try:
                item = next(iter_dataset)
                yield item
            except StopIteration:
                cur_dataset_iters.pop(selected_idx)
                weights = self._prepare_dataset_for_iter(
                    cur_dataset_iters=cur_dataset_iters,
                    remaining_dataset_indices=dataset_indices,
                )


def _get_batch_num(batch_size: int, num_samples: int, drop_last: bool) -> int:
    if drop_last:
        return num_samples // batch_size
    else:
        return (num_samples + batch_size - 1) // batch_size


def _get_total_batch_num(
    rows: int, num_workers: int, batch_size: int = 1, drop_last: bool = False
) -> int:
    """Calculate the total number of batches for the dataset.

    Pytorch `DataLoader` with multiple workers will shard the dataset into
    `num_workers` shards, and the default method to calculate the total
    number of batches does not consider the sharding, which will cause
    inaccurate total batch number when using multiple workers. This method
    provides a way to calculate the actual batch number.

    Note:
        The parameters should be the same as the parameters used in the
        DataLoader, otherwise the calculated batch number may
        be inaccurate.

    Args:
        rows (int): The total number of rows in the dataset.
        num_workers (int): The number of workers to use for loading
            the data.
        batch_size (int, optional): The batch size to use for loading
            the data. Defaults to 1.
        drop_last (bool, optional): Whether to drop the last incomplete
            batch. Defaults to False.

    """
    if num_workers <= 1:
        return _get_batch_num(
            batch_size=batch_size,
            num_samples=rows,
            drop_last=drop_last,
        )
    total_batches = 0
    for worker_id in range(num_workers):
        worker_num_samples = rows // num_workers
        if worker_id < rows % num_workers:
            worker_num_samples += 1

        total_batches += _get_batch_num(
            batch_size=batch_size,
            num_samples=worker_num_samples,
            drop_last=drop_last,
        )
    return total_batches
