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
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import socket
import tempfile
import unicodedata
import warnings
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    overload,
)

import numpy as np
import torch
from datasets import IterableDataset as HFIterableDataset
from filelock import FileLock
from pydantic import Field
from robo_orchard_core.utils.config import ClassType, Config
from robo_orchard_core.utils.logging import LoggerManager
from torch.utils.data import (
    DataLoader as TorchDataLoader,
    Dataset as TorchDataset,
    IterableDataset as TorchIterableDataset,
)
from typing_extensions import Self, TypeVar

from robo_orchard_lab.dataset.robot._prefetch import (
    close_iterators_best_effort,
    create_prefetch_iterator,
)
from robo_orchard_lab.dataset.sampler import (
    ChunkedIndiceTable,
    IndiceTable,
    IndiceTableSampler,
    ShardStrategy,
    Sized,
)

logger = LoggerManager().get_child(__name__)

__all__ = [
    "ShardConfig",
    "BatchLoaderConfig",
    "DataLoader",
    "ShuffleConfig",
    "IterableDatasetMixin",
    "DatasetWithIndices",
    "IterableWithLenDataset",
    "DatasetItemNode",
    "DatasetItem",
    "DatasetItemGroup",
    "DictIterableDataset",
]


DatasetType = TypeVar("DatasetType", bound=TorchDataset)
_TORCH_DATALOADER_INIT_SIGNATURE = inspect.signature(TorchDataLoader.__init__)
_DEFAULT_VIRTUAL_GETITEMS_BATCH_SIZE = 32
_DATASET_ITEM_NAME_SEGMENT_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]*")


class ShardConfig(Config):
    contiguous: bool = True
    shard_strategy: ShardStrategy = None


class BatchLoaderConfig(Config):
    batch_size: int = 1
    collate_fn: Callable | None = None
    drop_last: bool = False


def _collate_self_batched_item(
    batch: list[Any], user_collate_fn: Callable | None = None
) -> Any:
    if len(batch) != 1:
        raise ValueError(
            "Self-batched datasets expect DataLoader to receive exactly "
            f"one item per batch, but got {len(batch)} items."
        )
    item = batch[0]
    if user_collate_fn is None:
        return item
    return user_collate_fn(item)


def _should_use_dataset_batch_loader(
    dataset: Any, use_dataset_side_batching: bool
) -> bool:
    if (
        isinstance(dataset, IterableDatasetMixin)
        and dataset.batch_loader_kwargs is not None
    ):
        return True

    return (
        use_dataset_side_batching
        and isinstance(dataset, (IterableWithLenDataset, DictIterableDataset))
        and dataset.batch_loader_kwargs is None
    )


def _normalize_shuffle_for_non_iterable_dataset_mixin(
    dataset: Any,
    dataloader_kwargs: dict[str, Any],
) -> dict[str, Any]:
    dataloader_shuffle = dataloader_kwargs.get("shuffle")
    if not isinstance(dataloader_shuffle, ShuffleConfig):
        return dataloader_kwargs

    if dataloader_shuffle.chunk_size is not None:
        warnings.warn(
            "`ShuffleConfig.chunk_size` is only supported for "
            "IterableDatasetMixin datasets. Falling back to the boolean "
            "`shuffle` value for this DataLoader.",
            UserWarning,
        )

    if isinstance(dataset, TorchIterableDataset):
        if dataloader_shuffle.shuffle:
            warnings.warn(
                "Non-IterableDatasetMixin iterable datasets do not support "
                "outer DataLoader shuffling. Resetting `shuffle=False`.",
                UserWarning,
            )
        dataloader_kwargs["shuffle"] = False
        return dataloader_kwargs

    dataloader_kwargs["shuffle"] = dataloader_shuffle.shuffle
    return dataloader_kwargs


def _wrap_with_prefetch_if_needed(
    iterator: Iterator[Any],
    shuffle_config: ShuffleConfig,
    generator: torch.Generator | np.random.Generator | None,
    batch_loader_kwargs: BatchLoaderConfig | None,
) -> Iterator[Any]:
    """Return an iterator wrapped with sample-level prefetching when needed.

    Prefetching is a dataset-side optimization for shuffled sample iteration.
    Dataset-side batching already owns its own iteration boundary, so this
    helper leaves batched iteration unwrapped.
    """
    prefetch_size = shuffle_config.prefetch_size
    if _uses_sample_prefetch(shuffle_config, batch_loader_kwargs):
        assert prefetch_size is not None
        logger.debug(
            "Applying prefetching with prefetch size: %d", prefetch_size
        )
        return create_prefetch_iterator(
            iterator,
            prefetch_size,
            shuffle=shuffle_config.shuffle,
            generator=generator,
        )

    logger.debug(
        "No prefetching applied, shuffle: %s",
        shuffle_config.shuffle,
    )
    return iterator


def _uses_sample_prefetch(
    shuffle_config: ShuffleConfig,
    batch_loader_kwargs: BatchLoaderConfig | None,
) -> bool:
    """Return whether one iteration owns a threaded sample prefetcher.

    A size-one prefetch configuration is the documented pass-through path in
    ``create_prefetch_iterator``. It has no producer thread or reservoir, so
    this predicate must leave the sampler's explicit generator untouched.
    """

    prefetch_size = shuffle_config.prefetch_size
    if prefetch_size is not None and prefetch_size <= 0:
        raise ValueError("prefetch_size must be greater than 0.")

    return (
        prefetch_size is not None
        and prefetch_size > 1
        and shuffle_config.shuffle
        and batch_loader_kwargs is None
    )


def _split_prefetch_generator(
    generator: torch.Generator | np.random.Generator | None,
) -> tuple[
    torch.Generator | np.random.Generator | None,
    torch.Generator | np.random.Generator | None,
]:
    """Derive source and reservoir RNG streams before the producer starts.

    Source resampling runs in the prefetch producer thread while reservoir
    shuffle runs in the consumer thread. Splitting an explicit generator in
    the calling thread prevents their draw order from depending on thread
    scheduling or backing-store latency. ``None`` retains the pre-existing
    unseeded behavior because no caller-owned stream is shared.
    """

    if generator is None:
        return None, None
    if isinstance(generator, torch.Generator):
        seed_tensor = torch.empty((), dtype=torch.int64)
        source_seed = int(seed_tensor.random_(generator=generator).item())
        reservoir_seed = int(seed_tensor.random_(generator=generator).item())
        source_generator = torch.Generator(device=generator.device)
        source_generator.manual_seed(source_seed)
        reservoir_generator = torch.Generator(device=generator.device)
        reservoir_generator.manual_seed(reservoir_seed)
        return source_generator, reservoir_generator
    if isinstance(generator, np.random.Generator):
        max_seed = np.iinfo(np.int64).max
        source_seed = int(generator.integers(max_seed, dtype=np.int64))
        reservoir_seed = int(generator.integers(max_seed, dtype=np.int64))
        return np.random.default_rng(source_seed), np.random.default_rng(
            reservoir_seed
        )
    raise TypeError(
        "Prefetch shuffle generator must be a torch.Generator, "
        "numpy.random.Generator, or None."
    )


class DataLoader(TorchDataLoader):
    """A thin wrapper around PyTorch ``DataLoader``.

    For iterable datasets this loader can operate with two batching layers:

    1. The ordinary outer ``TorchDataLoader`` batching layer.
    2. A dataset-side batching layer driven by ``batch_loader_kwargs``.

    For iterable datasets that already yield batches through
    ``batch_loader_kwargs``, this loader clones the input dataset, aligns the
    dataset-side batch settings with the caller-provided dataloader batch
    arguments, and then configures the outer ``TorchDataLoader`` to forward one
    already-formed batch at a time.

    In that self-batched mode the outer loader may expose ``batch_size == 1``
    because it is only transporting one ready-made batch per iteration. The
    effective sample batch size is tracked separately and is the value used by
    ``__len__`` and the iterable dataset batch-count helpers.

    When ``use_dataset_side_batching`` is True and the input dataset is a
    supported iterable dataset without ``batch_loader_kwargs``, this loader
    will internally enable aligned ``batch_loader_kwargs`` on a cloned dataset.

    Args:
        dataset: The dataset to load.
        use_dataset_side_batching: When True and ``dataset`` is a supported
            iterable dataset without ``batch_loader_kwargs``, enable
            dataset-side batch loading on a cloned dataset.
        *args: Positional arguments forwarded to ``TorchDataLoader``.
        **kwargs: Keyword arguments forwarded to ``TorchDataLoader``. Relevant
            batch-related arguments, and ``shuffle`` when supported by the
            dataset, are also aligned into dataset-side configuration when
            self-batched loading is enabled.
    """

    @overload
    def __init__(
        self,
        dataset: Any,
        batch_size: int | None = 1,
        shuffle: bool | ShuffleConfig | None = None,
        sampler: Any | None = None,
        batch_sampler: None = None,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Callable | None = None,
        multiprocessing_context: Any = None,
        generator: torch.Generator | None = None,
        *,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
        in_order: bool = True,
        use_dataset_side_batching: bool = False,
    ) -> None: ...

    @overload
    def __init__(
        self,
        dataset: Any,
        batch_size: None = None,
        shuffle: bool | ShuffleConfig | None = None,
        sampler: None = None,
        batch_sampler: Any = None,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Callable | None = None,
        multiprocessing_context: Any = None,
        generator: torch.Generator | None = None,
        *,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
        in_order: bool = True,
        use_dataset_side_batching: bool = False,
    ) -> None: ...

    def __init__(
        self,
        dataset,
        *args,
        use_dataset_side_batching: bool = False,
        **kwargs,
    ):
        dataloader_kwargs = self._bind_dataloader_kwargs(
            dataset=dataset,
            args=args,
            kwargs=kwargs,
        )
        aligned_batch_loader_kwargs = None
        if isinstance(dataset, IterableDatasetMixin):
            (
                dataset,
                self._uses_dataset_batch_loader,
                aligned_batch_loader_kwargs,
            ) = self._clone_iterable_dataset_for_dataloader(
                dataset=dataset,
                dataloader_kwargs=dataloader_kwargs,
                use_dataset_side_batching=use_dataset_side_batching,
            )
            dataloader_kwargs["dataset"] = dataset
        else:
            self._uses_dataset_batch_loader = False
            dataloader_kwargs = (
                _normalize_shuffle_for_non_iterable_dataset_mixin(
                    dataset=dataset,
                    dataloader_kwargs=dataloader_kwargs,
                )
            )

        batch_size = dataloader_kwargs.get("batch_size", 1)
        self._effective_batch_size = 1 if batch_size is None else batch_size
        self._effective_drop_last = dataloader_kwargs.get("drop_last", False)

        if aligned_batch_loader_kwargs is not None:
            self._effective_batch_size = aligned_batch_loader_kwargs.batch_size
            self._effective_drop_last = aligned_batch_loader_kwargs.drop_last
            dataloader_kwargs = (
                self._normalize_outer_dataloader_for_self_batched_dataset(
                    dataloader_kwargs
                )
            )

        super().__init__(**dataloader_kwargs)

    def __len__(self) -> int:
        """Return the batch count using the effective batching layer.

        For iterable datasets this may differ from the outer dataloader's
        visible ``batch_size`` because dataset-side batching normalizes the
        outer loader to forward one already-built batch at a time.
        """
        if isinstance(self.dataset, IterableDatasetMixin):
            return self.dataset.get_total_batch_num(
                num_workers=self.num_workers,
                batch_size=self._effective_batch_size,
                drop_last=self._effective_drop_last,
            )

        return super().__len__()

    @staticmethod
    def _bind_dataloader_kwargs(
        dataset: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        bound = _TORCH_DATALOADER_INIT_SIGNATURE.bind_partial(
            None, dataset, *args, **kwargs
        )
        dataloader_kwargs = dict(bound.arguments)
        dataloader_kwargs.pop("self", None)
        return dataloader_kwargs

    @staticmethod
    def _clone_iterable_dataset_for_dataloader(
        dataset: IterableDatasetMixin,
        dataloader_kwargs: dict[str, Any],
        use_dataset_side_batching: bool,
    ) -> tuple[IterableDatasetMixin, bool, BatchLoaderConfig | None]:
        """Clone iterable datasets when loader-local state must diverge.

        The clone keeps caller-owned dataset objects immutable while this
        dataloader rewrites shuffle or dataset-side batching configuration for
        its own execution.
        """
        uses_dataset_batch_loader = _should_use_dataset_batch_loader(
            dataset=dataset,
            use_dataset_side_batching=use_dataset_side_batching,
        )
        should_clone_for_shuffle = (
            not uses_dataset_batch_loader and "shuffle" in dataloader_kwargs
        )
        should_clone_for_offload = (
            isinstance(dataset, DictIterableDataset)
            and dataset.offload_num_batches is not None
        )
        if not (
            uses_dataset_batch_loader
            or should_clone_for_shuffle
            or should_clone_for_offload
        ):
            return dataset, False, None

        aligned_batch_loader_kwargs = (
            DataLoader._align_batch_loader_kwargs(
                dataset=dataset,
                dataloader_kwargs=dataloader_kwargs,
            )
            if uses_dataset_batch_loader
            else None
        )

        aligned_shuffle_config = DataLoader._align_dataset_shuffle_config(
            dataset=dataset,
            dataloader_shuffle=dataloader_kwargs.get("shuffle"),
        )
        logger.debug("new shuffle cfg: %s", aligned_shuffle_config)

        if isinstance(dataset, IterableWithLenDataset):
            cloned_dataset: IterableDatasetMixin = IterableWithLenDataset(
                dataset=dataset.dataset,
                indices=dataset.indice_sampler.table,
                shuffle=aligned_shuffle_config,
                shard_kwargs=dataset.shard_kwargs,
                generator=dataset.indice_sampler.generator,
                batch_loader_kwargs=aligned_batch_loader_kwargs,
                resample_ratio=dataset.resample_ratio,
            )
        elif isinstance(dataset, DictIterableDataset):
            cloned_dataset = DictIterableDataset(
                datasets=dataset.logical_dataset_items,
                shuffle=aligned_shuffle_config,
                shard_kwargs=dataset.shard_kwargs,
                generator=dataset._generator,
                batch_loader_kwargs=aligned_batch_loader_kwargs,
                max_dataset_concurrency=dataset._max_dataset_concurrency,
                resample_ratios=dataset._logical_resample_ratios,
                offload_num_batches=dataset.offload_num_batches,
                offload_dir=dataset.offload_dir,
            )
            cloned_dataset._offload_batch_size = (
                aligned_batch_loader_kwargs.batch_size
                if aligned_batch_loader_kwargs is not None
                else dataloader_kwargs.get("batch_size", 1) or 1
            )
        else:
            raise TypeError(
                "Iterable dataset cloning only supports "
                "IterableWithLenDataset and DictIterableDataset."
            )

        if should_clone_for_shuffle:
            dataloader_kwargs["shuffle"] = False

        return (
            cloned_dataset,
            uses_dataset_batch_loader,
            aligned_batch_loader_kwargs,
        )

    @staticmethod
    def _align_batch_loader_kwargs(
        dataset: IterableDatasetMixin,
        dataloader_kwargs: dict[str, Any],
    ) -> BatchLoaderConfig:
        """Merge dataset batch defaults with explicit dataloader arguments.

        ``batch_size``, ``collate_fn`` and ``drop_last`` from the caller win
        over the dataset defaults so the cloned dataset behaves as if those
        arguments had been supplied at dataset construction time.
        """
        dataset_batch_loader_kwargs = dataset.batch_loader_kwargs
        aligned_batch_loader_kwargs = (
            BatchLoaderConfig(**dataset_batch_loader_kwargs.to_dict())
            if dataset_batch_loader_kwargs is not None
            else BatchLoaderConfig()
        )
        for key in BatchLoaderConfig.model_fields:
            if key in dataloader_kwargs:
                setattr(
                    aligned_batch_loader_kwargs,
                    key,
                    dataloader_kwargs[key],
                )
        return aligned_batch_loader_kwargs

    @staticmethod
    def _align_dataset_shuffle_config(
        dataset: IterableDatasetMixin,
        dataloader_shuffle: bool | ShuffleConfig | None,
    ) -> ShuffleConfig:
        """Translate dataloader shuffle requests into dataset shuffle state.

        A boolean request only replaces the ``shuffle`` flag. A full
        ``ShuffleConfig`` replaces the whole configuration so the caller can
        override chunking and prefetch-related settings as well.
        """
        if isinstance(dataset, IterableWithLenDataset):
            dataset_shuffle = dataset._shuffle_config
        elif isinstance(dataset, DictIterableDataset):
            dataset_shuffle = dataset._shuffle
        else:
            raise TypeError(
                "Dataset shuffle alignment only supports "
                "IterableWithLenDataset and DictIterableDataset."
            )

        aligned_shuffle_config = ShuffleConfig(**dataset_shuffle.to_dict())
        if dataloader_shuffle is None:
            return aligned_shuffle_config
        if isinstance(dataloader_shuffle, ShuffleConfig):
            return ShuffleConfig(**dataloader_shuffle.to_dict())

        aligned_shuffle_config.shuffle = dataloader_shuffle
        return aligned_shuffle_config

    @staticmethod
    def _normalize_outer_dataloader_for_self_batched_dataset(
        dataloader_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize outer DataLoader kwargs for self-batched datasets.

        In this mode the dataset itself already yields complete batches. The
        outer ``TorchDataLoader`` should therefore only transport one dataset
        item at a time and unwrap it, instead of trying to batch samples again.

        Any user ``collate_fn`` has already been aligned into the dataset-side
        ``batch_loader_kwargs``. The outer loader only needs to unwrap the
        single item it receives from the dataset.
        """
        dataloader_kwargs["batch_size"] = 1
        dataloader_kwargs["collate_fn"] = partial(_collate_self_batched_item)
        # ``drop_last`` has already been applied by the inner dataset batch
        # generation logic via ``batch_loader_kwargs``. The outer dataloader is
        # only used to forward one already-formed batch at a time, so keeping
        # ``drop_last=True`` here would risk dropping an entire final batch at
        # the wrong layer.
        dataloader_kwargs["drop_last"] = False
        dataloader_kwargs["shuffle"] = False
        return dataloader_kwargs


class ShuffleConfig(Config):
    """Configuration for shuffling the dataset indices.

    Args:
        shuffle (bool): Whether to shuffle the dataset indices.
        chunk_size (int | None): The chunk size for the indices. If provided,
            the indices will be split into chunks of the given size, and each
            chunk will be treated as a unit for sharding. This can help reduce
            the overhead of sharding when the dataset is very large. If None,
            then no chunking will be done and the indices will be treated as
            individual samples. Defaults to None.
        prefetch_factor (int): The factor to determine the prefetch size for
            prefetching the dataset. The prefetch size will be calculated as
            `chunk_size * prefetch_factor` if `chunk_size` is provided, otherwise
            the prefetch size will be `None` and no prefetching will be applied.
            This argument is usually only valid when `chunk_size` is provided and
            `shuffle` is True. Defaults to 4.

    """  # noqa: E501

    shuffle: bool = False
    chunk_size: int | None = None
    prefetch_factor: int = 4

    @property
    def prefetch_size(self) -> int | None:
        if self.chunk_size is not None:
            return self.chunk_size * self.prefetch_factor
        return None


class IterableDatasetMixin(metaclass=ABCMeta):
    @property
    @abstractmethod
    def batch_loader_kwargs(self) -> BatchLoaderConfig | None:
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def shard_kwargs(self) -> ShardConfig:
        raise NotImplementedError

    @abstractmethod
    def shard(self, num_shards: int, index: int):
        """Shard the dataset into multiple shards.

        Args:
            num_shards (int): The total number of shards to create.
            index (int): The ID of the shard to return. Must be in the
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
        index: int,
        contiguous: bool = True,
        shard_strategy: ShardStrategy | None = None,
    ):
        """Shard the dataset into multiple shards.

        Args:
            num_shards (int): The total number of shards to create.
            index (int): The ID of the shard to return. Must be in the
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
                shard_id=index,
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

    def to_iterable_dataset(
        self,
        shuffle: bool | ShuffleConfig = False,
        shard_kwargs: ShardConfig | None = None,
        generator: torch.Generator | np.random.Generator | None = None,
        batch_loader_kwargs: BatchLoaderConfig | dict | None = None,
        resample_ratio: float = 1.0,
    ) -> IterableWithLenDataset[DatasetType]:
        """Create a length-aware iterable view over the selected indices.

        The returned view owns shuffling, PyTorch worker sharding, optional
        row-level resampling, and optional dataset-side batching. A non-unit
        resampling ratio is only valid when shuffling is enabled.

        Args:
            shuffle (bool | ShuffleConfig, optional): Shuffle policy for each
                natural pass over the selected indices. Defaults to False.
            shard_kwargs (ShardConfig | None, optional): Worker/process shard
                policy. Defaults to None, which uses :class:`ShardConfig`
                defaults.
            generator (torch.Generator | np.random.Generator | None, optional):
                Random generator used by shuffling. Defaults to None.
            batch_loader_kwargs (BatchLoaderConfig | dict | None, optional):
                Dataset-side batch construction. Defaults to None, which
                yields individual samples.
            resample_ratio (float, optional): Finite positive multiplier for
                the selected row count. Values other than 1.0 require
                ``shuffle=True``. Defaults to 1.0.

        Returns:
            IterableWithLenDataset[DatasetType]: An iterable dataset view that
                preserves the selected index table.

        Raises:
            TypeError: If ``resample_ratio`` is not numeric.
            ValueError: If the ratio is invalid or non-unit resampling is
                requested without shuffling.
        """
        return IterableWithLenDataset(
            dataset=self.dataset,
            indices=self.indices,
            shuffle=shuffle,
            shard_kwargs=shard_kwargs,
            generator=generator,
            batch_loader_kwargs=batch_loader_kwargs,
            resample_ratio=resample_ratio,
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


class IterableWithLenDataset(
    TorchIterableDataset, IterableDatasetMixin, Generic[DatasetType]
):
    """Expose an indexable dataset as a length-aware iterable dataset.

    Use this wrapper when the backing dataset remains indexable but callers
    need iterable loading, exact logical lengths, PyTorch worker sharding, or
    dataset-side batching. With multiple workers, the physical index table is
    sharded first and each worker resamples only its own shard; worker targets
    are allocated so their sum remains the exact global logical row count.

    Note:
        The purpose of this class is to provide a way to wrap an indexable
        dataset as an iterable dataset. This is useful when we partition
        a very large dataset into multiple subsets/chunks that can be indexed,
        but we want to load them in an iterable way to save resources.
        The input dataset should be indexable with an IndiceTable, and the
        indices should be compatible with the sharding strategy used in
        the DataLoader.

        At runtime this wrapper has two distinct iteration modes:

        1. If ``batch_loader_kwargs`` is None, it yields individual samples by
           resolving indices from ``indice_sampler``. Outer PyTorch worker
           sharding is applied directly to the sampler.
        2. If ``batch_loader_kwargs`` is set, it builds an inner single-process
           dataloader over the worker-local view. The outer loader only forwards
           those ready-made batches.

        ``__iter__`` wraps either mode with optional prefetch buffering when
        sample-level iteration is active.

        Use ``dataset.iter(batch_size=32)`` to override the batch size and
        tail policy for one iterator while retaining the configured collation.

    Args:
        dataset (DatasetType): The underlying dataset to wrap.
        indices (IndiceTable | None): An optional IndiceTable to specify which
            indices of the dataset to use. If None, all indices will be used.
        shuffle (bool | ShuffleConfig, optional): Whether to shuffle the dataset
            indices. If a ShuffleConfig is provided, it will be used to configure
            the shuffling behavior. Defaults to False, which means no shuffling
            will be applied.
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
        resample_ratio (float, optional): Finite positive multiplier for the
            logical row count. Ratios below one take a shuffled prefix; ratios
            above one repeat natural shuffled cycles plus a final prefix.
            Values other than 1.0 require ``shuffle=True``. Defaults to 1.0.

    Raises:
        TypeError: If ``resample_ratio`` is not numeric.
        ValueError: If indices cannot be inferred, the ratio is not finite and
            positive, or non-unit resampling is requested without shuffling.

    """  # noqa: E501

    dataset: DatasetType
    indice_sampler: IndiceTableSampler
    _batch_loader_kwargs: BatchLoaderConfig | None

    def __init__(
        self,
        dataset: DatasetType,
        indices: IndiceTable | ChunkedIndiceTable | None = None,
        shuffle: bool | ShuffleConfig = False,
        shard_kwargs: ShardConfig | None = None,
        generator: torch.Generator | np.random.Generator | None = None,
        batch_loader_kwargs: BatchLoaderConfig | dict | None = None,
        resample_ratio: float = 1.0,
    ):
        logger.debug(
            "Initializing IterableWithLenDataset with shuffle config: %s, "
            "shard config: %s and batch loader kwargs: %s",
            shuffle,
            shard_kwargs,
            batch_loader_kwargs,
        )
        self.dataset = dataset
        if indices is None:
            if not isinstance(dataset, Sized):
                raise ValueError(
                    "Dataset does not have a length, indices must be provided."
                )
            indices = IndiceTable(len(dataset))
        self._shuffle_config = (
            ShuffleConfig(shuffle=shuffle)
            if isinstance(shuffle, bool)
            else shuffle
        )
        self._resample_ratio = _normalize_resample_ratio(
            resample_ratio,
            name="resample_ratio",
        )
        if self.resample_ratio != 1.0 and not self._shuffle_config.shuffle:
            raise ValueError(
                "resample_ratio values other than 1.0 require shuffle=True."
            )

        self.indice_sampler = IndiceTableSampler(
            indices=indices,
            shuffle=self._shuffle_config.shuffle,
            generator=generator,
            shuffle_chunk_size=(
                self._shuffle_config.chunk_size
                if not isinstance(indices, ChunkedIndiceTable)
                else None
            ),
        )

        # add to base classes but not inherit to avoid unnecessary methods.
        # prefer modifying class bases, but allow instance-level fallback
        # _add_hf_iterable_cls(self.__class__, instance=self)

        self._shard_kwargs = (
            shard_kwargs if shard_kwargs is not None else ShardConfig()
        )
        self._batch_loader_kwargs = (
            BatchLoaderConfig(**batch_loader_kwargs)
            if isinstance(batch_loader_kwargs, dict)
            else batch_loader_kwargs
        )
        self._prepared_indices = False

    @property
    def batch_loader_kwargs(self) -> BatchLoaderConfig | None:
        return self._batch_loader_kwargs

    @property
    def shard_kwargs(self) -> ShardConfig:
        return self._shard_kwargs

    @property
    def resample_ratio(self) -> float:
        """Return the logical row-count multiplier for this dataset view."""
        return self._resample_ratio

    def shuffle_indices(self):
        """Shuffle the dataset indices."""
        self.indice_sampler.shuffle_indices()

    def shard(self, num_shards: int, index: int):
        """Shard the dataset into multiple shards.

        Args:
            num_shards (int): The total number of shards to create.
            index (int): The ID of the shard to return. Must be in the
                range [0, num_shards - 1].

        Returns:
            IterableWithLenDataset[DatasetType]: A new dataset view with the
                same shuffle and batching configuration, but restricted to the
                selected shard of indices.
        """
        shard_sampler = self.indice_sampler.shard(
            num_shards=num_shards,
            shard_id=index,
            contiguous=self.shard_kwargs.contiguous,
        )
        return IterableWithLenDataset(
            dataset=self.dataset,
            indices=shard_sampler.table,
            shard_kwargs=self.shard_kwargs,
            shuffle=self._shuffle_config,
            generator=shard_sampler.generator,
            batch_loader_kwargs=self.batch_loader_kwargs,
            resample_ratio=self.resample_ratio,
        )

    def take(
        self, key: int | slice | range | Iterator[int]
    ) -> IterableWithLenDataset[DatasetType]:
        """Return a new IterableWithLenDataset with the rows specified by key."""  # noqa: E501
        return IterableWithLenDataset(
            dataset=self.dataset,
            indices=self.indice_sampler.table.take(key),
            shard_kwargs=self.shard_kwargs,
            shuffle=self._shuffle_config,
            generator=self.indice_sampler.generator,
            batch_loader_kwargs=self.batch_loader_kwargs,
            resample_ratio=self.resample_ratio,
        )

    def iter(
        self, batch_size: int, drop_last_batch: bool = False
    ) -> Iterator[Any]:
        """Yield batches using this dataset's sampling and collation rules.

        Batch size and tail policy apply only to this iterator. An existing
        ``collate_fn`` is retained; otherwise PyTorch's default collation is
        used. Raw samples are batched exactly once, including when the dataset
        already has dataset-side batching configured. Worker sharding,
        resampling, prefetching and cleanup follow normal dataset-side
        batching.

        Args:
            batch_size (int): Positive number of samples per batch.
            drop_last_batch (bool, optional): Whether to discard the final
                incomplete batch. Defaults to False.

        Yields:
            Any: Batches in the existing collation's output format.

        Raises:
            ValueError: If batch size is not a positive integer or
                ``drop_last_batch`` is not a bool.
        """
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer.")
        if not isinstance(drop_last_batch, bool):
            raise ValueError("drop_last_batch must be a bool.")
        batch_config = (
            self.batch_loader_kwargs
            if self.batch_loader_kwargs is not None
            else BatchLoaderConfig()
        ).replace(batch_size=batch_size, drop_last=drop_last_batch)
        yield from self._iter_batches(
            self.indice_sampler,
            batch_config,
        )

    def __iter__(self):
        """Yield the worker-local logical stream with optional prefetching.

        PyTorch worker sharding is applied before resampling. Prefetch
        buffering is only added while iteration remains sample-level;
        dataset-side batching stays the single source of batch construction.
        When explicit seeded shuffling and prefetching are both active, source
        resampling and reservoir shuffle use deterministic independent streams.

        Yields:
            Any: Individual samples or ready-made batches, depending on
            ``batch_loader_kwargs``.
        """
        indice_sampler = self.indice_sampler
        prefetch_generator = indice_sampler.generator
        if not self._prepared_indices and _uses_sample_prefetch(
            self._shuffle_config,
            self.batch_loader_kwargs,
        ):
            source_generator, prefetch_generator = _split_prefetch_generator(
                indice_sampler.generator
            )
            indice_sampler = IndiceTableSampler(
                indices=indice_sampler.table,
                shuffle=indice_sampler.shuffle,
                generator=source_generator,
            )
        iterator = _wrap_with_prefetch_if_needed(
            self._iter_impl(indice_sampler),
            shuffle_config=self._shuffle_config,
            generator=prefetch_generator,
            batch_loader_kwargs=self.batch_loader_kwargs,
        )
        primary_exc: BaseException | None = None
        try:
            for item in iterator:
                yield item
        except BaseException as exc:
            primary_exc = exc
            raise
        finally:
            close_iterators_best_effort(
                [iterator],
                primary_exc=primary_exc,
            )

    def _prepare_worker_index_stream(
        self,
    ) -> tuple[Iterable[int], torch.Generator | np.random.Generator | None]:
        """Prepare the original complete worker-local logical index stream.

        Only indices are consumed here, before any sample prefetch starts.
        The returned reservoir RNG remains independent of the source sampler
        and can survive finite reader residencies without retaining tables.
        """
        sampler = self.indice_sampler
        source_generator = sampler.generator
        prefetch_generator = source_generator
        if _uses_sample_prefetch(self._shuffle_config, None):
            source_generator, prefetch_generator = _split_prefetch_generator(
                source_generator
            )
            sampler = IndiceTableSampler(
                sampler.table,
                shuffle=sampler.shuffle,
                generator=source_generator,
            )
        return (
            self._prepare_torch_indices(sampler),
            prefetch_generator,
        )

    def _iter_prepared_indices(
        self,
        indices: IndiceTable,
        prefetch_generator: torch.Generator | np.random.Generator | None,
    ) -> Iterator[Any]:
        """Read a finite plan already sharded, shuffled and resampled.

        The finite view reaches the sample reader before prefetching. The
        ordinary sample prefetcher and optional inner loader retain their
        batching responsibilities; no worker sharding or sampling is repeated.
        """
        prepared = IterableWithLenDataset(
            dataset=self.dataset,
            indices=indices,
            generator=prefetch_generator,
        )
        prepared._prepared_indices = True
        prepared._shuffle_config = self._shuffle_config
        iterator = (
            iter(prepared)
            if self.batch_loader_kwargs is None
            else iter(
                TorchDataLoader(
                    prepared,
                    num_workers=0,
                    **self.batch_loader_kwargs.to_dict(),
                )
            )
        )
        primary_exc: BaseException | None = None
        try:
            yield from iterator
        except BaseException as exc:
            primary_exc = exc
            raise
        finally:
            close_iterators_best_effort([iterator], primary_exc=primary_exc)

    @property
    def total_iterator_length(self) -> int:
        """Return the exact logical row count after resampling."""
        return round(len(self.indice_sampler) * self.resample_ratio)

    @property
    def total_dataset_length(self) -> int:
        """Return the backing dataset length before index selection.

        Raises:
            ValueError: If the backing dataset does not expose a length.
        """
        if not isinstance(self.dataset, Sized):
            raise ValueError(
                "Underlying dataset does not have a length, cannot get "
                "total dataset length."
            )
        return len(self.dataset)

    def get_total_batch_num(
        self, num_workers: int, batch_size: int = 1, drop_last: bool = False
    ) -> int:
        """Calculate the batches produced by the matching DataLoader setup.

        The calculation applies the same shard-first proportional target
        allocation as runtime iteration, then applies batching independently
        to each worker's logical rows.

        Args:
            num_workers (int): DataLoader worker count. Values below two use
                one logical worker.
            batch_size (int, optional): DataLoader batch size. Defaults to 1.
            drop_last (bool, optional): Whether each worker drops its final
                incomplete batch. Defaults to False.

        Returns:
            int: Total number of batches yielded across all workers.
        """
        worker_rows = _distribute_rows_to_workers(
            base_rows=len(self.indice_sampler),
            target_rows=self.total_iterator_length,
            num_workers=num_workers,
        )
        return sum(
            _get_batch_num(
                batch_size=batch_size,
                num_samples=rows,
                drop_last=drop_last,
            )
            for rows in worker_rows
        )

    @property
    def n_shards(self) -> int:
        """Return the accelerate-compatible shard-count hint.

        The logical row count preserves the existing compatibility behavior
        expected by ``accelerate.prepare_data_loader``.
        """
        return self.total_iterator_length

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({repr(self.dataset)}, "
            f"indices={repr(self.indice_sampler)}, "
            f"resample_ratio={self.resample_ratio})"
        )

    def _iter_indices(self, indice_iter: Iterable[int]) -> Iterator[Any]:
        """Yield samples for the provided indices.

        When the wrapped dataset implements ``__getitems__``, this helper uses
        small index batches to amortize indexing overhead while still exposing
        a sample-by-sample iterator to callers.
        """
        dataset = self.dataset
        if not hasattr(dataset, "__getitems__"):
            for idx in indice_iter:
                yield dataset[idx]
            return

        batch_indices: list[int] = []
        for idx in indice_iter:
            batch_indices.append(int(idx))
            if len(batch_indices) >= _DEFAULT_VIRTUAL_GETITEMS_BATCH_SIZE:
                yield from dataset.__getitems__(batch_indices)  # type: ignore[attr-defined]
                batch_indices = []

        if batch_indices:
            yield from dataset.__getitems__(batch_indices)  # type: ignore[attr-defined]

    def _iter_resampled_indices(
        self,
        indice_sampler: IndiceTableSampler,
        target_rows: int,
    ) -> Iterator[int]:
        """Yield exactly ``target_rows`` from one physical index shard.

        Re-entering the sampler starts a new natural cycle. A shuffled
        sampler therefore produces a fresh permutation for every oversampling
        cycle without materializing a repeated index table.

        Args:
            indice_sampler (IndiceTableSampler): Worker-local physical index
                shard to consume.
            target_rows (int): Exact logical rows to yield from that shard.

        Yields:
            int: Backing-dataset indices from complete natural cycles followed
            by at most one partial cycle.

        Raises:
            RuntimeError: If sampler length metadata disagrees with iteration.
        """
        remaining = target_rows
        while remaining > 0:
            cycle_target = min(len(indice_sampler), remaining)
            cycle_count = 0
            for index in indice_sampler:
                cycle_count += 1
                yield index
                if cycle_count >= cycle_target:
                    break
            if cycle_count == 0:
                raise RuntimeError(
                    "Indice sampler reported a positive row count but "
                    "produced no rows."
                )
            if cycle_count != cycle_target:
                raise RuntimeError(
                    "Indice sampler row metadata does not match iteration: "
                    f"expected {cycle_target} rows in a cycle, but got "
                    f"{cycle_count}."
                )
            remaining -= cycle_count

    def _iter_batches(
        self,
        indice_sampler: IndiceTableSampler,
        batch_loader_kwargs: BatchLoaderConfig,
    ) -> Iterator[Any]:
        """Yield dataset-side batches and close the inner iterator.

        The inner loader uses ``num_workers=0`` and borrows the backing
        dataset. Its sample view prepares worker indices and prefetching only
        when iteration advances, after the loader's RNG initialization.
        Completion, failure and early close all close the inner iterator;
        an iteration failure remains primary if cleanup also fails.

        Args:
            indice_sampler (IndiceTableSampler): Index table and generator
                passed to the sample view without sampling them here.
            batch_loader_kwargs (BatchLoaderConfig): Batch size, tail policy
                and collation for this iterator; the config is not modified.

        Yields:
            Any: Batches built with the provided batch loader configuration.
        """
        logger.debug(
            "Iterating with batch loader, shuffle: %s, batch loader: %s",
            self._shuffle_config,
            batch_loader_kwargs,
        )
        inner_loader = torch.utils.data.DataLoader(
            dataset=IterableWithLenDataset(
                dataset=self.dataset,
                indices=indice_sampler.table,
                shard_kwargs=self.shard_kwargs,
                shuffle=self._shuffle_config,
                generator=indice_sampler.generator,
                batch_loader_kwargs=None,
                resample_ratio=self.resample_ratio,
            ),
            num_workers=0,
            **batch_loader_kwargs.to_dict(),
        )
        inner_iter = iter(inner_loader)
        primary_exc: BaseException | None = None
        try:
            for item in inner_iter:
                yield item
        except BaseException as exc:
            primary_exc = exc
            raise
        finally:
            close_iterators_best_effort(
                [inner_iter],
                primary_exc=primary_exc,
            )

    def _iter_impl(
        self,
        indice_sampler: IndiceTableSampler | None = None,
    ) -> Iterator[Any]:
        """Prepare indices, then read samples through the common reader.

        Dataset-side batching delegates to the inner loader, whose sample
        iterable uses this same preparation/read path. Prepared offload plans
        have already resolved worker ownership and sampling.
        """
        if indice_sampler is None:
            indice_sampler = self.indice_sampler
        if self.batch_loader_kwargs is not None:
            yield from self._iter_batches(
                indice_sampler, self.batch_loader_kwargs
            )
            return

        yield from self._iter_indices(
            self._prepare_torch_indices(indice_sampler)
        )

    def _prepare_torch_indices(
        self, indice_sampler: IndiceTableSampler
    ) -> Iterable[int]:
        """Resolve the current worker's final index stream before reading.

        Prepared plans already contain worker-local row anchors in their final
        sampled order. Ordinary indices are physically sharded first, then
        resampled to the allocated logical row target without materializing
        repeats. Both sample reading and offload persistence use this stage.
        """
        if self._prepared_indices:
            return indice_sampler.table

        target_rows = self.total_iterator_length
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            indice_sampler = indice_sampler.shard(
                num_shards=worker_info.num_workers,
                shard_id=worker_info.id,
                contiguous=self.shard_kwargs.contiguous,
            )
            target_rows = _distribute_rows_to_workers(
                base_rows=len(self.indice_sampler),
                target_rows=target_rows,
                num_workers=worker_info.num_workers,
            )[worker_info.id]
        return self._iter_resampled_indices(indice_sampler, target_rows)


class DatasetItemNode(Config, metaclass=ABCMeta):
    """Shared configuration contract for physical items and logical groups.

    Every node has one stable logical name and one static shard assignment.
    ``get_dataset_row_num`` always reports the physical row count before that
    assignment; :class:`DictIterableDataset` uses the shared sharded-count
    rule for ratios, summaries, and worker batch accounting.

    A :class:`DatasetItem` is one materializable physical source. A
    :class:`DatasetItemGroup` instead owns one logical name and expands to
    several physical sources only when a ``DictIterableDataset`` is built.
    In particular, a group's name is not copied to its physical members: it
    identifies the group for named resampling and summary display.
    """

    name: str | None = Field(
        default=None,
        description=(
            "Optional stable name for this dataset item or group. Dict-valued "
            "resample_ratios require a lowercase slash-separated leaf path."
        ),
    )

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
        """Return the physical row count before this node's static shard."""
        raise NotImplementedError(
            "get_dataset_row_num must be implemented by subclasses "
            "of DatasetItemNode."
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

    @property
    def is_sharded(self) -> bool:
        return self.num_shards > 1

    def shard(self, num_shards: int, index: int) -> Self:
        """Return this node's deterministic child shard configuration.

        The returned node keeps its source configuration but combines the new
        shard with any existing static shard assignment:

        - new_num_shards: self.num_shards * num_shards
        - new_shard_id: self.shard_id * num_shards + index
        """
        if index >= num_shards:
            raise ValueError(
                f"index must be in the range [0, num_shards - 1], but got "
                f"index={index} and num_shards={num_shards}."
            )
        if index < 0:
            raise ValueError(
                f"index must be non-negative, but got index={index}."
            )
        if num_shards < 1:
            raise ValueError(
                "num_shards must be at least 1, "
                f"but got num_shards={num_shards}."
            )

        return self.replace(
            num_shards=self.num_shards * num_shards,
            shard_id=self.shard_id * num_shards + index,
        )


class DatasetItem(DatasetItemNode, Generic[DatasetType], metaclass=ABCMeta):
    """Configuration for one independently materialized physical dataset.

    Subclasses provide a raw indexable dataset through `_create_dataset`.
    ``create_dataset`` wraps that dataset with the current static shard's
    physical index table. Logical groups intentionally do not inherit this
    materialization contract.

    Override :meth:`get_index_cache_key` when the default configuration digest
    cannot describe the source's stable, unsharded row selection.
    """

    class_type: ClassType[DatasetType]

    @abstractmethod
    def _create_dataset(self) -> DatasetType:
        """Create the unsharded physical dataset from this configuration."""
        raise NotImplementedError(
            "_create_dataset must be implemented by subclasses of DatasetItem."
        )

    def create_dataset(
        self, shard_config: ShardConfig
    ) -> DatasetWithIndices[DatasetType]:
        """Materialize this physical item with its configured index shard."""
        dataset = self._create_dataset()
        try:
            ret = DatasetWithIndices(dataset=dataset)
            if self.is_sharded:
                return ret.shard(
                    num_shards=self.num_shards,
                    index=self.shard_id,
                    **shard_config.to_dict(),
                )
            return ret
        except BaseException as exc:
            close_iterators_best_effort([dataset], primary_exc=exc)
            raise

    def get_index_cache_key(self) -> str:
        """Return a stable key for this source's unsharded row selection.

        Equal keys must describe the same reader/source version, row anchors
        and order across processes and epochs. Overrides may use any
        deterministic scheme and must not materialize readers or indices.
        The cache separately namespaces keys by node and working directory.

        The default hashes canonical JSON configuration, excluding display
        names, static shards and runtime private caches. Override this method
        for opaque configuration values or a source-specific version key;
        JSON serialization is not required of custom implementations.

        Returns:
            str: An opaque source/selection identity, independent of shuffle,
            resampling and iteration progress.
        """
        config = self.to_dict(
            mode="json",
            include_config_type=True,
            exclude={"name", "num_shards", "shard_id"},
        )
        encoded = json.dumps(config, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(slots=True)
class _DatasetItemGroupLayout:
    """Reader-free physical members and their aggregate row count."""

    items: tuple[DatasetItem[Any], ...]
    total_rows: int


class DatasetItemGroup(DatasetItemNode, metaclass=ABCMeta):
    """Configuration-only group of several physical dataset items.

    Use a group when several sources should share one logical name, resample
    ratio, and :meth:`DictIterableDataset.summary` leaf, while continuing to
    be read and scheduled as independent physical :class:`DatasetItem`
    values. A group never creates a concatenated runtime dataset or a shared
    reader. ``DictIterableDataset`` expands it to physical leaves before
    iteration, preserving the established reader lifecycle and worker path.

    The group's resolved ratio is inherited by every member. Each member
    independently targets ``round(member_sharded_rows * ratio)``; a group
    therefore has no group-level rounding. Its static shard is likewise
    applied independently to every member. Dataset-side batches never span
    members of a group.

    Subclasses implement :meth:`iter_dataset_items` to describe their leaves.
    The sequence must be fresh and deterministic, and every value must be an
    unsharded physical :class:`DatasetItem` configuration rather than a reader
    object. Nested groups are deliberately unsupported in V1.
    """

    @abstractmethod
    def iter_dataset_items(self) -> Iterator[DatasetItem[Any]]:
        """Yield this group's fresh, unsharded physical item configurations.

        The order must be deterministic. Yield physical ``DatasetItem``
        configurations, not instantiated readers or another
        ``DatasetItemGroup``. Leaves do not need logical names: the parent
        group owns the name used for resampling and summary display.
        """
        raise NotImplementedError

    def get_dataset_row_num(self) -> int:
        """Return the sum of all physical member row counts.

        Each call builds and validates the reader-free member configuration
        layout from the current group fields. A member may perform its own
        documented metadata fallback; the group does not concatenate or
        retain runtime datasets.
        """
        return self._get_layout().total_rows

    def get_sharded_row_num(self, shard_config: ShardConfig) -> int:
        """Return the sum of independently sharded physical member rows.

        This is intentionally not equivalent to sharding the aggregate count:
        every member receives the group's static shard independently.
        """
        return sum(
            item.shard(
                num_shards=self.num_shards,
                index=self.shard_id,
            ).get_sharded_row_num(shard_config)
            for item in self._get_layout().items
        )

    def _get_layout(
        self, *, isolate_indices: bool = False
    ) -> _DatasetItemGroupLayout:
        """Build a layout from the current mutable configuration."""

        items = tuple(self.iter_dataset_items())
        total_rows = 0
        for item_index, item in enumerate(items):
            if isinstance(item, DatasetItemGroup):
                raise TypeError(
                    "DatasetItemGroup.iter_dataset_items must yield physical "
                    "DatasetItem leaves, not nested DatasetItemGroup values."
                )
            if not isinstance(item, DatasetItem):
                raise TypeError(
                    "DatasetItemGroup.iter_dataset_items must yield "
                    f"DatasetItem values, but item {item_index} is "
                    f"{type(item).__name__}."
                )
            if item.is_sharded:
                raise ValueError(
                    "DatasetItemGroup member items must be unsharded because "
                    "the group shard is applied independently to every member."
                )
            count_item = _fresh_config(item) if isolate_indices else item
            row_count = count_item.get_dataset_row_num()
            if isinstance(row_count, bool) or not isinstance(row_count, int):
                raise TypeError(
                    "DatasetItem.get_dataset_row_num must return an int, but "
                    f"item {item_index} returned {row_count!r}."
                )
            if row_count < 0:
                raise ValueError(
                    "DatasetItem.get_dataset_row_num must not return a "
                    "negative value, but item "
                    f"{item_index} returned {row_count}."
                )
            total_rows += row_count

        return _DatasetItemGroupLayout(
            items=items,
            total_rows=total_rows,
        )


class DictIterableDataset(TorchIterableDataset, IterableDatasetMixin):
    """Mix multiple dataset items with optional per-item resampling.

    Use this dataset to combine physical ``DatasetItem`` sources and
    configuration-only ``DatasetItemGroup`` sources. Groups are expanded into
    their physical members before iteration, so every reader, shard, batch,
    and scheduler path remains the existing ``DatasetItem`` path. A group
    nevertheless retains one logical name, ratio, and summary leaf.

    ``resample_ratios`` is a row-count multiplier, not a probability. A group
    ratio is inherited by every member, and each member independently uses
    ``round(member_sharded_rows * ratio)``. Thus groups do not introduce a
    concatenated runtime dataset or group-level rounding.

    Shuffled mixing weights both reader admission and active sample/batch
    selection by effective worker-local epoch row targets, compensating for
    each selection's output size. This preserves the bounded reader pool and
    exact epoch quotas; it does not guarantee IID finite-window proportions.

    ``resample_ratios`` has four construction-time forms:

    - ``None``: every item uses ``1.0``.
    - One ``float``: apply the same ratio to every item.
    - An ``Iterable[float]``: provide one ratio per logical input, in
      ``datasets`` order.
    - A ``dict[str, float]``: apply ratios by logical item name path prefix.

    Dict keys and item names are lowercase relative paths whose segments match
    ``[a-z0-9][a-z0-9._-]*``. Matching is by complete path segment: ``"a"``
    matches ``"a"`` and ``"a/..."``, but not ``"abc"``. The deepest matching
    key wins, and an item with no matching key uses ``1.0``. In dict mode, every
    logical input must have a unique leaf name: ``"a"`` cannot coexist with
    ``"a/b"``. A group's physical members do not need names; its parent name
    is the one validated and matched.
    Every dict key must match at least one item, so misspelled rules fail during
    construction instead of silently doing nothing. Scalar and iterable forms
    do not require item names.

    For every form, ratios must be finite and strictly positive; any final ratio
    other than ``1.0`` requires ``shuffle=True``. Ratios are resolved once at
    construction, so later mutation of an item's name does not alter sampling.

    Examples:
        Use no explicit ratio when every selected item should contribute its
        original row count:

        .. code-block:: python

            dataset = DictIterableDataset(datasets=items)
            # Every item uses 1.0. Names are optional in this form.

        Apply one multiplier to every item, or provide one multiplier in the
        same order as ``datasets``:

        .. code-block:: python

            uniformly_downsampled = DictIterableDataset(
                datasets=items,
                shuffle=True,
                resample_ratios=0.5,
            )
            explicitly_weighted = DictIterableDataset(
                datasets=[training_items["a/b"], training_items["d/e"]],
                shuffle=True,
                resample_ratios=[2.0, 0.5],
            )
            # The resolved ratios are [2.0, 0.5], in datasets order.

        Use dict keys to weight a path subtree, then override a more specific
        leaf or subtree:

        .. code-block:: python

            path_weighted = DictIterableDataset(
                datasets=[
                    training_items["a/b"],
                    training_items["a/c"],
                    training_items["d/e"],
                ],
                shuffle=True,
                resample_ratios={"a": 0.5, "a/b": 2.0},
            )
            # Resolved ratios: [2.0, 0.5, 1.0].

        Group several physical RODataset roots under one logical path without
        concatenating them. The ratio is resolved from the group's name and
        then independently applied to each path:

        .. code-block:: python

            from robo_orchard_lab.dataset.robot.dataset import (
                RODatasetItemGroup,
            )

            bridge = RODatasetItemGroup(
                name="open_x/bridge",
                dataset_path=[
                    "/datasets/bridge-000",
                    "/datasets/bridge-001",
                ],
            )
            grouped = DictIterableDataset(
                datasets=[bridge],
                shuffle=True,
                resample_ratios={"open_x": 0.5, "open_x/bridge": 2.0},
            )
            # Each physical path targets round(its own sharded row count * 2.0).
            # summary() keeps one logical "open_x/bridge" leaf.

        Dict paths are a namespace, not arbitrary string prefixes. For
        example, this configuration is rejected because dataset item names
        must be leaves of one tree:

        .. code-block:: python

            invalid = DictIterableDataset(
                datasets=[training_items["a"], training_items["a/b"]],
                shuffle=True,
                resample_ratios={"a": 1.0},
            )
            # Raises ValueError: "a" is an ancestor of "a/b".

    In multi-worker loading, every child dataset is physically sharded first
    and resampled only within that shard.

    When dataset-side batching is configured, resampling still happens at the
    row level. Every physical item owns its own batch stream, so a group does
    not form batches across member boundaries and ``drop_last`` applies to
    each member's final incomplete batch.

    Item names for :meth:`summary` come from :attr:`DatasetItemNode.name`.
    Unnamed items use ``(unnamed #0)``, ``(unnamed #1)``, and so on.

    Args:
        datasets (Iterable[DatasetItemNode]): Physical items or logical groups
            to mix. Ratio iterables have one entry per input here, not one per
            physical member expanded from a group.
        shuffle (bool | ShuffleConfig, optional): Whether to shuffle the dataset
            indices. If a ShuffleConfig is provided, it will be used to configure
            the shuffling behavior. Defaults to False, which means no shuffling
            will be applied.
        shard_kwargs (ShardConfig | None, optional): Configuration for
            sharding the dataset. Sharding will be applied when using multiple
            processors in `accelerate`. Defaults to None, which means the
            default sharding strategy will be used (contiguous shards).
        generator (torch.Generator | np.random.Generator | None, optional): An
            optional generator to use for shuffling. If None, a new generator
            will be created with a random seed. Defaults to None.
        batch_loader_kwargs (BatchLoaderConfig | dict | None, optional): An
            optional configuration for using a batch loader. If provided, the
            dataset will be wrapped with a DataLoader to return batches of
            data. Defaults to None, which means no batch loader will be used.
        max_dataset_concurrency (int, optional): Maximum number of physical
            item iterators kept active by the weighted scheduler. One group can
            expand to several such iterators. Defaults to 4.
        resample_ratios (float | Iterable[float] | dict | None, optional):
            Row-count multiplier configuration. Use one of the four forms
            described above. In dict form, keys are path strings and values are
            floats. Group members inherit their group's resolved multiplier and
            round independently. Defaults to None, which uses 1.0 for each
            logical input.
        offload_num_batches (int | None, optional): Positive target batch
            count per worker/item reader residency. Complete worker-local
            shuffled/resampled indices are persisted before finite segments
            are read. A short remainder merges into the preceding segment;
            final tail-batch rules stay unchanged. ``None`` disables offload.
            Sample mode uses this repository's DataLoader batch size, or one
            sample per unit for direct iteration/external loaders.
        offload_dir (str | None, optional): Parent of the node-shared immutable
            base index cache and private iteration plans. Defaults to the
            system temporary directory. Base files survive epochs; private
            plans are removed after normal exhaustion and retained after an
            interrupted iteration. Inputs must stay immutable while cached.

    Offload supports both shuffle and batching modes. Enabling it changes
    mixture/reservoir delivery order; it does not provide training-checkpoint
    recovery. At construction it snapshots input items and groups from their
    declared fields, omitting runtime private caches without changing caller
    objects. Later caller configuration changes do not update those snapshots.
    Only active readers retain index mappings. The concurrency limit applies
    per mixture/worker instance, not to process RSS or batches already delivered
    to the caller.

    Raises:
        TypeError: If ratios are not numeric.
        ValueError: If iterable argument lengths do not match ``datasets``;
            dict paths are invalid, unmatched, or not unique leaf input names;
            a group yields an invalid member; or a ratio is non-finite, not
            positive, or non-unit while shuffling is disabled.

    """  # noqa: E501

    logical_dataset_items: list[DatasetItemNode]
    """Logical configurations used for names, ratios and summaries.

    Offload retains field-only snapshots without caller runtime caches;
    ordinary iteration retains the original input objects.
    """

    dataset_items: list[DatasetItem[Any]]
    """Physical items expanded from :attr:`logical_dataset_items`.

    These are the items used at runtime.
    """

    def __init__(
        self,
        datasets: Iterable[DatasetItemNode],
        shuffle: bool | ShuffleConfig = False,
        shard_kwargs: ShardConfig | None = None,
        generator: torch.Generator | np.random.Generator | None = None,
        batch_loader_kwargs: BatchLoaderConfig | dict | None = None,
        max_dataset_concurrency: int = 4,
        resample_ratios: float
        | Iterable[float]
        | dict[str, float]
        | None = None,
        offload_num_batches: int | None = None,
        offload_dir: str | None = None,
    ):
        # try to make this instance compatible with HF Iterable at class-level
        # or instance-level if class-level MRO change fails
        # _add_hf_iterable_cls(self.__class__, instance=self)
        self.logical_dataset_items = list(datasets)

        if offload_num_batches is not None and (
            isinstance(offload_num_batches, bool)
            or not isinstance(offload_num_batches, int)
            or offload_num_batches <= 0
        ):
            raise ValueError("offload_num_batches must be a positive integer.")
        self.offload_num_batches = offload_num_batches
        self.offload_dir = (
            os.fspath(offload_dir) if offload_dir is not None else None
        )
        self._offload_batch_size = 1
        self._item_sharded_rows: dict[int, int] = {}
        if self.offload_num_batches is not None:
            self.logical_dataset_items = [
                _fresh_config(item)
                if isinstance(item, DatasetItemNode)
                else item
                for item in self.logical_dataset_items
            ]

        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)

        if isinstance(shuffle, bool):
            shuffle = ShuffleConfig(shuffle=shuffle)

        self._shard_kwargs = (
            shard_kwargs if shard_kwargs is not None else ShardConfig()
        )
        self._generator = generator
        self._shuffle = shuffle
        if isinstance(batch_loader_kwargs, dict):
            batch_loader_kwargs = BatchLoaderConfig(**batch_loader_kwargs)
        self._batch_loader_kwargs = batch_loader_kwargs
        self._max_dataset_concurrency = max_dataset_concurrency

        if isinstance(resample_ratios, dict):
            logical_resample_ratios = _resolve_named_resample_ratios(
                dataset_items=self.logical_dataset_items,
                resample_ratios=resample_ratios,
            )
        else:
            if resample_ratios is None:
                ratio_values: list[Any] = [1.0] * len(
                    self.logical_dataset_items
                )
            elif isinstance(resample_ratios, Iterable) and not isinstance(
                resample_ratios, (str, bytes)
            ):
                try:
                    ratio_values = list(resample_ratios)
                except TypeError as exc:
                    raise TypeError(
                        "resample_ratios must be a float or an iterable of "
                        "floats."
                    ) from exc
            else:
                ratio_values = [resample_ratios] * len(
                    self.logical_dataset_items
                )

            if len(ratio_values) != len(self.logical_dataset_items):
                raise ValueError(
                    "resample_ratios must contain one value per dataset item, "
                    f"but got {len(ratio_values)} values for "
                    f"{len(self.logical_dataset_items)} items."
                )
            logical_resample_ratios = [
                _normalize_resample_ratio(
                    ratio,
                    name=f"resample_ratios[{index}]",
                )
                for index, ratio in enumerate(ratio_values)
            ]
        if (
            any(ratio != 1.0 for ratio in logical_resample_ratios)
            and not self._shuffle.shuffle
        ):
            raise ValueError(
                "resample_ratios values other than 1.0 require shuffle=True."
            )
        self._logical_resample_ratios = logical_resample_ratios
        self.dataset_items = []
        self._logical_item_indices: list[tuple[int, ...]] = []
        self._resample_ratios: list[float] = []
        for logical_item, ratio in zip(
            self.logical_dataset_items,
            self._logical_resample_ratios,
            strict=True,
        ):
            start_index = len(self.dataset_items)
            expanded_items = self._expand_dataset_item_node(logical_item)
            self.dataset_items.extend(expanded_items)
            self._resample_ratios.extend([ratio] * len(expanded_items))
            self._logical_item_indices.append(
                tuple(range(start_index, start_index + len(expanded_items)))
            )
        self._total_dataset_length: list[int] | None = None
        self._total_indices_length: list[int] | None = None

    def __getstate__(self) -> dict[str, Any]:
        """Serialize Torch RNG state without multiprocessing shared storage.

        A ``torch.Generator`` stores its state in a tensor. Under a spawned
        DataLoader worker, Torch's default ``file_descriptor`` IPC strategy
        transports that tensor as shared storage, which is not reliable across
        every supported runtime. Plain bytes preserve the exact RNG sequence
        while keeping dataset startup independent of Torch storage IPC.
        """

        state = self.__dict__.copy()
        generator = state.get("_generator")
        if isinstance(generator, torch.Generator):
            state["_generator"] = None
            state["_serialized_torch_generator"] = (
                str(generator.device),
                bytes(generator.get_state().tolist()),
            )
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a Torch generator serialized by :meth:`__getstate__`."""

        serialized_generator = state.pop(
            "_serialized_torch_generator",
            None,
        )
        if serialized_generator is not None:
            device, generator_state = serialized_generator
            generator = torch.Generator(device=device)
            generator.set_state(
                torch.tensor(list(generator_state), dtype=torch.uint8)
            )
            state["_generator"] = generator
        self.__dict__.update(state)

    @property
    def batch_loader_kwargs(self) -> BatchLoaderConfig | None:
        return self._batch_loader_kwargs

    @property
    def shard_kwargs(self) -> ShardConfig:
        return self._shard_kwargs

    def shard(self, num_shards: int, index: int) -> DictIterableDataset:
        """Shard every logical item before expanding group members."""
        sharded_items = [
            item.shard(num_shards=num_shards, index=index)
            for item in self.logical_dataset_items
        ]
        result = DictIterableDataset(
            datasets=sharded_items,
            shuffle=self._shuffle,
            generator=self._generator,
            batch_loader_kwargs=self.batch_loader_kwargs,
            max_dataset_concurrency=self._max_dataset_concurrency,
            shard_kwargs=self.shard_kwargs,
            resample_ratios=self._logical_resample_ratios,
            offload_num_batches=self.offload_num_batches,
            offload_dir=self.offload_dir,
        )
        result._offload_batch_size = self._offload_batch_size
        return result

    def __repr__(self) -> str:
        """Return a safe summary repr for notebook and console display.

        The runtime class also inherits from Hugging Face's
        ``IterableDataset`` for compatibility with downstream integrations.
        That base class expects internal attributes such as ``_info`` and
        ``_ex_iterable`` to exist when building its repr, which this custom
        iterable does not initialize. Defining a local repr keeps interactive
        display and debugging safe without changing the dataset's iteration
        behavior.

        Returns:
            str: Concise summary of the iterable dataset configuration.
        """
        dataset_items_repr = ",\n    ".join(
            repr(item) for item in self.logical_dataset_items
        )
        if dataset_items_repr:
            dataset_items_repr = f"[\n    {dataset_items_repr}\n  ]"
        else:
            dataset_items_repr = "[]"

        return (
            f"{self.__class__.__name__}("
            f"dataset_items={len(self.logical_dataset_items)}, "
            f"items={dataset_items_repr}, "
            f"shuffle={self._shuffle.shuffle}, "
            f"batch_loader_kwargs={self.batch_loader_kwargs!r}, "
            f"max_dataset_concurrency={self._max_dataset_concurrency})"
        )

    def summary(self) -> str:
        """Return a display tree for the current shard's dataset mixture.

        ``sample_ratio`` uses resampled row counts, while ``frame_ratio`` and
        ``length`` use the real sharded row counts before resampling. These
        sharded lengths can be smaller than :attr:`total_dataset_length`.
        ``resampled_length`` reports rows after per-physical-member resampling;
        its total equals :attr:`total_iterator_length`.
        Multipliers display at most five significant digits; sampling uses
        their full precision.
        With dataset-side batching, row-level sample ratios can differ slightly
        from the observed batch ratio because each dataset item forms batches
        independently and may drop one final incomplete batch.

        A :class:`DatasetItemGroup` remains one leaf in this table. Its
        ``length`` and row-level ratios aggregate its physical members, while
        its ``resample_ratio`` is the one multiplier configured for the group.
        Because members are rounded independently, the group's resampled count
        need not equal ``round(group_length * resample_ratio)``.

        Path-like names are shown as a compact tree. Aggregate nodes summarize
        every descendant but show ``—`` for ``resample_ratio`` because their
        children can use different multipliers. Leaf rows show the resolved
        multiplier. The display tree is intentionally tolerant: scalar/list
        ratio callers and post-construction name changes can therefore expose
        duplicate, overlapping, missing, or malformed names without making
        this diagnostic method fail. Dict-form ``resample_ratios`` remains the
        separate strict construction-time validation boundary.

        Returns:
            str: A display-width-aligned table suitable for logs or consoles.

        Example:
            ``print(dataset.summary())`` renders a table like::

                name              sample_ratio [frame_ratio] [length] resample_ratio [resampled_length]
                ├─ source               100.00%       100.00%       20              —                 40
                │  ├─ rgb                25.00%        50.00%       10             ×1                 10
                │  └─ depth              75.00%        50.00%       10             ×3                 30
                └─ total                100.00%       100.00%       20              —                 40
        """  # noqa: E501

        def char_width(char: str) -> int:
            if unicodedata.combining(char):
                return 0
            return 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1

        def display_width(value: str) -> int:
            return sum(char_width(char) for char in value)

        def align_display(
            value: str,
            width: int,
            *,
            left: bool = False,
        ) -> str:
            padding = " " * (width - display_width(value))
            return value + padding if left else padding + value

        _ = self.total_iterator_length
        assert self._total_indices_length is not None
        scaled_rows = [
            sum(self._total_indices_length[index] for index in item_indices)
            for item_indices in self._logical_item_indices
        ]
        base_rows = [
            sum(self._get_item_row_num(index) for index in item_indices)
            for item_indices in self._logical_item_indices
        ]
        total_scaled_rows = sum(scaled_rows)
        total_base_rows = sum(base_rows)
        summary_root, root_item_labels = _build_path_tree(
            item.name for item in self.logical_dataset_items
        )
        summary_rows = _format_path_tree(summary_root, root_item_labels)
        summary_rows.append(
            ("└─ total", tuple(range(len(self.logical_dataset_items))), None)
        )
        name_width = max(
            len("name"),
            30,
            *(display_width(label) for label, _, _ in summary_rows),
        )
        sample_ratio_width = len("sample_ratio")
        frame_ratio_width = len("frame_ratio")
        length_width = max(len("length"), len(str(total_base_rows)))
        resampled_length_width = max(
            len("resampled_length"), len(str(total_scaled_rows))
        )
        resample_ratio_width = len("resample_ratio")

        def format_summary_row(
            label: str,
            item_indices: tuple[int, ...],
            leaf_item_index: int | None,
        ) -> str:
            scaled_length = sum(scaled_rows[index] for index in item_indices)
            base_length = sum(base_rows[index] for index in item_indices)
            sample_ratio = (
                scaled_length / total_scaled_rows
                if total_scaled_rows > 0
                else 0.0
            )
            frame_ratio = (
                base_length / total_base_rows if total_base_rows > 0 else 0.0
            )
            resample_ratio = (
                f"×{self._logical_resample_ratios[leaf_item_index]:.5g}"
                if leaf_item_index is not None
                else "—"
            )
            return (
                f"{align_display(label, name_width, left=True)} "
                f"{sample_ratio:>{sample_ratio_width}.2%} "
                f"[{frame_ratio:>{frame_ratio_width}.2%}] "
                f"[{base_length:>{length_width}}] "
                f"{align_display(resample_ratio, resample_ratio_width)} "
                f"[{scaled_length:>{resampled_length_width}}]"
            )

        lines = [
            f"{align_display('name', name_width, left=True)} "
            f"{'sample_ratio':>{sample_ratio_width}} "
            f"[{'frame_ratio':>{frame_ratio_width}}] "
            f"[{'length':>{length_width}}] "
            f"{align_display('resample_ratio', resample_ratio_width)} "
            f"[{'resampled_length':>{resampled_length_width}}]"
        ]
        lines.extend(format_summary_row(*row) for row in summary_rows)
        return "\n".join(lines)

    @property
    def total_dataset_length(self) -> int:
        """Return the physical row count before configured sharding."""
        if self._total_dataset_length is None:
            self._total_dataset_length = [
                self._get_item_row_num(index, sharded=False)
                for index in range(len(self.dataset_items))
            ]
        return sum(self._total_dataset_length)

    @property
    def total_iterator_length(self) -> int:
        """Return the exact combined logical row count after resampling."""
        if self._total_indices_length is None:
            self._total_indices_length = [
                round(self._get_item_row_num(index) * ratio)
                for index, ratio in enumerate(self._resample_ratios)
            ]
        return sum(self._total_indices_length)

    def get_total_batch_num(
        self, num_workers: int, batch_size: int, drop_last: bool
    ) -> int:
        """Calculate batches for the matching mixed-dataset DataLoader setup.

        Row targets are first allocated per dataset item and physical worker
        shard. With dataset-side batching, each item-worker stream forms its
        own batches; otherwise each worker batches the combined item streams.

        Args:
            num_workers (int): DataLoader worker count. Values below two use
                one logical worker.
            batch_size (int): Batch size applied by the relevant batching
                layer.
            drop_last (bool): Whether each independently batched stream drops
                its final incomplete batch.

        Returns:
            int: Total number of batches yielded across all workers.
        """
        _ = self.total_iterator_length
        assert self._total_indices_length is not None
        base_rows = [
            self._get_item_row_num(index)
            for index in range(len(self.dataset_items))
        ]
        item_worker_rows = [
            _distribute_rows_to_workers(
                base_rows=item_base_rows,
                target_rows=target_rows,
                num_workers=num_workers,
            )
            for item_base_rows, target_rows in zip(
                base_rows,
                self._total_indices_length,
                strict=True,
            )
        ]

        if self.batch_loader_kwargs is not None:
            return sum(
                _get_batch_num(
                    batch_size=batch_size,
                    num_samples=worker_rows,
                    drop_last=drop_last,
                )
                for item_rows in item_worker_rows
                for worker_rows in item_rows
            )

        if num_workers <= 1:
            return _get_batch_num(
                batch_size=batch_size,
                num_samples=self.total_iterator_length,
                drop_last=drop_last,
            )

        return sum(
            _get_batch_num(
                batch_size=batch_size,
                num_samples=sum(
                    item_rows[worker_id] for item_rows in item_worker_rows
                ),
                drop_last=drop_last,
            )
            for worker_id in range(num_workers)
        )

    @property
    def n_shards(self) -> int:
        """Return an accelerate-compatible shard count hint.

        ``accelerate.prepare_data_loader`` only uses the native Hugging Face
        iterable-dataset sharding path when ``n_shards > num_processes``.
        Keep this value strictly larger than the current process count so
        accelerate prefers dataset-native sharding over its much slower
        ``IterableDatasetShard`` wrapper.
        """
        from accelerate.state import AcceleratorState

        state = AcceleratorState()
        return max(self.total_iterator_length, state.num_processes + 1)

    def __iter__(self):
        """Yield a row-weighted mixture and close all materialized datasets.

        Shuffled iteration uses compensated reader-admission and active-yield
        weights. Ordered iteration keeps the original source order. A completed
        segment closes after normal EOF and successful owned cleanup before
        its unfinished source can re-enter, including immediate reselection;
        sources need not complete a physical round before entering again.
        Failure or caller abandonment terminates this iteration without
        committing the interrupted segment.

        Yields:
            Any: One sample or dataset-side batch from an active dataset item.
        """
        cur_dataset_iters: list[tuple[int, Iterator[Any]]] = []
        dataset_indices = list(range(len(self.dataset_items)))
        _ = self.total_iterator_length
        assert self._total_indices_length is not None
        batch_size = (
            self.batch_loader_kwargs.batch_size
            if self.batch_loader_kwargs is not None
            else 1
        )
        segment_remaining: dict[int, int] = {}
        with self._iteration_sources() as (
            make_iter,
            offload_cursors,
            worker_targets,
        ):
            primary_exc: BaseException | None = None
            try:
                weights = self._prepare_dataset_for_iter(
                    cur_dataset_iters,
                    dataset_indices,
                    make_iter,
                    worker_targets,
                    segment_remaining,
                    offload_cursors,
                )
                while cur_dataset_iters:
                    selected_idx = (
                        self._sample_dataset_index(weights)
                        if self._shuffle.shuffle
                        else 0
                    )
                    item_idx, iter_dataset = cur_dataset_iters[selected_idx]
                    remaining = segment_remaining[item_idx]
                    try:
                        item = next(iter_dataset)
                    except StopIteration as exc:
                        raise RuntimeError(
                            "Dataset item exhausted before its allocated "
                            "output rows."
                        ) from exc
                    unit_rows = min(batch_size, remaining)
                    segment_remaining[item_idx] -= unit_rows
                    yield item
                    if segment_remaining[item_idx] == 0:
                        # Resume the owner through EOF/cleanup before requeue.
                        # Early close after the last yield must not commit.
                        try:
                            next(iter_dataset)
                        except StopIteration:
                            pass
                        else:
                            raise RuntimeError(
                                "Dataset item exceeded its allocated "
                                "output rows."
                            )
                        cur_dataset_iters.pop(selected_idx)
                        del segment_remaining[item_idx]
                        close_iterators_best_effort([iter_dataset])
                        if (
                            offload_cursors is not None
                            and not offload_cursors[item_idx].exhausted
                        ):
                            dataset_indices.append(item_idx)
                        weights = self._prepare_dataset_for_iter(
                            cur_dataset_iters,
                            dataset_indices,
                            make_iter,
                            worker_targets,
                            segment_remaining,
                            offload_cursors,
                        )
                    elif segment_remaining[item_idx] < batch_size:
                        # A non-drop short batch is a different selection unit.
                        weights = self._prepare_dataset_for_iter(
                            cur_dataset_iters,
                            dataset_indices,
                            make_iter,
                            worker_targets,
                            segment_remaining,
                            offload_cursors,
                        )
            except BaseException as exc:
                primary_exc = exc
                raise
            finally:
                close_iterators_best_effort(
                    [iter_dataset for _, iter_dataset in cur_dataset_iters],
                    primary_exc=primary_exc,
                )

    def _sample_dataset_index(self, weights: np.ndarray) -> int:
        """Draw a scheduling choice using only this mixture's explicit RNG."""
        if isinstance(self._generator, np.random.Generator):
            return int(self._generator.choice(len(weights), p=weights))
        if isinstance(self._generator, torch.Generator):
            return int(
                torch.multinomial(
                    torch.from_numpy(weights), 1, generator=self._generator
                ).item()
            )
        raise ValueError(
            "Generator must be either a torch.Generator "
            "or a numpy.random.Generator."
        )

    def _expand_dataset_item_node(
        self, item_node: DatasetItemNode
    ) -> tuple[DatasetItem[Any], ...]:
        """Expand one logical input node into independently scheduled leaves.

        A physical item remains one leaf. A group applies its static shard to
        every generated member, but otherwise has no runtime representation.
        With offload enabled, count on fresh configs so retained leaves do not
        cache selection indices; otherwise keep the normal counting path.
        """
        if isinstance(item_node, DatasetItem):
            return (item_node,)
        if isinstance(item_node, DatasetItemGroup):
            return tuple(
                item.shard(
                    num_shards=item_node.num_shards,
                    index=item_node.shard_id,
                )
                for item in item_node._get_layout(
                    isolate_indices=self.offload_num_batches is not None
                ).items
            )
        raise TypeError(
            "DictIterableDataset only supports DatasetItem or "
            f"DatasetItemGroup values, but got {type(item_node).__name__}."
        )

    def _iter_dataset_item(
        self,
        data_item: DatasetItem[Any],
        *,
        resample_ratio: float,
        expected_rows: int,
    ) -> Iterator[Any]:
        """Iterate one materialized item and close its resources.

        ``DictIterableDataset`` owns every dataset materialized from a
        ``DatasetItem``. Closing this generator therefore closes both the
        child iterator stack and the underlying dataset on exhaustion, early
        break, or failure. Resampling itself belongs to the child
        ``IterableWithLenDataset``.

        Args:
            data_item (DatasetItem): Lazy physical configuration to materialize
                for this iterator.
            resample_ratio (float): Validated ratio forwarded to the child
                iterable.
            expected_rows (int): Logical rows derived from item metadata.

        Yields:
            Any: Samples or dataset-side batches from the child iterable.

        Raises:
            RuntimeError: If materialized row metadata disagrees with the
                configured item metadata.
        """

        dataset: Any = None
        dataset_iter: Iterator[Any] | None = None
        primary_exc: BaseException | None = None
        try:
            dataset_with_indices = data_item.create_dataset(
                shard_config=self.shard_kwargs
            )
            dataset = dataset_with_indices.dataset
            iterable_dataset = dataset_with_indices.to_iterable_dataset(
                shuffle=self._shuffle,
                shard_kwargs=self.shard_kwargs,
                generator=self._generator,
                batch_loader_kwargs=self.batch_loader_kwargs,
                resample_ratio=resample_ratio,
            )
            _validate_dataset_item_rows(
                iterable_dataset.total_iterator_length, expected_rows
            )
            dataset_iter = iter(iterable_dataset)
            yield from dataset_iter
        except BaseException as exc:
            primary_exc = exc
            raise
        finally:
            close_iterators_best_effort(
                [dataset_iter, dataset],
                primary_exc=primary_exc,
            )

    def _get_item_row_num(self, index: int, *, sharded: bool = True) -> int:
        """Reuse fixed scalar counts without retaining offload selections."""
        if sharded and index in self._item_sharded_rows:
            return self._item_sharded_rows[index]
        item = self.dataset_items[index]
        if self.offload_num_batches is not None:
            item = _fresh_config(item)
        rows = (
            item.get_sharded_row_num(shard_config=self.shard_kwargs)
            if sharded
            else item.get_dataset_row_num()
        )
        if sharded:
            self._item_sharded_rows[index] = rows
        return rows

    @contextmanager
    def _iteration_sources(
        self,
    ) -> Iterator[
        tuple[
            Callable[[int], Iterator[Any] | None],
            dict[int, _OffloadCursor] | None,
            list[int],
        ]
    ]:
        """Bind lazy item creation and own this iteration's private plans.

        Enter after resolving logical row targets. With offload enabled,
        scalar metadata is warmed before any reader opens, and worker-local
        cursors share the node's immutable base cache. The returned factory
        creates generators without advancing them; ``None`` means an offload
        cursor has no remaining worker-local rows. Ordinary empty-worker
        streams keep their existing lazy admission behavior.

        The caller must close active iterators inside this context. Private
        plans are deleted only after normal context exit and successful caller
        cleanup. Read failures, early close and cleanup failures propagate
        through the yield and retain the private directory. Shared base files
        are retained across iterations.

        Yields:
            tuple: Lazy item factory, optional continuation cursors, and raw
                worker-local targets shared by cursor and mixture accounting.
        """
        targets = self._total_indices_length
        assert targets is not None
        offload_cursors: dict[int, _OffloadCursor] | None = None
        private_dir: str | None = None
        worker = torch.utils.data.get_worker_info()
        worker_targets = [
            _distribute_rows_to_workers(
                base_rows=self._get_item_row_num(idx),
                target_rows=target,
                num_workers=worker.num_workers if worker else 1,
            )[worker.id if worker else 0]
            for idx, target in enumerate(targets)
        ]
        if self.offload_num_batches is not None:
            # Warm all counts before resident readers fill the active slots.
            _ = self.total_dataset_length
            parent = self.offload_dir
            if parent is None:
                parent = os.path.join(
                    tempfile.gettempdir(),
                    f"robo-orchard-offload-{os.getuid()}",
                )
            node = hashlib.sha256(socket.gethostname().encode()).hexdigest()[
                :16
            ]
            cache_root = os.path.join(parent, node)
            os.makedirs(cache_root, mode=0o700, exist_ok=True)
            private_dir = tempfile.mkdtemp(prefix="iteration-", dir=cache_root)
            batch_size = (
                self.batch_loader_kwargs.batch_size
                if self.batch_loader_kwargs is not None
                else self._offload_batch_size
            )
            segment_rows = self.offload_num_batches * batch_size
            offload_cursors = {}
            for idx, worker_rows in enumerate(worker_targets):
                offload_cursors[idx] = _OffloadCursor(
                    cache_root=cache_root,
                    plan_path=os.path.join(private_dir, f"item-{idx}.arrow"),
                    total_rows=worker_rows,
                    segment_rows=segment_rows,
                )

        def make_iter(index: int) -> Iterator[Any] | None:
            data_item = self.dataset_items[index]
            if offload_cursors is None:
                return self._iter_dataset_item(
                    data_item,
                    resample_ratio=self._resample_ratios[index],
                    expected_rows=targets[index],
                )
            cursor = offload_cursors[index]
            if cursor.exhausted:
                return None
            return cursor.iter_segment(
                data_item,
                shard_kwargs=self.shard_kwargs,
                shuffle=self._shuffle,
                generator=self._generator,
                batch_loader_kwargs=self.batch_loader_kwargs,
                resample_ratio=self._resample_ratios[index],
                expected_rows=targets[index],
            )

        yield make_iter, offload_cursors, worker_targets
        if private_dir is not None:
            shutil.rmtree(private_dir)

    def _prepare_dataset_for_iter(
        self,
        cur_dataset_iters: list[tuple[int, Iterator[Any]]],
        remaining_dataset_indices: list[int],
        make_iter: Callable[[int], Iterator[Any] | None],
        worker_targets: list[int],
        segment_remaining: dict[int, int],
        offload_cursors: dict[int, _OffloadCursor] | None,
    ) -> np.ndarray:
        """Admit lazy residences with target/output-length compensation.

        ``T`` is the fixed worker-local effective epoch output-row quota after
        selection (including splits), process/worker sharding and resampling.
        Inner drop-last excludes its final incomplete batch from ``T``, never
        from the raw cursor/index plan. ``B`` is the next residence's actual
        output rows, including tail merging. ``u`` is the next yield's rows:
        one sample, a full inner batch, or its non-dropped short tail.

        Admission uses ``T/B`` because one entry produces ``B`` rows; active
        selection uses ``T/u`` because one service draw produces ``u`` rows.
        The two divisions compensate different selection units. An unfinished
        source becomes eligible again only after its previous reader closes;
        it can then win admission even while other sources remain pending.

        In a non-exhausting closed-cycle model with fixed targets, residence
        lengths and positive yield-size cycles, these paired weights have
        stationary row shares ``T/sum(T)``. An initially empty pool, finite
        quotas, exhaustion and changing tails do not satisfy that model, so
        individual prefixes need not match those shares. Without offload,
        ``B=T`` gives uniform admission and no short-window mixing guarantee.

        Mutates the active/pending lists and iteration-local output counters.
        Zero-output streams exhaust normally, preserving initialization checks
        and cleanup errors. Readers open only when advanced.
        """
        assert self._total_indices_length is not None
        batch = self.batch_loader_kwargs
        batch_size = batch.batch_size if batch is not None else 1

        def output_rows(rows: int) -> int:
            return (
                rows // batch_size * batch_size
                if batch is not None and batch.drop_last
                else rows
            )

        targets = [output_rows(rows) for rows in worker_targets]
        while (
            len(cur_dataset_iters) < self._max_dataset_concurrency
            and remaining_dataset_indices
        ):
            residence_rows = [
                output_rows(
                    offload_cursors[idx].next_segment_rows
                    if offload_cursors is not None
                    else worker_targets[idx]
                )
                for idx in remaining_dataset_indices
            ]
            # Zero output has no sampling weight, but must still run the
            # original lazy initialization/EOF validation before discarding it.
            if self._shuffle.shuffle and all(residence_rows):
                admission = np.array(
                    [
                        targets[idx] / rows
                        for idx, rows in zip(
                            remaining_dataset_indices,
                            residence_rows,
                            strict=True,
                        )
                    ],
                    dtype=np.float64,
                )
                selected = self._sample_dataset_index(
                    admission / admission.sum()
                )
            else:
                selected = (
                    residence_rows.index(0)
                    if self._shuffle.shuffle and 0 in residence_rows
                    else 0
                )
            idx = remaining_dataset_indices.pop(selected)
            rows = residence_rows[selected]
            if self._total_indices_length[idx] <= 0:
                continue
            dataset_iter = make_iter(idx)
            if dataset_iter is None:
                continue
            if rows == 0:
                try:
                    try:
                        next(dataset_iter)
                    except StopIteration:
                        pass
                    else:
                        raise RuntimeError(
                            "Zero-output dataset item unexpectedly "
                            "yielded data."
                        )
                except BaseException as exc:
                    close_iterators_best_effort(
                        [dataset_iter], primary_exc=exc
                    )
                    raise
                close_iterators_best_effort([dataset_iter])
                continue
            cur_dataset_iters.append((idx, dataset_iter))
            segment_remaining[idx] = rows
        weights = np.array(
            [
                targets[idx] / min(batch_size, segment_remaining[idx])
                for idx, _ in cur_dataset_iters
            ],
            dtype=np.float64,
        )
        if len(weights):
            weights /= weights.sum()
        return weights


@dataclass(slots=True)
class _OffloadCursor:
    """Read finite segments of one item in one worker/mixture iteration.

    Construction and ``iter_segment`` calls are lazy: only advancing the
    returned generator opens a reader and prepares the full logical plan.
    Between segments this owner retains only paths, scalars and the reservoir
    RNG. Readers, tables and prefetch buffers belong to the segment generator.
    Shared base files outlive this cursor; the mixture owns private-directory
    cleanup after normal exhaustion. Reading configuration stays fixed within
    one mixture iteration.
    """

    cache_root: str
    plan_path: str
    total_rows: int
    segment_rows: int
    cursor: int = 0
    initialized: bool = False
    prefetch_generator: torch.Generator | np.random.Generator | None = None

    @property
    def exhausted(self) -> bool:
        """Whether every allocated row has completed successful cleanup."""
        return self.cursor >= self.total_rows

    @property
    def next_segment_rows(self) -> int:
        """Predict raw residency rows without opening or advancing."""
        remaining = self.total_rows - self.cursor
        rows = min(self.segment_rows, remaining)
        if remaining - rows < self.segment_rows:
            rows = remaining
        return rows

    def iter_segment(
        self,
        data_item: DatasetItem[Any],
        *,
        shard_kwargs: ShardConfig,
        shuffle: ShuffleConfig,
        generator: torch.Generator | np.random.Generator | None,
        batch_loader_kwargs: BatchLoaderConfig | None,
        resample_ratio: float,
        expected_rows: int,
    ) -> Iterator[Any]:
        """Read the next segment and commit only after EOF and cleanup.

        The first advance validates item metadata and persists the complete
        worker-local index stream through the existing iterable preparation
        path. Reload opens only the raw reader; stored row anchors already
        include selection, sharding and sampling. Non-final segments contain
        whole batches, with a short remainder merged into the last segment.

        This generator owns its reader and child iterator. Normal exhaustion
        closes both before advancing the cursor. Early close, read failure or
        cleanup failure leaves the cursor uncommitted and propagates through
        the existing cleanup helper; the mixture then terminates the run.

        Args:
            data_item (DatasetItem): Source configuration, without mutating it.
            shard_kwargs (ShardConfig): Original physical shard policy.
            shuffle (ShuffleConfig): Existing index and prefetch policy.
            generator (torch.Generator | np.random.Generator | None): Existing
                mixture RNG, consumed only when first preparing this item.
            batch_loader_kwargs (BatchLoaderConfig | None): Existing batching
                and collation policy, or sample-level reading when ``None``.
            resample_ratio (float): Original logical row multiplier.
            expected_rows (int): Global item target before worker allocation.

        Yields:
            Any: Samples or dataset-side batches from this finite segment.

        Raises:
            RuntimeError: If source metadata or persisted plan length differs
                from its allocated row target.
        """
        dataset: Any = None
        dataset_iter: Iterator[Any] | None = None
        primary_exc: BaseException | None = None
        try:
            if not self.initialized:
                dataset_with_indices = self._open_base(data_item, shard_kwargs)
                dataset = dataset_with_indices.dataset
                iterable_dataset = dataset_with_indices.to_iterable_dataset(
                    shuffle=shuffle,
                    shard_kwargs=shard_kwargs,
                    generator=generator,
                    batch_loader_kwargs=batch_loader_kwargs,
                    resample_ratio=resample_ratio,
                )
                _validate_dataset_item_rows(
                    iterable_dataset.total_iterator_length, expected_rows
                )
                stream, prefetch_generator = (
                    iterable_dataset._prepare_worker_index_stream()
                )
                written_rows = IndiceTable.save_index_stream(
                    self.plan_path, stream
                )
                if written_rows != self.total_rows:
                    raise RuntimeError(
                        "Worker-local index plan length disagrees with "
                        "the allocated row target."
                    )
                self.prefetch_generator = prefetch_generator
                self.initialized = True
            else:
                # Stored indices already contain original selected row anchors.
                dataset = _fresh_config(data_item)._create_dataset()
                iterable_dataset = IterableWithLenDataset(
                    dataset=dataset,
                    indices=IndiceTable([]),
                    shuffle=shuffle,
                    batch_loader_kwargs=batch_loader_kwargs,
                )
            end = self.cursor + self.next_segment_rows
            plan = IndiceTable(self.plan_path)
            segment = plan.take(slice(self.cursor, end))
            dataset_iter = iterable_dataset._iter_prepared_indices(
                segment, self.prefetch_generator
            )
            batch_size = (
                batch_loader_kwargs.batch_size
                if batch_loader_kwargs is not None
                else 1
            )
            expected_yields = _get_batch_num(
                batch_size=batch_size,
                num_samples=end - self.cursor,
                drop_last=(
                    batch_loader_kwargs.drop_last
                    if batch_loader_kwargs is not None
                    else False
                ),
            )
            count = 0
            for item in dataset_iter:
                count += 1
                if count > expected_yields:
                    raise RuntimeError(
                        "Offload segment exceeded its output quota."
                    )
                yield item
            if count != expected_yields:
                raise RuntimeError(
                    "Offload segment exhausted before its output quota."
                )
        except BaseException as exc:
            primary_exc = exc
            raise
        finally:
            close_iterators_best_effort(
                [dataset_iter, dataset], primary_exc=primary_exc
            )
        # Only normal exhaustion followed by successful cleanup reaches here.
        self.cursor = end

    def _open_base(
        self, item: DatasetItem[Any], shard_kwargs: ShardConfig
    ) -> DatasetWithIndices[Any]:
        """Open the selected shared base, publishing it once per node/key.

        Source/selection inputs must remain immutable for this cache lifetime.
        A fresh config isolates subclass selection caches. The first writer
        reuses its raw reader for the first segment; cache hits open only the
        raw reader and never regenerate the full base table.
        """
        source = _fresh_config(item).replace(
            num_shards=1, shard_id=0, name=None
        )
        source_key = source.get_index_cache_key()
        if not isinstance(source_key, str):
            raise TypeError(
                "DatasetItem.get_index_cache_key() must return a string."
            )
        identity = json.dumps(
            {
                "source": source_key,
                "working_directory": os.getcwd(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        key = hashlib.sha256(identity.encode()).hexdigest()
        base_dir = os.path.join(self.cache_root, "base")
        os.makedirs(base_dir, exist_ok=True)
        path = os.path.join(base_dir, f"{key}.arrow")
        view: DatasetWithIndices[Any] | None = None
        try:
            if not os.path.exists(path):
                with FileLock(path + ".lock"):
                    if not os.path.exists(path):
                        view = source.create_dataset(shard_kwargs)
                        fd, pending = tempfile.mkstemp(
                            prefix=key + "-", suffix=".tmp", dir=base_dir
                        )
                        os.close(fd)
                        try:
                            view.indices.save_to_file(pending)
                            check = IndiceTable(pending)
                            if len(check) != len(view.indices):
                                raise RuntimeError(
                                    "Published base index count is incomplete."
                                )
                            del check
                            os.replace(pending, path)
                        finally:
                            if os.path.exists(pending):
                                os.unlink(pending)
            if view is None:
                base_indices = IndiceTable(path)
                view = DatasetWithIndices(
                    source._create_dataset(), base_indices
                )
            else:
                view.indices = IndiceTable(path)
            if item.is_sharded:
                view = view.shard(
                    num_shards=item.num_shards,
                    index=item.shard_id,
                    **shard_kwargs.to_dict(),
                )
            return view
        except BaseException as exc:
            if view is not None:
                close_iterators_best_effort([view.dataset], primary_exc=exc)
            raise


def _validate_dataset_item_rows(actual_rows: int, expected_rows: int) -> None:
    """Check logical source counts before worker allocation or sample reads.

    Ordinary reads and initial offload preparation enforce the same item
    metadata contract. Both counts include resampling and exclude worker
    allocation; a mismatch raises before consuming any source rows.
    """
    if actual_rows != expected_rows:
        detail = (
            "produced no rows"
            if actual_rows == 0 and expected_rows > 0
            else f"produced {actual_rows} rows"
        )
        raise RuntimeError(
            "DatasetItem row metadata does not match the "
            f"materialized dataset: expected {expected_rows} "
            f"rows but {detail}."
        )


def _fresh_config(config: Config) -> Any:
    """Rebuild declared fields without inheriting runtime private caches."""
    fields = config.to_dict(include_config_type=True)
    # The concrete outer type is already known, including function-local
    # DatasetItem subclasses; only nested polymorphic fields need lookup.
    fields.pop("__config_type__", None)
    return type(config).from_dict(fields)


@dataclass(slots=True)
class _PathTreeNode:
    """Store one generic display-path tree node with ordered item indices."""

    children: dict[str, _PathTreeNode] = field(default_factory=dict)
    item_indices: list[int] = field(default_factory=list)
    first_item_index: int | None = None


def _build_path_tree(
    item_names: Iterable[str | None],
) -> tuple[_PathTreeNode, dict[int, str]]:
    """Build a tolerant display tree for arbitrary slash-separated names."""
    root = _PathTreeNode()
    root_item_labels: dict[int, str] = {}
    for item_index, item_name in enumerate(item_names):
        if item_name is None:
            root.item_indices.append(item_index)
            root_item_labels[item_index] = f"(unnamed #{item_index})"
            continue
        if not isinstance(item_name, str):
            root.item_indices.append(item_index)
            root_item_labels[item_index] = (
                f"(invalid name #{item_index}: {type(item_name).__name__})"
            )
            continue

        node = root
        for segment in item_name.split("/"):
            child = node.children.get(segment)
            if child is None:
                child = _PathTreeNode(first_item_index=item_index)
                node.children[segment] = child
            node = child
        node.item_indices.append(item_index)
    return root, root_item_labels


def _format_path_tree(
    root: _PathTreeNode,
    root_item_labels: dict[int, str],
) -> list[tuple[str, tuple[int, ...], int | None]]:
    """Format a path tree into compact branch labels and item-index groups.

    Each returned tuple contains the tree label, all item indices summarized by
    that row, and the single leaf item index when the row is a concrete item.
    The helper preserves source order, compresses path nodes with one child and
    no direct item, and makes overlapping or duplicate names explicit through
    ``(self)`` rows. It does not know dataset metrics or resampling semantics.
    """

    def node_item_indices(node: _PathTreeNode) -> tuple[int, ...]:
        return tuple(
            sorted(
                (
                    *node.item_indices,
                    *(
                        item_index
                        for child in node.children.values()
                        for item_index in node_item_indices(child)
                    ),
                )
            )
        )

    rows: list[tuple[str, tuple[int, ...], int | None]] = []

    def append_leaf(label: str, item_index: int) -> None:
        rows.append((label, (item_index,), item_index))

    def append_node(
        node: _PathTreeNode,
        path_segments: tuple[str, ...],
        prefix: str,
        is_last: bool,
    ) -> None:
        while not node.item_indices and len(node.children) == 1:
            segment, node = next(iter(node.children.items()))
            path_segments = (*path_segments, segment)

        connector = "└─ " if is_last else "├─ "
        label = (
            prefix
            + connector
            + "/".join(
                _format_path_segment(segment) for segment in path_segments
            )
        )
        if len(node.item_indices) == 1 and not node.children:
            append_leaf(label, node.item_indices[0])
            return

        rows.append((label, node_item_indices(node), None))
        child_prefix = prefix + ("   " if is_last else "│  ")
        direct_item_positions = {
            item_index: position
            for position, item_index in enumerate(node.item_indices)
        }
        child_nodes_by_first_item_index = {
            child.first_item_index: (segment, child)
            for segment, child in node.children.items()
        }
        entry_indices = sorted(
            (*direct_item_positions, *child_nodes_by_first_item_index)
        )
        for position, item_index in enumerate(entry_indices):
            child_is_last = position == len(entry_indices) - 1
            if item_index in direct_item_positions:
                direct_position = direct_item_positions[item_index]
                direct_label = (
                    "(self)"
                    if len(node.item_indices) == 1
                    else f"(self #{direct_position})"
                )
                append_leaf(
                    child_prefix
                    + ("└─ " if child_is_last else "├─ ")
                    + direct_label,
                    item_index,
                )
                continue

            segment, child = child_nodes_by_first_item_index[item_index]
            append_node(child, (segment,), child_prefix, child_is_last)

    root_entries = sorted(
        (*root.item_indices, *root.children.values()),
        key=lambda entry: (
            entry if isinstance(entry, int) else entry.first_item_index
        ),
    )
    root_children_by_node_id = {
        id(child): segment for segment, child in root.children.items()
    }
    for entry in root_entries:
        if isinstance(entry, int):
            append_leaf(f"├─ {root_item_labels[entry]}", entry)
            continue
        append_node(
            entry,
            (root_children_by_node_id[id(entry)],),
            prefix="",
            is_last=False,
        )
    return rows


def _format_path_segment(segment: str) -> str:
    """Render one arbitrary path segment without terminal control effects."""
    if not segment:
        return "(empty)"

    escaped_chars: list[str] = []
    for char in segment:
        if char == "\\":
            escaped_chars.append("\\\\")
            continue
        if char == "\n":
            escaped_chars.append("\\n")
            continue
        if char == "\r":
            escaped_chars.append("\\r")
            continue
        if char == "\t":
            escaped_chars.append("\\t")
            continue
        if not unicodedata.category(char).startswith("C"):
            escaped_chars.append(char)
            continue

        codepoint = ord(char)
        if codepoint <= 0xFF:
            escaped_chars.append(f"\\x{codepoint:02x}")
        elif codepoint <= 0xFFFF:
            escaped_chars.append(f"\\u{codepoint:04x}")
        else:
            escaped_chars.append(f"\\U{codepoint:08x}")
    return "".join(escaped_chars)


@dataclass(slots=True)
class _ResamplePathNode:
    """Store one private path-trie node during ratio resolution."""

    children: dict[str, _ResamplePathNode] = field(default_factory=dict)
    item_index: int | None = None
    first_item_name: str | None = None
    resample_ratio: float | None = None
    rule_name: str | None = None


def _resolve_named_resample_ratios(
    *,
    dataset_items: list[DatasetItemNode],
    resample_ratios: dict[str, float],
) -> list[float]:
    """Resolve path-prefix rules into the existing per-item ratio sequence.

    The temporary trie validates that dataset item names are unique leaves and
    propagates the deepest matching rule to each leaf. It is discarded before
    iteration, so sharding and DataLoader cloning continue to use the resolved
    list of ratios.

    Args:
        dataset_items: Items whose names define the leaf set.
        resample_ratios: Positive row multipliers keyed by logical paths.

    Returns:
        One normalized ratio per item, in ``dataset_items`` order.

    Raises:
        TypeError: If a rule ratio is not numeric.
        ValueError: If a path is invalid, item paths overlap, a rule does not
            match an item, or a ratio is not finite and positive.
    """
    root = _ResamplePathNode()
    for item_index, item in enumerate(dataset_items):
        item_name = item.name
        path_segments = _split_resample_path(
            item_name,
            name=f"dataset_items[{item_index}].name",
        )
        node = root
        for segment in path_segments:
            if node.item_index is not None:
                raise ValueError(
                    "Dataset item names must be unique leaf paths, but "
                    f"{node.first_item_name!r} is an ancestor of "
                    f"{item_name!r}."
                )
            node = node.children.setdefault(segment, _ResamplePathNode())
            if node.first_item_name is None:
                node.first_item_name = item_name

        if node.item_index is not None:
            raise ValueError(
                "Dataset item names must be unique, but got "
                f"{item_name!r} more than once."
            )
        if node.children:
            raise ValueError(
                "Dataset item names must be unique leaf paths, but "
                f"{item_name!r} is an ancestor of "
                f"{node.first_item_name!r}."
            )
        node.item_index = item_index

    for rule_name, ratio in resample_ratios.items():
        path_segments = _split_resample_path(
            rule_name,
            name="resample_ratios key",
        )
        node = root
        for segment in path_segments:
            node = node.children.setdefault(segment, _ResamplePathNode())
        node.resample_ratio = _normalize_resample_ratio(
            ratio,
            name=f"resample_ratios[{rule_name!r}]",
        )
        node.rule_name = rule_name

    ratio_values = [1.0] * len(dataset_items)

    def resolve_node(
        node: _ResamplePathNode,
        inherited_ratio: float | None,
    ) -> bool:
        effective_ratio = (
            node.resample_ratio
            if node.resample_ratio is not None
            else inherited_ratio
        )
        has_item = node.item_index is not None
        if node.item_index is not None:
            ratio_values[node.item_index] = (
                effective_ratio if effective_ratio is not None else 1.0
            )
        for child in node.children.values():
            has_item = resolve_node(child, effective_ratio) or has_item
        if node.rule_name is not None and not has_item:
            raise ValueError(
                "resample_ratios key does not match any dataset item name: "
                f"{node.rule_name!r}."
            )
        return has_item

    resolve_node(root, inherited_ratio=None)
    return ratio_values


def _split_resample_path(value: Any, *, name: str) -> tuple[str, ...]:
    """Validate and split one lowercase logical dataset path."""
    if not isinstance(value, str):
        raise ValueError(
            f"{name} must be a lowercase slash-separated path, got {value!r}."
        )
    path_segments = tuple(value.split("/"))
    if not value or not all(
        _DATASET_ITEM_NAME_SEGMENT_PATTERN.fullmatch(segment)
        for segment in path_segments
    ):
        raise ValueError(
            f"{name} must be a lowercase slash-separated path, got {value!r}."
        )
    return path_segments


def _normalize_resample_ratio(value: float, *, name: str) -> float:
    """Normalize one caller-provided resampling ratio.

    Args:
        value (float): Caller-provided ratio to normalize. Booleans and text
            are rejected.
        name (str): Parameter label used in validation errors.

    Returns:
        float: A finite, strictly positive ratio.

    Raises:
        TypeError: If ``value`` is not numeric.
        ValueError: If the normalized ratio is not finite and positive.
    """
    if isinstance(value, (bool, str, bytes)):
        raise TypeError(f"{name} must be a float, but got {value!r}.")
    try:
        ratio = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a float, but got {value!r}.") from exc
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError(
            f"{name} must be finite and positive, but got {ratio!r}."
        )
    return ratio


def _distribute_rows_to_workers(
    base_rows: int,
    target_rows: int,
    num_workers: int,
) -> list[int]:
    """Allocate an exact resampled target across physical worker shards.

    The allocation mirrors the quotient-and-remainder worker sharding used by
    :class:`IndiceTableSampler`, then applies a largest-remainder allocation.
    The returned targets therefore sum exactly to ``target_rows``, and workers
    with empty physical shards receive no logical rows.

    Args:
        base_rows (int): Physical rows before worker sharding.
        target_rows (int): Logical rows required after resampling.
        num_workers (int): Requested worker count. Values below two use one
            logical worker.

    Returns:
        list[int]: Per-worker logical row targets in worker-ID order.
    """
    if num_workers <= 1:
        return [target_rows]
    if base_rows == 0:
        return [0] * num_workers

    rows_per_worker = base_rows // num_workers
    residual_rows = base_rows % num_workers
    base_rows_by_worker = [
        rows_per_worker + (worker_id < residual_rows)
        for worker_id in range(num_workers)
    ]
    target_numerators = [
        target_rows * worker_rows for worker_rows in base_rows_by_worker
    ]
    targets = [numerator // base_rows for numerator in target_numerators]
    remaining = target_rows - sum(targets)
    remainder_order = sorted(
        range(num_workers),
        key=lambda worker_id: (
            -(target_numerators[worker_id] % base_rows),
            worker_id,
        ),
    )
    for worker_id in remainder_order[:remaining]:
        targets[worker_id] += 1
    return targets


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


if not TYPE_CHECKING:
    _IterableWithLenDataset = IterableWithLenDataset
    _DictIterableDataset = DictIterableDataset

    class IterableWithLenDataset(
        _IterableWithLenDataset[DatasetType], HFIterableDataset
    ):
        def __init__(self, *args, **kwargs):
            _IterableWithLenDataset.__init__(self, *args, **kwargs)
            self._epoch = 0

    class DictIterableDataset(_DictIterableDataset, HFIterableDataset):
        def __init__(self, *args, **kwargs):
            _DictIterableDataset.__init__(self, *args, **kwargs)
            self._epoch = 0
