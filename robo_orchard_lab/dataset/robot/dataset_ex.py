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
import inspect
import math
import unicodedata
import warnings
from abc import ABCMeta, abstractmethod
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
from pydantic import Field
from robo_orchard_core.utils.config import ClassType, Config
from robo_orchard_core.utils.logging import LoggerManager
from torch.utils.data import (
    DataLoader as TorchDataLoader,
    Dataset as TorchDataset,
    IterableDataset as TorchIterableDataset,
)
from typing_extensions import TypeVar

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
    "DatasetItem",
    "DictIterableDataset",
]


DatasetType = TypeVar("DatasetType", bound=TorchDataset)
_TORCH_DATALOADER_INIT_SIGNATURE = inspect.signature(TorchDataLoader.__init__)
_DEFAULT_VIRTUAL_GETITEMS_BATCH_SIZE = 32


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


def _batched_iterator_with_indices(
    dataset: TorchDataset,
    indice_iter: Iterable[int],
    batch_size: int = _DEFAULT_VIRTUAL_GETITEMS_BATCH_SIZE,
) -> Iterator[Any]:
    if not hasattr(dataset, "__getitems__"):
        for idx in indice_iter:
            yield dataset[idx]
        return

    batch_indices: list[int] = []
    for idx in indice_iter:
        batch_indices.append(int(idx))
        if len(batch_indices) >= batch_size:
            yield from dataset.__getitems__(batch_indices)  # type: ignore[attr-defined]
            batch_indices = []

    if batch_indices:
        yield from dataset.__getitems__(batch_indices)  # type: ignore[attr-defined]


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
        if not uses_dataset_batch_loader and not should_clone_for_shuffle:
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
                datasets=dataset.dataset_items,
                shuffle=aligned_shuffle_config,
                shard_kwargs=dataset.shard_kwargs,
                generator=dataset._generator,
                batch_loader_kwargs=aligned_batch_loader_kwargs,
                max_dataset_concurrency=dataset._max_dataset_concurrency,
                resample_ratios=dataset._resample_ratios,
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
        indices = self._resolve_indices(dataset, indices)
        self._shuffle_config = self._normalize_shuffle_config(shuffle)
        self._resample_ratio = _normalize_resample_ratio(
            resample_ratio,
            name="resample_ratio",
        )
        if self.resample_ratio != 1.0 and not self._shuffle_config.shuffle:
            raise ValueError(
                "resample_ratio values other than 1.0 require shuffle=True."
            )

        self.indice_sampler = self._create_indice_sampler(
            indices=indices,
            shuffle_config=self._shuffle_config,
            generator=generator,
        )

        # add to base classes but not inherit to avoid unnecessary methods.
        # prefer modifying class bases, but allow instance-level fallback
        # _add_hf_iterable_cls(self.__class__, instance=self)

        self._shard_kwargs = (
            shard_kwargs if shard_kwargs is not None else ShardConfig()
        )
        self._batch_loader_kwargs = self._normalize_batch_loader_kwargs(
            batch_loader_kwargs
        )

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

    def iter(self):
        """Iterate over the current dataset view.

        This method does not apply outer PyTorch worker sharding by itself;
        ``__iter__`` chooses the worker-local view first and then delegates
        here.

        Yields:
            Any: Individual samples or ready-made batches, depending on
            whether ``batch_loader_kwargs`` is configured.

        """
        yield from self._iter_with_indice_sampler(self.indice_sampler)

    def _iter_with_indice_sampler(
        self,
        indice_sampler: IndiceTableSampler,
    ) -> Iterator[Any]:
        """Iterate using the sampler selected for one logical stream."""

        if self.batch_loader_kwargs is None:
            logger.debug("Iterating without batch loader,...")
            yield from self._iter_indices(
                self._iter_resampled_indices(
                    indice_sampler,
                    self.total_iterator_length,
                )
            )
            return

        logger.debug(
            "Iterating with batch loader, shuffle: %s, batch loader: %s",
            self._shuffle_config,
            self.batch_loader_kwargs,
        )
        inner_loader = self._create_inner_batch_loader(indice_sampler)
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
        if _uses_sample_prefetch(
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
            self._torch_iter(indice_sampler),
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

    @staticmethod
    def _normalize_shuffle_config(
        shuffle: bool | ShuffleConfig,
    ) -> ShuffleConfig:
        if isinstance(shuffle, bool):
            return ShuffleConfig(shuffle=shuffle)
        return shuffle

    @staticmethod
    def _resolve_indices(
        dataset: DatasetType,
        indices: IndiceTable | ChunkedIndiceTable | None,
    ) -> IndiceTable | ChunkedIndiceTable:
        if indices is not None:
            return indices
        if isinstance(dataset, Sized):
            return IndiceTable(len(dataset))
        raise ValueError(
            "Dataset does not have a length, indices must be provided."
        )

    @staticmethod
    def _create_indice_sampler(
        indices: IndiceTable | ChunkedIndiceTable,
        shuffle_config: ShuffleConfig,
        generator: torch.Generator | np.random.Generator | None,
    ) -> IndiceTableSampler:
        return IndiceTableSampler(
            indices=indices,
            shuffle=shuffle_config.shuffle,
            generator=generator,
            shuffle_chunk_size=(
                shuffle_config.chunk_size
                if not isinstance(indices, ChunkedIndiceTable)
                else None
            ),
        )

    @staticmethod
    def _normalize_batch_loader_kwargs(
        batch_loader_kwargs: BatchLoaderConfig | dict | None,
    ) -> BatchLoaderConfig | None:
        if isinstance(batch_loader_kwargs, dict):
            return BatchLoaderConfig(**batch_loader_kwargs)
        return batch_loader_kwargs

    def _iter_indices(self, indice_iter: Iterable[int]) -> Iterator[Any]:
        """Yield samples for the provided indices.

        When the wrapped dataset implements ``__getitems__``, this helper uses
        small index batches to amortize indexing overhead while still exposing
        a sample-by-sample iterator to callers.
        """
        yield from _batched_iterator_with_indices(
            self.dataset,
            indice_iter,
        )

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

    def _create_inner_batch_loader(
        self,
        indice_sampler: IndiceTableSampler,
    ) -> TorchDataLoader:
        """Build the inner dataloader used for dataset-side batching.

        The inner loader always uses ``num_workers=0``. Worker/process sharding
        has already been decided by the surrounding ``IterableWithLenDataset``
        instance, so spawning another worker pool here would duplicate that
        logic and make nested batching much harder to reason about.
        """
        assert self.batch_loader_kwargs is not None
        return torch.utils.data.DataLoader(
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
            **self.batch_loader_kwargs.to_dict(),
        )

    def _torch_iter(
        self,
        indice_sampler: IndiceTableSampler | None = None,
    ) -> Iterator[Any]:
        """Iterate over the dataset and yield data samples.

        This method is designed to be compatible with PyTorch's DataLoader with
        multiple workers.

        In plain sample mode, worker sharding happens here before the logical
        target is allocated and sampled. In dataset-side batching mode, the
        outer worker initializer has already built a worker-local dataset view,
        so this method delegates to :meth:`iter` and lets its inner loader form
        batches from that view.
        """
        if indice_sampler is None:
            indice_sampler = self.indice_sampler
        if (
            self.batch_loader_kwargs is not None
            or not self._is_torch_multi_worker()
        ):
            yield from self._iter_with_indice_sampler(indice_sampler)
            return

        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is not None
        worker_sampler = self._get_multi_worker_sharded_indices(indice_sampler)
        target_rows = _distribute_rows_to_workers(
            base_rows=len(self.indice_sampler),
            target_rows=self.total_iterator_length,
            num_workers=worker_info.num_workers,
        )[worker_info.id]
        yield from self._iter_indices(
            self._iter_resampled_indices(worker_sampler, target_rows)
        )

    def _get_multi_worker_sharded_indices(
        self,
        indice_sampler: IndiceTableSampler,
    ) -> IndiceTableSampler:
        """Return the current PyTorch worker's physical index shard.

        This method slices ``indice_sampler`` directly instead of calling
        :meth:`shard`, which would create another iterable wrapper and risk
        recursive worker sharding.
        """
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is not None
        # do not call shard() here to avoid recursive sharding.
        return indice_sampler.shard(
            num_shards=worker_info.num_workers,
            shard_id=worker_info.id,
            contiguous=self.shard_kwargs.contiguous,
        )

    def _is_torch_multi_worker(self) -> bool:
        import torch.utils.data

        worker_info = torch.utils.data.get_worker_info()
        return worker_info is not None and worker_info.num_workers > 1


class DatasetItem(Config, Generic[DatasetType], metaclass=ABCMeta):
    """A configuration for creating a dataset.

    User should inherit this class and implement the `_create_dataset` method
    to create a dataset from the configuration, and implement the
    `get_dataset_row_num` method to return the number of rows in the dataset
    before sharding.

    ``name`` optionally provides a stable source identity for consumers such
    as :class:`DictIterableDataset` summaries. This class also includes the
    sharding information, and the `create_dataset`
    method will apply the sharding to the created dataset. This is useful when
    we want to create a sharded dataset directly from the configuration.
    """

    class_type: ClassType[DatasetType]

    name: str | None = Field(
        default=None,
        description="Optional stable name for this dataset item.",
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
                index=self.shard_id,
                **shard_config.to_dict(),
            )
        return ret

    @property
    def is_sharded(self) -> bool:
        return self.num_shards > 1

    def shard(self, num_shards: int, index: int) -> DatasetItem[DatasetType]:
        """Shard the dataset item by returning a new DatasetItem.

        The new DatasetItem will have the same configuration as the original
        one, but with the updated shard_id and num_shards. The new sharding
        information will be calculated by:
        - new_num_shards: self.num_shards * num_shards
        - new_shard_id: self.shard_id * num_shards + index

        Note that the sharding information is always calculated based on the
        original dataset.

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


class DictIterableDataset(TorchIterableDataset, IterableDatasetMixin):
    """Mix multiple dataset items with optional per-item resampling.

    This dataset will create a DatasetWithIndices for each DatasetItem, and
    iterate over the datasets in a weighted round-robin way. Resampling scales
    each item's sharded row count before scheduling: ratios below one truncate
    the sample stream, while ratios above one restart it for additional rows.
    Non-unit ratios require shuffling. In multi-worker loading, every child
    dataset is physically sharded first and resampled only within that shard.

    When dataset-side batching is configured, resampling still happens at the
    row level. Batching is applied once to the complete resampled row stream,
    so ``drop_last`` only affects its final incomplete batch.

    Item names for :meth:`summary` come from :attr:`DatasetItem.name`.
    Unnamed items use ``item_0``, ``item_1``, and so on.

    Args:
        datasets (Iterable[DatasetItem]): An iterable of DatasetItems to create
            the dataset from.
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
        max_dataset_concurrency (int, optional): Maximum number of dataset-item
            iterators kept active by the weighted scheduler. Defaults to 4.
        resample_ratios (float | Iterable[float] | None, optional): A finite,
            positive row-count multiplier applied to every dataset item, or
            one multiplier per item. Defaults to None, which uses 1.0 for each
            item. Values other than 1.0 require ``shuffle=True``. Ratios are
            fixed at construction time.

    Raises:
        TypeError: If ratios are not numeric.
        ValueError: If iterable argument lengths do not match ``datasets``, or
            a ratio is non-finite, not positive, or non-unit while shuffling is
            disabled.

    """  # noqa: E501

    dataset_items: list[DatasetItem]

    def __init__(
        self,
        datasets: Iterable[DatasetItem],
        shuffle: bool | ShuffleConfig = False,
        shard_kwargs: ShardConfig | None = None,
        generator: torch.Generator | np.random.Generator | None = None,
        batch_loader_kwargs: BatchLoaderConfig | dict | None = None,
        max_dataset_concurrency: int = 4,
        resample_ratios: float | Iterable[float] | None = None,
    ):
        # try to make this instance compatible with HF Iterable at class-level
        # or instance-level if class-level MRO change fails
        # _add_hf_iterable_cls(self.__class__, instance=self)
        self.dataset_items = list(datasets)

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

        if resample_ratios is None:
            ratio_values: list[Any] = [1.0] * len(self.dataset_items)
        elif isinstance(resample_ratios, Iterable) and not isinstance(
            resample_ratios, (str, bytes)
        ):
            try:
                ratio_values = list(resample_ratios)
            except TypeError as exc:
                raise TypeError(
                    "resample_ratios must be a float or an iterable of floats."
                ) from exc
        else:
            ratio_values = [resample_ratios] * len(self.dataset_items)

        if len(ratio_values) != len(self.dataset_items):
            raise ValueError(
                "resample_ratios must contain one value per dataset item, "
                f"but got {len(ratio_values)} values for "
                f"{len(self.dataset_items)} items."
            )
        self._resample_ratios = [
            _normalize_resample_ratio(
                ratio,
                name=f"resample_ratios[{index}]",
            )
            for index, ratio in enumerate(ratio_values)
        ]
        if (
            any(ratio != 1.0 for ratio in self._resample_ratios)
            and not self._shuffle.shuffle
        ):
            raise ValueError(
                "resample_ratios values other than 1.0 require shuffle=True."
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
        """Shard the dataset by sharding each dataset item."""
        sharded_items = [
            item.shard(num_shards=num_shards, index=index)
            for item in self.dataset_items
        ]
        return DictIterableDataset(
            datasets=sharded_items,
            shuffle=self._shuffle,
            generator=self._generator,
            batch_loader_kwargs=self.batch_loader_kwargs,
            max_dataset_concurrency=self._max_dataset_concurrency,
            shard_kwargs=self.shard_kwargs,
            resample_ratios=self._resample_ratios,
        )

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
            repr(item) for item in self.dataset_items
        )
        if dataset_items_repr:
            dataset_items_repr = f"[\n    {dataset_items_repr}\n  ]"
        else:
            dataset_items_repr = "[]"

        return (
            f"{self.__class__.__name__}("
            f"dataset_items={len(self.dataset_items)}, "
            f"items={dataset_items_repr}, "
            f"shuffle={self._shuffle.shuffle}, "
            f"batch_loader_kwargs={self.batch_loader_kwargs!r}, "
            f"max_dataset_concurrency={self._max_dataset_concurrency})"
        )

    def summary(self) -> str:
        """Return the current shard's row-level dataset mixture summary.

        ``sample_ratio`` uses resampled row counts, while ``frame_ratio`` and
        ``length`` use the real sharded row counts before resampling. These
        sharded lengths can be smaller than :attr:`total_dataset_length`.
        With dataset-side batching, row-level sample ratios can differ slightly
        from the observed batch ratio because each dataset item forms batches
        independently and may drop one final incomplete batch.

        Returns:
            str: A display-width-aligned table suitable for logs or consoles.

        Example:
            ``print(dataset.summary())`` renders a table like::

                                            name sample_ratio [frame_ratio] [length]
                ├---------------------------aaa:       25.00% [     50.00%] [    10]
                ├-------------------------数_据:       75.00% [     50.00%] [    10]
                ├-------------------------total:      100.00% [    100.00%] [    20]
        """  # noqa: E501

        def char_width(char: str) -> int:
            if unicodedata.combining(char):
                return 0
            return 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1

        def format_name(name: str, width: int) -> str:
            sanitized = "".join(
                "_" if unicodedata.category(char).startswith("C") else char
                for char in name
            )
            kept_reversed: list[str] = []
            display_width = 0
            for char in reversed(sanitized):
                next_width = char_width(char)
                if display_width + next_width > width:
                    break
                kept_reversed.append(char)
                display_width += next_width
            visible_name = "".join(reversed(kept_reversed))
            return "-" * (width - display_width) + visible_name

        _ = self.total_iterator_length
        assert self._total_indices_length is not None
        scaled_rows = self._total_indices_length
        base_rows = [
            item.get_sharded_row_num(shard_config=self.shard_kwargs)
            for item in self.dataset_items
        ]
        total_scaled_rows = sum(scaled_rows)
        total_base_rows = sum(base_rows)
        length_width = max(len("length"), len(str(total_base_rows)))
        name_width = 30
        sample_ratio_width = len("sample_ratio")
        frame_ratio_width = len("frame_ratio")
        lines = [
            f"{'name':>{name_width + 2}} "
            f"{'sample_ratio':>{sample_ratio_width}} "
            f"[{'frame_ratio':>{frame_ratio_width}}] "
            f"[{'length':>{length_width}}]"
        ]
        for index, (item, scaled_length, base_length) in enumerate(
            zip(
                self.dataset_items,
                scaled_rows,
                base_rows,
                strict=True,
            )
        ):
            name = item.name if item.name is not None else f"item_{index}"
            sample_ratio = (
                scaled_length / total_scaled_rows
                if total_scaled_rows > 0
                else 0.0
            )
            frame_ratio = (
                base_length / total_base_rows if total_base_rows > 0 else 0.0
            )
            lines.append(
                f"├{format_name(name, name_width)}: "
                f"{sample_ratio:>{sample_ratio_width}.2%} "
                f"[{frame_ratio:>{frame_ratio_width}.2%}] "
                f"[{base_length:>{length_width}}]"
            )

        total_sample_ratio = 1.0 if total_scaled_rows > 0 else 0.0
        total_frame_ratio = 1.0 if total_base_rows > 0 else 0.0
        lines.append(
            f"├{format_name('total', name_width)}: "
            f"{total_sample_ratio:>{sample_ratio_width}.2%} "
            f"[{total_frame_ratio:>{frame_ratio_width}.2%}] "
            f"[{total_base_rows:>{length_width}}]"
        )
        return "\n".join(lines)

    @property
    def total_dataset_length(self) -> int:
        """Return the physical row count before configured sharding."""
        if self._total_dataset_length is None:
            self._total_dataset_length = [
                item.get_dataset_row_num() for item in self.dataset_items
            ]
        return sum(self._total_dataset_length)

    @property
    def total_iterator_length(self) -> int:
        """Return the exact combined logical row count after resampling."""
        if self._total_indices_length is None:
            self._total_indices_length = [
                round(
                    item.get_sharded_row_num(shard_config=self.shard_kwargs)
                    * ratio
                )
                for item, ratio in zip(
                    self.dataset_items,
                    self._resample_ratios,
                    strict=True,
                )
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
            item.get_sharded_row_num(shard_config=self.shard_kwargs)
            for item in self.dataset_items
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
        """Yield the weighted mixture and close all materialized datasets.

        At most ``max_dataset_concurrency`` item iterators are active. Shuffled
        iteration selects among them using resampled row-count weights; ordered
        iteration drains each item in sequence. Exhausted, failed, and
        caller-abandoned streams all close their child iterators.

        Yields:
            Any: One sample or dataset-side batch from an active dataset item.
        """
        cur_dataset_iters: list[tuple[int, Iterator[Any]]] = []
        dataset_indices = list(
            IndiceTableSampler(
                len(self.dataset_items),
                shuffle=self._shuffle.shuffle,
                generator=self._generator,
            )
        )
        _ = self.total_iterator_length
        assert self._total_indices_length is not None
        weights = self._prepare_dataset_for_iter(
            cur_dataset_iters=cur_dataset_iters,
            remaining_dataset_indices=dataset_indices,
        )

        primary_exc: BaseException | None = None
        try:
            while len(cur_dataset_iters) > 0:
                if self._shuffle.shuffle:
                    if isinstance(self._generator, np.random.Generator):
                        selected_idx = self._generator.choice(
                            len(cur_dataset_iters), p=weights, replace=False
                        )
                    elif isinstance(self._generator, torch.Generator):
                        selected_idx = int(
                            torch.multinomial(
                                torch.tensor(weights),
                                1,
                                generator=self._generator,
                            ).item()
                        )
                    else:
                        raise ValueError(
                            "Generator must be either a torch.Generator or a "
                            "numpy.random.Generator."
                        )
                else:
                    selected_idx = 0
                _, iter_dataset = cur_dataset_iters[selected_idx]
                try:
                    item = next(iter_dataset)
                    yield item
                except StopIteration:
                    cur_dataset_iters.pop(selected_idx)
                    close_iterators_best_effort([iter_dataset])
                    weights = self._prepare_dataset_for_iter(
                        cur_dataset_iters=cur_dataset_iters,
                        remaining_dataset_indices=dataset_indices,
                    )
        except BaseException as exc:
            primary_exc = exc
            raise
        finally:
            close_iterators_best_effort(
                [iter_dataset for _, iter_dataset in cur_dataset_iters],
                primary_exc=primary_exc,
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
            data_item (DatasetItem[Any]): Lazy dataset configuration to
                materialize for this iterator.
            resample_ratio (float): Validated ratio forwarded to the child
                iterable.
            expected_rows (int): Logical rows derived from item metadata.

        Yields:
            Any: Samples or dataset-side batches from the child iterable.

        Raises:
            RuntimeError: If materialized row metadata disagrees with the
                configured item metadata.
        """

        dataset_with_indices = data_item.create_dataset(
            shard_config=self.shard_kwargs
        )
        dataset_iter: Iterator[Any] | None = None
        primary_exc: BaseException | None = None
        try:
            iterable_dataset = dataset_with_indices.to_iterable_dataset(
                shuffle=self._shuffle,
                shard_kwargs=self.shard_kwargs,
                generator=self._generator,
                batch_loader_kwargs=self.batch_loader_kwargs,
                resample_ratio=resample_ratio,
            )
            actual_rows = iterable_dataset.total_iterator_length
            if actual_rows != expected_rows:
                detail = (
                    "produced no rows"
                    if actual_rows == 0 and expected_rows > 0
                    else f"produced {actual_rows} rows"
                )
                raise RuntimeError(
                    "DatasetItem row metadata does not match the materialized "
                    f"dataset: expected {expected_rows} rows but {detail}."
                )
            dataset_iter = iter(iterable_dataset)
            yield from dataset_iter
        except BaseException as exc:
            primary_exc = exc
            raise
        finally:
            close_iterators_best_effort(
                [dataset_iter, dataset_with_indices.dataset],
                primary_exc=primary_exc,
            )

    def _prepare_dataset_for_iter(
        self,
        cur_dataset_iters: list[tuple[int, Iterator[Any]]],
        remaining_dataset_indices: list[int],
    ) -> np.ndarray:
        """Fill active iterator slots and return their sampling weights.

        This method mutates both input lists: zero-target items are discarded,
        and positive-target items are materialized until the concurrency limit
        is reached. Returned weights follow ``cur_dataset_iters`` order and use
        each item's global logical target.

        Args:
            cur_dataset_iters (list[tuple[int, Iterator[Any]]]): Active item
                indices and their owned iterators.
            remaining_dataset_indices (list[int]): Pending item indices,
                consumed from the front.

        Returns:
            np.ndarray: Normalized weights for active iterators, or an empty
                array when no positive-target items remain.
        """
        assert self._total_indices_length is not None
        while (
            len(cur_dataset_iters) < self._max_dataset_concurrency
            and len(remaining_dataset_indices) > 0
        ):
            idx = remaining_dataset_indices.pop(0)
            target_rows = self._total_indices_length[idx]
            if target_rows <= 0:
                continue
            data_item = self.dataset_items[idx]
            dataset_iter = self._iter_dataset_item(
                data_item,
                resample_ratio=self._resample_ratios[idx],
                expected_rows=target_rows,
            )
            cur_dataset_iters.append((idx, dataset_iter))
        weights = [
            self._total_indices_length[idx] for idx, _ in cur_dataset_iters
        ]
        weights = np.array(weights, dtype=np.float32)
        if len(weights) > 0:
            weights = weights / weights.sum()
        return weights


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
