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

import gc
import multiprocessing as mp
import os
import threading
import time
import weakref
from typing import Any, cast

import pytest
import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration
from robo_orchard_core.utils.config import ClassType
from torch.utils.data import (
    Dataset,
    IterableDataset as TorchIterableDataset,
)

from robo_orchard_lab.dataset.robot import (
    BatchLoaderConfig,
    DataLoader,
    DatasetItem,
    IterableDatasetMixin,
    IterableWithLenDataset,
    ShardConfig,
    ShuffleConfig,
)
from robo_orchard_lab.dataset.robot.dataset_ex import (
    _DEFAULT_VIRTUAL_GETITEMS_BATCH_SIZE,
    _PREFETCH_CLOSE_JOIN_TIMEOUT_SEC,
    DictIterableDataset,
    _close_dataloader_iterator,
    _create_prefetch_iterator,
)
from robo_orchard_lab.dataset.sampler import ShardStrategy
from robo_orchard_lab.utils.accelerate import (
    configure_data_loader_for_accelerate,
)


class ArrayDataset(Dataset):
    def __init__(self, data: list):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class BatchedArrayDataset(ArrayDataset):
    def __init__(self, data: list):
        super().__init__(data)
        self.getitem_calls: list[int] = []
        self.getitems_calls: list[list[int]] = []

    def __getitem__(self, idx):
        self.getitem_calls.append(idx)
        return super().__getitem__(idx)

    def __getitems__(self, indices: list[int]) -> list:
        self.getitems_calls.append(list(indices))
        return [self.data[idx] for idx in indices]


class ArrayIterableDataset(TorchIterableDataset):
    def __init__(self, data: list[int]):
        self.data = data

    def __iter__(self):
        yield from self.data

    def __len__(self):
        return len(self.data)


class ArrayDatasetItem(DatasetItem[ArrayDataset]):
    class_type: ClassType[ArrayDataset] = ArrayDataset

    data: list

    def get_dataset_row_num(self) -> int:
        return len(self.data)

    def _create_dataset(self) -> ArrayDataset:
        return ArrayDataset(self.data)


def _get_dataloader_multiprocessing_context(
    num_workers: int,
) -> str | None:
    """Return the multiprocessing context for DataLoader tests.

    Default to a non-fork context because running the full dataset suite after
    importing many native extensions can make `fork`-based DataLoader workers
    unstable. Allow an explicit environment override for local speed tests.
    """

    if num_workers <= 0:
        return None

    start_methods = mp.get_all_start_methods()
    override = os.environ.get(
        "ROBO_ORCHARD_TEST_DATALOADER_MP_CONTEXT",
    )
    if override:
        if override not in start_methods:
            raise ValueError(
                "Unsupported multiprocessing context "
                f"{override!r}. Available contexts: {start_methods}."
            )
        return override

    if "forkserver" in start_methods:
        return "forkserver"
    if "spawn" in start_methods:
        return "spawn"
    if "fork" in start_methods:
        return "fork"
    return None


@pytest.fixture()
def dummy_array_dataset():
    return ArrayDataset(data=list(range(0, 10)))


class TestIterableDatasetMixin:
    def _check_dataloader_total_batch_consistency(
        self,
        dataloader: DataLoader,
        dataset: IterableDatasetMixin,
        batch_size: int,
        drop_last: bool,
    ):
        total_batches = 0
        for _ in dataloader:
            total_batches += 1

        calculated_batches = dataset.get_total_batch_num(
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=dataloader.num_workers,
        )
        assert total_batches == calculated_batches, (
            f"Total batches from dataloader ({total_batches}) does not match "
            f"calculated total batches ({calculated_batches})"
        )

    def _check_dataloader_item_consistency(
        self,
        dataloader: DataLoader,
        dataset: IterableDatasetMixin,
        need_sort: bool,
    ):
        dataloader_items = []
        for batch in dataloader:
            dataloader_items.extend(batch)

        dataset_items = []

        for item in dataset:
            if isinstance(item, list):
                dataset_items.extend(item)
            elif isinstance(item, torch.Tensor):
                dataset_items.extend(item.tolist())
            else:
                dataset_items.append(item)

        assert len(dataloader_items) == len(dataset_items), (
            f"Total items from dataloader ({len(dataloader_items)}) "
            f"does not match total items from dataset ({len(dataset_items)})"
        )
        # sort both lists before comparison, since dataloader may shuffle
        # the data
        if need_sort:
            dataloader_items.sort()
            dataset_items.sort()
        assert dataloader_items == dataset_items, (
            f"Items from dataloader do not match items from dataset.\n"
            f"Dataloader items: {dataloader_items}\n"
            f"Dataset items: {dataset_items}"
        )


class TestNonIterableDatasetMixinDataLoader:
    def test_map_dataset_accepts_shuffle_config(self):
        dataset = ArrayDataset(data=list(range(10)))

        with pytest.warns(UserWarning) as warning_records:
            dataloader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=ShuffleConfig(shuffle=True, chunk_size=4),
                num_workers=0,
            )

        flattened_items: list[int] = []
        for batch in dataloader:
            flattened_items.extend(cast(torch.Tensor, batch).tolist())

        assert sorted(flattened_items) == dataset.data
        assert any(
            "ShuffleConfig.chunk_size" in str(record.message)
            for record in warning_records
        )

    def test_iterable_dataset_accepts_shuffle_config_without_error(self):
        dataset = ArrayIterableDataset(data=list(range(10)))

        with pytest.warns(UserWarning) as warning_records:
            dataloader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=ShuffleConfig(shuffle=True, chunk_size=4),
                num_workers=0,
            )

        flattened_items: list[int] = []
        for batch in dataloader:
            flattened_items.extend(cast(torch.Tensor, batch).tolist())

        assert flattened_items == dataset.data
        warning_messages = [str(record.message) for record in warning_records]
        assert any(
            "ShuffleConfig.chunk_size" in message
            for message in warning_messages
        )
        assert any(
            "Resetting `shuffle=False`" in message
            for message in warning_messages
        )


class TestIterableWithLenDataset(TestIterableDatasetMixin):
    @pytest.fixture(params=["dummy_array_dataset"])
    def total_batch_consistency_test_dataset(self, request):
        return request.getfixturevalue(request.param)

    @pytest.mark.parametrize(
        "batch_size, num_workers, drop_last",
        [
            (3, 0, False),
            (4, 0, False),
            (6, 0, False),
            (3, 0, True),
            (4, 0, True),
            (6, 0, True),
            (3, 3, False),
            (4, 3, False),
            (6, 3, False),
            (3, 3, True),
            (4, 3, True),
            (6, 3, True),
        ],
    )
    def test_total_batch_consistency(
        self,
        total_batch_consistency_test_dataset: Dataset,
        batch_size: int,
        num_workers: int,
        drop_last: bool,
    ):
        dataset = IterableWithLenDataset(total_batch_consistency_test_dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=drop_last,
            multiprocessing_context=_get_dataloader_multiprocessing_context(
                num_workers
            ),
        )
        self._check_dataloader_total_batch_consistency(
            dataloader=dataloader,
            dataset=dataset,
            batch_size=batch_size,
            drop_last=drop_last,
        )

        # check batched reader
        dataset = IterableWithLenDataset(
            total_batch_consistency_test_dataset,
            batch_loader_kwargs=BatchLoaderConfig(
                batch_size=batch_size,
                drop_last=drop_last,
            ),
        )

        dataloader = DataLoader(
            dataset,
            num_workers=num_workers,
            multiprocessing_context=_get_dataloader_multiprocessing_context(
                num_workers
            ),
        )
        self._check_dataloader_total_batch_consistency(
            dataloader=dataloader,
            dataset=dataset,
            batch_size=batch_size,
            drop_last=drop_last,
        )

    def test_dataloader_item_consistency(self, dummy_array_dataset: Dataset):
        dataset = IterableWithLenDataset(dummy_array_dataset)
        batch_size = 3
        num_workers = 0
        drop_last = False
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=drop_last,
            multiprocessing_context=_get_dataloader_multiprocessing_context(
                num_workers
            ),
        )
        self._check_dataloader_item_consistency(
            dataloader=dataloader,
            dataset=dataset,
            need_sort=False,
        )

        # check batched reader
        dataset = IterableWithLenDataset(
            dummy_array_dataset,
            batch_loader_kwargs=BatchLoaderConfig(
                batch_size=batch_size,
                drop_last=drop_last,
            ),
        )

        dataloader = DataLoader(
            dataset,
            num_workers=num_workers,
            multiprocessing_context=_get_dataloader_multiprocessing_context(
                num_workers
            ),
        )
        self._check_dataloader_item_consistency(
            dataloader=dataloader,
            dataset=dataset,
            need_sort=False,
        )

    def test_unbatched_iterable_len(self, dummy_array_dataset: ArrayDataset):
        dataset = IterableWithLenDataset(dummy_array_dataset)

        dataloader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=0,
        )

        dataloader_items = list(dataloader)

        assert dataloader_items == dummy_array_dataset.data
        assert len(dataloader) == len(dummy_array_dataset)
        assert len(dataloader_items) == len(dataloader)

    def test_iterable_with_len_self_batched_overrides_dataset_config(
        self, dummy_array_dataset: Dataset
    ):
        dataset = IterableWithLenDataset(
            dummy_array_dataset,
            batch_loader_kwargs=BatchLoaderConfig(
                batch_size=4,
                drop_last=False,
            ),
        )

        dataloader = DataLoader(
            dataset,
            batch_size=3,
            drop_last=True,
            num_workers=0,
        )

        assert dataloader.dataset is not dataset
        assert dataset.batch_loader_kwargs is not None
        assert isinstance(dataloader.dataset, IterableWithLenDataset)
        assert dataloader.dataset.batch_loader_kwargs is not None
        assert dataset.batch_loader_kwargs.batch_size == 4
        assert dataset.batch_loader_kwargs.drop_last is False
        assert dataloader.dataset.batch_loader_kwargs.batch_size == 3
        assert dataloader.dataset.batch_loader_kwargs.drop_last is True
        assert len(list(dataloader)) == len(dataloader)

    def test_use_dataset_side_batching_supports_iterable_with_len(
        self, dummy_array_dataset: Dataset
    ):
        dataset = IterableWithLenDataset(dummy_array_dataset)

        dataloader = DataLoader(
            dataset,
            batch_size=3,
            drop_last=False,
            num_workers=0,
            use_dataset_side_batching=True,
        )

        assert dataloader.dataset is not dataset
        assert dataset.batch_loader_kwargs is None
        assert isinstance(dataloader.dataset, IterableWithLenDataset)
        assert dataloader.dataset.batch_loader_kwargs is not None
        assert dataloader.dataset.batch_loader_kwargs.batch_size == 3
        assert dataloader.dataset.batch_loader_kwargs.drop_last is False
        assert len(list(dataloader)) == len(dataloader)

    def test_use_dataset_side_batching_aligns_user_collate_fn(
        self, dummy_array_dataset: Dataset
    ):
        dataset = IterableWithLenDataset(dummy_array_dataset)
        collate_inputs: list[list[int]] = []

        def collate_fn(batch: list[int]) -> dict[str, Any]:
            batch_list = list(batch)
            collate_inputs.append(batch_list)
            return {"values": batch_list, "size": len(batch_list)}

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            drop_last=True,
            num_workers=0,
            collate_fn=collate_fn,
            use_dataset_side_batching=True,
        )

        assert list(dataloader) == [
            {"values": [0, 1, 2, 3], "size": 4},
            {"values": [4, 5, 6, 7], "size": 4},
        ]
        assert collate_inputs == [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ]

    def test_iterable_with_len_accepts_dataloader_shuffle_true(
        self, dummy_array_dataset: Dataset
    ):
        dataset = IterableWithLenDataset(dummy_array_dataset, shuffle=False)

        dataloader = DataLoader(
            dataset,
            batch_size=3,
            shuffle=True,
            num_workers=0,
        )

        assert isinstance(dataloader.dataset, IterableWithLenDataset)
        assert dataloader.dataset is not dataset
        assert dataset._shuffle_config.shuffle is False
        assert dataloader.dataset._shuffle_config.shuffle is True
        assert len(list(dataloader)) == len(dataloader)

    def test_iterable_with_len_self_batched_aligns_shuffle_config(
        self, dummy_array_dataset: Dataset
    ):
        dataset = IterableWithLenDataset(
            dummy_array_dataset,
            shuffle=ShuffleConfig(
                shuffle=False,
                chunk_size=4,
                prefetch_factor=3,
            ),
            batch_loader_kwargs=BatchLoaderConfig(batch_size=2),
        )

        dataloader = DataLoader(
            dataset,
            batch_size=3,
            shuffle=True,
            num_workers=0,
        )

        assert isinstance(dataloader.dataset, IterableWithLenDataset)
        assert dataloader.dataset is not dataset
        assert dataset._shuffle_config.shuffle is False
        assert dataset._shuffle_config.chunk_size == 4
        assert dataset._shuffle_config.prefetch_factor == 3
        assert dataloader.dataset._shuffle_config.shuffle is True
        assert dataloader.dataset._shuffle_config.chunk_size == 4
        assert dataloader.dataset._shuffle_config.prefetch_factor == 3
        assert len(list(dataloader)) == len(dataloader)

    def test_iterable_with_len_uses_virtual_batch_getitems(self):
        dataset = BatchedArrayDataset(data=list(range(10)))

        iterable_dataset = IterableWithLenDataset(dataset)

        assert list(iterable_dataset) == list(range(10))
        assert dataset.getitem_calls == []
        assert dataset.getitems_calls
        assert sum(len(batch) for batch in dataset.getitems_calls) == len(
            dataset
        )
        assert all(
            len(batch) <= _DEFAULT_VIRTUAL_GETITEMS_BATCH_SIZE
            for batch in dataset.getitems_calls
        )

    def test_iterable_with_len_shard_uses_accelerate_signature(
        self, dummy_array_dataset: ArrayDataset
    ):
        dataset = IterableWithLenDataset(dummy_array_dataset)

        sharded_dataset = dataset.shard(num_shards=3, index=1)

        assert list(sharded_dataset) == [4, 5, 6]

    def test_iterable_with_len_shard_preserves_shard_kwargs(
        self, dummy_array_dataset: ArrayDataset
    ):
        dataset = IterableWithLenDataset(
            dummy_array_dataset,
            shard_kwargs=ShardConfig(
                contiguous=False,
                shard_strategy="pad_last",
            ),
        )

        sharded_dataset = dataset.shard(num_shards=3, index=1)

        assert list(sharded_dataset) == [1, 4, 7]
        assert sharded_dataset.shard_kwargs.contiguous is False
        assert sharded_dataset.shard_kwargs.shard_strategy == "pad_last"


class TestDictIterableDataset(TestIterableDatasetMixin):
    @pytest.fixture()
    def dummy_dataset_items(self):
        return [
            ArrayDatasetItem(
                data=list(range(0, 10)),
            ),
            ArrayDatasetItem(data=list(range(100, 110))),
        ]

    def test_dataloader_item_consistency(
        self, dummy_dataset_items: list[DatasetItem]
    ):
        dataset = DictIterableDataset(dummy_dataset_items)
        batch_size = 3
        num_workers = 0
        drop_last = False
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=drop_last,
            multiprocessing_context=_get_dataloader_multiprocessing_context(
                num_workers
            ),
        )
        self._check_dataloader_item_consistency(
            dataloader=dataloader,
            dataset=dataset,
            need_sort=False,
        )

        # check batched reader
        dataset = DictIterableDataset(
            dummy_dataset_items,
            batch_loader_kwargs=BatchLoaderConfig(
                batch_size=batch_size,
                drop_last=drop_last,
            ),
        )

        dataloader = DataLoader(
            dataset,
            num_workers=num_workers,
            multiprocessing_context=_get_dataloader_multiprocessing_context(
                num_workers
            ),
        )
        self._check_dataloader_item_consistency(
            dataloader=dataloader,
            dataset=dataset,
            need_sort=False,
        )

    def test_repr_does_not_require_hf_internal_info(
        self, dummy_dataset_items: list[DatasetItem]
    ):
        dataset = DictIterableDataset(dummy_dataset_items)

        repr_text = repr(dataset)

        assert "DictIterableDataset(" in repr_text
        assert "dataset_items=2" in repr_text

    def test_use_dataset_side_batching_option(
        self, dummy_dataset_items: list[DatasetItem]
    ):
        dataset = DictIterableDataset(dummy_dataset_items)

        dataloader = DataLoader(
            dataset,
            batch_size=3,
            num_workers=0,
            drop_last=False,
            use_dataset_side_batching=True,
        )

        assert isinstance(dataloader.dataset, DictIterableDataset)
        assert dataloader.dataset is not dataset
        assert dataloader.dataset.batch_loader_kwargs is not None
        assert dataloader.dataset.batch_loader_kwargs.batch_size == 3
        assert dataloader.dataset.batch_loader_kwargs.drop_last is False

        batches = []
        for batch in dataloader:
            if isinstance(batch, torch.Tensor):
                batches.append(batch.tolist())
            else:
                batches.append(list(batch))

        assert [9, 100, 101] not in batches
        for batch in batches:
            assert batch, "Batch should not be empty."
            assert all(item < 100 for item in batch) or all(
                item >= 100 for item in batch
            )

    def test_close_closes_active_child_iterators(
        self,
        dummy_dataset_items: list[DatasetItem],
        monkeypatch,
    ):
        class _CloseTrackingIterator:
            def __init__(self, items: list[int]) -> None:
                self._items = iter(items)
                self.closed = False

            def __iter__(self):
                return self

            def __next__(self) -> int:
                return next(self._items)

            def close(self) -> None:
                self.closed = True

        dataset = DictIterableDataset(dummy_dataset_items, shuffle=False)
        dataset._total_indices_length = [2, 2]
        child_iters = [
            _CloseTrackingIterator([0, 1]),
            _CloseTrackingIterator([100, 101]),
        ]

        def fake_prepare_dataset_for_iter(
            cur_dataset_iters: list[tuple[int, Any]],
            remaining_dataset_indices: list[int],
        ) -> list[float]:
            del remaining_dataset_indices
            if not cur_dataset_iters:
                cur_dataset_iters.extend(
                    [(0, child_iters[0]), (1, child_iters[1])]
                )
            return [0.5, 0.5]

        monkeypatch.setattr(
            dataset,
            "_prepare_dataset_for_iter",
            fake_prepare_dataset_for_iter,
        )

        dataset_iter = iter(dataset)

        assert next(dataset_iter) == 0

        cast(Any, dataset_iter).close()

        assert child_iters[0].closed is True
        assert child_iters[1].closed is True

    def test_use_dataset_side_batching_aligns_batch_loader_kwargs(
        self, dummy_dataset_items: list[DatasetItem]
    ):
        dataset = DictIterableDataset(dummy_dataset_items)

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            num_workers=0,
            drop_last=True,
            use_dataset_side_batching=True,
        )

        assert dataset.batch_loader_kwargs is None
        assert isinstance(dataloader.dataset, DictIterableDataset)
        assert dataloader.dataset is not dataset
        assert dataloader.dataset.batch_loader_kwargs is not None
        assert dataloader.dataset.batch_loader_kwargs.batch_size == 4
        assert dataloader.dataset.batch_loader_kwargs.drop_last is True

        batches = []
        for batch in dataloader:
            if isinstance(batch, torch.Tensor):
                batches.append(batch.tolist())
            else:
                batches.append(list(batch))

        assert len(batches) == len(dataloader)
        assert all(len(batch) == 4 for batch in batches)
        for batch in batches:
            assert all(item < 100 for item in batch) or all(
                item >= 100 for item in batch
            )

    def test_use_dataset_side_batching_accepts_shuffle_config(
        self, dummy_dataset_items: list[DatasetItem]
    ):
        dataset = DictIterableDataset(
            dummy_dataset_items,
            shuffle=ShuffleConfig(
                shuffle=False,
                chunk_size=5,
                prefetch_factor=2,
            ),
        )

        dataloader = DataLoader(
            dataset,
            batch_size=3,
            num_workers=0,
            use_dataset_side_batching=True,
            shuffle=ShuffleConfig(
                shuffle=True,
                chunk_size=3,
                prefetch_factor=4,
            ),
        )

        assert isinstance(dataloader.dataset, DictIterableDataset)
        assert dataloader.dataset is not dataset
        assert dataset._shuffle.shuffle is False
        assert dataset._shuffle.chunk_size == 5
        assert dataset._shuffle.prefetch_factor == 2
        assert dataloader.dataset._shuffle.shuffle is True
        assert dataloader.dataset._shuffle.chunk_size == 3
        assert dataloader.dataset._shuffle.prefetch_factor == 4

        batches = []
        for batch in dataloader:
            if isinstance(batch, torch.Tensor):
                batches.append(batch.tolist())
            else:
                batches.append(list(batch))

        assert batches
        for batch in batches:
            assert all(item < 100 for item in batch) or all(
                item >= 100 for item in batch
            )

    def test_dict_iterable_accepts_dataloader_shuffle_true(
        self, dummy_dataset_items: list[DatasetItem]
    ):
        dataset = DictIterableDataset(dummy_dataset_items, shuffle=False)

        dataloader = DataLoader(
            dataset,
            batch_size=3,
            shuffle=True,
            num_workers=0,
        )

        assert isinstance(dataloader.dataset, DictIterableDataset)
        assert dataloader.dataset is not dataset
        assert dataset._shuffle.shuffle is False
        assert dataloader.dataset._shuffle.shuffle is True
        assert len(list(dataloader)) == len(dataloader)

    @pytest.mark.parametrize(
        "batch_size, num_workers, drop_last",
        [
            (3, 0, False),
            (4, 0, False),
            (6, 0, False),
            (3, 0, True),
            (4, 0, True),
            (6, 0, True),
            (3, 3, False),
            (4, 3, False),
            (6, 3, False),
            (3, 3, True),
            (4, 3, True),
            (6, 3, True),
        ],
    )
    def test_total_batch_consistency(
        self,
        dummy_dataset_items: list[DatasetItem],
        batch_size: int,
        num_workers: int,
        drop_last: bool,
    ):
        dataset = DictIterableDataset(dummy_dataset_items)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=drop_last,
            multiprocessing_context=_get_dataloader_multiprocessing_context(
                num_workers
            ),
        )
        self._check_dataloader_total_batch_consistency(
            dataloader=dataloader,
            dataset=dataset,
            batch_size=batch_size,
            drop_last=drop_last,
        )

        # check batched reader
        dataset = DictIterableDataset(
            dummy_dataset_items,
            batch_loader_kwargs=BatchLoaderConfig(
                batch_size=batch_size,
                drop_last=drop_last,
            ),
        )

        dataloader = DataLoader(
            dataset,
            num_workers=num_workers,
            multiprocessing_context=_get_dataloader_multiprocessing_context(
                num_workers
            ),
        )
        self._check_dataloader_total_batch_consistency(
            dataloader=dataloader,
            dataset=dataset,
            batch_size=batch_size,
            drop_last=drop_last,
        )

    def test_dict_iterable_self_batched_overrides_dataset_config(
        self, dummy_dataset_items: list[DatasetItem]
    ):
        dataset = DictIterableDataset(
            dummy_dataset_items,
            batch_loader_kwargs=BatchLoaderConfig(
                batch_size=4,
                drop_last=False,
            ),
        )

        dataloader = DataLoader(
            dataset,
            batch_size=3,
            drop_last=True,
            num_workers=0,
        )

        assert dataloader.dataset is not dataset
        assert dataset.batch_loader_kwargs is not None
        assert isinstance(dataloader.dataset, DictIterableDataset)
        assert dataloader.dataset.batch_loader_kwargs is not None
        assert dataset.batch_loader_kwargs.batch_size == 4
        assert dataset.batch_loader_kwargs.drop_last is False
        assert dataloader.dataset.batch_loader_kwargs.batch_size == 3
        assert dataloader.dataset.batch_loader_kwargs.drop_last is True
        assert len(list(dataloader)) == len(dataloader)

    def test_dataset_item_shard_uses_accelerate_signature(self):
        item = ArrayDatasetItem(data=list(range(10)))

        sharded_item = item.shard(num_shards=3, index=1)

        assert sharded_item.num_shards == 3
        assert sharded_item.shard_id == 1

    def test_dict_iterable_shard_uses_accelerate_signature(
        self, dummy_dataset_items: list[DatasetItem]
    ):
        dataset = DictIterableDataset(dummy_dataset_items)

        sharded_dataset = dataset.shard(num_shards=2, index=1)

        assert isinstance(sharded_dataset, DictIterableDataset)
        assert len(sharded_dataset.dataset_items) == len(dataset.dataset_items)
        assert [item.shard_id for item in sharded_dataset.dataset_items] == [
            1,
            1,
        ]
        assert [item.num_shards for item in sharded_dataset.dataset_items] == [
            2,
            2,
        ]

    @staticmethod
    def _collect_rank_local_dataset_trace(
        rank: int,
        shard_strategy: ShardStrategy,
    ) -> tuple[list[str], list[int]]:
        generator = torch.Generator()
        generator.manual_seed(123)
        dataset = DictIterableDataset(
            [
                ArrayDatasetItem(data=list(range(21))),
                ArrayDatasetItem(data=list(range(100, 105))),
            ],
            shuffle=True,
            generator=generator,
            shard_kwargs=ShardConfig(
                contiguous=True,
                shard_strategy=shard_strategy,
            ),
        ).shard(num_shards=2, index=rank)

        trace = ["A" if item < 100 else "B" for item in dataset]
        assert dataset._total_indices_length is not None
        return trace, dataset._total_indices_length

    def test_rank_local_schedule_can_diverge_without_even_shards(self):
        # Multi-process iterable training relies on each rank observing the
        # same per-dataset mixture weights. Without an even shard strategy,
        # local shard lengths can differ and the rank-local dataset schedule
        # will drift even when the same RNG seed is used.
        rank0_trace, rank0_lengths = self._collect_rank_local_dataset_trace(
            rank=0,
            shard_strategy=None,
        )
        rank1_trace, rank1_lengths = self._collect_rank_local_dataset_trace(
            rank=1,
            shard_strategy=None,
        )

        assert rank0_lengths != rank1_lengths
        assert rank0_trace != rank1_trace, (
            "Expected the rank-local dataset schedule to diverge when "
            "sharded lengths are uneven, but both ranks produced the same "
            f"trace: {''.join(rank0_trace)}"
        )

    def test_rank_local_schedule_stays_aligned_with_pad_last(self):
        rank0_trace, rank0_lengths = self._collect_rank_local_dataset_trace(
            rank=0,
            shard_strategy="pad_last",
        )
        rank1_trace, rank1_lengths = self._collect_rank_local_dataset_trace(
            rank=1,
            shard_strategy="pad_last",
        )

        assert rank0_lengths == rank1_lengths
        assert rank0_trace == rank1_trace


class TestCreatePrefetchIterator:
    def _count_prefetch_threads(self) -> int:
        return sum(
            thread.name == "dataset-prefetch-producer"
            for thread in threading.enumerate()
        )

    def test_waits_for_full_prefetch_window_before_shuffle(self):
        # When shuffling is enabled, the iterator should wait for a full
        # prefetch window before yielding any item so that the shuffle has a
        # meaningful sample pool.
        allow_second_item = threading.Event()
        first_item_ready = threading.Event()
        yielded_items = []

        def blocking_iter():
            yield 0
            if not allow_second_item.wait(timeout=1):
                raise TimeoutError("Timed out waiting for the second item.")
            yield 1

        generator = torch.Generator()
        generator.manual_seed(0)
        prefetch_iter = _create_prefetch_iterator(
            iter(blocking_iter()),
            prefetch_size=2,
            shuffle=True,
            generator=generator,
        )

        def consume_first_item():
            yielded_items.append(next(prefetch_iter))
            first_item_ready.set()

        consumer_thread = threading.Thread(target=consume_first_item)
        consumer_thread.start()

        # The consumer must still be blocked because only one sample is ready
        # and `prefetch_size=2` has not been satisfied yet.
        assert not first_item_ready.wait(timeout=0.1)

        # Once the second sample arrives, the first shuffled item can be
        # yielded and the remaining item should still be preserved.
        allow_second_item.set()
        assert first_item_ready.wait(timeout=1)
        assert yielded_items[0] in {0, 1}

        consumer_thread.join(timeout=1)
        remaining_items = list(prefetch_iter)
        assert sorted(yielded_items + remaining_items) == [0, 1]

    def test_prefetches_next_window_while_consuming_current_window(self):
        # After the first window is handed to the consumer, the producer should
        # immediately start filling the next window instead of waiting for the
        # current one to be fully drained.
        attempted_refill = threading.Event()
        allow_tail_items = threading.Event()

        def blocking_iter():
            yield 0
            yield 1
            attempted_refill.set()
            if not allow_tail_items.wait(timeout=1):
                raise TimeoutError("Timed out waiting for tail items.")
            yield 2
            yield 3

        prefetch_iter = _create_prefetch_iterator(
            iter(blocking_iter()),
            prefetch_size=2,
            shuffle=False,
            generator=None,
        )

        assert next(prefetch_iter) == 0
        # Reaching this event means the producer has already advanced to the
        # next refill stage while the current window is still being consumed.
        assert attempted_refill.wait(timeout=1)

        allow_tail_items.set()
        remaining_items = list(prefetch_iter)
        assert remaining_items == [1, 2, 3]

    def test_close_stops_prefetch_thread_waiting_on_full_queue(self):
        # Closing the generator early should notify the producer and let the
        # background thread exit instead of remaining blocked on a full queue.
        allow_refill = threading.Event()

        def blocking_iter():
            yield 0
            yield 1
            if not allow_refill.wait(timeout=1):
                raise TimeoutError("Timed out waiting for refill release.")
            yield 2

        baseline_threads = self._count_prefetch_threads()
        prefetch_iter = _create_prefetch_iterator(
            iter(blocking_iter()),
            prefetch_size=2,
            shuffle=False,
            generator=None,
        )

        assert next(prefetch_iter) == 0
        cast(Any, prefetch_iter).close()

        # Release the upstream iterator and wait briefly for the prefetch
        # thread count to return to its baseline.
        allow_refill.set()
        for _ in range(20):
            if self._count_prefetch_threads() == baseline_threads:
                break
            threading.Event().wait(0.05)

        assert self._count_prefetch_threads() == baseline_threads

    def test_close_waits_for_short_inflight_prefetch_item(self):
        tail_item_started = threading.Event()

        def slow_tail_iter():
            yield 0
            yield 1
            tail_item_started.set()
            time.sleep(0.1)
            yield 2

        baseline_threads = self._count_prefetch_threads()
        prefetch_iter = _create_prefetch_iterator(
            iter(slow_tail_iter()),
            prefetch_size=2,
            shuffle=False,
            generator=None,
        )

        assert next(prefetch_iter) == 0
        assert tail_item_started.wait(timeout=1)

        cast(Any, prefetch_iter).close()

        assert self._count_prefetch_threads() == baseline_threads

    def test_close_returns_when_inflight_prefetch_item_blocks(self):
        tail_item_started = threading.Event()
        allow_tail_item = threading.Event()

        def blocked_tail_iter():
            yield 0
            yield 1
            tail_item_started.set()
            allow_tail_item.wait()
            yield 2

        baseline_threads = self._count_prefetch_threads()
        prefetch_iter = _create_prefetch_iterator(
            iter(blocked_tail_iter()),
            prefetch_size=2,
            shuffle=False,
            generator=None,
        )

        assert next(prefetch_iter) == 0
        assert tail_item_started.wait(timeout=1)

        start = time.monotonic()
        with pytest.warns(
            UserWarning,
            match="Prefetch producer thread did not exit within",
        ):
            cast(Any, prefetch_iter).close()
        elapsed = time.monotonic() - start

        assert elapsed < _PREFETCH_CLOSE_JOIN_TIMEOUT_SEC + 0.5

        allow_tail_item.set()
        for _ in range(40):
            if self._count_prefetch_threads() == baseline_threads:
                break
            threading.Event().wait(0.05)

        assert self._count_prefetch_threads() == baseline_threads

    def test_close_closes_nested_dataloader_prefetch_iterator(self):
        baseline_threads = self._count_prefetch_threads()
        dataset = DictIterableDataset(
            [
                ArrayDatasetItem(data=list(range(16))),
                ArrayDatasetItem(data=list(range(100, 116))),
            ],
            shuffle=ShuffleConfig(
                shuffle=True,
                chunk_size=4,
                prefetch_factor=2,
            ),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            shuffle=dataset._shuffle,
        )
        accelerator = Accelerator(
            dataloader_config=DataLoaderConfiguration(
                dispatch_batches=False,
                split_batches=False,
                even_batches=False,
            )
        )
        configure_data_loader_for_accelerate(
            accelerator=accelerator,
            data_loader=dataloader,
        )
        dataloader = accelerator.prepare(dataloader)
        baseline_ref_count = sum(
            ref is not None
            for ref in accelerator.gradient_state.dataloader_references
        )

        dataloader_iter = iter(dataloader)
        next(dataloader_iter)

        _close_dataloader_iterator(dataloader_iter)
        assert accelerator.gradient_state.in_dataloader
        assert (
            sum(
                ref is not None
                for ref in accelerator.gradient_state.dataloader_references
            )
            > baseline_ref_count
        )
        accelerator.end_training()

        for _ in range(40):
            if self._count_prefetch_threads() == baseline_threads:
                break
            threading.Event().wait(0.05)

        assert self._count_prefetch_threads() == baseline_threads

    def test_close_keeps_persistent_workers_reusable(self):
        num_workers = 1
        dataloader = DataLoader(
            ArrayDataset(data=list(range(16))),
            batch_size=4,
            num_workers=num_workers,
            persistent_workers=True,
            multiprocessing_context=_get_dataloader_multiprocessing_context(
                num_workers
            ),
        )

        iterator = iter(dataloader)
        first_batch = cast(torch.Tensor, next(iterator))

        _close_dataloader_iterator(cast(Any, iterator))

        iterator = iter(dataloader)
        second_batch = cast(torch.Tensor, next(iterator))

        assert first_batch.tolist() == [0, 1, 2, 3]
        assert second_batch.tolist() == [0, 1, 2, 3]

    def test_raises_producer_error_without_draining_ready_queue(self):
        # If the producer fails after the current window has been handed off,
        # the consumer should observe that failure on the next pull instead of
        # silently draining the rest of the ready queue first.
        fail_now = threading.Event()
        failure_branch_reached = threading.Event()

        def failing_iter():
            yield 0
            yield 1
            if not fail_now.wait(timeout=1):
                raise TimeoutError("Timed out waiting to trigger failure.")
            failure_branch_reached.set()
            raise RuntimeError("producer failed")

        prefetch_iter = _create_prefetch_iterator(
            iter(failing_iter()),
            prefetch_size=2,
            shuffle=False,
            generator=None,
        )

        assert next(prefetch_iter) == 0
        fail_now.set()
        # Wait until the producer has actually reached the failing branch so
        # the next `next()` call deterministically checks error propagation.
        assert failure_branch_reached.wait(timeout=1)
        with pytest.raises(RuntimeError, match="producer failed"):
            next(prefetch_iter)


class TestDataLoaderEarlyBreakCleanup:
    def _count_prefetch_threads(self) -> int:
        return sum(
            thread.name == "dataset-prefetch-producer"
            for thread in threading.enumerate()
        )

    def _iterate_with_early_break(
        self,
        dataloader: DataLoader,
        max_batches: int,
    ) -> list[Any]:
        dataloader_iter = iter(dataloader)
        collected_batches = []
        try:
            for batch_idx, batch in enumerate(dataloader_iter):
                collected_batches.append(batch)
                if batch_idx + 1 >= max_batches:
                    break
        finally:
            _close_dataloader_iterator(cast(Any, dataloader_iter))

        return collected_batches

    def _wait_for_prefetch_threads(self, expected_count: int) -> None:
        for _ in range(40):
            if self._count_prefetch_threads() == expected_count:
                return
            time.sleep(0.05)

        assert self._count_prefetch_threads() == expected_count

    def _active_child_count(self) -> int:
        return len(mp.active_children())

    def _wait_for_active_child_count(self, expected_count: int) -> None:
        for _ in range(60):
            if self._active_child_count() == expected_count:
                return
            time.sleep(0.05)

        assert self._active_child_count() == expected_count

    def _build_dataset(
        self,
        dataset_kind: str,
    ) -> IterableDatasetMixin:
        shuffle = ShuffleConfig(
            shuffle=True,
            chunk_size=4,
            prefetch_factor=2,
        )
        if dataset_kind == "iterable":
            return IterableWithLenDataset(
                ArrayDataset(data=list(range(32))),
                shuffle=shuffle,
            )

        return DictIterableDataset(
            [
                ArrayDatasetItem(data=list(range(6))),
                ArrayDatasetItem(data=list(range(100, 106))),
                ArrayDatasetItem(data=list(range(200, 206))),
            ],
            shuffle=shuffle,
            max_dataset_concurrency=2,
        )

    @pytest.mark.parametrize("use_dataset_side_batching", [False, True])
    def test_dict_iterable_repeated_early_break_releases_dynamic_sub_iterators(
        self,
        monkeypatch,
        use_dataset_side_batching: bool,
    ):
        tracked_state = {"created": 0, "finalized": 0}
        live_generators = weakref.WeakSet()
        original_iter = IterableWithLenDataset.__iter__

        def mark_finalized() -> None:
            tracked_state["finalized"] += 1

        def tracked_iter(self):
            generator = original_iter(self)
            tracked_state["created"] += 1
            live_generators.add(generator)
            weakref.finalize(generator, mark_finalized)
            return generator

        monkeypatch.setattr(
            IterableWithLenDataset,
            "__iter__",
            tracked_iter,
        )

        dataset = DictIterableDataset(
            [
                ArrayDatasetItem(data=list(range(3))),
                ArrayDatasetItem(data=list(range(100, 103))),
                ArrayDatasetItem(data=list(range(200, 203))),
            ],
            shuffle=False,
            max_dataset_concurrency=1,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=2 if use_dataset_side_batching else 1,
            num_workers=0,
            use_dataset_side_batching=use_dataset_side_batching,
        )

        for _ in range(5):
            batches = self._iterate_with_early_break(
                dataloader,
                max_batches=4,
            )
            assert batches
            gc.collect()
            assert len(live_generators) == 0
            assert tracked_state["finalized"] == tracked_state["created"]

        assert tracked_state["created"] >= 10

    @pytest.mark.parametrize("use_dataset_side_batching", [False, True])
    def test_repeated_early_break_keeps_historical_iterators_clean(
        self,
        use_dataset_side_batching: bool,
    ):
        baseline_threads = self._count_prefetch_threads()
        shuffle = ShuffleConfig(
            shuffle=True,
            chunk_size=4,
            prefetch_factor=2,
        )
        batch_size = 2 if use_dataset_side_batching else 1
        dataset = IterableWithLenDataset(
            ArrayDataset(data=list(range(32))),
            shuffle=shuffle,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=shuffle,
            use_dataset_side_batching=use_dataset_side_batching,
        )

        first_values = []
        for _ in range(6):
            batches = self._iterate_with_early_break(
                dataloader,
                max_batches=2,
            )
            assert batches
            first_batch = batches[0]
            if isinstance(first_batch, torch.Tensor):
                first_values.append(first_batch.tolist())
            else:
                first_values.append(list(first_batch))
            self._wait_for_prefetch_threads(baseline_threads)

        assert len(first_values) == 6

    @pytest.mark.parametrize("dataset_kind", ["iterable", "dict"])
    @pytest.mark.parametrize("use_dataset_side_batching", [False, True])
    @pytest.mark.parametrize(
        "num_workers,persistent_workers",
        [
            (0, False),
            (1, False),
            (1, True),
            (2, False),
            (2, True),
        ],
    )
    def test_repeated_early_break_keeps_dataloader_reusable(
        self,
        dataset_kind: str,
        use_dataset_side_batching: bool,
        num_workers: int,
        persistent_workers: bool,
    ):
        baseline_child_count = self._active_child_count()
        dataloader_kwargs = {
            "batch_size": 2 if use_dataset_side_batching else 1,
            "num_workers": num_workers,
            "use_dataset_side_batching": use_dataset_side_batching,
        }
        if num_workers > 0:
            dataloader_kwargs["persistent_workers"] = persistent_workers
            dataloader_kwargs["multiprocessing_context"] = (
                _get_dataloader_multiprocessing_context(num_workers)
            )

        dataloader = DataLoader(
            self._build_dataset(dataset_kind),
            **dataloader_kwargs,
        )
        expected_cycle_child_count = baseline_child_count
        if num_workers > 0 and persistent_workers:
            expected_cycle_child_count += num_workers

        for _ in range(4):
            batches = self._iterate_with_early_break(
                dataloader,
                max_batches=2,
            )
            assert batches
            self._wait_for_active_child_count(expected_cycle_child_count)

        del dataloader
        gc.collect()
        self._wait_for_active_child_count(baseline_child_count)
