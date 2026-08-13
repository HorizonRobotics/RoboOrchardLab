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
import pickle
import threading
import time
import weakref
from typing import Any, cast

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration
from robo_orchard_core.utils.config import ClassType
from torch.utils.data import (
    DataLoader as TorchDataLoader,
    Dataset,
    IterableDataset as TorchIterableDataset,
)

import robo_orchard_lab.dataset.robot._prefetch as prefetch_module
from robo_orchard_lab.dataset.robot import (
    BatchLoaderConfig,
    DataLoader,
    DatasetItem,
    DatasetWithIndices,
    IterableDatasetMixin,
    IterableWithLenDataset,
    ShardConfig,
    ShuffleConfig,
)
from robo_orchard_lab.dataset.robot._prefetch import (
    DataloaderCloseReason,
    _close_dataloader_iterator,
    _close_dataloader_owner_resources,
    close_dataloader_resources,
    create_prefetch_iterator,
)
from robo_orchard_lab.dataset.robot.dataset_ex import (
    _DEFAULT_VIRTUAL_GETITEMS_BATCH_SIZE,
    DictIterableDataset,
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


class DelayedBatchedArrayDataset(BatchedArrayDataset):
    def __init__(self, data: list, delay: float) -> None:
        super().__init__(data)
        self._delay = delay
        self._getitems_call_count = 0

    def __getitems__(self, indices: list[int]) -> list:
        if self._delay > 0 and self._getitems_call_count % 2 == 0:
            time.sleep(self._delay)
        self._getitems_call_count += 1
        return super().__getitems__(indices)


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
        "resample_ratio,expected_length",
        [
            (0.5, 5),
            (1.0, 10),
            (2.5, 25),
        ],
    )
    def test_resample_ratio_controls_exact_logical_row_count(
        self,
        dummy_array_dataset: ArrayDataset,
        resample_ratio: float,
        expected_length: int,
    ):
        dataset = DatasetWithIndices(dummy_array_dataset).to_iterable_dataset(
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
            resample_ratio=resample_ratio,
        )

        items = list(dataset)

        assert len(items) == dataset.total_iterator_length == expected_length
        if resample_ratio < 1:
            assert len(set(items)) == expected_length
        else:
            for cycle_start in range(0, expected_length - 9, 10):
                assert sorted(items[cycle_start : cycle_start + 10]) == list(
                    range(10)
                )

    @pytest.mark.parametrize("generator_kind", ["torch", "numpy"])
    def test_oversampling_prefetch_is_independent_of_source_latency(
        self,
        generator_kind: str,
    ):
        """Seeded source cycles and reservoir shuffle use separate streams."""

        def collect(delay: float) -> list[int]:
            generator: torch.Generator | np.random.Generator
            if generator_kind == "torch":
                generator = torch.Generator().manual_seed(7)
            else:
                generator = np.random.default_rng(7)
            dataset = DatasetWithIndices(
                DelayedBatchedArrayDataset(list(range(12)), delay)
            ).to_iterable_dataset(
                shuffle=ShuffleConfig(
                    shuffle=True,
                    chunk_size=2,
                    prefetch_factor=2,
                ),
                generator=generator,
                resample_ratio=10.0,
            )
            return list(dataset)

        fast_items = collect(delay=0.0)
        delayed_items = collect(delay=0.001)

        assert fast_items == delayed_items
        assert len(fast_items) == 120
        assert all(fast_items.count(value) == 10 for value in range(12))

    @pytest.mark.parametrize("generator_kind", ["torch", "numpy"])
    def test_size_one_prefetch_preserves_sampling_generator_stream(
        self,
        generator_kind: str,
    ):
        """Size-one prefetch is a pass-through without RNG splitting."""

        if generator_kind == "torch":
            prefetch_generator: torch.Generator | np.random.Generator = (
                torch.Generator().manual_seed(7)
            )
            direct_generator: torch.Generator | np.random.Generator = (
                torch.Generator().manual_seed(7)
            )
        else:
            prefetch_generator = np.random.default_rng(7)
            direct_generator = np.random.default_rng(7)

        shuffle_config = ShuffleConfig(
            shuffle=True,
            chunk_size=1,
            prefetch_factor=1,
        )
        prefetch_items = list(
            DatasetWithIndices(
                ArrayDataset(list(range(12)))
            ).to_iterable_dataset(
                shuffle=shuffle_config,
                generator=prefetch_generator,
                resample_ratio=2.0,
            )
        )
        direct_dataset = DatasetWithIndices(
            ArrayDataset(list(range(12)))
        ).to_iterable_dataset(
            shuffle=shuffle_config,
            generator=direct_generator,
            resample_ratio=2.0,
        )
        direct_items = list(
            direct_dataset._torch_iter(direct_dataset.indice_sampler)
        )

        assert prefetch_items == direct_items
        if isinstance(prefetch_generator, torch.Generator):
            assert isinstance(direct_generator, torch.Generator)
            assert torch.equal(
                prefetch_generator.get_state(),
                direct_generator.get_state(),
            )
        else:
            assert isinstance(direct_generator, np.random.Generator)
            assert (
                prefetch_generator.bit_generator.state
                == direct_generator.bit_generator.state
            )

    @pytest.mark.parametrize("prefetch_factor", [0, -1])
    def test_nonpositive_sample_prefetch_size_still_raises(
        self,
        dummy_array_dataset: ArrayDataset,
        prefetch_factor: int,
    ):
        dataset = DatasetWithIndices(dummy_array_dataset).to_iterable_dataset(
            shuffle=ShuffleConfig(
                shuffle=True,
                chunk_size=1,
                prefetch_factor=prefetch_factor,
            ),
        )

        with pytest.raises(
            ValueError,
            match="prefetch_size must be greater than 0",
        ):
            list(dataset)

    @pytest.mark.parametrize(
        "resample_ratio,exception",
        [
            (0.0, ValueError),
            (-1.0, ValueError),
            (float("nan"), ValueError),
            (float("inf"), ValueError),
            (True, TypeError),
            ("1.0", TypeError),
        ],
    )
    def test_resample_ratio_validates_public_contract(
        self,
        dummy_array_dataset: ArrayDataset,
        resample_ratio: Any,
        exception: type[Exception],
    ):
        with pytest.raises(exception):
            IterableWithLenDataset(
                dummy_array_dataset,
                shuffle=True,
                resample_ratio=resample_ratio,
            )

    @pytest.mark.parametrize("resample_ratio", [0.5, 2.0])
    def test_resample_ratio_requires_shuffle(
        self,
        dummy_array_dataset: ArrayDataset,
        resample_ratio: float,
    ):
        with pytest.raises(ValueError, match="require shuffle=True"):
            IterableWithLenDataset(
                dummy_array_dataset,
                shuffle=False,
                resample_ratio=resample_ratio,
            )

    def test_resample_ratio_batches_the_complete_stream_once(self):
        dataset = IterableWithLenDataset(
            ArrayDataset([0, 1]),
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
            resample_ratio=2.0,
            batch_loader_kwargs=BatchLoaderConfig(
                batch_size=4,
                drop_last=True,
            ),
        )

        batches = list(dataset)

        assert len(batches) == 1
        assert sorted(batches[0].tolist()) == [0, 0, 1, 1]

    def test_resample_ratio_survives_views_and_dataloader_clone(
        self,
        dummy_array_dataset: ArrayDataset,
    ):
        dataset = IterableWithLenDataset(
            dummy_array_dataset,
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
            resample_ratio=2.0,
        )
        sharded = dataset.shard(num_shards=2, index=0)
        taken = dataset.take(slice(0, 3))
        dataloader = DataLoader(
            dataset,
            batch_size=3,
            use_dataset_side_batching=True,
        )

        assert sharded.resample_ratio == 2.0
        assert sharded.total_iterator_length == 10
        assert taken.resample_ratio == 2.0
        assert taken.total_iterator_length == 6
        assert isinstance(dataloader.dataset, IterableWithLenDataset)
        assert dataloader.dataset.resample_ratio == 2.0

    @pytest.mark.parametrize("use_dataset_side_batching", [False, True])
    def test_worker_sharding_precedes_oversampling_for_tiny_dataset(
        self,
        use_dataset_side_batching: bool,
    ):
        num_workers = 2
        dataset = IterableWithLenDataset(
            ArrayDataset([7]),
            shuffle=True,
            resample_ratio=2.0,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=num_workers,
            use_dataset_side_batching=use_dataset_side_batching,
            multiprocessing_context=_get_dataloader_multiprocessing_context(
                num_workers
            ),
        )

        batches = list(dataloader)

        assert len(batches) == len(dataloader) == 1
        assert batches[0].tolist() == [7, 7]

    def test_worker_targets_preserve_global_downsample_length(self):
        num_workers = 2
        dataset = IterableWithLenDataset(
            ArrayDataset([0, 1, 2]),
            shuffle=True,
            resample_ratio=0.5,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=num_workers,
            multiprocessing_context=_get_dataloader_multiprocessing_context(
                num_workers
            ),
        )

        items = list(dataloader)

        assert len(items) == dataset.total_iterator_length == 2
        assert len(set(items)) == 2

    @pytest.mark.parametrize("resample_ratio", [0.75, 2.0])
    @pytest.mark.parametrize("use_dataset_side_batching", [False, True])
    def test_shuffled_multi_worker_resampling_uses_disjoint_worker_shards(
        self,
        resample_ratio: float,
        use_dataset_side_batching: bool,
    ):
        num_workers = 2
        dataset = IterableWithLenDataset(
            ArrayDataset(list(range(8))),
            shuffle=True,
            resample_ratio=resample_ratio,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=2 if use_dataset_side_batching else None,
            num_workers=num_workers,
            generator=torch.Generator().manual_seed(0),
            use_dataset_side_batching=use_dataset_side_batching,
            multiprocessing_context=_get_dataloader_multiprocessing_context(
                num_workers
            ),
        )

        dataloader_items = list(dataloader)
        items = (
            [int(item) for batch in dataloader_items for item in batch]
            if use_dataset_side_batching
            else [int(item) for item in dataloader_items]
        )

        if resample_ratio < 1:
            assert len(items) == 6
            assert len(set(items)) == 6
        else:
            assert sorted(items) == sorted([*range(8), *range(8)])

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

    def test_resample_ratios_control_exact_row_counts(self):
        downsampled = DictIterableDataset(
            [
                ArrayDatasetItem(data=list(range(10))),
                ArrayDatasetItem(data=list(range(100, 110))),
            ],
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
            resample_ratios=[0.5, 0.5],
        )
        oversampled = DictIterableDataset(
            [ArrayDatasetItem(data=list(range(10)))],
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
            resample_ratios=2.5,
        )

        downsampled_items = list(downsampled)
        first = [item for item in downsampled_items if item < 100]
        second = [item for item in downsampled_items if item >= 100]
        oversampled_items = list(oversampled)

        assert len(first) == len(set(first)) == 5
        assert len(second) == len(set(second)) == 5
        assert downsampled.total_iterator_length == 10
        assert sorted(oversampled_items[:10]) == list(range(10))
        assert sorted(oversampled_items[10:20]) == list(range(10))
        assert len(set(oversampled_items[20:])) == 5
        assert oversampled.total_iterator_length == 25

    def test_resample_ratio_input_forms_preserve_scale_one_behavior(
        self,
        dummy_dataset_items: list[DatasetItem],
    ):
        outputs = [
            list(
                DictIterableDataset(
                    dummy_dataset_items,
                    shuffle=False,
                    resample_ratios=ratios,
                )
            )
            for ratios in (None, 1.0, [1.0, 1.0])
        ]

        assert outputs[0] == outputs[1] == outputs[2]

    @pytest.mark.parametrize(
        "resample_ratios,exception",
        [
            ([], ValueError),
            ([0.0], ValueError),
            ([-1.0], ValueError),
            ([float("nan")], ValueError),
            ([float("inf")], ValueError),
            ([True], TypeError),
            (["1.0"], TypeError),
        ],
    )
    def test_resample_ratios_validate_public_contract(
        self,
        resample_ratios: Any,
        exception: type[Exception],
    ):
        with pytest.raises(exception):
            DictIterableDataset(
                [ArrayDatasetItem(data=[0])],
                shuffle=True,
                resample_ratios=resample_ratios,
            )

    @pytest.mark.parametrize("resample_ratios", [0.5, 2.0, [1.0, 2.0]])
    def test_resample_ratios_require_shuffle(
        self,
        resample_ratios: float | list[float],
    ):
        datasets = [ArrayDatasetItem(data=[0])]
        if isinstance(resample_ratios, list):
            datasets.append(ArrayDatasetItem(data=[1]))

        with pytest.raises(ValueError, match="require shuffle=True"):
            DictIterableDataset(
                datasets,
                shuffle=False,
                resample_ratios=resample_ratios,
            )

    def test_zero_target_items_are_skipped_without_zero_weight_division(self):
        mixed = DictIterableDataset(
            [
                ArrayDatasetItem(data=[0]),
                ArrayDatasetItem(data=[100, 101]),
            ],
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
            resample_ratios=[0.5, 1.0],
        )
        all_zero = DictIterableDataset(
            [ArrayDatasetItem(data=[0])],
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
            resample_ratios=[0.5],
        )

        assert sorted(mixed) == [100, 101]
        assert mixed._total_indices_length == [0, 2]
        assert list(all_zero) == []
        assert all_zero.total_iterator_length == 0

    def test_resampling_batches_the_complete_target_row_stream_once(self):
        dataset = DictIterableDataset(
            [ArrayDatasetItem(data=list(range(10)))],
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
            resample_ratios=2.0,
            batch_loader_kwargs=BatchLoaderConfig(
                batch_size=4,
                drop_last=False,
            ),
        )

        batches = list(dataset)
        flattened = [int(item) for batch in batches for item in batch]

        assert len(batches) == dataset.get_total_batch_num(
            num_workers=0,
            batch_size=4,
            drop_last=False,
        )
        assert sorted(flattened) == sorted([*range(10), *range(10)])

    def test_resampling_batches_across_natural_cycles_before_drop_last(self):
        dataset = DictIterableDataset(
            [ArrayDatasetItem(data=[0, 1])],
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
            resample_ratios=2.0,
            batch_loader_kwargs=BatchLoaderConfig(
                batch_size=4,
                drop_last=True,
            ),
        )

        batches = list(dataset)

        assert len(batches) == 1
        assert sorted(batches[0].tolist()) == [0, 0, 1, 1]
        assert (
            dataset.get_total_batch_num(
                num_workers=0,
                batch_size=4,
                drop_last=True,
            )
            == 1
        )

    def test_resampling_fails_when_positive_metadata_yields_no_rows(self):
        class _EmptyDespiteMetadataItem(ArrayDatasetItem):
            def get_dataset_row_num(self) -> int:
                return 1

        dataset = DictIterableDataset(
            [_EmptyDespiteMetadataItem(data=[])],
            shuffle=True,
            resample_ratios=2.0,
        )

        with pytest.raises(RuntimeError, match="produced no rows"):
            list(dataset)

    def test_resample_metadata_and_item_names_survive_shard_and_clone(self):
        dataset = DictIterableDataset(
            [
                ArrayDatasetItem(data=list(range(10)), name="first"),
                ArrayDatasetItem(data=list(range(100, 110)), name="second"),
            ],
            shuffle=True,
            resample_ratios=[0.5, 2.0],
        )

        sharded = dataset.shard(num_shards=2, index=1)
        dataloader = DataLoader(
            dataset,
            batch_size=3,
            use_dataset_side_batching=True,
        )

        assert sharded._resample_ratios == [0.5, 2.0]
        assert [item.name for item in sharded.dataset_items] == [
            "first",
            "second",
        ]
        assert isinstance(dataloader.dataset, DictIterableDataset)
        assert dataloader.dataset._resample_ratios == [0.5, 2.0]
        assert [item.name for item in dataloader.dataset.dataset_items] == [
            "first",
            "second",
        ]

    def test_summary_reports_resampled_and_real_sharded_ratios(self):
        dataset = DictIterableDataset(
            [
                ArrayDatasetItem(data=list(range(10)), name="aaa"),
                ArrayDatasetItem(data=list(range(10, 20)), name="数\n据"),
            ],
            shuffle=True,
            resample_ratios=[1.0, 3.0],
        )

        summary = dataset.summary()
        lines = summary.splitlines()

        assert len(lines) == 4
        assert lines[0].startswith(" " * 28 + "name sample_ratio")
        assert "[frame_ratio]" in lines[0]
        assert lines[1].startswith("├" + "-" * 27 + "aaa:")
        assert lines[2].startswith("├" + "-" * 25 + "数_据:")
        assert "25.00%" in lines[1]
        assert "75.00%" in lines[2]
        assert "50.00%" in lines[1]
        assert "50.00%" in lines[2]
        assert lines[3].endswith("[    20]")
        assert lines[0].index("name") + len("name") == (
            lines[1].index(":") + 1
        )
        assert lines[0].index("sample_ratio") + len("sample_ratio") == (
            lines[1].index("25.00%") + len("25.00%")
        )
        assert lines[0].index("frame_ratio") + len("frame_ratio") == (
            lines[1].index("50.00%") + len("50.00%")
        )
        assert lines[0].index("length") + len("length") == (
            lines[1].index("10") + len("10")
        )

    def test_summary_handles_an_empty_shard(self):
        dataset = DictIterableDataset(
            [ArrayDatasetItem(data=[0])],
            shard_kwargs=ShardConfig(shard_strategy="drop_last"),
        ).shard(num_shards=2, index=1)

        summary = dataset.summary()

        assert "item_0" in summary
        assert summary.count("0.00%") == 4
        assert summary.splitlines()[-1].endswith("[     0]")

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

    @pytest.mark.skipif(
        (
            "spawn" not in mp.get_all_start_methods()
            or "file_descriptor"
            not in torch.multiprocessing.get_all_sharing_strategies()
        ),
        reason="requires spawn workers with file-descriptor Torch IPC",
    )
    def test_torch_generator_is_safe_for_spawn_workers(
        self,
        dummy_dataset_items: list[DatasetItem],
    ) -> None:
        """Spawn workers restore Torch RNG state without shared storage FDs."""

        generator = torch.Generator().manual_seed(17)
        dataset = DictIterableDataset(
            dummy_dataset_items,
            shuffle=False,
            generator=generator,
        )
        # The project DataLoader clones iterable datasets before Torch starts
        # workers. Use Torch's loader directly to exercise the serialization
        # boundary used by the profiling runner.
        dataloader = TorchDataLoader(
            dataset,
            batch_size=2,
            num_workers=1,
            multiprocessing_context="spawn",
        )

        sharing_strategy = torch.multiprocessing.get_sharing_strategy()
        try:
            # The test suite normally selects ``file_system`` IPC globally.
            # Production uses Torch's Linux ``file_descriptor`` default, whose
            # generator-state transport is the boundary under test here.
            torch.multiprocessing.set_sharing_strategy("file_descriptor")
            batches = list(dataloader)
        finally:
            torch.multiprocessing.set_sharing_strategy(sharing_strategy)

        assert batches
        assert sorted(
            item
            for batch in batches
            for item in cast(torch.Tensor, batch).tolist()
        ) == [*range(10), *range(100, 110)]

    def test_torch_generator_pickle_preserves_rng_state(
        self,
        dummy_dataset_items: list[DatasetItem],
    ) -> None:
        """Dataset serialization must not change Torch shuffle sequences."""

        generator = torch.Generator().manual_seed(23)
        dataset = DictIterableDataset(
            dummy_dataset_items,
            shuffle=True,
            generator=generator,
        )
        expected_state = generator.get_state().clone()

        restored = pickle.loads(pickle.dumps(dataset))

        assert isinstance(restored._generator, torch.Generator)
        assert restored._generator is not generator
        assert torch.equal(restored._generator.get_state(), expected_state)
        assert "_serialized_torch_generator" not in restored.__dict__

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

    @pytest.mark.parametrize("stop_early", [False, True])
    @pytest.mark.parametrize("resample_ratio", [1.0, 2.5])
    @pytest.mark.parametrize("use_dataset_side_batching", [False, True])
    def test_iteration_closes_datasets_created_by_items(
        self,
        dummy_dataset_items: list[DatasetItem],
        monkeypatch: pytest.MonkeyPatch,
        stop_early: bool,
        resample_ratio: float,
        use_dataset_side_batching: bool,
    ) -> None:
        """Close item-created datasets on exhaustion and early exit."""

        class _CloseTrackingArrayDataset(ArrayDataset):
            def __init__(self, data: list) -> None:
                super().__init__(data)
                self.closed = False

            def close(self) -> None:
                self.closed = True

        created_datasets: list[_CloseTrackingArrayDataset] = []

        def create_closeable_dataset(
            item: ArrayDatasetItem,
        ) -> _CloseTrackingArrayDataset:
            created = _CloseTrackingArrayDataset(item.data)
            created_datasets.append(created)
            return created

        monkeypatch.setattr(
            ArrayDatasetItem,
            "_create_dataset",
            create_closeable_dataset,
        )
        dataset = DictIterableDataset(
            dummy_dataset_items,
            shuffle=resample_ratio != 1.0,
            generator=torch.Generator().manual_seed(7),
            resample_ratios=resample_ratio,
            batch_loader_kwargs=(
                BatchLoaderConfig(batch_size=3)
                if use_dataset_side_batching
                else None
            ),
        )
        dataset_iter = iter(dataset)

        if stop_early:
            first_item = next(dataset_iter)
            if use_dataset_side_batching:
                assert len(first_item) == 3
            else:
                assert first_item in {*range(10), *range(100, 110)}
            cast(Any, dataset_iter).close()
        else:
            list(dataset_iter)

        assert created_datasets
        assert all(created.closed for created in created_datasets)

    def test_iteration_failure_closes_dataset_created_by_item(
        self,
        dummy_dataset_items: list[DatasetItem],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Close the item-created dataset when row iteration fails."""

        class _FailingArrayDataset(ArrayDataset):
            def __init__(self, data: list) -> None:
                super().__init__(data)
                self.closed = False

            def __getitem__(self, index: int) -> Any:
                raise RuntimeError(f"row {index} failed")

            def close(self) -> None:
                self.closed = True

        created_datasets: list[_FailingArrayDataset] = []

        def create_failing_dataset(
            item: ArrayDatasetItem,
        ) -> _FailingArrayDataset:
            created = _FailingArrayDataset(item.data)
            created_datasets.append(created)
            return created

        monkeypatch.setattr(
            ArrayDatasetItem,
            "_create_dataset",
            create_failing_dataset,
        )
        dataset = DictIterableDataset(dummy_dataset_items, shuffle=False)

        with pytest.raises(RuntimeError, match="row 0 failed"):
            list(dataset)

        assert len(created_datasets) == 1
        assert created_datasets[0].closed is True

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

    @pytest.mark.parametrize("resample_ratio", [0.5, 2.0])
    @pytest.mark.parametrize("use_dataset_side_batching", [False, True])
    def test_resampled_total_batch_consistency_across_workers(
        self,
        dummy_dataset_items: list[DatasetItem],
        resample_ratio: float,
        use_dataset_side_batching: bool,
    ):
        batch_size = 4
        num_workers = 2
        dataset = DictIterableDataset(
            dummy_dataset_items,
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
            resample_ratios=resample_ratio,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
            use_dataset_side_batching=use_dataset_side_batching,
            multiprocessing_context=_get_dataloader_multiprocessing_context(
                num_workers
            ),
        )

        self._check_dataloader_total_batch_consistency(
            dataloader=dataloader,
            dataset=(
                cast(IterableDatasetMixin, dataloader.dataset)
                if use_dataset_side_batching
                else dataset
            ),
            batch_size=batch_size,
            drop_last=False,
        )

    @pytest.mark.parametrize("use_dataset_side_batching", [False, True])
    def test_worker_sharding_precedes_oversampling_for_tiny_dataset(
        self,
        use_dataset_side_batching: bool,
    ):
        num_workers = 2
        dataset = DictIterableDataset(
            [ArrayDatasetItem(data=[7])],
            shuffle=True,
            resample_ratios=2.0,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=num_workers,
            use_dataset_side_batching=use_dataset_side_batching,
            multiprocessing_context=_get_dataloader_multiprocessing_context(
                num_workers
            ),
        )

        batches = list(dataloader)

        assert len(batches) == len(dataloader) == 1
        assert batches[0].tolist() == [7, 7]

    @pytest.mark.parametrize("use_dataset_side_batching", [False, True])
    def test_shuffled_multi_worker_resampling_uses_each_item_worker_shard(
        self,
        use_dataset_side_batching: bool,
    ):
        num_workers = 2
        dataset = DictIterableDataset(
            [
                ArrayDatasetItem(data=list(range(8))),
                ArrayDatasetItem(data=list(range(100, 108))),
            ],
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
            resample_ratios=0.75,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=2 if use_dataset_side_batching else None,
            num_workers=num_workers,
            generator=torch.Generator().manual_seed(0),
            use_dataset_side_batching=use_dataset_side_batching,
            multiprocessing_context=_get_dataloader_multiprocessing_context(
                num_workers
            ),
        )

        dataloader_items = list(dataloader)
        items = (
            [int(item) for batch in dataloader_items for item in batch]
            if use_dataset_side_batching
            else [int(item) for item in dataloader_items]
        )
        first = [item for item in items if item < 100]
        second = [item for item in items if item >= 100]

        assert len(first) == len(set(first)) == 6
        assert len(second) == len(set(second)) == 6

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

    def test_dataset_item_name_is_optional_and_survives_sharding(self):
        unnamed_item = ArrayDatasetItem(data=[0])
        item = ArrayDatasetItem(data=list(range(10)), name="source/action")

        sharded_item = item.shard(num_shards=3, index=1)
        restored_item = ArrayDatasetItem.model_validate(item.model_dump())

        assert unnamed_item.name is None
        assert item.model_dump()["name"] == "source/action"
        assert restored_item.name == "source/action"
        assert sharded_item.name == "source/action"
        assert sharded_item.num_shards == 3
        assert sharded_item.shard_id == 1

    def test_dict_iterable_summary_reads_current_dataset_item_names(self):
        named_item = ArrayDatasetItem(data=[0], name="named")
        dataset = DictIterableDataset(
            [
                named_item,
                ArrayDatasetItem(data=[1]),
            ]
        )
        named_item.name = "renamed"

        summary = dataset.summary()

        assert "renamed" in summary
        assert "item_1" in summary

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

    def test_waits_for_k_plus_t_candidates_before_first_partition(self):
        # K=12 uses an effective T=2. The producer must publish the replacement
        # chunk atomically, so thirteen live-source candidates are
        # insufficient.
        allow_last_chunk_item = threading.Event()
        first_item_ready = threading.Event()
        yielded_items = []

        def blocking_iter():
            yield from range(13)
            if not allow_last_chunk_item.wait(timeout=1):
                raise TimeoutError(
                    "Timed out waiting for the last chunk item."
                )
            yield 13

        generator = torch.Generator()
        generator.manual_seed(0)
        prefetch_iter = create_prefetch_iterator(
            iter(blocking_iter()),
            prefetch_size=12,
            shuffle=True,
            generator=generator,
        )

        def consume_first_item():
            yielded_items.append(next(prefetch_iter))
            first_item_ready.set()

        consumer_thread = threading.Thread(target=consume_first_item)
        consumer_thread.start()

        # The producer has one item in its reserved local chunk. It must not
        # publish or wake the consumer until the second item is available.
        assert not first_item_ready.wait(timeout=0.1)

        allow_last_chunk_item.set()
        assert first_item_ready.wait(timeout=1)
        assert yielded_items[0] in range(14)

        consumer_thread.join(timeout=1)
        remaining_items = list(prefetch_iter)
        assert sorted(yielded_items + remaining_items) == list(range(14))

    def test_chunk_partition_matches_one_torch_randperm(self):
        actual_generator = torch.Generator().manual_seed(0)
        prefetch_iter = create_prefetch_iterator(
            iter(range(14)),
            prefetch_size=12,
            shuffle=True,
            generator=actual_generator,
        )

        actual_outputs = [next(prefetch_iter) for _ in range(2)]

        expected_generator = torch.Generator().manual_seed(0)
        pool = list(range(14))
        indices = torch.randperm(14, generator=expected_generator).tolist()
        assert actual_outputs == [pool[index] for index in indices[:2]]
        assert cast(Any, prefetch_iter)._shuffle_reservoir == [
            pool[index] for index in indices[2:]
        ]
        assert torch.equal(
            actual_generator.get_state(),
            expected_generator.get_state(),
        )
        cast(Any, prefetch_iter).close()

    def test_chunk_partition_matches_one_numpy_permutation(self):
        actual_generator = np.random.default_rng(0)
        prefetch_iter = create_prefetch_iterator(
            iter(range(14)),
            prefetch_size=12,
            shuffle=True,
            generator=actual_generator,
        )

        actual_outputs = [next(prefetch_iter) for _ in range(2)]

        expected_generator = np.random.default_rng(0)
        pool = list(range(14))
        indices = expected_generator.permutation(14).tolist()
        assert actual_outputs == [pool[index] for index in indices[:2]]
        assert cast(Any, prefetch_iter)._shuffle_reservoir == [
            pool[index] for index in indices[2:]
        ]
        assert (
            actual_generator.bit_generator.state
            == expected_generator.bit_generator.state
        )
        cast(Any, prefetch_iter).close()

    def test_partial_eof_chunk_partitions_only_available_candidates(self):
        actual_generator = torch.Generator().manual_seed(11)
        prefetch_iter = create_prefetch_iterator(
            iter(range(13)),
            prefetch_size=12,
            shuffle=True,
            generator=actual_generator,
        )

        actual_output = next(prefetch_iter)

        expected_generator = torch.Generator().manual_seed(11)
        pool = list(range(13))
        indices = torch.randperm(13, generator=expected_generator).tolist()
        assert actual_output == pool[indices[0]]
        assert cast(Any, prefetch_iter)._shuffle_reservoir == [
            pool[index] for index in indices[1:]
        ]
        assert torch.equal(
            actual_generator.get_state(),
            expected_generator.get_state(),
        )
        cast(Any, prefetch_iter).close()

    def test_non_divisible_k_waits_for_full_live_partition_chunk(self):
        allow_last_chunk_item = threading.Event()
        first_item_ready = threading.Event()
        outputs: list[int] = []

        def blocking_iter():
            yield from range(14)
            if not allow_last_chunk_item.wait(timeout=1):
                raise TimeoutError(
                    "Timed out waiting for the last chunk item."
                )
            yield 14

        actual_generator = torch.Generator().manual_seed(12)
        prefetch_iter = create_prefetch_iterator(
            iter(blocking_iter()),
            prefetch_size=13,
            shuffle=True,
            generator=actual_generator,
        )

        def consume_first_item():
            outputs.append(next(prefetch_iter))
            first_item_ready.set()

        consumer_thread = threading.Thread(target=consume_first_item)
        consumer_thread.start()
        assert not first_item_ready.wait(timeout=0.1)

        allow_last_chunk_item.set()
        assert first_item_ready.wait(timeout=1)
        consumer_thread.join(timeout=1)

        expected_generator = torch.Generator().manual_seed(12)
        pool = list(range(15))
        indices = torch.randperm(15, generator=expected_generator).tolist()
        assert outputs == [pool[indices[0]]]
        assert list(cast(Any, prefetch_iter)._ready_queue) == [
            pool[indices[1]]
        ]
        assert cast(Any, prefetch_iter)._shuffle_reservoir == [
            pool[index] for index in indices[2:]
        ]
        assert torch.equal(
            actual_generator.get_state(),
            expected_generator.get_state(),
        )
        assert sorted(outputs + list(prefetch_iter)) == list(range(15))

    def test_non_divisible_k_reuses_partial_queue_without_deadlock(self):
        source_length = 200
        prefetch_iter = create_prefetch_iterator(
            iter(range(source_length)),
            prefetch_size=38,
            shuffle=True,
            generator=torch.Generator().manual_seed(13),
        )
        consume_done = threading.Event()
        outputs: list[int] = []
        errors: list[BaseException] = []

        def consume_all():
            try:
                outputs.extend(prefetch_iter)
            except BaseException as exc:
                errors.append(exc)
            finally:
                consume_done.set()

        consumer_thread = threading.Thread(target=consume_all)
        consumer_thread.start()
        if not consume_done.wait(timeout=2):
            cast(Any, prefetch_iter).close(timeout=1)
        consumer_thread.join(timeout=1)

        assert consume_done.is_set()
        assert errors == []
        assert sorted(outputs) == list(range(source_length))

    @pytest.mark.parametrize("prefetch_size", range(6, 129))
    def test_chunk_layout_keeps_two_chunks_within_1_5k(
        self,
        prefetch_size: int,
    ):
        prefetch_iter = create_prefetch_iterator(
            iter(range(1000)),
            prefetch_size=prefetch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(1),
        )
        state = cast(Any, prefetch_iter)._state

        chunk_size = state.producer_chunk_size
        incoming_capacity = state.buffer_capacity
        headroom = prefetch_size // 2
        expected_chunk_size = min(16, headroom // 3)
        expected_capacity = (
            (headroom - expected_chunk_size) // expected_chunk_size
        ) * expected_chunk_size
        assert chunk_size == expected_chunk_size
        assert incoming_capacity == expected_capacity
        assert incoming_capacity >= 2 * chunk_size
        assert incoming_capacity % chunk_size == 0
        assert (
            prefetch_size + chunk_size + incoming_capacity
            <= prefetch_size + prefetch_size // 2
        )

        cast(Any, prefetch_iter).close()

    @pytest.mark.parametrize("prefetch_size", range(2, 6))
    def test_small_k_scalar_fallback_shares_1_5k_headroom(
        self,
        prefetch_size: int,
    ):
        produced_count = 0

        def counted_iter():
            nonlocal produced_count
            for item in range(100):
                produced_count += 1
                yield item

        prefetch_iter = create_prefetch_iterator(
            iter(counted_iter()),
            prefetch_size=prefetch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(1),
        )
        state = cast(Any, prefetch_iter)._state
        assert state.producer_chunk_size == 1
        assert state.buffer_capacity == prefetch_size // 2
        assert cast(Any, prefetch_iter)._shuffle_uses_shared_headroom_credit

        outputs = []
        for item in prefetch_iter:
            outputs.append(item)
            assert produced_count - len(outputs) <= (
                prefetch_size + prefetch_size // 2
            )

        assert sorted(outputs) == list(range(100))

    def test_small_k_releases_credit_only_on_the_next_request(self):
        produced_count = 0

        def counted_iter():
            nonlocal produced_count
            for item in range(100):
                produced_count += 1
                yield item

        prefetch_iter = create_prefetch_iterator(
            iter(counted_iter()),
            prefetch_size=2,
            shuffle=True,
            generator=torch.Generator().manual_seed(1),
        )
        state = cast(Any, prefetch_iter)._state

        first_item = next(prefetch_iter)
        assert first_item in range(3)
        assert produced_count == 3
        with state.condition:
            assert state.consumer_reserved_size == 1
            assert len(state.incoming_queue) == 0

        time.sleep(0.05)
        assert produced_count == 3

        second_item = next(prefetch_iter)
        assert second_item in range(4)
        cast(Any, prefetch_iter).close()

    def test_chunk_credits_keep_total_materialization_within_1_5k(self):
        produced_count = 0

        def counted_iter():
            nonlocal produced_count
            for item in range(300):
                produced_count += 1
                yield item

        prefetch_iter = create_prefetch_iterator(
            iter(counted_iter()),
            prefetch_size=128,
            shuffle=True,
            generator=torch.Generator().manual_seed(1),
        )
        state = cast(Any, prefetch_iter)._state

        for _ in range(100):
            if produced_count == 48:
                break
            time.sleep(0.01)
        assert produced_count == 48
        with state.condition:
            assert state.producer_chunk_size == 16
            assert state.buffer_capacity == 48
            assert len(state.incoming_queue) == 48
            assert state.producer_reserved_size == 0

        yielded_item = next(prefetch_iter)
        assert yielded_item in range(192)

        # The yielded item plus K reservoir, T-1 active outputs, and N shared
        # producer-side items account for exactly 1.5K materialized samples.
        for _ in range(100):
            if produced_count == 192:
                break
            time.sleep(0.01)
        assert produced_count == 192
        time.sleep(0.05)
        assert produced_count == 192
        with state.condition:
            iterator_owned = (
                len(cast(Any, prefetch_iter)._shuffle_reservoir)
                + len(cast(Any, prefetch_iter)._ready_queue)
                + len(cast(Any, prefetch_iter)._shuffle_pending_incoming)
                + len(state.incoming_queue)
                + state.producer_reserved_size
            )
        assert iterator_owned == 191
        assert iterator_owned + 1 <= 128 + 128 // 2

        del yielded_item
        cast(Any, prefetch_iter).close()

    def test_slow_producer_does_not_drain_shuffle_reservoir(self):
        allow_next_item = threading.Event()
        producer_waiting = threading.Event()
        second_item_ready = threading.Event()
        yielded_items: list[int] = []

        def blocking_iter():
            yield from range(14)
            producer_waiting.set()
            if not allow_next_item.wait(timeout=1):
                raise TimeoutError("Timed out waiting for the next item.")
            yield 14
            yield 15

        prefetch_iter = create_prefetch_iterator(
            iter(blocking_iter()),
            prefetch_size=12,
            shuffle=True,
            generator=torch.Generator().manual_seed(2),
        )

        yielded_items.append(next(prefetch_iter))
        assert producer_waiting.wait(timeout=1)
        assert len(cast(Any, prefetch_iter)._ready_queue) == 1

        # The second output was already partitioned with the first one, so a
        # slow producer cannot add another per-sample handoff here.
        yielded_items.append(next(prefetch_iter))

        def consume_third_item():
            yielded_items.append(next(prefetch_iter))
            second_item_ready.set()

        consumer_thread = threading.Thread(target=consume_third_item)
        consumer_thread.start()
        assert not second_item_ready.wait(timeout=0.1)

        allow_next_item.set()
        assert second_item_ready.wait(timeout=1)
        consumer_thread.join(timeout=1)

        remaining_items = list(prefetch_iter)
        assert sorted(yielded_items + remaining_items) == list(range(16))

    @pytest.mark.parametrize("source_length", [0, 1, 3, 4, 5, 6, 20])
    def test_streaming_shuffle_preserves_input_multiset(
        self,
        source_length: int,
    ):
        prefetch_iter = create_prefetch_iterator(
            iter(range(source_length)),
            prefetch_size=4,
            shuffle=True,
            generator=torch.Generator().manual_seed(3),
        )

        assert sorted(prefetch_iter) == list(range(source_length))

    @pytest.mark.parametrize("generator_kind", ["torch", "numpy"])
    def test_producer_timing_does_not_change_shuffle_order(
        self,
        generator_kind: str,
    ):
        def collect(delay: float) -> list[int]:
            def source_iter():
                for item in range(30):
                    if delay > 0 and item % 3 == 0:
                        time.sleep(delay)
                    yield item

            if generator_kind == "torch":
                generator: torch.Generator | np.random.Generator = (
                    torch.Generator().manual_seed(4)
                )
            else:
                generator = np.random.default_rng(4)
            return list(
                create_prefetch_iterator(
                    iter(source_iter()),
                    prefetch_size=4,
                    shuffle=True,
                    generator=generator,
                )
            )

        assert collect(delay=0.0) == collect(delay=0.001)

    def test_torch_chunk_rng_matches_repeated_randperm(self):
        prefetch_size = 12
        chunk_size = 2
        output_count = 4
        actual_generator = torch.Generator().manual_seed(5)
        prefetch_iter = create_prefetch_iterator(
            iter(range(100)),
            prefetch_size=prefetch_size,
            shuffle=True,
            generator=actual_generator,
        )

        actual_outputs = [next(prefetch_iter) for _ in range(output_count)]
        cast(Any, prefetch_iter).close()

        expected_generator = torch.Generator().manual_seed(5)
        reservoir = list(range(prefetch_size))
        expected_outputs = []
        for chunk_start in range(
            prefetch_size,
            prefetch_size + output_count,
            chunk_size,
        ):
            pool = reservoir + list(
                range(chunk_start, chunk_start + chunk_size)
            )
            indices = torch.randperm(
                len(pool),
                generator=expected_generator,
            ).tolist()
            expected_outputs.extend(
                pool[index] for index in indices[:chunk_size]
            )
            reservoir = [pool[index] for index in indices[chunk_size:]]

        assert actual_outputs == expected_outputs
        assert torch.equal(
            actual_generator.get_state(),
            expected_generator.get_state(),
        )

    @pytest.mark.parametrize("generator_kind", ["torch", "numpy"])
    def test_chunk_rng_matches_reference_for_all_supported_k(
        self,
        generator_kind: str,
    ):
        for prefetch_size in range(2, 129):
            headroom = prefetch_size // 2
            chunk_size = min(16, headroom // 3) if headroom >= 3 else 1
            source = list(range(prefetch_size + 2 * chunk_size + 1))
            if generator_kind == "torch":
                actual_generator: torch.Generator | np.random.Generator = (
                    torch.Generator().manual_seed(17)
                )
                expected_generator: torch.Generator | np.random.Generator = (
                    torch.Generator().manual_seed(17)
                )
            else:
                actual_generator = np.random.default_rng(17)
                expected_generator = np.random.default_rng(17)

            actual = list(
                create_prefetch_iterator(
                    iter(source),
                    prefetch_size=prefetch_size,
                    shuffle=True,
                    generator=actual_generator,
                )
            )

            reservoir = source[:prefetch_size]
            expected = []
            for offset in range(prefetch_size, len(source), chunk_size):
                incoming = source[offset : offset + chunk_size]
                pool = reservoir + incoming
                if isinstance(expected_generator, torch.Generator):
                    indices = torch.randperm(
                        len(pool),
                        generator=expected_generator,
                    ).tolist()
                else:
                    indices = expected_generator.permutation(
                        len(pool)
                    ).tolist()
                expected.extend(
                    pool[index] for index in indices[: len(incoming)]
                )
                reservoir = [pool[index] for index in indices[len(incoming) :]]

            if isinstance(expected_generator, torch.Generator):
                tail_indices = torch.randperm(
                    len(reservoir),
                    generator=expected_generator,
                ).tolist()
            else:
                tail_indices = expected_generator.permutation(
                    len(reservoir)
                ).tolist()
            expected.extend(reservoir[index] for index in tail_indices)

            assert actual == expected, prefetch_size
            if isinstance(actual_generator, torch.Generator):
                assert isinstance(expected_generator, torch.Generator)
                assert torch.equal(
                    actual_generator.get_state(),
                    expected_generator.get_state(),
                ), prefetch_size
            else:
                assert isinstance(expected_generator, np.random.Generator)
                assert (
                    actual_generator.bit_generator.state
                    == expected_generator.bit_generator.state
                ), prefetch_size

    def test_early_close_commits_the_whole_chunk_rng_draw(self):
        actual_generator = torch.Generator().manual_seed(6)
        prefetch_iter = create_prefetch_iterator(
            iter(range(100)),
            prefetch_size=12,
            shuffle=True,
            generator=actual_generator,
        )

        next(prefetch_iter)
        cast(Any, prefetch_iter).close()

        expected_generator = torch.Generator().manual_seed(6)
        torch.randperm(14, generator=expected_generator)
        assert torch.equal(
            actual_generator.get_state(),
            expected_generator.get_state(),
        )

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

        prefetch_iter = create_prefetch_iterator(
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
        item_requests = 0

        def blocking_iter():
            nonlocal item_requests
            item_requests += 1
            yield 0
            item_requests += 1
            yield 1
            item_requests += 1
            yield 2

        baseline_threads = self._count_prefetch_threads()
        prefetch_iter = create_prefetch_iterator(
            iter(blocking_iter()),
            prefetch_size=2,
            shuffle=False,
            generator=None,
        )

        for _ in range(40):
            if item_requests == 2:
                break
            time.sleep(0.05)
        assert item_requests == 2

        cast(Any, prefetch_iter).close()

        for _ in range(20):
            if self._count_prefetch_threads() == baseline_threads:
                break
            threading.Event().wait(0.05)

        assert self._count_prefetch_threads() == baseline_threads

    def test_close_waits_for_short_inflight_prefetch_item(
        self,
        caplog,
        monkeypatch: pytest.MonkeyPatch,
    ):
        tail_item_started = threading.Event()

        monkeypatch.setattr(
            prefetch_module,
            "_PREFETCH_CLOSE_JOIN_TIMEOUT_SEC",
            0.02,
        )
        caplog.set_level("INFO")

        def slow_tail_iter():
            yield 0
            yield 1
            tail_item_started.set()
            time.sleep(0.1)
            yield 2

        baseline_threads = self._count_prefetch_threads()
        prefetch_iter = create_prefetch_iterator(
            iter(slow_tail_iter()),
            prefetch_size=2,
            shuffle=False,
            generator=None,
        )

        assert next(prefetch_iter) == 0
        assert tail_item_started.wait(timeout=1)

        cast(Any, prefetch_iter).close()

        assert self._count_prefetch_threads() == baseline_threads
        assert "continuing the soft close wait" in caplog.text
        assert "exited and was joined" in caplog.text

    def test_close_warns_when_inflight_prefetch_item_blocks(self, caplog):
        tail_item_started = threading.Event()
        allow_tail_item = threading.Event()

        def blocked_tail_iter():
            yield 0
            yield 1
            tail_item_started.set()
            allow_tail_item.wait()
            yield 2

        baseline_threads = self._count_prefetch_threads()
        prefetch_iter = create_prefetch_iterator(
            iter(blocked_tail_iter()),
            prefetch_size=2,
            shuffle=False,
            generator=None,
        )

        assert next(prefetch_iter) == 0
        assert tail_item_started.wait(timeout=1)

        start = time.monotonic()
        cast(Any, prefetch_iter).close(timeout=0.2)
        elapsed = time.monotonic() - start

        assert elapsed < 0.7
        assert "close() is returning while the producer remains alive" in (
            caplog.text
        )

        allow_tail_item.set()
        for _ in range(40):
            if self._count_prefetch_threads() == baseline_threads:
                break
            threading.Event().wait(0.05)

        assert self._count_prefetch_threads() == baseline_threads

    @pytest.mark.parametrize("observation_path", ["next", "close"])
    def test_producer_error_is_raised_before_blocking_source_close(
        self,
        observation_path: str,
    ):
        class _ErrorThenBlockingCloseIterator:
            def __init__(self) -> None:
                self.next_started = threading.Event()
                self.allow_error = threading.Event()
                self.close_started = threading.Event()
                self.allow_close = threading.Event()

            def __iter__(self):
                return self

            def __next__(self) -> int:
                self.next_started.set()
                if not self.allow_error.wait(timeout=1):
                    raise TimeoutError("Timed out waiting to raise error.")
                raise RuntimeError("primary source error")

            def close(self) -> None:
                self.close_started.set()
                if not self.allow_close.wait(timeout=1):
                    raise TimeoutError("Timed out waiting to close source.")
                raise RuntimeError("secondary source close error")

        source_iter = _ErrorThenBlockingCloseIterator()
        prefetch_iter = create_prefetch_iterator(
            source_iter,
            prefetch_size=4,
            shuffle=True,
            generator=torch.Generator().manual_seed(6),
        )
        assert source_iter.next_started.wait(timeout=1)

        observer_done = threading.Event()
        observer_errors: list[BaseException] = []

        def observe_producer_error():
            try:
                if observation_path == "next":
                    next(prefetch_iter)
                else:
                    cast(Any, prefetch_iter).close(timeout=2.0)
            except BaseException as exc:
                observer_errors.append(exc)
            finally:
                observer_done.set()

        observer_thread = threading.Thread(target=observe_producer_error)
        if observation_path == "next":
            observer_thread.start()
            time.sleep(0.05)
            source_iter.allow_error.set()
            assert source_iter.close_started.wait(timeout=1)
        else:
            source_iter.allow_error.set()
            assert source_iter.close_started.wait(timeout=1)
            observer_thread.start()

        try:
            assert observer_done.wait(timeout=0.5)
            assert len(observer_errors) == 1
            assert isinstance(observer_errors[0], RuntimeError)
            assert str(observer_errors[0]) == "primary source error"
            if observation_path == "next":
                start = time.monotonic()
                cast(Any, prefetch_iter).close(timeout=2.0)
                assert time.monotonic() - start < 0.5
        finally:
            source_iter.allow_close.set()
            observer_thread.join(timeout=1)
            cast(Any, prefetch_iter).close(timeout=1)

    def test_close_observes_producer_error_arriving_during_wait(self):
        class _LateErrorThenBlockingCloseIterator:
            def __init__(self) -> None:
                self.next_started = threading.Event()
                self.allow_error = threading.Event()
                self.close_started = threading.Event()
                self.allow_close = threading.Event()

            def __iter__(self):
                return self

            def __next__(self) -> int:
                self.next_started.set()
                if not self.allow_error.wait(timeout=2):
                    raise TimeoutError("Timed out waiting to raise error.")
                raise RuntimeError("late primary source error")

            def close(self) -> None:
                self.close_started.set()
                if not self.allow_close.wait(timeout=2):
                    raise TimeoutError("Timed out waiting to close source.")

        source_iter = _LateErrorThenBlockingCloseIterator()
        prefetch_iter = create_prefetch_iterator(
            source_iter,
            prefetch_size=4,
            shuffle=True,
            generator=torch.Generator().manual_seed(9),
        )
        assert source_iter.next_started.wait(timeout=1)

        close_done = threading.Event()
        close_errors: list[BaseException] = []

        def close_prefetch_iter():
            try:
                cast(Any, prefetch_iter).close(timeout=2.0)
            except BaseException as exc:
                close_errors.append(exc)
            finally:
                close_done.set()

        close_thread = threading.Thread(target=close_prefetch_iter)
        close_thread.start()
        state = cast(Any, prefetch_iter)._state
        for _ in range(100):
            with state.condition:
                if state.consumer_closed:
                    break
            time.sleep(0.01)
        with state.condition:
            assert state.consumer_closed

        source_iter.allow_error.set()
        try:
            assert source_iter.close_started.wait(timeout=1)
            assert close_done.wait(timeout=0.5)
            assert len(close_errors) == 1
            assert isinstance(close_errors[0], RuntimeError)
            assert str(close_errors[0]) == "late primary source error"
        finally:
            source_iter.allow_close.set()
            close_thread.join(timeout=1)
            cast(Any, prefetch_iter).close(timeout=1)

    def test_close_releases_buffered_sample_references(self):
        class _Payload:
            pass

        payload_refs: list[weakref.ReferenceType[_Payload]] = []

        def payload_iter():
            for _ in range(100):
                payload = _Payload()
                payload_refs.append(weakref.ref(payload))
                yield payload

        prefetch_iter = create_prefetch_iterator(
            iter(payload_iter()),
            prefetch_size=4,
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
        )
        yielded_item = next(prefetch_iter)
        for _ in range(40):
            if len(payload_refs) == 6:
                break
            time.sleep(0.01)
        assert len(payload_refs) == 6

        del yielded_item
        cast(Any, prefetch_iter).close()
        gc.collect()

        assert all(payload_ref() is None for payload_ref in payload_refs)

    def test_close_releases_producer_local_sample_before_source_close(self):
        class _Payload:
            pass

        class _OnePayloadThenBlockingCloseIterator:
            def __init__(self) -> None:
                self.produced = False
                self.payload_ref: weakref.ReferenceType[_Payload] | None = None
                self.close_started = threading.Event()
                self.allow_close = threading.Event()

            def __iter__(self):
                return self

            def __next__(self) -> _Payload:
                if self.produced:
                    raise StopIteration
                self.produced = True
                payload = _Payload()
                self.payload_ref = weakref.ref(payload)
                return payload

            def close(self) -> None:
                self.close_started.set()
                self.allow_close.wait()

        source_iter = _OnePayloadThenBlockingCloseIterator()
        prefetch_iter = create_prefetch_iterator(
            source_iter,
            prefetch_size=2,
            shuffle=True,
            generator=torch.Generator().manual_seed(10),
        )
        state = cast(Any, prefetch_iter)._state
        for _ in range(100):
            with state.condition:
                if state.incoming_queue:
                    break
            time.sleep(0.01)
        with state.condition:
            assert len(state.incoming_queue) == 1
        assert source_iter.payload_ref is not None

        cast(Any, prefetch_iter).close(
            raise_producer_errors=False,
            timeout=0.0,
        )
        try:
            assert source_iter.close_started.wait(timeout=1)
            gc.collect()
            assert source_iter.payload_ref() is None
        finally:
            source_iter.allow_close.set()
            cast(Any, prefetch_iter).close(timeout=1)

    def test_producer_error_releases_buffered_sample_references(self):
        class _Payload:
            pass

        payload_refs: list[weakref.ReferenceType[_Payload]] = []

        def failing_payload_iter():
            for _ in range(7):
                payload = _Payload()
                payload_refs.append(weakref.ref(payload))
                yield payload
            raise RuntimeError("payload source failed")

        prefetch_iter = create_prefetch_iterator(
            iter(failing_payload_iter()),
            prefetch_size=4,
            shuffle=True,
            generator=torch.Generator().manual_seed(8),
        )

        with pytest.raises(RuntimeError, match="payload source failed"):
            while True:
                yielded_item = next(prefetch_iter)
                del yielded_item

        cast(Any, prefetch_iter).close()
        gc.collect()

        assert all(payload_ref() is None for payload_ref in payload_refs)

    def test_close_logs_producer_error_in_gc_close_path(self, caplog):
        baseline_threads = self._count_prefetch_threads()

        def failing_iter():
            yield 0
            yield 1
            raise RuntimeError("producer failed during close")

        prefetch_iter = create_prefetch_iterator(
            iter(failing_iter()),
            prefetch_size=3,
            shuffle=False,
            generator=None,
        )

        for _ in range(40):
            if self._count_prefetch_threads() == baseline_threads:
                break
            time.sleep(0.05)

        cast(Any, prefetch_iter).close(raise_producer_errors=False)

        assert "producer-side exception" in caplog.text

    def test_close_calls_source_iter_close_from_producer_thread(self):
        class _CloseTrackingIterator:
            def __init__(self) -> None:
                self.item_count = 0
                self.window_ready = threading.Event()
                self.closed = threading.Event()
                self.close_thread_name: str | None = None

            def __iter__(self):
                return self

            def __next__(self) -> int:
                if self.item_count >= 2:
                    raise StopIteration
                value = self.item_count
                self.item_count += 1
                if self.item_count == 2:
                    self.window_ready.set()
                return value

            def close(self) -> None:
                self.close_thread_name = threading.current_thread().name
                self.closed.set()

        source_iter = _CloseTrackingIterator()
        prefetch_iter = create_prefetch_iterator(
            source_iter,
            prefetch_size=2,
            shuffle=False,
            generator=None,
        )

        assert source_iter.window_ready.wait(timeout=1)
        cast(Any, prefetch_iter).close()

        assert source_iter.closed.is_set()
        assert source_iter.close_thread_name == "dataset-prefetch-producer"

    def test_source_iter_close_error_is_raised_without_timeout(self):
        class _FailingCloseIterator:
            def __init__(self) -> None:
                self.item_count = 0
                self.window_ready = threading.Event()
                self.closed = threading.Event()

            def __iter__(self):
                return self

            def __next__(self) -> int:
                if self.item_count >= 2:
                    raise StopIteration
                value = self.item_count
                self.item_count += 1
                if self.item_count == 2:
                    self.window_ready.set()
                return value

            def close(self) -> None:
                self.closed.set()
                raise RuntimeError("source close failed")

        source_iter = _FailingCloseIterator()
        prefetch_iter = create_prefetch_iterator(
            source_iter,
            prefetch_size=2,
            shuffle=False,
            generator=None,
        )

        assert source_iter.window_ready.wait(timeout=1)
        with pytest.raises(RuntimeError, match="source close failed"):
            cast(Any, prefetch_iter).close(timeout=0.2)

        assert source_iter.closed.is_set()

    def test_prefetch_thread_start_failure_closes_source_iterator(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        class _CloseTrackingIterator:
            def __init__(self) -> None:
                self.closed = False

            def __iter__(self):
                return self

            def __next__(self) -> int:
                return 0

            def close(self) -> None:
                self.closed = True

        class _FailingThread:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def start(self) -> None:
                raise RuntimeError("can't start new thread")

        source_iter = _CloseTrackingIterator()
        monkeypatch.setattr(
            prefetch_module.threading,
            "Thread",
            _FailingThread,
        )

        with pytest.raises(RuntimeError, match="can't start new thread"):
            create_prefetch_iterator(
                source_iter,
                prefetch_size=2,
                shuffle=False,
                generator=None,
            )

        assert source_iter.closed is True

    def test_second_close_consumes_error_after_first_close_timeout(
        self,
        caplog,
    ):
        source_blocked = threading.Event()
        unblock_source = threading.Event()

        def blocked_then_failing_iter():
            yield 0
            yield 1
            source_blocked.set()
            unblock_source.wait()
            raise RuntimeError("producer failed after timeout")

        baseline_threads = self._count_prefetch_threads()
        prefetch_iter = create_prefetch_iterator(
            iter(blocked_then_failing_iter()),
            prefetch_size=2,
            shuffle=False,
            generator=None,
        )

        assert next(prefetch_iter) == 0
        assert source_blocked.wait(timeout=1)
        cast(Any, prefetch_iter).close(timeout=0.05)
        assert "close() is returning while the producer remains alive" in (
            caplog.text
        )

        unblock_source.set()
        for _ in range(40):
            if self._count_prefetch_threads() == baseline_threads:
                break
            time.sleep(0.05)

        with pytest.raises(
            RuntimeError,
            match="producer failed after timeout",
        ):
            cast(Any, prefetch_iter).close(timeout=1.0)

    def test_close_closes_nested_prefetch_iterator_but_not_wrapper_state(
        self,
    ):
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
        # `_close_dataloader_iterator()` only tears down iterator-owned
        # resources. The prepared wrapper still owns Accelerate state cleanup.
        assert accelerator.gradient_state.in_dataloader
        assert (
            sum(
                ref is not None
                for ref in accelerator.gradient_state.dataloader_references
            )
            > baseline_ref_count
        )
        dataloader.end()
        assert not accelerator.gradient_state.in_dataloader
        assert (
            sum(
                ref is not None
                for ref in accelerator.gradient_state.dataloader_references
            )
            == baseline_ref_count
        )

        for _ in range(40):
            if self._count_prefetch_threads() == baseline_threads:
                break
            threading.Event().wait(0.05)

        assert self._count_prefetch_threads() == baseline_threads

    def test_close_closes_single_process_closeable_dataset_iterator(self):
        class _CloseTrackingIterator:
            def __init__(self) -> None:
                self._items = iter([0, 1, 2])
                self.closed = False

            def __iter__(self):
                return self

            def __next__(self) -> int:
                return next(self._items)

            def close(self) -> None:
                self.closed = True

        class _CloseTrackingDataset(TorchIterableDataset):
            def __init__(self) -> None:
                self.last_iter: Any | None = None

            def __iter__(self):
                self.last_iter = _CloseTrackingIterator()
                return self.last_iter

        dataset = _CloseTrackingDataset()
        dataloader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=0,
        )

        dataloader_iter = iter(dataloader)

        assert next(dataloader_iter) == 0

        _close_dataloader_iterator(cast(Any, dataloader_iter))

        assert cast(Any, dataset).last_iter.closed is True

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

    def test_close_resources_shutdowns_persistent_workers_for_early_break(
        self,
    ):
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

        close_dataloader_resources(
            dataloader,
            iterator,
            reason=DataloaderCloseReason.EARLY_BREAK,
        )

        assert getattr(dataloader, "_iterator", None) is None
        iterator = iter(dataloader)
        second_batch = cast(torch.Tensor, next(iterator))
        close_dataloader_resources(
            dataloader,
            iterator,
            reason=DataloaderCloseReason.EPOCH_EXHAUSTED,
        )

        assert first_batch.tolist() == [0, 1, 2, 3]
        assert second_batch.tolist() == [0, 1, 2, 3]

    def test_close_owner_resources_shutdowns_kept_persistent_iterator(self):
        """Final teardown closes persistent workers kept across epochs."""
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
        close_dataloader_resources(
            dataloader,
            iterator,
            reason=DataloaderCloseReason.EPOCH_EXHAUSTED,
        )

        assert getattr(dataloader, "_iterator", None) is not None

        _close_dataloader_owner_resources(dataloader)

        assert getattr(dataloader, "_iterator", None) is None
        iterator = iter(dataloader)
        second_batch = cast(torch.Tensor, next(iterator))
        close_dataloader_resources(
            dataloader,
            iterator,
            reason=DataloaderCloseReason.TRAINER_TEARDOWN,
        )

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

        prefetch_iter = create_prefetch_iterator(
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
