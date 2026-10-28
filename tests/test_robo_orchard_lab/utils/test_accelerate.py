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

import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from accelerate.utils.dataclasses import DataLoaderConfiguration
from robo_orchard_core.utils.config import ClassType
from torch.utils.data import DataLoader, Dataset

import robo_orchard_lab.utils.accelerate as accelerate_utils
from robo_orchard_lab.dataset.robot import (
    DatasetItem,
    DictIterableDataset,
    IterableWithLenDataset,
)

prepare_data_loader = accelerate_utils.prepare_data_loader


class TestPrepareDataLoader:
    class _ArrayDataset(Dataset):
        def __init__(self, data: list[int]):
            self.data = data

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx: int) -> int:
            return self.data[idx]

    @pytest.fixture()
    def iterable_dataloader(self) -> DataLoader:
        dataset = IterableWithLenDataset(self._ArrayDataset(list(range(10))))
        return DataLoader(dataset, batch_size=2)

    def test_resets_unsupported_iterable_flags(
        self,
        iterable_dataloader: DataLoader,
    ):
        accelerator = MagicMock()
        accelerator.num_processes = 2
        accelerator.dataloader_config = DataLoaderConfiguration(
            dispatch_batches=False,
            even_batches=True,
            split_batches=True,
        )
        accelerator.dataloader_config.dispatch_batches = None  # type: ignore[assignment]
        accelerator.prepare_data_loader = MagicMock(
            return_value=iterable_dataloader
        )

        with pytest.warns(UserWarning) as warning_records:
            ret = prepare_data_loader(
                accelerator,
                iterable_dataloader,
                put_on_device=False,
            )

        assert ret is iterable_dataloader
        accelerator.prepare_data_loader.assert_called_once_with(
            iterable_dataloader,
            put_on_device=False,
        )

        dataset = iterable_dataloader.dataset
        assert isinstance(dataset, IterableWithLenDataset)
        assert dataset.shard_kwargs.shard_strategy == "pad_last"
        assert accelerator.dataloader_config.dispatch_batches is False
        assert accelerator.dataloader_config.even_batches is False
        assert accelerator.dataloader_config.split_batches is False

        warning_messages = [str(record.message) for record in warning_records]
        assert len(warning_messages) == 4
        assert any(
            "Reset the shard strategy to 'pad_last'" in msg
            for msg in warning_messages
        )
        assert any(
            "dispatch_batches" in msg and "reset it to False" in msg
            for msg in warning_messages
        )
        assert any(
            "even_batches" in msg and "reset it to False" in msg
            for msg in warning_messages
        )
        assert any(
            "split_batches" in msg and "reset it to False" in msg
            for msg in warning_messages
        )

    def test_keeps_flags_for_single_process(
        self,
        iterable_dataloader: DataLoader,
    ):
        accelerator = MagicMock()
        accelerator.num_processes = 1
        accelerator.dataloader_config = DataLoaderConfiguration(
            dispatch_batches=True,
            even_batches=True,
            split_batches=True,
        )
        accelerator.prepare_data_loader = MagicMock(
            return_value=iterable_dataloader
        )

        with warnings.catch_warnings(record=True) as warning_records:
            warnings.simplefilter("always")
            ret = prepare_data_loader(accelerator, iterable_dataloader)

        assert ret is iterable_dataloader
        dataset = iterable_dataloader.dataset
        assert isinstance(dataset, IterableWithLenDataset)
        assert dataset.shard_kwargs.shard_strategy is None
        assert accelerator.dataloader_config.dispatch_batches is True
        assert accelerator.dataloader_config.even_batches is True
        assert accelerator.dataloader_config.split_batches is True
        assert len(warning_records) == 0

    def test_warns_when_prepare_returns_iterable_dataset_shard(
        self,
        iterable_dataloader: DataLoader,
        monkeypatch: pytest.MonkeyPatch,
    ):
        class FakeIterableDatasetShard:
            pass

        class FakeDataLoaderShard:
            def __init__(self, dataset: object):
                self.base_dataloader = SimpleNamespace(dataset=dataset)

        monkeypatch.setattr(
            accelerate_utils,
            "DataLoaderShard",
            FakeDataLoaderShard,
        )
        monkeypatch.setattr(
            accelerate_utils,
            "IterableDatasetShard",
            FakeIterableDatasetShard,
        )

        dataset = iterable_dataloader.dataset
        assert isinstance(dataset, IterableWithLenDataset)
        dataset.shard_kwargs.shard_strategy = "pad_last"

        accelerator = MagicMock()
        accelerator.num_processes = 2
        accelerator.dataloader_config = DataLoaderConfiguration(
            dispatch_batches=False,
            even_batches=False,
            split_batches=False,
        )
        ret = FakeDataLoaderShard(FakeIterableDatasetShard())
        accelerator.prepare_data_loader = MagicMock(return_value=ret)

        with pytest.warns(UserWarning) as warning_records:
            prepared = prepare_data_loader(
                accelerator,
                iterable_dataloader,
                put_on_device=False,
            )

        assert prepared is ret
        assert any(
            "IterableDatasetShard" in str(record.message)
            for record in warning_records
        )

    def test_warns_when_prepare_returns_dataloader_dispatcher(
        self,
        iterable_dataloader: DataLoader,
        monkeypatch: pytest.MonkeyPatch,
    ):
        class FakeDataLoaderDispatcher:
            pass

        monkeypatch.setattr(
            accelerate_utils,
            "DataLoaderDispatcher",
            FakeDataLoaderDispatcher,
        )

        dataset = iterable_dataloader.dataset
        assert isinstance(dataset, IterableWithLenDataset)
        dataset.shard_kwargs.shard_strategy = "pad_last"

        accelerator = MagicMock()
        accelerator.num_processes = 2
        accelerator.dataloader_config = DataLoaderConfiguration(
            dispatch_batches=False,
            even_batches=False,
            split_batches=False,
        )
        ret = FakeDataLoaderDispatcher()
        accelerator.prepare_data_loader = MagicMock(return_value=ret)

        with pytest.warns(UserWarning) as warning_records:
            prepared = prepare_data_loader(
                accelerator,
                iterable_dataloader,
                put_on_device=False,
            )

        assert prepared is ret
        assert any(
            "DataLoaderDispatcher" in str(record.message)
            for record in warning_records
        )

    def test_dict_iterable_dataset_n_shards_exceeds_process_count(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        class ArrayDatasetItem(
            DatasetItem[TestPrepareDataLoader._ArrayDataset]
        ):
            class_type: ClassType[TestPrepareDataLoader._ArrayDataset] = (
                TestPrepareDataLoader._ArrayDataset
            )

            data: list[int]

            def get_dataset_row_num(self) -> int:
                return len(self.data)

            def _create_dataset(self) -> TestPrepareDataLoader._ArrayDataset:
                return TestPrepareDataLoader._ArrayDataset(self.data)

        monkeypatch.setattr(
            "accelerate.state.AcceleratorState",
            lambda: SimpleNamespace(num_processes=4),
        )

        dataset = DictIterableDataset(
            [
                ArrayDatasetItem(data=[1]),
            ]
        )

        assert dataset.n_shards == 5


class _ManagedRngDataset(Dataset):
    def __init__(self, data: list[int]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class _ManagedRngItem(DatasetItem[_ManagedRngDataset]):
    class_type: ClassType[_ManagedRngDataset] = _ManagedRngDataset
    data: list[int]

    def get_dataset_row_num(self):
        return len(self.data)

    def _create_dataset(self):
        return _ManagedRngDataset(self.data)


@pytest.mark.parametrize("already_common", [False, True])
def test_bind_fallback_preserves_caller_epoch_and_generator(already_common):
    from accelerate.data_loader import DataLoaderShard, IterableDatasetShard

    source = DictIterableDataset(
        [_ManagedRngItem(data=list(range(24)))], shuffle=True, seed=42
    )
    source._sampling_rank = None if already_common else 0
    expected = list(source)
    wrapper = IterableDatasetShard(
        source, batch_size=1, num_processes=2, process_index=0
    )
    prepared = DataLoaderShard(wrapper, batch_size=1)
    accelerator = SimpleNamespace(process_index=0)
    assert accelerate_utils.bind_prepared_data_loader_sampling(
        accelerator, prepared
    )
    owned = prepared.dataset.dataset
    assert owned is not source
    assert owned._sampling_rank is None
    prepared.set_epoch(3)
    assert list(source) == expected
    assert source._sampling_epoch.value == 0
    assert owned._sampling_epoch.value == 3
    rows = list(owned)
    assert accelerate_utils.bind_prepared_data_loader_sampling(
        accelerator, prepared
    )
    assert prepared.dataset.dataset is not owned
    assert list(prepared.dataset.dataset) == rows
    assert prepared.dataset.dataset._sampling_epoch.value == 3


def test_dispatcher_binding_leaves_shared_source_untouched():
    from accelerate import Accelerator
    from accelerate.data_loader import DataLoaderDispatcher

    source = DictIterableDataset(
        [_ManagedRngItem(data=list(range(24)))], shuffle=True, seed=42
    )
    loader = DataLoader(source, batch_size=2)
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(dispatch_batches=True),
    )
    prepared = accelerator.prepare_data_loader(loader)
    assert isinstance(prepared, DataLoaderDispatcher)
    assert prepared.dataset is source
    expected = list(source)
    state = source._generator.get_state().clone()
    assert (
        accelerate_utils.bind_prepared_data_loader_sampling(
            accelerator, prepared
        )
        is False
    )
    assert prepared.dataset is source
    assert source._sampling_seed == 42
    assert source._generator.get_state().equal(state)
    assert list(source) == expected


def test_fallback_owner_can_be_safely_reused_in_native_loader():
    from accelerate.data_loader import DataLoaderShard, IterableDatasetShard

    from robo_orchard_lab.dataset.robot.dataset_ex import (
        DataLoader as RODataLoader,
    )

    source = DictIterableDataset(
        [_ManagedRngItem(data=list(range(24)))], shuffle=True, seed=42
    )
    fallback = DataLoaderShard(
        IterableDatasetShard(
            source, batch_size=1, num_processes=2, process_index=0
        ),
        batch_size=1,
    )
    accelerator = SimpleNamespace(process_index=0)
    assert accelerate_utils.bind_prepared_data_loader_sampling(
        accelerator, fallback
    )
    common_owner = fallback.dataset.dataset
    assert common_owner._sampling_rank is None

    loader = RODataLoader(common_owner, batch_size=1)
    native = DataLoaderShard(loader.dataset, batch_size=1)
    assert accelerate_utils.bind_prepared_data_loader_sampling(
        accelerator, native
    )
    fresh = DictIterableDataset(
        [_ManagedRngItem(data=list(range(24)))], shuffle=True, seed=42
    )
    assert list(native.dataset) == list(fresh)
    assert common_owner._sampling_rank is None

    shared_torch_loader = DataLoader(common_owner, batch_size=1)
    shared_native = DataLoaderShard(shared_torch_loader.dataset, batch_size=1)
    with pytest.raises(ValueError, match="without a new owner"):
        accelerate_utils.bind_prepared_data_loader_sampling(
            accelerator, shared_native
        )
    assert common_owner._sampling_rank is None


def test_fallback_rebinds_a_preexisting_static_shard_across_ranks():
    """Static dataset sharding cannot be mistaken for final native mode."""
    from collections import Counter

    from accelerate import Accelerator
    from accelerate.data_loader import DataLoaderShard, IterableDatasetShard

    Accelerator()
    sources = [
        DictIterableDataset(
            [_ManagedRngItem(data=list(range(24)))],
            shuffle=True,
            seed=42,
        ).shard(num_shards=2, index=0)
        for rank in (0, 1)
    ]
    expected = list(sources[0])
    prepared = [
        DataLoaderShard(
            IterableDatasetShard(
                sources[rank],
                batch_size=1,
                num_processes=2,
                process_index=rank,
            ),
            batch_size=1,
        )
        for rank in (0, 1)
    ]
    for rank, loader in enumerate(prepared):
        assert accelerate_utils.bind_prepared_data_loader_sampling(
            SimpleNamespace(process_index=rank), loader
        )
    rows = [
        int(value)
        for loader in prepared
        for batch in loader
        for value in batch
    ]
    assert Counter(rows) == Counter(expected)
    assert all(source._sampling_rank == 0 for source in sources)
