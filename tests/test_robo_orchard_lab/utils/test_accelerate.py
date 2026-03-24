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
from unittest.mock import MagicMock

import pytest
from accelerate.utils.dataclasses import DataLoaderConfiguration
from torch.utils.data import DataLoader, Dataset

from robo_orchard_lab.dataset.robot import IterableWithLenDataset
from robo_orchard_lab.utils.accelerate import prepare_data_loader


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
            dispatch_batches=None,
            even_batches=True,
            split_batches=True,
        )
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
