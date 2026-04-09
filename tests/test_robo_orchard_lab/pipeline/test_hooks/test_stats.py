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

import time
from typing import Any, Iterator, cast
from unittest.mock import MagicMock

import pytest
import torch
from accelerate.data_loader import IterableDatasetShard
from torch.utils.data import Dataset

from robo_orchard_lab.dataset.robot import (
    BatchLoaderConfig,
    DataLoader,
    IterableWithLenDataset,
)
from robo_orchard_lab.pipeline.hooks import StatsMonitorConfig
from robo_orchard_lab.pipeline.hooks.mixin import PipelineHookArgs


class ArrayDataset(Dataset):
    def __init__(self, data: list[int]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> int:
        return self.data[idx]


@pytest.fixture(scope="function")
def mock_accelerator():
    """Fixture to create a mock Accelerator."""
    accelerator = MagicMock()
    accelerator.num_processes = 4
    accelerator.is_main_process = True
    return accelerator


@pytest.fixture(scope="function")
def mock_dataloader():
    """Fixture to create a mock DataLoader."""
    return torch.utils.data.DataLoader(
        ArrayDataset(list(range(3200))),
        batch_size=32,
    )


@pytest.fixture(scope="function")
def mock_hook_args(mock_accelerator, mock_dataloader):
    """Fixture to create hook args backed by a real dataloader."""
    optimizer = torch.optim.Adam(
        params=[
            {"params": torch.nn.Parameter(torch.randn(10)), "lr": 0.01},
            {"params": torch.nn.Parameter(torch.randn(5)), "lr": 0.02},
        ],
        lr=0.0,
    )

    return PipelineHookArgs(
        accelerator=mock_accelerator,
        dataloader=mock_dataloader,
        epoch_id=0,
        step_id=0,
        global_step_id=0,
        start_step=0,
        start_epoch=0,
        max_step=1000,
        max_epoch=10,
        optimizer=cast(Any, optimizer),
        model_outputs=None,
        reduce_loss=torch.tensor(0.0),
    )


def test_stats_monitor_initialization():
    """Test StatsMonitor initialization."""
    monitor = StatsMonitorConfig(
        batch_size=32, steps_per_epoch=100, step_log_freq=10, epoch_log_freq=1
    )()
    assert monitor.batch_size == 32
    assert monitor.steps_per_epoch == 100
    assert monitor.step_log_freq == 10
    assert monitor.epoch_log_freq == 1


def test_estimate_data_stats(mock_accelerator, mock_dataloader):
    """Test data statistics estimation."""
    monitor = StatsMonitorConfig()()
    monitor._estimate_data_stats(mock_accelerator, mock_dataloader)
    assert monitor.batch_size == 32
    assert monitor.total_batch_size == 32 * 4
    assert monitor.steps_per_epoch == 100


def test_estimate_data_stats_uses_dataset_side_batch_size(
    mock_accelerator,
):
    """Dataset-side batching should override outer DataLoader metadata."""
    dataset = IterableWithLenDataset(ArrayDataset(list(range(16))))
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        num_workers=0,
        use_dataset_side_batching=True,
    )

    assert dataloader.batch_size == 1

    monitor = StatsMonitorConfig()()
    monitor._estimate_data_stats(mock_accelerator, dataloader)

    assert monitor.batch_size == 8
    assert monitor.total_batch_size == 8 * 4
    assert monitor.steps_per_epoch == 2


def test_estimate_data_stats_prefers_dataset_side_batch_size_for_shard(
    mock_accelerator,
    monkeypatch: pytest.MonkeyPatch,
):
    """Prepared shard wrappers should not trust the outer shard batch size."""

    class FakeDataLoaderShard:
        def __init__(self, dataset, total_batch_size: int, batch_size: int):
            self.dataset = dataset
            self.total_batch_size = total_batch_size
            self.batch_size = batch_size

        def __len__(self) -> int:
            return 7

        def __iter__(self) -> Iterator[int]:
            return iter(())

    monkeypatch.setattr(
        "robo_orchard_lab.pipeline.hooks.stats.DataLoaderShard",
        FakeDataLoaderShard,
    )

    dataset = IterableWithLenDataset(
        ArrayDataset(list(range(32))),
        batch_loader_kwargs=BatchLoaderConfig(batch_size=8, drop_last=False),
    )
    sharded_dataset = IterableDatasetShard(
        dataset,
        batch_size=1,
        drop_last=False,
        num_processes=4,
        process_index=0,
        split_batches=False,
    )
    dataloader = FakeDataLoaderShard(
        dataset=sharded_dataset,
        total_batch_size=4,
        batch_size=1,
    )

    monitor = StatsMonitorConfig()()
    monitor._estimate_data_stats(mock_accelerator, dataloader)

    assert monitor.batch_size == 8
    assert monitor.total_batch_size == 8 * 4
    assert monitor.steps_per_epoch == 7


def test_estimate_data_stats_rejects_multi_prepared_dataloader(
    mock_accelerator,
    monkeypatch: pytest.MonkeyPatch,
):
    """StatsMonitor should reject iterable dataloaders prepared twice."""

    class FakeDataLoaderShard:
        def __init__(self, dataset, total_batch_size: int, batch_size: int):
            self.dataset = dataset
            self.total_batch_size = total_batch_size
            self.batch_size = batch_size

        def __len__(self) -> int:
            return 7

        def __iter__(self) -> Iterator[int]:
            return iter(())

    monkeypatch.setattr(
        "robo_orchard_lab.pipeline.hooks.stats.DataLoaderShard",
        FakeDataLoaderShard,
    )

    dataset = IterableWithLenDataset(
        ArrayDataset(list(range(32))),
        batch_loader_kwargs=BatchLoaderConfig(batch_size=8, drop_last=False),
    )
    prepared_once_dataset = IterableDatasetShard(
        dataset,
        batch_size=1,
        drop_last=False,
        num_processes=4,
        process_index=0,
        split_batches=False,
    )
    prepared_twice_dataset = IterableDatasetShard(
        prepared_once_dataset,
        batch_size=1,
        drop_last=False,
        num_processes=4,
        process_index=0,
        split_batches=False,
    )
    dataloader = FakeDataLoaderShard(
        dataset=prepared_twice_dataset,
        total_batch_size=4,
        batch_size=1,
    )

    monitor = StatsMonitorConfig()()

    with pytest.raises(ValueError, match="prepared more than once"):
        monitor._estimate_data_stats(mock_accelerator, dataloader)


def test_estimate_remaining_time():
    """Test remaining time estimation."""
    monitor = StatsMonitorConfig()()
    elapsed_time = 60  # 60 seconds
    current_step = 50
    current_epoch = 1
    start_step = 0
    start_epoch = 0
    max_step = 1000
    max_epoch = 10
    steps_per_epoch = 100

    remaining_time = monitor._estimate_remaining_time(
        elapsed_time,
        current_step,
        current_epoch,
        start_step,
        start_epoch,
        max_step,
        max_epoch,
        steps_per_epoch,
    )
    assert remaining_time is not None and remaining_time > 0


def test_on_step_end(mocker, mock_hook_args):
    """Test the on_step_end method."""
    mock_logger = mocker.patch("robo_orchard_lab.pipeline.hooks.stats.logger")

    monitor = StatsMonitorConfig(
        batch_size=32, steps_per_epoch=100, step_log_freq=10
    )()
    monitor._start_time = time.time() - 60  # Simulate 1 minute elapsed
    monitor.total_batch_size = 32 * 4
    monitor._step_start_time = (
        time.time() - 0.5
    )  # Simulate 0.5 seconds per step

    mock_hook_args.global_step_id = 10  # Step 10 not reached

    with monitor.begin("on_step", mock_hook_args):
        pass
    mock_logger.info.assert_not_called()

    mock_hook_args.global_step_id = 9
    with monitor.begin("on_step", mock_hook_args):
        pass
    mock_logger.info.assert_called_once()


def test_on_epoch_end(mocker, mock_hook_args):
    """Test the on_epoch_end method."""
    mock_logger = mocker.patch("robo_orchard_lab.pipeline.hooks.stats.logger")

    monitor = StatsMonitorConfig(
        batch_size=32, steps_per_epoch=100, epoch_log_freq=1
    )()

    with monitor.begin("on_epoch", mock_hook_args) as monitor_hook_args:
        monitor._start_time = time.time() - 300  # Simulate 5 minutes elapsed
        monitor.total_batch_size = 32 * 4
        monitor._epoch_start_time = time.time() - 60  # Simulate 1-minute epoch

        monitor_hook_args.epoch_id = 0
        monitor_hook_args.global_step_id = 10

        pass

    mock_logger.info.assert_called_once()
