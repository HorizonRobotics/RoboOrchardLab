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

"""Shared in-memory datasets and assertions for DataLoader tests."""

import multiprocessing as mp
import os

import pytest
import torch
from robo_orchard_core.utils.config import ClassType
from torch.utils.data import (
    Dataset,
)

from robo_orchard_lab.dataset.robot import (
    DataLoader,
    DatasetItem,
    IterableDatasetMixin,
)

__all__ = [
    "ArrayDataset",
    "BatchedArrayDataset",
    "ArrayDatasetItem",
    "_get_dataloader_multiprocessing_context",
    "TestIterableDatasetMixin",
    "isolated_accelerate_state",
]


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


@pytest.fixture
def isolated_accelerate_state(monkeypatch):
    """Keep CPU prepare independent of earlier Accelerator instances."""
    from accelerate.state import AcceleratorState, GradientState, PartialState

    for state in (AcceleratorState, PartialState, GradientState):
        monkeypatch.setattr(state, "_shared_state", {})
