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

import pytest
import torch
from robo_orchard_core.utils.config import ClassType
from torch.utils.data import DataLoader, Dataset

from robo_orchard_lab.dataset.robot import (
    BatchLoaderConfig,
    DatasetItem,
    IterableDatasetMixin,
    IterableWithLenDataset,
)
from robo_orchard_lab.dataset.robot.dataset_ex import DictIterableDataset


class ArrayDataset(Dataset):
    def __init__(self, data: list):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ArrayDatasetItem(DatasetItem[ArrayDataset]):
    class_type: ClassType[ArrayDataset] = ArrayDataset

    data: list

    def get_dataset_row_num(self) -> int:
        return len(self.data)

    def _create_dataset(self) -> ArrayDataset:
        return ArrayDataset(self.data)


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
            persistent_workers=num_workers > 0,
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

        def collate_fn_unbatch(data):
            return data[0]

        dataloader = DataLoader(
            dataset,
            batch_size=1,  # batch size is 1 since the dataset already returns
            num_workers=num_workers,
            drop_last=drop_last,
            persistent_workers=num_workers > 0,
            collate_fn=collate_fn_unbatch,
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
            persistent_workers=num_workers > 0,
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

        def collate_fn_unbatch(data):
            return data[0]

        dataloader = DataLoader(
            dataset,
            batch_size=1,  # batch size is 1 since the dataset already returns
            num_workers=num_workers,
            drop_last=drop_last,
            persistent_workers=num_workers > 0,
            collate_fn=collate_fn_unbatch,
        )
        self._check_dataloader_item_consistency(
            dataloader=dataloader,
            dataset=dataset,
            need_sort=False,
        )


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
            persistent_workers=num_workers > 0,
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

        def collate_fn_unbatch(data):
            return data[0]

        dataloader = DataLoader(
            dataset,
            batch_size=1,  # batch size is 1 since the dataset already returns
            num_workers=num_workers,
            drop_last=drop_last,
            persistent_workers=num_workers > 0,
            collate_fn=collate_fn_unbatch,
        )
        self._check_dataloader_item_consistency(
            dataloader=dataloader,
            dataset=dataset,
            need_sort=False,
        )

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
            persistent_workers=num_workers > 0,
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

        def collate_fn_unbatch(data):
            return data[0]

        dataloader = DataLoader(
            dataset,
            batch_size=1,  # batch size is 1 since the dataset already returns
            num_workers=num_workers,
            drop_last=drop_last,
            persistent_workers=num_workers > 0,
            collate_fn=collate_fn_unbatch,
        )
        self._check_dataloader_total_batch_consistency(
            dataloader=dataloader,
            dataset=dataset,
            batch_size=batch_size,
            drop_last=drop_last,
        )
