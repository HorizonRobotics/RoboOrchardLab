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

"""Shared dataset owners for sampling and prepared-iteration tests."""

from robo_orchard_core.utils.config import ClassType
from torch.utils.data import Dataset

from robo_orchard_lab.dataset.robot import DatasetItem

__all__ = ["_ManagedRngItem"]


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
