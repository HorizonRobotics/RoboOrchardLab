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

from accelerate import Accelerator
from accelerate.data_loader import DataLoaderDispatcher
from torch.utils.data import DataLoader

__all__ = ["prepare_data_loader"]


def prepare_data_loader(
    accelerator: Accelerator,
    data_loader: DataLoader,
    **kwargs,
) -> DataLoader:
    """Prepare the dataloader using accelerator.prepare_data_loader.

    This function is a wrapper around accelerator.prepare_data_loader to handle
    the case when the dataset is `IterableDatasetMixin`.

    """

    from robo_orchard_lab.dataset.robot.dataset_ex import IterableDatasetMixin

    dataset = data_loader.dataset
    if (
        isinstance(dataset, IterableDatasetMixin)
        and accelerator.num_processes > 1
    ):
        if dataset.shard_kwargs.shard_strategy is None:
            warnings.warn(
                "The dataset is an iterable dataset and the shard strategy "
                "is not set for multi-process training. This may lead to "
                "unbalanced data loading and potential system hang. "
                "Reset the shard strategy to 'pad_last'. ",
                UserWarning,
            )
            dataset.shard_kwargs.shard_strategy = "pad_last"
        if accelerator.dataloader_config.dispatch_batches is not False:
            warnings.warn(
                "Using IterableDatasetMixin with multi-process training and "
                "dispatch_batches is not set to False will lead to "
                "inefficient data loading! Please set dispatch_batches to "
                "False in the dataloader config. ",
                UserWarning,
            )
        if accelerator.dataloader_config.even_batches is not False:
            warnings.warn(
                "even_batches in accelerator dataloader config is not "
                "supported for IterableDataset. You should set drop_last in "
                "the dataloader to get rid of the last incomplete batch "
                "instead. ",
                UserWarning,
            )
        if accelerator.dataloader_config.split_batches is not False:
            warnings.warn(
                "Using IterableDatasetMixin with multi-process training and "
                "split_batches is not set to False will lead to "
                "inefficient data loading! Please set split_batches to "
                "False in the dataloader config. ",
                UserWarning,
            )
    ret = accelerator.prepare_data_loader(data_loader, **kwargs)

    if (
        isinstance(ret, DataLoaderDispatcher)
        and isinstance(dataset, IterableDatasetMixin)
        and accelerator.num_processes > 1
    ):
        warnings.warn(
            "The prepared dataloader is a DataLoaderDispatcher, which may "
            "have performance issues when used with an IterableDatasetMixin "
            "and multi-process training. Please set dataloader_config "
            "properly to get better performance. ",
            UserWarning,
        )

    return ret
