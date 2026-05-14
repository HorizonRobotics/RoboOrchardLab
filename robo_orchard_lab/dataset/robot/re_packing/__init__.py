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
from typing import Iterable, Sequence

from robo_orchard_lab.dataset.robot.dataset import RODataset
from robo_orchard_lab.dataset.robot.re_packing.contracts import (
    IdentityRODatasetRepackTransform,
    RODatasetEpisodeContext,
    RODatasetEpisodeSelection,
    RODatasetFrameContext,
    RODatasetRepackContext,
    RODatasetRepackTransform,
)
from robo_orchard_lab.dataset.robot.re_packing.helpers import (
    DefaultRePackingEpisodeHelper,
    RePackingDatasetHelper as RePackingDatasetHelper,
    RePackingEpisodeType,
    helper_repack_dataset,
)
from robo_orchard_lab.dataset.robot.re_packing.transform import (
    transform_repack_dataset,
)

__all__ = [
    "repack_dataset",
    "DefaultRePackingEpisodeHelper",
    "RODatasetRepackTransform",
    "IdentityRODatasetRepackTransform",
    "RODatasetRepackContext",
    "RODatasetEpisodeSelection",
    "RODatasetEpisodeContext",
    "RODatasetFrameContext",
]


def repack_dataset(
    source_dataset: RODataset,
    target_path: str,
    frame_indices: Iterable[int] | None = None,
    columns: str | list[str] | None = None,
    writer_batch_size: int = 1024,
    max_shard_size: str | int = "8GB",
    force_overwrite: bool = False,
    packing_impl: type[RePackingEpisodeType] = DefaultRePackingEpisodeHelper,
    transforms: Sequence[RODatasetRepackTransform] | None = None,
    fail_fast: bool | None = None,
):
    """Re-package a RoboOrchard dataset with selected frames for each episode.

    Args:
        source_dataset (RODataset): The source dataset to re-package from.
        target_path (str): The path to save the re-packaged dataset.
        frame_indices (Iterable[int] | None): An iterable of frame indices
            to include in the re-packaged dataset. Frames from the same
            episode should be grouped together! If None, all frames will be
            included. Default is None.
        columns (str | list[str] | None): The columns to include in the
            re-packaged dataset. If None, all columns will be included. If
            a string is provided, it will be treated as a single column name.
            Default is None.
        writer_batch_size (int): The batch size for writing the arrow file.
            This may affect the performance of packaging or reading the
            dataset later. Default is 1024.
        max_shard_size (str | int | None): The maximum size of each shard.
            If None, no sharding will be applied. This can be a string
            like '10GB' or an integer representing the size in bytes.
            Default is '8GB'.
        force_overwrite (bool): Whether to overwrite the target path if it
            already exists. In transform mode, an existing target is replaced
            only after the transformed dataset is fully packaged. Default is
            False.
        packing_impl (type[RePackingEpisodeType]): The compatibility helper
            class for packaging each episode. This path is used only when
            ``transforms`` is None. Default is
            `DefaultRePackingEpisodeHelper`.
        transforms (Sequence[RODatasetRepackTransform] | None, optional):
            Transform pipeline to apply. If None, uses the compatibility
            ``packing_impl`` path. Any non-None value, including an empty
            sequence, enables transform mode. Default is None.
        fail_fast (bool | None, optional): Failure policy for the
            compatibility ``packing_impl`` path. Transform mode is always
            fail-fast and rejects False. Default is None.

    """
    if columns is not None:
        dataset = source_dataset.select_columns(columns, include_index=True)
    else:
        dataset = source_dataset

    if transforms is None:
        helper_repack_dataset(
            dataset=dataset,
            target_path=target_path,
            frame_indices=frame_indices,
            writer_batch_size=writer_batch_size,
            max_shard_size=max_shard_size,
            force_overwrite=force_overwrite,
            packing_impl=packing_impl,
            fail_fast=bool(fail_fast),
        )
        return

    if packing_impl is not DefaultRePackingEpisodeHelper:
        raise ValueError("packing_impl cannot be combined with transforms.")
    if fail_fast is False:
        raise ValueError("transform mode is always fail-fast.")

    transform_repack_dataset(
        dataset=dataset,
        target_path=target_path,
        frame_indices=frame_indices,
        writer_batch_size=writer_batch_size,
        max_shard_size=max_shard_size,
        force_overwrite=force_overwrite,
        transforms=list(transforms),
    )
