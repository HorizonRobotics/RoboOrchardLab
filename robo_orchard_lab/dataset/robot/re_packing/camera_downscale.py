# Project RoboOrchard
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
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

"""RODataset repack transform for downscaling encoded camera columns."""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Collection, Sequence

import datasets as hg_datasets

from robo_orchard_lab.dataset.datatypes import BatchCameraDataEncodedFeature
from robo_orchard_lab.dataset.robot.camera_downscale import (
    CameraDownscaleConfig,
    downscale_camera_data,
)
from robo_orchard_lab.dataset.robot.columns import PreservedColumnsKeys
from robo_orchard_lab.dataset.robot.dataset import RODataset
from robo_orchard_lab.dataset.robot.packaging import (
    DataFrame,
)
from robo_orchard_lab.dataset.robot.re_packing import repack_dataset
from robo_orchard_lab.dataset.robot.re_packing.contracts import (
    IdentityRODatasetRepackTransform,
    RODatasetFrameContext,
    RODatasetRepackContext,
)

__all__ = ["downscale_ro_dataset", "CameraDownscaleTransform"]


def downscale_ro_dataset(
    source_dataset_path: str,
    target_dataset_path: str,
    *,
    downscale: float,
    columns: list[str] | None = None,
    depth_columns: list[str] | None = None,
    jpeg_quality: int = 95,
    png_compression: int | None = None,
    max_shard_size: str | int = "8GB",
    writer_batch_size: int = 4096,
    force_overwrite: bool = False,
) -> None:
    """Write a RODataset copy with encoded camera columns downscaled.

    Args:
        source_dataset_path (str): Existing RODataset path to read.
        target_dataset_path (str): New RODataset path to write.
        downscale (float): Scale factor applied to selected camera columns.
        columns (list[str] | None, optional): Encoded camera columns to
            process. If None, all encoded camera columns are processed.
            Default is None.
        depth_columns (list[str] | None, optional): Selected columns that
            should be decoded as uint16 PNG depth and resized with nearest
            interpolation. If None, ``*_depth`` columns are treated as depth.
            Default is None.
        jpeg_quality (int, optional): JPEG quality for re-encoded RGB/JPEG
            columns. Default is 95.
        png_compression (int | None, optional): PNG compression level for
            re-encoded PNG columns. Default is None.
        max_shard_size (str | int, optional): Maximum Hugging Face shard size.
            Default is "8GB".
        writer_batch_size (int, optional): Hugging Face writer batch size.
            Default is 4096.
        force_overwrite (bool, optional): Whether to overwrite the target
            dataset path. Default is False.
    """

    source_dataset = RODataset(source_dataset_path)
    config = CameraDownscaleConfig(
        downscale=downscale,
        jpeg_quality=jpeg_quality,
        png_compression=png_compression,
    )
    repack_dataset(
        source_dataset=source_dataset,
        target_path=target_dataset_path,
        transforms=[
            CameraDownscaleTransform(
                config=config,
                camera_columns=columns,
                depth_columns=depth_columns,
            )
        ],
        writer_batch_size=writer_batch_size,
        max_shard_size=max_shard_size,
        force_overwrite=force_overwrite,
    )


class CameraDownscaleTransform(IdentityRODatasetRepackTransform):
    """Downscale encoded camera columns during RODataset repack."""

    def __init__(
        self,
        *,
        config: CameraDownscaleConfig,
        camera_columns: Sequence[str] | None = None,
        depth_columns: Sequence[str] | None = None,
    ) -> None:
        self.config = config
        self.camera_columns = (
            list(camera_columns) if camera_columns is not None else None
        )
        self.depth_columns = (
            list(depth_columns) if depth_columns is not None else None
        )
        self._resolved_columns: _ResolvedCameraDownscaleColumns | None = None

    def required_columns(
        self,
        context: RODatasetRepackContext,
    ) -> Collection[str]:
        return self._resolve_columns(context.target_features).selected_columns

    def prepare(self, context: RODatasetRepackContext) -> None:
        self._resolved_columns = self._resolve_columns(context.target_features)

    def update_features(
        self,
        features: hg_datasets.Features,
    ) -> hg_datasets.Features:
        return features

    def transform_frame(
        self,
        frame: DataFrame,
        context: RODatasetFrameContext,
    ) -> DataFrame:
        if self.config.downscale == 1.0:
            return frame

        resolved_columns = self._require_resolved_columns()
        depth_column_set = set(resolved_columns.depth_columns)
        features = frame.features.copy()
        source_frame_index = context.source_frame_row.get(
            "frame_index", context.frame_index
        )
        for column in resolved_columns.selected_columns:
            features[column] = downscale_camera_data(
                features[column],
                config=self.config,
                is_depth=column in depth_column_set,
                context=(
                    f"episode={context.episode_index} column={column} "
                    f"frame={source_frame_index}"
                ),
            )
        return replace(frame, features=features)

    def _resolve_columns(
        self,
        features: hg_datasets.Features,
    ) -> _ResolvedCameraDownscaleColumns:
        return _resolve_camera_downscale_columns(
            features,
            columns=self.camera_columns,
            depth_columns=self.depth_columns,
        )

    def _require_resolved_columns(self) -> _ResolvedCameraDownscaleColumns:
        if self._resolved_columns is None:
            raise RuntimeError(
                "CameraDownscaleTransform.prepare() must run before "
                "episode or frame transforms."
            )
        return self._resolved_columns


@dataclass(frozen=True, slots=True)
class _ResolvedCameraDownscaleColumns:
    image_columns: list[str]
    depth_columns: list[str]

    @property
    def selected_columns(self) -> list[str]:
        return self.image_columns + self.depth_columns


def _resolve_camera_downscale_columns(
    features: hg_datasets.Features,
    *,
    columns: Sequence[str] | None,
    depth_columns: Sequence[str] | None,
) -> _ResolvedCameraDownscaleColumns:
    camera_columns = [
        name
        for name, feature in features.items()
        if name not in PreservedColumnsKeys
        and isinstance(feature, BatchCameraDataEncodedFeature)
    ]
    if columns is None:
        selected_columns = camera_columns
    else:
        selected_columns = list(columns)
        _validate_encoded_camera_columns(
            features,
            selected_columns,
            label="columns",
        )

    if not selected_columns:
        raise ValueError("No encoded camera columns selected for downscale.")

    if depth_columns is None:
        depth_column_set = {
            column for column in selected_columns if _is_depth_column(column)
        }
    else:
        _validate_encoded_camera_columns(
            features,
            depth_columns,
            label="depth_columns",
        )
        selected_column_set = set(selected_columns)
        outside_selection = [
            column
            for column in depth_columns
            if column not in selected_column_set
        ]
        if outside_selection:
            raise ValueError(
                "depth_columns must be included in columns; got "
                f"{outside_selection!r} outside selected columns."
            )
        depth_column_set = set(depth_columns)

    image_columns = [
        column for column in selected_columns if column not in depth_column_set
    ]
    resolved_depth_columns = [
        column for column in selected_columns if column in depth_column_set
    ]
    return _ResolvedCameraDownscaleColumns(
        image_columns=image_columns,
        depth_columns=resolved_depth_columns,
    )


def _validate_encoded_camera_columns(
    features: hg_datasets.Features,
    columns: Sequence[str],
    *,
    label: str,
) -> None:
    for column in columns:
        if column not in features:
            raise ValueError(f"{label} contains unknown column {column!r}.")
        if not isinstance(features[column], BatchCameraDataEncodedFeature):
            raise ValueError(
                f"{label} column {column!r} is not an encoded camera column."
            )


def _is_depth_column(column: str) -> bool:
    return column.endswith("_depth") or column.endswith("_camera_depth")
