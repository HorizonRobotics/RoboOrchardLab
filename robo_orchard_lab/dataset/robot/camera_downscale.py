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

"""Utilities for downscaling encoded robot camera columns."""

from __future__ import annotations

import torch
from pydantic import BaseModel, ConfigDict, Field
from robo_orchard_core.datatypes.camera_data import (
    BatchCameraDataEncoded,
    ImageChannelLayout,
)

__all__ = [
    "CameraDownscaleConfig",
    "downscale_camera_data",
]


class CameraDownscaleConfig(BaseModel):
    """Configuration for downscaling one encoded camera column."""

    model_config = ConfigDict(frozen=True)

    downscale: float = Field(gt=0.0, le=1.0)
    """Scale factor applied to image height and width."""

    jpeg_quality: int = Field(default=95, ge=1, le=100)
    """JPEG quality used when re-encoding JPEG/JPG camera images."""

    png_compression: int | None = Field(default=None, ge=0, le=9)
    """PNG compression level. If None, the encoder default is used."""

    def target_hw(self, image_shape: tuple[int, int]) -> tuple[int, int]:
        """Return ``round_min_1`` scaled image shape."""
        height, width = image_shape
        return (
            max(1, round(height * self.downscale)),
            max(1, round(width * self.downscale)),
        )


def downscale_camera_data(
    data: BatchCameraDataEncoded,
    *,
    config: CameraDownscaleConfig,
    is_depth: bool,
    context: str | None = None,
) -> BatchCameraDataEncoded:
    """Downscale one encoded camera batch and preserve camera metadata.

    Args:
        data (BatchCameraDataEncoded): Encoded camera batch to downscale.
        config (CameraDownscaleConfig): Downscale and codec settings.
        is_depth (bool): If True, treat data as PNG uint16 depth and resize
            with nearest-neighbor interpolation. Otherwise resize with
            area interpolation.
        context (str | None, optional): Caller-provided context included in
            decode, encode, and format errors. Default is None.

    Returns:
        BatchCameraDataEncoded: Downscaled encoded camera batch.
    """
    if config.downscale == 1.0:
        return data.model_copy(deep=False)

    _validate_format(data=data, is_depth=is_depth, context=context)
    decoded_data = None
    image_shape = data.image_shape
    if image_shape is None:
        try:
            decoded_data = data.decode()
            image_shape = decoded_data.image_shape
        except Exception as exc:
            raise ValueError(
                _format_error_context(
                    context=context,
                    frame_index=None,
                    format=data.format,
                    reason=str(exc),
                )
            ) from exc
    if image_shape is None:
        raise ValueError(
            _format_error_context(
                context=context,
                frame_index=None,
                format=data.format,
                reason="image_shape is unavailable after decoding.",
            )
        )
    target_hw = config.target_hw(image_shape)
    jpeg_quality = (
        config.jpeg_quality if data.format in ("jpeg", "jpg") else None
    )
    png_compression = config.png_compression if data.format == "png" else None
    try:
        if decoded_data is not None:
            if is_depth and decoded_data.sensor_data.dtype != torch.uint16:
                raise ValueError(
                    "decoded sensor_data dtype must be "
                    f"{torch.uint16}, got {decoded_data.sensor_data.dtype}."
                )
            resized = decoded_data.resize2d(
                target_hw=target_hw,
                inter_mode="nearest" if is_depth else "area",
            )
            return resized.encode(
                format=data.format,
                jpeg_quality=jpeg_quality,
                png_compression=png_compression,
                channel_layout=ImageChannelLayout.HWC,
            )

        return data.resize2d(
            target_hw=target_hw,
            inter_mode="nearest" if is_depth else "area",
            expected_sensor_dtype=torch.uint16 if is_depth else None,
            jpeg_quality=jpeg_quality,
            png_compression=png_compression,
        )
    except Exception as exc:
        raise ValueError(
            _format_error_context(
                context=context,
                frame_index=None,
                format=data.format,
                reason=str(exc),
            )
        ) from exc


def _validate_format(
    *,
    data: BatchCameraDataEncoded,
    is_depth: bool,
    context: str | None,
) -> None:
    if is_depth and data.format != "png":
        raise ValueError(
            _format_error_context(
                context=context,
                frame_index=None,
                format=data.format,
                reason="depth camera data must use PNG encoding",
            )
        )
    if not is_depth and data.format not in ("jpeg", "jpg", "png"):
        raise ValueError(
            _format_error_context(
                context=context,
                frame_index=None,
                format=data.format,
                reason="unsupported camera image format",
            )
        )


def _format_error_context(
    *,
    context: str | None,
    frame_index: int | None,
    format: str,
    reason: str,
) -> str:
    parts = []
    if context:
        parts.append(context)
    if frame_index is not None:
        parts.append(f"frame={frame_index}")
    parts.append(f"format={format}")
    parts.append(reason)
    return "; ".join(parts)
