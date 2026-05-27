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

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
from foxglove_schemas_protobuf.RawImage_pb2 import RawImage
from google.protobuf.timestamp import from_seconds
from google.protobuf.timestamp_pb2 import Timestamp

from robo_orchard_lab.dataset.experimental.mcap.msg_converter.base import (
    MessageConverterConfig,
    MessageConverterStateless,
)

__all__ = [
    "NumpyImageMsg",
    "Numpy2CompressedImage",
    "Numpy2CompressedImageConfig",
    "CompressedImage2Numpy",
    "CompressedImage2NumpyConfig",
    "RawImage2CompressedImage",
    "RawImage2CompressedImageConfig",
]


@dataclass
class NumpyImageMsg:
    data: np.ndarray
    """Message class for storing numpy image data."""

    target_format: Literal["jpeg", "png"]
    """Target format for the image, e.g., 'jpg' or 'png'."""

    frame_id: str = ""
    """Frame ID for the image."""

    timestamp: Timestamp | None = None
    """Timestamp for the image, defaults to zero if not provided."""


class Numpy2CompressedImage(
    MessageConverterStateless[NumpyImageMsg, CompressedImage]
):
    """Convert numpy array to CompressedImage.

    Args:
        cfg (Numpy2CompressedImageConfig | None, optional): Codec quality
            settings. Defaults to ``Numpy2CompressedImageConfig()`` when None.
    """

    def __init__(self, cfg: Numpy2CompressedImageConfig | None = None):
        if cfg is None:
            cfg = Numpy2CompressedImageConfig()

        self.jpeg_quality = cfg.jpeg_quality
        self.png_compression = cfg.png_compression

    def convert(self, data: NumpyImageMsg) -> CompressedImage:
        """Convert numpy array to CompressedImage."""

        if data.timestamp is None:
            data.timestamp = from_seconds(0)

        encode_param = (
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            if data.target_format == "jpg"
            else [int(cv2.IMWRITE_PNG_COMPRESSION), self.png_compression]
        )

        cv2_encode_success, encoded_img = cv2.imencode(
            f".{data.target_format}", data.data, encode_param
        )
        if not cv2_encode_success:
            raise ValueError(
                f"Failed to encode image to {data.target_format} format."
            )
        compressed_image = CompressedImage(
            data=encoded_img.tobytes(),
            format=data.target_format,
            frame_id=data.frame_id,
            timestamp=data.timestamp,
        )
        return compressed_image


class Numpy2CompressedImageConfig(
    MessageConverterConfig[Numpy2CompressedImage]
):
    """Configuration class for Numpy2CompressedImage."""

    class_type: type[Numpy2CompressedImage] = Numpy2CompressedImage

    jpeg_quality: int = 90
    """JPEG quality for encoding."""

    png_compression: int = 3
    """PNG compression level for encoding."""


class RawImage2CompressedImage(
    MessageConverterStateless[RawImage, CompressedImage]
):
    """Convert Foxglove RawImage messages to CompressedImage messages.

    This stateless converter is intended for writer-side topic converters,
    where a final ``RawImage`` topic should be stored as a Foxglove
    ``CompressedImage`` payload without changing the topic name.

    Args:
        cfg (RawImage2CompressedImageConfig | None, optional): Target format
            and codec settings. Defaults to JPEG with default quality when
            None.
    """

    def __init__(self, cfg: RawImage2CompressedImageConfig | None = None):
        if cfg is None:
            cfg = RawImage2CompressedImageConfig()

        self.format = cfg.format
        self.jpeg_quality = cfg.jpeg_quality
        self.png_compression = cfg.png_compression

    def convert(self, data: RawImage) -> CompressedImage:
        """Encode a RawImage payload with OpenCV.

        Args:
            data (RawImage): Foxglove raw image message.

        Returns:
            CompressedImage: Compressed image with timestamp and frame ID
            copied from ``data``.

        Raises:
            ValueError: If the raw image encoding cannot be compressed with
                the configured format, if row metadata is inconsistent, or if
                OpenCV fails to encode the image.
        """

        image = self._raw_image_to_cv2_image(data)
        encode_param = (
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            if self.format == "jpg"
            else [int(cv2.IMWRITE_PNG_COMPRESSION), self.png_compression]
        )
        ok, encoded_img = cv2.imencode(f".{self.format}", image, encode_param)
        if not ok:
            raise ValueError(f"Failed to encode image to {self.format}.")
        return CompressedImage(
            timestamp=data.timestamp,
            frame_id=data.frame_id,
            data=encoded_img.tobytes(),
            format=self.format,
        )

    def _raw_image_to_cv2_image(self, data: RawImage) -> np.ndarray:
        encoding = data.encoding.lower()
        if encoding == "32fc1":
            raise ValueError("RawImage encoding 32FC1 cannot be compressed.")

        if encoding in ("rgb8", "bgr8"):
            dtype = np.uint8
            channels = 3
            row_bytes = int(data.width) * channels
        elif encoding == "mono8":
            dtype = np.uint8
            channels = 1
            row_bytes = int(data.width)
        elif encoding == "mono16":
            if self.format == "jpg":
                raise ValueError(
                    "RawImage encoding mono16 can only be compressed as png."
                )
            dtype = np.uint16
            channels = 1
            row_bytes = int(data.width) * np.dtype(dtype).itemsize
        else:
            raise ValueError(
                f"Unsupported RawImage encoding: {data.encoding}."
            )

        step = int(data.step)
        height = int(data.height)
        if step < row_bytes:
            raise ValueError(
                f"RawImage step {step} is smaller than row size {row_bytes}."
            )
        if len(data.data) < step * height:
            raise ValueError(
                "RawImage data is shorter than height * step: "
                f"{len(data.data)} < {step * height}."
            )

        rows = [
            np.frombuffer(
                data.data[row_idx * step : row_idx * step + row_bytes],
                dtype=dtype,
            )
            for row_idx in range(height)
        ]
        image = np.stack(rows, axis=0)
        if channels == 3:
            image = image.reshape(height, int(data.width), channels)
            if encoding == "rgb8":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image
        return image.reshape(height, int(data.width))


class RawImage2CompressedImageConfig(
    MessageConverterConfig[RawImage2CompressedImage]
):
    """Configuration class for RawImage2CompressedImage."""

    class_type: type[RawImage2CompressedImage] = RawImage2CompressedImage

    format: Literal["jpg", "png"] = "jpg"
    """Target compressed image format."""

    jpeg_quality: int = 90
    """JPEG quality for encoding."""

    png_compression: int = 3
    """PNG compression level for encoding."""


class CompressedImage2Numpy(
    MessageConverterStateless[CompressedImage, NumpyImageMsg]
):
    """Convert CompressedImage to numpy array.

    Args:
        cfg (CompressedImage2NumpyConfig | None, optional): Converter config.
            Currently accepted for API symmetry and defaults when None.
    """

    def __init__(self, cfg: CompressedImage2NumpyConfig | None = None):
        if cfg is None:
            cfg = CompressedImage2NumpyConfig()

    def convert(self, data: CompressedImage) -> NumpyImageMsg:
        """Convert CompressedImage to numpy array."""

        img = cv2.imdecode(
            np.frombuffer(data.data, np.uint8), cv2.IMREAD_UNCHANGED
        )
        return NumpyImageMsg(
            data=img,  # type: ignore
            target_format=data.format,  # type: ignore
            frame_id=data.frame_id,
            timestamp=data.timestamp if data.HasField("timestamp") else None,
        )


class CompressedImage2NumpyConfig(
    MessageConverterConfig[CompressedImage2Numpy]
):
    """Configuration class for CompressedImage2Numpy."""

    class_type: type[CompressedImage2Numpy] = CompressedImage2Numpy
