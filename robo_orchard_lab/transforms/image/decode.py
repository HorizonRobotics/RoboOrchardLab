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
import io
from typing import Literal, Sequence

import numpy as np
import torch
from pydantic import Field

from robo_orchard_lab.dataset.datatypes import (
    BatchCameraData,
    BatchCameraDataEncoded,
    BatchImageData,
    ImageMode,
)
from robo_orchard_lab.transforms.base import (
    ClassType,
    DictTransform,
    DictTransformConfig,
)

__all__ = [
    "ImageDecode",
    "ImageDecodeConfig",
]


class ImageDecode(DictTransform):
    """A transform to decode BatchCameraDataEncoded to BatchCameraData."""

    cfg: ImageDecodeConfig
    is_variadic: bool = True

    def __init__(self, cfg: ImageDecodeConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def transform(self, **kwargs) -> dict[str, BatchCameraData]:
        """Decode the image data from bytes to PIL images."""
        if self.cfg.backend == "pil":
            # return img.decode(self._decode_impl_pil)
            impl = self._decode_impl_pil
        elif self.cfg.backend == "cv2":
            # return img.decode(self._decode_impl_cv2)
            impl = self._decode_impl_cv2
        else:
            raise ValueError(f"Unsupported backend: {self.cfg.backend}")

        ret = {}
        for key, value in kwargs.items():
            if isinstance(value, BatchCameraDataEncoded):
                # Decode the image data
                assert isinstance(value, BatchCameraDataEncoded)
                out = value.decode(impl)
                # handle RGB/BGR inversion if needed
                if self.cfg.invert_rgb and out.pix_fmt == ImageMode.RGB:
                    out.pix_fmt = ImageMode.BGR
                elif self.cfg.invert_rgb and out.pix_fmt == ImageMode.BGR:
                    out.pix_fmt = ImageMode.RGB
                ret[key] = out
            else:
                raise TypeError(
                    f"Expected BatchCameraDataEncoded for key '{key}', "
                    f"but got {type(value)}."
                )
        return ret

    def _decode_impl_pil(self, data: bytes, format: str) -> BatchImageData:
        """Decode the image data from bytes to PIL images."""
        from PIL import Image as PILImage

        img = PILImage.open(io.BytesIO(data))
        # # convert img to numpy
        img_tensor = torch.asarray(np.array(img))
        ret_mode = ImageMode(img.mode)
        if img_tensor.ndim == 2:
            img_tensor = img_tensor.unsqueeze(-1)
        assert img_tensor.ndim == 3, "Image tensor must be 3D (H, W, C)"
        return BatchImageData(
            sensor_data=img_tensor.unsqueeze(0),  # Add batch dimension
            pix_fmt=ret_mode,
        )

    def _decode_impl_cv2(self, data: bytes, format: str) -> BatchImageData:
        """Decode the image data from bytes to OpenCV images."""
        import cv2

        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        img_tensor = torch.from_numpy(img)

        if img_tensor.ndim == 2:
            img_tensor = img_tensor.unsqueeze(-1)

        if img_tensor.shape[-1] == 3 and img_tensor.dtype == torch.uint8:
            ret_mode = ImageMode.BGR
        elif img_tensor.shape[-1] == 1 and img_tensor.dtype == torch.uint8:
            ret_mode = ImageMode.L
        elif img_tensor.shape[-1] == 1 and img_tensor.dtype == torch.uint16:
            ret_mode = ImageMode.I16
        else:
            raise ValueError(
                f"Unsupported image format: {img_tensor.shape} "
                f"with dtype {img_tensor.dtype}"
            )

        assert img_tensor.ndim == 3, "Image tensor must be 3D (H, W, C)"
        return BatchImageData(
            sensor_data=img_tensor.unsqueeze(0),  # Add batch dimension
            pix_fmt=ret_mode,
        )


class ImageDecodeConfig(DictTransformConfig[ImageDecode]):
    class_type: ClassType[ImageDecode] = ImageDecode

    backend: Literal["pil", "cv2"] = "pil"

    input_columns: Sequence[str] = Field(
        description="The columns to decode.",
    )

    invert_rgb: bool = False
    """Whether to invert RGB to BGR.

    For some datasets, the actual image data may be incorrectly stored as RGB
    but should be treated as BGR. This flag allows to invert the channels
    accordingly.
    """
