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
from typing import Any, Sequence, TypeVar

from PIL import Image
from robo_orchard_core.datatypes.camera_data import (
    ImageChannelLayout,
    ImageMode,
)
from robo_orchard_core.utils.torch_utils import dtype_str2torch

from robo_orchard_lab.dataset.datatypes import BatchCameraData, BatchImageData
from robo_orchard_lab.transforms.base import (
    ClassType,
    DictTransform,
    DictTransformConfig,
)

__all__ = [
    "ImageChannelLayout",
    "ImageMode",
    "ImageLayoutFormatter",
    "ImageLayoutFormatterConfig",
    "AsPILImages",
    "AsPILImagesConfig",
]
BatchImageDataType = TypeVar(
    "BatchImageDataType", bound=BatchImageData | BatchCameraData
)


class ImageLayoutFormatter(DictTransform):
    """A transform to format the image layout.

    This transform changes the layout of the image data to HWC or CHW format,
    depending on the specified output layout. For training, CHW is often
    preferred, while for visualization, HWC is more common. The transform also
    ensures that the image data is in the specified dtype.

    """

    cfg: ImageLayoutFormatterConfig
    is_variadic: bool = True

    def __init__(self, cfg: ImageLayoutFormatterConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def _format_layout(self, data: BatchImageDataType) -> BatchImageDataType:
        """Format the image layout."""
        target_dtype = (
            dtype_str2torch(self.cfg.output_dtype)
            if self.cfg.output_dtype
            else data.sensor_data.dtype
        )
        out = data.model_copy(deep=False)

        layout = out.channel_layout
        if layout != self.cfg.output_layout:
            if self.cfg.output_layout == ImageChannelLayout.CHW:
                assert layout == ImageChannelLayout.HWC, (
                    f"Expected input layout {ImageChannelLayout.HWC}, "
                    f"but got {layout}."
                )
                out.sensor_data = out.sensor_data.permute(0, 3, 1, 2)
            elif self.cfg.output_layout == ImageChannelLayout.HWC:
                assert layout == ImageChannelLayout.CHW, (
                    f"Expected input layout {ImageChannelLayout.CHW}, "
                    f"but got {layout}."
                )
                out.sensor_data = out.sensor_data.permute(0, 2, 3, 1)
            else:
                raise ValueError(
                    f"Unsupported output layout: {self.cfg.output_layout}"
                )
        if out.sensor_data.dtype != target_dtype:
            out.sensor_data = out.sensor_data.to(target_dtype)

        return out

    def transform(
        self, **kwargs: dict[str, BatchImageDataType]
    ) -> dict[str, BatchImageDataType]:
        """Format the image layout."""
        ret = {}
        for key, value in kwargs.items():
            if isinstance(value, (BatchCameraData, BatchImageData)):
                # Format the image layout
                ret[key] = self._format_layout(value)
            else:
                raise TypeError(
                    f"Expected BatchCameraData for key '{key}', "
                    f"but got {type(value)}."
                )
        return ret


class ImageLayoutFormatterConfig(DictTransformConfig[ImageLayoutFormatter]):
    class_type: ClassType[ImageLayoutFormatter] = ImageLayoutFormatter

    input_columns: Sequence[str]
    "The columns to format."
    output_layout: ImageChannelLayout = ImageChannelLayout.CHW
    "The output image channel layout."
    output_dtype: str | None = None
    "The output image dtype. If None, the input dtype is used."


class AsPILImages(DictTransform):
    """A transform to convert BatchCameraData to PIL images."""

    cfg: AsPILImagesConfig
    is_variadic: bool = True

    def __init__(self, cfg: AsPILImagesConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def transform(self, **kwargs) -> dict[str, Any]:
        """Convert BatchCameraData to a list of PIL images."""
        ret = {}
        for key, value in kwargs.items():
            if isinstance(value, BatchCameraData):
                # Convert the image data to PIL images
                assert isinstance(value, BatchCameraData)
                ret[key] = self._convert_to_pil(value)
            else:
                raise TypeError(
                    f"Expected BatchCameraData for key '{key}', "
                    f"but got {type(value)}."
                )
        return ret

    def _convert_to_pil(self, data: BatchCameraData) -> list[Image.Image]:
        """Convert BatchCameraData to a list of PIL images."""
        sensor_data = data.sensor_data
        if data.channel_layout == ImageChannelLayout.CHW:
            sensor_data = sensor_data.permute(0, 2, 3, 1)  # Convert to HWC
        elif data.channel_layout != ImageChannelLayout.HWC:
            raise NotImplementedError(
                f"Unsupported channel layout: {data.channel_layout}. "
                "Expected HWC or CHW."
            )
        mode = data.pix_fmt
        sensor_data = sensor_data.numpy(force=False)
        if data.pix_fmt == ImageMode.BGR:
            sensor_data = sensor_data[..., ::-1]  # Convert BGR to RGB
            mode = ImageMode.RGB
        elif data.pix_fmt == ImageMode.RGB:
            pass
        elif data.pix_fmt in [
            ImageMode.I,
            ImageMode.I16,
            ImageMode.BIT,
            ImageMode.F,
            ImageMode.L,
        ]:
            sensor_data = sensor_data.squeeze(
                -1
            )  # Remove last channel for grayscale
        else:
            raise NotImplementedError(
                "Unsupported pixel format: "
                f"{data.pix_fmt}. Expected RGB, BGR, or grayscale."
            )
        pil_images = [Image.fromarray(img, mode=mode) for img in sensor_data]
        return pil_images


class AsPILImagesConfig(DictTransformConfig[AsPILImages]):
    class_type: ClassType[AsPILImages] = AsPILImages

    input_columns: Sequence[str]
