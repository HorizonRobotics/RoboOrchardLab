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
from typing import Any, Callable, Literal

import datasets as hg_datasets
from robo_orchard_core.datatypes.camera_data import (
    BatchCameraData as _BatchCameraData,
    BatchCameraDataEncoded as _BatchCameraDataEncoded,
    BatchCameraInfo as _BatchCameraInfo,
    BatchImageData,
    Distortion as _Distortion,
    ImageMode,
)
from robo_orchard_core.utils.torch_utils import Device

from robo_orchard_lab.dataset.datatypes.geometry import (
    BatchFrameTransformFeature,
)
from robo_orchard_lab.dataset.datatypes.hg_features import (
    RODictDataFeature,
    ToDataFeatureMixin,
    TypedDictFeatureDecode,
    check_fields_consistency,
    hg_dataset_feature,
)
from robo_orchard_lab.dataset.datatypes.hg_features.tensor import (
    AnyTensorFeature,
    TypedTensorFeature,
)

__all__ = [
    "ImageMode",
    "DistortionFeature",
    "Distortion",
    "BatchCameraInfoFeature",
    "BatchCameraInfo",
    "BatchCameraDataEncodedFeature",
    "BatchCameraDataEncoded",
    "BatchImageData",
    "BatchCameraDataFeature",
    "BatchCameraData",
]


class Distortion(_Distortion, ToDataFeatureMixin):
    """A class for distortion parameters with dataset feature support."""

    @classmethod
    def dataset_feature(
        cls, dtype: Literal["float32", "float64"] = "float32"
    ) -> DistortionFeature:
        ret = DistortionFeature(dtype=dtype)
        check_fields_consistency(cls, ret.pa_type)
        return ret


@hg_dataset_feature
@dataclass
class DistortionFeature(RODictDataFeature, TypedDictFeatureDecode):
    """A feature for storing distortion parameters in a dataset.

    The underlying data is stored as a serialized numpy array
    with additional metadata about the distortion parameters.
    """

    dtype: Literal["float32", "float64"] = "float32"
    decode: bool = True
    _decode_type: type = Distortion

    def __post_init__(self):
        self._dict = {
            "model": hg_datasets.features.features.Value("string"),
            "coefficients": TypedTensorFeature(
                dtype=self.dtype, as_torch_tensor=True
            ),
        }


class BatchCameraInfo(_BatchCameraInfo, ToDataFeatureMixin):
    """A class for batch camera info with dataset feature support."""

    @classmethod
    def dataset_feature(
        cls, dtype: Literal["float32", "float64"] = "float32"
    ) -> BatchCameraInfoFeature:
        ret = BatchCameraInfoFeature(dtype=dtype)
        check_fields_consistency(cls, ret.pa_type)
        return ret


@hg_dataset_feature
@dataclass
class BatchCameraInfoFeature(RODictDataFeature, TypedDictFeatureDecode):
    """A feature for storing batch camera info in a dataset.

    The underlying data is stored as a serialized numpy array
    with additional metadata about the camera info.
    """

    dtype: Literal["float32", "float64"] = "float32"
    decode: bool = True
    _decode_type: type = BatchCameraInfo

    def __post_init__(self):
        self._dict = {
            "topic": hg_datasets.features.features.Value("string"),
            "frame_id": hg_datasets.features.features.Value("string"),
            "image_shape": hg_datasets.features.features.Sequence(
                hg_datasets.features.Value("int32")
            ),
            "intrinsic_matrices": TypedTensorFeature(
                dtype=self.dtype, as_torch_tensor=True
            ),
            "distortion": DistortionFeature(dtype=self.dtype),
            "pose": BatchFrameTransformFeature(dtype=self.dtype),
        }


class BatchCameraDataEncoded(BatchCameraInfo, _BatchCameraDataEncoded):
    """A class for batch camera data with dataset feature support."""

    @classmethod
    def dataset_feature(
        cls, dtype: Literal["float32", "float64"] = "float32"
    ) -> BatchCameraDataEncodedFeature:
        ret = BatchCameraDataEncodedFeature(dtype=dtype)
        check_fields_consistency(cls, ret.pa_type)
        return ret

    def decode(
        self,
        decoder: Callable[[bytes, str], BatchImageData],
        device: Device = "cpu",
    ) -> BatchCameraData:
        return BatchCameraData(**super().decode(decoder, device).__dict__)


@hg_dataset_feature
@dataclass
class BatchCameraDataEncodedFeature(BatchCameraInfoFeature):
    """A feature for storing batch camera data in a dataset.

    The underlying data is stored as a serialized numpy array
    with additional metadata about the camera data.
    """

    dtype: Literal["float32", "float64"] = "float32"
    decode: bool = True
    _decode_type: type = BatchCameraDataEncoded

    def __post_init__(self):
        super().__post_init__()
        self._dict.update(
            {
                "sensor_data": hg_datasets.features.features.Sequence(
                    hg_datasets.features.Value("binary")
                ),
                "format": hg_datasets.features.features.Value("string"),
                "timestamps": hg_datasets.features.features.Sequence(
                    hg_datasets.features.Value("int64")
                ),
            }
        )


class BatchCameraData(BatchCameraInfo, _BatchCameraData):
    """A class for batch camera data with dataset feature support."""

    @classmethod
    def dataset_feature(
        cls, dtype: Literal["float32", "float64"] = "float32"
    ) -> BatchCameraDataFeature:
        ret = BatchCameraDataFeature(dtype=dtype)
        check_fields_consistency(cls, ret.pa_type)
        return ret


@hg_dataset_feature
@dataclass
class BatchCameraDataFeature(BatchCameraInfoFeature):
    """A feature for storing batch camera data in a dataset.

    The underlying data is stored as a serialized numpy array
    with additional metadata about the camera data.
    """

    dtype: Literal["float32", "float64"] = "float32"
    decode: bool = True
    _decode_type: type = BatchCameraData

    def __post_init__(self):
        super().__post_init__()
        self._dict.update(
            {
                "sensor_data": AnyTensorFeature(),
                "pix_fmt": hg_datasets.features.features.Value("string"),
                "timestamps": hg_datasets.features.features.Sequence(
                    hg_datasets.features.Value("int64")
                ),
            }
        )

    def encode_example(self, value: BatchCameraData) -> Any:
        return super().encode_example(value.__dict__)
