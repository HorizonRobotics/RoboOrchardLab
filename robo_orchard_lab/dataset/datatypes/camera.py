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
from typing import Any, Literal

import datasets as hg_datasets
from robo_orchard_core.datatypes.camera_data import (
    BatchCameraData,
    BatchCameraDataEncoded,
    BatchCameraInfo,
    BatchImageData,
    Distortion,
    ImageMode,
)

from robo_orchard_lab.dataset.datatypes.geometry import (
    BatchFrameTransformFeature,
)
from robo_orchard_lab.dataset.datatypes.hg_features import (
    RODictDataFeature,
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


@classmethod
def _distortion_dataset_feature(
    cls, dtype: Literal["float32", "float64"] = "float32"
) -> DistortionFeature:
    """A class for distortion parameters with dataset feature support."""
    ret = DistortionFeature(dtype=dtype)
    check_fields_consistency(cls, ret.pa_type)
    return ret


Distortion.dataset_feature = _distortion_dataset_feature


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


@classmethod
def _camera_info_dataset_feature(
    cls, dtype: Literal["float32", "float64"] = "float32"
) -> BatchCameraInfoFeature:
    ret = BatchCameraInfoFeature(dtype=dtype)
    check_fields_consistency(cls, ret.pa_type)
    return ret


BatchCameraInfo.dataset_feature = _camera_info_dataset_feature


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


@classmethod
def _camera_data_encoded_dataset_feature(
    cls, dtype: Literal["float32", "float64"] = "float32"
) -> BatchCameraDataEncodedFeature:
    ret = BatchCameraDataEncodedFeature(dtype=dtype)
    check_fields_consistency(cls, ret.pa_type)
    return ret


BatchCameraDataEncoded.dataset_feature = (  # type: ignore
    _camera_data_encoded_dataset_feature
)


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


@classmethod
def _camera_data_dataset_feature(
    cls, dtype: Literal["float32", "float64"] = "float32"
) -> BatchCameraDataFeature:
    ret = BatchCameraDataFeature(dtype=dtype)
    check_fields_consistency(cls, ret.pa_type)
    return ret


BatchCameraData.dataset_feature = _camera_data_dataset_feature  # type: ignore


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
