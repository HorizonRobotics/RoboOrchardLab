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
import pyarrow as pa
from datasets.table import array_cast
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
    PickleFeature,
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
    cls,
    dtype: Literal["float32", "float64"] = "float32",
    use_pickle: bool = False,
) -> DistortionFeature | PickleFeature:
    """A class for distortion parameters with dataset feature support."""
    if use_pickle:
        return PickleFeature(class_type=Distortion)

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

    def __post_init__(self):
        self._decode_type = Distortion
        self._dict = {
            "model": hg_datasets.features.features.Value("string"),
            "coefficients": TypedTensorFeature(
                dtype=self.dtype, as_torch_tensor=True
            ),
        }


@classmethod
def _camera_info_dataset_feature(
    cls,
    dtype: Literal["float32", "float64"] = "float32",
    use_pickle: bool = False,
) -> BatchCameraInfoFeature | PickleFeature:
    if use_pickle:
        return PickleFeature(class_type=BatchCameraInfo)
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

    def __post_init__(self):
        self._decode_type = BatchCameraInfo
        self._dict = {
            "topic": hg_datasets.features.features.Value("string"),
            "frame_id": hg_datasets.features.features.Value("string"),
            "image_shape": hg_datasets.features.features.Sequence(
                hg_datasets.features.Value("int32")
            ),
            "intrinsic_matrices": TypedTensorFeature(
                dtype=self.dtype, as_torch_tensor=True
            ),
            "transform_matrices": TypedTensorFeature(
                dtype=self.dtype, as_torch_tensor=True
            ),
            "distortion": DistortionFeature(dtype=self.dtype),
            "pose": BatchFrameTransformFeature(dtype=self.dtype),
        }

    def cast_storage(self, storage: pa.StructArray) -> pa.StructArray:
        """Cast the storage array to the expected schema.

        When loading arrow table with old schema, we need to update
        the schema to current version.

        Note:
            This method only handles missing fields. If the field
            type is changed, it will not be handled here!

        """
        # Cast the storage array to the expected schema
        storage_type: pa.StructType = storage.type
        feature_type: pa.StructType = self.pa_type
        # we only handle the case when storage is a struct array,
        # and leave other cases to the default array_cast implementation.
        if pa.types.is_struct(storage_type):
            if storage_type == feature_type:
                return storage
            # find all field
            existing_fields = set(storage_type.names)

            # reconstruct storage with missing fields filled with null values
            arrays = []
            for field in feature_type:
                if field.name in existing_fields:
                    arrays.append(storage.field(field.name))
                else:
                    arrays.append(
                        pa.array([None] * len(storage), type=field.type)
                    )
            storage = pa.StructArray.from_arrays(
                arrays, names=feature_type.names, mask=storage.is_null()
            )
        # return storage
        return array_cast(storage, self.pa_type)


@classmethod
def _camera_data_encoded_dataset_feature(
    cls,
    dtype: Literal["float32", "float64"] = "float32",
    use_pickle: bool = False,
) -> BatchCameraDataEncodedFeature | PickleFeature:
    if use_pickle:
        return PickleFeature(class_type=BatchCameraDataEncoded)
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

    def __post_init__(self):
        super().__post_init__()
        self._decode_type = BatchCameraDataEncoded
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
    cls,
    dtype: Literal["float32", "float64"] = "float32",
    use_pickle: bool = False,
) -> BatchCameraDataFeature | PickleFeature:
    if use_pickle:
        return PickleFeature(class_type=BatchCameraData)
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

    def __post_init__(self):
        super().__post_init__()
        self._decode_type = BatchCameraData
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
