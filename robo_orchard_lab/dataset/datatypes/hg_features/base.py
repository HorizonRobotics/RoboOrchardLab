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
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import datasets as hg_datasets
import pyarrow as pa
from datasets.features.features import register_feature
from pydantic import BaseModel
from typing_extensions import TypeVar

__all__ = [
    "RODataFeature",
    "RODictDataFeature",
    "TypedDictFeatureDecode",
    "FeatureDecodeMixin",
    "ToDataFeatureMixin",
    "hg_dataset_feature",
    "check_fields_consistency",
    "guess_hg_features",
]


class FeatureDecodeMixin(metaclass=ABCMeta):
    """Mixin class for features that support decoding."""

    @abstractmethod
    def decode_example(self, value: Any, **kwargs) -> Any:
        """Decode the example value from its stored format."""
        raise NotImplementedError(
            "Subclasses must implement decode_example method."
        )


class TypedDictFeatureDecode(FeatureDecodeMixin):
    """Helper class for decoding typed dictionary features.

    This class uses `_dict` as the feature decoding schema to decode the
    example value. User should extend this class and set the `_dict`
    attribute to represent the schema.

    Note that `_dict` will not be included in the serialized output, and
    also not be used in the initialization of the class. You should
    initialize `_dict` in `__post_init__` or in the class body.

    """

    _dict: dict = field(init=False, repr=False)
    _decode_type: type = field(init=False, repr=False)
    decode: bool = True

    def decode_example(self, value: Any, **kwargs) -> Any:
        if not self.decode:
            raise RuntimeError(
                "This feature does not support decoding. "
                "Set decode=True to enable decoding."
            )
        ret: dict = hg_datasets.features.features.decode_nested_example(
            schema=self._dict, obj=value
        )
        return self._decode_type(**ret)


@dataclass
class RODataFeature(metaclass=ABCMeta):
    """Base class for RoboOrchard dataset features.

    User should implement the `pa_type` property and `encode_example` method
    to define the specific feature type and how to encode example values.

    This class does not include `decode_example` method, as it is not
    required for all features. If you need to decode the example values,
    you can inherit from `FeatureDecodeMixin`.

    """

    _type: str = field(init=False, repr=False)
    """The class name of the feature type. Needed for serialization
    and deserialization. Should be set in subclasses."""

    def __call__(self) -> pa.DataType:
        """Return the pyarrow data type for this feature."""
        return self.pa_type

    @property
    @abstractmethod
    def pa_type(self) -> pa.DataType:
        """Return the pyarrow data type for this feature."""
        raise NotImplementedError(
            "Subclasses must implement pa_type property."
        )

    @abstractmethod
    def encode_example(self, value: Any) -> Any:
        """Encode the example value into a format suitable for storage."""
        raise NotImplementedError(
            "Subclasses must implement encode_example method."
        )


class RODictDataFeature(RODataFeature):
    """A feature that is composed of a dictionary of features.

    This class provide a easy way to define a feature that is a dictionary
    of features. It is useful for defining complex features that are
    composed of multiple fields. The user should define the `_dict` attribute
    as a dictionary mapping field names to features. The keys of the dictionary
    are the field names, and the values are the features.

    This class does not include `decode_example` method, as it is not
    required for all features. If you need to decode the example values,
    you can inherit from `DictFeatureDecodeMixin`.

    """

    _dict: dict = field(init=False, repr=False)

    @property
    def pa_type(self) -> pa.DataType:
        """Return the pyarrow data type for this feature."""
        return hg_datasets.features.features.get_nested_type(self._dict)

    def encode_example(self, value: Any) -> Any:
        return hg_datasets.features.features.encode_nested_example(
            schema=self._dict, obj=value
        )

    def items(self):
        """Return the items of the dictionary."""
        return self._dict.items()

    def keys(self):
        """Return the keys of the dictionary."""
        return self._dict.keys()

    def values(self):
        """Return the values of the dictionary."""
        return self._dict.values()


class ToDataFeatureMixin(metaclass=ABCMeta):
    """Mixin class for features that can be converted to a pyarrow DataType."""

    @classmethod
    @abstractmethod
    def dataset_feature(cls) -> RODataFeature:
        raise NotImplementedError(
            "Subclasses must implement dataset_feature method."
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Get the value of the feature by key."""
        return getattr(self, key, default)


RODataFeatureType = TypeVar("RODataFeatureType", bound=RODataFeature)


def hg_dataset_feature(
    cls: type[RODataFeatureType],
) -> type[RODataFeatureType]:
    """Decorator to register a feature class with its type."""
    if not issubclass(cls, RODataFeature):
        raise TypeError("Feature class must inherit from RODataFeature.")
    cls._type = cls.__qualname__
    register_feature(cls, cls._type)
    return cls


def check_fields_consistency(
    cls: type[BaseModel],
    pa_struct: pa.StructType,
):
    pydantic_fields = set(cls.model_fields.keys())
    pa_fields = set([field.name for field in pa_struct.fields])
    if pydantic_fields != pa_fields:
        raise TypeError(
            f"Pydantic fields {pydantic_fields} do not match "
            f"pyarrow fields {pa_fields} for {cls.__name__}."
            " This means that the feature is not fully implemented."
        )


def guess_hg_features(
    data: dict,
) -> hg_datasets.features.Features:
    """Guess the Hugging Face dataset features from an dict.

    If the object contains a list or tuple, it will try to guess the feature
    type from the first non-null value. If all values are None or empty,
    it raises a ValueError.

    Try to avoid any None values in the input object, as it may lead to
    incorrect feature type inference.

    """

    if not isinstance(data, dict):
        raise TypeError(
            "Input data must be a dictionary mapping field names to values."
        )

    def guess_feature(obj: Any):
        if isinstance(obj, (hg_datasets.Features, RODataFeature)):
            return obj
        elif isinstance(obj, dict):
            return {k: guess_feature(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            idx, value = hg_datasets.features.features.first_non_null_value(
                obj
            )
            if idx < 0:
                raise ValueError(
                    "Cannot guess features from a list or tuple with all "
                    "None values or empty."
                )
            return hg_datasets.Sequence(feature=guess_feature(value))
        elif isinstance(obj, ToDataFeatureMixin):
            return obj.dataset_feature()
        else:
            return hg_datasets.features.features.generate_from_arrow_type(
                pa.array([obj]).type
            )

    feature_dict = guess_feature(data)
    assert isinstance(feature_dict, dict), (
        "The guessed features must be a dictionary mapping field names to "
        "Hugging Face dataset features."
    )
    return hg_datasets.Features(**(feature_dict))
