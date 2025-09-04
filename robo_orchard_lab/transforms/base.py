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
import copy
import functools
import inspect
from abc import ABCMeta, abstractmethod
from dataclasses import is_dataclass
from typing import Sequence

from pydantic import AliasChoices, BaseModel, Field
from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassType,
    Config,
    ConfigInstanceOf,
    load_from,
)
from typing_extensions import TypeVar

from robo_orchard_lab.utils.state import (
    State,
    StateList,
    StateSaveLoadMixin,
    obj2state,
    state2obj,
)

__all__ = [
    "Config",
    "ClassType",
    "ConfigInstanceOf",
    "DictTransform",
    "DictTransformType",
    "DictTransformConfig",
    "ConcatDictTransform",
    "ConcatDictTransformConfig",
]


class DictTransform(StateSaveLoadMixin, metaclass=ABCMeta):
    """A class that defines the interface for transforming a dict.

    The dict is usually a row in a dataset, and the transform
    will take the input columns, apply some transformation, and return
    a new dict with the transformed values added to the original dict.

    User should implement the `transform` method to define the specific
    transformation logic.

    If you use a dataclass or BaseModel as the return type of
    the transform method, `output_columns` will be automatically
    inferred from the fields of the dataclass or BaseModel. Otherwise,
    you should implement the `output_columns` property to return the
    expected output columns for the transform.


    For Transform whose input and output columns are not known
    at the time of configuration, the input and output columns
    properties should not be used, and the `check_return_columns`
    configuration should be set to False. This will prevent runtime
    errors when the transform is called with a row dict.

    """

    InitFromConfig: bool = True

    cfg: DictTransformConfig

    @abstractmethod
    def transform(self, **kwargs) -> dict | BaseModel:
        """Transform the input columns into a new row dict to be updated.

        All input columns will be passed as keyword arguments.
        """
        raise NotImplementedError

    def __call__(self, row: dict) -> dict:
        """Call the transform on a row dict.

        This method will extract the input columns from the row dict,
        call the transform method, and return a new row dict with the
        transformed values added to the original row dict.

        """
        mapped_input = row.copy()
        # mapping input columns if needed
        if isinstance(self.cfg.input_columns, dict):
            for src_name, dst_name in self.cfg.input_columns.items():
                if src_name not in mapped_input:
                    raise ValueError(
                        f"Input column {src_name} not found in row dict."
                    )
                mapped_input[dst_name] = mapped_input.pop(src_name)

        # take columns that required by the transform
        ts_input = {}
        for col in self.input_columns:
            if (
                col not in mapped_input
                and not self.cfg.missing_input_columns_as_none
            ):
                raise KeyError(f"Input column `{col}` not found in row dict.")
            ts_input[col] = mapped_input.get(col, None)

        columns_after = self.transform(**ts_input)
        if isinstance(columns_after, BaseModel):
            columns_after = {
                k: getattr(columns_after, k)
                for k in columns_after.model_fields
            }
        elif is_dataclass(columns_after):
            columns_after = {
                field.name: getattr(columns_after, field.name)
                for field in columns_after.__dataclass_fields__.values()
            }

        if not isinstance(columns_after, dict):
            raise TypeError(
                f"Transform {self.__class__.__name__} must return a dict, "
                f"got {type(columns_after)}."
            )

        # check that the output columns match the expected output columns
        if self.cfg.check_return_columns:
            for col in columns_after.keys():
                if col not in self.output_columns:
                    raise ValueError(
                        f"Output column {col} not in expected output columns: "
                        f"{self.output_columns}."
                    )

        for src_name, dst_name in self.cfg.output_column_mapping.items():
            if dst_name in columns_after:
                raise ValueError(
                    f"Output column {dst_name} already exists in transformed "
                    "columns."
                )
            if src_name in columns_after:
                columns_after[dst_name] = columns_after.pop(src_name)
        if self.cfg.keep_input_columns:
            ret = row.copy()
        else:
            ret = {}
        ret.update(columns_after)
        return ret

    @functools.cached_property
    def input_columns(self) -> list[str]:
        """The input columns that this transform requires.

        This should be a list of column names that are required
        for the transformation. The transform method will be called
        with these columns as keyword arguments.

        The input columns are determined by the signature of the
        `transform` method, and if the `input_columns` configuration
        is set, it will be used as well to determine the input columns.

        """
        # use inspect to get the parameters of the transform method
        self._input_columns = []

        def get_input_from_signature() -> list[str]:
            ret = []
            sig = inspect.signature(self.transform)
            for param in sig.parameters.values():
                if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                    ret.append(param.name)
                elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                    continue
                elif param.kind == inspect.Parameter.VAR_KEYWORD:
                    continue
                else:
                    continue
            return ret

        self._input_columns = get_input_from_signature()
        if isinstance(self.cfg.input_columns, dict):
            # if input_columns is a dict, we need to use the values as
            # input columns
            self._input_columns += list(self.cfg.input_columns.values())
        elif isinstance(self.cfg.input_columns, (list, tuple)):
            # if input_columns is a list or tuple, we use it directly
            self._input_columns += list(self.cfg.input_columns)
        else:
            raise TypeError(
                f"Expected input_columns to be a dict, list, tuple or None, "
                f"but got {type(self.cfg.input_columns)}."
            )
        # handle duplicate columns
        self._input_columns = list(set(self._input_columns))
        return self._input_columns

    @functools.cached_property
    def mapped_input_columns(self) -> list[str]:
        """The input columns that this transform requires for column mapping.

        If no mapping is required, this will be the same as input_columns.

        """

        old_input_columns = self.input_columns

        if isinstance(self.cfg.input_columns, dict):
            inverted_mapping = {
                v: k for k, v in self.cfg.input_columns.items()
            }
            mapped_input_columns = [
                inverted_mapping.get(col, col) for col in old_input_columns
            ]
        elif isinstance(self.cfg.input_columns, (list, tuple)):
            mapped_input_columns = list(self.cfg.input_columns)
        elif self.cfg.input_columns is None:
            mapped_input_columns = old_input_columns
        else:
            raise TypeError(
                f"Expected input_columns to be a dict, list, tuple or None, "
                f"but got {type(self.cfg.input_columns)}."
            )

        self._mapped_input_columns = mapped_input_columns
        return self._mapped_input_columns

    @functools.cached_property
    def output_columns(self) -> list[str]:
        """The output columns that this transform produces.

        This should be a list of column names that the transform will
        produce as output. The transform method will return a dict
        with these keys.

        Note that this property contains all possible output columns,
        not just the ones that are actually produced by the transform.
        The transform method may return a subset of these columns,
        but the output_columns property should list all columns that
        the transform can produce.

        """

        def generate_output_columns() -> list[str]:
            # using signature to get the return type of the transform method
            sig = inspect.signature(self.transform, eval_str=True)
            return_annotation = sig.return_annotation
            # get the classtype from the return annotation
            # if the return type is dataclass, we need to extract the fields
            if is_dataclass(return_annotation):
                return list(return_annotation.__dataclass_fields__.keys())
            # if the return type is a subclass of BaseModel,
            # we can use its fields
            elif isinstance(return_annotation, type) and issubclass(
                return_annotation, BaseModel
            ):
                return list(return_annotation.model_fields.keys())
            else:
                raise NotImplementedError(
                    "Cannot determine output columns for "
                    f"{self.__class__.__name__}. Return type "
                    f"{return_annotation} is not a dataclass or BaseModel. "
                    "You should implement the output_columns property to "
                    "return the expected output columns for this transform."
                )

        self._output_columns = generate_output_columns()
        return self._output_columns

    @functools.cached_property
    def mapped_output_columns(self) -> list[str]:
        """The output columns that this transform produces after mapping."""
        old_output_columns = self.output_columns
        mapped_output_columns = [
            self.cfg.output_column_mapping.get(col, col)
            for col in old_output_columns
        ]
        self._mapped_output_columns = mapped_output_columns
        return self._mapped_output_columns

    def __repr__(self) -> str:
        ret = f"{self.__class__.__name__}("
        ret += f"cfg={self.cfg.to_dict(exclude_defaults=True)}"
        ret += ")"
        return ret

    def __getstate__(self) -> State:
        """Return the state of the object for pickling."""
        state = self.__dict__.copy()
        # Remove any cached properties to avoid pickling issues
        for key in [
            "_input_columns",
            "_output_columns",
            "_mapped_input_columns",
            "_mapped_output_columns",
        ]:
            state.pop(key, None)

        config = state.pop("cfg")
        state = obj2state(state)

        return State(
            state=state,
            config=config,
            parameters=None,
            save_to_path=None,
            class_type=type(self),
        )

    def __setstate__(self, state: State) -> None:
        """Set the state of the object from the unpickled state."""
        self.cfg = state.config  # type: ignore
        self.__dict__.update(state.parameters or {})
        self.__dict__.update(state2obj(state.state))

    def __add__(
        self, other: DictTransform | ConcatDictTransform
    ) -> ConcatDictTransform:
        """Concatenate another DictTransform to this one.

        This will modify the current configuration to include the transforms
        from the other configuration.
        """
        if not isinstance(other, DictTransform):
            raise TypeError(
                "Can only concatenate DictTransform or "
                "ConcatDictTransform objects."
            )
        if not isinstance(self, ConcatDictTransform):
            # if self is not a ConcatDictTransform, we need to create one
            new_transform = ConcatDictTransform.__new__(ConcatDictTransform)
            new_transform.cfg = ConcatDictTransformConfig(
                transforms=[self.cfg],
                input_columns=None,
            )
            new_transform._transforms = [self]
        else:
            new_transform = copy.copy(self)
        new_transform += other
        return new_transform

    @classmethod
    def from_config(
        cls: type[DictTransformType], config_path: str
    ) -> DictTransformType:
        """Load a DictTransform from a configuration file."""
        cfg = load_from(config_path, ensure_type=DictTransformConfig)
        return cfg()


DictTransformType = TypeVar(
    "DictTransformType", bound=DictTransform, covariant=True
)


class DictTransformConfig(ClassConfig[DictTransformType]):
    class_type: ClassType[DictTransformType]

    input_columns: dict[str, str] | Sequence[str] = Field(
        validation_alias=AliasChoices("input_column_mapping", "input_columns"),
    )
    """The input columns that need to be mapped to fit
    the transform's input_columns, or a list of input columns
    as extra input columns.

    If this is a dict, it should map the source column names
    to the destination column names, and the destination column names
    will be used as the input columns for the transform.

    """

    missing_input_columns_as_none: bool = Field(default=False)
    """If True, missing input columns will be set to None.

    If False, an error will be raised if any input column is missing."""

    output_column_mapping: dict[str, str] = Field(default_factory=dict)
    """The output columns that the transform will produce.
    This should be a mapping from the output column names to the
    names that the transform will use to return the transformed values.
    If the transform does not produce any output, this can be an empty dict.
    """

    check_return_columns: bool = False
    """Whether to check that the output columns of the transform
    match the expected output columns. If this is set to True,
    the transform will raise an error if the output columns do not
    match the expected output columns.

    For Transform that does not properly implement the `output_columns`
    property, or if the output columns are not known at the time of
    configuration, this should be set to False to avoid runtime errors.
    """

    keep_input_columns: bool = True
    """Whether to keep the input columns in the output dict.

    If this is set to True, the input columns will be included in the
    output dict. If this is set to False, the input columns will be
    removed from the output dict.
    """

    # overload + operator to return concat
    def __add__(
        self, other: DictTransformConfig | ConcatDictTransformConfig
    ) -> ConcatDictTransformConfig:
        """Concatenate two DictTransformConfig objects.

        This will create a new ConcatDictTransformConfig object that
        combines the transforms from both configurations.
        """
        if not isinstance(other, DictTransformConfig):
            raise TypeError(
                "Can only concatenate DictTransformConfig objects."
            )
        if not isinstance(self, ConcatDictTransformConfig):
            # if self is not a ConcatDictTransformConfig, we need to create one
            new_cfg = ConcatDictTransformConfig(
                transforms=[self],
                input_columns=None,
            )
        else:
            new_cfg = self.model_copy(deep=False)
        new_cfg += other
        return new_cfg


class ConcatDictTransform(DictTransform):
    cfg: ConcatDictTransformConfig

    def __init__(self, cfg: ConcatDictTransformConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self._transforms = [
            t() for t in self.cfg.transforms
        ]  # Instantiate all transforms

    def __getitem__(self, index: int) -> DictTransform:
        return self._transforms[index]

    def __iadd__(
        self, other: DictTransform | ConcatDictTransform
    ) -> ConcatDictTransform:
        """Concatenate another DictTransform to this one.

        This will modify the current configuration to include the transforms
        from the other configuration.
        """
        if not isinstance(other, DictTransform):
            raise TypeError(
                "Can only concatenate DictTransform or "
                "ConcatDictTransform objects."
            )
        if isinstance(other, ConcatDictTransform):
            self._transforms.extend(other._transforms)
        else:
            self._transforms.append(other)
        self.cfg += other.cfg
        return self

    @property
    def input_columns(self) -> list[str]:
        raise RuntimeError(
            "ConcatDictTransform does not implement input_columns. "
            "Use the mapped_input_columns property instead."
        )

    def _init_input_output_columns(self):
        input_columns: set[str] = set()
        output_columns: set[str] = set()
        for transform in self._transforms:
            cur_input_set = set(transform.mapped_input_columns)
            # consume the output columns if required as input
            to_consume = cur_input_set.intersection(output_columns)
            output_columns.difference_update(to_consume)
            # remove the input that is taken from the previous output
            # and update the input columns
            cur_input_set.difference_update(to_consume)
            input_columns.update(cur_input_set)
            # update the output columns
            output_columns.update(transform.mapped_output_columns)

        self._mapped_input_columns = list(input_columns)
        self._mapped_output_columns = list(output_columns)

    @functools.cached_property
    def mapped_input_columns(self) -> list[str]:
        """Get the input columns for this transform."""

        self._init_input_output_columns()
        return self._mapped_input_columns.copy()

    @property
    def output_columns(self) -> list[str]:
        """Get the output columns for this transform."""
        raise RuntimeError(
            "ConcatDictTransform does not implement output_columns. "
            "Use the mapped_output_columns property instead."
        )

    @functools.cached_property
    def mapped_output_columns(self) -> list[str]:
        """Get the output columns for this transform."""
        self._init_input_output_columns()
        return self._mapped_output_columns.copy()

    def transform(self, **kwargs) -> dict:
        raise RuntimeError(
            "ConcatDictTransform does not implement transform method. "
            "Use the __call__ method instead."
        )

    def __call__(self, row: dict) -> dict:
        """Concatenate the input columns into a new row dict.

        This method will apply all transforms in the order they are defined
        in the configuration, and return a new row dict with the transformed
        values added to the original row dict.
        """
        for transform in self._transforms:
            row = transform(row)
        return row

    def __getstate__(self) -> State:
        state = obj2state(
            dict(
                # use StateList to enable pickling the transforms separately
                transforms=StateList(self._transforms, save_to_path=True)
            )
        )
        return State(
            state=state,
            config=self.cfg,
            save_to_path=None,
            class_type=type(self),
        )

    def __setstate__(self, state: State) -> None:
        """Set the state of the object from the unpickled state."""
        self.cfg = state.config  # type: ignore
        self._transforms = state2obj(state.state["transforms"])


class ConcatDictTransformConfig(
    DictTransformConfig[ConcatDictTransform],
):
    class_type: ClassType[ConcatDictTransform] = ConcatDictTransform

    transforms: Sequence[
        ConfigInstanceOf[DictTransformConfig[DictTransform]]
    ] = Field(min_length=1)

    """A sequence of transforms to apply to the input columns."""

    input_columns: None = None
    """Input columns are not used in ConcatDictTransform, as the input columns
    are determined by the transforms themselves. This should be set to None.
    """

    def __getitem__(self, item):
        return self.transforms[item]

    # overload += operator
    def __iadd__(
        self, other: DictTransformConfig | ConcatDictTransformConfig
    ) -> ConcatDictTransformConfig:
        """Concatenate two DictTransformConfig objects in place.

        This will modify the current configuration to include the transforms
        from the other configuration.
        """
        if not isinstance(other, DictTransformConfig):
            raise TypeError(
                "Can only concatenate DictTransformConfig objects."
            )
        if isinstance(other, ConcatDictTransformConfig):
            self.transforms = list(self.transforms) + list(other.transforms)
        else:
            self.transforms = list(self.transforms) + [other]
        return self
