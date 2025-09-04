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
import pickle
from dataclasses import dataclass
from typing import Sequence

import pytest
import torch
from pydantic import AliasChoices, BaseModel, Field

from robo_orchard_lab.dataset.transforms import (
    ClassType,
    ConcatDictTransform,
    ConcatDictTransformConfig,
    DictTransform,
    DictTransformConfig,
)
from robo_orchard_lab.utils.state import State


@dataclass
class DummyDataclassReturn:
    value: int


class DummyBaseModelReturn(BaseModel):
    value: int


class DummyTransform(DictTransform):
    """A simple dummy transform for testing purposes."""

    cfg: DummyTransformConfig

    def __init__(self, cfg: DummyTransformConfig) -> None:
        self.cfg = cfg

    @property
    def output_columns(self):
        return ["value"]

    def transform(self, value: int) -> dict:
        return {"value": value + self.cfg.add_value}


class DummyDataclassTransform(DictTransform):
    """A dummy transform that returns a dataclass."""

    cfg: DummyTransformConfig

    def __init__(self, cfg: DummyTransformConfig) -> None:
        self.cfg = cfg

    def transform(self, value: int) -> DummyDataclassReturn:
        return DummyDataclassReturn(value=value + self.cfg.add_value)


class DummyBaseModelTransform(DictTransform):
    """A dummy transform that returns a BaseModel."""

    cfg: DummyTransformConfig

    def __init__(self, cfg: DummyTransformConfig) -> None:
        self.cfg = cfg

    def transform(self, value: int) -> DummyBaseModelReturn:
        return DummyBaseModelReturn(value=value + self.cfg.add_value)


class DummyTransformNoOutputColumns(DictTransform):
    """A simple dummy transform for testing purposes."""

    cfg: DummyTransformConfig

    def __init__(self, cfg: DummyTransformConfig) -> None:
        self.cfg = cfg

    def transform(self, value: int) -> dict:
        return {"value": value + self.cfg.add_value}


DummyTransformType = (
    DummyTransform
    | DummyDataclassTransform
    | DummyBaseModelTransform
    | DummyTransformNoOutputColumns
)


class DummyTransformConfig(DictTransformConfig[DummyTransformType]):
    """Configuration for the DummyTransform."""

    class_type: ClassType[DummyTransformType] = DummyTransform

    add_value: int

    input_columns: dict[str, str] | Sequence[str] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("input_column_mapping", "input_columns"),
    )


class TestDictTransforms:
    """Tests for the DictTransform and ConcatDictTransform."""

    def test_dict_transform(self):
        cfg = DummyTransformConfig(add_value=10)
        transform = DummyTransform(cfg)
        src = {"value": 5}
        result = transform(src)
        assert result == {"value": 15}
        assert set(transform.input_columns) == set(["value"])
        assert set(transform.mapped_input_columns) == set(["value"])
        assert set(transform.output_columns) == set(["value"])
        assert set(transform.mapped_output_columns) == set(["value"])

    def test_dataclass_transform(self):
        cfg = DummyTransformConfig(
            add_value=10, class_type=DummyDataclassTransform
        )
        transform = cfg()
        src = {"value": 5}
        result = transform(src)
        assert result == {"value": 15}
        assert set(transform.input_columns) == set(["value"])
        assert set(transform.mapped_input_columns) == set(["value"])
        assert set(transform.output_columns) == set(["value"])
        assert set(transform.mapped_output_columns) == set(["value"])

    def test_basemodel_transform(self):
        cfg = DummyTransformConfig(
            add_value=10, class_type=DummyBaseModelTransform
        )
        transform = cfg()
        src = {"value": 5}
        result = transform(src)
        assert result == {"value": 15}
        assert set(transform.input_columns) == set(["value"])
        assert set(transform.mapped_input_columns) == set(["value"])
        assert set(transform.output_columns) == set(["value"])
        assert set(transform.mapped_output_columns) == set(["value"])

    def test_transform_no_output_columns(self):
        cfg = DummyTransformConfig(
            add_value=10, class_type=DummyTransformNoOutputColumns
        )
        transform = cfg()
        with pytest.raises(NotImplementedError) as e:
            # No output_columns defined, should raise an error
            assert set(transform.output_columns) == set(["value"])
        print(e)

    def test_transform_input_dismatch(self):
        cfg = DummyTransformConfig(add_value=10)
        transform = DummyTransform(cfg)
        src = {"data": 5}
        with pytest.raises(KeyError) as e:
            transform(src)
        print(e)

    def test_transform_input_mapping(self):
        cfg = DummyTransformConfig(
            add_value=10,
            input_column_mapping={"input_value": "value"},
        )
        transform = DummyTransform(cfg)
        src = {"input_value": 5}
        result = transform(src)
        assert result["value"] == 15
        assert result["input_value"] == 5
        assert set(transform.input_columns) == set(["value"])
        assert set(transform.mapped_input_columns) == set(["input_value"])
        assert set(transform.output_columns) == set(["value"])
        assert set(transform.mapped_output_columns) == set(["value"])

    def test_transform_output_mapping(self):
        cfg = DummyTransformConfig(
            add_value=10,
            input_column_mapping={"input_value": "value"},
            output_column_mapping={"value": "output_value"},
        )
        transform = DummyTransform(cfg)
        src = {"input_value": 5}
        result = transform(src)
        assert result["output_value"] == 15
        assert result["input_value"] == 5
        assert set(transform.input_columns) == set(["value"])
        assert set(transform.mapped_input_columns) == set(["input_value"])
        assert set(transform.output_columns) == set(["value"])
        assert set(transform.mapped_output_columns) == set(["output_value"])

    def test_transform_output_mapping_overwrite(self):
        cfg = DummyTransformConfig(
            add_value=10,
            input_column_mapping={"input_value": "value"},
            output_column_mapping={"value": "input_value"},
        )
        transform = DummyTransform(cfg)
        src = {"input_value": 5}
        result = transform(src)
        assert result["input_value"] == 15
        assert len(result.keys()) == 1

    def test_pickle_dumps(self):
        cfg = DummyTransformConfig(
            add_value=10, class_type=DummyBaseModelTransform
        )
        ts = cfg()
        ts.tensor = torch.tensor([1, 2, 3])  # type: ignore
        ts_bytes = pickle.dumps(ts)
        state = ts.__getstate__()
        assert "tensor" in state.state

        ts_recovered = pickle.loads(ts_bytes)
        print(ts_recovered)
        print(ts_recovered.tensor)


class TestConcatDictTransform:
    """Tests for the ConcatDictTransform."""

    def test_concat_transform(self):
        cfg = ConcatDictTransformConfig(
            transforms=[
                DummyTransformConfig(add_value=10),
                DummyTransformConfig(add_value=20),
            ]
        )
        transform = ConcatDictTransform(cfg)
        src = {"value": 5}
        result = transform(src)
        assert result["value"] == 35  # 5 + 10 + 20
        assert set(transform.mapped_input_columns) == set(["value"])
        assert set(transform.mapped_output_columns) == set(["value"])

    def test_concat_transform_input_mapping(self):
        cfg = ConcatDictTransformConfig(
            transforms=[
                DummyTransformConfig(
                    add_value=10, input_column_mapping={"input_value": "value"}
                ),
                DummyTransformConfig(add_value=20),
            ]
        )
        transform = ConcatDictTransform(cfg)
        src = {"input_value": 5}
        result = transform(src)
        assert result["value"] == 35
        assert result["input_value"] == 5
        assert set(transform.mapped_input_columns) == set(["input_value"])
        assert set(transform.mapped_output_columns) == set(["value"])

    def test_pickle_dumps(self):
        cfg = ConcatDictTransformConfig(
            transforms=[
                DummyTransformConfig(add_value=10),
                DummyTransformConfig(add_value=20),
            ]
        )
        transform = cfg()
        transform._transforms[0].tensor = torch.tensor([1, 2, 3])  # type: ignore

        ts_bytes = pickle.dumps(transform)
        transform_recovered = pickle.loads(ts_bytes)
        print(transform_recovered._transforms[0].tensor)

    def test_getstate_no_mixin_cls(self):
        cfg = ConcatDictTransformConfig(
            transforms=[
                DummyTransformConfig(add_value=10),
                DummyTransformConfig(add_value=20),
            ]
        )
        transform = cfg()
        transform._transforms[0].tensor = torch.tensor([1, 2, 3])  # type: ignore

        state = transform.__getstate__()
        print("state: ", state)
        for st in state.state["transforms"]:
            assert isinstance(st, State)
