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
import glob
import importlib
import os
import pickle
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

import pytest
import torch
import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from robo_orchard_core.utils.config import ClassType

import robo_orchard_lab.transforms as transforms_pkg
import robo_orchard_lab.transforms.base as transforms_base
from robo_orchard_lab.transforms import (
    DictRowTransform,
    DictRowTransformConfig,
    DictTransform,
    DictTransformConfig,
    DictTransformPipeline,
    DictTransformPipelineConfig,
)
from robo_orchard_lab.transforms.base import (
    ConcatDictTransform,
    ConcatDictTransformConfig,
    semantic_output_to_dict,
)


@dataclass
class DummyDataclassReturn:
    value: int


@dataclass
class DummyTensorDataclassReturn:
    value: torch.Tensor


class DummyBaseModelReturn(BaseModel):
    value: int


class DummyTensorBaseModelReturn(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: torch.Tensor


class DummyTransform(DictTransform[dict[str, int]]):
    """A simple dummy transform for testing purposes."""

    cfg: DummyTransformConfig

    def __init__(self, cfg: DummyTransformConfig) -> None:
        self.cfg = cfg

    @property
    def output_columns(self):
        return ["value"]

    def transform(self, value: int) -> dict:
        return {"value": value + self.cfg.add_value}


class DummyDataclassTransform(DictTransform[DummyDataclassReturn]):
    """A dummy transform that returns a dataclass."""

    cfg: DummyTransformConfig

    def __init__(self, cfg: DummyTransformConfig) -> None:
        self.cfg = cfg

    def transform(self, value: int) -> DummyDataclassReturn:
        return DummyDataclassReturn(value=value + self.cfg.add_value)


class DummyBaseModelTransform(DictTransform[DummyBaseModelReturn]):
    """A dummy transform that returns a BaseModel."""

    cfg: DummyTransformConfig

    def __init__(self, cfg: DummyTransformConfig) -> None:
        self.cfg = cfg

    def transform(self, value: int) -> DummyBaseModelReturn:
        return DummyBaseModelReturn(value=value + self.cfg.add_value)


class DummyTensorDataclassTransform(DictTransform[DummyTensorDataclassReturn]):
    """A dummy transform that returns a dataclass tensor payload."""

    cfg: DummyTransformConfig

    def __init__(self, cfg: DummyTransformConfig) -> None:
        self.cfg = cfg

    def transform(self, value: torch.Tensor) -> DummyTensorDataclassReturn:
        return DummyTensorDataclassReturn(value=value)


class DummyTensorBaseModelTransform(DictTransform[DummyTensorBaseModelReturn]):
    """A dummy transform that returns a BaseModel tensor payload."""

    cfg: DummyTransformConfig

    def __init__(self, cfg: DummyTransformConfig) -> None:
        self.cfg = cfg

    def transform(self, value: torch.Tensor) -> DummyTensorBaseModelReturn:
        return DummyTensorBaseModelReturn(value=value)


class DummyTransformNoOutputColumns(DictTransform[dict[str, int]]):
    """A simple dummy transform for testing purposes."""

    cfg: DummyTransformConfig

    def __init__(self, cfg: DummyTransformConfig) -> None:
        self.cfg = cfg

    def transform(self, value: int) -> dict:
        return {"value": value + self.cfg.add_value}


class StatefulCounterTransform(DictTransform[dict[str, int]]):
    """A transform with mutable runtime state for shared-reference tests."""

    cfg: StatefulCounterTransformConfig

    def __init__(self, cfg: StatefulCounterTransformConfig) -> None:
        self.cfg = cfg
        self.calls = 0

    @property
    def output_columns(self) -> list[str]:
        return ["value"]

    def transform(self, value: int) -> dict[str, int]:
        self.calls += 1
        return {"value": value + self.cfg.add_value + self.calls}


class OrderedColumnsTransform(DictTransform[dict[str, int]]):
    """A helper transform with explicit ordered input and output metadata."""

    cfg: OrderedColumnsTransformConfig

    def __init__(self, cfg: OrderedColumnsTransformConfig) -> None:
        self.cfg = cfg

    @property
    def output_columns(self) -> list[str]:
        return list(self.cfg.output_columns_order)

    def transform(self, **kwargs: int) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.output_columns)}


DummyTransformType = (
    DummyTransform
    | DummyDataclassTransform
    | DummyBaseModelTransform
    | DummyTensorDataclassTransform
    | DummyTensorBaseModelTransform
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


class StatefulCounterTransformConfig(
    DictTransformConfig[StatefulCounterTransform]
):
    class_type: ClassType[StatefulCounterTransform] = StatefulCounterTransform

    add_value: int = 0


class OrderedColumnsTransformConfig(
    DictTransformConfig[OrderedColumnsTransform]
):
    class_type: ClassType[OrderedColumnsTransform] = OrderedColumnsTransform

    output_columns_order: Sequence[str] = Field(default_factory=tuple)
    input_columns: dict[str, str] | Sequence[str] = Field(
        default_factory=tuple,
        validation_alias=AliasChoices("input_column_mapping", "input_columns"),
    )


class KwargOnlyTransform(DictTransform[dict[str, int]]):
    """A transform that relies on kwargs-based config instantiation."""

    InitFromConfig = False

    cfg: KwargOnlyTransformConfig

    def __init__(self, add_value: int, **kwargs: Any) -> None:
        self.cfg = KwargOnlyTransformConfig(add_value=add_value, **kwargs)

    @property
    def output_columns(self) -> list[str]:
        return ["value"]

    def transform(self, value: int) -> dict[str, int]:
        return {"value": value + self.cfg.add_value}


class KwargOnlyTransformConfig(DictTransformConfig[KwargOnlyTransform]):
    class_type: ClassType[KwargOnlyTransform] = KwargOnlyTransform

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

    def test_apply_returns_structured_dataclass_and_final_row(self):
        cfg = DummyTransformConfig(
            add_value=10, class_type=DummyDataclassTransform
        )
        transform = cfg()

        structured_result, final_row = transform.apply({"value": 5})

        assert isinstance(structured_result, DummyDataclassReturn)
        assert structured_result.value == 15
        assert final_row == {"value": 15}
        assert transform({"value": 5}) == final_row

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

    def test_apply_returns_structured_basemodel_and_final_row(self):
        cfg = DummyTransformConfig(
            add_value=10, class_type=DummyBaseModelTransform
        )
        transform = cfg()

        structured_result, final_row = transform.apply({"value": 5})

        assert isinstance(structured_result, DummyBaseModelReturn)
        assert structured_result.value == 15
        assert final_row == {"value": 15}
        assert transform({"value": 5}) == final_row

    def test_semantic_output_to_dict_supports_structured_outputs(self):
        assert semantic_output_to_dict({"value": 1}) == {"value": 1}
        assert semantic_output_to_dict(DummyDataclassReturn(value=2)) == {
            "value": 2
        }
        assert semantic_output_to_dict(DummyBaseModelReturn(value=3)) == {
            "value": 3
        }
        with pytest.raises(TypeError):
            semantic_output_to_dict(object())

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
            input_columns={"input_value": "value"},
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
            input_columns={"input_value": "value"},
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

    def test_apply_preserves_raw_output_before_output_mapping(self):
        cfg = DummyTransformConfig(
            add_value=10,
            output_column_mapping={"value": "output_value"},
            check_return_columns=True,
        )
        transform = cfg()

        structured_result, final_row = transform.apply({"value": 5})

        assert structured_result == {"value": 15}
        assert final_row == {"value": 5, "output_value": 15}

    def test_transform_output_mapping_overwrite(self):
        cfg = DummyTransformConfig(
            add_value=10,
            input_columns={"input_value": "value"},
            output_column_mapping={"value": "input_value"},
        )
        transform = DummyTransform(cfg)
        src = {"input_value": 5}
        result = transform(src)
        assert result["input_value"] == 15
        assert len(result.keys()) == 1

    def test_apply_preserves_dataclass_tensor_reference(self):
        cfg = DummyTransformConfig(
            add_value=0,
            class_type=DummyTensorDataclassTransform,
        )
        transform = cfg()
        tensor = torch.tensor([1, 2, 3])

        structured_result, final_row = transform.apply({"value": tensor})

        assert isinstance(structured_result, DummyTensorDataclassReturn)
        assert structured_result.value is tensor
        assert final_row["value"] is tensor

    def test_apply_preserves_basemodel_tensor_reference(self):
        cfg = DummyTransformConfig(
            add_value=0,
            class_type=DummyTensorBaseModelTransform,
        )
        transform = cfg()
        tensor = torch.tensor([1, 2, 3])

        structured_result, final_row = transform.apply({"value": tensor})

        assert isinstance(structured_result, DummyTensorBaseModelReturn)
        assert structured_result.value is tensor
        assert final_row["value"] is tensor

    def test_pickle_dumps(self):
        cfg = DummyTransformConfig(
            add_value=10, class_type=DummyBaseModelTransform
        )
        ts = cfg()
        ts.tensor = torch.tensor([1, 2, 3])  # type: ignore
        ts_bytes = pickle.dumps(ts)
        state = ts._get_state()
        assert "tensor" in state.state

        ts_recovered = pickle.loads(ts_bytes)
        recovered_any = cast(Any, ts_recovered)
        ts_any = cast(Any, ts)
        print(ts_recovered)
        print(recovered_any.tensor)
        assert torch.equal(recovered_any.tensor, ts_any.tensor)

    def test_save_and_load(self, tmp_local_folder: str):
        cfg = DummyTransformConfig(
            add_value=10, class_type=DummyBaseModelTransform
        )
        ts = cfg()

        ts.tensor = torch.tensor([1, 2, 3])  # type: ignore

        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            ts.save(save_path)
            print(glob.glob(os.path.join(save_path, "**"), recursive=True))
            ts_recovered = DummyTransform.load(save_path)
            recovered_any = cast(Any, ts_recovered)
            ts_any = cast(Any, ts)
            print(ts_recovered)
            print(recovered_any.tensor)
            assert torch.equal(recovered_any.tensor, ts_any.tensor)

    def test_structural_field_mutation_updates_live_metadata_and_runtime(self):
        cfg = DummyTransformConfig(
            add_value=10,
            input_columns={"input_value": "value"},
            output_column_mapping={"value": "output_value"},
        )
        transform = cfg()

        assert isinstance(cfg.input_columns, dict)
        cfg.input_columns.pop("input_value")
        cfg.input_columns["renamed_input"] = "value"
        cfg.output_column_mapping["value"] = "renamed_value"

        assert transform.mapped_input_columns == ["renamed_input"]
        assert transform.mapped_output_columns == ["renamed_value"]
        assert transform({"renamed_input": 5}) == {
            "renamed_input": 5,
            "renamed_value": 15,
        }

    def test_reflection_metadata_is_cached_across_repeated_access(self):
        class CachedReflectionTransform(DictTransform[DummyDataclassReturn]):
            cfg: DummyTransformConfig

            def __init__(self, cfg: DummyTransformConfig) -> None:
                self.cfg = cfg

            def transform(
                self,
                value: int,
                optional_bias: int = 0,
            ) -> DummyDataclassReturn:
                return DummyDataclassReturn(
                    value=value + self.cfg.add_value + optional_bias
                )

        transforms_base._get_cached_transform_reflection_metadata.cache_clear()
        signature_calls = 0
        original_signature = transforms_base.inspect.signature

        def counting_signature(*args: Any, **kwargs: Any):
            nonlocal signature_calls
            signature_calls += 1
            return original_signature(*args, **kwargs)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            transforms_base.inspect,
            "signature",
            counting_signature,
        )
        try:
            transform = CachedReflectionTransform(
                DummyTransformConfig(add_value=10)
            )

            assert transform.input_columns == ["value", "optional_bias"]
            assert transform.output_columns == ["value"]
            assert transform.input_columns == ["value", "optional_bias"]
            assert transform.output_columns == ["value"]
            assert transform.apply({"value": 5}) == (
                DummyDataclassReturn(value=15),
                {"value": 15},
            )
            assert signature_calls == 1
        finally:
            monkeypatch.undo()
            transforms_base._get_cached_transform_reflection_metadata.cache_clear()

    def test_input_column_metadata_cache_tracks_cfg_values(self):
        transforms_base._get_cached_input_column_views.cache_clear()
        try:
            cfg = DummyTransformConfig(
                add_value=10,
                input_columns={"input_value": "value"},
            )
            transform = cfg()

            assert transform.input_columns == ["value"]
            assert transform.mapped_input_columns == ["input_value"]

            input_cache_info = (
                transforms_base._get_cached_input_column_views.cache_info()
            )
            assert input_cache_info.misses == 1

            assert transform.input_columns == ["value"]
            assert transform.mapped_input_columns == ["input_value"]

            input_cache_info = (
                transforms_base._get_cached_input_column_views.cache_info()
            )
            assert input_cache_info.hits >= 3

            input_columns_mapping = cast(dict[str, str], cfg.input_columns)
            input_columns_mapping.pop("input_value")
            input_columns_mapping["renamed_input"] = "value"

            assert transform.input_columns == ["value"]
            assert transform.mapped_input_columns == ["renamed_input"]

            input_cache_info = (
                transforms_base._get_cached_input_column_views.cache_info()
            )
            assert input_cache_info.misses == 2
        finally:
            transforms_base._get_cached_input_column_views.cache_clear()


class TestTransformDeprecation:
    def test_dataset_transform_package_reload_warns(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            module = importlib.import_module(
                "robo_orchard_lab.dataset.transforms"
            )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            reloaded_module = importlib.reload(module)

        assert reloaded_module is module
        assert [str(item.message) for item in caught] == [
            "`robo_orchard_lab.dataset.transforms` is deprecated. "
            "Use `robo_orchard_lab.transforms` instead."
        ]

    def test_concat_compatibility_classes_emit_deprecation_warnings(self):
        with pytest.deprecated_call(
            match="ConcatDictTransformConfig is deprecated"
        ):
            cfg = ConcatDictTransformConfig(
                transforms=[DummyTransformConfig(add_value=10)]
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg = ConcatDictTransformConfig(
                transforms=[DummyTransformConfig(add_value=10)]
            )

        with pytest.deprecated_call(match="ConcatDictTransform is deprecated"):
            transform = ConcatDictTransform(cfg)

        assert transform({"value": 5}) == {"value": 15}


@pytest.mark.filterwarnings(
    "ignore:ConcatDictTransform is deprecated.*:DeprecationWarning"
)
@pytest.mark.filterwarnings(
    "ignore:ConcatDictTransformConfig is deprecated.*:DeprecationWarning"
)
class TestConcatDictTransform:
    """Tests for the ConcatDictTransform."""

    def test_dataset_transform_root_matches_canonical_transform_root(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            dataset_transforms = importlib.import_module(
                "robo_orchard_lab.dataset.transforms"
            )

        assert dataset_transforms.__all__ == transforms_pkg.__all__

    def test_package_roots_keep_compatibility_exports(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            dataset_transforms = importlib.import_module(
                "robo_orchard_lab.dataset.transforms"
            )

        compat_exports = {
            "ConcatDictTransform",
            "ConcatDictTransformConfig",
            "Config",
            "ClassType",
            "ConfigInstanceOf",
            "DictTransformType",
        }

        assert compat_exports.issubset(set(transforms_pkg.__all__))
        assert compat_exports.issubset(set(dataset_transforms.__all__))
        assert transforms_pkg.ConcatDictTransform is ConcatDictTransform
        assert (
            transforms_pkg.ConcatDictTransformConfig
            is ConcatDictTransformConfig
        )
        assert dataset_transforms.ConcatDictTransform is ConcatDictTransform
        assert (
            dataset_transforms.ConcatDictTransformConfig
            is ConcatDictTransformConfig
        )
        assert transforms_pkg.Config is transforms_base.Config
        assert transforms_pkg.ClassType is ClassType
        assert (
            dataset_transforms.ConfigInstanceOf
            is transforms_base.ConfigInstanceOf
        )

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
        assert isinstance(transform, DictTransformPipeline)
        assert isinstance(transform, DictTransform)
        assert transform.cfg.transforms[0] is transform[0].cfg

    def test_concat_weak_dicttransform_compatibility_is_explicit(self):
        transform = ConcatDictTransformConfig(
            transforms=[DummyTransformConfig(add_value=10)]
        )()

        assert isinstance(transform, DictTransform)
        with pytest.raises(
            RuntimeError, match="does not implement input_columns"
        ):
            _ = transform.input_columns
        with pytest.raises(
            RuntimeError, match="does not implement output_columns"
        ):
            _ = transform.output_columns
        with pytest.raises(RuntimeError, match="does not implement transform"):
            transform.transform(value=5)

    def test_concat_transform_input_mapping(self):
        cfg = ConcatDictTransformConfig(
            transforms=[
                DummyTransformConfig(
                    add_value=10, input_columns={"input_value": "value"}
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

    def test_concat_apply_returns_last_child_structured_output(self):
        cfg = ConcatDictTransformConfig(
            transforms=[
                DummyTransformConfig(add_value=10),
                DummyTransformConfig(
                    add_value=20,
                    class_type=DummyDataclassTransform,
                ),
            ]
        )
        transform = ConcatDictTransform(cfg)

        structured_result, final_row = transform.apply({"value": 5})

        assert isinstance(structured_result, DummyDataclassReturn)
        assert structured_result.value == 35
        assert final_row == {"value": 35}
        assert transform({"value": 5}) == final_row

    @pytest.mark.parametrize(
        ("field_name", "field_value", "message"),
        [
            (
                "missing_input_columns_as_none",
                True,
                "does not support missing_input_columns_as_none",
            ),
            (
                "output_column_mapping",
                {"value": "renamed_value"},
                "does not support output_column_mapping",
            ),
            (
                "check_return_columns",
                True,
                "does not support check_return_columns",
            ),
            (
                "keep_input_columns",
                False,
                "does not support keep_input_columns",
            ),
        ],
    )
    def test_concat_config_rejects_unsupported_outer_fields(
        self,
        field_name: str,
        field_value: object,
        message: str,
    ):
        kwargs = {
            "transforms": [DummyTransformConfig(add_value=10)],
            field_name: field_value,
        }

        with pytest.raises(ValueError, match=message):
            ConcatDictTransformConfig(**kwargs)

    def test_concat_config_accepts_explicit_legacy_default_fields(self):
        cfg = ConcatDictTransformConfig.model_validate(
            {
                "transforms": [
                    DummyTransformConfig(add_value=10).model_dump()
                ],
                "input_columns": None,
                "missing_input_columns_as_none": False,
                "output_column_mapping": {},
                "check_return_columns": False,
                "keep_input_columns": True,
            }
        )

        assert cfg.input_columns is None
        assert cfg.missing_input_columns_as_none is False
        assert cfg.output_column_mapping == {}
        assert cfg.check_return_columns is False
        assert cfg.keep_input_columns is True

    def test_concat_iadd_updates_mapped_columns(self):
        transform = ConcatDictTransformConfig(
            transforms=[DummyTransformConfig(add_value=10)]
        )()

        transform += DummyTransformConfig(
            add_value=20,
            output_column_mapping={"value": "output_value"},
        )()

        assert set(transform.mapped_input_columns) == {"value"}
        assert set(transform.mapped_output_columns) == {"output_value"}

    @pytest.mark.parametrize(
        ("field_name", "field_value", "message"),
        [
            (
                "missing_input_columns_as_none",
                True,
                "does not support missing_input_columns_as_none",
            ),
            (
                "output_column_mapping",
                {"value": "renamed_value"},
                "does not support output_column_mapping",
            ),
            (
                "check_return_columns",
                True,
                "does not support check_return_columns",
            ),
            (
                "keep_input_columns",
                False,
                "does not support keep_input_columns",
            ),
        ],
    )
    def test_concat_config_rejects_unsupported_outer_field_assignment(
        self,
        field_name: str,
        field_value: object,
        message: str,
    ):
        cfg = ConcatDictTransformConfig(
            transforms=[DummyTransformConfig(add_value=10)]
        )

        with pytest.raises(ValueError, match=message):
            setattr(cfg, field_name, field_value)

        assert hasattr(cfg, field_name)

    @pytest.mark.parametrize(
        ("field_name", "field_value"),
        [
            ("input_columns", None),
            ("missing_input_columns_as_none", False),
            ("output_column_mapping", {}),
            ("check_return_columns", False),
            ("keep_input_columns", True),
        ],
    )
    def test_concat_config_allows_default_outer_field_assignment(
        self,
        field_name: str,
        field_value: object,
    ):
        cfg = ConcatDictTransformConfig(
            transforms=[DummyTransformConfig(add_value=10)]
        )

        setattr(cfg, field_name, field_value)

        assert getattr(cfg, field_name) == field_value

    def test_concat_preserves_mapped_column_order(self):
        transform = ConcatDictTransformConfig(
            transforms=[
                OrderedColumnsTransformConfig(
                    input_columns=["src_b", "src_a"],
                    output_columns_order=["mid_b", "mid_a"],
                ),
                OrderedColumnsTransformConfig(
                    input_columns=["mid_a", "mid_b", "src_c"],
                    output_columns_order=["final_a", "final_b"],
                ),
            ]
        )()

        assert transform.mapped_input_columns == [
            "src_b",
            "src_a",
            "src_c",
        ]
        assert transform.mapped_output_columns == [
            "final_a",
            "final_b",
        ]

    def test_concat_add_normalizes_to_pipeline_type(self):
        transform = ConcatDictTransformConfig(
            transforms=[DummyTransformConfig(add_value=10)]
        )()

        extended = transform + DummyTransformConfig(add_value=20)()

        assert isinstance(extended, DictTransformPipeline)
        assert not isinstance(extended, ConcatDictTransform)
        assert isinstance(extended.cfg, DictTransformPipelineConfig)
        assert extended[0] is transform[0]
        assert extended.cfg.transforms[0] is transform.cfg.transforms[0]

    def test_leaf_add_concat_normalizes_to_pipeline_type(self):
        leaf = DummyTransformConfig(add_value=10)()
        concat = ConcatDictTransformConfig(
            transforms=[DummyTransformConfig(add_value=20)]
        )()

        extended = leaf + concat

        assert isinstance(extended, DictTransformPipeline)
        assert not isinstance(extended, ConcatDictTransform)
        assert isinstance(extended.cfg, DictTransformPipelineConfig)
        assert extended[0] is leaf
        assert extended[1] is concat[0]

    def test_leaf_config_add_concat_config_normalizes_to_pipeline_type(self):
        leaf_cfg = DummyTransformConfig(add_value=10)
        concat_cfg = ConcatDictTransformConfig(
            transforms=[DummyTransformConfig(add_value=20)]
        )

        extended_cfg = leaf_cfg + concat_cfg

        assert isinstance(extended_cfg, DictTransformPipelineConfig)
        assert not isinstance(extended_cfg, ConcatDictTransformConfig)
        assert extended_cfg.transforms[0] is leaf_cfg
        assert extended_cfg.transforms[1] is concat_cfg.transforms[0]

    def test_leaf_add_leaf_returns_pipeline_type(self):
        extended = (
            DummyTransformConfig(add_value=10)()
            + DummyTransformConfig(add_value=20)()
        )
        extended_cfg = DummyTransformConfig(
            add_value=10
        ) + DummyTransformConfig(add_value=20)

        assert isinstance(extended, DictTransformPipeline)
        assert not isinstance(extended, ConcatDictTransform)
        assert isinstance(extended.cfg, DictTransformPipelineConfig)
        assert isinstance(extended_cfg, DictTransformPipelineConfig)
        assert not isinstance(extended_cfg, ConcatDictTransformConfig)

    def test_pipeline_add_concat_normalizes_to_pipeline_type(self):
        pipeline = (
            DummyTransformConfig(add_value=10)()
            + DummyTransformConfig(add_value=20)()
        )
        concat = ConcatDictTransformConfig(
            transforms=[DummyTransformConfig(add_value=30)]
        )()

        extended = pipeline + concat

        assert isinstance(extended, DictTransformPipeline)
        assert not isinstance(extended, ConcatDictTransform)
        assert isinstance(extended.cfg, DictTransformPipelineConfig)
        assert extended({"value": 5}) == {"value": 65}

    def test_pipeline_config_add_concat_config_normalizes_to_pipeline_type(
        self,
    ):
        pipeline_cfg = DummyTransformConfig(
            add_value=10
        ) + DummyTransformConfig(add_value=20)
        concat_cfg = ConcatDictTransformConfig(
            transforms=[DummyTransformConfig(add_value=30)]
        )

        extended_cfg = pipeline_cfg + concat_cfg

        assert isinstance(extended_cfg, DictTransformPipelineConfig)
        assert not isinstance(extended_cfg, ConcatDictTransformConfig)
        assert extended_cfg()({"value": 5}) == {"value": 65}

    def test_concat_config_output_column_mapping_is_read_only(self):
        cfg = ConcatDictTransformConfig(
            transforms=[DummyTransformConfig(add_value=10)]
        )

        with pytest.raises(TypeError, match="read-only"):
            cfg.output_column_mapping["value"] = "renamed_value"

        assert cfg()({"value": 5})["value"] == 15

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
        recovered_first = cast(Any, transform_recovered._transforms[0])
        print(recovered_first.tensor)

    def test_save_and_load(self, tmp_local_folder: str):
        cfg = ConcatDictTransformConfig(
            transforms=[
                DummyTransformConfig(add_value=10),
                DummyTransformConfig(add_value=20),
            ]
        )
        transform = cfg()
        transform._transforms[0].tensor = torch.tensor([1, 2, 3])  # type: ignore

        with tempfile.TemporaryDirectory(dir=tmp_local_folder) as save_path:
            transform.save(save_path)

            transform_recovered = ConcatDictTransform.load(save_path)
            recovered_first = cast(Any, transform_recovered._transforms[0])
            first = cast(Any, transform._transforms[0])
            print(transform_recovered)
            print(recovered_first.tensor)
            assert torch.equal(recovered_first.tensor, first.tensor)
            assert (
                transform_recovered.cfg.transforms[0]
                is transform_recovered[0].cfg
            )

    def test_getstate_no_mixin_cls(self):
        cfg = ConcatDictTransformConfig(
            transforms=[
                DummyTransformConfig(add_value=10),
                DummyTransformConfig(add_value=20),
            ]
        )
        transform = cfg()
        transform._transforms[0].tensor = torch.tensor([1, 2, 3])  # type: ignore

        state = transform._get_state()
        print("state: ", state)
        for st in state.state["transforms"]:
            assert st is not None


class TestDictTransformPipeline:
    """Tests for the preferred pipeline composition path."""

    def test_dicttransform_from_config(self, tmp_path: Path):
        config_path = tmp_path / "transform.yaml"
        config_path.write_text(
            yaml.safe_dump(
                DummyTransformConfig(add_value=10).model_dump(mode="json")
            ),
            encoding="utf-8",
        )

        transform = DummyTransform.from_config(str(config_path))

        assert isinstance(transform, DummyTransform)
        assert transform({"value": 5}) == {"value": 15}

    def test_dicttransform_from_config_loads_pipeline_yaml(
        self, tmp_path: Path
    ):
        config_path = tmp_path / "pipeline_transform.yaml"
        config_path.write_text(
            yaml.safe_dump(
                DictTransformPipelineConfig(
                    transforms=[
                        DummyTransformConfig(add_value=10),
                        DummyTransformConfig(add_value=20),
                    ]
                ).model_dump(mode="json")
            ),
            encoding="utf-8",
        )

        transform = DictTransform.from_config(str(config_path))

        assert isinstance(transform, DictTransformPipeline)
        assert transform({"value": 5}) == {"value": 35}

    def test_dicttransform_from_config_loads_legacy_concat_yaml(
        self, tmp_path: Path
    ):
        config_path = tmp_path / "concat_transform.yaml"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config_path.write_text(
                yaml.safe_dump(
                    ConcatDictTransformConfig(
                        transforms=[DummyTransformConfig(add_value=10)]
                    ).model_dump(mode="json")
                ),
                encoding="utf-8",
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            transform = DictTransform.from_config(str(config_path))
            concat_transform = ConcatDictTransform.from_config(
                str(config_path)
            )

        assert isinstance(transform, ConcatDictTransform)
        assert isinstance(concat_transform, ConcatDictTransform)
        assert transform({"value": 5}) == {"value": 15}
        assert concat_transform({"value": 5}) == {"value": 15}

    def test_legacy_concat_config_dump_uses_base_module_paths(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg_dump = ConcatDictTransformConfig(
                transforms=[DummyTransformConfig(add_value=10)]
            ).model_dump(mode="json")

        assert (
            cfg_dump["__config_type__"]
            == "robo_orchard_lab.transforms.base:ConcatDictTransformConfig"
        )
        assert (
            cfg_dump["class_type"]
            == "robo_orchard_lab.transforms.base:ConcatDictTransform"
        )

    def test_nested_pipeline_config_constructor_flattens_and_runs(self):
        inner_cfg = DictTransformPipelineConfig(
            transforms=[
                DummyTransformConfig(add_value=10),
                DummyTransformConfig(add_value=20),
            ]
        )
        outer_cfg = DictTransformPipelineConfig(
            transforms=[inner_cfg, DummyTransformConfig(add_value=30)]
        )

        transform = outer_cfg()

        assert len(outer_cfg.transforms) == 3
        assert outer_cfg.transforms[0] is inner_cfg.transforms[0]
        assert outer_cfg.transforms[1] is inner_cfg.transforms[1]
        assert transform({"value": 5}) == {"value": 65}

    def test_pipeline_constructor_respects_config_call_contract(self):
        cfg = KwargOnlyTransformConfig(add_value=10)

        assert cfg()({"value": 5}) == {"value": 15}

        pipeline = DictTransformPipelineConfig(transforms=[cfg])()

        assert pipeline.cfg.transforms[0] is pipeline[0].cfg
        assert pipeline({"value": 5}) == {"value": 15}

        pipeline.cfg.transforms[0].add_value = 20

        assert pipeline({"value": 5}) == {"value": 25}

    def test_nested_concat_config_constructor_flattens_and_runs(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            inner_cfg = ConcatDictTransformConfig(
                transforms=[
                    DummyTransformConfig(add_value=10),
                    DummyTransformConfig(add_value=20),
                ]
            )
            outer_cfg = ConcatDictTransformConfig(
                transforms=[inner_cfg, DummyTransformConfig(add_value=30)]
            )
            transform = outer_cfg()

        assert len(outer_cfg.transforms) == 3
        assert outer_cfg.transforms[0] is inner_cfg.transforms[0]
        assert outer_cfg.transforms[1] is inner_cfg.transforms[1]
        assert transform({"value": 5}) == {"value": 65}

    def test_add_returns_pipeline(self):
        pipeline = (
            DummyTransformConfig(add_value=10)()
            + DummyTransformConfig(add_value=20)()
        )

        assert isinstance(pipeline, DictTransformPipeline)
        assert pipeline({"value": 5}) == {"value": 35}
        assert set(pipeline.mapped_input_columns) == {"value"}
        assert set(pipeline.mapped_output_columns) == {"value"}

    def test_add_does_not_mutate_existing_pipeline(self):
        pipeline = (
            DummyTransformConfig(add_value=10)()
            + DummyTransformConfig(add_value=20)()
        )
        _ = pipeline.mapped_output_columns

        extended = pipeline + DummyTransformConfig(add_value=30)()

        assert pipeline({"value": 5}) == {"value": 35}
        assert extended({"value": 5}) == {"value": 65}
        assert len(pipeline.cfg.transforms) == 2
        assert len(extended.cfg.transforms) == 3
        assert pipeline._transforms is not extended._transforms
        assert pipeline.cfg is not extended.cfg

    def test_add_reuses_child_runtime_state(self):
        pipeline = (
            StatefulCounterTransformConfig(add_value=0)()
            + DummyTransformConfig(add_value=0)()
        )

        extended = pipeline + DummyTransformConfig(add_value=0)()

        assert extended({"value": 0}) == {"value": 1}
        assert pipeline({"value": 0}) == {"value": 2}
        assert pipeline[0] is extended[0]
        assert pipeline.cfg.transforms[0] is extended.cfg.transforms[0]

    def test_stateful_stage_contract_differs_between_runtime_and_cfg_paths(
        self,
    ):
        cfg = StatefulCounterTransformConfig(add_value=0)
        runtime_stage = StatefulCounterTransform(cfg)

        runtime_pipeline = runtime_stage + runtime_stage
        config_pipeline = DictTransformPipelineConfig(transforms=[cfg, cfg])()

        assert runtime_pipeline({"value": 0}) == {"value": 3}
        assert config_pipeline({"value": 0}) == {"value": 2}
        assert runtime_pipeline[0] is runtime_pipeline[1]
        assert config_pipeline[0] is not config_pipeline[1]
        assert config_pipeline[0].cfg is cfg
        assert config_pipeline[1].cfg is cfg

    def test_runtime_pipeline_cfg_is_single_source_of_truth(self):
        pipeline = (
            DummyTransformConfig(add_value=10)()
            + DummyTransformConfig(add_value=20)()
        )

        pipeline.cfg.transforms[0].add_value = 100

        assert pipeline.cfg.transforms[0] is pipeline[0].cfg
        assert pipeline({"value": 5}) == {"value": 125}

    def test_pipeline_iadd_syncs_runtime_and_cfg(self):
        pipeline = DictTransformPipelineConfig(
            transforms=[DummyTransformConfig(add_value=10)]
        )()

        pipeline += DummyTransformConfig(add_value=20)()

        assert len(pipeline.cfg.transforms) == 2
        assert len(pipeline._transforms) == 2
        assert pipeline.cfg.transforms[0] is pipeline[0].cfg
        assert pipeline.cfg.transforms[1] is pipeline[1].cfg
        assert pipeline({"value": 5}) == {"value": 35}

    def test_attached_cfg_transforms_assignment_does_not_rewire_runtime(
        self,
    ):
        pipeline = DictTransformPipelineConfig(
            transforms=[DummyTransformConfig(add_value=10)]
        )()
        _ = pipeline.mapped_output_columns

        pipeline.cfg.transforms = (
            pipeline.cfg.transforms[0],
            DummyTransformConfig(
                add_value=20,
                output_column_mapping={"value": "output_value"},
            ),
        )

        assert len(pipeline.cfg.transforms) == 2
        assert len(pipeline._transforms) == 1
        assert pipeline.cfg.transforms[0] is pipeline[0].cfg
        assert pipeline.mapped_output_columns == ["value"]
        assert pipeline({"value": 5}) == {"value": 15}

    def test_pipeline_apply_returns_last_child_structured_output(self):
        pipeline = (
            DummyTransformConfig(add_value=10)()
            + DummyTransformConfig(
                add_value=20,
                class_type=DummyDataclassTransform,
            )()
        )

        structured_result, final_row = pipeline.apply({"value": 5})

        assert isinstance(structured_result, DummyDataclassReturn)
        assert structured_result.value == 35
        assert final_row == {"value": 35}

    def test_config_add_returns_pipeline_config(self):
        left_cfg = DummyTransformConfig(add_value=10)
        right_cfg = DummyTransformConfig(add_value=20)
        combined_cfg = left_cfg + right_cfg
        extended_cfg = combined_cfg + DummyTransformConfig(add_value=30)
        left_cfg.add_value = 100

        assert isinstance(combined_cfg, DictTransformPipelineConfig)
        assert len(combined_cfg.transforms) == 2
        assert len(extended_cfg.transforms) == 3
        assert combined_cfg.transforms[0] is left_cfg
        assert combined_cfg.transforms[1] is right_cfg
        assert extended_cfg.transforms[0] is left_cfg

        pipeline = combined_cfg()
        assert isinstance(pipeline, DictTransformPipeline)
        assert pipeline.cfg.transforms[0] is left_cfg
        assert pipeline[0].cfg is left_cfg
        assert pipeline({"value": 5}) == {"value": 125}

    def test_pipeline_preserves_mapped_column_order(self):
        pipeline = (
            OrderedColumnsTransformConfig(
                input_columns=["src_b", "src_a"],
                output_columns_order=["mid_b", "mid_a"],
            )()
            + OrderedColumnsTransformConfig(
                input_columns=["mid_a", "mid_b", "src_c"],
                output_columns_order=["final_a", "final_b"],
            )()
        )

        assert pipeline.mapped_input_columns == [
            "src_b",
            "src_a",
            "src_c",
        ]
        assert pipeline.mapped_output_columns == [
            "final_a",
            "final_b",
        ]

    def test_pipeline_metadata_cache_tracks_child_metadata_values(self):
        transforms_base._get_cached_ordered_mapped_columns.cache_clear()
        try:
            pipeline = (
                OrderedColumnsTransformConfig(
                    input_columns=["src_b", "src_a"],
                    output_columns_order=["mid_b", "mid_a"],
                )()
                + OrderedColumnsTransformConfig(
                    input_columns=["mid_a", "mid_b", "src_c"],
                    output_columns_order=["final_a", "final_b"],
                )()
            )

            assert pipeline.mapped_input_columns == [
                "src_b",
                "src_a",
                "src_c",
            ]
            assert pipeline.mapped_output_columns == [
                "final_a",
                "final_b",
            ]

            cache_info = (
                transforms_base._get_cached_ordered_mapped_columns.cache_info()
            )
            assert cache_info.misses == 1

            assert pipeline.mapped_input_columns == [
                "src_b",
                "src_a",
                "src_c",
            ]
            assert pipeline.mapped_output_columns == [
                "final_a",
                "final_b",
            ]

            cache_info = (
                transforms_base._get_cached_ordered_mapped_columns.cache_info()
            )
            assert cache_info.hits >= 3

            pipeline[1].cfg.output_columns_order = ("final_x", "final_y")

            assert pipeline.mapped_output_columns == [
                "final_x",
                "final_y",
            ]

            cache_info = (
                transforms_base._get_cached_ordered_mapped_columns.cache_info()
            )
            assert cache_info.misses == 2
        finally:
            transforms_base._get_cached_ordered_mapped_columns.cache_clear()


class TestDictRowTransformInterface:
    def _assert_row_transform_contract(
        self,
        transform: DictRowTransform[Any],
        *,
        expected_input_columns: list[str],
        expected_output_columns: list[str],
        expected_final_row: dict[str, int],
    ) -> None:
        assert isinstance(transform, DictRowTransform)
        assert transform.mapped_input_columns == expected_input_columns
        assert transform.mapped_output_columns == expected_output_columns

        structured_result, final_row = transform.apply({"value": 5})

        assert final_row == expected_final_row
        assert transform({"value": 5}) == expected_final_row
        assert structured_result is not None

    def test_leaf_and_pipeline_share_row_transform_contract(self):
        self._assert_row_transform_contract(
            DummyTransformConfig(add_value=10)(),
            expected_input_columns=["value"],
            expected_output_columns=["value"],
            expected_final_row={"value": 15},
        )
        self._assert_row_transform_contract(
            DictTransformPipelineConfig(
                transforms=[
                    DummyTransformConfig(add_value=10),
                    DummyTransformConfig(add_value=20),
                ]
            )(),
            expected_input_columns=["value"],
            expected_output_columns=["value"],
            expected_final_row={"value": 35},
        )

    def test_leaf_and_pipeline_configs_share_row_transform_config_contract(
        self,
    ):
        def build_extended_pipeline(
            cfg: DictRowTransformConfig[Any],
        ) -> DictTransformPipeline:
            assert isinstance(cfg, DictRowTransformConfig)
            combined_cfg = cfg + DummyTransformConfig(add_value=20)
            assert isinstance(combined_cfg, DictTransformPipelineConfig)
            return combined_cfg()

        leaf_pipeline = build_extended_pipeline(
            DummyTransformConfig(add_value=10)
        )
        nested_pipeline = build_extended_pipeline(
            DictTransformPipelineConfig(
                transforms=[DummyTransformConfig(add_value=10)]
            )
        )

        assert leaf_pipeline({"value": 5}) == {"value": 35}
        assert nested_pipeline({"value": 5}) == {"value": 35}
