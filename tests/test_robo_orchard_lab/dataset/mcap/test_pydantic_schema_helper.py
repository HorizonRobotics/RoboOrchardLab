# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
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

from typing import Any

import pytest
from pydantic import BaseModel

from tests.test_robo_orchard_lab.dataset._mcap_pydantic_schema_helper import (
    assert_mcap_compatible_pydantic_schema,
)


class _NestedPayload(BaseModel):
    value: int


class _CompatiblePayload(BaseModel):
    value: str | None = None


class _NullableNestedPayload(BaseModel):
    nested: _NestedPayload | None = None


class _UnsupportedUnionPayload(BaseModel):
    value: _NestedPayload | str | None = None


class _AnyPayload(BaseModel):
    value: Any


class _DictAnyPayload(BaseModel):
    value: dict[str, Any]


def test_helper_accepts_nullable_single_scalar_alternative() -> None:
    assert_mcap_compatible_pydantic_schema(_CompatiblePayload)


def test_helper_accepts_nullable_nested_model_alternative() -> None:
    assert_mcap_compatible_pydantic_schema(_NullableNestedPayload)


@pytest.mark.parametrize(
    ("model_type", "match"),
    [
        (_UnsupportedUnionPayload, "anyOf.*exactly one non-null"),
        (_AnyPayload, "untyped schema"),
        (_DictAnyPayload, "unconstrained object schema"),
    ],
)
def test_helper_rejects_schema_shapes_unsupported_by_mcap_readers(
    model_type: type[BaseModel],
    match: str,
) -> None:
    with pytest.raises(AssertionError, match=match):
        assert_mcap_compatible_pydantic_schema(model_type)
