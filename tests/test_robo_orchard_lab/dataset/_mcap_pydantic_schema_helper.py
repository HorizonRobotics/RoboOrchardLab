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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel

_SCHEMA_MARKER_KEYS = frozenset(
    {
        "$ref",
        "type",
        "const",
        "enum",
        "anyOf",
        "allOf",
        "oneOf",
    }
)
_SCHEMA_MAP_KEYS = frozenset(
    {
        "$defs",
        "definitions",
        "properties",
        "patternProperties",
        "dependentSchemas",
    }
)
_SCHEMA_VALUE_KEYS = frozenset(
    {
        "additionalProperties",
        "contains",
        "items",
        "not",
        "propertyNames",
        "unevaluatedProperties",
    }
)
_SCHEMA_LIST_KEYS = frozenset(
    {
        "allOf",
        "anyOf",
        "oneOf",
        "prefixItems",
    }
)


@dataclass(frozen=True)
class _SchemaIssue:
    path: str
    message: str


def assert_mcap_compatible_pydantic_schema(
    model_type: type[BaseModel],
) -> None:
    """Assert that a Pydantic model's JSON schema is safe for MCAP readers."""

    schema = model_type.model_json_schema(mode="serialization", by_alias=True)
    issues = list(_iter_schema_issues(schema, "$"))
    if schema.get("type") != "object":
        issues.insert(
            0,
            _SchemaIssue(
                "$",
                "top-level Pydantic MCAP schemas must be JSON objects",
            ),
        )
    if not issues:
        return

    details = "\n".join(
        f"- {model_type.__name__} {issue.path}: {issue.message}"
        for issue in issues
    )
    raise AssertionError(
        f"{model_type.__name__} is not MCAP-compatible:\n{details}"
    )


def _iter_schema_issues(
    schema: object,
    path: str,
) -> list[_SchemaIssue]:
    if isinstance(schema, bool):
        if schema:
            return [
                _SchemaIssue(
                    path,
                    "boolean true schema is unconstrained",
                )
            ]
        return []
    if not isinstance(schema, Mapping):
        return []

    issues: list[_SchemaIssue] = []
    if not schema:
        issues.append(_SchemaIssue(path, "untyped schema is not allowed"))
        return issues
    if not _SCHEMA_MARKER_KEYS.intersection(schema):
        issues.append(_SchemaIssue(path, "untyped schema is not allowed"))

    any_of = schema.get("anyOf")
    if isinstance(any_of, Sequence) and not isinstance(any_of, str):
        non_null_alternatives = [
            item
            for item in any_of
            if not (isinstance(item, Mapping) and item.get("type") == "null")
        ]
        if len(non_null_alternatives) != 1:
            issues.append(
                _SchemaIssue(
                    f"{path}.anyOf",
                    "anyOf must have exactly one non-null alternative",
                )
            )
        elif not (
            isinstance(non_null_alternatives[0], Mapping)
            and non_null_alternatives[0].get("type") is not None
        ):
            issues.append(
                _SchemaIssue(
                    f"{path}.anyOf",
                    "anyOf non-null alternatives must define a concrete type",
                )
            )

    additional_properties = schema.get("additionalProperties")
    if additional_properties is True:
        issues.append(
            _SchemaIssue(
                f"{path}.additionalProperties",
                "unconstrained additionalProperties is not allowed",
            )
        )
    if (
        schema.get("type") == "object"
        and "properties" not in schema
        and "patternProperties" not in schema
        and "additionalProperties" not in schema
        and "unevaluatedProperties" not in schema
    ):
        issues.append(
            _SchemaIssue(
                path,
                "unconstrained object schema is not allowed",
            )
        )

    for key in sorted(_SCHEMA_MAP_KEYS):
        value = schema.get(key)
        if isinstance(value, Mapping):
            for child_name, child_schema in value.items():
                issues.extend(
                    _iter_schema_issues(
                        child_schema,
                        f"{path}.{key}.{child_name}",
                    )
                )

    for key in sorted(_SCHEMA_VALUE_KEYS):
        if key not in schema:
            continue
        if key == "additionalProperties" and schema[key] is True:
            continue
        issues.extend(_iter_schema_issues(schema[key], f"{path}.{key}"))

    for key in sorted(_SCHEMA_LIST_KEYS):
        value = schema.get(key)
        if isinstance(value, Sequence) and not isinstance(value, str):
            for index, child_schema in enumerate(value):
                issues.extend(
                    _iter_schema_issues(
                        child_schema,
                        f"{path}.{key}[{index}]",
                    )
                )

    return issues
