# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
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

"""Canonical RoboOrchard metadata schema models for robot datasets.

The classes in this module define tagged, strict JSON-compatible metadata
payloads used by dataset canonical writers. Legacy dataset adapters should
translate old shapes into these models at the dataset layer.
"""

from __future__ import annotations
import math
from typing import (
    Any,
    ClassVar,
    Collection,
    Literal,
    Mapping,
    TypeAlias,
    cast,
    get_args,
    get_origin,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic.types import JsonValue as _PydanticJsonValue

__all__ = [
    "JsonDict",
    "JsonValue",
    "MetadataSchemaError",
    "MetadataSchemaNotAllowed",
    "MetadataSchemaNotTagged",
    "MetadataSchemaPartialTag",
    "MetadataSchemaUnknown",
    "MetadataSchemaValidationError",
    "ROVersionedMetadata",
    "ROMetadataSchema",
    "ROInstructionContent",
    "InstructionSubtask",
    "ROEpisodeInfo",
    "EpisodeTimingInfo",
    "EpisodeMediaInfo",
    "DepthEncodingInfo",
    "parse_registered_metadata_schema",
    "resolve_metadata_schema",
    "validate_json_value",
]

JsonValue: TypeAlias = _PydanticJsonValue
JsonDict: TypeAlias = dict[str, JsonValue]


class MetadataSchemaError(ValueError):
    """Base error for canonical metadata schema parsing failures.

    Catch this type when a caller handles all tagged metadata parsing failures
    uniformly, and catch the subclasses when legacy fallback or hard-fail
    behavior needs to be distinguished.
    """


class MetadataSchemaNotTagged(MetadataSchemaError):  # noqa: N818
    """Raised when metadata has no schema tag and may use legacy parsing."""


class MetadataSchemaPartialTag(MetadataSchemaError):  # noqa: N818
    """Raised when metadata has an incomplete or malformed schema tag."""


class MetadataSchemaUnknown(MetadataSchemaError):  # noqa: N818
    """Raised when no registered schema matches the metadata tag."""


class MetadataSchemaNotAllowed(MetadataSchemaError):  # noqa: N818
    """Raised when a registered schema is not allowed at this call boundary."""


class MetadataSchemaValidationError(MetadataSchemaError):
    """Raised when tagged metadata fails validation for its schema."""


def validate_json_value(value: Any) -> None:
    """Validate that a value can be stored as strict JSON data.

    Use this helper at metadata boundaries that still receive plain Python
    containers instead of one of the Pydantic schema classes. The accepted
    shape is the JSON data model: strings, integers, finite floats, booleans,
    null, arrays, and objects with string keys.

    Args:
        value (Any): Candidate value to validate.

    Raises:
        TypeError: If the value contains a non-JSON type or a non-string
            object key.
        ValueError: If the value contains a non-finite float.
    """
    if value is None or type(value) in (str, bool, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON float values must be finite.")
        return
    if isinstance(value, list):
        for item in value:
            validate_json_value(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON dict key must be a string.")
            validate_json_value(item)
        return
    raise TypeError(
        f"JSON value must be str, int, float, bool, None, list, or dict; "
        f"got {type(value).__name__}."
    )


def _normalize_descriptions(value: Any) -> Any:
    if value is None:
        return value
    if not isinstance(value, list):
        return value
    ret: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            return value
        text = item.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ret.append(text)
    return ret


def _literal_values(annotation: Any) -> tuple[str, ...]:
    if get_origin(annotation) is not Literal:
        return ()
    values = get_args(annotation)
    if not all(isinstance(value, str) for value in values):
        raise TypeError("metadata schema Literal values must be strings.")
    return cast(tuple[str, ...], values)


class _ROBaseModel(BaseModel):
    """Strict base for canonical metadata models and nested payloads.

    The optional ``extras`` field is an opaque JSON object reserved for
    dataset-specific annotations. Empty ``extras`` objects normalize to
    ``None`` so storage output stays compact when no real extension data is
    present.
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )

    extras: JsonDict | None = None
    """Optional JSON object for dataset-specific annotations."""

    @field_validator("extras", mode="before", check_fields=False)
    @classmethod
    def _normalize_extras(cls, value: Any) -> Any:
        if value == {}:
            return None
        if value is not None:
            validate_json_value(value)
        return value


class ROVersionedMetadata(_ROBaseModel):
    """Base class for strict JSON metadata payloads with schema tags.

    Subclasses identify themselves with literal ``schema_id`` and
    ``schema_version`` fields. This base provides the shared versioned JSON
    contract, but it does not register subclasses for top-level metadata
    parsing. Use :class:`ROMetadataSchema` for canonical top-level storage
    payloads that should be resolved by
    :func:`parse_registered_metadata_schema`.
    """

    schema_id: str
    """Stable identifier of this versioned metadata schema."""

    schema_version: str
    """Version of this metadata schema."""

    def to_json_dict(self) -> JsonDict:
        """Dump this metadata as a strict JSON-compatible dict.

        Returns:
            JsonDict: JSON-compatible metadata with ``None`` fields omitted.

        Raises:
            TypeError: If a field value cannot be represented as JSON.
            ValueError: If a field value contains a non-finite float.
        """
        value = cast(
            JsonDict,
            self.model_dump(mode="json", exclude_none=True),
        )
        validate_json_value(value)
        return value

    @classmethod
    def schema_tag_keys(cls) -> tuple[tuple[str, str], ...]:
        """Return tag keys declared by this class's schema literals."""
        schema_id_field = cls.model_fields.get("schema_id")
        schema_version_field = cls.model_fields.get("schema_version")
        if schema_id_field is None or schema_version_field is None:
            return ()

        schema_ids = _literal_values(schema_id_field.annotation)
        if not schema_ids:
            return ()
        if len(schema_ids) != 1:
            raise TypeError("schema_id must be a single-value Literal.")
        if schema_id_field.default not in schema_ids:
            raise TypeError("schema_id default must match its Literal value.")

        schema_versions = _literal_values(schema_version_field.annotation)
        if not schema_versions:
            raise TypeError("schema_version must be a Literal.")
        if schema_version_field.default not in schema_versions:
            raise TypeError(
                "schema_version default must be one of its Literal values."
            )
        return tuple(
            (schema_ids[0], schema_version)
            for schema_version in schema_versions
        )


class ROMetadataSchema(ROVersionedMetadata):
    """Base class for top-level tagged RoboOrchard metadata payloads.

    Subclasses are registered automatically and may be parsed by
    :func:`parse_registered_metadata_schema`. Use this class for canonical
    storage-entry payloads such as ``Episode.info`` or
    ``Instruction.json_content``. Nested versioned records should inherit
    :class:`ROVersionedMetadata` instead and use a boundary-specific parser.
    """

    _registry: ClassVar[dict[tuple[str, str], type["ROMetadataSchema"]]] = {}

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        for key in cls.schema_tag_keys():
            previous = ROMetadataSchema._registry.get(key)
            if previous is not None and previous is not cls:
                raise ValueError(f"Duplicate metadata schema key: {key}")
            ROMetadataSchema._registry[key] = cls


class InstructionSubtask(_ROBaseModel):
    """A semantic subtask span inside an instruction episode.

    Spans use half-open frame indices ``[frame_index_begin, frame_index_end)``.
    Individual spans only validate their own bounds; overlap checks happen in
    :class:`ROInstructionContent` so sparse spans with holes remain valid.
    """

    descriptions: list[str] = Field(default_factory=list)
    """Equivalent natural-language descriptions of the subtask."""

    frame_index_begin: int = Field(ge=0)
    """Inclusive zero-based frame index where the subtask span begins."""

    frame_index_end: int = Field(gt=0)
    """Exclusive zero-based frame index where the subtask span ends."""

    @field_validator("descriptions", mode="before")
    @classmethod
    def _normalize_descriptions(cls, value: Any) -> Any:
        return _normalize_descriptions(value)

    @model_validator(mode="after")
    def _validate_span(self) -> "InstructionSubtask":
        if self.frame_index_end <= self.frame_index_begin:
            raise ValueError(
                "frame_index_end must be greater than frame_index_begin."
            )
        return self


class ROInstructionContent(ROMetadataSchema):
    """Canonical metadata stored in ``Instruction.json_content``.

    ``descriptions`` contains equivalent natural-language instructions; order
    is not a training priority contract. Empty descriptions are allowed so
    downstream datasets can decide whether instruction text is required.
    Optional subtasks may annotate sparse, non-overlapping frame spans.
    """

    schema_id: Literal["robo_orchard.instruction_content"] = (
        "robo_orchard.instruction_content"
    )
    """Schema identifier for canonical instruction metadata."""

    schema_version: Literal["1.0"] = "1.0"
    """Schema version for canonical instruction metadata."""

    descriptions: list[str] = Field(default_factory=list)
    """Equivalent natural-language descriptions for the instruction."""

    subtasks: list[InstructionSubtask] | None = None
    """Optional sparse, non-overlapping semantic subtask spans."""

    @field_validator("descriptions", mode="before")
    @classmethod
    def _normalize_descriptions(cls, value: Any) -> Any:
        return _normalize_descriptions(value)

    @model_validator(mode="after")
    def _normalize_subtasks(self) -> "ROInstructionContent":
        if self.subtasks is None:
            return self
        self.subtasks = sorted(
            self.subtasks,
            key=lambda item: (item.frame_index_begin, item.frame_index_end),
        )
        previous: InstructionSubtask | None = None
        for current in self.subtasks:
            if (
                previous is not None
                and current.frame_index_begin < previous.frame_index_end
            ):
                raise ValueError("subtasks must not overlap.")
            previous = current
        return self


class EpisodeTimingInfo(_ROBaseModel):
    """Compact episode-level timing summary metadata.

    The fields summarize episode duration and nominal frame rate only. Store
    detailed timestamp arrays in dataset-specific data columns instead of this
    metadata payload.
    """

    duration_s: FiniteFloat | None = Field(default=None, ge=0)
    """Episode duration in seconds when known."""

    average_fps: FiniteFloat | None = Field(default=None, ge=0)
    """Nominal average episode frame rate when known."""


class DepthEncodingInfo(_ROBaseModel):
    """Canonical PNG uint16 depth image decoding contract for one camera.

    Depth frames are stored as ``uint16`` PNG data and decoded as meters with
    ``depth_m = stored_value / scale``. Camera names are owned by
    :class:`EpisodeMediaInfo`, which stores one encoding entry per depth
    camera.
    """

    format: Literal["png"] = "png"
    """Depth image container format used in storage."""

    storage_dtype: Literal["uint16"] = "uint16"
    """Integer dtype of stored depth image pixels."""

    unit: Literal["m"] = "m"
    """Physical unit produced after applying the depth scale."""

    scale: FiniteFloat = Field(gt=0)
    """Divisor that converts stored uint16 depth to meters."""

    invalid_value: int | None = Field(default=0, ge=0, le=65535)
    """Stored uint16 value that represents invalid depth."""


class EpisodeMediaInfo(_ROBaseModel):
    """Episode-level media decoding metadata.

    This payload records decoding contracts that apply to media columns.
    Depth encodings are keyed by camera name so the stored metadata is always
    explicit at each camera boundary.
    """

    depth_encodings: dict[str, DepthEncodingInfo] | None = None
    """Depth image decoding contracts keyed by camera name."""


class ROEpisodeInfo(ROMetadataSchema):
    """Canonical episode metadata stored in ``Episode.info``.

    The payload is reserved for episode-level facts needed to interpret
    training data, not for arbitrary provenance. Dataset-specific annotations
    can be carried in ``extras`` when they are still JSON-compatible.
    """

    schema_id: Literal["robo_orchard.episode_info"] = (
        "robo_orchard.episode_info"
    )
    """Schema identifier for canonical episode metadata."""

    schema_version: Literal["1.0"] = "1.0"
    """Schema version for canonical episode metadata."""

    episode_id: str | None = None
    """Optional stable episode identifier from the source data."""

    timing: EpisodeTimingInfo | None = None
    """Optional episode-level timing summary."""

    media: EpisodeMediaInfo | None = None
    """Optional media decoding metadata for this episode."""


def resolve_metadata_schema(
    schema_id: str,
    schema_version: str,
) -> type[ROMetadataSchema]:
    """Resolve a registered metadata schema class by exact schema tag.

    Use this when caller code already validated that a payload is tagged and
    only needs the model class for the tag. Use
    :func:`parse_registered_metadata_schema` to validate and instantiate a
    payload in one step.

    Args:
        schema_id (str): Stable schema identifier in the metadata payload.
        schema_version (str): Schema version in the metadata payload.

    Returns:
        type[ROMetadataSchema]: Registered schema class for the exact tag.

    Raises:
        MetadataSchemaUnknown: If the tag has no registered schema class.
    """
    cls = ROMetadataSchema._registry.get((schema_id, schema_version))
    if cls is None:
        raise MetadataSchemaUnknown(
            f"unknown metadata schema {schema_id!r} version "
            f"{schema_version!r}."
        )
    return cls


def parse_registered_metadata_schema(
    value: Mapping[str, Any],
    *,
    allowed_schema_ids: Collection[str] | None = None,
) -> ROMetadataSchema:
    """Parse a tagged metadata mapping into a registered schema model.

    This is the canonical read boundary for new metadata. A
    :class:`MetadataSchemaNotTagged` error is the signal that callers may try a
    dataset-local legacy parser; partial, unknown, disallowed, and invalid
    tags are hard failures for the tagged metadata path.

    Args:
        value (Mapping[str, Any]): Raw metadata mapping from storage.
        allowed_schema_ids (Collection[str], optional): Optional allow-list
            for the storage location being parsed. Default is ``None``.

    Returns:
        ROMetadataSchema: Validated metadata model for the declared tag.

    Raises:
        MetadataSchemaNotTagged: If both ``schema_id`` and
            ``schema_version`` are absent.
        MetadataSchemaPartialTag: If only one tag field is present, or if
            either tag field is not a string.
        MetadataSchemaNotAllowed: If ``schema_id`` is outside the allow-list.
        MetadataSchemaUnknown: If no registered class matches the tag.
        MetadataSchemaValidationError: If the registered class rejects the
            payload content.
    """
    has_schema_id = "schema_id" in value
    has_schema_version = "schema_version" in value
    if not has_schema_id and not has_schema_version:
        raise MetadataSchemaNotTagged(
            "metadata has no schema_id/schema_version."
        )
    if has_schema_id != has_schema_version:
        raise MetadataSchemaPartialTag(
            "metadata must contain both schema_id and schema_version."
        )

    schema_id = value["schema_id"]
    schema_version = value["schema_version"]
    if not isinstance(schema_id, str) or not isinstance(schema_version, str):
        raise MetadataSchemaPartialTag(
            "schema_id and schema_version must be strings."
        )
    if allowed_schema_ids is not None and schema_id not in allowed_schema_ids:
        raise MetadataSchemaNotAllowed(
            f"metadata schema_id {schema_id!r} is not in allowed set "
            f"{sorted(allowed_schema_ids)!r}."
        )

    cls = resolve_metadata_schema(schema_id, schema_version)
    try:
        return cls.model_validate(value)
    except (ValidationError, TypeError, ValueError) as exc:
        raise MetadataSchemaValidationError(
            f"metadata schema {schema_id!r} version {schema_version!r} "
            f"failed validation: {exc}"
        ) from exc
