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

from typing import Literal

import pytest
from pydantic import ValidationError

from robo_orchard_lab.dataset.robot.metadata_schema import (
    DepthEncodingInfo,
    EpisodeMediaInfo,
    EpisodeTimingInfo,
    InstructionSubtask,
    MetadataSchemaNotAllowed,
    MetadataSchemaNotTagged,
    MetadataSchemaPartialTag,
    MetadataSchemaUnknown,
    MetadataSchemaValidationError,
    ROEpisodeInfo,
    ROInstructionContent,
    ROVersionedMetadata,
    parse_registered_metadata_schema,
    resolve_metadata_schema,
    validate_json_value,
)


def test_instruction_content_dumps_tagged_json() -> None:
    content = ROInstructionContent(
        descriptions=["Pick cup", " Pick cup ", "", "Lift cup"],
        extras={"source": "unit-test"},
    )

    assert content.descriptions == ["Pick cup", "Lift cup"]
    assert content.to_json_dict() == {
        "schema_id": "robo_orchard.instruction_content",
        "schema_version": "1.0",
        "descriptions": ["Pick cup", "Lift cup"],
        "extras": {"source": "unit-test"},
    }


def test_instruction_content_allows_empty_descriptions() -> None:
    content = ROInstructionContent(descriptions=[])

    assert content.descriptions == []
    assert content.to_json_dict()["descriptions"] == []


def test_subtasks_allow_holes_and_are_sorted() -> None:
    content = ROInstructionContent(
        descriptions=["Do task"],
        subtasks=[
            InstructionSubtask(
                descriptions=["finish"],
                frame_index_begin=8,
                frame_index_end=10,
            ),
            InstructionSubtask(
                descriptions=["start"],
                frame_index_begin=1,
                frame_index_end=3,
            ),
        ],
    )

    assert [
        (item.frame_index_begin, item.frame_index_end)
        for item in content.subtasks or []
    ] == [(1, 3), (8, 10)]


def test_subtasks_reject_overlap() -> None:
    with pytest.raises(ValidationError, match="must not overlap"):
        ROInstructionContent(
            descriptions=["Do task"],
            subtasks=[
                InstructionSubtask(
                    descriptions=["a"],
                    frame_index_begin=0,
                    frame_index_end=5,
                ),
                InstructionSubtask(
                    descriptions=["b"],
                    frame_index_begin=4,
                    frame_index_end=7,
                ),
            ],
        )


def test_episode_info_dumps_timing_and_depth_encoding() -> None:
    episode_info = ROEpisodeInfo(
        episode_id="episode-1",
        timing=EpisodeTimingInfo(duration_s=1.5, average_fps=30.0),
        media=EpisodeMediaInfo(
            depth_encodings={
                "left": DepthEncodingInfo(scale=500.0),
                "right": DepthEncodingInfo(scale=1000.0),
            }
        ),
    )

    assert episode_info.to_json_dict() == {
        "schema_id": "robo_orchard.episode_info",
        "schema_version": "1.0",
        "episode_id": "episode-1",
        "timing": {"duration_s": 1.5, "average_fps": 30.0},
        "media": {
            "depth_encodings": {
                "left": {
                    "format": "png",
                    "storage_dtype": "uint16",
                    "unit": "m",
                    "scale": 500.0,
                    "invalid_value": 0,
                },
                "right": {
                    "format": "png",
                    "storage_dtype": "uint16",
                    "unit": "m",
                    "scale": 1000.0,
                    "invalid_value": 0,
                },
            }
        },
    }


def test_depth_encoding_requires_camera_local_scale() -> None:
    with pytest.raises(ValidationError):
        DepthEncodingInfo.model_validate({})

    with pytest.raises(ValidationError):
        DepthEncodingInfo(scale=0)


def test_episode_media_depth_encodings_are_keyed_by_camera() -> None:
    media = EpisodeMediaInfo(
        depth_encodings={
            "front": DepthEncodingInfo(scale=1000.0),
            "wrist": DepthEncodingInfo(scale=2000.0, invalid_value=65535),
        }
    )

    assert media.depth_encodings is not None
    assert media.depth_encodings["front"].scale == 1000.0
    assert media.depth_encodings["wrist"].scale == 2000.0
    assert media.depth_encodings["wrist"].invalid_value == 65535


def test_depth_encoding_rejects_invalid_value_outside_uint16_range() -> None:
    with pytest.raises(ValidationError):
        DepthEncodingInfo(scale=1000.0, invalid_value=-1)

    with pytest.raises(ValidationError):
        DepthEncodingInfo(scale=1000.0, invalid_value=65536)


def test_episode_info_rejects_nan_timing() -> None:
    with pytest.raises(ValidationError):
        EpisodeTimingInfo(duration_s=float("nan"))


def test_resolve_metadata_schema_returns_registered_class() -> None:
    assert (
        resolve_metadata_schema(
            "robo_orchard.instruction_content",
            "1.0",
        )
        is ROInstructionContent
    )


def test_versioned_metadata_base_does_not_register_top_level_schema() -> None:
    class LocalProcessingRecord(ROVersionedMetadata):
        schema_id: Literal["robo_orchard.processing.local"] = (
            "robo_orchard.processing.local"
        )
        schema_version: Literal["1.0"] = "1.0"
        value: int

    record = LocalProcessingRecord(value=1, extras={})

    assert record.extras is None
    assert record.to_json_dict() == {
        "schema_id": "robo_orchard.processing.local",
        "schema_version": "1.0",
        "value": 1,
    }
    with pytest.raises(MetadataSchemaUnknown):
        resolve_metadata_schema(
            "robo_orchard.processing.local",
            "1.0",
        )


def test_parse_registered_metadata_schema_returns_model() -> None:
    parsed = parse_registered_metadata_schema(
        {
            "schema_id": "robo_orchard.instruction_content",
            "schema_version": "1.0",
            "descriptions": ["Pick cup"],
        },
        allowed_schema_ids={"robo_orchard.instruction_content"},
    )

    assert isinstance(parsed, ROInstructionContent)
    assert parsed.descriptions == ["Pick cup"]


def test_parse_registered_metadata_schema_not_tagged() -> None:
    with pytest.raises(MetadataSchemaNotTagged):
        parse_registered_metadata_schema({"description": "legacy"})


def test_parse_registered_metadata_schema_partial_tag_fails_hard() -> None:
    with pytest.raises(MetadataSchemaPartialTag):
        parse_registered_metadata_schema(
            {"schema_id": "robo_orchard.instruction_content"}
        )


def test_parse_registered_metadata_schema_unknown_fails_hard() -> None:
    with pytest.raises(MetadataSchemaUnknown):
        parse_registered_metadata_schema(
            {"schema_id": "unknown", "schema_version": "1.0"}
        )


def test_parse_registered_metadata_schema_not_allowed_fails_hard() -> None:
    with pytest.raises(MetadataSchemaNotAllowed):
        parse_registered_metadata_schema(
            {
                "schema_id": "robo_orchard.episode_info",
                "schema_version": "1.0",
            },
            allowed_schema_ids={"robo_orchard.instruction_content"},
        )


def test_parse_registered_metadata_schema_validation_error() -> None:
    with pytest.raises(MetadataSchemaValidationError) as exc_info:
        parse_registered_metadata_schema(
            {
                "schema_id": "robo_orchard.instruction_content",
                "schema_version": "1.0",
                "descriptions": [1],
            }
        )

    assert isinstance(exc_info.value.__cause__, ValidationError)


def test_parse_registered_metadata_schema_wraps_extras_type_error() -> None:
    with pytest.raises(MetadataSchemaValidationError) as exc_info:
        parse_registered_metadata_schema(
            {
                "schema_id": "robo_orchard.instruction_content",
                "schema_version": "1.0",
                "extras": {1: "bad"},
            }
        )

    assert isinstance(exc_info.value.__cause__, TypeError)
    assert "JSON dict key must be a string." in str(exc_info.value)


def test_validate_json_value_rejects_non_string_dict_key() -> None:
    with pytest.raises(TypeError, match="dict key"):
        validate_json_value({1: "bad"})


def test_validate_json_value_rejects_non_finite_float() -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_json_value({"value": float("inf")})


def test_to_json_dict_runs_final_json_validation() -> None:
    content = ROInstructionContent(
        descriptions=["ok"],
        extras={"nested": ["value", None, True, 1, 1.5]},
    )
    payload = content.to_json_dict()
    extras = payload["extras"]

    assert isinstance(extras, dict)
    assert extras["nested"] == [
        "value",
        None,
        True,
        1,
        1.5,
    ]


def test_empty_extras_normalize_to_none() -> None:
    content = ROInstructionContent(descriptions=["ok"], extras={})
    subtask = InstructionSubtask(
        descriptions=["subtask"],
        frame_index_begin=0,
        frame_index_end=1,
        extras={},
    )

    assert content.extras is None
    assert subtask.extras is None
    assert "extras" not in content.to_json_dict()


def test_to_json_dict_preserves_user_extras_payload() -> None:
    content = ROInstructionContent(
        descriptions=["ok"],
        extras={"source": {"unit_test": {"extras": {}}}},
    )

    assert content.to_json_dict()["extras"] == {
        "source": {"unit_test": {"extras": {}}}
    }
