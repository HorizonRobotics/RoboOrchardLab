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

import json
from pathlib import Path

import pytest
from foxglove_schemas_protobuf.RawImage_pb2 import RawImage
from google.protobuf.timestamp import from_nanoseconds
from pydantic import BaseModel, Field, RootModel

from robo_orchard_lab.dataset.experimental.mcap import msg_encoder
from robo_orchard_lab.dataset.experimental.mcap.foxglove import create_channel
from robo_orchard_lab.dataset.experimental.mcap.foxglove_writer import (
    FoxgloveMcapWriter,
)
from robo_orchard_lab.dataset.experimental.mcap.msg_converter import (
    dataset_meta,
)
from robo_orchard_lab.dataset.experimental.mcap.msg_converter.base import (
    MessageConverterConfig,
    MessageConverterFactoryConfig,
    MessageConverterStateful,
    MessageConverterStateless,
)
from robo_orchard_lab.dataset.experimental.mcap.reader import (
    MakeIterMsgArgs,
    McapReader,
)
from tests.test_robo_orchard_lab.dataset._mcap_pydantic_schema_helper import (
    assert_mcap_compatible_pydantic_schema,
)


class _PydanticPayload(BaseModel):
    value: int


class _OtherPydanticPayload(BaseModel):
    value: int


class _AliasedPydanticPayload(BaseModel):
    internal_name: int = Field(serialization_alias="externalName")


class _RootPayload(RootModel[int]):
    pass


class _ObjectRootPayload(RootModel[dict[str, int]]):
    pass


class _StatefulRawImageConverter(MessageConverterStateful[RawImage, RawImage]):
    def __init__(self, cfg: "_StatefulRawImageConverterConfig"):
        _ = cfg

    def convert(self, src: RawImage | None):
        if src is not None:
            yield src

    def flush(self) -> list[RawImage]:
        return []


class _StatefulRawImageConverterConfig(
    MessageConverterConfig[_StatefulRawImageConverter]
):
    class_type: type[_StatefulRawImageConverter] = _StatefulRawImageConverter


class _ZeroRawImageConverter(MessageConverterStateless[RawImage, RawImage]):
    def __init__(self, cfg: "_ZeroRawImageConverterConfig"):
        _ = cfg

    def convert(self, src: RawImage) -> RawImage:
        return src

    def __call__(self, src: RawImage):
        _ = src
        return iter(())


class _ZeroRawImageConverterConfig(
    MessageConverterConfig[_ZeroRawImageConverter]
):
    class_type: type[_ZeroRawImageConverter] = _ZeroRawImageConverter


def _raw_rgb_image() -> RawImage:
    return RawImage(
        timestamp=from_nanoseconds(123),
        frame_id="camera",
        width=2,
        height=1,
        encoding="rgb8",
        step=6,
        data=bytes([255, 0, 0, 0, 255, 0]),
    )


def test_message_converter_factory_uses_neutral_name_for_lookup() -> None:
    from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage

    from robo_orchard_lab.dataset.experimental.mcap.msg_converter import (
        RawImage2CompressedImageConfig,
    )

    factory = MessageConverterFactoryConfig(
        converters={
            "/camera/image": RawImage2CompressedImageConfig(format="png")
        }
    )()

    converter = factory.converter_for("/camera/image")
    assert converter is not None
    converted = list(converter(_raw_rgb_image()))
    assert len(converted) == 1
    assert isinstance(converted[0], CompressedImage)

    convert_impl = factory.convert_for(name="/camera/image")
    assert convert_impl is not None
    converted_again = list(convert_impl(_raw_rgb_image()))
    assert len(converted_again) == 1
    assert isinstance(converted_again[0], CompressedImage)

    assert factory.converter_for("/missing") is None
    assert factory.convert_for(name="/missing") is None


def test_create_channel_uses_pydantic_serialization_schema() -> None:
    channel = create_channel("/pydantic", _PydanticPayload)

    assert channel.message_encoding == "json"
    schema = channel.schema()
    assert schema is not None
    assert schema.encoding == "jsonschema"
    assert json.loads(schema.data.decode("utf-8")) == (
        _PydanticPayload.model_json_schema(mode="serialization", by_alias=True)
    )


@pytest.mark.parametrize(
    "model_type",
    [
        dataset_meta._RobotMcapMetadata,
    ],
)
def test_schema_fixed_pydantic_payloads_are_mcap_compatible(
    model_type: type[BaseModel],
) -> None:
    assert_mcap_compatible_pydantic_schema(model_type)


def test_encoder_preserves_pydantic_model_until_writer_boundary() -> None:
    msg = _PydanticPayload(value=3)

    encoded = msg_encoder.FoxgloveEncoder(None).encode_message(
        topic="/pydantic",
        msg=msg,
        log_time=123,
    )

    assert encoded.channel.message_encoding == "json"
    assert encoded.message.data is msg
    assert encoded.message.log_time == 123


def test_encoder_applies_topic_converter_before_schema_creation() -> None:
    from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage

    from robo_orchard_lab.dataset.experimental.mcap.msg_converter import (
        RawImage2CompressedImageConfig,
    )

    encoder = msg_encoder.FoxgloveEncoder(
        None,
        converters={
            "/camera/image": RawImage2CompressedImageConfig(format="png")
        },
    )

    encoded = encoder.encode_message(
        topic="/camera/image",
        msg=_raw_rgb_image(),
        log_time=123,
    )

    assert isinstance(encoded.message.data, CompressedImage)
    assert encoded.schema is not None
    assert encoded.schema.name == CompressedImage.DESCRIPTOR.full_name

    encoder.reset()
    encoded_after_reset = encoder.encode_message(
        topic="/camera/image",
        msg=_raw_rgb_image(),
        log_time=456,
    )
    assert isinstance(encoded_after_reset.message.data, CompressedImage)


def test_encoder_rejects_stateful_topic_converter() -> None:
    encoder = msg_encoder.FoxgloveEncoder(
        None,
        converters={"/camera/image": _StatefulRawImageConverterConfig()},
    )

    with pytest.raises(TypeError, match="stateless"):
        encoder.encode_message("/camera/image", _raw_rgb_image(), log_time=1)


def test_encoder_rejects_topic_converter_without_exactly_one_output() -> None:
    encoder = msg_encoder.FoxgloveEncoder(
        None,
        converters={"/camera/image": _ZeroRawImageConverterConfig()},
    )

    with pytest.raises(ValueError, match="exactly one"):
        encoder.encode_message("/camera/image", _raw_rgb_image(), log_time=1)


def test_writer_dumps_pydantic_payload_at_log_boundary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pydantic.mcap"

    with path.open("wb") as f, FoxgloveMcapWriter(f) as writer:
        writer.write_message(
            topic="/pydantic",
            message=_PydanticPayload(value=7),
            log_time=456,
        )

    with path.open("rb") as f:
        messages = list(
            McapReader.make_reader(f).iter_messages(
                MakeIterMsgArgs(topics=["/pydantic"])
            )
        )

    assert len(messages) == 1
    assert messages[0].channel.message_encoding == "json"
    assert messages[0].message.log_time == 456
    assert json.loads(messages[0].message.data.decode("utf-8")) == {"value": 7}


def test_writer_uses_pydantic_serialization_aliases(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pydantic_alias.mcap"

    with path.open("wb") as f, FoxgloveMcapWriter(f) as writer:
        writer.write_message(
            topic="/pydantic_alias",
            message=_AliasedPydanticPayload(internal_name=9),
            log_time=789,
        )

    with path.open("rb") as f:
        messages = list(
            McapReader.make_reader(f).iter_messages(
                MakeIterMsgArgs(topics=["/pydantic_alias"])
            )
        )

    assert len(messages) == 1
    schema = messages[0].schema
    assert schema is not None
    assert json.loads(schema.data.decode("utf-8")) == (
        _AliasedPydanticPayload.model_json_schema(
            mode="serialization", by_alias=True
        )
    )
    assert json.loads(messages[0].message.data.decode("utf-8")) == {
        "externalName": 9
    }


def test_root_model_pydantic_schema_is_rejected() -> None:
    with pytest.raises(TypeError, match="/root.*_RootPayload"):
        create_channel("/root", _RootPayload)
    with pytest.raises(TypeError, match="/root_object.*_ObjectRootPayload"):
        create_channel("/root_object", _ObjectRootPayload)
    with pytest.raises(TypeError, match="/root_schema.*_ObjectRootPayload"):
        create_channel(
            "/root_schema",
            _ObjectRootPayload,
            schema={"type": "object", "title": "RootSchema"},
        )


def test_pydantic_topic_rejects_different_model_class() -> None:
    encoder = msg_encoder.FoxgloveEncoder(None)
    encoder.encode_message("/topic", _PydanticPayload(value=1), log_time=1)

    with pytest.raises(TypeError, match="/topic"):
        encoder.encode_message(
            "/topic", _OtherPydanticPayload(value=2), log_time=2
        )


def test_pydantic_topic_rejects_schemaless_json_mix() -> None:
    encoder = msg_encoder.FoxgloveEncoder(None)
    encoder.encode_message("/topic", _PydanticPayload(value=1), log_time=1)

    with pytest.raises(TypeError, match="/topic"):
        encoder.encode_message("/topic", {"value": 2}, log_time=2)


def test_schemaless_json_topic_allows_dict_and_list_mix() -> None:
    encoder = msg_encoder.FoxgloveEncoder(None)
    first = encoder.encode_message("/json", {"value": 1}, log_time=1)
    second = encoder.encode_message("/json", [1, 2], log_time=2)

    assert first.channel.message_encoding == "json"
    assert second.channel.message_encoding == "json"
    assert second.message.data == [1, 2]
