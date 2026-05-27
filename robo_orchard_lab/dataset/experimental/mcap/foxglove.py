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

"""Foxglove channel helpers for RoboOrchard MCAP writers."""

from typing import Any, Type, cast

from foxglove import Channel, Context, Schema
from google.protobuf.message import Message as ProtobufMessage
from mcap_protobuf.schema import build_file_descriptor_set
from pydantic import BaseModel, RootModel

__all__ = [
    "JsonSchema",
    "create_schema",
    "create_channel",
    "create_channels_from_examples",
]


JsonSchema = dict[str, Any]


def create_schema(
    msg_type: Type[Any] | Any,
    *,
    topic: str | None = None,
) -> Schema | JsonSchema | None:
    """Create a Foxglove schema for a supported message class.

    Protobuf message classes are encoded as Foxglove protobuf schemas.
    Pydantic models are encoded as JSON schemas and must have a top-level
    JSON object schema. Schemaless JSON topics return ``None``.
    """

    if _is_type_subclass(msg_type, ProtobufMessage):
        file_descriptor_set = build_file_descriptor_set(msg_type)
        return Schema(
            name=msg_type.DESCRIPTOR.full_name,
            encoding="protobuf",
            data=file_descriptor_set.SerializeToString(),
        )

    pydantic_model_type = _validate_pydantic_model_type(topic, msg_type)
    if pydantic_model_type is not None:
        schema = pydantic_model_type.model_json_schema(
            mode="serialization", by_alias=True
        )
        if schema.get("type") != "object":
            raise TypeError(
                f"{_format_channel_model(topic, pydantic_model_type)} must "
                "have a top-level object JSON schema."
            )
        return schema

    return None


def create_channel(
    topic: str,
    msg_type: Type[Any],
    context: Context | None = None,
    schema: Schema | JsonSchema | None = None,
    metadata: dict[str, str] | None = None,
) -> Channel:
    """Create a Foxglove channel for protobuf or JSON MCAP messages.

    Args:
        topic: MCAP topic name for the channel.
        msg_type: Message class for the channel. Protobuf message classes,
            ``dict``/``list`` JSON containers, and Pydantic ``BaseModel``
            classes are supported.
        context: Optional Foxglove context that owns the channel.
        schema: Optional prebuilt schema. When omitted, this function creates
            a schema for protobuf and Pydantic message classes.
        metadata: Optional Foxglove channel metadata.

    Raises:
        TypeError: If a Pydantic model cannot be represented as a Foxglove
            JSON object schema.
        ValueError: If ``msg_type`` is not supported.
    """

    pydantic_model_type = _validate_pydantic_model_type(topic, msg_type)
    if schema is None:
        schema = create_schema(msg_type, topic=topic)

    message_encoding = None
    if _is_type_subclass(msg_type, ProtobufMessage):
        message_encoding = "protobuf"
    elif _is_type_subclass(msg_type, (dict, list)) or (
        pydantic_model_type is not None
    ):
        message_encoding = "json"
    else:
        raise ValueError(
            f"Unsupported message type {msg_type} for channel {topic}. "
            "Only protobuf messages, dicts, lists, and Pydantic BaseModel "
            "types are supported."
        )

    return Channel(
        topic=topic,
        schema=schema,
        context=context,
        message_encoding=message_encoding,
        metadata=metadata,
    )


def create_channels_from_examples(
    msgs: dict[str, Any],
    context: Context | None = None,
) -> dict[str, Channel]:
    """Create Foxglove channels from example messages keyed by topic."""

    channels = {}
    for topic, msg in msgs.items():
        channels[topic] = create_channel(topic, type(msg), context=context)
    return channels


def _is_type_subclass(
    msg_type: Type[Any] | Any, base: type[Any] | tuple[type[Any], ...]
) -> bool:
    return isinstance(msg_type, type) and issubclass(msg_type, base)


def _validate_pydantic_model_type(
    topic: str | None,
    msg_type: Type[Any] | Any,
) -> type[BaseModel] | None:
    if not _is_type_subclass(msg_type, BaseModel):
        return None

    pydantic_model_type = cast(type[BaseModel], msg_type)
    if _is_type_subclass(pydantic_model_type, RootModel) or bool(
        getattr(pydantic_model_type, "__pydantic_root_model__", False)
    ):
        raise TypeError(
            f"{_format_channel_model(topic, pydantic_model_type)} must not be "
            "a RootModel."
        )
    return pydantic_model_type


def _format_channel_model(topic: str | None, msg_type: type[BaseModel]) -> str:
    if topic is None:
        return f"Pydantic model {msg_type.__name__}"
    return f"Channel {topic} Pydantic model {msg_type.__name__}"
