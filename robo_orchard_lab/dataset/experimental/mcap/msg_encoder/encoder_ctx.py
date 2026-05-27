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

"""Message encoder contexts used by MCAP writers."""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from foxglove import Channel, Context
from google.protobuf.message import Message as ProtobufMessage
from pydantic import BaseModel
from typing_extensions import Self

from robo_orchard_lab.dataset.experimental.mcap.foxglove import (
    create_channel,
)
from robo_orchard_lab.dataset.experimental.mcap.messages import (
    McapMessageTuple,
    StampedMessage,
)
from robo_orchard_lab.dataset.experimental.mcap.msg_converter.base import (
    MessageConverter,
    MessageConverterConfig,
    MessageConverterFactoryConfig,
)

__all__ = ["McapEncoderContext", "FoxgloveEncoder"]


@dataclass(frozen=True)
class _TopicMessageKind:
    kind: str
    message_type: type[Any] | None = None


class McapEncoderContext(metaclass=ABCMeta):
    """Context-managed encoder interface for MCAP writer backends."""

    @abstractmethod
    def reset(self) -> None:
        """Reset any schema, channel, or topic state owned by the encoder."""

        raise NotImplementedError(
            "reset() method is not implemented in the encoder mixin."
        )

    def __enter__(self) -> Self:
        """Enter the context manager."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_value: Optional[BaseException],
        traceback: Optional[Any],
    ) -> None:
        """Exit the context manager."""
        self.reset()
        if exc_type is not None and exc_value is not None:
            raise exc_value

    def encode_message(
        self,
        topic: str,
        msg: Any,
        log_time: int,
        pub_time: int | None = None,
    ) -> McapMessageTuple:
        """Prepare a message tuple with MCAP channel metadata.

        Args:
            topic (str): The topic name of the message.
            msg (Any): The original message data.

        Returns:
            McapMessageTuple: The channel metadata and stamped original
                message.
        """
        raise NotImplementedError(
            "encode_message() method is not implemented in the encoder mixin."
        )


class FoxgloveEncoder(McapEncoderContext):
    """Encoder for Foxglove-compatible MCAP messages.

    The encoder creates one Foxglove channel per topic and keeps a stable
    message kind for each topic. Message payloads stay in their original Python
    form; the writer performs any SDK-required serialization at the final log
    boundary.

    Args:
        ctx (Context | None): Foxglove context used to create channels. A new
            context is created when None is passed.
        converters (
            Mapping[str, MessageConverterConfig[MessageConverter]]
            | None, optional): Topic-keyed converters applied before
            channel/schema resolution in ``encode_message``. Converters must
            be stateless and emit exactly one message per input.
    """

    def __init__(
        self,
        ctx: Context | None,
        converters: (
            Mapping[str, MessageConverterConfig[MessageConverter]] | None
        ) = None,
    ):
        self._converter_factory = MessageConverterFactoryConfig(
            converters=dict(converters or {})
        )()
        self.reset(ctx=ctx)

    def reset(self, ctx: Context | None = None) -> None:
        """Reset channel and topic-kind caches for a Foxglove context."""

        if ctx is None:
            ctx = Context()
        self._ctx = ctx
        self._channel_dict: dict[str, Channel] = {}
        self._topic_kinds: dict[str, _TopicMessageKind] = {}
        self._converters: dict[str, MessageConverter] = {}

    def encode_message(
        self,
        topic: str,
        msg: Any,
        log_time: int,
        pub_time: int | None = None,
    ) -> McapMessageTuple:
        msg = self._convert_topic_message(topic, msg)
        self._check_topic_kind(topic, msg)

        if topic not in self._channel_dict:
            channel = create_channel(topic, type(msg), context=self._ctx)
            self._channel_dict[topic] = channel
        else:
            channel = self._channel_dict[topic]

        data = StampedMessage(data=msg, log_time=log_time, pub_time=pub_time)

        return McapMessageTuple(
            schema=channel.schema(), channel=channel, message=data
        )

    def _convert_topic_message(self, topic: str, msg: Any) -> Any:
        converter = self._converters.get(topic)
        if converter is None:
            converter = self._converter_factory.converter_for(topic)
            if converter is None:
                return msg
            if not converter.stateless:
                raise TypeError(
                    f"FoxgloveEncoder topic converter for '{topic}' must be "
                    "stateless."
                )
            self._converters[topic] = converter

        converted = list(converter(msg))
        if len(converted) != 1:
            raise ValueError(
                f"FoxgloveEncoder topic converter for '{topic}' must emit "
                f"exactly one message, got {len(converted)}."
            )
        return converted[0]

    def _get_message_kind(self, msg: Any) -> _TopicMessageKind:
        msg_type = type(msg)
        if isinstance(msg, ProtobufMessage):
            return _TopicMessageKind("protobuf", msg_type)
        if isinstance(msg, BaseModel):
            return _TopicMessageKind("pydantic", msg_type)
        if isinstance(msg, (dict, list)):
            return _TopicMessageKind("schemaless_json")
        return _TopicMessageKind("unsupported", msg_type)

    def _check_topic_kind(self, topic: str, msg: Any) -> _TopicMessageKind:
        current_kind = self._get_message_kind(msg)
        previous_kind = self._topic_kinds.get(topic)
        if previous_kind is None:
            if current_kind.kind != "unsupported":
                self._topic_kinds[topic] = current_kind
            return current_kind
        if previous_kind == current_kind:
            return current_kind
        raise TypeError(
            f"Topic '{topic}' was first written as {previous_kind.kind} "
            f"({previous_kind.message_type}), but got {current_kind.kind} "
            f"({current_kind.message_type})."
        )
