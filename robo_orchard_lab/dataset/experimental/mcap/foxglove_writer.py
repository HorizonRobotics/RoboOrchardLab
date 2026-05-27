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

"""Foxglove SDK based MCAP writer."""

from typing import Any, BinaryIO, Mapping

from foxglove import (
    Channel,
    Context,
    open_mcap,
)
from foxglove.mcap import MCAPWriteOptions
from google.protobuf.message import Message as ProtobufMessage
from pydantic import BaseModel

from robo_orchard_lab.dataset.experimental.mcap.messages import StampedMessage
from robo_orchard_lab.dataset.experimental.mcap.msg_converter.base import (
    MessageConverter,
    MessageConverterConfig,
)
from robo_orchard_lab.dataset.experimental.mcap.msg_encoder import (
    FoxgloveEncoder,
)

__all__ = [
    "FoxgloveMcapWriter",
]


class FoxgloveMcapWriter:
    """Write protobuf and JSON messages through the Foxglove MCAP SDK.

    Args:
        path (BinaryIO): Open binary file-like destination to write.
        writer_options (MCAPWriteOptions | None, optional): Foxglove SDK
            MCAP writer options such as compression or chunking. Default is
            the SDK default.
        converters (
            Mapping[str, MessageConverterConfig[MessageConverter]]
            | None, optional): Topic-keyed writer-side converters. The key
            is the final MCAP topic passed to ``write_message`` after any
            upstream source-topic conversion has already happened. Converters
            run immediately before ``encode_message`` creates or resolves the
            Foxglove channel.
    """

    def __init__(
        self,
        path: BinaryIO,
        writer_options: MCAPWriteOptions | None = None,
        converters: (
            Mapping[str, MessageConverterConfig[MessageConverter]] | None
        ) = None,
    ):
        self._ctx = Context()
        self._encoder = FoxgloveEncoder(self._ctx, converters=converters)
        self._mcap = open_mcap(
            path,
            context=self._ctx,
            writer_options=writer_options,
        )

    def write_message(
        self,
        topic: str,
        message: Any,
        log_time: int,
        publish_time: int | None = None,
    ) -> None:
        """Write one original message to a Foxglove MCAP channel.

        The writer keeps internal ``StampedMessage`` values in their original
        Python form and only converts payloads to SDK-accepted data just before
        calling ``Channel.log``.

        Args:
            topic (str): Final MCAP topic to write. This topic is also used
                to look up writer-side converters configured on the writer.
            message (Any): Original message payload for this topic.
            log_time (int): MCAP log time in nanoseconds.
            publish_time (int | None, optional): Optional publish time in
                nanoseconds. Default is None.

        Raises:
            TypeError: If a configured topic converter is stateful, or if the
                converted payload does not match the Foxglove channel type.
            ValueError: If a configured topic converter emits zero or multiple
                payloads for one input message.
        """

        msg = self._encoder.encode_message(
            topic, message, log_time, publish_time
        )
        channel = msg.channel
        assert isinstance(channel, Channel)
        assert isinstance(msg.message, StampedMessage)
        payload = self._to_log_payload(
            topic=topic,
            channel=channel,
            message=msg.message.data,
        )
        channel.log(
            msg=payload,
            log_time=msg.message.log_time,
        )

    def __enter__(self):
        """Open the underlying MCAP writer context."""

        self._mcap.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the underlying MCAP writer context."""

        self._mcap.__exit__(exc_type, exc_value, traceback)

    def close(self) -> None:
        """Close the underlying MCAP writer."""

        self._mcap.close()

    def _to_log_payload(
        self,
        topic: str,
        channel: Channel,
        message: Any,
    ) -> Any:
        if channel.message_encoding == "protobuf":
            if not isinstance(message, ProtobufMessage):
                raise TypeError(
                    f"Expected ProtobufMessage for topic '{topic}', "
                    f"but got {type(message)}."
                )
            return message.SerializeToString()

        if isinstance(message, BaseModel):
            try:
                return message.model_dump(mode="json", by_alias=True)
            except Exception as exc:
                raise TypeError(
                    f"Failed to dump Pydantic message for topic '{topic}' "
                    f"and model {type(message).__name__}."
                ) from exc

        return message
