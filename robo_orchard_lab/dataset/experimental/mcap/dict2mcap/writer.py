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

"""Topic-map based MCAP writer facade."""

from __future__ import annotations
import os
import warnings
from typing import Any, Iterable, Mapping

import fsspec

from robo_orchard_lab.dataset.experimental.mcap.dict2mcap.converters import (
    ToMcapMessageConverter,
    ToMcapMessageFactory,
    align_converted_topic_map,
)
from robo_orchard_lab.dataset.experimental.mcap.dict2mcap.topic_map import (
    Dict2McapWriteSummary,
    TopicStampedMessage,
    flatten_input_order_records,
    write_records,
)
from robo_orchard_lab.dataset.experimental.mcap.foxglove_writer import (
    FoxgloveMcapWriter as McapWriter,
)
from robo_orchard_lab.dataset.experimental.mcap.messages import StampedMessage
from robo_orchard_lab.dataset.experimental.mcap.msg_converter.base import (
    MessageConverter,
    MessageConverterConfig,
)

__all__ = ["Dict2Mcap"]


class Dict2Mcap:
    """Write stamped source topic maps to MCAP.

    Input data is keyed by source topic. For each source topic, the writer
    resolves a converter from the optional user factory first and then from the
    default factory. If no converter is found, messages are written directly to
    the same topic as protobuf, schemaless JSON, or Pydantic JSON messages.
    ``writer_converters`` are applied after those source-topic converters, at
    the final MCAP writer topic boundary.

    Args:
        converter_factory (ToMcapMessageFactory | None, optional): Source
            message converter factory. These converters receive the input
            source topic and may emit messages under one or more final MCAP
            topics.
        writer_converters (
            Mapping[str, MessageConverterConfig[MessageConverter]]
            | None, optional): Final-topic keyed converters passed to
            ``McapWriter`` when ``save_to_mcap`` opens the writer itself. Use
            this for last-mile message conversion such as ``RawImage`` to
            ``CompressedImage``.
    """

    def __init__(
        self,
        converter_factory: ToMcapMessageFactory | None = None,
        *,
        writer_converters: (
            Mapping[str, MessageConverterConfig[MessageConverter]] | None
        ) = None,
    ):
        from robo_orchard_lab.dataset.experimental.mcap.dict2mcap.default_converters import (  # noqa: E501
            create_default_to_mcap_message_factory,
        )

        self._converter_factory = converter_factory
        self._default_factory = create_default_to_mcap_message_factory()
        self._source_converter_cache: dict[
            tuple[str, type[Any]], ToMcapMessageConverter | None
        ] = {}
        self._writer_converters = dict(writer_converters or {})

    def iter_records(
        self, data: Mapping[str, Iterable[StampedMessage[Any]]]
    ) -> Iterable[TopicStampedMessage]:
        """Stream-convert source topic messages into final MCAP records.

        The returned iterator is the source-conversion boundary: records are
        keyed by final MCAP topic, source converters are resolved and cached by
        ``(source_topic, source_data_type)``, and output order follows source
        topic iteration order. The iterator does not own an MCAP writer and can
        be materialized by callers that need to inspect final-record bounds
        before writing.
        """
        idx = 0
        for source_topic, messages in data.items():
            source_data_type: type[Any] | None = None
            for message in messages:
                if not isinstance(message, StampedMessage):
                    raise ValueError(
                        f"Expected StampedMessage for topic '{source_topic}', "
                        f"but got {type(message)}."
                    )
                current_data_type = type(message.data)
                if source_data_type is None:
                    source_data_type = current_data_type
                elif current_data_type is not source_data_type:
                    raise TypeError(
                        f"Messages of source topic '{source_topic}' have "
                        "inconsistent data type: "
                        f"{source_data_type} vs {current_data_type}."
                    )

                cache_key = (source_topic, current_data_type)
                if cache_key not in self._source_converter_cache:
                    self._source_converter_cache[cache_key] = (
                        self._resolve_converter(
                            source_topic=source_topic,
                            sample_data=message.data,
                        )
                    )
                converter = self._source_converter_cache[cache_key]

                if converter is None:
                    yield TopicStampedMessage(
                        topic=source_topic, idx=idx, message=message
                    )
                    idx += 1
                    continue

                converted_data = converter.convert(message)
                converted_data = align_converted_topic_map(
                    converted_data, message.log_time
                )
                for record in flatten_input_order_records(
                    converted_data, start_idx=idx
                ):
                    yield record
                    idx = record.idx + 1

    def write_records(
        self,
        records: Iterable[TopicStampedMessage],
        mcap: McapWriter,
    ) -> Dict2McapWriteSummary:
        """Write already-converted final records to an opened MCAP writer.

        The caller owns the writer lifecycle and any writer-side converters.
        This method only consumes the final records and returns a lightweight
        summary of what was written.
        """
        return write_records(records, mcap)

    def save_to_mcap(
        self,
        data: Mapping[str, Iterable[StampedMessage[Any]]],
        mcap: str | os.PathLike[str] | McapWriter | None = None,
        *,
        mcap_path: str | os.PathLike[str] | None = None,
    ) -> Dict2McapWriteSummary:
        """Save a stamped source topic map to an MCAP file or writer.

        Use this facade when the caller does not need to inspect converted
        records before writing. It internally streams
        ``iter_records(data) -> write_records(...)`` and returns a lightweight
        summary without exposing or retaining the converted record collection.

        Args:
            data: Source-topic map containing stamped source messages.
            mcap: Destination path or an opened ``McapWriter``. When a path is
                provided, ``writer_converters`` are installed on the newly
                opened writer. When an opened writer is provided, that writer
                must already own any final-topic converters it needs.
            mcap_path: Deprecated destination path keyword. Use ``mcap``.

        Returns:
            Summary of the final records actually written.

        Raises:
            TypeError: If destination arguments are missing or conflicting, or
                if ``writer_converters`` were configured while saving to an
                already-open writer.
            ValueError: If any input or converted output is not a
                ``StampedMessage`` where one is required.
        """
        if mcap_path is not None:
            warnings.warn(
                "`mcap_path=` is deprecated; use `mcap=` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if mcap is not None:
                raise TypeError("Use either `mcap` or `mcap_path`, not both.")
            mcap = mcap_path
        if mcap is None:
            raise TypeError("Missing required MCAP destination `mcap`.")

        if isinstance(mcap, McapWriter):
            if self._writer_converters:
                raise TypeError(
                    "`Dict2Mcap(writer_converters=...)` cannot apply "
                    "converters to an already-open McapWriter. Pass "
                    "`converters=` to the McapWriter constructor instead."
                )
            return self.write_records(
                self.iter_records(data),
                mcap=mcap,
            )

        if not isinstance(mcap, (str, os.PathLike)):
            raise TypeError(
                "`mcap` must be a path-like object or an opened McapWriter, "
                f"got {type(mcap)}."
            )

        with (
            fsspec.open(os.fspath(mcap), "wb") as f,
            McapWriter(
                f,
                converters=self._writer_converters,
            ) as writer,
        ):  # type: ignore[arg-type]
            return self.write_records(
                self.iter_records(data),
                mcap=writer,
            )

    def _resolve_converter(
        self, source_topic: str, sample_data: Any
    ) -> ToMcapMessageConverter | None:
        converter = None
        if self._converter_factory is not None:
            converter = self._converter_factory.create_converter(
                topic=source_topic, data=sample_data
            )
        if converter is None:
            converter = self._default_factory.create_converter(
                topic=source_topic, data=sample_data
            )
        return converter
