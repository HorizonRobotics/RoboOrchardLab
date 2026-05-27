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

"""Utilities for final MCAP topic maps and write-order records."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping

from robo_orchard_lab.dataset.experimental.mcap.foxglove_writer import (
    FoxgloveMcapWriter as McapWriter,
)
from robo_orchard_lab.dataset.experimental.mcap.messages import StampedMessage

__all__ = [
    "TopicStampedMessage",
    "flatten_input_order_records",
    "write_records",
]


@dataclass
class TopicStampedMessage:
    """One final MCAP topic plus a stamped message in input order."""

    topic: str
    idx: int
    message: StampedMessage[Any]

    def __lt__(self, other: TopicStampedMessage) -> bool:
        return self.idx < other.idx


def flatten_input_order_records(
    data: Mapping[str, Iterable[StampedMessage[Any]]],
    start_idx: int = 0,
) -> Iterator[TopicStampedMessage]:
    """Flatten a final topic map without cross-topic log-time sorting.

    The yielded ``idx`` values preserve the iteration order of the input topic
    map and the message order inside each topic.
    """

    idx = start_idx
    for topic, messages in data.items():
        for message in messages:
            if not isinstance(message, StampedMessage):
                raise ValueError(
                    f"Expected StampedMessage for topic '{topic}', "
                    f"but got {type(message)}."
                )
            yield TopicStampedMessage(topic=topic, idx=idx, message=message)
            idx += 1


def write_records(
    records: Iterable[TopicStampedMessage],
    mcap_writer: McapWriter,
) -> tuple[int | None, int | None]:
    """Write final records and return the min/max log time seen."""

    min_log_time: int | None = None
    max_log_time: int | None = None
    for record in records:
        log_time = record.message.log_time
        mcap_writer.write_message(
            topic=record.topic,
            message=record.message.data,
            log_time=log_time,
            publish_time=record.message.pub_time,
        )
        if log_time is None:
            continue
        min_log_time = (
            log_time if min_log_time is None else min(min_log_time, log_time)
        )
        max_log_time = (
            log_time if max_log_time is None else max(max_log_time, log_time)
        )

    return min_log_time, max_log_time
