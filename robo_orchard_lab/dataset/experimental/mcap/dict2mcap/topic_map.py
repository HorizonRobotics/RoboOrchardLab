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
    "Dict2McapWriteSummary",
    "LogTimeBounds",
    "TopicStampedMessage",
    "flatten_input_order_records",
    "summarize_records",
    "write_records",
]


@dataclass(frozen=True, slots=True)
class LogTimeBounds:
    """Min and max final-record log time in nanoseconds."""

    min_log_time: int
    max_log_time: int


@dataclass(frozen=True, slots=True)
class Dict2McapWriteSummary:
    """Lightweight summary of final records written to MCAP."""

    record_count: int
    topics: tuple[str, ...]
    log_time_bounds: LogTimeBounds | None


@dataclass(slots=True)
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


class _RecordSummaryAccumulator:
    def __init__(self) -> None:
        self._record_count = 0
        self._topics: list[str] = []
        self._seen_topics: set[str] = set()
        self._min_log_time: int | None = None
        self._max_log_time: int | None = None

    def add(self, record: TopicStampedMessage) -> None:
        self._record_count += 1
        if record.topic not in self._seen_topics:
            self._seen_topics.add(record.topic)
            self._topics.append(record.topic)
        log_time = record.message.log_time
        if log_time is None:
            return
        self._min_log_time = (
            log_time
            if self._min_log_time is None
            else min(self._min_log_time, log_time)
        )
        self._max_log_time = (
            log_time
            if self._max_log_time is None
            else max(self._max_log_time, log_time)
        )

    def finish(self) -> Dict2McapWriteSummary:
        bounds = (
            None
            if self._min_log_time is None or self._max_log_time is None
            else LogTimeBounds(
                min_log_time=self._min_log_time,
                max_log_time=self._max_log_time,
            )
        )
        return Dict2McapWriteSummary(
            record_count=self._record_count,
            topics=tuple(self._topics),
            log_time_bounds=bounds,
        )


def summarize_records(
    records: Iterable[TopicStampedMessage],
) -> Dict2McapWriteSummary:
    """Summarize already-converted final MCAP records."""
    accumulator = _RecordSummaryAccumulator()
    for record in records:
        accumulator.add(record)
    return accumulator.finish()


def write_records(
    records: Iterable[TopicStampedMessage],
    mcap_writer: McapWriter,
) -> Dict2McapWriteSummary:
    """Write final records and return a lightweight write summary."""
    accumulator = _RecordSummaryAccumulator()
    for record in records:
        mcap_writer.write_message(
            topic=record.topic,
            message=record.message.data,
            log_time=record.message.log_time,
            publish_time=record.message.pub_time,
        )
        accumulator.add(record)

    return accumulator.finish()
