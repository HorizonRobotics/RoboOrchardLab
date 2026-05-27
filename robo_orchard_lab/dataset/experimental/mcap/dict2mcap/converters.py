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

"""Converter interfaces for topic-map based MCAP export."""

from __future__ import annotations
from typing import Any, Callable, Generic, Protocol, Type, TypeVar, overload

from robo_orchard_core.utils.registry import Registry
from typing_extensions import Concatenate, ParamSpec, TypeAlias

from robo_orchard_lab.dataset.experimental.mcap.messages import StampedMessage

__all__ = [
    "MessageDict",
    "ToMcapMessageConverter",
    "ToMcapMessageFactory",
    "align_converted_topic_map",
    "BatchFormatConverter",
]

_P = ParamSpec("_P")

MessageDict: TypeAlias = dict[str, list[StampedMessage[Any]]]


class ToMcapMessageConverter(Protocol):
    """Convert one source stamped message into a final topic map."""

    def convert(self, message: StampedMessage[Any]) -> MessageDict:
        """Return final-topic messages for one source stamped message."""

        ...


_ConverterCreatorType = Callable[
    Concatenate[str, Any, _P], ToMcapMessageConverter
]


class ToMcapMessageFactory:
    """Build reusable MCAP converters from source-topic sample messages.

    Converter creators are registered by exact source data type. A creator sees
    the source topic, the first source data object, and any caller kwargs, then
    returns a converter that can be reused for later messages of that topic.
    """

    def __init__(self) -> None:
        self._registry: Registry = Registry(name="ToMcapMessageRegistry")

    @overload
    def register(
        self,
        data_type: Type[Any],
        converter_factory: _ConverterCreatorType,
    ) -> None: ...

    @overload
    def register(
        self, data_type: Type[Any], converter_factory: None = None
    ) -> Callable[[_ConverterCreatorType], _ConverterCreatorType]: ...

    def register(
        self,
        data_type: Type[Any],
        converter_factory: _ConverterCreatorType | None = None,
    ) -> None | Callable[[_ConverterCreatorType], _ConverterCreatorType]:
        """Register a converter creator, optionally as a decorator."""

        if converter_factory is None:

            def decorator(
                func: _ConverterCreatorType,
            ) -> _ConverterCreatorType:
                self._registry.register(func, name=str(data_type))
                return func

            return decorator

        self._registry.register(converter_factory, name=str(data_type))
        return None

    def create_converter(
        self, topic: str, data: Any, **kwargs: Any
    ) -> ToMcapMessageConverter | None:
        """Create a converter for a source topic and sample data.

        Returns ``None`` when the factory has no creator for ``type(data)``.
        """

        converter_factory: _ConverterCreatorType | None = self._registry.get(
            str(type(data)), raise_not_exist=False
        )
        if converter_factory is None:
            return None
        return converter_factory(topic, data, **kwargs)


def align_converted_topic_map(
    topic_map: MessageDict, log_time: int | None
) -> MessageDict:
    """Align converted messages to a source log time in place.

    The converter contract allows a converter to preserve relative timing
    among multiple output messages. When ``log_time`` is provided, the first
    converted message in each output topic is shifted to that source log time
    and the remaining messages keep their relative offsets.
    """

    if log_time is None:
        return topic_map

    for topic, messages in topic_map.items():
        if len(messages) == 0:
            continue
        first_log_time = messages[0].log_time
        if first_log_time is None:
            raise ValueError(
                f"Converted messages for topic '{topic}' must have log_time."
            )
        log_time_offset = log_time - first_log_time
        if log_time_offset == 0:
            continue
        for message in messages:
            if message.log_time is None:
                raise ValueError(
                    f"Converted messages for topic '{topic}' have "
                    "inconsistent log_time."
                )
            message.log_time += log_time_offset

    return topic_map


T = TypeVar("T")


class BatchFormatConverter(Generic[T]):
    """Adapt batch encoder formatter callables to the converter contract."""

    def __init__(self, formatter: Callable[[T], MessageDict]) -> None:
        self._formatter = formatter

    def convert(self, message: StampedMessage[T]) -> MessageDict:
        """Convert a stamped batch message with the configured formatter."""

        return self._formatter(message.data)
