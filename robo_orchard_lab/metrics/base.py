# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
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

from typing import Any, Protocol, runtime_checkable

from robo_orchard_core.utils.config import ClassConfig
from typing_extensions import TypeVar

__all__ = ["MetricProtocol", "MetricDict", "MetricConfig", "MetricDictConfig"]


@runtime_checkable
class MetricProtocol(Protocol):
    """Protocol for metric functions used in policy evaluation."""

    def __call__(self, *args: Any, **kwds: Any): ...

    def reset(self, **kwargs: Any): ...

    def compute(self) -> Any: ...

    def update(self, *args: Any, **kwds: Any): ...


class MetricDict(dict[str, MetricProtocol]):
    """A dictionary that holds metric functions.

    This class extends the standard dictionary to ensure that all values
    are callable metric functions that follow the MetricProtocol.
    """

    def __setitem__(self, key: str, value: MetricProtocol) -> None:
        if not isinstance(value, MetricProtocol):
            raise ValueError(
                f"The value for key '{key}' must follow the MetricProtocol."
            )
        super().__setitem__(key, value)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        for metric in self.values():
            metric(*args, **kwds)

    def update(self, *args: Any, **kwds: Any) -> None:
        for metric in self.values():
            metric.update(*args, **kwds)

    def reset(self, **kwargs: Any) -> None:
        for metric in self.values():
            metric.reset(**kwargs)

    def compute(self) -> dict[str, Any]:
        return {name: metric.compute() for name, metric in self.items()}


T = TypeVar("T")


class MetricConfig(ClassConfig[T]):
    def __call__(self) -> T:
        return self.class_type(self)  # type: ignore


class MetricDictConfig(dict):
    """A dictionary that holds MetricConfig instances."""

    def __call__(self) -> MetricDict:
        return MetricDict({name: cfg() for name, cfg in self.items()})
