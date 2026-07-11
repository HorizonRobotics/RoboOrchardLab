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

"""Adapter registry: string routing key → concrete adapter class.

Routing is explicit; the case manifest declares ``"adapter": "interna1"`` and
the registry looks the class up. Missing key → hard error listing the
registered adapters, so a user typo does not silently fall back to a default.
"""

from __future__ import annotations
from typing import Callable, Type

from projects.holobrain_internal.common.urdf_tools.adapters.base import (
    DatasetAdapter,
)

_REGISTRY: dict[str, Type[DatasetAdapter]] = {}


def register_adapter(
    key: str,
) -> Callable[[Type[DatasetAdapter]], Type[DatasetAdapter]]:
    """Decorator: register `cls` under `key` for manifest lookup."""

    def _wrap(cls: Type[DatasetAdapter]) -> Type[DatasetAdapter]:
        if key in _REGISTRY:
            raise ValueError(
                f"adapter '{key}' is already registered as "
                f"{_REGISTRY[key].__name__}"
            )
        _REGISTRY[key] = cls
        return cls

    return _wrap


def get_adapter(key: str) -> DatasetAdapter:
    """Return an instance of the adapter registered under `key`.

    Raises:
        KeyError: If no adapter is registered for `key`; the exception message
            lists all registered adapters to help typo diagnosis.
    """

    _ensure_registered()

    if key not in _REGISTRY:
        registered = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(
            f"no adapter registered for '{key}'. "
            f"Registered adapters: {registered}"
        )
    return _REGISTRY[key]()


def registered_adapters() -> tuple[str, ...]:
    _ensure_registered()
    return tuple(sorted(_REGISTRY))


def _ensure_registered() -> None:
    # Import adapter implementations for their side-effect of registering
    # themselves. Kept inside accessors so simply importing this module does
    # not force packer config imports.
    from projects.holobrain_internal.common.urdf_tools.adapters import (  # noqa: F401,E501
        abc130k,
        agibot,
        agilex,
        behavior,
        droid,
        interna1,
        rh20t,
        robotwin,
        table30v2,
    )


__all__ = [
    "DatasetAdapter",
    "get_adapter",
    "register_adapter",
    "registered_adapters",
]
