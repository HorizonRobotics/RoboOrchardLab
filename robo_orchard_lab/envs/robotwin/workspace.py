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

from __future__ import annotations
import functools
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager

__all__ = [
    "config_robotwin_path",
    "in_robotwin_workspace",
]


@functools.lru_cache(maxsize=1)
def config_robotwin_path() -> str:
    """Return configured RoboTwin root and make it importable."""
    robo_twin_path = os.environ.get("RoboTwin_PATH", default=None)
    if robo_twin_path is None:
        raise ValueError(
            "RoboTwin_PATH environment variable is not set. "
            "Please set it to the path of the RoboTwin package."
        )
    if robo_twin_path not in sys.path:
        sys.path.append(robo_twin_path)
    return robo_twin_path


@contextmanager
def in_robotwin_workspace() -> Iterator[None]:
    """Temporarily switch cwd to the configured RoboTwin root."""
    robotwin_root = config_robotwin_path()
    original_cwd = os.getcwd()
    os.chdir(robotwin_root)
    try:
        yield
    finally:
        os.chdir(original_cwd)
