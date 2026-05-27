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

"""Shared topic helpers for experimental MCAP export."""

from __future__ import annotations
import os

__all__ = ["camera_image_topic"]


def camera_image_topic(source_topic: str) -> str:
    """Return the neutral image leaf topic for one camera source topic.

    Args:
        source_topic (str): Camera source topic before image/calibration/TF
            expansion.

    Returns:
        str: Final image payload topic under ``source_topic``.
    """

    return os.path.join(source_topic, "image")
