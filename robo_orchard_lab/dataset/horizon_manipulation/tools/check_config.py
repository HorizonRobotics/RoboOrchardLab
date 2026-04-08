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

from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class InspectConfig:
    """Thresholds used by rule-based MCAP inspection.

    This dataclass centralizes the default thresholds shared by JSON config
    loading, rule evaluation, and validation reporting.
    """

    camera_topics_mean_fps_limit: float = 25.0
    camera_topics_min_fps_limit: float = 10.0
    robot_state_topics_mean_fps_limit: float = 180.0
    robot_state_topics_min_fps_limit: float = 50.0
    timestamp_limit: float = 0.5
    alignment_time_diff_limit: float = 0.5
    joint_limit_tolerance: float = 0.1
    joint_change_tolerance: float = 0.1
    master_slave_joint_tolerance: float = 0.1
    ee_pose_position_tolerance: float = 0.02
    ee_pose_orientation_tolerance: float = 0.05


def build_default_inspect_config() -> InspectConfig:
    """Build the default inspection thresholds.

    Returns:
        InspectConfig: Default inspection thresholds.
    """

    return InspectConfig()


def load_inspect_config(config_path: str | None = None) -> InspectConfig:
    """Load inspection thresholds from JSON and merge them with defaults.

    Args:
        config_path: Optional JSON file path that overrides default thresholds.

    Returns:
        InspectConfig: Effective inspection thresholds.
    """

    if not config_path:
        return build_default_inspect_config()
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    default_payload = asdict(build_default_inspect_config())
    default_payload.update(payload)
    return InspectConfig(**default_payload)
