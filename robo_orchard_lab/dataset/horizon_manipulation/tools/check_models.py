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
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EpisodeData:
    """Normalized episode payload consumed by rule evaluation.

    This dataclass keeps the normalized arrays, topic statistics, camera
    payloads, and parse warnings consumed by rule evaluation and reporting.
    """

    uuid: str
    meta: dict[str, Any]
    topic_counts: dict[str, int]
    topic_frequencies: dict[str, float | Any]
    base_time: np.ndarray
    joint_positions: np.ndarray
    master_joint_positions: np.ndarray
    fk_ee_poses: np.ndarray
    recorded_ee_poses: np.ndarray | None
    images: dict[str, dict[str, Any]]
    depths: dict[str, dict[str, Any]]
    required_topics: list[str]
    observed_topics: set[str] = field(default_factory=set)
    joint_limit_names: list[str] = field(default_factory=list)
    joint_lower_limits: np.ndarray | None = None
    joint_upper_limits: np.ndarray | None = None
    parse_warnings: list[str] = field(default_factory=list)
    cam_names: list[str] = field(default_factory=list)
    extrinsic: dict[str, np.ndarray] = field(default_factory=dict)
    intrinsic: dict[str, np.ndarray] = field(default_factory=dict)
    topic_summaries: dict[str, dict[str, Any]] = field(default_factory=dict)
    alignment_time_diff_stats: dict[str, dict[str, float]] = field(
        default_factory=dict
    )
    static_filter_applied: bool = False


@dataclass(frozen=True)
class RuleResult:
    """Structured result for one inspection rule.

    It captures the stable rule id, status values such as ``pass`` or
    ``warning``, the rule severity, a human-readable message, and optional
    metrics.
    """

    rule_id: str
    status: str
    severity: str
    message: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the rule result to a plain dictionary.

        Returns:
            dict[str, Any]: Serializable rule payload.
        """

        return {
            "rule_id": self.rule_id,
            "status": self.status,
            "severity": self.severity,
            "message": self.message,
            "metrics": self.metrics,
        }


@dataclass
class EpisodeInspection:
    """Aggregated rule output for one episode before rendering.

    It groups the episode status, per-rule results, render hints, and
    aggregate metrics before serialization.
    """

    episode_status: str
    rule_results: list[RuleResult]
    render_hints: list[str]
    episode_metrics: dict[str, Any]


@dataclass
class EpisodeReport:
    """Serializable episode-level report emitted by the checker.

    It stores the serialized inspection outcome together with optional video
    export and runtime error metadata.
    """

    uuid: str
    episode_status: str
    rule_results: list[RuleResult]
    episode_metrics: dict[str, Any]
    video_file: str | None
    runtime_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the episode report to a plain dictionary.

        Returns:
            dict[str, Any]: Serializable episode report payload.
        """

        return {
            "uuid": self.uuid,
            "episode_status": self.episode_status,
            "rule_results": [item.to_dict() for item in self.rule_results],
            "episode_metrics": self.episode_metrics,
            "video_file": self.video_file,
            "runtime_error": self.runtime_error,
        }
